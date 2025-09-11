from pathlib import Path
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dep_tools.processors import Processor
from dep_tools.s2_utils import mask_clouds
from odc.algo import mask_cleanup
from odc.stac import load
from pystac import Item
from xarray import DataArray, Dataset
import cv2
import torch
from torchvision.transforms import Normalize, ToTensor
import pickle
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt
import logging

BATCH_SIZE = 6
NUM_WORKERS = 0
PREFETCH_FACTOR = None
SUPPORTED_TASKS = ["allveg10m", "height10m"]
IMAGENET_NORMALIZER = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

log = logging.getLogger(f"{__name__}")


def pickle_load(file):
    with open(file, "rb") as f:
        x = pickle.load(f)
    f.close()
    return x


def resize_figure(fig, factor=2):
    return fig.set_size_inches(*(fig.get_size_inches() * factor))


def imshow(
    im, fig=None, ax=None, alpha=1, cmap=None, label="", return_true=False, resize=False
):
    """
    im: np array [c,h,w], value range from 0 to 255
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if torch.is_tensor(im):
        im = im.detach().cpu().numpy()

    # if im.max() <= 1:
    #     im = im * 255

    im = im.astype(np.uint8)
    if im.shape[0] == 3:  # if is an RGB image, not a mask
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(im.transpose([1, 2, 0]))
    else:  # if a mask
        if len(im.shape) == 2:
            im = im[None]
        if ax is None:
            fig, ax = plt.subplots(1, len(im))
        if len(im) == 1:
            f = ax.imshow(im[0], alpha=alpha, cmap=cmap)
            divider = make_axes_locatable(ax)
            # if 1 in im.shape: # add a colorbar
            cax = divider.append_axes("right", size="7%", pad=0.05)
            fig.colorbar(f, cax=cax, ax=ax)
            fig.tight_layout()
            ax.set_title(label)
        else:
            for i in range(im.shape[0]):
                f = ax[i].imshow(im[i], alpha=alpha, cmap=cmap)
                ax[i].axis("off")
                divider = make_axes_locatable(ax[i])
                # if 1 in im.shape: # add a colorbar
                cax = divider.append_axes("right", size="7%", pad=0.05)
                fig.colorbar(f, cax=cax, ax=ax[i])
                fig.tight_layout()
                ax[i].set_title(label + " " + str(i))

    if resize:
        resize_figure(fig, resize)
    if return_true:
        return fig, ax


class PatchDataset(torch.utils.data.Dataset):
    """
    Patch dataset for inference that inputs the path to a raster and returns the patches by using rasterio reading in window.
    The list of indices are generated similar to patchify2, but no mirroring is used. Instead, the right and bottom borders are sampled to be overlapped.
    This dataset arises from the need to inference highly overlapping patches for smooth inference result between patches, hence saving each patch to external storage then read them like in PatchDataset becomes inefficient. This becomes much more important when inferencing huge rasters.
    Args:
        img_path: path to raster. Tested on .tif
        s, s1: int, int - size of each patch and sub-size of each patch
        standardize: bool - if True, load the mean, std of the training data from NZ data. Then create mapping from input image to reference mean, std. Then apply the mapping to each patch loaded.
        normalizer: torchvision.transforms.Normalizer - the normalizer used duing training.
        rgb_order: list of 3 int - the order of the bands to read to ensure the read data is RGB. Useful when input data is BGR (then this should be set to [3,2,1])
    """

    def __init__(
        self,
        img: np.ndarray,
        s: int,
        s1: int,
        normalizer=None,
        check_empty_patches=False,
    ):
        self.img = img
        self.has_alpha = (
            img.shape[0] == 4
        )  # if img has 4 channels, means it has alpha mask
        self.s = s
        self.s1 = s1
        assert s > s1
        self.normalizer = normalizer

        # get info from raster
        self.h = img.shape[1]
        self.w = img.shape[2]

        self.get_indices(check_empty_patches)

    def get_indices(self, check_empty_patches):
        log.info("Creating patches...")
        i_id = list(range(0, self.h - self.s, self.s1))
        j_id = list(range(0, self.w - self.s, self.s1))
        i_id += [self.h - self.s]
        j_id += [self.w - self.s]
        self.indices = np.stack(np.meshgrid(i_id, j_id, indexing="ij"), axis=-1)
        self.indices = self.indices.reshape([-1, 2]).tolist()

        n_all_patches = len(self.indices)
        # filter all-zero patches
        if check_empty_patches:
            log.info(f"Total {n_all_patches} patches.\nFiltering all-zero patches...")
            for i in tqdm(self.indices):
                valid_mask = self.img[:, i[0] : i[0] + self.s, i[1] : i[1] + self.s]
                if not valid_mask.any():  # if all is zero
                    self.indices.remove(i)
            n_valid_patches = len(self.indices)
            log.info(f"{n_valid_patches} patches remaining.\n")

    def load(self, idx):
        i, j = self.indices[idx]
        x = self.img[:, i : i + self.s, j : j + self.s]
        return x, i, j  # x: nparray [3, s, s] uint8

    def __getitem__(self, idx):
        x, i, j = self.load(idx)
        x = x.transpose(
            [1, 2, 0]
        )  # transpose to later use torchvision ToTensor to convert back to [c,s,s] and scale to range [0., 1.]
        # assert x.dtype == np.uint8
        x = ToTensor()(x)  # [3,s,s]
        if self.normalizer is not None:
            x = self.normalizer(x).float()
        return x, (i, j)

    def __len__(self):
        return len(self.indices)


def inference(model, dataloader, n_classes, device, segmentation_class):
    """
    inference a big image from the window patches returned by PatchDataset2.
    returns big_pred:tensor [n,h,w] - the output of the model, as-is without any post processing (no sigmoid, no scaling...)
    Args:
        model: torch jit model - the treemask_height model, which has cfg.segmentation_class == height but outputs 2 channels: tree = out[0] and height=out[1]
        dataloader: torch.utils.dataset.DataLoader - dataloader created from PatchDataset2
        n_classes: int - number of channels of model output
        device: str - to create big_out that collects the model output
        average: str - method to average (exp, uniform, max, None)
    """
    s = dataloader.dataset.s
    s1 = dataloader.dataset.s1
    d1 = (s - s1) // 2
    d2 = d1 + s1
    h = dataloader.dataset.h
    w = dataloader.dataset.w

    assert segmentation_class in SUPPORTED_TASKS

    if segmentation_class == "allveg10m":
        result_dtype = torch.uint8
    elif segmentation_class == "height10m":
        result_dtype = torch.float32

    big_out = torch.zeros((n_classes, h, w), device=device, dtype=result_dtype)
    model.eval()
    log.info("Finish initiation. Start inferencing...")
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for sub_im_normalized, (i, j) in dataloader:
            pbar.update()
            sub_im_normalized = sub_im_normalized.cuda()
            # breakpoint()
            out = model(sub_im_normalized).to(device)  # [bs, c, s, s]
            if segmentation_class == "allveg10m":
                out = torch.argmax(out, dim=1, keepdim=True)  # [bs,1,s,s]
            elif segmentation_class == "height10m":
                out = out * 60
                out = out.clamp(min=1) - 1

            # other unsupported tasks
            # elif ('height' in segmentation_class and n_classes == 1) or \
            #      (segmentation_class == 'heightstat10m'):
            #     out = output_transform(out)
            #     out = out.clamp(min=1) - 1
            # elif 'tree' in segmentation_class:
            #     out = torch.nn.functional.sigmoid(out)
            # elif 'canopycover' in segmentation_class:
            #     # out = torch.nn.functional.sigmoid(out)
            #     out = out.clip(min=0, max=1)
            #     out = output_transform(out) # multiply by 100 to outputs percentage
            # elif 'building' in segmentation_class:
            #     out = torch.nn.functional.sigmoid(out)
            #     # pass

            if result_dtype == torch.uint8:
                out = out.round().to(torch.uint8)
            elif result_dtype == torch.float32:
                out = torch.round(out, decimals=4).to(torch.float32)

            for k in range(len(out)):
                try:
                    di1 = 0 if i[k] == 0 else d1
                    di2 = s if i[k] == h - s else d2
                    dj1 = 0 if j[k] == 0 else d1
                    dj2 = s if j[k] == w - s else d2
                    big_out[:, i[k] + di1 : i[k] + di2, j[k] + dj1 : j[k] + dj2] = (
                        copy.deepcopy(out[k][:, di1:di2, dj1:dj2])
                    )

                    # overlap_count[:, i[k]+di1:i[k]+di2, j[k]+dj1:j[k]+dj2] += 1

                except Exception as e:
                    log.info(e)
                    # breakpoint()
        pbar.close()
    return big_out.cpu().numpy()


class VegProcessor(Processor):
    """input: image as nparray, dtype anytype"""

    send_area_to_processor = False

    def __init__(
        self, ref_stats_filepath="models/traindata_hist_quantile_mean_std(veg_only).pkl"
    ):
        self.allveg_model = torch.jit.load("models/model_allveg10m.pth").cuda()
        self.height_model = torch.jit.load("models/model_height10m.pth").cuda()
        self.ref_stats_filepath = ref_stats_filepath
        # load ref
        _, _, self.ref_mean, self.ref_std = pickle_load(self.ref_stats_filepath)
        self.ref_mean = self.ref_mean.reshape((3, 1, 1))
        self.ref_std = self.ref_std.reshape((3, 1, 1))
        log.info(f"Loaded referenced mean: {self.ref_mean}")
        log.info(f"Loaded referenced std: {self.ref_std}")

    def preprocess(self, img: np.ndarray) -> np.ndarray:

        # img = img.to_array().squeeze().values # took 46s
        # img.shape = [3,9600,9600] dtype uint16

        # get alpha mask
        if img.shape[0] == 3:
            self.mask = ~get_mask(img)[0]  # [h,w] # True - valid, False - invalid
            self.mask = self.mask.astype("uint8")
            # dilate then erode the mask to leave out the valid pixels with values = (0,0,0) inside the raster
            kernel = np.ones((5, 5), np.uint8)
            self.mask = cv2.dilate(self.mask, kernel, iterations=1)
            self.mask = cv2.erode(self.mask, kernel, iterations=1)

        elif img.shape[0] == 4:
            log.info(
                f"Fourth channel detected, which has min = {img[-1].min()}, max={img[-1].max()}. It will be treated as alpha channel"
            )
            self.mask = img[3].clip(max=1).astype("uint8")

        self.mask = self.mask.astype("bool")

        # Collapsing to range 0-255
        for i in range(3):
            vmin, vmax = np.quantile(img[i], [0.001, 0.99])
            img[i] = img[i].clip(min=vmin, max=vmax)

        rgb = img[:3].transpose([1, 2, 0])  # [h,w,3]
        rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        rgb = rgb.transpose([2, 0, 1])  # [3,h,w]
        final_data = np.empty((3, *img.shape[1:]), dtype=np.uint8)

        # log.info('+1 to zero valid pixels...')
        # valid_zero_pixels = np.bitwise_and(utils.get_mask(rgb)[0], mask).astype('bool')
        # log.info(valid_zero_pixels.sum())
        # rgb[rgb == 0] = 1
        rgb[np.repeat(~self.mask[None], 3, 0)] = 0
        final_data = rgb
        # self.mask *= 255 # scale alpha mask from [0, 1] to [0, 255]
        return final_data  # [3,h,w]

    def standardize(self, img: np.ndarray) -> np.ndarray:
        """apply the transform to img: img = (img - shift_factor)*scale factor
        where shift/scale factors are calculated from img's mean/std and reference mean/std
        Args:
            img (np.ndarray): img array shape [3, h, w] uint8

        Returns:
            np.ndarray: output array shape [3,h,w] uint8
        """
        self.mask = self.mask.astype("bool")
        # calculate img stats (mean/std) channel-wise
        mean, std = [], []
        for i in img:
            mean.append(i[self.mask].mean())
            std.append(i[self.mask].std())
        mean = np.array(mean)
        std = np.array(std)
        mean = mean.reshape((3, 1, 1))
        std = std.reshape((3, 1, 1))
        # apply shift, scale
        img = img.astype("float32")
        img = (img - mean) / std
        img = img * self.ref_std + self.ref_mean

        img = img.round().clip(min=0, max=255).astype("uint8")
        return img

    def patchify(self, img: np.ndarray, s: int, s1: int) -> None:
        """create patches of size [3,s,s] from the img of size [3,h,w]; at the same time create a Dataset and DataLoader for fetching those patches

        Args:
            img (np.ndarray): size [3, h, w]

        Returns:
            None, but registered self.dataset, self.dataloader

        """
        self.dataset = PatchDataset(
            img, s, s1, IMAGENET_NORMALIZER, check_empty_patches=True
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=False,
        )
        return None

    def infer_veg_segmentation(self, img: np.ndarray, standardize=False) -> np.ndarray:
        # img shape: [3, h, w] uint8
        # steps: 1. Standardize img (optional) - 2. patchify, create dataloader - 3. infer, reassemble to large array

        # Step 1: standardize img
        if standardize:
            img = self.standardize(img)

        # step 2: patchify
        self.patchify(img, s=512, s1=256)

        # step 3: inference
        allveg = inference(
            self.allveg_model,
            self.dataloader,
            n_classes=1,
            device="cpu",
            segmentation_class="allveg10m",
        )

        return allveg  # [1, h, w], uint8, 0=nodata, 1=non-veg, 2=low-veg, 3=high-veg

    def infer_height_estimation(
        self, img: np.ndarray[np.uint8], standardize=False
    ) -> np.ndarray:
        # img shape: [3, h, w] uint8
        # steps: 1. Standardize img (optional) - 2. patchify, create dataloader - 3. infer, reassemble to large array

        # Step 1: standardize img
        log.info("standardize")
        if standardize:
            img = self.standardize(img)

        # step 2: patchify
        log.info("patchify")
        self.patchify(img, s=512, s1=256)  # TODO: change s1 to 64 later

        # step 3: inference
        log.info("start height inference")
        height = inference(
            self.height_model,
            self.dataloader,
            n_classes=1,
            device="cpu",
            segmentation_class="height10m",
        )
        return height  # [1, h, w], float32

    def process(self, img: np.ndarray | Dataset) -> Dataset:

        if isinstance(img, Dataset):
            coords = {dim: img.coords[dim] for dim in img.dims}
            img = img.to_array().squeeze().values
        log.info("Preprocess input...")
        img = self.preprocess(img)
        # Step 1: infer img with veg_model to obtain allveg mask
        log.info("Step 1")
        allveg = self.infer_veg_segmentation(img, standardize=False)  # [1, h, w]

        # Step 2: use allveg mask to extract pixels for standardization, perform standardization
        log.info("Step 2")
        nonvegmask = allveg <= 1  # True if not veg, False if veg
        self.mask = np.bitwise_and(
            self.mask, ~nonvegmask[0]
        )  # remember self.mask.shape=[h,w]
        nonvegmask = np.repeat(nonvegmask, 3, axis=0)
        img[nonvegmask] = 0
        # Step 3: infer standardized data with height_model
        log.info("step 3")
        height = self.infer_height_estimation(
            img, standardize=True
        )  # height [1, h, w] float16

        # convert to xr Dataset
        height_da = xr.DataArray(
            height,  # remove channel dimension, then add 1 channel for time, so no action needed
            dims=["time", "y", "x"],
            coords=coords,
            name="height",
        )

        height_ds = height_da.to_dataset()
        height_ds["height"] = height_ds["height"].where(self.mask)
        height_ds["height"].attrs["nodata"] = float("nan")
        height_ds["height"].attrs["_FillValue"] = float("nan")
        # height_ds["vegmask"] = xr.DataArray(
        #     self.mask[None], #shape [1,h,w]
        #     dims=['time', 'y', 'x'],
        #     coords=coords,
        #     name="mask",
        # ).astype('uint8')
        return height_ds


def get_mask(im, channel_axis=0, nodata=0):
    """
    im: nparray [c,h,w], pixel value=0 in all channels means no data
    Useful for masked_im = np.ma.array(im, mask=get_mask(im))

    returns: mask [c, h, w], where mask[i,j,k] = True if im[:, j, k] == 0
    """
    mask = np.repeat(
        np.all(im == nodata, axis=channel_axis, keepdims=True),
        repeats=im.shape[channel_axis],
        axis=channel_axis,
    )
    return mask


# maybe useful in future development
# def apply_mask(
#     ds: Dataset,
#     mask: DataArray,
#     ds_to_mask: Dataset | None = None,
#     return_mask: bool = False,
# ) -> Dataset:
#     """Applies a mask to a dataset"""
#     to_mask = ds if ds_to_mask is None else ds_to_mask
#     masked = to_mask.where(mask)

#     if return_mask:
#         return masked, mask
#     else:
#         return masked


# def mask_land(
#     ds: Dataset, ds_to_mask: Dataset | None = None, return_mask: bool = False
# ) -> Dataset:
#     """Masks out land pixels based on the NDWI and MNDWI indices.

#     Args:
#         ds (Dataset): Dataset to mask
#         ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
#         return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

#     Returns:
#         Dataset: Masked dataset
#     """
#     land = (ds.mndwi + ds.ndwi).squeeze() < 0
#     mask = mask_cleanup(land, [["dilation", 5], ["erosion", 5]])

#     # Inverting the mask here
#     mask = ~mask

#     return apply_mask(ds, mask, ds_to_mask, return_mask)


# def do_prediction(
#     ds: Dataset, model: RegressorMixin, output_name: str | None = None
# ) -> Dataset | DataArray:
#     """Predicts the model on the dataset and adds the prediction as a new variable.

#     Args:
#         ds (Dataset): Dataset to predict on
#         model (RegressorMixin): Model to predict with

#     Returns:
#         Dataset: Dataset with the prediction as a new variable
#     """
#     mask = ds.red.isnull()  # Probably should check more bands

#     # Convert to a stacked array of observations
#     stacked_arrays = ds.to_array().stack(dims=["y", "x"])

#     # Replace any infinities with NaN
#     stacked_arrays = stacked_arrays.where(stacked_arrays != float("inf"))
#     stacked_arrays = stacked_arrays.where(stacked_arrays != float("-inf"))

#     # Replace any NaN values with 0
#     df = stacked_arrays.squeeze().fillna(0).transpose().to_pandas()

#     # Remove the all-zero rows
#     zero_mask: pd.Series[bool] = (df == 0).all(axis=1)
#     non_zero_df = df.loc[~zero_mask]

#     # Create a new array to hold the predictions
#     full_pred = pd.Series(np.nan, index=df.index)

#     # Only run the prediction if there are non-zero rows
#     if not non_zero_df.empty:
#         # Predict the classes
#         preds = model.predict(non_zero_df)

#         # Fill the new array with the predictions, skipping those old zero rows
#         full_pred.loc[~zero_mask] = preds

#     # Reshape back to the original 2D array
#     array = full_pred.to_numpy().reshape(ds.y.size, ds.x.size)

#     # Convert to an xarray again, because it's easier to work with
#     predicted_da = xr.DataArray(array, coords={"y": ds.y, "x": ds.x}, dims=["y", "x"])

#     # Mask the prediction with the original mask
#     predicted_da = predicted_da.where(~mask)

#     # If we have a name, return dataset, else the dataarray
#     if output_name is None:
#         return predicted_da
#     else:
#         return predicted_da.to_dataset(name=output_name)
