import copy
import io
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Tuple, Any

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import xarray as xr
from dep_tools.aws import write_to_s3
from dep_tools.grids import COUNTRIES_AND_CODES
from dep_tools.namers import S3ItemPath
from dep_tools.processors import Processor
from dep_tools.writers import StacWriter
from pyproj import CRS, Transformer
from rasterio import rio
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter
from shapely.geometry import box
from torchvision.transforms import Normalize, ToTensor
from tqdm.auto import tqdm
from xarray import Dataset

# This EPSG code is what we're using for now
# but it's not ideal, as its not an equal area projection...
PACIFIC_EPSG = "EPSG:3832"

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


def inference(model, dataloader, n_classes, segmentation_class):
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

    big_out = torch.zeros((n_classes, h, w), device="cpu", dtype=result_dtype)
    model.eval()
    log.info("Finish initiation. Start inferencing...")
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for sub_im_normalized, (i, j) in dataloader:
            pbar.update()
            sub_im_normalized = sub_im_normalized.cuda()
            # breakpoint()
            out = model(sub_im_normalized).cpu()  # [bs, c, s, s]
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
        """fill nan to 0 in img -> get mask of pixels that have all RGB=0 -> collapse to range 0-255 with quantile [0.001, 0.99] channel-wise
        -> set all pixels in mask back to 0 again

        Args:
            img (np.ndarray size [3(RGB) or 4(RGBA), h, w]): the raw image loaded from xdataset

        Returns:
            np.ndarray size [3, h, w]: the processed image
        """
        # img.shape = [3,9600,9600] dtype uint16
        # replace nan with zero
        img = np.nan_to_num(img, nan=0)

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
            self.img = self.standardize(img)

        # step 2: patchify
        log.info("patchify")
        self.patchify(self.img, s=512, s1=256)  # TODO: change s1 to 64 later

        # step 3: inference
        log.info("start height inference")
        height = inference(
            self.height_model,
            self.dataloader,
            n_classes=1,
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
        img[nonvegmask] = (
            0  # IMPORTANT - set all non-veg pixels to 0 so that they don't affect standarization (histogram matching )
        )
        # Step 3: infer standardized data with height_model
        log.info("step 3")
        height = self.infer_height_estimation(
            img, standardize=True
        )  # height [1, h, w] float16

        # convert to xr Dataset
        if isinstance(img, Dataset):
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
        else:
            return height
        # height_ds["vegmask"] = xr.DataArray(
        #     self.mask[None], #shape [1,h,w]
        #     dims=['time', 'y', 'x'],
        #     coords=coords,
        #     name="mask",
        # ).astype('uint8')
        return height_ds


class VegProcessorKeepNonVegPixels(Processor):
    """input: image as nparray, dtype anytype"""

    send_area_to_processor = False

    def __init__(
        self,
        ref_stats_filepath="models/traindata_hist_quantile_mean_std.pkl",
        osm_land_polygons_file="models/land-polygons-complete-4326/land_polygons.shp",
        gadm_land_polygons_file="models/gadm_pacific_union.gpkg",
        land_mask_src="combined",
        testrun=False,
    ):
        # self.allveg_model = torch.jit.load("models/model_allveg10m.pth").cuda()
        self.height_model = torch.jit.load("models/model_height10m.pth").cuda()
        self.ref_stats_filepath = ref_stats_filepath
        # load ref
        _, _, self.ref_mean, self.ref_std = pickle_load(self.ref_stats_filepath)
        self.ref_mean = self.ref_mean.reshape((3, 1, 1))
        self.ref_std = self.ref_std.reshape((3, 1, 1))
        log.info(f"Loaded referenced mean: {self.ref_mean}")
        log.info(f"Loaded referenced std: {self.ref_std}")
        # land polygons
        self.osm_land_polygons_file = Path(osm_land_polygons_file)
        self.gadm_land_polygons_file = Path(gadm_land_polygons_file)
        self.land_mask_src = land_mask_src

        # set in processing
        self.tce_img = None
        self.img = None
        self.landmask = None

        self.testrun = testrun

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """fill nan to 0 in img -> get mask of pixels that have all RGB=0 -> collapse to range 0-255 with quantile [0.001, 0.99] channel-wise
        -> set all pixels in mask back to 0 again

        Args:
            img (np.ndarray size [3(RGB) or 4(RGBA), h, w]): the raw image loaded from xdataset

        Returns:
            np.ndarray size [3, h, w]: the processed image
            also registered self.mask
        """
        # img = img.to_array().squeeze().values # took 46s
        # img.shape = [3,9600,9600] dtype uint16

        # replace nan with zero
        img = np.nan_to_num(img, nan=0)
        # Collapsing to range 0-255
        for i in range(3):
            vmin, vmax = np.quantile(img[i][self.mask], [0.001, 0.99])
            img[i][self.mask] = img[i][self.mask].clip(min=vmin, max=vmax)
            img[i][self.mask] = cv2.normalize(
                img[i][self.mask], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
            ).reshape(img[i][self.mask].shape)

        return img  # [3,h,w]

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
        img[np.repeat(~self.mask[None], 3, axis=0)] = 0  # reset masked pixels to 0
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

    def get_land_mask(self, bbox_4326):
        """Fetch either land mask from GADM or OSM and load only bbox.
        Default return gadm
        """
        # 2) Read only polygons intersecting this bbox (still in 4326)
        # allow for parquet files too
        if self.land_mask_src == "osm":
            landpolygon = get_osm_bbox(bbox_4326, self.osm_land_polygons_file)
        elif self.land_mask_src == "combined":
            all_polys = pd.concat(
                [
                    get_gadm_bbox(bbox_4326, self.gadm_land_polygons_file),
                    get_osm_bbox(bbox_4326, self.osm_land_polygons_file),
                ]
            )
            landpolygon = all_polys.dissolve()[["geometry"]]
        else:
            landpolygon = get_gadm_bbox(bbox_4326, self.gadm_land_polygons_file)
        return landpolygon

    def infer_height_estimation(
        self, img: np.ndarray[np.uint8], standardize=False
    ) -> np.ndarray:
        # img shape: [3, h, w] uint8
        # steps: 1. Standardize img (optional) - 2. patchify, create dataloader - 3. infer, reassemble to large array

        # Step 1: standardize img
        log.info("standardize")
        if standardize:
            self.img = self.standardize(img)

        # step 2: patchify
        log.info("patchify")
        self.patchify(self.img, s=512, s1=256)  # TODO: change s1 to 64 later

        # step 3: inference
        log.info("start height inference")
        height = inference(
            self.height_model,
            self.dataloader,
            n_classes=1,
            segmentation_class="height10m",
        )
        return height  # [1, h, w], float32

    def process(
        self, img: Dataset, use_tce_processing=False, rgb_to_output=False
    ) -> Dataset:
        # img: [4 = RGB + observations, h, w] dtype float32
        coords = {dim: img.coords[dim] for dim in img.dims}
        geobox = img.odc.geobox
        output_data = []
        # Get true color dataset
        # Keep original values and coords
        # rgb app
        if rgb_to_output:
            rgb0_ds = self.get_rgb_ds(img)
            output_data.append(rgb0_ds)

        self.tce_img = self.apply_rgb_enhancements(img["B04"], img["B03"], img["B02"])

        # Observation processing should all be done on using xarray rasterise land mask could also produce xr DataArray
        # obs = img["observations"]
        # Could potententially look at just using NDWI instead
        # 1) Compute bounding box of geobox in EPSG:4326 to efficiently subset polygons
        bbox_4326 = geobox_bounds_in_crs(geobox, target_crs="EPSG:4326")
        landpolygon = self.get_land_mask(bbox_4326)
        landmask = rasterize_land_mask_for_geobox(geobox, landpolygon)  # [9600,9600]

        img = img.to_array().squeeze().values
        obs = img[-1]  # obs

        log.info("Preprocess input...")
        self.landmask = landmask.astype("bool")
        self.mask = self.landmask
        # Step 1: fill NaN=0, convert from uint16 or float to uint8
        # True color actually reduces height slightly
        # smaller difference between grass and trees I think as opposed to rgb
        if use_tce_processing:
            img = self.preprocess(self.tce_img.squeeze().values)
        else:
            img = self.preprocess(img[:-1])
        # Step 2: set all non-land pixels to 0
        # remember self.mask.shape=[h,w]
        invalid_mask = np.repeat(~self.mask[None], 3, axis=0)
        # IMPORTANT - set all non-land pixels to 0 so that they don't affect standarization (histogram matching )
        img[invalid_mask] = 0

        # Step 3: standardize image then infer it with height_model
        log.info("Infer height")
        if not self.testrun:
            height = self.infer_height_estimation(
                img, standardize=True
            )  # height [1, h, w] float16
        else:
            height = self.mask[None]

        # Step 4: calculate confidence from obs
        obs = np.nan_to_num(obs, nan=0)  # set no-observation from nan to 0
        # Compute local average with a 101x101 kernel
        # confidence = uniform_filter(obs, size=51, mode='constant', cval=0)
        confidence = uniform_filter(obs, size=51, mode="reflect")
        confidence = confidence.clip(max=10) / 10 * self.mask.astype("uint8")
        obs *= self.mask.astype("uint8")  # non-land pixels set to 0 observation
        # ---- wrap outputs into xr DataArrays ----
        height_da = xr.DataArray(
            height,  # [time, y, x]
            dims=["time", "y", "x"],
            coords=coords,
            name="height",
        )

        conf_da = xr.DataArray(
            confidence[None, ...],  # add time axis -> [1, h, w]
            dims=["time", "y", "x"],
            coords=coords,
            name="confidence",
        )

        # ---- to datasets + merge ----
        height_ds = height_da.to_dataset()
        conf_ds = conf_da.to_dataset()
        out_ds = xr.merge([height_ds, conf_ds, self.tce_img, *output_data])

        # ---- mask + nodata attrs ----
        out_ds["height"] = out_ds["height"].where(self.mask)
        out_ds["confidence"] = out_ds["confidence"].where(self.mask)

        for v in ["height", "confidence"]:
            out_ds[v].attrs["nodata"] = float("nan")
            out_ds[v].attrs["_FillValue"] = float("nan")
        return out_ds

    def get_rgb_ds(self, img: Dataset) -> Dataset:
        r0 = img["B04"]
        g0 = img["B03"]
        b0 = img["B02"]

        # ensure each has a band label
        r0 = r0.expand_dims(band=["red"])
        g0 = g0.expand_dims(band=["green"])
        b0 = b0.expand_dims(band=["blue"])

        rgb0_da = xr.concat([r0, g0, b0], dim="band").rename("rgb")
        rgb0_ds = rgb0_da.to_dataset()
        return rgb0_ds

    @staticmethod
    def apply_rgb_enhancements(
        red_band: xr.DataArray, green_band: xr.DataArray, blue_band: xr.DataArray
    ) -> xr.DataArray:
        """Apply contrast enhancement and saturation adjustments for RGB."""
        # Constants
        MAX_REFLECTANCE = 3.0
        MID_REFLECTANCE = 0.13
        SATURATION = 1.2
        GAMMA = 1.8
        SCALE_FACTOR = 10000.0

        def clip_enc(value: xr.DataArray) -> xr.DataArray:
            """Clip enc."""
            return value.clip(0, 1)

        def adj(
            value: xr.DataArray, tx: float, ty: float, max_c: float
        ) -> xr.DataArray:
            """Adj."""
            a_ratio = clip_enc(value / max_c)
            numerator = a_ratio * (a_ratio * (tx / max_c + ty - 1) - ty)
            denominator = a_ratio * (2 * tx / max_c - 1) - tx / max_c
            with np.errstate(divide="ignore", invalid="ignore"):
                result = numerator / denominator
                result = result.where(denominator != 0, 0)
            return result

        def adj_gamma(value: xr.DataArray) -> xr.DataArray:
            """Adj gamma."""
            g_offset = 0.01
            g_off_pow = g_offset**GAMMA
            g_off_range = ((1 + g_offset) ** GAMMA) - g_off_pow
            return ((value + g_offset) ** GAMMA - g_off_pow) / g_off_range

        def s_adj(value: xr.DataArray) -> xr.DataArray:
            """S adj."""
            return adj_gamma(adj(value, MID_REFLECTANCE, 1, MAX_REFLECTANCE))

        def sat_enh(
            r: xr.DataArray, g: xr.DataArray, b: xr.DataArray
        ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
            """Sat enh."""
            avg = (r + g + b) / 3.0 * (1 - SATURATION)
            return (
                clip_enc(avg + r * SATURATION),
                clip_enc(avg + g * SATURATION),
                clip_enc(avg + b * SATURATION),
            )

        def s_rgb(c: xr.DataArray) -> xr.DataArray:
            """S rgb."""
            return xr.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) - 0.055)

        # Scale and process bands
        r_scaled = red_band / SCALE_FACTOR
        g_scaled = green_band / SCALE_FACTOR
        b_scaled = blue_band / SCALE_FACTOR

        # Apply adjustments
        r_adj = s_adj(r_scaled)
        g_adj = s_adj(g_scaled)
        b_adj = s_adj(b_scaled)

        # Apply enhancements
        r_enh, g_enh, b_enh = sat_enh(r_adj, g_adj, b_adj)

        # Convert to sRGB
        r_srgb = s_rgb(r_enh)
        g_srgb = s_rgb(g_enh)
        b_srgb = s_rgb(b_enh)

        # Assign band coordinates before concatenation
        r_srgb = r_srgb.assign_coords(band="red")
        g_srgb = g_srgb.assign_coords(band="green")
        b_srgb = b_srgb.assign_coords(band="blue")

        # Concatenate with band dimension
        output = xr.concat([r_srgb, g_srgb, b_srgb], dim="band")
        output = output.assign_coords(
            band=["red", "green", "blue"]
        )  # clearer than per-array assign

        # Scale to 0-255 and convert to uint8
        rgb_out = (output * 255).clip(0, 255).astype("uint8")
        rgb_out.attrs = output.attrs.copy()

        # Copy spatial reference info if present
        for key in ["crs", "transform", "_crs"]:
            if key in red_band.attrs:
                rgb_out.attrs[key] = red_band.attrs[key]

        rgb_out = rgb_out.rename("true_color")
        return rgb_out


def get_osm_bbox(bbox, osm_polygons_file) -> gpd.GeoDataFrame:
    if osm_polygons_file.suffix.lower() == ".parquet":
        box = gpd.read_parquet(osm_polygons_file, bbox=bbox)
    else:
        box = gpd.read_file(osm_polygons_file, bbox=bbox)
    return box


def get_gadm_bbox(bbox, gadm_polygon_file):
    if not gadm_polygon_file.exists():
        log.info(f"Downloading GADM land data to {gadm_polygon_file}...")
        all_polys = pd.concat(
            [
                gpd.read_file(
                    f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{code}.gpkg",
                    layer="ADM_ADM_0",
                )
                for code in COUNTRIES_AND_CODES.values()
            ]
        )

        all_polys.dissolve()[["geometry"]].to_file(gadm_polygon_file)
    return gpd.read_file(gadm_polygon_file, bbox=bbox)


def download_and_extract_land_polygons(
    url: str = "https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip",
    out_dir: str | Path = "land_polygons",
) -> Path:
    """
    Download the OSM land polygons zip and extract it to `out_dir`.
    Returns the path to the main .shp file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find shapefile
    shp_files = list(out_dir.rglob("*.shp"))
    if not shp_files:
        # Download
        resp = requests.get(url)
        resp.raise_for_status()

        # Extract
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(out_dir)

        shp_files = list(out_dir.rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No .shp file found in extracted land polygons.")


def geobox_raster_transform(geobox):
    """
    Get rasterio-compatible transform and shape from a GeoBox.
    Works with datacube GeoBox.
    """
    # GeoBox usually has .affine and .shape, newer versions also .transform
    transform = getattr(geobox, "transform", getattr(geobox, "affine"))
    height, width = geobox.shape  # (rows, cols)
    return transform, (height, width)


def geobox_bounds_in_crs(geobox, target_crs="EPSG:4326"):
    """
    Get bounding box of GeoBox transformed into target_crs.
    Returns (minx, miny, maxx, maxy) in target_crs.
    """
    # GeoBox extent is in its own CRS
    extent = geobox.extent  # datacube geometry with .boundingbox
    left, bottom, right, top = extent.boundingbox

    src_crs = CRS.from_user_input(geobox.crs)
    dst_crs = CRS.from_user_input(target_crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Transform corners (approx; good for moderate-sized areas)
    minx, miny = transformer.transform(left, bottom)
    maxx, maxy = transformer.transform(right, top)

    # Ensure ordering is correct
    minx, maxx = min(minx, maxx), max(minx, maxx)
    miny, maxy = min(miny, maxy), max(miny, maxy)
    return minx, miny, maxx, maxy


def rasterize_land_mask_for_geobox(
    geobox, land: gpd.GeoDataFrame, all_touched: bool = False
) -> np.ndarray:
    """
    Given a GeoBox (in EPSG:3832) and a path to the OSM land polygons shapefile
    (in EPSG:4326), return a rasterized mask aligned to the GeoBox:

        - 1 where land intersects the GeoBox
        - 0 elsewhere

    Returns: numpy.ndarray of shape (geobox.height, geobox.width), dtype uint8.
    """

    if land.empty:
        # No land polygons intersect this geobox: return all zeros
        _, (height, width) = geobox_raster_transform(geobox)
        return np.zeros((height, width), dtype=np.uint8)

    # 3) Reproject polygons to the GeoBox CRS (EPSG:3832)
    land = land.to_crs(geobox.crs)

    # 4) Clip polygons to the GeoBox's exact extent in its CRS (for efficiency)
    extent = geobox.extent
    left, bottom, right, top = extent.boundingbox
    geobox_poly = box(left, bottom, right, top)

    land["geometry"] = land.geometry.intersection(geobox_poly)
    land = land[~land.geometry.is_empty & land.geometry.notnull()]

    if land.empty:
        _, (height, width) = geobox_raster_transform(geobox)
        return np.zeros((height, width), dtype=np.uint8)

    # 5) Rasterize
    transform, (height, width) = geobox_raster_transform(geobox)

    shapes = ((geom, 1) for geom in land.geometry if not geom.is_empty)

    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )

    return mask


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


def normalize_rgb_to_uint8(rgb):
    """
    Normalize a float32 RGB array to the range [0, 255] and convert to uint8.
    rgb: np.ndarray of shape (3, H, W), dtype float32
    Returns: np.ndarray of shape (3, H, W), dtype uint8
    """
    rgb_min = 0.1
    rgb_max = np.quantile(rgb, 0.99)
    # Avoid division by zero
    if rgb_max == rgb_min:
        norm_rgb = np.zeros_like(rgb, dtype=np.uint8)
    else:
        norm_rgb = (
            ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).clip(0, 255).astype(np.uint8)
        )
    return norm_rgb


# Extract metadata from the xarray Dataset 'data' for saving as raster
def extract_raster_meta(data):
    """
    Extracts raster metadata from an xarray Dataset for rasterio saving.
    Returns a dictionary with keys: dtype, crs, transform, count, height, width.
    """
    # Use the first band for shape and dtype
    band = list(data.data_vars)[0]
    arr = data[band]
    dtype = str(arr.dtype)
    height, width = arr.shape[-2], arr.shape[-1]
    # Get CRS and transform from geobox if available
    crs = data.spatial_ref if hasattr(data, "spatial_ref") else None
    if hasattr(data, "odc") and hasattr(data.odc, "transform"):
        transform = data.odc.transform
    elif hasattr(data, "transform"):
        transform = data.transform
    else:
        transform = None
    # Try to get CRS string
    if hasattr(data, "odc") and hasattr(data.odc, "crs"):
        crs = data.odc.crs
    elif hasattr(data, "crs"):
        crs = data.crs
    elif hasattr(data, "spatial_ref"):
        crs = data.spatial_ref
    meta = {
        "dtype": dtype,
        "crs": str(crs) if crs is not None else None,
        "transform": transform,
        # "count": arr.shape[0] if arr.ndim == 3 else 1,  # number of channels
        # "height": height,
        # "width": width,
    }
    return meta


def save_raster(a, meta, save_path, nbits=8, overwrite=False, **kwargs):
    """
    a: np array [c,h,w] or [h,w]
    meta: dict
    save_path: ends with .tif
    nbits: number of bits to store each pixel. This is very helpful to compress the raster if the value range is small.
    overwrite: option to overwrite existing files
    """
    save_path = Path(save_path)
    save_path_exists = save_path.exists()
    if save_path_exists:
        print(f"file exists: {save_path}")
        if not overwrite:
            raise Exception(
                f"file exists: {save_path} and overwrite is set to False. Please set overwrite to True"
            )
        print(f"Overwriting file...")
    if len(a.shape) == 2:
        a = a[None]  # ensure a.shape = [c,h,w]
    meta["count"] = a.shape[0]
    meta["height"] = a.shape[1]
    meta["width"] = a.shape[2]
    meta["dtype"] = a.dtype
    try:
        with rio.open(save_path, "w", nbits=nbits, **meta, **kwargs) as dest:
            for i in range(a.shape[0]):
                dest.write(a[i], i + 1)
    except Exception as e:
        raise Exception(
            f"Error saving raster to {save_path} error {e}. Check the meta and data shape. Meta: {meta}, data shape: {a.shape}"
        )


def load_raster(raster_path):
    """
    Load a raster file as a numpy array and return array and metadata.
    Returns:
        arr: np.ndarray
        meta: dict (rasterio metadata)
    """
    raster_path = str(raster_path)
    with rasterio.open(raster_path) as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


class CustomAwsStacWriter(StacWriter):
    def __init__(
        self,
        itempath: S3ItemPath,
        **kwargs,
    ):
        write_stac_function = kwargs.pop("write_stac_function") or write_to_s3
        super().__init__(
            itempath=itempath,
            write_stac_function=write_stac_function,
            bucket=itempath.bucket,
            **kwargs,
        )


from datetime import date
import re


def quarter_start_dates(year_or_period: str):
    """
    Input:
        - '2024' -> all quarter starts in 2024
        - '2023-2024' -> all quarter starts from 2023 through 2024 inclusive
    Output:
        List of 'YYYY-MM-DD' strings for quarter starts (Jan 1, Apr 1, Jul 1, Oct 1).
    """
    if not isinstance(year_or_period, str):
        raise TypeError("Input must be a string like '2024' or '2023-2024'.")

    s = year_or_period.strip()

    # if a date, do nothing
    if is_date(s):
        return [s]

    # Match either "YYYY" or "YYYY-YYYY" with optional whitespace around hyphen
    m_single = re.fullmatch(r"(\d{4})", s)
    m_range = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", s)

    if m_single:
        start_year = end_year = int(m_single.group(1))
    elif m_range:
        start_year = int(m_range.group(1))
        end_year = int(m_range.group(2))
        if end_year < start_year:
            raise ValueError("End year must be >= start year.")
    else:
        raise ValueError("Invalid format. Use 'YYYY' or 'YYYY-YYYY'.")

    quarter_months = (1, 4, 7, 10)
    out = []
    for y in range(start_year, end_year + 1):
        for m in quarter_months:
            out.append(date(y, m, 1).isoformat())
    return out


def is_date(t: str):
    """
    Returns True if t matches the YYYY-MM-DD format, else False.
    """
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", t))
