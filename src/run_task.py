# from logging import INFO, Formatter, Logger, StreamHandler, getLogger
import logging
from pathlib import Path
from zipfile import ZipFile

import boto3
import requests
import typer
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator
from dep_tools.task import AwsStacTask as Task
from dep_tools.writers import AwsDsCogWriter
from odc.stac import configure_s3_access
from typing_extensions import Annotated

from utils import VegProcessor


# def get_logger(region_code: str) -> Logger:
#     """Set up a simple logger"""
#     console = StreamHandler()
#     time_format = "%Y-%m-%d %H:%M:%S"
#     console.setFormatter(
#         Formatter(
#             fmt=f"%(asctime)s %(levelname)s ({region_code}):  %(message)s",
#             datefmt=time_format,
#         )
#     )

#     log = getLogger("GEOMAD")
#     log.addHandler(console)
#     log.setLevel(INFO)
#     return log

# Configure logging ONCE here (root logger)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


def main(
    model_zip_uri: Annotated[str, typer.Option()],
    tile_id: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = "dep-public-staging",
    overwrite: Annotated[bool, typer.Option()] = False,
    datetime: Annotated[str, typer.Option()] = "2024",
) -> None:
    # log = get_logger(tile_id)
    log = logging.getLogger(tile_id)
    log.info("Starting processing")

    grid = PACIFIC_GRID_10
    # catalog = "https://earth-search.aws.element84.com/v1"
    # collection = "sentinel-2-l2a"
    catalog = "https://stac.digitalearthpacific.org"
    collection = "dep_s2_geomad"
    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)
    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor="s2",  ########################
        dataset_id="vegheight",  ########################
        version=version,
        time=datetime,
    )
    stac_document = itempath.stac_path(tile_id)
    stac_document = stac_document.replace("[", "").replace(
        "]", ""
    )  # remove brackets from tile_id for path

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document, client=client):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    ######################## where to upload model?
    # Download the model and unzip it
    Path("models").mkdir(parents=True, exist_ok=True)

    model_zip = "models/" + model_zip_uri.split("/")[-1]
    if not Path(model_zip).exists():
        log.info(f"Downloading model from {model_zip_uri}")
        r = requests.get(model_zip_uri)
        with open(model_zip, "wb") as f:
            f.write(r.content)

        log.info("Unzipping model")
        with ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall("models/")

    # Open the model ######################## this is done in the Processor
    # model = joblib.load(model_zip.replace(".zip", ".joblib"))

    tile_index = tuple(
        int(i) for i in tile_id.replace("[", "").replace("]", "").split(",")
    )
    geobox = grid.tile_geobox(tile_index)

    searcher = PystacSearcher(
        catalog=catalog,
        collections=[collection],
        datetime=datetime,
    )

    loader = OdcLoader(
        bands=["red", "green", "blue"],
        # chunks={"x": 3201, "y": 3201},######################## don't need it for geomad - small im size
        groupby="solar_day",  ######################## does this matter for our application?
        fail_on_error=False,
    )

    processor = (
        VegProcessor()
    )  ######################## this includes: loads 2 models (1GB each), loads one .pkl file for stats, numpy [3,h,w] uint16 for input,

    # Custom writer so we write multithreaded
    writer = AwsDsCogWriter(itempath, write_multithreaded=True)

    # STAC making thing
    stac_creator = StacCreator(
        itempath=itempath, remote=True, make_hrefs_https=True, with_raster=True
    )

    try:
        paths = Task(
            itempath=itempath,
            id=tile_index,
            area=geobox,
            searcher=searcher,
            loader=loader,
            processor=processor,
            writer=writer,
            logger=log,
            stac_creator=stac_creator,
        ).run()
    except EmptyCollectionError:
        log.info("No items found for this tile")
        raise typer.Exit()  # Exit with success
    except Exception as e:
        log.exception(f"Failed to process with error: {e}")
        raise typer.Exit(code=1)

    log.info(
        f"Completed processing. Wrote {len(paths)} items to https://{output_bucket}.s3.us-west-2.amazonaws.com/{stac_document}"
    )


if __name__ == "__main__":
    typer.run(main)
