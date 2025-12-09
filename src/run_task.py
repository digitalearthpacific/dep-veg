# from logging import INFO, Formatter, Logger, StreamHandler, getLogger
import logging
import os
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import boto3
import requests
import typer
from dask.distributed import Client
from dep_tools.aws import object_exists, write_to_s3
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath, LocalPath
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator

# from dep_tools.task import AwsStacTask as Task
from task import CopernicusReadAwsStacTask as Task
from dep_tools.writers import AwsDsCogWriter, LocalDsCogWriter
from odc.stac import configure_s3_access
from typing_extensions import Annotated

from utils import (
    VegProcessorKeepNonVegPixels,
    rasterize_land_mask_for_geobox,
    download_and_extract_land_polygons,
)

# Configure logging ONCE here (root logger)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


def main(
    # Required options (no defaults → must be passed)
    tile_id: Annotated[
        str, typer.Option("--tile-id", "-t", help="Tile ID of GeoMAD tile (e.g. 8,10)")
    ],
    version: Annotated[
        str,
        typer.Option(
            "--version", "-v", help="Version string for output data (e.g. 0.0.1)"
        ),
    ],
    # Options with defaults
    output_bucket: Annotated[
        str,
        typer.Option(
            "--output-bucket",
            help="""Where to save results. 
        Production: dep-public-data
        Staging:    dep-public-staging
        """,
        ),
    ] = "dep-public-staging",
    collection_url_root: Annotated[
        str,
        typer.Option(
            "--collection-url-root",
            help="""STAC Collection URL root.
        Production: https://stac.digitalearthpacific.org/collections
        Staging:    https://stac.staging.digitalearthpacific.io/collections

        """,
        ),
    ] = "https://stac.staging.digitalearthpacific.io/collections",
    model_zip_uri: Annotated[
        str, typer.Option("--model-zip-uri", help="Deep Learning Model download path")
    ] = "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_vegheight/models/dep-veg-model-v1.zip",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Overwrite existing results"),
    ] = False,
    datetime: Annotated[
        str, typer.Option("--datetime", help="Datetime string (e.g. 2024)")
    ] = "2024",
    land_mask: Annotated[
        str,
        typer.Option(
            "--land-mask",
            help="Land mask to use osm, gadm or combined. Default is combined.",
        ),
    ] = "combined",
    testrun: Annotated[
        bool,
        typer.Option("--testrun/--no-testrun", help="Testrun: run without model inferencing to test data reading/writing"),
    ] = False
) -> None:

    log = logging.getLogger(tile_id)
    log.info("Starting processing")
    log.info(
        f"Input received: \ntile_id: {tile_id}\n version: {version}\n output_bucket: {output_bucket}\n datetime: {datetime}\n overwrite: {overwrite}\n model_zip_uri: {model_zip_uri}\n collection_url_root: {collection_url_root} \ntestrun: {testrun}"
    )

    grid = PACIFIC_GRID_10
    catalog = "https://stac.dataspace.copernicus.eu/v1"
    collection = "sentinel-2-global-mosaics"

    assert ((collection_url_root == 'https://stac.digitalearthpacific.org/collections' and output_bucket=='dep-public-data') or
            (collection_url_root == 'https://stac.staging.digitalearthpacific.io/collections' and output_bucket=='dep-public-staging')
    ), f'output_bucket={output_bucket} and collection_url_root={collection_url_root} not matched. Check the python run_task.py command'

    if 'staging' in collection_url_root:
        profile = 'staging'
    else:
        profile = 'prod'

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(profile=profile, cloud_defaults=True)
    client = boto3.client("s3")
    # show client information

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor="s2",
        dataset_id="vegheight",
        version=version,
        time=datetime,
    )

    # tile_id is a string like "45,55"
    stac_document = itempath.stac_path(tile_id)
    log.info(
        f"result will be saved to https://{output_bucket}.s3.us-west-2.amazonaws.com/{stac_document}"
    )

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

    tile_index = tuple(int(i) for i in tile_id.split(","))
    geobox = grid.tile_geobox(tile_index)

    searcher = PystacSearcher(
        catalog=catalog,
        collections=[collection],
        datetime=datetime,
    )

    loader = OdcLoader(
        bands=["B04", "B03", "B02", "observations"],  # , "B08"],
        chunks={"x": 1024, "y": 1024},
    )
    ## This includes:
    # - loads height model (1GB),
    # - loads one .pkl file for stats,
    # - downloading input dataset into numpy array shape [3,h,w] float32
    processor = VegProcessorKeepNonVegPixels(land_mask_src=land_mask, testrun=testrun)

    # Make sure we pass original client down to s3 writer and stac writer
    dep_client_to_s3 = partial(write_to_s3, client=client)
    # Custom writer so we write multithreaded
    writer = AwsDsCogWriter(
        itempath, write_multithreaded=True, write_function=dep_client_to_s3
    )
    # STAC making thing
    stac_creator = StacCreator(
        itempath=itempath,
        collection_url_root=collection_url_root,
        remote=True,
        make_hrefs_https=True,
        with_raster=True,
        write_function=dep_client_to_s3,
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
    log.info(paths)


if __name__ == "__main__":
    typer.run(main)
