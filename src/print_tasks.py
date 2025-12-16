import json
import sys

import typer
from boto3 import client
from dep_tools.aws import object_exists
from dep_tools.namers import S3ItemPath
from grids import get_tiles
from dep_tools.searchers import PystacSearcher
from dep_tools.loaders import OdcLoader
from dep_tools.exceptions import EmptyCollectionError
from tqdm.auto import tqdm
from utils import quarter_start_dates
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer()


@app.command()
def print_tasks(
    overwrite: bool = False,
    limit: int = None,
    output_bucket: str = "dep-public-staging",
    datetime: str = "2024",
    version: str = "0.0.0",
):
    dates = quarter_start_dates(datetime)
    # Load the dep tiles
    tile_ids = get_tiles(resolution=10)
    if limit is not None:
        tile_ids = tile_ids[:limit]
    if not overwrite:
        s3_client = client("s3")
        valid_tile_ids = []

        catalog = "https://stac.dataspace.copernicus.eu/v1"
        collection = "sentinel-2-global-mosaics"
        copernicus_searcher = PystacSearcher(
            catalog=catalog,
            collections=[collection],
            datetime=datetime,
        )
        loader = OdcLoader(
            bands=["B04", "B03", "B02", "observations"],  # , "B08"],
            chunks={"x": 9600, "y": 9600},
        )
        for tile_id, geobox in tqdm(tile_ids):
            try:
                items = copernicus_searcher.search(geobox)
                data = loader.load(items, geobox)
                dates = [str(i).split("T")[0] for i in data.time.values]
            except EmptyCollectionError:
                continue
            for date in dates:
                itempath = S3ItemPath(
                    bucket=output_bucket,
                    sensor="s2",  ########################
                    dataset_id="vegheight",  ########################
                    version=version,
                    time=date,
                )
                stac_path = itempath.stac_path(tile_id)

                if not object_exists(output_bucket, stac_path, client=s3_client):
                    tile_id_ = ",".join([str(i) for i in tile_id])
                    valid_tile_ids.append(tile_id_)
                    break
        tile_ids = valid_tile_ids
    else:
        tile_ids = [",".join([str(i) for i in tile_id]) for tile_id, _ in tile_ids]

    json.dump(tile_ids, sys.stdout)


if __name__ == "__main__":
    app()
