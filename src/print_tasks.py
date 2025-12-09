import json
import sys

import requests
import typer
from boto3 import client
from dep_tools.aws import object_exists
from dep_tools.namers import S3ItemPath
from dep_tools.grids import get_tiles
app = typer.Typer()


@app.command()
def print_tasks(
    overwrite: bool = False,
    limit: int = None,
    output_bucket: str = "dep-public-staging",
    datetime: str = "2024",
    version: str = "0.0.0",
):
    # Load the dep tiles
    tile_ids = get_tiles(resolution=10)
    if limit is not None:
        tile_ids = tile_ids[:limit]

    if not overwrite:
        s3_client = client("s3")
        valid_tile_ids = []

        for tile_id, _ in tile_ids:
            itempath = S3ItemPath(
                bucket=output_bucket,
                sensor="s2",  ########################
                dataset_id="vegheight",  ########################
                version=version,
                time=datetime,
            )
            stac_path = itempath.stac_path(tile_id)

            if not object_exists(output_bucket, stac_path, client=s3_client):
                tile_id_ = ",".join([str(i) for i in tile_id])
                valid_tile_ids.append(tile_id_)

        tile_ids = valid_tile_ids
    else:
        tile_ids = [",".join([str(i) for i in tile_id]) for tile_id, _ in tile_ids]

    json.dump(tile_ids, sys.stdout)


if __name__ == "__main__":
    app()
