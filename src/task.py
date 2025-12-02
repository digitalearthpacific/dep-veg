import os
from contextlib import contextmanager
from logging import getLogger, Logger

import boto3
import rasterio as rio
from dep_tools.stac_utils import set_stac_properties  # wherever this lives
from dep_tools.task import AwsStacTask
from odc.loader._rio import GDAL_CLOUD_DEFAULTS
from rasterio.session import AWSSession


@contextmanager
def copernicus_read_env(profile_name="copernicus", **kwargs):
    """
    Temporary GDAL/AWS environment for reading from Copernicus S3.
    Restores environment after exit.
    If running in docker/k8s requires copernicus profile to be configured in ~/.aws/config
    [default]
    region = us-west-2

    [profile copernicus]
    aws_access_key_id=
    aws_secret_access_key=
    services = s3-specific

    [services s3-specific]
    s3 =
      endpoint_url = https://eodata.dataspace.copernicus.eu/
      # not needed but including
      max_concurrent_requests = 20
      max_bandwidth = 20MB/s
      use_accelerate_endpoint = true
    .

    Have added access key fallback
    CDSE_AWS_ACCESS_KEY_ID AND CDSE_AWS_SECRET_ACCESS_KEY preferred
    use AWS_ACCESS_KEY_ID AND AWS_SECRET_ACCESS_KEY as fallback
    -------------
    """
    old_env = os.environ.copy()
    try:
        # --- your setup ---
        gdal_opts = {**GDAL_CLOUD_DEFAULTS, **kwargs}
        gdal_opts["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
        gdal_opts["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
        gdal_opts["AWS_HTTPS"] = "YES"
        gdal_opts["AWS_VIRTUAL_HOSTING"] = "FALSE"
        gdal_opts["GDAL_HTTP_UNSAFESSL"] = "YES"

        # check if ~/.aws/config exists and has copernicus profile
        if not os.path.exists(os.path.expanduser("~/.aws/config")):
            # use aws credentials from env vars as fallback
            copernicus_access_key = os.environ.get("CDSE_AWS_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID"))
            copernicus_secret_key = os.environ.get("CDSE_AWS_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY"))
            session = boto3.Session(aws_access_key_id=copernicus_access_key,
                                    aws_secret_access_key=copernicus_secret_key)
        else:
            # Create a boto3 session from profile just for reads
            session = boto3.Session(profile_name=profile_name)
        aws = AWSSession(session)

        # rio.Env makes GDAL pick up this AWSSession for /vsis3/ reads
        with rio.Env(aws_session=aws, **gdal_opts):
            yield

    finally:
        # Restore env fully (important if writers rely on original creds)
        os.environ.clear()
        os.environ.update(old_env)


class CopernicusReadAwsStacTask(AwsStacTask):
    """
    Reads inputs using Copernicus creds+endpoint, writes outputs using default/original creds.
    """

    def __init__(
            self,
            *args,
            copernicus_profile="copernicus",
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.copernicus_profile = copernicus_profile
        self.logger: Logger = kwargs.get("logger", getLogger())

    def run(self):
        # ---- READ PHASE (temp copernicus session) ----
        with copernicus_read_env(profile_name=self.copernicus_profile):
            self.logger.info("Reading with Copernicus S3 credentials/session.")
            items = self.searcher.search(self.area)
            input_data = self.loader.load(items, self.area)

        # ---- PROCESS PHASE (no special creds needed) ----
        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        output_data = set_stac_properties(
            input_data,
            self.processor.process(input_data, **processor_kwargs),
        )

        if self.post_processor is not None:
            output_data = self.post_processor.process(output_data)

        # ---- WRITE PHASE (original/default creds restored) ----
        self.logger.info("Writing with original/default AWS credentials.")
        paths = self.writer.write(output_data, self.id)

        if self.stac_creator is not None and self.stac_writer is not None:
            stac_item = self.stac_creator.process(output_data, self.id)
            self.stac_writer.write(stac_item, self.id)

        return paths
