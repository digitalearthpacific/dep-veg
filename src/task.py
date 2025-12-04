import os
from contextlib import contextmanager
from logging import getLogger, Logger

import boto3
import rasterio as rio
from dep_tools.stac_utils import set_stac_properties  # wherever this lives
from dep_tools.task import AwsStacTask
from odc.loader._rio import GDAL_CLOUD_DEFAULTS, configure_rio
from rasterio.session import AWSSession

logger = getLogger(__name__)


@contextmanager
def copernicus_read_env(profile_name="copernicus", force_keys=False, **kwargs):
    """
    NOTE: Unfortunately contextlib.contextmanager cannot be nested well with dask delayed functions
    so this is of limited use inside dask tasks. Leaving for posterity. But feel free to clear
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
    original_aws_key_id = os.environ.pop("AWS_ACCESS_KEY_ID", None)
    original_aws_key_secret = os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    original_session = os.environ.pop("AWS_SESSION_TOKEN", None)
    try:
        gdal_opts = {**GDAL_CLOUD_DEFAULTS, **kwargs}
        gdal_opts.update({
            "GDAL_HTTP_TCP_KEEPALIVE": "YES",
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "GDAL_HTTP_UNSAFESSL": "YES",
        })

        use_profile = os.path.exists(os.path.expanduser("~/.aws/config")) and not force_keys

        if use_profile:
            logger.debug("Using copernicus profile")
            session = boto3.Session(profile_name=profile_name)
        else:
            logger.debug("Not using copernicus profile pulling keys")
            copernicus_access_key = os.environ.get("CDSE_AWS_ACCESS_KEY_ID") or original_aws_key_id
            copernicus_secret_key = os.environ.get("CDSE_AWS_SECRET_ACCESS_KEY") or original_aws_key_secret

            if not copernicus_access_key or not copernicus_secret_key:
                raise RuntimeError(
                    "Copernicus credentials not found. "
                    "Set CDSE_AWS_ACCESS_KEY_ID and CDSE_AWS_SECRET_ACCESS_KEY "
                    "or configure ~/.aws/config profile 'copernicus'."
                )

            session = boto3.Session(
                aws_access_key_id=copernicus_access_key,
                aws_secret_access_key=copernicus_secret_key,
                aws_session_token=original_session,
            )

        aws = AWSSession(session)
        with rio.Env(aws_session=aws, **gdal_opts) as env:
            yield env

    finally:
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
            force_keys=False,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.copernicus_profile = copernicus_profile
        self.logger: Logger = kwargs.get("logger", getLogger())

    def run(self):
        # ---- READ PHASE (temp copernicus session) ----
        aws, boto_session, gdal_opts = get_copernicus_rio_config()
        # using raserio env probably unnecessary as configure_rio is actually passing down the config to the dask client

        configure_rio(cloud_defaults=True, aws={"session": boto_session}, **gdal_opts)
        items = self.searcher.search(self.area)
        input_data = self.loader.load(items, self.area)

        # ---- PROCESS PHASE (no special creds needed) ----
        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        #
        output_data = set_stac_properties(
            input_data,
            self.processor.process(input_data, **processor_kwargs),
        )

        if self.post_processor is not None:
            output_data = self.post_processor.process(output_data)

        # ---- WRITE PHASE (original/default creds restored) ----
        # configure_s3_access(cloud_defaults=True)
        paths = self.writer.write(output_data, self.id)

        if self.stac_creator is not None and self.stac_writer is not None:
            stac_item = self.stac_creator.process(output_data, self.id)
            self.stac_writer.write(stac_item, self.id)

        return paths


def get_copernicus_rio_config(profile_name="copernicus", force_keys=False):
    """
    Returns the GDAL/Rasterio configuration dictionary for Copernicus.
    Does NOT set environment variables globally.
    """
    # 1. Determine Credentials
    use_profile = os.path.exists(os.path.expanduser("~/.aws/config")) and not force_keys

    access_key = None
    secret_key = None

    if use_profile:
        logger.debug("Using copernicus profile")
        boto_session = boto3.Session(profile_name=profile_name)
        creds = boto_session.get_credentials()
        if creds:
            access_key = creds.access_key
            secret_key = creds.secret_key
    else:
        logger.debug("Not using copernicus profile, pulling keys from env")
        # specific CDSE keys or fallback to standard AWS keys
        access_key = os.environ.get("CDSE_AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("CDSE_AWS_SECRET_ACCESS_KEY")
        boto_session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    if not access_key or not secret_key:
        raise RuntimeError(
            "Copernicus credentials not found. "
            "Set CDSE_AWS_ACCESS_KEY_ID and CDSE_AWS_SECRET_ACCESS_KEY "
            "or configure ~/.aws/config profile 'copernicus'."
        )

    # 2. Build the GDAL Configuration Dictionary
    # These settings will be embedded into the dask graph
    config = {
        "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
        "AWS_HTTPS": "YES",
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "GDAL_HTTP_UNSAFESSL": "YES",
        "GDAL_HTTP_TCP_KEEPALIVE": "YES",
    }

    aws = AWSSession(boto_session)
    return aws, boto_session, config
