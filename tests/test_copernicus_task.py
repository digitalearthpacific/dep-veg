# test_copernicus_read_aws_stac_task.py
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

# ---- Adjust this to your real module ----
MODULE_PATH = "src.task"


@pytest.fixture
def env_keys(monkeypatch):
    """
    Provide distinct default AWS and CDSE Copernicus creds.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "orig_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "orig_secret")
    monkeypatch.setenv("CDSE_AWS_ACCESS_KEY_ID", "cdse_key")
    monkeypatch.setenv("CDSE_AWS_SECRET_ACCESS_KEY", "cdse_secret")
    yield
    # monkeypatch fixture auto-restores


def test_get_copernicus_rio_config_uses_cdse_keys_not_default(monkeypatch, env_keys):
    """
    Ensures get_copernicus_rio_config() builds its boto3 session from CDSE_*
    (when no profile) and that these are different from AWS_* defaults.
    """
    import importlib

    mod = importlib.import_module(MODULE_PATH)

    # Force "no profile" path
    monkeypatch.setattr(mod.os.path, "exists", lambda *_: False)

    # Capture args passed to boto3.Session
    created_sessions = []

    class FakeCreds:
        access_key = "cdse_key"
        secret_key = "cdse_secret"

    class FakeSession:
        def __init__(self, **kwargs):
            created_sessions.append(kwargs)

        def get_credentials(self):
            return FakeCreds()

    monkeypatch.setattr(mod.boto3, "Session", lambda **kwargs: FakeSession(**kwargs))

    aws, boto_session, cfg = mod.get_copernicus_rio_config(force_keys=True)

    # Assert boto3.Session called with CDSE creds (not AWS defaults)
    assert created_sessions, "Expected boto3.Session to be constructed"
    kwargs = created_sessions[0]
    assert kwargs["aws_access_key_id"] == os.environ["CDSE_AWS_ACCESS_KEY_ID"]
    assert kwargs["aws_secret_access_key"] == os.environ["CDSE_AWS_SECRET_ACCESS_KEY"]

    # Sanity: defaults and CDSE creds are distinct
    assert os.environ["AWS_ACCESS_KEY_ID"] != os.environ["CDSE_AWS_ACCESS_KEY_ID"]
    assert (
        os.environ["AWS_SECRET_ACCESS_KEY"] != os.environ["CDSE_AWS_SECRET_ACCESS_KEY"]
    )

    # Basic shape of config
    assert "AWS_S3_ENDPOINT" in cfg


def test_run_phase_separation_and_restore(monkeypatch, env_keys):
    """
    Ensures:
      - read phase uses Copernicus session via configure_rio(... aws={"session": copernicus_aws})
      - configure_s3_access called after read Env exits
      - writer.write runs after configure_s3_access (write phase)
      - AWS_* env not changed by read phase
    """
    import importlib

    mod = importlib.import_module(MODULE_PATH)

    events = []

    # ---- Patch get_copernicus_rio_config to return a known copernicus aws object ----
    copernicus_aws = object()
    gdal_opts = {"AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu"}
    monkeypatch.setattr(
        mod, "get_copernicus_rio_config", lambda **_: (copernicus_aws, gdal_opts)
    )

    # ---- Patch rasterio.Env to record enter/exit ----
    class FakeEnv:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            events.append("env_enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("env_exit")
            return False

    monkeypatch.setattr(mod.rasterio, "Env", FakeEnv)

    # ---- Patch configure_rio / configure_s3_access ----
    configure_rio_mock = MagicMock(
        side_effect=lambda *a, **k: events.append(("configure_rio", k))
    )
    configure_s3_mock = MagicMock(
        side_effect=lambda *a, **k: events.append(("configure_s3_access", k))
    )
    monkeypatch.setattr(mod, "configure_rio", configure_rio_mock)
    monkeypatch.setattr(mod, "configure_s3_access", configure_s3_mock)

    # ---- Patch set_stac_properties to be transparent ----
    monkeypatch.setattr(
        mod, "set_stac_properties", lambda input_d, processed_d: processed_d
    )

    # ---- Build a task with mocked components ----
    searcher = SimpleNamespace(
        search=MagicMock(side_effect=lambda area: events.append("search") or ["item"])
    )
    loader = SimpleNamespace(
        load=MagicMock(
            side_effect=lambda items, area: events.append("load") or "input_data"
        )
    )

    processor = SimpleNamespace(
        send_area_to_processor=True,
        process=MagicMock(
            side_effect=lambda input_data, **kw: events.append(("process", kw))
            or "processed_data"
        ),
    )

    writer = SimpleNamespace(
        write=MagicMock(
            side_effect=lambda data, task_id: events.append("write") or ["s3://out"]
        )
    )

    task = mod.CopernicusReadAwsStacTask(
        id="task-1",
        itempath=MagicMock(),
        area="area-1",
        searcher=searcher,
        loader=loader,
        processor=processor,
        writer=writer,
        post_processor=None,
        stac_creator=None,
        stac_writer=None,
        logger=MagicMock(),
    )

    # Capture original env for later comparison
    orig_env = {
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
    }

    out = task.run()
    assert out == ["s3://out"]

    # ---- Assertions: configure_rio called with copernicus aws session ----
    assert configure_rio_mock.call_count == 1
    _, kwargs = configure_rio_mock.call_args
    assert kwargs["cloud_defaults"] is True
    assert kwargs["aws"]["session"] is copernicus_aws  # key check

    # ---- Assertions: call order / phase separation ----
    # Expected rough order:
    # env_enter -> configure_rio -> search -> load -> process -> env_exit -> configure_s3_access -> write
    # (set_stac_properties just passes through)
    assert "env_enter" in events
    assert "env_exit" in events
    assert events.index("env_enter") < events.index(("configure_rio", kwargs))
    assert events.index("search") < events.index("env_exit")
    assert events.index("load") < events.index("env_exit")
    assert events.index(("process", {"area": "area-1"})) < events.index("env_exit")
    assert events.index("env_exit") < events.index(
        ("configure_s3_access", {"cloud_defaults": True})
    )
    assert events.index(
        ("configure_s3_access", {"cloud_defaults": True})
    ) < events.index("write")

    # ---- Assertions: default AWS env still present (not mutated by read) ----
    assert os.environ["AWS_ACCESS_KEY_ID"] == orig_env["AWS_ACCESS_KEY_ID"]
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == orig_env["AWS_SECRET_ACCESS_KEY"]

    # ---- Sanity: CDSE and AWS creds are different during the run ----
    assert os.environ["AWS_ACCESS_KEY_ID"] != os.environ["CDSE_AWS_ACCESS_KEY_ID"]
    assert (
        os.environ["AWS_SECRET_ACCESS_KEY"] != os.environ["CDSE_AWS_SECRET_ACCESS_KEY"]
    )
