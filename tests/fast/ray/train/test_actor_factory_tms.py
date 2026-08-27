import pytest

from miles.ray.train.actor_factory import (
    TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV,
    TMS_TRAIN_DISK_BACKUP_DIR_ENV,
    _configure_train_tms_env,
)


def test_train_tms_defaults_to_cpu_backup(monkeypatch):
    monkeypatch.delenv(TMS_TRAIN_DISK_BACKUP_DIR_ENV, raising=False)
    monkeypatch.delenv(TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV, raising=False)
    env_vars = {
        "TMS_INIT_ENABLE_DISK_BACKUP": "1",
        "TMS_DISK_BACKUP_DIR": "/stale",
    }

    _configure_train_tms_env(env_vars, "/patched-tms.so")

    assert env_vars == {
        "LD_PRELOAD": "/patched-tms.so",
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }


def test_train_tms_can_select_disk_backup(monkeypatch):
    monkeypatch.setenv(
        TMS_TRAIN_DISK_BACKUP_DIR_ENV,
        "/workspace/tms-disk-backup",
    )
    monkeypatch.setenv(TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV, "128")
    env_vars = {"UNRELATED": "kept"}

    _configure_train_tms_env(env_vars, "/patched-tms.so")

    assert env_vars == {
        "UNRELATED": "kept",
        "LD_PRELOAD": "/patched-tms.so",
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
        "TMS_INIT_ENABLE_DISK_BACKUP": "1",
        "TMS_DISK_BACKUP_DIR": "/workspace/tms-disk-backup",
        "TMS_DISK_BACKUP_CHUNK_MB": "128",
    }


@pytest.mark.parametrize(
    ("directory", "chunk_mb", "match"),
    [
        ("relative/path", None, "must be absolute"),
        ("/workspace/tms", "0", "must be positive"),
        (None, "128", "requires"),
    ],
)
def test_train_tms_rejects_invalid_disk_backup_config(
    monkeypatch,
    directory,
    chunk_mb,
    match,
):
    if directory is None:
        monkeypatch.delenv(TMS_TRAIN_DISK_BACKUP_DIR_ENV, raising=False)
    else:
        monkeypatch.setenv(TMS_TRAIN_DISK_BACKUP_DIR_ENV, directory)
    if chunk_mb is None:
        monkeypatch.delenv(TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV, raising=False)
    else:
        monkeypatch.setenv(TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV, chunk_mb)

    with pytest.raises(ValueError, match=match):
        _configure_train_tms_env({}, "/patched-tms.so")
