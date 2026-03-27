from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def collect_environment_snapshot(*, device: str | None = None) -> dict[str, str]:
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "pandas": package_version("pandas"),
        "pyarrow": package_version("pyarrow"),
        "numpy": package_version("numpy"),
    }
    if device is not None:
        snapshot["device"] = device
    return snapshot


def canonical_json_digest(payload: dict[str, Any] | list[Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def runner_checkpoint_path(runner: Any) -> str | None:
    bundle = getattr(runner, "bundle", None)
    checkpoint_path = getattr(bundle, "model_name_or_path", None)
    if checkpoint_path is None:
        return None
    return str(Path(checkpoint_path).expanduser().resolve())


def runner_device_name(runner: Any) -> str | None:
    device = getattr(runner, "device", None)
    return None if device is None else str(device)
