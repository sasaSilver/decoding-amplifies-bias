from unittest.mock import patch

import pytest

from app.device import resolve_torch_device


def test_resolve_torch_device_defaults_to_cpu_when_no_accelerator_is_available() -> None:
    with (
        patch("app.device.torch.cuda.is_available", return_value=False),
        patch("app.device.torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_torch_device("auto") == "cpu"


def test_resolve_torch_device_accepts_explicit_cpu() -> None:
    assert resolve_torch_device("cpu") == "cpu"


def test_resolve_torch_device_rejects_unavailable_cuda() -> None:
    with patch("app.device.torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="device='cuda'"):
            resolve_torch_device("cuda")
