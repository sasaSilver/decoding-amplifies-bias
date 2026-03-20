import torch


def resolve_torch_device(device: str | None) -> str:
    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        return "cpu"

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("Requested device='cuda' but CUDA is not available.")

    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        raise ValueError("Requested device='mps' but MPS is not available.")

    raise ValueError(f"Unsupported device '{device}'. Use auto, cpu, cuda, or mps.")
