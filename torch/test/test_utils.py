import torch


def cuda_2d_arange(
    n: int,
    m: int,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
    **kwargs
):
    x = torch.arange(n * m, dtype=dtype, device=device, **kwargs) / (n * m)
    x = x.reshape(m, n).T  # simulate column-major order
    return x
