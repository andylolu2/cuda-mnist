import torch

def cuda_2d_arange(
    n: int, 
    m: int, 
    dtype=torch.float16, 
    device="cuda",
    **kwargs,
):
    x = torch.arange(
        n * m, dtype=dtype, device=device, **kwargs
    ) / (n * m)
    x = x.reshape(m, n).T
    return x

