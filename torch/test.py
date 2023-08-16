import torch
import time

if __name__ == "__main__":
    N = 8192
    TIMES = 1000
    WARMUP = 10

    print(f"Matrix size: {N}x{N}, times: {TIMES}")

    x = torch.randn(N, N, device="cuda", dtype=torch.float16)

    for _ in range(WARMUP):
        _ = x @ x
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(TIMES):
        _ = x @ x
    torch.cuda.synchronize()
    end = time.time()

    duration = end - start
    FLOPS = 2 * N * N * N * TIMES / duration
    iter_per_ms = TIMES / (duration * 1e3)

    print(f"iter/ms: {iter_per_ms:.4f}, {FLOPS / 1e12:.4f}TFLOPS")

