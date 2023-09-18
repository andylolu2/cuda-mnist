import sys
import torch
import time

from test_utils import cuda_2d_arange

if __name__ == "__main__":
    # M, N, K = 8, 8, 64
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    print_ab = False

    a = cuda_2d_arange(M, K)
    b = cuda_2d_arange(K, N)

    print(f"M: {M}, N: {N}, K: {K}")

    if print_ab:
        print(a)
        print(b)
    print(a @ b + 1)


    # TIMES = 1000
    # WARMUP = 10

    # print(f"Matrix size: {N}x{N}, times: {TIMES}")

    # x = torch.randn(N, N, device="cuda", dtype=torch.float16)

    # for _ in range(WARMUP):
    #     _ = x @ x
    # torch.cuda.synchronize()

    # start = time.time()
    # for _ in range(TIMES):
    #     _ = x @ x
    # torch.cuda.synchronize()
    # end = time.time()

    # duration = end - start
    # FLOPS = 2 * N * N * N * TIMES / duration
    # iter_per_ms = TIMES / (duration * 1e3)

    # print(f"iter/ms: {iter_per_ms:.4f}, {FLOPS / 1e12:.4f}TFLOPS")

