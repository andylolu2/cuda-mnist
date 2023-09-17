import torch
import time

if __name__ == "__main__":
    M, N, K = 3, 5, 9
    print_xy = True

    x = torch.arange(
        M * K, dtype=torch.float16, device="cuda"
    ) / (M * K)
    y = torch.arange(
        N * K, dtype=torch.float16, device="cuda"
    ) / (N * K)

    print(f"M: {M}, N: {N}, K: {K}")

    # print("version 1")
    # x_ = x.view(M, K)
    # y_ = y.view(K, N)
    # if print_xy:
    #     print(x_)
    #     print(y_)
    # print(x_ @ y_)

    # print("version 2")
    # x_ = x.view(M, K)
    # y_ = y.view(N, K).T
    # if print_xy:
    #     print(x_)
    #     print(y_)
    # print(x_ @ y_)

    # print("version 3")
    # x_ = x.view(K, M).T
    # y_ = y.view(K, N)
    # if print_xy:
    #     print(x_)
    #     print(y_)
    # print(x_ @ y_)

    print("version 4 <--")
    x_ = x.view(K, M).T
    y_ = y.view(N, K).T
    if print_xy:
        print(x_)
        print(y_)
    print(x_ @ y_)


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

