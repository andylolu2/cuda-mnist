# Training MNIST from Scratch with CUDA

## But why?

I wanted to know how much overhead is added by Python-based ML frameworks like PyTorch, and how much faster we can possibly get by going lower-level (CUDA). It is also a good exercise for me to learn CUDA.

## So... how slow is PyTorch?

It's... pretty slow, at least for small networks. Even using PyTorch 2.0's `torch.compile` functionality (with `mode="max-autotune"` and `fullgraph=True`, which is supposed to remove all Python overhead), it is can still be up to $6$ times slower than CUDA!

This overhead goes down as the network gets larger, never completely goes away. It asympotically approaches $\approx1.2$ times slower than CUDA.

<p align="center">
    <img src="./docs/time_graph.png" width="600" alt="Time graph">
</p>

There are a few reasons why PyTorch is (asymptotically) slower than CUDA:
1. The CUDA implementation pre-allocate all the tensors, so there is no memory allocation overhead. PyTorch (might) allocate and deallocate memory within each training step.
2. The CUDA implementation uses fp16 accumulation for matrix multiplication, which I found to be faster than fp32 accumulation. (I think) PyTorch uses fp32 accumulation.
3. I tuned the hyperparameters for the CUDA implementation specifically for my hardware. I'm not sure if `max-autotune` does the same for PyTorch.

> [!NOTE]
> I applied a few optimisations to both implementations.
> 1. I pre-loaded all data into memory in oroder to minimise the host-device data transfer overhead.
> 2. I allowed the PyTorch implementation to have a few warmup steps before timing, to allow the JIT compiler to compile the graph.

## Loss curve sanity check

Just to make sure my CUDA implementation is correct. The loss curves look pretty much identical.

<p align="center">
    <img src="./docs/loss_graph.png" width="600" alt="Loss graph">
</p>

## Things I learned

### How matrix multiplication works

The [CUTLASS docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/) has a good explanation of how matrix multiplication works. I'll try to summarise it here.

Matrix multiplication is often referred to as GEMM (General Matrix Multiplication) in the CUDA world. Efficient matrix multiplication is highly hardware-specific and so the design of the algorithm maps closely to the hardware architecture.

> [!NOTE]
> **Brief overview of CUDA architecture**
>
> | Level        | Memory hierarchy | Hardware feature(s)         |
> | ------------ | ---------------- | --------------------------- |
> | Device       | Global memory    | GPU                         |
> | Thread block | Shared memory    | Streaming multiprocessor    |
> | Warp         | -                | Warp scheduler, Tensor core |
> | Thread       | Registers        | ALUs                        |

### Thread block level

Suppose we want to multiply two matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ to make $C \in \mathbb{R}^{M \times N} = AB$. To parallelise this operation, we will split $A$ and $B$ into smaller matrices, mat-mul them individually and concatenate the results to form $C$.

Specifcally, we can partition $M$ into chunks of size $M'$ and $N$ into chunks of size $N'$. Mathematically, this looks like:

$$
\begin{align}
    A &= \begin{bmatrix}
        A_{1} \\
        A_{2} \\
        \vdots \\
        A_{M/M'} \\
    \end{bmatrix} \text{, where } A_{i} \in \mathbb{R}^{M' \times K} \\
    B &= \begin{bmatrix}
        B_{1} & B_{2} & \cdots & B_{N/N'} \\
    \end{bmatrix} \text{, where } B_{j} \in \mathbb{R}^{K \times N'} \\
    C &= \begin{bmatrix}
        A_{1}B_{1} & A_{1}B_{2} & \cdots & A_{1}B_{N/N'} \\
        A_{2}B_{1} & A_{2}B_{2} & \cdots & A_{2}B_{N/N'} \\
        \vdots & \vdots & \ddots & \vdots \\
        A_{M/M'}B_{1} & A_{M/M'}B_{2} & \cdots & A_{M/M'}B_{N/N'} \\
    \end{bmatrix}
\end{align}
$$

We can see that each sub-matrix $C_{ij}$ in $C$ are independent of each other, so we can easily parallelise the computation of each sub-matrix. We do this by assigning each sub-matrix-multiplication problem to a thread block.

### Warp level

TBC
