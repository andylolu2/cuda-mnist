# Training MNIST from Scratch with CUDA

## But why?

I wanted to know how much overhead is added by Python-based ML frameworks like PyTorch, and how much faster we can possibly get by going lower-level (CUDA). It is also a good exercise for me to learn CUDA.

## So... how slow is PyTorch?

It's... pretty slow, at least for small networks. Even using PyTorch 2.0's `torch.compile` functionality (with `mode="max-autotune"` and `fullgraph=True`, which is supposed to remove all Python overhead), it can still be up to $6$ times slower than CUDA!

This overhead goes down as the network gets larger, never completely goes away. It asymptotically approaches $\approx1.2$ times slower than CUDA.

<p align="center">
    <img src="./docs/time_graph.png" width="600" alt="Time graph">
</p>

There are a few reasons why PyTorch is (asymptotically) slower than CUDA:
1. The CUDA implementation pre-allocate all the tensors, so there is no memory allocation overhead. PyTorch (might) allocate and deallocate memory within each training step.
2. The CUDA implementation uses fp16 accumulation for matrix multiplication, which I found to be faster than fp32 accumulation. (I think) PyTorch uses fp32 accumulation.
3. I tuned the hyperparameters for the CUDA implementation specifically for my hardware. I'm not sure if `max-autotune` does the same for PyTorch.

> [!NOTE]
> I applied a few optimisations to both implementations.
> 1. I preloaded all data into memory in order to minimise the host-device data transfer overhead.
> 2. I allowed the PyTorch implementation to have a few warm-up steps before timing, to allow the JIT compiler to compile the graph.

## Loss curve sanity check

Just to make sure my CUDA implementation is correct. The loss curves look pretty much identical.

<p align="center">
    <img src="./docs/loss_graph.png" width="600" alt="Loss graph">
</p>

## Things I learned

### How matrix multiplication works

The [CUTLASS docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/) have a good explanation of how matrix multiplication works. I'll try to summarise it here.

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

Suppose we want to multiply two matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ to make $C \in \mathbb{R}^{M \times N} = AB$. We say that the global problem size is $(M, N, K)$. To parallelise this operation, we will split $A$ and $B$ into smaller matrices, matrix multiply them individually and concatenate the results to form $C$.

Specifically, we can partition $M$ into chunks of size $M'$ and $N$ into chunks of size $N'$. Mathematically, this looks like:

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

We can see that each sub-matrix $C_{ij} = A_i B_j$ in $C$ are independent of each other, so we can easily parallelise the computation of each sub-matrix. We do this by assigning each sub-matrix-multiplication problem of size $(M', N', K)$ to a **thread block**.

In practice, $K$ might be too large to directly compute on. Instead, a typical implementation will split $K$ into chunks of size $K'$ and sum over the partial results. Mathematically, this looks like:

$$
\begin{align}
    A_i &= \begin{bmatrix}
        A_i^{(1)} & A_i^{(2)} & \cdots & A_i^{(K/K')}
    \end{bmatrix}, \text{ where } A_i^{(k)} \in \mathbb{R}^{M' \times K'} \\
    B_j &= \begin{bmatrix}
        B_j^{(1)} \\
        B_j^{(2)} \\
        \vdots \\
        B_j^{(K/K')} \\
    \end{bmatrix}, \text{ where } B_j^{(k)} \in \mathbb{R}^{K' \times N'} \\
    C_{ij}^{(k)} &= A_i^{(k)} B_j^{(k)} \\
    C_{ij} &= \sum_{k=1}^{K/K'} C_{ij}^{(k)}
\end{align}
$$

This is known as **serial-K reduction**. In the implementation, we use $(M', N', K') = (128, 256, 64)$.

### Warp level

Warps within the same thread block must collective solve a sub-problem of size $(M', N', K')$. We further parallelise this across warps by splitting into sub-sub-problems of size $(M'', N'', K'')$. Mathematically, this looks like:

$$
\begin{align}
    A_i^{(k)} &= \begin{bmatrix}
        A_i^{(k)(1,1)} & \cdots & A_i^{(k)(1, K''/K')} \\
        \vdots & \ddots & \vdots \\
        A_i^{(k)(M''/M', 1)}  & \cdots & A_i^{(k)(M''/M', K''/K')} \\
    \end{bmatrix}, \text{ where } A_i^{(k)(m, n)} \in \mathbb{R}^{M'' \times K''} \\
    B_j^{(k)} &= \begin{bmatrix}
        B_j^{(k)(1,1)} & \cdots & B_j^{(k)(1, N''/N')} \\
        \vdots & \ddots & \vdots \\
        B_j^{(k)(K''/K', 1)}  & \cdots & B_j^{(k)(K''/K', N''/N')} \\
    \end{bmatrix}, \text{ where } B_j^{(k)(m, n)} \in \mathbb{R}^{K'' \times N''} \\
    C_{ij}^{(k)(m, n)} &= \sum_{l=1}^{K''/K'} A_i^{(k)(m, l)} B_j^{(k)(l, n)} \\
    C_{ij}^{(k)} &= \begin{bmatrix}
        C_{ij}^{(k)(1,1)} & \cdots & C_{ij}^{(k)(1, N''/N')} \\
        \vdots & \ddots & \vdots \\
        C_{ij}^{(k)(M''/M', 1)}  & \cdots & C_{ij}^{(k)(M''/M', N''/N')} \\
    \end{bmatrix} \\
\end{align}
$$

> [!NOTE]
>
> We didn't split the problem into chunks of $(M'', N'', K'')$ at the thread block level to improve memory-movement efficiency. The warps within the same thread block 
