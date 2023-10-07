# How Matrix Multiplication Works on the GPU

One day, I woke up and realised how little I knew about how matrix multiplication works on the GPU. Having done so many machine learning projects, I feel ashamed (not really) that I don't know how the most important operation in ML works. I decided that I must go out of my PyTorch/JAX bubble and **venture into the horrors of CUDA**.

Matrix multiplication is often referred to as GEMM (**Ge**neral **M**atrix **M**ultiplication) in the CUDA world. Efficient matrix multiplication is highly hardware-specific and so the design of the algorithm maps closely to the hardware architecture.

:::info
**Brief overview of CUDA architecture**
| Level        | Memory hierarchy | Definition                                     |
| ------------ | ---------------- | ---------------------------------------------- |
| Device       | Global memory    | -                                              |
| Thread block | Shared memory    | A collection of warps, executed on a single SM |
| Warp         | -                | 32 threads, scheduled by the warp scheduler    |
| Thread       | Registers        | Executed on a single CUDA core                 |
:::

## Parallelising matrix multiplication

Suppose we want to multiply two matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ to make $C \in \mathbb{R}^{M \times N} = AB$. In this case, we say that the problem size is $(M, N, K)$. To parallelise this operation, we will split $A$ and $B$ into smaller matrices, matrix multiply them individually and concatenate the results to form $C$.

Specifically, we can partition $A$ row-wise (i.e. $M$ into chunks of size $M'$) and $B$ column-wise (i.e. $N$ into chunks of size $N'$). Mathematically, this looks like:

$$
\begin{align}
    A &= \begin{bmatrix}
        A_{1} \\
        \vdots \\
        A_{M/M'} \\
    \end{bmatrix} \text{, where } A_{i} \in \mathbb{R}^{M' \times K} \\
    B &= \begin{bmatrix}
        B_{1} & \cdots & B_{N/N'} \\
    \end{bmatrix} \text{, where } B_{j} \in \mathbb{R}^{K \times N'} \\
    C_{i,j} &= A_i B_j \\
    C &= \begin{bmatrix}
        C_{1,1} & \cdots & C_{1,N/N'} \\
        \vdots & \ddots & \vdots \\
        C_{M/M',1}  & \cdots & C_{M/M',N/N'} \\
    \end{bmatrix}
\end{align}
$$

We can see that each sub-matrix $C_{i,j} = A_i B_j$ in $C$ are independent of each other, so we can easily parallelise the computation of each sub-matrix. 

In practice, $K$ might be too large to directly load into memory and compute on. Instead, a typical implementation will also split $K$ into chunks of size $K'$, iterate over each chunk, and accumulate (by summing) over the partial results. This is known as **serial-K reduction**. (As opposed to [**parallel-K reduction**](#parallel-k-reduction)). Mathematically, this looks like:

$$
\begin{align}
    A_i &= \begin{bmatrix}
        A_{i,1} & \cdots & A_{i,K/K'}
    \end{bmatrix}, \text{ where } A_{i,k} \in \mathbb{R}^{M' \times K'} \\
    B_j &= \begin{bmatrix}
        B_{1,j} \\
        \vdots \\
        B_{K/K',j} \\
    \end{bmatrix}, \text{ where } B_{k,j} \in \mathbb{R}^{K' \times N'} \\
    C_{ij} &= \sum_{k=1}^{K/K'} A_{i,k} B_{k,j}
\end{align}
$$

:::info
At any point where the problem size is not divisible by the partition size, we need to add *padding*. This is typically done implicitly when we load the partitioned inputs ($A_{i,k}$ and $B_{k,j}$) into lower-level memory, such that the loaded partition (of size $M' \times K'$ for $A_{i,k}$ and $K' \times N'$ for $B_{k,j}$) is always "full". Though special care needs to be taken when writing the results back to global memory (to avoid out-of-bounds writes).
:::

On a high level, **three nested partitions** happen to parallelise matrix multiplication on the GPU:
1. The first partition happens on the **threadblock** level. Each thread block is thus responsible for computing $C_{i,j} = A_i B_j$.
2. The second partition happens on the **warp** level. The threadblock-level problem $C_{i,j}$ is further partitioned such that each warp is responsible for computing $C_{i,j}^{(m,n)} = A_i^{(m)} B_j^{(n)}$.
3. The third partition happens on the **instruction** level. Some instructions expects inputs of particular sizes. For example, second generation Tensor Cores operate on problems of size $(16, 8, 8)$ for fp16, whereas a direct implementation on CUDA cores by multiplication would simply operate on size $(1, 1, 1)$. The warp-level problem is thus even further partitioned such that each chunk has a suitable size for the instruction: $C_{i,j}^{(m,n)|(a,b)} = A_i^{(m)|(a)} B_j^{(n)|(b)}$.

## Data redundancy

Matrix multiplication can easily become memory-bound if we naively re-fetch data from global memory to registers everytime we perform a computation. The key observation is that many of the sub-inputs $A_i$ and $B_j$ are reused across different sub-matrix multiplications. For example, $A_1$ is required for $C_{1,1}$, $C_{1,2}$, ..., $C_{1,N/N'}$ and $B_1$ is required for $C_{1,1}$, $C_{2,1}$, ..., $C_{M/M',1}$. We can get the highest throughput if we can minimise redundant data movement and reuse the loaded data as much as possible.

In CUDA, there are three types of user-accessible memory:

| Memory type   | Capacity             | Bandwidth & latency  |
| ------------- | -------------------- | -------------------- |
| Global memory | :star: :star: :star: | :star:               |
| Shared memory | :star: :star:        | :star: :star:        |
| Registers     | :star:               | :star: :star: :star: |

Here's a high-level view of how each memory type is utilised:
1. Each threadblock would first load its required inputs into **shared memory**. All subsequent accesses to those data would thus be served by the shared memory instead of by the slower global memory.
2. Each warp would first load its required inputs into **registers**. All subsequent accesses to those data would be served by the fast registers directly.

## Diving into the details

### Thread block level

On the thread block level, the problem is partitioned into sub-problems of size $(M', N', K')$. Thus, each thread block is responsible for computing a fragment of $C$, denoted as $C_{i,j} \in \mathbb{R}^{M' \times N'}$:

$$
C_{i,j} = \sum_{k=1}^{K/K'} A_{i,k} B_{k,j}
$$

where $A_{i,k} \in \mathbb{R}^{M' \times K'}$ and $B_{k,j} \in \mathbb{R}^{K' \times N'}$.

Redundant data movement is minimised by loading the sub-inputs $A_{i,k}$ and $B_{k,j}$ into **shared memory**. When we are done with computing $A_{i,k} B_{k,j}$, the next chunk ($A_{i,k+1} and B_{k+1,j}) will be loaded into shared memory.

In my implementation, a partition size of $(M', N', K') = (128, 256, 32)$ is used.

### Warp level

On the warp level, the sub-problem is further partitioned into sub-sub-problems of size $(M'', N'', K'')$. Thus, each *warp* is responsible for computing a fragment of $C_{i,j}$, denoted as $C_{i,j}^{(m,n)} \in \mathbb{R}^{M'' \times N''}$:

$$
C_{i,j}^{(m,n)} = \sum_{k=1}^{K/K'} \sum_{l=1}^{K'/K''} A_{i,k}^{(m,l)} B_{k,j}^{(l,n)}
$$

where $A_{i,k}^{(m,l)} \in \mathbb{R}^{M'' \times K''}$ and $B_{k,j}^{(l,n)} \in \mathbb{R}^{K'' \times N''}$.

Redundant data movement is minimised by loading the sub-inputs $A_{i,k}^{(m,l)}$ and $B_{k,j}^{(l,n)}$ into **registers**. Any accesses to $A_{i,k}^{(m,l)}$ and $B_{k,j}^{(l,n)}$ *within* a warp will then be served by the fast registers.

:::info
It is worth noting that registers are **thread-level only**. This means that inputs in a register cannot be accessed by other threads in a warp. The exact way of how $A_{i,k}^{(m,l)}$ and $B_{k,j}^{(l,n)}$ are partitioned into the registers of each thread depends on the specific instruction used. The NVIDIA docs on [Warp Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions) gives a detailed description for each instruction.
:::

In my implementation, a warp-level partition size of $(M'', N'', K'') = (64, 64, 32)$ is used.

### Tensor core level

To actually perform the matrix multiplication, we use the **Tensor Cores** on the GPU. My GPU (RTX 2060) has the second generation Tensor Cores, which are specialised to solve problems of size $(M''', N''', K''') = (16, 8, 8)$. Thus, we even further partition $C_{i,j}^{(m,n)}$ into $C_{i,j}^{(m,n)|(a,b)} \in \mathbb{R}^{16 \times 8}$:

$$
C_{i,j}^{(m,n)|(a,b)} = \sum_{k=1}^{K/K'} \sum_{l=1}^{K'/K''} \sum_{p=1}^{K''/8} A_{i,k}^{(m,l)|(a,p)} B_{k,j}^{(l,n)|(p,b)}
$$

where $A_{i,k}^{(m,l)|(a,p)} \in \mathbb{R}^{16 \times 8}$ and $B_{k,j}^{(l,n)|(p,b)} \in \mathbb{R}^{8 \times 8}$. Here, all the inputs are already in the registers and thus the data movement overhead is minimal. 

:::info
Tensor Core operations are **warp-level instructions**, meaning that all the threads in a warp need to execute the Tensor Core instruction at the same time, collaboratively preparing the data to be consumed by **one** Tensor Core.
:::

## Choosing the partition sizes

So, given that we want to minimise data movement, we should just choose a partition size as large as possible to use all shared memory and registers, *right?* Well, not quite.

### Thread block partition size

Asymptotically, as the problem size increases, yes, we do want to use as much shared memory and registers as possible. However, for small problem sizes, we might run into two problems:
1. Have a large partition size means that we will have fewer thread blocks. As a result, we will not be able to utilise all the **Streaming Multiprocessors** (SMs) on the GPU.
2. For problem sizes that are not divisible by the partition size, we will have to add more padding to the inputs. As a result, some threads will be doing redundant computation.

### Warp partition size

In general, having a large warp partition size means there will be less redundant data movement, but at the cost of having fewer warps. Having too few warps means that we will not be able to hide the latency of memory accesses (because we might run out of other warps to schedule while the current warp is waiting for data).

### Instruction partition size

This is completely determined by what instructions your GPU supports. For my RTX 2060, the ptx instruction for fp16 Tensor Core matrix multiplication (with fp16 accumulation) is `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`, which expects inputs of size $(16, 8, 8)$.

## Even more optimisations

### Parallel-K reduction

In cases where $M$ and $N$ are small, we might only have a few thread blocks. For example in my implementation, I chose the thread block partition size to be $(M', N') = (128, 256)$. If the original problem size has $M \leq 128$ and $N \leq 256$, we will only have one thread block. This is an issue because each thread block can only execute in one **Streaming Multiprocessor** (SM) but most GPUs have multiple SMs. For example, my RTX 2060 has 30 SMs. This means that we are only using $\frac{1}{30}$ of the GPU's compute power!

In cases where $K$ is large (even though $M$ and $N$ are small), we can utilise more parallelism by doing **parallel-K reduction**. Recall that in *serial*-K reduction, each thread block iterates over the following sum:

$$
C_{i,j} = \sum_{k=1}^{K/K'} A_{i,k} B_{k,j}
$$

and accumulates the intermediate results into $C_{i,j}$. In parallel-K reduction, we instead assign each thread block to only compute *one element of the sum* (i.e. $A_{i,k} B_{k,j}$). This allows us to increase the number of thread blocks by a factor of $K/K'$, thus utilising more SMs. 

The caveat is that now, we need to *allocate more memory* to store the results from each thread block, and *invoke a second kernel* to perform a final reduction over the partial results to get $C_{i,j}$.

### Software pipelining

Normally, CUDA hides the latency of memory accesses by scheduling other warps to execute while a warp is waiting for data. This requires us to have enough warps to mask the latency. 

However, the number of warps is typically relatively small when doing GEMM. This is because the number of warps is limited by $\frac{\text{Number of registers per thread block}}{\text{Number of registers per warp}}$, and for GEMM we use a lot of registers per warp to hold as much data as possible. As a result, we might not have enough warps to mask the latency.

> The CUTLASS docs mention that *"The accumulator elements typically occupy at least half a thread's total register budget"*. 

To mitigate this effect, we can use **software pipelining**. In essence, we (manually) preload the inputs for the next iteration of the loop asynchronously using special instructions. While the inputs are being loaded, we can continue to compute on the current iteration. It is summarised by the following diagram:

<p align="center">
    <img src="https://github.com/andylolu2/cuda-nn/assets/66584117/42ae2a55-a3e9-4cab-b451-09df603c553c" width="600" alt="Loss graph">
</p>

# Appendix

## Resources

1. CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/