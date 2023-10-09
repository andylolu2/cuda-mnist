# Matrix Multiplication on GPU

This blog came from a sudden realisation of how little I knew about how matrix multiplication works on the GPU. Having done so many ML projects, I feel like I ought to understand how the most important operation in ML works: What is this "Tensor Core" thing? Why does everyone say "*data movement is the bottleneck*"? How fast can GPUs actually go? 

To answer these questions, I decided that I must go out of my PyTorch bubble and **venture into the abyss of CUDA**. I wrote this blog to document all that I have learnt, and hopefully anyone reading this wouldn't have to go through the pain of digging through CUDA docs/code as I did.

If there is anything that I've learnt in this journey, it is **concurrent matrix multiplication is HARD**. Efficient matrix multiplication heavily depends on the specific hardware you are using and the problem size you are trying to solve. There is no one-size-fits-all solution.

Enough nagging, let's dig in!

## Recap on GPU architecture

Let's remind ourselves how (NVIDIA) GPUs work. A GPU achieves parallelism by running many **threads**. Each thread is executed on a single CUDA core, though at a given time, only a subset of the threads are active, so there can be many more threads than CUDA cores available. Each thread, no matter it is active or not, has its own set of **registers**.

A group of 32 threads is known as a **warp**. All threads in a warp must execute together (or be inactive together). In most cases, there are a lot more inactive warps than active warps, and the **warp scheduler** is responsible for choosing which warps to execute at a given time. This allows the GPU to hide the latency of memory accesses by scheduling other warps to execute while a warp is waiting for data.

A group of warps is known as a **threadblock**. All warps in a threadblock are executed in the same **Streaming Multiprocessor** (SM). Each threadblock has its own **shared memory** that can be accessed by all threads in the threadblock. 

:::info
:pencil: **Note: Newer architectures**

From Volta architecture onwards, each thread also has its own program counter and call stack etc. This means that each thread in a warp can execute different instructions at the same time. 

The Volta architecture also introduced **Tensor Cores** that are specialised to solve matrix multiplications of specific sizes. Each active warp have access to one Tensor Core.

In the newest Hopper architecture, there is a concept of **threadblock clusters** that represents a group of threadblocks. It gives the user more fine-grained control over the scheduling of threadblocks and allows the shared memory of one threadblock to be access by other threadblocks in the same cluster.
::: 

## Parallelising matrix multiplication

Suppose we want to multiply two matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ to make $C \in \mathbb{R}^{M \times N} = AB$. (We say that the problem size is $(M, N, K)$ in this case). To parallelise this operation, we can split $A$ and $B$ into smaller matrices, matrix multiply them individually and concatenate the results to form $C$.

Specifically, we can partition $A$ row-wise (i.e. $M$ into chunks of size $M'$) and $B$ column-wise (i.e. $N$ into chunks of size $N'$) to give:

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

We can see that each sub-matrix $C_{i,j} = A_i B_j$ are independent of each other, so we can easily parallelise the computation of each sub-matrix. 

In practice, $K$ might be too large to directly load into memory and compute on. Instead, a typical implementation will also split $K$ into chunks of size $K'$, iterate over each chunk, and accumulate (by summing) over the partial results. This is known as **serial-K reduction**. (As opposed to [**parallel-K reduction**](#Parallel-K-reduction)). Mathematically, this looks like:

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
:pencil: **Note: Padding**

At any point where the problem size is not divisible by the partition size, we need to add *padding*. This is typically done implicitly when we load the partitioned inputs ($A_{i,k}$ and $B_{k,j}$) into lower-level memory where we ensure the loaded partition (of size $M' \times K'$ for $A_{i,k}$ and $K' \times N'$ for $B_{k,j}$) is always "full", by adding zeros. Special care needs to be taken when writing the results back to global memory to avoid out-of-bounds errors.
:::

On a high level, **three nested partitions** happen to parallelise matrix multiplication on the GPU:
1. The first partition happens on the **threadblock** level. Each threadblock is responsible for computing $C_{i,j} = A_i B_j$.
2. The second partition happens on the **warp** level. The threadblock-level problem $C_{i,j}$ is further partitioned such that each warp is responsible for computing $C_{i,j}^{(m,n)} = A_i^{(m)} B_j^{(n)}$.
3. The third partition happens on the **instruction** level. Some instructions expect inputs of particular sizes. For example, second generation Tensor Cores operate on problems of size $(16, 8, 8)$ for fp16, whereas a direct implementation on CUDA cores by scalar multiplication would simply operate on size $(1, 1, 1)$. The warp-level problem is thus even further partitioned such that each chunk has a suitable size for the instruction: $C_{i,j}^{(m,n)|(a,b)} = A_i^{(m)|(a)} B_j^{(n)|(b)}$.

## Data redundancy

Matrix multiplication can easily become memory-bound if we naively re-fetch data from global memory to registers everytime we perform a computation. The key observation is that many of the sub-inputs $A_i$ and $B_j$ are reused across different sub-matrix multiplications. For example, $A_1$ is required for $C_{1,1}$, $C_{1,2}$, ..., $C_{1,N/N'}$ and $B_1$ is required for $C_{1,1}$, $C_{2,1}$, ..., $C_{M/M',1}$. We can get the highest throughput if we can minimise redundant data movement and reuse the loaded data as much as possible.

In CUDA, there are three types of user-accessible memory:

| Memory type   | Capacity             | Bandwidth & latency  |
| ------------- | -------------------- | -------------------- |
| Global memory | :star: :star: :star: | :star:               |
| Shared memory | :star: :star:        | :star: :star:        |
| Registers     | :star:               | :star: :star: :star: |

Here's a high-level view of how each memory type is utilised:
1. Each threadblock would first load its required inputs from global memory into **shared memory**. All subsequent accesses to those data would thus be served by the shared memory instead of by the slower global memory.
2. Each warp would first load its required inputs from shared memory into **registers**. All subsequent accesses to those data would be served by the fast registers directly.

## Diving into the details

### Threadblock level

On the threadblock level, the problem is partitioned into sub-problems of size $(M', N', K')$. Thus, each threadblock is responsible for computing a fragment of $C$, denoted as $C_{i,j} \in \mathbb{R}^{M' \times N'}$:

$$
\begin{align}
C_{i,j} &= \sum_{k=1}^{K/K'} A_{i,k} B_{k,j}
&& \text{ where } A_{i,k} \in \mathbb{R}^{M' \times K'} \text{ and } B_{k,j} \in \mathbb{R}^{K' \times N'}
\end{align}
$$

Redundant data movement is minimised by loading the sub-inputs $A_{i,k}$ and $B_{k,j}$ into **shared memory**. When we are done with computing $A_{i,k} B_{k,j}$, the next chunk ($A_{i,k+1}$ and $B_{k+1,j}$) will be loaded into shared memory.

### Warp level

On the warp level, the sub-problem is further partitioned into sub-sub-problems of size $(M'', N'', K'')$. Thus, each *warp* is responsible for computing a fragment of $C_{i,j}$, denoted as $C_{i,j}^{(m,n)} \in \mathbb{R}^{M'' \times N''}$:

$$
\begin{align}
C_{i,j}^{(m,n)} &= \sum_{k=1}^{K/K'} \sum_{l=1}^{K'/K''} A_{i,k}^{(m,l)} B_{k,j}^{(l,n)}
&& \text{ where } A_{i,k}^{(m,l)} \in \mathbb{R}^{M'' \times K''} \text{ and } B_{k,j}^{(l,n)} \in \mathbb{R}^{K'' \times N''}
\end{align}
$$

Redundant data movement is minimised by loading the sub-inputs $A_{i,k}^{(m,l)}$ and $B_{k,j}^{(l,n)}$ into **registers**. Any accesses to $A_{i,k}^{(m,l)}$ and $B_{k,j}^{(l,n)}$ *within* a warp will then be served by the fast registers.

:::info
:pencil: **Note: Distributing data across registers**

It is worth noting that registers are **thread-level only**. This means that inputs in a register cannot be accessed by other threads in a warp. The exact way of how $A_{i,k}^{(m,l)}$ and $B_{k,j}^{(l,n)}$ are partitioned into the registers of each thread depends on the specific instruction used. The NVIDIA docs on [Warp Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions) gives a detailed description for each instruction.
:::

### Tensor core level

To actually perform the matrix multiplication, we use the **Tensor Cores** on the GPU. My GPU (RTX 2060) has the second generation Tensor Cores, which are specialised to solve problems of size $(M''', N''', K''') = (16, 8, 8)$. Thus, we even further partition $C_{i,j}^{(m,n)}$ into $C_{i,j}^{(m,n)|(a,b)} \in \mathbb{R}^{16 \times 8}$:

$$
\begin{align}
C_{i,j}^{(m,n)|(a,b)} &= \sum_{k=1}^{K/K'} \sum_{l=1}^{K'/K''} \sum_{p=1}^{K''/8} A_{i,k}^{(m,l)|(a,p)} B_{k,j}^{(l,n)|(p,b)}
&& \text{ where } A_{i,k}^{(m,l)|(a,p)} \in \mathbb{R}^{16 \times 8} \text{ and } B_{k,j}^{(l,n)|(p,b)} \in \mathbb{R}^{8 \times 8}
\end{align}
$$

where $A_{i,k}^{(m,l)|(a,p)} \in \mathbb{R}^{16 \times 8}$ and $B_{k,j}^{(l,n)|(p,b)} \in \mathbb{R}^{8 \times 8}$. Here, all the inputs are already in the registers and thus the data movement overhead is minimal. 

:::info
:pencil: **Note**

Tensor Core operations are **warp-level instructions**, meaning that all the threads in a warp need to execute the Tensor Core instruction at the same time, collaboratively preparing the data to be consumed by **one** Tensor Core.
:::

## Choosing the partition sizes

So, given that we want to minimise data movement, we should just choose a partition size as large as possible to use all shared memory and registers, *right?* Well, not quite.

### Threadblock partition size

Asymptotically, as the problem size increases, yes, we do want to use as much shared memory and registers as possible. However, for small problem sizes, we might run into two problems:
1. Have a large partition size means that we will have fewer threadblocks. As a result, we will not be able to utilise all the SMs on the GPU.
2. For problem sizes that are not divisible by the partition size, we will have to add more padding to the inputs. As a result, some threads will be doing redundant computation.

A typical implementation might use a partition size of $(M', N', K') = (128, 256, 32)$.

### Warp partition size

In general, having a large warp partition size means there will be less redundant data movement, but at the cost of having fewer warps. Having too few warps means that we will not be able to hide the latency of memory accesses (because we might run out of other warps to schedule while the current warp is waiting for data).

A typical implementation might use a partition size of $(M'', N'', K'') = (64, 64, 32)$.

### Instruction partition size

This is completely determined by what instructions your GPU supports. For my RTX 2060, the ptx instruction for fp16 Tenor Core matrix multiplication (with fp16 accumulation) is `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`, which expects inputs of size $(16, 8, 8)$.

## Even more optimisations

The above techniques can get us close to the theoretical peak performance of the GPU when the problem size is large. However, for smaller problem sizes, they are not as efficient. There are two common techniques to further improve the performance of matrix multiplication: **parallel-K reduction** and **software pipelining**.

### Parallel-K reduction

In cases where $M$ and $N$ are small, we might only have a few threadblocks. For example in my implementation, I chose the threadblock partition size to be $(M', N') = (128, 256)$. If the original problem size has $M \leq 128$ and $N \leq 256$, we will only have one threadblock, and so we are only utilising a fraction of the GPU's compute power! (For example, my RTX 2060 has 30 SMs, so to maximise utilisation we want at least 30 threadblocks.)

In cases where $K$ is large (even though $M$ and $N$ are small), we can utilise more parallelism by doing **parallel-K reduction**. Recall that in *serial*-K reduction, each threadblock iterates over the following sum:

$$
C_{i,j} = \sum_{k=1}^{K/K'} A_{i,k} B_{k,j}
$$

and accumulates the intermediate results into $C_{i,j}$. In parallel-K reduction, we instead assign each threadblock to only compute *one element of the sum* (i.e. $A_{i,k} B_{k,j}$). This allows us to increase the number of threadblocks by a factor of $K/K'$, thus utilising more SMs. 

The caveat is that now, we need to *allocate more memory* to store the results from each threadblock, and *invoke a second kernel* to perform a final reduction over the partial results to get $C_{i,j}$.

### Software pipelining

Normally, CUDA hides the latency of memory accesses by scheduling other warps to execute while a warp is waiting for data. This requires us to have enough warps to mask the latency. 

However, the number of warps is typically relatively small when doing GEMM. This is because the number of warps is limited by $\frac{\text{Number of registers per threadblock}}{\text{Number of registers per warp}}$, and for GEMM we use a lot of registers per warp to hold as much data as possible. As a result, we might not have enough warps to mask the latency.

> The CUTLASS docs mention that *"The accumulator elements typically occupy at least half a thread's total register budget"*. 

To mitigate this effect, we can use **software pipelining**. In essence, we (manually) preload the inputs for the next iteration of the loop asynchronously using special instructions. While the inputs are being loaded, we can continue to compute on the current iteration. It is summarised by the following diagram:

<p align="center">
    <img src="https://github.com/andylolu2/cuda-nn/assets/66584117/42ae2a55-a3e9-4cab-b451-09df603c553c" width="600" alt="Loss graph">
</p>

This is made possible by the fact that the GPU is like any modern CPU: it can pipeline memory accesses and arithmetic operations as long as there is no data dependency between them. This is known as **instruction-level parallelism**.

## Matrix multiplication in action

If you want to see how all these concepts come together in a real implementation, check out my [implementation of training MNIST from scratch with CUDA](https://github.com/andylolu2/cuda-nn). There, I trained a multi-layer perceptron on MNIST using CUDA, achieving 6x speedup over optimised PyTorch for medium-sized networks.

## References

1. CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/
2. CUDA docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
3. CUTLASS examples: https://github.com/NVIDIA/cutlass/tree/main/examples
