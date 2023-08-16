#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

#include <array>
#include <iostream>

#include "lib/utils.h"

// Method for engine config generator based on heuristics
auto heurgen_method =
    [](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          .setHeurMode(CUDNN_HEUR_MODE_A)
                          .build();
    std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations "
              << std::endl;

    auto &engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    return engine_configs;
};

// Method for engine config generator based on fallback list
auto fallback_method =
    [](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        // .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                        .build();
    auto &fallback_list = fallback.getFallbackList();
    return fallback_list;
};

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s B N TIMES\n", argv[0]);
        return 0;
    }

    int64_t B = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    int64_t TIMES = atoi(argv[3]);
    printf("B = %ld, N = %ld, TIMES = %ld\n", B, N, TIMES);

    printf("CUDNN VERSION FROM cudnnGetVersion(): %zu\n", cudnnGetVersion());
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    int64_t dims[3] = {B, N, N};
    int64_t stride[3] = {N * N, N, 1};

    auto xTensor = cudnn_frontend::TensorBuilder()
                       .setDim(3, dims)
                       .setStride(3, stride)
                       .setId('x')
                       .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                       .setDataType(CUDNN_DATA_HALF)
                       .build();

    auto yTensor = cudnn_frontend::TensorBuilder()
                       .setDim(3, dims)
                       .setStride(3, stride)
                       .setId('y')
                       .setAlignment(16)
                       .setDataType(CUDNN_DATA_HALF)
                       .build();

    auto cTensor = cudnn_frontend::TensorBuilder()
                       .setDim(3, dims)
                       .setStride(3, stride)
                       .setId('c')
                       .setAlignment(16)
                       .setDataType(CUDNN_DATA_HALF)
                       .build();

    auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_HALF).build();

    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(xTensor)
                        .setbMatDesc(yTensor)
                        .setcMatDesc(cTensor)
                        .setmatmulDesc(matmulDesc)
                        .build();

    std::array<cudnn_frontend::Operation const *, 1> ops = {&matmulOp};

    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                       .setHandle(handle)
                       .setOperationGraph(ops.size(), ops.data())
                       .build();

    size_t xSize = B * N * N * sizeof(half);
    size_t ySize = B * N * N * sizeof(half);
    size_t cSize = B * N * N * sizeof(half);
    printf("xSize = %zu, ySize = %zu, cSize = %zu\n", xSize, ySize, cSize);
    void *x_ptr, *y_ptr, *c_ptr;
    check_cudnn_status(cudaMalloc(&x_ptr, xSize));
    check_cudnn_status(cudaMalloc(&y_ptr, ySize));
    check_cudnn_status(cudaMalloc(&c_ptr, cSize));

    void *data_ptrs[] = {x_ptr, y_ptr, c_ptr};
    int64_t uids[] = {'x', 'y', 'c'};

    auto variantPack =
        cudnn_frontend::VariantPackBuilder().setDataPointers(3, data_ptrs).setUids(3, uids).build();
    std::cout << "variantPack " << variantPack.describe() << std::endl;

    std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method,
                                                                    fallback_method};
    cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()),
                                                    sources.data());

    auto options = generator.cudnnFindPlan<
        cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(handle, opGraph,
                                                                                   variantPack);
    cudnn_frontend::ExecutionPlan plan = options.front();
    std::cout << "Plan chosen: " << plan.getTag() << std::endl;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < TIMES; i++) {
        checkCudnnErr(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    double nOps = (double)(2 * TIMES * B * N * N * N) / (double)(milliseconds / 1000.0);
    printf("N = %d, %.4f ops/ms, %.4f TFLOPS\n", N, TIMES / milliseconds, nOps / 1e12);

    check_cudnn_status(cudaFree(x_ptr));
    check_cudnn_status(cudaFree(y_ptr));
    check_cudnn_status(cudaFree(c_ptr));
    checkCudnnErr(cudnnDestroy(handle));
    return 0;
}
