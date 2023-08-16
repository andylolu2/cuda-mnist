#include <cudnn.h>
#include <cudnn_frontend.h>

void matmul_bias_relu_fwd(cudnn_frontend::Tensor& x, void* x_ptr, cudnn_frontend::Tensor& w,
                          void* w_ptr, cudnn_frontend::Tensor& b, void* b_ptr,
                          cudnn_frontend::Tensor& y, void* y_ptr, cudnnHandle_t handle);

cudnn_frontend::Tensor linear_backward(cudnn_frontend::Tensor& dLdy, cudnn_frontend::Tensor& x,
                                       std::string name,
                                       std::vector<cudnn_frontend::Operation>& ops,
                                       std::set<std::pair<uint64_t, void*>>& data_ptrs);