#include "optimizer.hpp"
#include "tensor.hpp"
#include "autograd.hpp"
#include <cuda_runtime.h>

// CUDA kernel to fill an array with a constant value.
static __global__ void fill_kernel(float* data, float val, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

static __global__ void sgd_update_kernel(float* param_data, const float* grad_data, float lr, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param_data[idx] -= lr * grad_data[idx];
    }
}

SGD::SGD(const vector<VarPtr>& params, float lr) : params(params), lr(lr) {}

void SGD::step() {
    for (auto param : params) {
        if (!param->grad_initialized) {
            cout << "[ERROR] Gradient not initialized for parameter!" << endl;
        }
        size_t n = param->data.size();
        if (param->data.device() == CUDA) {
            dim3 blockSize(256);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
            sgd_update_kernel<<<gridSize, blockSize>>>(param->data.data(), param->grad.data(), lr, n);
            cudaDeviceSynchronize();
        } else {
            for (size_t i = 0; i < n; ++i) {
                param->data.data()[i] -= lr * param->grad.data()[i];
            }
        }
    }
}

void SGD::zero_grad() {
    for (auto param : params) {
        size_t n = param->grad.size();
        if (param->grad.device() == CUDA) {
            dim3 blockSize(256);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
            fill_kernel<<<gridSize, blockSize>>>(param->grad.data(), 0.0f, n);
            cudaDeviceSynchronize();
        } else {
            param->grad.fill(0.0f);
        }
        param->grad_initialized = false;
        param->pending_count = 0;
    }
}