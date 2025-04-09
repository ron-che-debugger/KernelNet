#include "optimizer.hpp"

using namespace std;

// CUDA kernel to fill an array with a constant value.
static __global__ void fill_kernel(float *data, float val, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

// CUDA kernel for the SGD update step.
static __global__ void sgd_update_kernel(float *param_data, const float *grad_data, float lr, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param_data[idx] -= lr * grad_data[idx];
    }
}

// ----------------- Norm Clipping Kernels -----------------

// Kernel to compute the sum of squares of gradients.
// Uses shared memory reduction.
__global__ void compute_grad_norm_kernel(const float *grad, float *partial_sums, size_t size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // Sum squared gradients for this block.
    while (idx < size) {
        float val = grad[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel to scale gradients by a given factor.
__global__ void scale_grad_kernel(float *grad, float scale, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= scale;
    }
}

// ---------------------------------------------------------

SGD::SGD(const vector<VarPtr> &params, float lr, float clip_value)
    : params(params), lr(lr), clip_value(clip_value) {}

void SGD::step() {
    for (auto param : params) {
        if (!param->grad_initialized) {
            cout << "[ERROR] Gradient not initialized for parameter!" << endl;
        }
        size_t n = param->data.size();

        // Apply norm-based gradient clipping if clip_value > 0.
        if (clip_value > 0.0f) {
            if (param->grad.device() == CPU) {
                // Compute L2 norm over CPU.
                float norm = 0.0f;
                for (size_t i = 0; i < n; i++) {
                    float g = param->grad.data()[i];
                    norm += g * g;
                }
                norm = sqrt(norm);
                if (norm > clip_value) {
                    float scale = clip_value / norm;
                    for (size_t i = 0; i < n; i++) {
                        param->grad.data()[i] *= scale;
                    }
                }
            } else {
                // GPU branch: compute norm using a reduction kernel.
                dim3 blockSize(256);
                dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
                // Allocate temporary device memory for partial sums.
                float *d_partial;
                cudaMalloc(&d_partial, gridSize.x * sizeof(float));
                // Launch kernel with shared memory.
                compute_grad_norm_kernel<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(param->grad.data(), d_partial, n);
                cudaDeviceSynchronize();

                // Copy partial sums to host and compute total sum.
                vector<float> h_partial(gridSize.x);
                cudaMemcpy(h_partial.data(), d_partial, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(d_partial);
                float norm = 0.0f;
                for (int i = 0; i < gridSize.x; i++) {
                    norm += h_partial[i];
                }
                norm = sqrt(norm);
                if (norm > clip_value) {
                    float scale = clip_value / norm;
                    // Scale the gradient.
                    gridSize = dim3((n + blockSize.x - 1) / blockSize.x);
                    scale_grad_kernel<<<gridSize, blockSize>>>(param->grad.data(), scale, n);
                    cudaDeviceSynchronize();
                }
            }
        }

        // Update parameters using SGD.
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