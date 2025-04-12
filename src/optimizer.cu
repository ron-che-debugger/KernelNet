#include "optimizer.hpp"

using namespace std;

/**
 * @brief CUDA kernel to fill an array with a constant value.
 *
 * @param data Pointer to the data array.
 * @param val The constant value to fill.
 * @param size Total number of elements in the array.
 */
static __global__ void fill_kernel(float *data, float val, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

/**
 * @brief CUDA kernel for performing the SGD parameter update.
 *
 * Each element in the parameter array is updated as:
 *    param[i] = param[i] - lr * grad[i]
 *
 * @param param_data Pointer to the parameter array.
 * @param grad_data Pointer to the gradient array.
 * @param lr Learning rate.
 * @param size Total number of elements to update.
 */
static __global__ void sgd_update_kernel(float *param_data, const float *grad_data, float lr, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param_data[idx] -= lr * grad_data[idx];
    }
}

// ----------------- Norm Clipping Kernels -----------------

/**
 * @brief CUDA kernel to compute the sum of squares of gradient elements.
 *
 * Uses shared memory for block-level reduction.
 *
 * @param grad Pointer to the gradient array.
 * @param partial_sums Pointer to an array where each block writes its partial sum.
 * @param size Total number of elements in the gradient array.
 */
__global__ void compute_grad_norm_kernel(const float *grad, float *partial_sums, size_t size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // Stride loop over global array.
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

/**
 * @brief CUDA kernel to scale the gradient elements by a given factor.
 *
 * @param grad Pointer to the gradient array.
 * @param scale Scaling factor.
 * @param size Total number of gradient elements.
 */
__global__ void scale_grad_kernel(float *grad, float scale, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= scale;
    }
}

// ---------------------------------------------------------

/**
 * @brief Constructs an SGD optimizer.
 *
 * Initializes the optimizer with a set of parameters, a learning rate,
 * and an optional gradient clipping value.
 *
 * @param params Vector of parameters (as VarPtr) to optimize.
 * @param lr Learning rate.
 * @param clip_value Maximum allowed L2 norm for gradients (0 disables clipping).
 */
SGD::SGD(const vector<VarPtr> &params, float lr, float clip_value)
    : params(params), lr(lr), clip_value(clip_value) {}

/**
 * @brief Performs one update step for all parameters using SGD.
 *
 * For each parameter, if gradient clipping is enabled (clip_value > 0), then
 * the L2 norm of the gradient is computed and, if it exceeds clip_value,
 * the gradients are scaled accordingly. After clipping (if any), the parameters are updated.
 */
void SGD::step() {
    for (auto param : params) {
        if (!param->grad_initialized) {
            cout << "[ERROR] Gradient not initialized for parameter!" << endl;
        }
        size_t n = param->data.size();

        // Apply norm-based gradient clipping if clip_value > 0.
        if (clip_value > 0.0f) {
            if (param->grad.device() == CPU) {
                // Compute L2 norm on CPU.
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
                // GPU branch: compute norm using reduction kernel.
                dim3 blockSize(256);
                dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
                // Allocate temporary device memory for partial sums.
                float *d_partial;
                cudaMalloc(&d_partial, gridSize.x * sizeof(float));
                // Launch kernel with shared memory allocation.
                compute_grad_norm_kernel<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(param->grad.data(), d_partial, n);
                cudaDeviceSynchronize();

                // Copy partial sums to host to compute total norm.
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

/**
 * @brief Zeros out gradients for all parameters and resets gradient flags.
 *
 * Sets all gradient elements to 0 and marks gradients as not initialized.
 * Also resets the pending gradient count.
 */
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