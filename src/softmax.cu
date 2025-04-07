#include "softmax.hpp"

Softmax::Softmax(int batch_size, int num_classes)
    : batch_size(batch_size), num_classes(num_classes) {}

VarPtr Softmax::forward(const VarPtr &input) {
    return SoftmaxFunction::apply(input, batch_size, num_classes);
}

__global__ void softmax_forward_kernel(const float *input, float *output, int num_classes) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * num_classes + tid; // one sample per block
    float val = input[index];
    sdata[tid] = val;
    __syncthreads();

    // Compute next power of two greater than or equal to num_classes.
    int n = num_classes;
    int s = 1;
    while (s < n)
        s *= 2;

    // Reduction to compute maximum over valid elements.
    for (int stride = s / 2; stride > 0; stride /= 2) {
        if (tid < stride && (tid + stride) < n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Compute exp(x - max_val) for numerical stability.
    float exp_val = expf(val - max_val);
    sdata[tid] = exp_val;
    __syncthreads();

    // Reduction to compute the sum of exponentials.
    for (int stride = s / 2; stride > 0; stride /= 2) {
        if (tid < stride && (tid + stride) < n) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    output[index] = exp_val / sum_val;
}

Tensor softmax_forward(const Tensor &input, int batch_size, int num_classes) {
    Tensor output(input.size(), input.device());

    if (input.device() == CPU) {
        // CPU branch.
        const float *in_data = input.data();
        float *out_data = output.data();
        for (int b = 0; b < batch_size; ++b) {
            int offset = b * num_classes;
            // Find max for numerical stability
            float max_val = in_data[offset];
            for (int i = 1; i < num_classes; ++i) {
                max_val = max(max_val, in_data[offset + i]);
            }
            // Compute exponentials and accumulate their sum
            float sum_exp = 0.0f;
            for (int i = 0; i < num_classes; ++i) {
                float exp_val = exp(in_data[offset + i] - max_val);
                out_data[offset + i] = exp_val;
                sum_exp += exp_val;
            }
            // Normalize.
            for (int i = 0; i < num_classes; ++i) {
                out_data[offset + i] /= sum_exp;
            }
        }
    } else {
        // CUDA branch.
        const float *in_ptr = input.data(); // device pointer
        float *out_ptr = output.data();     // device pointer

        // Launch one block per sample with num_classes threads per block.
        dim3 gridSize(batch_size);
        dim3 blockSize(num_classes);
        size_t sharedMemSize = num_classes * sizeof(float);
        // Kernel launch (defined below).
        softmax_forward_kernel<<<gridSize, blockSize, sharedMemSize>>>(in_ptr, out_ptr, num_classes);
        cudaDeviceSynchronize();
    }

    return output;
}

VarPtr SoftmaxFunction::apply(const VarPtr &input, int batch_size, int num_classes) {
    auto func = make_shared<SoftmaxFunction>();
    func->batch_size = batch_size;
    func->num_classes = num_classes;
    func->saved_input = input;

    // Call the unified softmax forward helper.
    Tensor out_tensor = softmax_forward(input->data, batch_size, num_classes);
    func->softmax_output = out_tensor; // Save computed output for backward.

    auto out = make_shared<Variable>(out_tensor, input->requires_grad, "Softmax_out");
    out->set_creator(func);
    func->inputs.push_back(input);
    func->output = out;

    return out;
}

// CUDA kernel for softmax backward.
// Each block processes one batch (row) of length num_classes.
// It computes the dot product: dot = sum_j (grad_out[j] * y[j]) using a reduction
// with a next power-of-two padded block size, then computes for each element:
// grad_in[i] = y[i] * (grad_out[i] - dot)
__global__ void softmax_backward_kernel(const float *grad_out, const float *y, float *grad_in, int num_classes) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int b = blockIdx.x; // one block per sample
    int offset = b * num_classes;

    // Load grad_out * y into shared memory (with padding for threads beyond num_classes).
    float prod = (tid < num_classes) ? grad_out[offset + tid] * y[offset + tid] : 0.0f;
    sdata[tid] = prod;
    __syncthreads();

    // Compute next power-of-two padded size.
    int n = num_classes;
    int s = 1;
    while (s < n)
        s *= 2;

    // Reduction to compute the dot product.
    for (int stride = s / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (tid + stride < n) {
                sdata[tid] += sdata[tid + stride];
            }
        }
        __syncthreads();
    }
    float dot = sdata[0];

    // Compute gradient for each valid element.
    if (tid < num_classes) {
        grad_in[offset + tid] = y[offset + tid] * (grad_out[offset + tid] - dot);
    }
}

Tensor softmax_backward(const Tensor &grad_output, const Tensor &y, int batch_size, int num_classes) {
    Tensor grad_input(grad_output.size(), grad_output.device());
    if (grad_output.device() == CPU) {
        const float *grad_out_ptr = grad_output.data();
        const float *y_ptr = y.data();
        float *grad_in_ptr = grad_input.data();
        for (int b = 0; b < batch_size; ++b) {
            int offset = b * num_classes;
            float dot = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                dot += grad_out_ptr[offset + j] * y_ptr[offset + j];
            }
            for (int i = 0; i < num_classes; ++i) {
                grad_in_ptr[offset + i] = y_ptr[offset + i] * (grad_out_ptr[offset + i] - dot);
            }
        }
    } else {
        // Compute next power-of-two for block size.
        int s = 1;
        while (s < num_classes)
            s *= 2;
        dim3 gridSize(batch_size);
        dim3 blockSize(s);
        size_t sharedMemSize = s * sizeof(float);
        softmax_backward_kernel<<<gridSize, blockSize, sharedMemSize>>>(grad_output.data(), y.data(), grad_input.data(), num_classes);
        cudaDeviceSynchronize();
    }
    return grad_input;
}

vector<Tensor> SoftmaxFunction::backward(const Tensor &grad_output) {
    // For the backward pass, dispatch to our CUDA kernel if needed.
    Tensor grad_in = softmax_backward(grad_output, softmax_output, batch_size, num_classes);
    if (grad_output.device() != CPU) {
        grad_in.toCUDA();
    }
    return {grad_in};
}