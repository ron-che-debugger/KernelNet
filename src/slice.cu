#include "slice.hpp"

using namespace std;

// CUDA kernel for the forward pass.
// It copies the slice from the input to the output.
// Input is interpreted as a 2D array of shape [batch_size, total_width] stored in row-major order.
// The output has shape [batch_size, slice_len], with slice_len = end - start.
__global__ void slice_forward_kernel(const float *in, float *out, int total_width, int slice_start, int slice_len, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * slice_len;
    if (idx < total) {
        int b = idx / slice_len;
        int i = idx % slice_len;
        out[idx] = in[b * total_width + slice_start + i];
    }
}

// CUDA kernel for the backward pass.
// It writes the gradient from the sliced output into the correct positions of the
// input gradient tensor. The positions that are not part of the slice are set to zero.
// Input grad_output is of shape [batch_size, slice_len] and grad_input is of shape
// [batch_size, total_width].
__global__ void slice_backward_kernel(const float *grad_out, float *grad_in, int total_width, int slice_start, int slice_len, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * total_width;
    if (idx < total) {
        int b = idx / total_width;
        int j = idx % total_width;
        float grad = 0.0f;
        // If j is within the slice, copy the corresponding gradient.
        if (j >= slice_start && j < slice_start + slice_len) {
            int i = j - slice_start;
            grad = grad_out[b * slice_len + i];
        }
        grad_in[idx] = grad;
    }
}

VarPtr SliceFunction::apply(const VarPtr &input, int batch_size, int start, int end) {
    auto func = make_shared<SliceFunction>();
    // Save the input as a hard pointer for backward.
    func->saved_input = input;

    func->batch_size = batch_size;
    func->start = start;
    func->end = end;

    // Compute total_width from the input.
    int total_size = input->data.size();
    func->total_width = total_size / batch_size;
    int slice_len = end - start;

    Tensor out_tensor(batch_size * slice_len, input->data.device());
    int total = batch_size * slice_len;

    if (input->data.device() == CPU) {
        // CPU branch: simple loop.
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < slice_len; i++) {
                out_ptr[b * slice_len + i] = in_ptr[b * func->total_width + start + i];
            }
        }
    } else {
        // CUDA branch.
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        slice_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), func->total_width, start, slice_len, batch_size);
        cudaDeviceSynchronize();
    }

    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad, "Slice_out");
    out->set_creator(func);
    func->output = out;

    return out;
}

vector<Tensor> SliceFunction::backward(const Tensor &grad_output) {
    int slice_len = end - start;
    int grad_in_size = batch_size * total_width;
    Tensor grad_input(grad_in_size, grad_output.device());

    if (grad_output.device() == CPU) {
        // CPU branch: initialize grad_input to zeros and then copy.
        float *grad_in_ptr = grad_input.data();
        const float *grad_out_ptr = grad_output.data();
        // Zero initialize.
        for (int i = 0; i < grad_in_size; i++) {
            grad_in_ptr[i] = 0;
        }
        // For each batch, copy the corresponding slice from grad_output.
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < slice_len; i++) {
                grad_in_ptr[b * total_width + start + i] = grad_out_ptr[b * slice_len + i];
            }
        }
    } else {
        // CUDA branch.
        int total = batch_size * total_width;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        // Zero initialize grad_input on device.
        cudaMemset(grad_input.data(), 0, grad_in_size * sizeof(float));
        slice_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_input.data(), total_width, start, slice_len, batch_size);
        cudaDeviceSynchronize();
    }

    return {grad_input};
}