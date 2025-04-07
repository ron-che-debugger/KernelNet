#include "sigmoid.hpp"

using namespace std;

Sigmoid::Sigmoid() {
    // Constructor (no internal state needed).
}

VarPtr Sigmoid::forward(const VarPtr &input) {
    return SigmoidFunction::apply(input);
}

__global__ void sigmoid_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float *grad_out, const float *output, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = output[idx];
        grad_in[idx] = grad_out[idx] * y * (1.0f - y);
    }
}

VarPtr SigmoidFunction::apply(const VarPtr &input) {
    auto func = make_shared<SigmoidFunction>();

    func->saved_input = input;

    Tensor out_tensor(input->data.size(), input->data.device());
    size_t size = input->data.size();
    if (input->data.device() == CPU) {
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = 1.0f / (1.0f + exp(-in_ptr[i]));
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        sigmoid_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), size);
        cudaDeviceSynchronize();
    }
    func->sigmoid_output = out_tensor;
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad);
    out->set_creator(func);
    func->output = out;
    return out;
}

vector<Tensor> SigmoidFunction::backward(const Tensor &grad_output) {
    size_t size = grad_output.size();
    Tensor grad_input(size, grad_output.device());
    if (grad_output.device() == CPU) {
        // CPU branch: use loop
        const float *grad_out_ptr = grad_output.data();
        const float *y_ptr = sigmoid_output.data();
        float *grad_in_ptr = grad_input.data();
        for (size_t i = 0; i < size; i++) {
            float val = y_ptr[i];
            grad_in_ptr[i] = grad_out_ptr[i] * val * (1.0f - val);
        }
    } else {
        // CUDA branch: use backward kernel.
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        sigmoid_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), sigmoid_output.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}