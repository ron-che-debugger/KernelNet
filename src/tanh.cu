#include "tanh.hpp"

using namespace std;

Tanh::Tanh() {
    // No internal state is required.
}

VarPtr Tanh::forward(const VarPtr &input) {
    return TanhFunction::apply(input);
}

__global__ void tanh_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(in[idx]);
    }
}

__global__ void tanh_backward_kernel(const float *grad_out, const float *output, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = output[idx];
        grad_in[idx] = grad_out[idx] * (1.0f - y * y);
    }
}

VarPtr TanhFunction::apply(const VarPtr &input) {
    auto func = make_shared<TanhFunction>();
    // Save the input as a hard pointer.
    func->saved_input = input;

    Tensor out_tensor(input->data.size(), input->data.device());
    size_t size = input->data.size();
    if (input->data.device() == CPU) {
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = tanh(in_ptr[i]);
        }
    } else {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        tanh_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), size);
        cudaDeviceSynchronize();
    }
    func->tanh_output = out_tensor;
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad, "Tanh_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

vector<Tensor> TanhFunction::backward(const Tensor &grad_output) {
    size_t size = grad_output.size();
    Tensor grad_input(size, grad_output.device());
    if (grad_output.device() == CPU) {
        // CPU branch: use a simple loop.
        const float *grad_out_ptr = grad_output.data();
        const float *y_ptr = tanh_output.data();
        float *grad_in_ptr = grad_input.data();
        for (size_t i = 0; i < size; i++) {
            float val = y_ptr[i];
            grad_in_ptr[i] = grad_out_ptr[i] * (1.0f - val * val);
        }
    } else {
        // CUDA branch: use the backward kernel.
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        tanh_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), tanh_output.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}