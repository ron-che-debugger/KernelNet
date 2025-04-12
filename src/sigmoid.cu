#include "sigmoid.hpp"

using namespace std;

/**
 * @brief Default constructor for the Sigmoid module.
 *
 * No internal state is required.
 */
Sigmoid::Sigmoid() {
    // Constructor (no internal state needed).
}

/**
 * @brief Performs the forward pass of the Sigmoid module.
 *
 * This function computes the element-wise sigmoid activation on the input variable.
 *
 * @param input Input variable.
 * @return Output variable after applying the sigmoid function.
 */
VarPtr Sigmoid::forward(const VarPtr &input) {
    return SigmoidFunction::apply(input);
}

/**
 * @brief CUDA kernel for the forward pass of the sigmoid activation.
 *
 * Computes:
 *   out[i] = 1 / (1 + exp(-in[i]))
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param size Total number of elements.
 */
__global__ void sigmoid_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

/**
 * @brief CUDA kernel for the backward pass of the sigmoid activation.
 *
 * Computes the gradient with respect to the input as:
 *    grad_in[i] = grad_out[i] * y[i] * (1 - y[i])
 * where y is the output from the sigmoid activation.
 *
 * @param grad_out Pointer to the gradient from the next layer.
 * @param output Pointer to the output of the sigmoid (y).
 * @param grad_in Pointer to the gradient with respect to input.
 * @param size Total number of elements.
 */
__global__ void sigmoid_backward_kernel(const float *grad_out, const float *output, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = output[idx];
        grad_in[idx] = grad_out[idx] * y * (1.0f - y);
    }
}

/**
 * @brief Applies the sigmoid activation function and builds the autograd graph.
 *
 * Computes the forward pass on the CPU or CUDA depending on the device associated with the input tensor.
 * The computed output is cached for use during the backward pass.
 *
 * @param input Input variable.
 * @return Output variable after applying the sigmoid activation.
 */
VarPtr SigmoidFunction::apply(const VarPtr &input) {
    auto func = make_shared<SigmoidFunction>();

    // Save input for backward.
    func->saved_input = input;

    // Allocate output tensor with same size and on the same device as input.
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

/**
 * @brief Computes the backward pass for the sigmoid activation.
 *
 * Calculates the gradient with respect to the input using the stored output from the forward pass.
 * The derivative of the sigmoid function is computed as y * (1 - y).
 *
 * @param grad_output Gradient of the loss with respect to the output.
 * @return A vector containing the gradient with respect to the input.
 */
vector<Tensor> SigmoidFunction::backward(const Tensor &grad_output) {
    size_t size = grad_output.size();
    Tensor grad_input(size, grad_output.device());
    if (grad_output.device() == CPU) {
        const float *grad_out_ptr = grad_output.data();
        const float *y_ptr = sigmoid_output.data();
        float *grad_in_ptr = grad_input.data();
        for (size_t i = 0; i < size; i++) {
            float val = y_ptr[i];
            grad_in_ptr[i] = grad_out_ptr[i] * val * (1.0f - val);
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        sigmoid_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), sigmoid_output.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}