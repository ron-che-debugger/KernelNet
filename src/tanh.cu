#include "tanh.hpp"

namespace kernelnet {
namespace nn {
/**
 * @brief Default constructor for the Tanh module.
 *
 * No internal state is required for the Tanh activation.
 */
Tanh::Tanh() {
    // No internal state is required.
}

/**
 * @brief Performs the forward pass of the Tanh module.
 *
 * Applies the hyperbolic tangent activation function element-wise on the input variable.
 *
 * @param input Input variable.
 * @return Output variable after applying tanh.
 */
VarPtr Tanh::forward(const VarPtr &input) {
    return TanhFunction::apply(input);
}

/**
 * @brief CUDA kernel for the forward pass of the tanh activation.
 *
 * Computes the hyperbolic tangent for each element of the input:
 *     out[i] = tanhf(in[i])
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param size Total number of elements in the array.
 */
__global__ void tanh_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(in[idx]);
    }
}

/**
 * @brief CUDA kernel for the backward pass of the tanh activation.
 *
 * Computes the gradient with respect to the input using the derivative of tanh:
 *     grad_in[i] = grad_out[i] * (1 - y[i]^2)
 * where y[i] is the output from the forward pass.
 *
 * @param grad_out Pointer to the gradient of the loss with respect to the output.
 * @param output Pointer to the output array from the forward pass.
 * @param grad_in Pointer to the output gradient array to be computed.
 * @param size Total number of elements.
 */
__global__ void tanh_backward_kernel(const float *grad_out, const float *output, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = output[idx];
        grad_in[idx] = grad_out[idx] * (1.0f - y * y);
    }
}

/**
 * @brief Applies the Tanh activation function and builds the autograd graph.
 *
 * Computes the forward pass on either the CPU or CUDA, storing the resulting output
 * for use in the backward pass.
 *
 * @param input Input variable.
 * @return Output variable after applying tanh.
 */
VarPtr TanhFunction::apply(const VarPtr &input) {
    auto func = make_shared<TanhFunction>();
    // Save input for backward propagation.
    func->saved_input = input;

    // Allocate output tensor with the same size and device as the input.
    Tensor out_tensor(input->data.size(), input->data.device());
    size_t size = input->data.size();

    if (input->data.device() == CPU) {
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = tanh(in_ptr[i]);
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        tanh_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), size);
        cudaDeviceSynchronize();
    }

    // Cache the computed output for the backward pass.
    func->tanh_output = out_tensor;
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad, "Tanh_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the tanh activation.
 *
 * Uses the derivative of tanh, where the gradient is computed as:
 *     grad_in[i] = grad_out[i] * (1 - tanh(x)^2)
 * This function supports both CPU and CUDA implementations.
 *
 * @param grad_output Gradient tensor from the next layer.
 * @return A vector containing the gradient with respect to the input.
 */
vector<Tensor> TanhFunction::backward(const Tensor &grad_output) {
    size_t size = grad_output.size();
    Tensor grad_input(size, grad_output.device());
    if (grad_output.device() == CPU) {
        // CPU implementation.
        const float *grad_out_ptr = grad_output.data();
        const float *y_ptr = tanh_output.data();
        float *grad_in_ptr = grad_input.data();
        for (size_t i = 0; i < size; i++) {
            float val = y_ptr[i];
            grad_in_ptr[i] = grad_out_ptr[i] * (1.0f - val * val);
        }
    } else {
        // CUDA implementation.
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        tanh_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), tanh_output.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}
} // namespace nn
} // namespace kernelnet