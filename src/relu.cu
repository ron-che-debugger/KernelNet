#include "relu.hpp"

/**
 * @brief CUDA kernel for the forward pass of ReLU.
 *
 * Applies the ReLU activation element-wise:
 *    out[i] = (in[i] > 0.0f) ? in[i] : 0.0f
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param size Total number of elements in the input array.
 */
__global__ void relu_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx];
        out[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

/**
 * @brief CUDA kernel for the backward pass of ReLU.
 *
 * Computes the gradient with respect to input:
 *    grad_in[i] = grad_out[i] * ((output[i] > 0.0f) ? 1.0f : 0.0f)
 *
 * @param grad_out Pointer to the gradient of the loss with respect to the output.
 * @param output Pointer to the output array from the forward pass.
 * @param grad_in Pointer to the output gradient array to be computed.
 * @param size Total number of elements.
 */
__global__ void relu_backward_kernel(const float *grad_out, const float *output, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = grad_out[idx] * ((output[idx] > 0.0f) ? 1.0f : 0.0f);
    }
}

/**
 * @brief Default constructor for the ReLU module.
 *
 * No internal state is maintained.
 */
ReLU::ReLU() {
    // No internal state needed.
}

/**
 * @brief Performs the forward pass of the ReLU module.
 *
 * Calls ReLUFunction::apply to compute the ReLU activation on the input variable.
 *
 * @param input Input variable.
 * @return Output variable after applying ReLU.
 */
VarPtr ReLU::forward(const VarPtr &input) {
    return ReLUFunction::apply(input);
}

/**
 * @brief Applies the ReLU function element-wise and builds the autograd graph.
 *
 * Allocates an output tensor of the same size and device as the input.
 * The forward pass is computed on CPU or CUDA accordingly. The computed output
 * is stored in the function's internal state for use during the backward pass.
 *
 * @param input Input variable.
 * @return Output variable after applying ReLU.
 */
VarPtr ReLUFunction::apply(const VarPtr &input) {
    auto func = make_shared<ReLUFunction>();
    func->saved_input = input;

    // Allocate an output tensor matching input size and device.
    Tensor out_tensor(input->data.size(), input->data.device());
    size_t size = input->data.size();

    if (input->data.device() == CPU) {
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (size_t i = 0; i < size; ++i) {
            out_ptr[i] = (in_ptr[i] > 0.0f) ? in_ptr[i] : 0.0f;
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        relu_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), size);
        cudaDeviceSynchronize();
    }

    func->relu_output = out_tensor;
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad, "ReLU_out");
    out->set_creator(func);
    func->inputs.push_back(input);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the ReLU activation.
 *
 * Uses the stored output from the forward pass to compute the gradient with respect to the input.
 * The derivative of ReLU is 1 for positive outputs and 0 otherwise.
 *
 * @param grad_output Gradient of the loss with respect to the output.
 * @return A vector containing a single tensor: the gradient with respect to the input.
 */
vector<Tensor> ReLUFunction::backward(const Tensor &grad_output) {
    size_t size = grad_output.size();
    Tensor grad_input(size, grad_output.device());

    if (grad_output.device() == CPU) {
        const float *grad_out_ptr = grad_output.data();
        const float *out_ptr = relu_output.data();
        float *grad_in_ptr = grad_input.data();
        for (size_t i = 0; i < size; ++i) {
            grad_in_ptr[i] = grad_out_ptr[i] * ((out_ptr[i] > 0.0f) ? 1.0f : 0.0f);
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        relu_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), relu_output.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}