#include "softmax.hpp"

namespace kernelnet {
namespace nn {

/**
 * @brief Constructor for the Softmax module.
 *
 * @param batch_size Number of samples in each batch.
 * @param num_classes Number of classes for softmax normalization.
 */
Softmax::Softmax(int batch_size, int num_classes)
    : batch_size(batch_size), num_classes(num_classes) {}

/**
 * @brief Performs the forward pass of the Softmax module.
 *
 * @param input Input variable (tensor) on which softmax is to be computed.
 * @return Output variable after applying softmax.
 */
VarPtr Softmax::forward(const VarPtr &input) {
    return SoftmaxFunction::apply(input, batch_size, num_classes);
}

/**
 * @brief Computes the softmax forward operation on the CPU.
 *
 * @param input Input tensor.
 * @param batch_size Number of samples in a batch.
 * @param num_classes Number of classes (elements per sample).
 * @return Tensor containing the softmax output.
 */
Tensor softmax_forward(const Tensor &input, int batch_size, int num_classes) {
    // Create a CPU tensor for input data.
    Tensor cpu_input(input.size(), CPU);
    if (input.device() == CUDA) {
        // Copy data from CUDA to CPU.
        cudaMemcpy(cpu_input.data(), input.data(), input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        memcpy(cpu_input.data(), input.data(), input.size() * sizeof(float));
    }

    // Compute softmax on CPU.
    Tensor cpu_output(cpu_input.size(), CPU);
    const float *in_data = cpu_input.data();
    float *out_data = cpu_output.data();
    for (int b = 0; b < batch_size; ++b) {
        int offset = b * num_classes;
        float max_val = in_data[offset];
        // Find maximum value for numerical stability.
        for (int i = 1; i < num_classes; ++i) {
            max_val = std::max(max_val, in_data[offset + i]);
        }
        float sum_exp = 0.0f;
        // Calculate exponentials and their sum.
        for (int i = 0; i < num_classes; ++i) {
            float exp_val = expf(in_data[offset + i] - max_val);
            out_data[offset + i] = exp_val;
            sum_exp += exp_val;
        }
        // Normalize to get softmax probabilities.
        for (int i = 0; i < num_classes; ++i) {
            out_data[offset + i] /= sum_exp;
        }
    }

    // If the original tensor was on CUDA, transfer the computed softmax output to CUDA.
    if (input.device() == CUDA) {
        cpu_output.toCUDA();
    }
    return cpu_output;
}

/**
 * @brief Computes the softmax backward operation on the CPU.
 *
 * @param grad_output Gradient of the loss with respect to the softmax output.
 * @param y Softmax output from the forward pass.
 * @param batch_size Number of samples in a batch.
 * @param num_classes Number of classes (elements per sample).
 * @return Tensor representing the gradient with respect to the input.
 */
Tensor softmax_backward(const Tensor &grad_output, const Tensor &y, int batch_size, int num_classes) {
    // Bring grad_output and y to CPU memory.
    Tensor cpu_grad(grad_output.size(), CPU);
    Tensor cpu_y(y.size(), CPU);
    if (grad_output.device() == CUDA) {
        cudaMemcpy(cpu_grad.data(), grad_output.data(), grad_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        memcpy(cpu_grad.data(), grad_output.data(), grad_output.size() * sizeof(float));
    }
    if (y.device() == CUDA) {
        cudaMemcpy(cpu_y.data(), y.data(), y.size() * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        memcpy(cpu_y.data(), y.data(), y.size() * sizeof(float));
    }

    // Compute softmax backward using CPU loops.
    Tensor cpu_grad_input(cpu_grad.size(), CPU);
    const float *grad_out_ptr = cpu_grad.data();
    const float *y_ptr = cpu_y.data();
    float *grad_in_ptr = cpu_grad_input.data();
    for (int b = 0; b < batch_size; ++b) {
        int offset = b * num_classes;
        float dot = 0.0f;
        // Dot product of grad_output and softmax output for sample 'b'.
        for (int j = 0; j < num_classes; ++j) {
            dot += grad_out_ptr[offset + j] * y_ptr[offset + j];
        }
        // Compute gradient with respect to input.
        for (int i = 0; i < num_classes; ++i) {
            grad_in_ptr[offset + i] = y_ptr[offset + i] * (grad_out_ptr[offset + i] - dot);
        }
    }

    // Transfer back to CUDA if needed.
    if (grad_output.device() == CUDA) {
        cpu_grad_input.toCUDA();
    }
    return cpu_grad_input;
}

/**
 * @brief Applies the Softmax activation function and builds the autograd graph.
 *
 * @param input Input variable (tensor) on which softmax is applied.
 * @param batch_size Number of samples in each batch.
 * @param num_classes Number of classes.
 * @return Output variable after applying the softmax activation.
 */
VarPtr SoftmaxFunction::apply(const VarPtr &input, int batch_size, int num_classes) {
    auto func = make_shared<SoftmaxFunction>();
    func->batch_size = batch_size;
    func->num_classes = num_classes;
    // Save input for use in the backward pass.
    func->saved_input = input;

    // Compute softmax forward using the CPU function.
    Tensor out_tensor = softmax_forward(input->data, batch_size, num_classes);
    // Cache the softmax output for use during backpropagation.
    func->softmax_output = out_tensor;

    // Create the output variable with the computed tensor.
    auto out = make_shared<Variable>(out_tensor, input->requires_grad, "Softmax_out");
    out->set_creator(func);
    func->inputs.push_back(input);
    func->output = out;

    return out;
}

/**
 * @brief Computes the backward pass for the softmax activation.
 *
 * @param grad_output Gradient of the loss with respect to the softmax output.
 * @return A vector containing a single tensor representing the gradient with respect to the input.
 */
vector<Tensor> SoftmaxFunction::backward(const Tensor &grad_output) {
    // Compute softmax backward using the CPU function.
    Tensor grad_in = softmax_backward(grad_output, softmax_output, batch_size, num_classes);
    // Ensure consistency of device placement: if grad_output was on CUDA, transfer the gradient back.
    if (grad_output.device() != CPU) {
        grad_in.toCUDA();
    }
    return {grad_in};
}

} // namespace nn
} // namespace kernelnet