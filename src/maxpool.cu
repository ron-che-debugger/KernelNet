#include "maxpool.hpp"
namespace kernelnet {
namespace nn {
/**
 * @brief Constructor for the MaxPool2D module.
 *
 * Initializes pooling parameters and input dimensions.
 *
 * @param kernel_size Size of the square pooling window.
 * @param stride Stride used for pooling.
 * @param batch_size Number of samples in the batch.
 * @param channels Number of channels in the input.
 * @param input_height Height of the input image.
 * @param input_width Width of the input image.
 */
MaxPool2D::MaxPool2D(int kernel_size, int stride,
                     int batch_size, int channels,
                     int input_height, int input_width)
    : kernel_size(kernel_size), stride(stride),
      batch_size(batch_size), channels(channels),
      input_height(input_height), input_width(input_width) {}

/**
 * @brief Forward pass for MaxPool2D.
 *
 * Applies max pooling on the input variable and returns the pooled output.
 *
 * @param input Input variable.
 * @return Output variable after max pooling.
 */
VarPtr MaxPool2D::forward(const VarPtr &input) {
    return MaxPool2DFunction::apply(input, batch_size, channels, input_height, input_width, kernel_size, stride);
}

/**
 * @brief CUDA kernel for the forward pass of max pooling.
 *
 * For each output element, finds the maximum value over the corresponding
 * pooling window and stores the index of the maximum element.
 *
 * @param in_data Pointer to the input tensor data.
 * @param out_data Pointer to the output tensor data.
 * @param d_max_indices Pointer to an array to store indices of max values.
 * @param batch_size Number of samples in the batch.
 * @param channels Number of channels in the input.
 * @param input_height Height of the input image.
 * @param input_width Width of the input image.
 * @param kernel_size Size of the square pooling window.
 * @param stride Stride used for pooling.
 * @param output_height Calculated height of the output.
 * @param output_width Calculated width of the output.
 */
__global__ void maxpool_forward_kernel(const float *in_data, float *out_data, int *d_max_indices,
                                       int batch_size, int channels, int input_height, int input_width,
                                       int kernel_size, int stride, int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_height * output_width;
    if (idx < total) {
        // Decode output index into b, c, oh, ow.
        int ow = idx % output_width;
        int tmp = idx / output_width;
        int oh = tmp % output_height;
        tmp = tmp / output_height;
        int c = tmp % channels;
        int b = tmp / channels;

        float max_val = -FLT_MAX;
        int max_idx = -1;
        int start_h = oh * stride;
        int start_w = ow * stride;

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int ih = start_h + i;
                int iw = start_w + j;
                int in_index = b * (channels * input_height * input_width) +
                               c * (input_height * input_width) +
                               ih * input_width + iw;
                float val = in_data[in_index];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_index;
                }
            }
        }
        out_data[idx] = max_val;
        d_max_indices[idx] = max_idx;
    }
}

/**
 * @brief Helper function for the forward pass of max pooling.
 *
 * Computes output dimensions, allocates the output tensor, and performs max pooling
 * on CPU or CUDA. Also returns the indices of maximum values for use in the backward pass.
 *
 * @param input Input tensor.
 * @param batch_size Number of samples in the batch.
 * @param channels Number of channels in the input.
 * @param input_height Height of the input image.
 * @param input_width Width of the input image.
 * @param kernel_size Size of the pooling kernel.
 * @param stride Pooling stride.
 * @param output_height (Output) Calculated output height.
 * @param output_width (Output) Calculated output width.
 * @param max_indices (Output) Vector to store indices of maximum elements.
 * @return The output tensor after max pooling.
 */
Tensor maxpool_forward(const Tensor &input, int batch_size, int channels,
                       int input_height, int input_width,
                       int kernel_size, int stride,
                       int &output_height, int &output_width,
                       vector<int> &max_indices) {
    // Compute output dimensions.
    output_height = (input_height - kernel_size) / stride + 1;
    output_width = (input_width - kernel_size) / stride + 1;
    size_t output_size = batch_size * channels * output_height * output_width;

    // Allocate the output tensor on the same device as the input.
    Tensor out_tensor(output_size, input.device());
    max_indices.resize(output_size, -1);

    if (input.device() == CPU) {
        // CPU branch.
        const float *in_data = input.data();
        float *out_data = out_tensor.data();

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        int out_index = b * (channels * output_height * output_width) +
                                        c * (output_height * output_width) +
                                        oh * output_width + ow;
                        float max_val = -FLT_MAX;
                        int max_idx = -1;
                        int start_h = oh * stride;
                        int start_w = ow * stride;
                        for (int i = 0; i < kernel_size; ++i) {
                            for (int j = 0; j < kernel_size; ++j) {
                                int ih = start_h + i;
                                int iw = start_w + j;
                                int in_index = b * (channels * input_height * input_width) +
                                               c * (input_height * input_width) +
                                               ih * input_width + iw;
                                float val = in_data[in_index];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = in_index;
                                }
                            }
                        }
                        out_data[out_index] = max_val;
                        max_indices[out_index] = max_idx;
                    }
                }
            }
        }
    } else {
        // CUDA branch.
        int *d_max_indices;
        cudaMalloc(&d_max_indices, output_size * sizeof(int));

        const float *in_data = input.data(); // device pointer
        float *out_data = out_tensor.data(); // device pointer

        dim3 blockSize(256);
        dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x);
        maxpool_forward_kernel<<<gridSize, blockSize>>>(in_data, out_data, d_max_indices,
                                                        batch_size, channels, input_height, input_width,
                                                        kernel_size, stride, output_height, output_width);
        cudaDeviceSynchronize();

        // Copy the computed indices from device to host.
        cudaMemcpy(max_indices.data(), d_max_indices, output_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_max_indices);
    }

    return out_tensor;
}

/**
 * @brief Applies the MaxPool2DFunction forward pass.
 *
 * Sets up pooling parameters and calls the helper function to perform max pooling.
 *
 * @param input Input variable.
 * @param batch_size Number of samples in the batch.
 * @param channels Number of channels in the input.
 * @param input_height Height of the input image.
 * @param input_width Width of the input image.
 * @param kernel_size Size of the pooling kernel.
 * @param stride Pooling stride.
 * @return Output variable after max pooling.
 */
VarPtr MaxPool2DFunction::apply(const VarPtr &input,
                                int batch_size, int channels,
                                int input_height, int input_width,
                                int kernel_size, int stride) {
    auto func = make_shared<MaxPool2DFunction>();
    func->saved_input = input;
    func->batch_size = batch_size;
    func->channels = channels;
    func->input_height = input_height;
    func->input_width = input_width;
    func->kernel_size = kernel_size;
    func->stride = stride;

    int output_height, output_width;
    vector<int> max_indices;
    // Call the helper forward function.
    Tensor out_tensor = maxpool_forward(input->data, batch_size, channels,
                                        input_height, input_width,
                                        kernel_size, stride,
                                        output_height, output_width,
                                        max_indices);
    func->output_height = output_height;
    func->output_width = output_width;
    func->max_indices = max_indices;

    auto out = make_shared<Variable>(out_tensor, input->requires_grad, "Maxpool_out");
    out->set_creator(func);
    func->inputs.push_back(input);
    func->output = out;

    return out;
}

/**
 * @brief CUDA kernel for the backward pass of max pooling.
 *
 * For each output element, uses the stored index of the maximum input element
 * to propagate the gradient.
 *
 * @param grad_out Pointer to the gradient output tensor.
 * @param grad_in Pointer to the gradient input tensor (to be updated).
 * @param max_indices Pointer to the array of indices stored during forward.
 * @param output_size Total number of output elements.
 */
__global__ void maxpool_backward_kernel(const float *grad_out, float *grad_in, const int *max_indices, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        int in_idx = max_indices[idx];
        atomicAdd(&grad_in[in_idx], grad_out[idx]);
    }
}

/**
 * @brief Computes the backward pass of the MaxPool2DFunction.
 *
 * Propagates gradients from the output back to the input using the stored max indices.
 *
 * @param grad_output Gradient tensor from the next layer.
 * @return A vector containing a single tensor representing the gradient with respect to the input.
 */
vector<Tensor> MaxPool2DFunction::backward(const Tensor &grad_output) {
    if (grad_output.device() == CPU) {
        Tensor grad_input(batch_size * channels * input_height * input_width, CPU);
        grad_input.fill(0.0f);
        int output_size = batch_size * channels * output_height * output_width;
        const float *grad_out_data = grad_output.data();
        float *grad_in_data = grad_input.data();
        for (int idx = 0; idx < output_size; ++idx) {
            int in_idx = max_indices[idx];
            grad_in_data[in_idx] += grad_out_data[idx];
        }
        return {grad_input};
    } else {
        // CUDA branch.
        int output_size = batch_size * channels * output_height * output_width;
        // Allocate grad_input with size = input size.
        Tensor grad_input(batch_size * channels * input_height * input_width, CUDA);
        grad_input.fill(0.0f);

        // Allocate device memory for max_indices.
        int *d_max_indices;
        cudaMalloc(&d_max_indices, output_size * sizeof(int));
        cudaMemcpy(d_max_indices, max_indices.data(), output_size * sizeof(int), cudaMemcpyHostToDevice);

        // Launch the kernel to compute gradients.
        dim3 blockSize(256);
        dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x);
        maxpool_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_input.data(), d_max_indices, output_size);
        cudaDeviceSynchronize();
        cudaFree(d_max_indices);

        return {grad_input};
    }
}
} // namespace nn
} // namespace kernelnet