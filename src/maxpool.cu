#include "maxpool.hpp"

MaxPool2D::MaxPool2D(int kernel_size, int stride,
                     int batch_size, int channels,
                     int input_height, int input_width)
    : kernel_size(kernel_size), stride(stride),
      batch_size(batch_size), channels(channels),
      input_height(input_height), input_width(input_width) {}

VarPtr MaxPool2D::forward(const VarPtr &input) {
    return MaxPool2DFunction::apply(input, batch_size, channels, input_height, input_width, kernel_size, stride);
}

// CUDA kernel for the forward pass.
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

// Helper function that handles the forward pass.
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

        int blockSize = 256;
        int gridSize = (output_size + blockSize - 1) / blockSize;
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

// Updated apply function that simply calls the helper forward function.
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

vector<Tensor> MaxPool2DFunction::backward(const Tensor &grad_output) {
    // Ensure backward is computed on CPU.
    Tensor grad_out_cpu = grad_output;
    if (grad_output.device() != CPU) {
        grad_out_cpu.toCPU();
    }

    int input_size = batch_size * channels * input_height * input_width;
    Tensor grad_input(input_size, CPU);
    grad_input.fill(0.0f);

    int output_size = batch_size * channels * output_height * output_width;
    const float *grad_out_data = grad_out_cpu.data();
    float *grad_in_data = grad_input.data();

    // Propagate each output gradient to its corresponding max index.
    for (int idx = 0; idx < output_size; ++idx) {
        int in_idx = max_indices[idx];
        grad_in_data[in_idx] += grad_out_data[idx];
    }

    if (grad_output.device() != CPU) {
        grad_input.toCUDA();
    }

    return {grad_input};
}