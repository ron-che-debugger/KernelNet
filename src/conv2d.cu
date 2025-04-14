#include "conv2d.hpp"

namespace kernelnet {
namespace nn {
/**
 * @brief Conv2D constructor.
 *
 * Initializes the convolution layer by creating the weight and bias tensors.
 *
 * The weight tensor is initialized using a uniform distribution within [–limit, limit],
 * where limit = sqrt(6/(fan_in + fan_out)). The bias tensor is initialized to zero.
 *
 * @param in_channels Number of input channels.
 * @param out_channels Number of output channels.
 * @param kernel_h Height of the convolution kernel.
 * @param kernel_w Width of the convolution kernel.
 * @param input_height Height of the input images.
 * @param input_width Width of the input images.
 * @param stride Stride of the convolution.
 * @param padding Padding added to the input.
 * @param device Target device for the tensors (CPU or CUDA).
 */
Conv2D::Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
               int input_height, int input_width, int stride, int padding, Device device)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_h(kernel_h), kernel_w(kernel_w),
      stride(stride), padding(padding),
      input_height(input_height), input_width(input_width) {
    size_t weight_size = out_channels * in_channels * kernel_h * kernel_w;
    // Create weight tensor on CPU first.
    Tensor w(weight_size, CPU);

    // Initialize weights uniformly: range = [-limit, limit].
    float limit = sqrt(6.0f / (in_channels * kernel_h * kernel_w + out_channels));
    for (size_t i = 0; i < weight_size; i++) {
        w.data()[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    }

    // Transfer weights to CUDA if needed.
    if (device == CUDA) {
        w.toCUDA();
    }
    weight = make_shared<Variable>(w, true, "Conv2D_weight");

    Tensor b(out_channels, device);
    b.fill(0.0f);
    bias = make_shared<Variable>(b, true, "Conv2D_bias");
}

/**
 * @brief Conv2D forward pass.
 *
 * Calls the Conv2DFunction apply method to perform the convolution.
 *
 * @param input Input variable.
 * @return The output variable after applying the convolution.
 */
VarPtr Conv2D::forward(const VarPtr &input) {
    return Conv2DFunction::apply(input, weight, bias,
                                 in_channels, input_height, input_width,
                                 out_channels, kernel_h, kernel_w, stride, padding);
}

/**
 * @brief Returns the parameters of the Conv2D layer.
 *
 * @return A vector containing the weight and bias variables.
 */
vector<VarPtr> Conv2D::parameters() {
    return {weight, bias};
}

/**
 * @brief CUDA kernel for 2D convolution forward pass.
 *
 * Computes one output element based on the local receptive field.
 *
 * @param input Pointer to the input tensor (shape: [B, IC, IH, IW]).
 * @param weight Pointer to the weight tensor (shape: [OC, IC, KH, KW]).
 * @param bias Pointer to the bias tensor (shape: [OC]).
 * @param output Pointer to the output tensor (shape: [B, OC, OH, OW]).
 * @param batch_size Number of samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Stride of the convolution.
 * @param padding Padding on each side.
 * @param out_height Height of the output.
 * @param out_width Width of the output.
 */
__global__ void conv2d_kernel(const float *input, const float *weight, const float *bias, float *output,
                              int batch_size, int in_channels, int input_height, int input_width,
                              int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                              int out_height, int out_width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_height * out_width;
    if (index < total) {
        // Decode index: index = (((b * out_channels + oc) * out_height) + oh) * out_width + ow
        int ow = index % out_width;
        int tmp = index / out_width;
        int oh = tmp % out_height;
        tmp = tmp / out_height;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;
        float sum = bias[oc];

        // Loop over the receptive field.
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_y = oh * stride - padding + kh;
                    int in_x = ow * stride - padding + kw;
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_index = b * (in_channels * input_height * input_width) + ic * (input_height * input_width) + in_y * input_width + in_x;
                        int weight_index = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
        output[index] = sum;
    }
}

/**
 * @brief Performs the forward convolution operation on CPU or CUDA.
 *
 * Computes the convolution given the input, weight, and bias tensors.
 *
 * @param input Input tensor.
 * @param weight Weight tensor.
 * @param bias Bias tensor.
 * @param batch_size Number of samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Convolution stride.
 * @param padding Convolution padding.
 * @param out_height Height of the output.
 * @param out_width Width of the output.
 * @return Output tensor after convolution.
 */
Tensor conv2d_forward(const Tensor &input, const Tensor &weight, const Tensor &bias,
                      int batch_size, int in_channels, int input_height, int input_width,
                      int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                      int out_height, int out_width) {
    int output_size = batch_size * out_channels * out_height * out_width;
    Device device = input.device();
    Tensor output(output_size, device);
    if (device == CPU) {
        const float *input_ptr = input.data();
        const float *weight_ptr = weight.data();
        const float *bias_ptr = bias.data();
        float *output_ptr = output.data();
        for (int b = 0; b < batch_size; b++) {
            for (int oc = 0; oc < out_channels; oc++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        int out_index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                        float sum = bias_ptr[oc];
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kh = 0; kh < kernel_h; kh++) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    int in_y = oh * stride - padding + kh;
                                    int in_x = ow * stride - padding + kw;
                                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                                        int in_index = b * (in_channels * input_height * input_width) + ic * (input_height * input_width) + in_y * input_width + in_x;
                                        int w_index = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                                        sum += input_ptr[in_index] * weight_ptr[w_index];
                                    }
                                }
                            }
                        }
                        output_ptr[out_index] = sum;
                    }
                }
            }
        }
    } else {
        int total = output.size();
        dim3 blockSize(256);
        dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
        conv2d_kernel<<<gridSize, blockSize>>>(input.data(), weight.data(), bias.data(), output.data(),
                                               batch_size, in_channels, input_height, input_width,
                                               out_channels, kernel_h, kernel_w, stride, padding,
                                               out_height, out_width);
        cudaDeviceSynchronize();
    }
    return output;
}

/**
 * @brief CUDA kernel for backward pass with respect to bias.
 *
 * For each output channel, sums the gradients across all batches and spatial locations.
 *
 * @param grad_output Pointer to the gradient output tensor.
 * @param grad_bias Pointer to the gradient bias tensor.
 * @param batch_size Number of samples.
 * @param out_channels Number of output channels.
 * @param out_height Height of output.
 * @param out_width Width of output.
 */
__global__ void conv2d_backward_bias_kernel(const float *grad_output, float *grad_bias,
                                            int batch_size, int out_channels, int out_height, int out_width) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc < out_channels) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    int index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                    sum += grad_output[index];
                }
            }
        }
        grad_bias[oc] = sum;
    }
}

/**
 * @brief Computes the gradient with respect to the bias.
 *
 * Sums gradients along the batch and spatial dimensions.
 *
 * @param grad_output Gradient tensor from the next layer.
 * @param batch_size Number of samples.
 * @param out_channels Number of output channels.
 * @param out_height Height of the output.
 * @param out_width Width of the output.
 * @return Gradient tensor for the bias.
 */
Tensor conv2d_backward_bias(const Tensor &grad_output, int batch_size,
                            int out_channels, int out_height, int out_width) {
    Tensor grad_bias(out_channels, grad_output.device());
    if (grad_output.device() == CPU) {
        float *grad_bias_data = grad_bias.data();
        const float *grad_out_data = grad_output.data();
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        int index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                        sum += grad_out_data[index];
                    }
                }
            }
            grad_bias_data[oc] = sum;
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((out_channels + blockSize.x - 1) / blockSize.x);
        conv2d_backward_bias_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_bias.data(),
                                                             batch_size, out_channels, out_height, out_width);
        cudaDeviceSynchronize();
    }
    return grad_bias;
}

/**
 * @brief CUDA kernel for backward pass with respect to weights.
 *
 * Computes the gradient contribution for each weight based on the corresponding receptive fields.
 *
 * @param grad_output Pointer to the gradient output tensor.
 * @param input Pointer to the input tensor.
 * @param grad_weight Pointer to the weight gradient tensor.
 * @param batch_size Number of samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Convolution stride.
 * @param padding Convolution padding.
 * @param out_height Height of output.
 * @param out_width Width of output.
 * @param weight_size Total number of weight elements.
 */
__global__ void conv2d_backward_weight_kernel(const float *grad_output, const float *input, float *grad_weight,
                                              int batch_size, int in_channels, int input_height, int input_width,
                                              int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                                              int out_height, int out_width, int weight_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < weight_size) {
        // Decode weight index: index = (((oc * in_channels + ic) * kernel_h) + kh) * kernel_w + kw
        int kw = index % kernel_w;
        int tmp = index / kernel_w;
        int kh = tmp % kernel_h;
        tmp = tmp / kernel_h;
        int ic = tmp % in_channels;
        int oc = tmp / in_channels;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    int grad_out_index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                    int in_y = oh * stride - padding + kh;
                    int in_x = ow * stride - padding + kw;
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_index = b * (in_channels * input_height * input_width) + ic * (input_height * input_width) + in_y * input_width + in_x;
                        sum += grad_output[grad_out_index] * input[input_index];
                    }
                }
            }
        }
        grad_weight[index] = sum;
    }
}

/**
 * @brief Computes the gradient with respect to the weight tensor.
 *
 * Calculates gradients on CPU or CUDA.
 *
 * @param grad_output Gradient tensor from the next layer.
 * @param input Input tensor.
 * @param batch_size Number of samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Convolution stride.
 * @param padding Convolution padding.
 * @param out_height Height of output.
 * @param out_width Width of output.
 * @return Gradient tensor for weights.
 */
Tensor conv2d_backward_weight(const Tensor &grad_output, const Tensor &input,
                              int batch_size, int in_channels, int input_height, int input_width,
                              int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                              int out_height, int out_width) {
    size_t weight_size = out_channels * in_channels * kernel_h * kernel_w;
    Tensor grad_weight(weight_size, grad_output.device());
    grad_weight.fill(0.0f);
    if (grad_output.device() == CPU) {
        float *grad_weight_data = grad_weight.data();
        const float *grad_out_data = grad_output.data();
        const float *input_data = input.data();
        for (int b = 0; b < batch_size; b++) {
            for (int oc = 0; oc < out_channels; oc++) {
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            float sum = 0.0f;
                            for (int oh = 0; oh < out_height; oh++) {
                                for (int ow = 0; ow < out_width; ow++) {
                                    int in_y = oh * stride - padding + kh;
                                    int in_x = ow * stride - padding + kw;
                                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                                        int input_index = b * (in_channels * input_height * input_width) + ic * (input_height * input_width) + in_y * input_width + in_x;
                                        int grad_out_index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                                        sum += grad_out_data[grad_out_index] * input_data[input_index];
                                    }
                                }
                            }
                            int weight_index = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            grad_weight_data[weight_index] += sum;
                        }
                    }
                }
            }
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((weight_size + blockSize.x - 1) / blockSize.x);
        conv2d_backward_weight_kernel<<<gridSize, blockSize>>>(grad_output.data(), input.data(), grad_weight.data(),
                                                               batch_size, in_channels, input_height, input_width,
                                                               out_channels, kernel_h, kernel_w, stride, padding,
                                                               out_height, out_width, weight_size);
        cudaDeviceSynchronize();
    }
    return grad_weight;
}

/**
 * @brief CUDA kernel for computing the input gradient (backward pass).
 *
 * For each element in the input tensor, accumulates contributions from all receptive fields.
 *
 * @param grad_output Pointer to the gradient output tensor.
 * @param weight Pointer to the weight tensor.
 * @param grad_input Pointer to the gradient input tensor.
 * @param batch_size Number of samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Stride of the convolution.
 * @param padding Padding.
 * @param out_height Height of output tensor.
 * @param out_width Width of output tensor.
 * @param input_size Total number of elements in the input tensor.
 */
__global__ void conv2d_backward_input_kernel(const float *grad_output, const float *weight, float *grad_input,
                                             int batch_size, int in_channels, int input_height, int input_width,
                                             int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                                             int out_height, int out_width, int input_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size) {
        // Decode the input tensor index: index = (b, ic, ih, iw)
        int iw = index % input_width;
        int tmp = index / input_width;
        int ih = tmp % input_height;
        tmp = tmp / input_height;
        int ic = tmp % in_channels;
        int b = tmp / in_channels;

        float sum = 0.0f;
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    int in_y_min = oh * stride - padding;
                    int in_x_min = ow * stride - padding;
                    if ((iw >= in_x_min) && (iw < in_x_min + kernel_w) &&
                        (ih >= in_y_min) && (ih < in_y_min + kernel_h)) {
                        int kh = ih - in_y_min;
                        int kw = iw - in_x_min;
                        int weight_index = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                        int grad_out_index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                        sum += grad_output[grad_out_index] * weight[weight_index];
                    }
                }
            }
        }
        grad_input[index] = sum;
    }
}

/**
 * @brief Computes the gradient with respect to the input tensor.
 *
 * Given the gradients from the next layer and the layer's weights, this function
 * calculates the gradient for the input on CPU or CUDA.
 *
 * @param grad_output Gradient tensor from the next layer.
 * @param weight Weight tensor.
 * @param batch_size Number of samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Convolution stride.
 * @param padding Padding.
 * @param out_height Height of the output.
 * @param out_width Width of the output.
 * @return Gradient tensor for the input.
 */
Tensor conv2d_backward_input(const Tensor &grad_output, const Tensor &weight,
                             int batch_size, int in_channels, int input_height, int input_width,
                             int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                             int out_height, int out_width) {
    size_t input_size = batch_size * in_channels * input_height * input_width;
    Tensor grad_input(input_size, grad_output.device());
    grad_input.fill(0.0f);
    if (grad_output.device() == CPU) {
        float *grad_input_data = grad_input.data();
        const float *grad_out_data = grad_output.data();
        const float *weight_data = weight.data();
        for (int b = 0; b < batch_size; b++) {
            for (int oc = 0; oc < out_channels; oc++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        int grad_out_index = b * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
                        float grad_val = grad_out_data[grad_out_index];
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kh = 0; kh < kernel_h; kh++) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        int input_index = b * (in_channels * input_height * input_width) + ic * (input_height * input_width) + ih * input_width + iw;
                                        int weight_index = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                                        grad_input_data[input_index] += grad_val * weight_data[weight_index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        int input_size_int = grad_input.size();
        dim3 blockSize(256);
        dim3 gridSize((input_size_int + blockSize.x - 1) / blockSize.x);
        conv2d_backward_input_kernel<<<gridSize, blockSize>>>(grad_output.data(), weight.data(), grad_input.data(),
                                                              batch_size, in_channels, input_height, input_width,
                                                              out_channels, kernel_h, kernel_w, stride, padding,
                                                              out_height, out_width, input_size_int);
        cudaDeviceSynchronize();
    }
    return grad_input;
}

/**
 * @brief Applies the Conv2D function as part of the autograd graph.
 *
 * Sets up the convolution parameters, saves inputs for backward, and computes the forward
 * convolution output.
 *
 * @param input Input variable.
 * @param weight Weight variable.
 * @param bias Bias variable.
 * @param in_channels Number of input channels.
 * @param input_height Height of input images.
 * @param input_width Width of input images.
 * @param out_channels Number of output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Convolution stride.
 * @param padding Convolution padding.
 * @return Output variable after convolution.
 */
VarPtr Conv2DFunction::apply(const VarPtr &input, const VarPtr &weight, const VarPtr &bias,
                             int in_channels, int input_height, int input_width,
                             int out_channels, int kernel_h, int kernel_w, int stride, int padding) {
    auto func = make_shared<Conv2DFunction>();
    int total = input->data.size();
    int single = in_channels * input_height * input_width;
    int batch_size = total / single;
    func->batch_size = batch_size;
    func->in_channels = in_channels;
    func->input_height = input_height;
    func->input_width = input_width;
    func->out_channels = out_channels;
    func->kernel_h = kernel_h;
    func->kernel_w = kernel_w;
    func->stride = stride;
    func->padding = padding;
    func->out_height = (input_height - kernel_h + 2 * padding) / stride + 1;
    func->out_width = (input_width - kernel_w + 2 * padding) / stride + 1;

    func->saved_input = input;
    func->saved_weight = weight;
    func->saved_bias = bias;
    func->inputs.push_back(input);
    func->inputs.push_back(weight);
    func->inputs.push_back(bias);
    Tensor out_data = conv2d_forward(input->data, weight->data, bias->data,
                                     batch_size, in_channels, input_height, input_width,
                                     out_channels, kernel_h, kernel_w, stride, padding,
                                     func->out_height, func->out_width);
    bool req_grad = input->requires_grad || weight->requires_grad || bias->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad, "Conv2D_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the Conv2D function.
 *
 * Calculates gradients with respect to the input, weights, and bias, using the
 * previously saved inputs.
 *
 * @param grad_output Gradient tensor from the next layer.
 * @return A vector containing gradients for input, weight, and bias.
 */
vector<Tensor> Conv2DFunction::backward(const Tensor &grad_output) {
    Tensor grad_input = conv2d_backward_input(grad_output, saved_weight->data,
                                              batch_size, in_channels, input_height, input_width,
                                              out_channels, kernel_h, kernel_w, stride, padding,
                                              out_height, out_width);
    Tensor grad_weight = conv2d_backward_weight(grad_output, saved_input->data,
                                                batch_size, in_channels, input_height, input_width,
                                                out_channels, kernel_h, kernel_w, stride, padding,
                                                out_height, out_width);
    Tensor grad_bias = conv2d_backward_bias(grad_output, batch_size,
                                            out_channels, out_height, out_width);
    return {grad_input, grad_weight, grad_bias};
}
} // namespace nn
} // namespace kernelnet