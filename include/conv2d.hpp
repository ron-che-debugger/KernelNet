#pragma once

#include "module.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

class Conv2D : public Module {
  public:
    VarPtr weight; // Shape: (out_channels, in_channels, kernel_h, kernel_w)
    VarPtr bias;   // Shape: (out_channels)
    int in_channels, out_channels, kernel_h, kernel_w;
    int stride, padding;
    int input_height, input_width;

    Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int input_height, int input_width, int stride = 1, int padding = 0, Device device = Device::CPU);

    VarPtr forward(const VarPtr &input);

    vector<VarPtr> parameters() override;
};

Tensor conv2d_forward(const Tensor &input, const Tensor &weight, const Tensor &bias,
                      int batch_size, int in_channels, int input_height, int input_width,
                      int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                      int out_height, int out_width);

Tensor conv2d_backward_bias(const Tensor &grad_output, int batch_size,
                            int out_channels, int out_height, int out_width);

Tensor conv2d_backward_weight(const Tensor &grad_output, const Tensor &input,
                              int batch_size, int in_channels, int input_height, int input_width,
                              int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                              int out_height, int out_width);

Tensor conv2d_backward_input(const Tensor &grad_output, const Tensor &weight,
                             int batch_size, int in_channels, int input_height, int input_width,
                             int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                             int out_height, int out_width);

class Conv2DFunction : public Function {
  public:
    VarPtr saved_input;
    VarPtr saved_weight;
    VarPtr saved_bias;
    int batch_size, in_channels, input_height, input_width;
    int out_channels, kernel_h, kernel_w, stride, padding, out_height, out_width;

    static VarPtr apply(const VarPtr &input, const VarPtr &weight, const VarPtr &bias,
                        int in_channels, int input_height, int input_width,
                        int out_channels, int kernel_h, int kernel_w, int stride, int padding);

    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};