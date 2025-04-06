#pragma once
#include "module.hpp"
#include "tensor.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class MaxPool2DFunction : public Function {
  public:
    // Save the input variable for backward.
    VarPtr saved_input;

    int batch_size, channels, input_height, input_width;
    int kernel_size, stride;
    int output_height, output_width;

    // For CPU: store max indices.
    vector<int> max_indices;

    // Compute forward max pooling and set up autograd info.
    static VarPtr apply(const VarPtr &input,
                        int batch_size, int channels,
                        int input_height, int input_width,
                        int kernel_size, int stride);

    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class MaxPool2D : public Module {
  public:
    int kernel_size, stride;
    int batch_size, channels, input_height, input_width;

    // Constructor takes pooling parameters and input dimensions.
    MaxPool2D(int kernel_size, int stride,
              int batch_size, int channels,
              int input_height, int input_width);

    VarPtr forward(const VarPtr &input);
};