#pragma once
#include "single_input_module.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class SoftmaxFunction : public Function {
  public:
    int batch_size, num_classes;
    // Save the computed softmax output for use in the backward pass.
    Tensor softmax_output;
    // Save the input variable as a hard pointer.
    VarPtr saved_input;

    // Compute the softmax output given the input.
    static VarPtr apply(const VarPtr &input, int batch_size, int num_classes);

    // Backward pass: given grad_output, compute grad_input.
    vector<Tensor> backward(const Tensor &grad_output) override;
};

class Softmax : public SingleInputModule {
    using SingleInputModule::forward;

  public:
    int batch_size, num_classes;

    Softmax(int batch_size, int num_classes);
    VarPtr forward(const VarPtr &input) override;
};
