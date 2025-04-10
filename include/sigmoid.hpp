#pragma once
#include "single_input_module.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class SigmoidFunction : public Function {
  public:
    VarPtr saved_input;
    Tensor sigmoid_output;

    // Compute the sigmoid output given the input.
    static VarPtr apply(const VarPtr &input);

    // Backward pass: given grad_output, compute grad_input.
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class Sigmoid : public SingleInputModule {
  public:
    using SingleInputModule::forward;
    Sigmoid();
    VarPtr forward(const VarPtr &input) override;
};