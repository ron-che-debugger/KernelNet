#pragma once
#include "single_input_module.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class TanhFunction : public Function {
  public:
    VarPtr saved_input;

    Tensor tanh_output;

    // Compute the tanh output given the input.
    static VarPtr apply(const VarPtr &input);

    // Backward pass: given grad_output, compute grad_input.
    vector<Tensor> backward(const Tensor &grad_output) override;
};

class Tanh : public SingleInputModule {
  public:
    using SingleInputModule::forward;
    Tanh();
    VarPtr forward(const VarPtr &input) override;
};