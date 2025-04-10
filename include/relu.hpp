#pragma once
#include "module.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace std;

class ReLUFunction : public Function {
public:
    VarPtr saved_input;
    Tensor relu_output;

    // Compute the ReLU output given the input.
    static VarPtr apply(const VarPtr &input);

    // Backward pass: given grad_output, compute grad_input.
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class ReLU : public Module {
public:
    ReLU();
    VarPtr forward(const VarPtr &input);
};