#pragma once
#include "autograd.hpp"
#include "tensor.hpp"
#include <vector>

using namespace std;

class Module {
public:
    virtual vector<Variable*> parameters() {return {};}
    virtual void zero_grad() {
        for (auto param : parameters()){
            param->grad.fill(0.0f);
            param->grad_initialized = false;
        }
    }
};

class Dense : public Module {
public:
    Variable* weight;
    Variable* bias;
    int input_dim, output_dim;

    Dense(int input_dim, int output_dim, Device device = CPU);
    ~Dense();

    Variable* forward(Variable* input);

    vector<Variable*> parameters() override;
};