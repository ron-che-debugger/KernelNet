#pragma once
#include "autograd.hpp"
#include "tensor.hpp"
#include <vector>

using namespace std;

class Module {
public:
    virtual vector<VarPtr> parameters() {return {};}
    virtual void zero_grad() {
        for (auto param : parameters()){
            param->grad.fill(0.0f);
            param->grad_initialized = false;
        }
    }
};

class Dense : public Module {
public:
    VarPtr weight;
    VarPtr bias;
    int input_dim, output_dim;

    Dense(int input_dim, int output_dim, Device device = CPU);
    ~Dense() = default;

    VarPtr forward(const VarPtr& input);

    vector<VarPtr> parameters() override;
};