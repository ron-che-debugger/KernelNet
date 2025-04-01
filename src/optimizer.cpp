#include "optimizer.hpp"
#include "tensor.hpp"

SGD::SGD(const vector<Variable*>& params, float lr) : params(params), lr(lr) {}

void SGD::step(){
    for (auto param: params) {
        for (size_t i = 0; i < param->data.size(); ++i){
            param->data.data()[i] -= lr * param->grad.data()[i];
        }
    }
}

void SGD::zero_grad(){
    for (auto param: params) {
        param->grad.fill(0.0f);
        param->grad_initialized = false;
    }
}