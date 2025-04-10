#pragma once

#include "autograd.hpp"
#include <cassert>
#include <vector>

using namespace std;

class Module {
  public:
    virtual vector<VarPtr> forward(const vector<VarPtr> &inputs) = 0;

    virtual vector<VarPtr> parameters() { return {}; }

    virtual void zero_grad() {
        for (auto param : parameters()) {
            param->grad.fill(0.0f);
            param->grad_initialized = false;
        }
    }
};