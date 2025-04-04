#pragma once

#include "autograd.hpp"
#include <vector>

using namespace std;

class Module {
  public:
    virtual vector<VarPtr> parameters() { return {}; }
    virtual void zero_grad() {
        for (auto param : parameters()) {
            param->grad.fill(0.0f);
            param->grad_initialized = false;
        }
    }
};