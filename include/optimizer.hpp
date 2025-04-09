#pragma once

#include "autograd.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class SGD {
  public:
    vector<VarPtr> params;
    float lr;
    float clip_value; // if > 0 then gradients are norm–clipped to clip_value

    // Constructor accepts optional clip_value (default 0 means no clipping).
    SGD(const vector<VarPtr> &params, float lr, float clip_value = 0.0f);

    void step();
    void zero_grad();
};