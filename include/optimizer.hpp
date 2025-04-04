#pragma once

#include "autograd.hpp"
#include <vector>

using namespace std;

class SGD {
  public:
    vector<VarPtr> params;
    float lr;

    SGD(const vector<VarPtr> &params, float lr);

    void step();
    void zero_grad();
};