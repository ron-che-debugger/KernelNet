#pragma once
#include "autograd.hpp"
#include <vector>

using namespace std;

class SGD {
public:
    vector<Variable*> params;
    float lr;

    SGD(const vector<Variable*>& params, float lr);

    void step();
    void zero_grad();
};