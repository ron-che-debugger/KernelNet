#pragma once

#include "module.hpp"
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

class Dense : public Module {
  public:
    VarPtr weight;
    VarPtr bias;
    int input_dim, output_dim;

    Dense(int input_dim, int output_dim, Device device = Device::CPU);
    ~Dense() = default;

    VarPtr forward(const VarPtr &input);

    vector<VarPtr> parameters() override;
};