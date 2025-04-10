#pragma once

#include "single_input_module.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class Dense : public SingleInputModule {
  public:
    using SingleInputModule::forward;
    VarPtr weight;
    VarPtr bias;
    int input_dim, output_dim;

    Dense(int input_dim, int output_dim, Device device = Device::CPU);
    ~Dense() = default;

    VarPtr forward(const VarPtr &input) override;

    vector<VarPtr> parameters() override;
};