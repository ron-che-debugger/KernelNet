#include "dense.hpp"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

__global__ void replicate_bias_kernel(const float *bias, float *out, int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_dim;

    if (idx < total) {
        int j = idx % output_dim;
        out[idx] = bias[j];
    }
}

Dense::Dense(int input_dim, int output_dim, Device device) : input_dim(input_dim), output_dim(output_dim) {
    // Create on CPU regardless of the intended device.
    Tensor w(input_dim * output_dim, CPU);

    // Initialize on CPU.
    float limit = sqrt(6.0f / (input_dim + output_dim));
    for (size_t i = 0; i < w.size(); ++i) {
        w.data()[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    }

    // Transfer to CUDA if needed.
    if (device == CUDA) {
        w.toCUDA();
    }

    weight = make_shared<Variable>(w, true, "Dense_weight");

    Tensor b(output_dim, device);
    b.fill(0.0f);
    bias = make_shared<Variable>(b, true, "Dense_bias");
}

VarPtr Dense::forward(const VarPtr &input) {
    int batch_size = input->data.size() / input_dim;
    auto z = MatMulFunction::apply(input, weight, batch_size, input_dim, output_dim);

    auto out = AddFunction::apply(z, bias);

    return out;
}

vector<VarPtr> Dense::parameters() {
    return {weight, bias};
}