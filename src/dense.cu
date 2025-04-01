#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "tensor.hpp"
#include "dense.hpp"

using namespace std;

__global__ void replicate_bias_kernel(const float* bias, float* out, int batch_size, int output_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_dim;

    if (idx < total){
        int j = idx % output_dim;
        out[idx] = bias[j];
    }
}

Dense::Dense(int input_dim, int output_dim, Device device) : input_dim(input_dim), output_dim(output_dim) {
    Tensor w(input_dim * output_dim, device);

    for (size_t i = 0; i < w.size(); ++i){
        w.data()[i] = ((float) rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    if (device == CUDA){
        w.toCUDA();
    }

    weight = new Variable(w, true);

    Tensor b(output_dim, device);
    b.fill(0.0f);
    bias = new Variable(b, true);
}

Dense::~Dense(){
    delete weight;
    delete bias;
}

Variable* Dense::forward(Variable* input){
    int batch_size = input->data.size() / input_dim;    
    Variable* z = MatMulFunction::apply(input, weight, batch_size, input_dim, output_dim);

    Tensor bias_rep(batch_size * output_dim, z->data.device());

    if (z->data.device() == CPU) {
        for (int i = 0; i < batch_size; i++){
            for (int j = 0; j < output_dim; j++){
                bias_rep.data()[i * output_dim + j] = bias->data.data()[j];
            }
        }
    } else {
        int total = batch_size * output_dim;
        dim3 blockSize(256);
        dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
        replicate_bias_kernel<<<gridSize, blockSize>>>(bias->data.data(), bias_rep.data(), batch_size, output_dim);
        cudaDeviceSynchronize();
    }

    Variable* bias_var = new Variable(bias_rep, bias->requires_grad);
    Variable* out = AddFunction::apply(z, bias_var);

    return out;
}

vector<Variable*> Dense::parameters(){
    return {weight, bias};
}