#include "dense.hpp"

using namespace std;

/**
 * @brief CUDA kernel to replicate bias for each sample in the batch.
 *
 * Copies the bias vector into an output array such that for each sample in the batch,
 * the same bias vector is replicated.
 *
 * @param bias Pointer to the bias array (length = output_dim).
 * @param out Pointer to the output array (length = batch_size * output_dim).
 * @param batch_size Number of samples in the batch.
 * @param output_dim Dimensionality of the output (and bias).
 */
__global__ void replicate_bias_kernel(const float *bias, float *out, int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_dim;

    if (idx < total) {
        int j = idx % output_dim;
        out[idx] = bias[j];
    }
}

/**
 * @brief Constructs a Dense (fully-connected) layer.
 *
 * Initializes the weight and bias tensors. Weights are initialized using a uniform distribution
 * in the range [-limit, limit] where limit = sqrt(6 / (input_dim + output_dim)). The weight tensor
 * is created on CPU and transferred to CUDA if needed.
 *
 * @param input_dim Number of features in the input.
 * @param output_dim Number of neurons in the layer (output dimension).
 * @param device Target device for the tensors (CPU or CUDA).
 */
Dense::Dense(int input_dim, int output_dim, Device device)
    : input_dim(input_dim), output_dim(output_dim) {
    // Create weight tensor on CPU.
    Tensor w(input_dim * output_dim, CPU);

    // Initialize weights uniformly.
    float limit = sqrt(6.0f / (input_dim + output_dim));
    for (size_t i = 0; i < w.size(); ++i) {
        w.data()[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    }

    // Transfer weights to CUDA if needed.
    if (device == CUDA) {
        w.toCUDA();
    }
    weight = make_shared<Variable>(w, true, "Dense_weight");

    // Create and initialize bias tensor.
    Tensor b(output_dim, device);
    b.fill(0.0f);
    bias = make_shared<Variable>(b, true, "Dense_bias");
}

/**
 * @brief Performs the forward pass of the Dense layer.
 *
 * The forward pass calculates:
 *    z = input x weight^T
 *    out = z + bias
 * where the bias is replicated for each batch sample.
 *
 * @param input Input variable with shape (batch_size, input_dim) flattened to 1D.
 * @return Output variable with shape (batch_size, output_dim).
 */
VarPtr Dense::forward(const VarPtr &input) {
    int batch_size = input->data.size() / input_dim;
    // Perform matrix multiplication: (batch_size, input_dim) x (input_dim, output_dim)
    auto z = MatMulFunction::apply(input, weight, batch_size, input_dim, output_dim);
    // Add bias (broadcasted over batch).
    auto out = AddFunction::apply(z, bias);
    return out;
}

/**
 * @brief Returns all learnable parameters of the Dense layer.
 *
 * @return A vector containing the weight and bias variables.
 */
vector<VarPtr> Dense::parameters() {
    return {weight, bias};
}