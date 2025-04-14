#include "embedding.hpp"

namespace kernelnet {
namespace nn {

/**
 * @brief Constructs an Embedding layer.
 *
 * Creates and initializes a learnable embedding weight tensor on the CPU and transfers it
 * to CUDA if requested.
 *
 * @param vocab_size Number of tokens in the vocabulary.
 * @param embed_dim Dimensionality of each embedding vector.
 * @param dev The target device (CPU or CUDA) for the weight tensor.
 */
Embedding::Embedding(int vocab_size, int embed_dim, Device dev)
    : vocab_size(vocab_size), embed_dim(embed_dim) {
    // Create weight tensor on CPU.
    Tensor w(vocab_size * embed_dim, CPU);
    // Compute uniform initialization limit.
    float limit = sqrt(6.0f / (vocab_size + embed_dim));
    // Initialize each element with a uniform random value in [-limit, limit].
    for (size_t i = 0; i < w.size(); i++) {
        w.data()[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * limit;
    }
    // If the target device is CUDA, transfer the weight tensor to the GPU.
    if (dev == CUDA)
        w.toCUDA();
    // Wrap the tensor in a Variable and mark it as trainable.
    weight = make_shared<Variable>(w, true, "Embedding_weight");
}

/**
 * @brief Performs the forward pass of the Embedding layer.
 *
 * Given an input tensor containing token indices, performs a lookup in the weight matrix
 * and returns the corresponding embedding vectors.
 *
 * @param input Input variable with flattened token indices.
 * @return Output variable containing the looked-up embeddings.
 */
VarPtr Embedding::forward(const VarPtr &input) {
    return EmbeddingLookupFunction::apply(input, weight, embed_dim);
}

/**
 * @brief Returns the learnable parameters of the Embedding layer.
 *
 * @return A vector containing the embedding weight variable.
 */
vector<VarPtr> Embedding::parameters() {
    return {weight};
}

/**
 * @brief CUDA kernel for the forward pass of the Embedding lookup.
 *
 * For each token index, converts the float value to an integer index,
 * and copies the corresponding embedding vector from the weight tensor to output.
 *
 * @param weight Pointer to the embedding weight array.
 * @param indices Pointer to the input indices array.
 * @param output Pointer to the output embeddings array.
 * @param embed_dim Dimensionality of the embedding vectors.
 * @param n Total number of token indices.
 */
__global__ void embedding_forward_kernel(const float *weight,
                                         const float *indices,
                                         float *output,
                                         int embed_dim,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Convert the float token index to int.
        int idx = static_cast<int>(indices[i]);
        // Copy the embedding vector corresponding to token index 'idx'
        for (int j = 0; j < embed_dim; j++) {
            output[i * embed_dim + j] = weight[idx * embed_dim + j];
        }
    }
}

/**
 * @brief Applies the embedding lookup and builds the autograd graph.
 *
 * This function extracts token indices from the input (converting to CPU data if needed),
 * stores the indices in a CPU-side vector, then performs a lookup either using a CPU loop
 * (for CPU input) or launching a CUDA kernel (for GPU input). The resulting embeddings and
 * necessary state for the backward pass are saved.
 *
 * @param indices Input variable containing token indices (flattened).
 * @param weight Embedding weight variable.
 * @param embed_dim Dimensionality of the embedding vectors.
 * @return Output variable containing the corresponding embeddings.
 */
VarPtr EmbeddingLookupFunction::apply(const VarPtr &indices,
                                      const VarPtr &weight,
                                      int embed_dim) {
    auto func = make_shared<EmbeddingLookupFunction>();
    func->embed_dim = embed_dim;
    func->saved_weight = weight;

    if (weight->requires_grad) {
        weight->pending_count++;
        cout << "[DEBUG] EmbeddingLookupFunction::apply: incremented pending_count for " << weight->debug_name
             << ", new pending_count = " << weight->pending_count << endl;
    }

    // Only the weight is differentiable, so push only the weight into inputs.
    func->inputs.push_back(weight);
    size_t n = indices->data.size();

    // Create a separate CPU copy of the indices data.
    vector<float> indices_cpu(n);
    if (indices->data.device() == CPU) {
        // Already on CPU.
        const float *in_ptr = indices->data.data();
        for (size_t i = 0; i < n; i++) {
            indices_cpu[i] = in_ptr[i];
        }
    } else {
        // Allocate temporary CPU memory and copy from GPU.
        indices_cpu.resize(n);
        cudaMemcpy(indices_cpu.data(), indices->data.data(), n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Save the integer conversion from the independent CPU copy.
    for (size_t i = 0; i < n; i++) {
        func->indices.push_back(static_cast<int>(indices_cpu[i]));
    }

    // Create an output tensor of size (n * embed_dim) on the same device as indices.
    Tensor out(n * embed_dim, indices->data.device());
    // If using CPU for indices, perform the lookup with a simple loop.
    if (indices->data.device() == CPU) {
        float *weight_data = weight->data.data();
        float *out_data = out.data();
        for (size_t i = 0; i < n; i++) {
            int idx = func->indices[i];
            for (int j = 0; j < embed_dim; j++) {
                out_data[i * embed_dim + j] = weight_data[idx * embed_dim + j];
            }
        }
    } else {
        // For GPU, launch the CUDA kernel.
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        embedding_forward_kernel<<<gridSize, blockSize>>>(
            weight->data.data(),
            indices->data.data(),
            out.data(),
            embed_dim,
            n);
        cudaDeviceSynchronize();
    }

    // Determine if gradients are required.
    bool req_grad = weight->requires_grad; // Indices are non-differentiable.
    auto out_var = make_shared<Variable>(out, req_grad, "Embedding_out");
    out_var->set_creator(func);
    cout << "[DEBUG] Set creator for Embedding_out: " << (out_var->creator != nullptr) << endl;
    func->output = out_var;
    return out_var;
}

/**
 * @brief Computes the backward pass for the Embedding lookup.
 *
 * Since the token indices are non-differentiable, only the gradient for the embedding weight
 * is computed. For each token in the batch, its corresponding gradient slice from grad_output
 * is accumulated into the proper row of the weight gradient. For GPU devices, the gradient is
 * temporarily copied to CPU for accumulation and then moved back.
 *
 * @param grad_output Upstream gradient tensor with shape (n * embed_dim).
 * @return A vector containing a single tensor: the gradient for the embedding weight.
 */
vector<Tensor> EmbeddingLookupFunction::backward(const Tensor &grad_output) {
    cout << "[DEBUG] EmbeddingLookupFunction::backward called, number of saved indices: "
         << indices.size() << endl;
    size_t n = indices.size();
    // Initialize grad_weight with zeros on the same device as grad_output.
    Tensor grad_weight(saved_weight->data.size(), grad_output.device());
    grad_weight.fill(0.0f);

    if (grad_output.device() == CPU) {
        const float *grad_out_ptr = grad_output.data();
        // For each token, add its gradient slice to the correct row.
        for (size_t i = 0; i < n; i++) {
            int idx = indices[i];
            for (int j = 0; j < embed_dim; j++) {
                grad_weight.data()[idx * embed_dim + j] += grad_out_ptr[i * embed_dim + j];
            }
        }
    } else {
        // For GPU, copy grad_output to CPU for safe accumulation.
        Tensor grad_output_cpu(grad_output.size(), CPU);
        cudaMemcpy(grad_output_cpu.data(), grad_output.data(), grad_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Ensure grad_weight data is on CPU.
        grad_weight.toCPU();
        const float *grad_out_ptr = grad_output_cpu.data();
        // Accumulate gradients on the CPU.
        for (size_t i = 0; i < n; i++) {
            int idx = indices[i];
            for (int j = 0; j < embed_dim; j++) {
                grad_weight.data()[idx * embed_dim + j] += grad_out_ptr[i * embed_dim + j];
            }
        }
        // Transfer the accumulated gradient back to CUDA.
        grad_weight.toCUDA();
    }
    
    return {grad_weight};
}

} // namespace nn
} // namespace kernelnet