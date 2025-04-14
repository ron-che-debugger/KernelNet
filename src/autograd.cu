#include "autograd.hpp"

namespace kernelnet {
namespace autograd {
/**
 * @brief CUDA kernel to fill an array with a constant value.
 *
 * @param out Pointer to the output array.
 * @param value The constant value to fill.
 * @param size Total number of elements in the array.
 */
static __global__ void fill_kernel(float *out, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = value;
    }
}

/**
 * @brief CUDA kernel to compute the sum over repeated segments for gradient broadcasting.
 *
 * @param grad Pointer to the input gradient array.
 * @param out Pointer to the output array where the summed gradient is stored.
 * @param total_size Total number of elements in grad.
 * @param small_size Number of elements in the smaller dimension (the reduction axis).
 */
static __global__ void broadcast_sum_kernel(const float *grad, float *out, size_t total_size, size_t small_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < small_size) {
        float sum_val = 0.0f;
        int repeat = total_size / small_size;
        for (int i = 0; i < repeat; i++) {
            sum_val += grad[i * small_size + idx];
        }
        out[idx] = sum_val;
    }
}

// -------------------- Variable Implementation --------------------

/**
 * @brief Constructs a Variable with the given data.
 *
 * If gradient tracking is enabled, initializes the gradient to zeros.
 *
 * @param data Tensor data.
 * @param requires_grad Flag indicating if gradient tracking is enabled.
 * @param name Optional debug name.
 */
Variable::Variable(const Tensor &data, bool requires_grad, const string &name)
    : data(data), requires_grad(requires_grad), grad_initialized(false), pending_count(0), debug_name(name) {
    if (requires_grad) {
        grad = Tensor(data.size(), data.device());
        grad.fill(0.0f);
    }
}

/**
 * @brief Sets the creator function (the function that produced this variable).
 *
 * @param func Shared pointer to the creator function.
 */
void Variable::set_creator(const FuncPtr &func) {
    creator = func;
}

/**
 * @brief Performs the backward pass for gradient propagation.
 *
 * If this variable requires gradient, accumulates the gradient and, once all
 * contributions are received, propagates backward to inputs.
 *
 * @param grad_output The gradient from the next layer.
 */
void Variable::backward(const Tensor &grad_output) {
    if (requires_grad) {
        /**
        // Print out the first few elements of grad_output (up to 10 elements)
        cout << "[DEBUG] grad_output for " << debug_name << ": ";

        size_t numToPrint = std::min(grad_output.size(), (size_t)10);
        if (grad_output.device() == CPU) {
            const float *ptr = grad_output.data();
            for (size_t i = 0; i < numToPrint; ++i) {
                cout << ptr[i] << " ";
            }
        } else {
            // For CUDA, copy data to CPU first.
            vector<float> host_buffer(numToPrint);
            cudaMemcpy(host_buffer.data(), grad_output.data(), numToPrint * sizeof(float), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < numToPrint; ++i) {
                cout << host_buffer[i] << " ";
            }
        }
        cout << endl;
        cout << "[DEBUG] In backward for " << debug_name
             << " pending_count=" << pending_count
             << " creator=" << (creator ? "yes" : "no") << endl;
        */
        if (!grad_initialized) {
            grad = grad_output;
            grad_initialized = true;
            // cout << "[DEBUG] Initial gradient set for " << debug_name << endl;
        } else {
            grad = Tensor::add(grad, grad_output);
            // cout << "[DEBUG] Updated gradient for " << debug_name << endl;
        }
        if (pending_count > 0)
            pending_count--;
        if (creator && pending_count == 0) {
            // cout << "[DEBUG] Backward called on function output: " << debug_name << endl;
            // cout << "[DEBUG] Function has " << creator->inputs.size() << " inputs" << endl;
            vector<Tensor> input_grads = creator->backward(grad);
            for (size_t i = 0; i < creator->inputs.size(); ++i) {
                if (auto inp = creator->inputs[i].lock()) {
                    if (inp->requires_grad) {
                        inp->backward(input_grads[i]);
                    }
                }
            }
            creator = nullptr;
        }
    }
}

/**
 * @brief Returns a copy of this variable detached from the autograd graph.
 *
 * @return A new Variable containing the same data but not tracking gradients.
 */
VarPtr Variable::detach() {
    return make_shared<Variable>(data, false);
}

// -------------------- AddFunction Implementation --------------------

/**
 * @brief Applies the element-wise addition function.
 *
 * Saves the input variables and creates the output as the broadcasted sum of the inputs.
 *
 * @param a First input variable.
 * @param b Second input variable.
 * @return The output variable representing a + b.
 */
VarPtr AddFunction::apply(const VarPtr &a, const VarPtr &b) {
    auto func = make_shared<AddFunction>();
    if (a->requires_grad)
        a->pending_count++;
    if (b->requires_grad)
        b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::broadcast_add(a->data, b->data);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad, "Add_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the addition function.
 *
 * Propagates the gradient to both inputs, summing appropriately if broadcasting occurred.
 *
 * @param grad_output The gradient of the loss with respect to the output.
 * @return A vector containing the gradients for each input.
 */
vector<Tensor> AddFunction::backward(const Tensor &grad_output) {
    auto inp0 = saved_a;
    auto inp1 = saved_b;
    if (inp0->data.size() == inp1->data.size()) {
        return {grad_output, grad_output};
    }
    if (inp0->data.size() > inp1->data.size()) {
        size_t small_size = inp1->data.size();
        size_t repeat = grad_output.size() / small_size;
        Tensor grad_inp1(small_size, grad_output.device());
        if (grad_output.device() == CPU) {
            for (size_t j = 0; j < small_size; j++) {
                float sum_val = 0;
                for (size_t i = 0; i < repeat; i++) {
                    size_t index = i * small_size + j;
                    sum_val += grad_output.data()[index];
                }
                grad_inp1.data()[j] = sum_val;
            }
        } else {
            dim3 blockSize(256);
            dim3 gridSize((small_size + blockSize.x - 1) / blockSize.x);
            broadcast_sum_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_inp1.data(), grad_output.size(), small_size);
            cudaDeviceSynchronize();
        }
        return {grad_output, grad_inp1};
    }
    if (inp1->data.size() > inp0->data.size()) {
        size_t small_size = inp0->data.size();
        size_t repeat = grad_output.size() / small_size;
        Tensor grad_inp0(small_size, grad_output.device());
        if (grad_output.device() == CPU) {
            for (size_t j = 0; j < small_size; j++) {
                float sum_val = 0;
                for (size_t i = 0; i < repeat; i++) {
                    size_t index = i * small_size + j;
                    sum_val += grad_output.data()[index];
                }
                grad_inp0.data()[j] = sum_val;
            }
        } else {
            dim3 blockSize(256);
            dim3 gridSize((small_size + blockSize.x - 1) / blockSize.x);
            broadcast_sum_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_inp0.data(), grad_output.size(), small_size);
            cudaDeviceSynchronize();
        }
        return {grad_inp0, grad_output};
    }
    return {grad_output, grad_output};
}

// -------------------- SubtractFunction Implementation --------------------

/**
 * @brief Applies the element-wise subtraction function.
 *
 * Computes a - b and saves the input variables.
 *
 * @param a First input variable.
 * @param b Second input variable.
 * @return The output variable representing a - b.
 */
VarPtr SubtractFunction::apply(const VarPtr &a, const VarPtr &b) {
    auto func = make_shared<SubtractFunction>();
    if (a->requires_grad)
        a->pending_count++;
    if (b->requires_grad)
        b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::subtract(a->data, b->data);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad, "Subtract_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for subtraction.
 *
 * The gradient with respect to the first input is unchanged,
 * while the gradient for the second input is multiplied by -1.
 *
 * @param grad_output The gradient of the loss with respect to the output.
 * @return A vector containing gradients for the inputs.
 */
vector<Tensor> SubtractFunction::backward(const Tensor &grad_output) {
    Tensor neg_one(grad_output.size(), grad_output.device());
    neg_one.fill(-1.0f);
    Tensor grad_b = Tensor::multiply(grad_output, neg_one);
    if (!saved_b->requires_grad) {
        grad_b = Tensor(saved_b->data.size(), saved_b->data.device());
        grad_b.fill(0.0f);
    }
    return {grad_output, grad_b};
}

// -------------------- MultiplyFunction Implementation --------------------

/**
 * @brief Applies the element-wise multiplication function.
 *
 * Computes a * b and stores both inputs for use in the backward pass.
 *
 * @param a First input variable.
 * @param b Second input variable.
 * @return The output variable representing a * b.
 */
VarPtr MultiplyFunction::apply(const VarPtr &a, const VarPtr &b) {
    auto func = make_shared<MultiplyFunction>();
    if (a->requires_grad)
        a->pending_count++;
    if (b->requires_grad)
        b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::multiply(a->data, b->data);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad, "Multiply_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for multiplication.
 *
 * Uses the chain rule: grad_a = grad_output * b and grad_b = grad_output * a.
 *
 * @param grad_output The gradient of the loss with respect to the output.
 * @return A vector containing gradients for each input.
 */
vector<Tensor> MultiplyFunction::backward(const Tensor &grad_output) {
    Tensor grad_a = Tensor::multiply(grad_output, saved_b->data);
    Tensor grad_b = Tensor::multiply(grad_output, saved_a->data);
    return {grad_a, grad_b};
}

// -------------------- MatMulFunction Implementation --------------------

/**
 * @brief Applies the matrix multiplication function.
 *
 * Computes the product of two matrices with shapes:
 *   - A: (M, K)
 *   - B: (K, N)
 *
 * @param a First input variable.
 * @param b Second input variable.
 * @param M Number of rows in the first matrix.
 * @param K Shared dimension.
 * @param N Number of columns in the second matrix.
 * @return The output variable representing the matrix product.
 */
VarPtr MatMulFunction::apply(const VarPtr &a, const VarPtr &b, int M, int K, int N) {
    auto func = make_shared<MatMulFunction>();
    if (a->requires_grad)
        a->pending_count++;
    if (b->requires_grad)
        b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->M = M;
    func->K = K;
    func->N = N;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::matmul(a->data, b->data, M, K, N);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad, "MatMul_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for matrix multiplication.
 *
 * Computes gradients using transposes:
 *   - grad_a = grad_output * B^T
 *   - grad_b = A^T * grad_output
 *
 * @param grad_output The gradient of the loss with respect to the output.
 * @return A vector containing gradients with respect to each input.
 */
vector<Tensor> MatMulFunction::backward(const Tensor &grad_output) {
    auto inp1 = saved_a;
    auto inp2 = saved_b;
    Tensor b_t = Tensor::transpose(inp2->data, K, N);
    Tensor grad_a = Tensor::matmul(grad_output, b_t, M, N, K);
    Tensor a_t = Tensor::transpose(inp1->data, M, K);
    Tensor grad_b = Tensor::matmul(a_t, grad_output, K, M, N);
    return {grad_a, grad_b};
}

// -------------------- SumFunction Implementation --------------------

/**
 * @brief Applies the summation function to reduce all elements to a scalar.
 *
 * @param input The input variable to sum.
 * @return A variable containing the scalar sum.
 */
VarPtr SumFunction::apply(const VarPtr &input) {
    auto func = make_shared<SumFunction>();
    func->saved_input = input;
    func->inputs.push_back(input);
    float total = input->data.sum();
    Tensor out_data(1, input->data.device());
    out_data.fill(total);
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad, "Sum_out");
    out->set_creator(func);
    func->output = out;
    func->input_size = input->data.size();
    return out;
}

/**
 * @brief Computes the backward pass for the summation function.
 *
 * The gradient for every element is the upstream gradient.
 *
 * @param grad_output The gradient of the loss with respect to the sum output.
 * @return A vector containing the gradient for the input.
 */
vector<Tensor> SumFunction::backward(const Tensor &grad_output) {
    size_t n = saved_input->data.size();
    Tensor grad_input(n, saved_input->data.device());
    if (saved_input->data.device() == CUDA) {
        float grad_val;
        cudaMemcpy(&grad_val, grad_output.data(), sizeof(float), cudaMemcpyDeviceToHost);
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        fill_kernel<<<gridSize, blockSize>>>(grad_input.data(), grad_val, n);
        cudaDeviceSynchronize();
    } else {
        for (size_t i = 0; i < n; ++i) {
            grad_input.data()[i] = grad_output.data()[0];
        }
    }
    return {grad_input};
}

// -------------------- LogFunction Implementation --------------------

/**
 * @brief CUDA kernel for the forward pass of the element-wise logarithm.
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param size Number of elements in the array.
 */
__global__ void log_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1e-8f;
    if (idx < size) {
        out[idx] = logf(in[idx] + epsilon);
    }
}

/**
 * @brief CUDA kernel for the backward pass of the logarithm.
 *
 * Computes derivative 1/x * grad_out.
 *
 * @param grad_out Pointer to the gradient from the next layer.
 * @param in Pointer to the original input array.
 * @param grad_in Pointer to the computed input gradient array.
 * @param size Number of elements.
 */
__global__ void log_backward_kernel(const float *grad_out, const float *in, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1e-8f;
    if (idx < size) {
        grad_in[idx] = grad_out[idx] / (in[idx] + epsilon);
    }
}

/**
 * @brief Applies the element-wise logarithm function.
 *
 * Computes log(input + epsilon) element-wise.
 *
 * @param input The input variable.
 * @return The output variable after applying the logarithm.
 */
VarPtr LogFunction::apply(const VarPtr &input) {
    auto func = make_shared<LogFunction>();
    func->saved_input = input;
    func->inputs.push_back(input);
    size_t size = input->data.size();
    Tensor out_tensor(size, input->data.device());
    if (input->data.device() == CPU) {
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = logf(in_ptr[i] + 1e-8f);
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        log_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), size);
        cudaDeviceSynchronize();
    }
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad, "Log_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the logarithm function.
 *
 * Uses the derivative: 1 / (input + epsilon) multiplied element-wise with grad_output.
 *
 * @param grad_output The gradient of the loss with respect to the output.
 * @return A vector containing the gradient with respect to the input.
 */
vector<Tensor> LogFunction::backward(const Tensor &grad_output) {
    size_t size = grad_output.size();
    Tensor grad_input(size, grad_output.device());
    if (grad_output.device() == CPU) {
        const float *grad_out_ptr = grad_output.data();
        const float *in_ptr = saved_input->data.data();
        float *grad_in_ptr = grad_input.data();
        for (size_t i = 0; i < size; i++) {
            grad_in_ptr[i] = grad_out_ptr[i] / (in_ptr[i] + 1e-8f);
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        log_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), saved_input->data.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}

// -------------------- MSEFunction Implementation --------------------

/**
 * @brief Applies the Mean Squared Error (MSE) loss function.
 *
 * Computes loss = mean((prediction - target)^2).
 *
 * @param prediction The predicted output variable.
 * @param target The target tensor.
 * @return A variable representing the scalar MSE loss.
 */
VarPtr MSEFunction::apply(const VarPtr &prediction, const Tensor &target) {
    auto target_var = make_shared<Variable>(target, false, "target");
    auto diff = SubtractFunction::apply(prediction, target_var);
    diff->debug_name = "diff";
    auto sq = MultiplyFunction::apply(diff, diff);
    sq->debug_name = "sq";
    auto sum_loss = SumFunction::apply(sq);
    sum_loss->debug_name = "sum_loss";

    Tensor div_tensor(1, sum_loss->data.device());
    div_tensor.fill(1.0f / prediction->data.size());
    auto scale = make_shared<Variable>(div_tensor, false, "scale");

    auto mse_loss = MultiplyFunction::apply(sum_loss, scale);
    mse_loss->debug_name = "mse_loss";
    return mse_loss;
}

// -------------------- CrossEntropyLossFunction Implementation --------------------

/**
 * @brief Applies the Cross-Entropy Loss function.
 *
 * Computes loss = -sum(target ⊙ log(prediction))/batch_size when num_classes > 0,
 * otherwise multiplies the summed loss by -1.
 *
 * @param prediction The predicted output variable.
 * @param target The target tensor (typically one-hot encoded).
 * @param num_classes The number of classes.
 * @return A variable representing the scalar cross-entropy loss.
 */
VarPtr CrossEntropyLossFunction::apply(const VarPtr &prediction, const Tensor &target, int num_classes) {
    auto target_var = make_shared<Variable>(target, false, "target");
    auto log_pred = LogFunction::apply(prediction);
    log_pred->debug_name = "log_pred";
    auto prod = MultiplyFunction::apply(target_var, log_pred);
    prod->debug_name = "prod";
    auto sum_loss = SumFunction::apply(prod);
    sum_loss->debug_name = "sum_loss";

    if (num_classes > 0) {
        int batch_size = target.size() / num_classes;
        Tensor scale_tensor(1, sum_loss->data.device());
        scale_tensor.fill(-1.0f / static_cast<float>(batch_size));
        auto scale = make_shared<Variable>(scale_tensor, false, "scale");
        auto cross_entropy_loss = MultiplyFunction::apply(sum_loss, scale);
        cross_entropy_loss->debug_name = "cross_entropy_loss";
        return cross_entropy_loss;
    } else {
        Tensor neg_tensor(1, sum_loss->data.device());
        neg_tensor.fill(-1.0f);
        auto neg_one = make_shared<Variable>(neg_tensor, false, "neg_one");
        auto cross_entropy_loss = MultiplyFunction::apply(sum_loss, neg_one);
        cross_entropy_loss->debug_name = "cross_entropy_loss";
        return cross_entropy_loss;
    }
}

// -------------------- SliceFunction Implementation --------------------

/**
 * @brief CUDA kernel for the forward pass of the slice function.
 *
 * Copies a slice from the input tensor to the output tensor.
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param total_width Total width of the input tensor (number of columns).
 * @param slice_start Starting index for the slice (inclusive).
 * @param slice_len Length of the slice.
 * @param batch_size Number of rows in the input tensor.
 */
__global__ void slice_forward_kernel(const float *in, float *out, int total_width, int slice_start, int slice_len, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * slice_len;
    if (idx < total) {
        int b = idx / slice_len;
        int i = idx % slice_len;
        out[idx] = in[b * total_width + slice_start + i];
    }
}

/**
 * @brief CUDA kernel for the backward pass of the slice function.
 *
 * Propagates gradients from the sliced output back into the proper locations
 * of the input gradient tensor, setting non-sliced elements to zero.
 *
 * @param grad_out Pointer to the gradient output array (sliced).
 * @param grad_in Pointer to the gradient input array (full tensor).
 * @param total_width Total width of the input tensor.
 * @param slice_start Starting index of the slice.
 * @param slice_len Length of the slice.
 * @param batch_size Number of rows in the input tensor.
 */
__global__ void slice_backward_kernel(const float *grad_out, float *grad_in, int total_width, int slice_start, int slice_len, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * total_width;
    if (idx < total) {
        int b = idx / total_width;
        int j = idx % total_width;
        float grad = 0.0f;
        if (j >= slice_start && j < slice_start + slice_len) {
            int i = j - slice_start;
            grad = grad_out[b * slice_len + i];
        }
        grad_in[idx] = grad;
    }
}

/**
 * @brief Applies the slice function to extract a subset of features.
 *
 * Interprets the input tensor as a 2D array with shape [batch_size, total_width]
 * and extracts columns in the interval [start, end). The output tensor has shape
 * [batch_size, slice_len] where slice_len = end - start.
 *
 * @param input Input variable.
 * @param batch_size Number of rows in the input.
 * @param start Starting column index (inclusive).
 * @param end Ending column index (non-inclusive).
 * @return Output variable containing the sliced tensor.
 */
VarPtr SliceFunction::apply(const VarPtr &input, int batch_size, int start, int end) {
    auto func = make_shared<SliceFunction>();

    func->saved_input = input;
    func->inputs.push_back(input);
    func->batch_size = batch_size;
    func->start = start;
    func->end = end;
    int total_size = input->data.size();
    func->total_width = total_size / batch_size;
    int slice_len = end - start;
    Tensor out_tensor(batch_size * slice_len, input->data.device());
    int total = batch_size * slice_len;
    if (input->data.device() == CPU) {
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < slice_len; i++) {
                out_ptr[b * slice_len + i] = in_ptr[b * func->total_width + start + i];
            }
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
        slice_forward_kernel<<<gridSize, blockSize>>>(input->data.data(), out_tensor.data(), func->total_width, start, slice_len, batch_size);
        cudaDeviceSynchronize();
    }
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_tensor, req_grad, "Slice_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the slice function.
 *
 * Maps the gradient from the sliced output back to the corresponding positions
 * in the input gradient tensor, setting elements outside the slice to zero.
 *
 * @param grad_output Gradient tensor from the next layer (of shape [batch_size, slice_len]).
 * @return A vector containing the gradient tensor for the input.
 */
vector<Tensor> SliceFunction::backward(const Tensor &grad_output) {
    int slice_len = end - start;
    int grad_in_size = batch_size * total_width;
    Tensor grad_input(grad_in_size, grad_output.device());
    if (grad_output.device() == CPU) {
        float *grad_in_ptr = grad_input.data();
        const float *grad_out_ptr = grad_output.data();
        for (int i = 0; i < grad_in_size; i++) {
            grad_in_ptr[i] = 0;
        }
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < slice_len; i++) {
                grad_in_ptr[b * total_width + start + i] = grad_out_ptr[b * slice_len + i];
            }
        }
    } else {
        int total = batch_size * total_width;
        dim3 blockSize(256);
        dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
        cudaMemset(grad_input.data(), 0, grad_in_size * sizeof(float));
        slice_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_input.data(), total_width, start, slice_len, batch_size);
        cudaDeviceSynchronize();
    }
    return {grad_input};
}

/**
 * @brief Concatenates a list of input tensors into a single output tensor.
 *
 * @param inputs Vector of input variables to concatenate.
 * @return Output variable containing the concatenated tensor.
 */
VarPtr ConcatFunction::apply(const vector<VarPtr> &inputs) {
    // Create a new instance of the ConcatFunction.
    auto func = make_shared<ConcatFunction>();
    // Save the inputs for use during the backward pass.
    func->saved_input = inputs;
    size_t total_size = 0;

    for (const auto &inp : inputs) {
        func->inputs.push_back(inp);
    }

    // Iterate over each input:
    // - Record the size of the input tensor.
    // - If the input requires gradient, increment its pending count.
    for (auto &inp : inputs) {
        size_t sz = inp->data.size();
        func->sizes.push_back(sz);
        total_size += sz;
    }

    // Create the output tensor using the total size.
    Tensor out_data(total_size, inputs[0]->data.device());
    float *out_ptr = out_data.data();

    // Copy the data from each input into the output tensor.
    // Use memcpy for CPU and cudaMemcpy for GPU.
    if (inputs[0]->data.device() == CPU) {
        for (auto &inp : inputs) {
            const float *in_ptr = inp->data.data();
            memcpy(out_ptr, in_ptr, inp->data.size() * sizeof(float));
            out_ptr += inp->data.size();
        }
    } else {
        for (auto &inp : inputs) {
            const float *in_ptr = inp->data.data();
            cudaMemcpy(out_ptr, in_ptr, inp->data.size() * sizeof(float), cudaMemcpyDeviceToDevice);
            out_ptr += inp->data.size();
        }
    }

    // Determine if any input requires gradient.
    bool req_grad = false;
    for (auto &inp : inputs)
        req_grad = req_grad || inp->requires_grad;

    // Create the output variable and set the current function as its creator.
    auto out = make_shared<Variable>(out_data, req_grad, "Concat_out");
    out->set_creator(func);
    func->output = out;
    return out;
}

/**
 * @brief Computes the backward pass for the concatenation function.
 *
 * @param grad_output Gradient tensor from the next layer corresponding to the concatenated output.
 * @return Vector of gradient tensors for each input variable.
 */
vector<Tensor> ConcatFunction::backward(const Tensor &grad_output) {
    vector<Tensor> grads;
    // Pointer to traverse the gradient output tensor.
    if (grad_output.device() == CPU) {
        const float *grad_ptr = grad_output.data();
        // For each recorded input size, create a tensor for the gradient and copy the corresponding data.
        for (auto sz : sizes) {
            Tensor grad_in(sz, grad_output.device());
            memcpy(grad_in.data(), grad_ptr, sz * sizeof(float));
            grad_ptr += sz;
            grads.push_back(grad_in);
        }
    } else {
        const float *grad_ptr = grad_output.data();
        for (auto sz : sizes) {
            Tensor grad_in(sz, grad_output.device());
            cudaMemcpy(grad_in.data(), grad_ptr, sz * sizeof(float), cudaMemcpyDeviceToDevice);
            grad_ptr += sz;
            grads.push_back(grad_in);
        }
    }
    return grads;
}
} // namespace autograd
} // namespace kernelnet