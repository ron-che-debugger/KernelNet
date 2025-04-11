#include "autograd.hpp"

using namespace std;

// ---------- CUDA Kernels ----------

static __global__ void fill_kernel(float *out, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = value;
    }
}

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

// ---------- Variable Implementation ----------

Variable::Variable(const Tensor &data, bool requires_grad, const string &name)
    : data(data), requires_grad(requires_grad), grad_initialized(false), pending_count(0), debug_name(name) {
    if (requires_grad) {
        grad = Tensor(data.size(), data.device());
        grad.fill(0.0f);
    }
}

void Variable::set_creator(const FuncPtr &func) {
    creator = func;
}

void Variable::backward(const Tensor &grad_output) {
    if (requires_grad) {
        if (!grad_initialized) {
            grad = grad_output;
            grad_initialized = true;
        } else {
            grad = Tensor::add(grad, grad_output);
        }
        if (pending_count > 0)
            pending_count--;
        if (creator && pending_count == 0) {
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

/*
void Variable::backward(const Tensor &grad_output) {
    string name = debug_name.empty() ? ("@" + to_string((uintptr_t)this)) : debug_name;
    cout << "[DEBUG] Enter backward for Variable " << name
         << " | size: " << data.size() << endl;

    if (requires_grad) {
        if (!grad_initialized) {
            grad = grad_output;
            grad_initialized = true;
            cout << "[DEBUG] Gradient initialized for Variable " << name << endl;
        } else {
            grad = Tensor::add(grad, grad_output);
            cout << "[DEBUG] Gradient accumulated for Variable " << name << endl;
        }

        if (pending_count > 0)
            pending_count--;
        cout << "[DEBUG] Variable " << name << " pending_count now " << pending_count << endl;

        if (creator && pending_count == 0) {
            cout << "[DEBUG] Variable " << name << " has creator, invoking backward." << endl;
            vector<Tensor> input_grads = creator->backward(grad);
            cout << "[DEBUG] Creator backward returned for Variable " << name << endl;
            for (size_t i = 0; i < creator->inputs.size(); ++i) {
                if (auto inp = creator->inputs[i].lock()) {
                    string inp_name = inp->debug_name.empty() ? ("@" + to_string((uintptr_t)inp.get())) : inp->debug_name;
                    cout << "[DEBUG] Propagating to input " << i << " (Variable "
                         << inp_name << ", size: " << inp->data.size() << ")" << endl;
                    if (inp->requires_grad) {
                        inp->backward(input_grads[i]);
                    } else {
                        cout << "[DEBUG] Skipped input " << i << " (no grad required)" << endl;
                    }
                } else {
                    cout << "[ERROR] Input " << i << " is expired or null!" << endl;
                }
            }
        }
    }
}
*/

VarPtr Variable::detach() {
    return make_shared<Variable>(data, false);
}

// ---------- AddFunction Implementation ----------

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

// ---------- SubtractFunction Implementation ----------

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

// ---------- MultiplyFunction Implementation ----------

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

vector<Tensor> MultiplyFunction::backward(const Tensor &grad_output) {
    Tensor grad_a = Tensor::multiply(grad_output, saved_b->data);
    Tensor grad_b = Tensor::multiply(grad_output, saved_a->data);
    return {grad_a, grad_b};
}

// ---------- MatMulFunction Implementation ----------

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

vector<Tensor> MatMulFunction::backward(const Tensor &grad_output) {
    auto inp1 = saved_a;
    auto inp2 = saved_b;
    Tensor b_t = Tensor::transpose(inp2->data, K, N);
    Tensor grad_a = Tensor::matmul(grad_output, b_t, M, N, K);
    Tensor a_t = Tensor::transpose(inp1->data, M, K);
    Tensor grad_b = Tensor::matmul(a_t, grad_output, K, M, N);
    return {grad_a, grad_b};
}

// ---------- SumFunction Implementation ----------

VarPtr SumFunction::apply(const VarPtr &input) {
    auto func = make_shared<SumFunction>();
    if (input->requires_grad)
        input->pending_count++;
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

// ---------- LogFunction Implementation ----------
// Kernel for the forward pass: compute element-wise log.
__global__ void log_forward_kernel(const float *in, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1e-8f;
    if (idx < size) {
        out[idx] = logf(in[idx] + epsilon);
    }
}
// Kernel for the backward pass: compute derivative of log (1/x) multiplied by grad_output.
__global__ void log_backward_kernel(const float *grad_out, const float *in, float *grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1e-8f;
    if (idx < size) {
        grad_in[idx] = grad_out[idx] / (in[idx] + epsilon);
    }
}

// ---------- LogFunction Implementation ----------

VarPtr LogFunction::apply(const VarPtr &input) {
    // Create an instance of LogFunction to hold saved state.
    auto func = make_shared<LogFunction>();

    // Save input for backward computation.
    func->saved_input = input;
    func->inputs.push_back(input);

    size_t size = input->data.size();
    // Allocate an output tensor on the same device as the input.
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

// ---------- MSEFunction Implementation ----------

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

// ---------- CrossEntropyLossFunction Implementation ----------

// Implementation of the cross-entropy loss function.
VarPtr CrossEntropyLossFunction::apply(const VarPtr &prediction, const Tensor &target, int num_classes) {
    // Wrap the target Tensor in a Variable (no gradient tracking).
    auto target_var = make_shared<Variable>(target, false, "target");

    // Compute element-wise natural logarithm of the predictions.
    auto log_pred = LogFunction::apply(prediction);
    log_pred->debug_name = "log_pred";

    // Multiply element-wise: target ⊙ log(prediction)
    auto prod = MultiplyFunction::apply(target_var, log_pred);
    prod->debug_name = "prod";

    // Sum over all elements (aggregates the loss across all dimensions)
    auto sum_loss = SumFunction::apply(prod);
    sum_loss->debug_name = "sum_loss";

    if (num_classes > 0) {
        // Compute batch size by dividing the total number of elements in target by num_classes.
        int batch_size = target.size() / num_classes;

        // Create a scaling tensor holding -1.0 / batch_size.
        Tensor scale_tensor(1, sum_loss->data.device());
        scale_tensor.fill(-1.0f / static_cast<float>(batch_size));
        auto scale = make_shared<Variable>(scale_tensor, false, "scale");

        // Multiply the summed loss by the scaling factor so that the loss is averaged per sample.
        auto cross_entropy_loss = MultiplyFunction::apply(sum_loss, scale);
        cross_entropy_loss->debug_name = "cross_entropy_loss";
        return cross_entropy_loss;
    } else {
        // Regular behavior: simply multiply by -1.
        Tensor neg_tensor(1, sum_loss->data.device());
        neg_tensor.fill(-1.0f);
        auto neg_one = make_shared<Variable>(neg_tensor, false, "neg_one");

        auto cross_entropy_loss = MultiplyFunction::apply(sum_loss, neg_one);
        cross_entropy_loss->debug_name = "cross_entropy_loss";
        return cross_entropy_loss;
    }
}

// CUDA kernel for the forward pass.
// It copies the slice from the input to the output.
// Input is interpreted as a 2D array of shape [batch_size, total_width] stored in row-major order.
// The output has shape [batch_size, slice_len], with slice_len = end - start.
__global__ void slice_forward_kernel(const float *in, float *out, int total_width, int slice_start, int slice_len, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * slice_len;
    if (idx < total) {
        int b = idx / slice_len;
        int i = idx % slice_len;
        out[idx] = in[b * total_width + slice_start + i];
    }
}

// CUDA kernel for the backward pass.
// It writes the gradient from the sliced output into the correct positions of the
// input gradient tensor. The positions that are not part of the slice are set to zero.
// Input grad_output is of shape [batch_size, slice_len] and grad_input is of shape
// [batch_size, total_width].
__global__ void slice_backward_kernel(const float *grad_out, float *grad_in, int total_width, int slice_start, int slice_len, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * total_width;
    if (idx < total) {
        int b = idx / total_width;
        int j = idx % total_width;
        float grad = 0.0f;
        // If j is within the slice, copy the corresponding gradient.
        if (j >= slice_start && j < slice_start + slice_len) {
            int i = j - slice_start;
            grad = grad_out[b * slice_len + i];
        }
        grad_in[idx] = grad;
    }
}

VarPtr SliceFunction::apply(const VarPtr &input, int batch_size, int start, int end) {
    auto func = make_shared<SliceFunction>();
    // Save the input as a hard pointer for backward.
    func->saved_input = input;

    func->batch_size = batch_size;
    func->start = start;
    func->end = end;

    // Compute total_width from the input.
    int total_size = input->data.size();
    func->total_width = total_size / batch_size;
    int slice_len = end - start;

    Tensor out_tensor(batch_size * slice_len, input->data.device());
    int total = batch_size * slice_len;

    if (input->data.device() == CPU) {
        // CPU branch: simple loop.
        const float *in_ptr = input->data.data();
        float *out_ptr = out_tensor.data();
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < slice_len; i++) {
                out_ptr[b * slice_len + i] = in_ptr[b * func->total_width + start + i];
            }
        }
    } else {
        // CUDA branch.
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

vector<Tensor> SliceFunction::backward(const Tensor &grad_output) {
    int slice_len = end - start;
    int grad_in_size = batch_size * total_width;
    Tensor grad_input(grad_in_size, grad_output.device());

    if (grad_output.device() == CPU) {
        // CPU branch: initialize grad_input to zeros and then copy.
        float *grad_in_ptr = grad_input.data();
        const float *grad_out_ptr = grad_output.data();
        // Zero initialize.
        for (int i = 0; i < grad_in_size; i++) {
            grad_in_ptr[i] = 0;
        }
        // For each batch, copy the corresponding slice from grad_output.
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < slice_len; i++) {
                grad_in_ptr[b * total_width + start + i] = grad_out_ptr[b * slice_len + i];
            }
        }
    } else {
        // CUDA branch.
        int total = batch_size * total_width;
        dim3 blockSize(256);
        dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
        // Zero initialize grad_input on device.
        cudaMemset(grad_input.data(), 0, grad_in_size * sizeof(float));
        slice_backward_kernel<<<gridSize, blockSize>>>(grad_output.data(), grad_input.data(), total_width, start, slice_len, batch_size);
        cudaDeviceSynchronize();
    }

    return {grad_input};
}