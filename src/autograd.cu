#pragma once

#include "autograd.hpp"
#include "tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// ============================================
// CUDA Kernel Definitions for Autograd
// ============================================

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

// ============================================
// Variable Class Implementations with Debug Logging
// ============================================

Variable::Variable(const Tensor &data, bool requires_grad, const string &name)
    : data(data), requires_grad(requires_grad), grad_initialized(false), pending_count(0), debug_name(name) {
    if (requires_grad) {
        grad = Tensor(data.size(), data.device());
        grad.fill(0.0f);
        cout << "[DEBUG] Variable " << debug_name << " created with grad required." << endl;
    } else {
        cout << "[DEBUG] Variable " << debug_name << " created without grad." << endl;
    }
}

void Variable::set_creator(const FuncPtr &func) {
    creator = func;
}

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
            creator = nullptr;
        }
    }
}

VarPtr Variable::detach() {
    return make_shared<Variable>(data, false);
}

// ============================================
// AddFunction Implementations with Debug Logging
// ============================================

VarPtr AddFunction::apply(const VarPtr &a, const VarPtr &b) {
    cout << "[DEBUG] AddFunction::apply() called for Variables " << a->debug_name << " and " << b->debug_name << endl;
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
    cout << "[DEBUG] AddFunction::apply() finished. Returning Variable 'Add_out'" << endl;
    return out;
}

vector<Tensor> AddFunction::backward(const Tensor &grad_output) {
    cout << "[DEBUG] AddFunction::backward() called." << endl;
    auto inp0 = saved_a;
    auto inp1 = saved_b;

    if (inp0->data.size() == inp1->data.size()) {
        cout << "[DEBUG] AddFunction: equal sizes, returning grad_output for both inputs." << endl;
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
        cout << "[DEBUG] AddFunction::backward() finished for second input." << endl;
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
        cout << "[DEBUG] AddFunction::backward() finished for first input." << endl;
        return {grad_inp0, grad_output};
    }

    cout << "[DEBUG] AddFunction::backward() returning default gradients." << endl;
    return {grad_output, grad_output};
}

// ============================================
// SubtractFunction Implementations with Debug Logging
// ============================================

VarPtr SubtractFunction::apply(const VarPtr &a, const VarPtr &b) {
    cout << "[DEBUG] SubtractFunction::apply() called for Variables " << a->debug_name << " and " << b->debug_name << endl;
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
    cout << "[DEBUG] SubtractFunction::apply() finished. Returning Variable 'Subtract_out'" << endl;
    return out;
}

vector<Tensor> SubtractFunction::backward(const Tensor &grad_output) {
    cout << "[DEBUG] SubtractFunction::backward() called." << endl;
    Tensor neg_one(grad_output.size(), grad_output.device());
    neg_one.fill(-1.0f);
    Tensor grad_b = Tensor::multiply(grad_output, neg_one);
    if (!saved_b->requires_grad) {
        grad_b = Tensor(saved_b->data.size(), saved_b->data.device());
        grad_b.fill(0.0f);
    }
    cout << "[DEBUG] SubtractFunction::backward() finished." << endl;
    return {grad_output, grad_b};
}

// ============================================
// MultiplyFunction Implementations with Debug Logging
// ============================================

VarPtr MultiplyFunction::apply(const VarPtr &a, const VarPtr &b) {
    cout << "[DEBUG] MultiplyFunction::apply() called for Variables " << a->debug_name << " and " << b->debug_name << endl;
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
    cout << "[DEBUG] MultiplyFunction::apply() finished. Returning Variable 'Multiply_out'" << endl;
    return out;
}

vector<Tensor> MultiplyFunction::backward(const Tensor &grad_output) {
    cout << "[DEBUG] MultiplyFunction::backward() called." << endl;
    Tensor grad_a = Tensor::multiply(grad_output, saved_b->data);
    Tensor grad_b = Tensor::multiply(grad_output, saved_a->data);
    cout << "[DEBUG] MultiplyFunction::backward() finished." << endl;
    return {grad_a, grad_b};
}

// ============================================
// MatMulFunction Implementations with Debug Logging
// ============================================

VarPtr MatMulFunction::apply(const VarPtr &a, const VarPtr &b, int M, int K, int N) {
    cout << "[DEBUG] MatMulFunction::apply() called for Variables " << a->debug_name << " and " << b->debug_name << endl;
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
    cout << "[DEBUG] MatMulFunction::apply() finished. Returning Variable 'MatMul_out'" << endl;
    return out;
}

vector<Tensor> MatMulFunction::backward(const Tensor &grad_output) {
    cout << "[DEBUG] MatMulFunction::backward() called." << endl;
    auto inp1 = saved_a;
    auto inp2 = saved_b;
    Tensor b_t = Tensor::transpose(inp2->data, K, N);
    Tensor grad_a = Tensor::matmul(grad_output, b_t, M, N, K);
    Tensor a_t = Tensor::transpose(inp1->data, M, K);
    Tensor grad_b = Tensor::matmul(a_t, grad_output, K, M, N);
    cout << "[DEBUG] MatMulFunction::backward() finished." << endl;
    return {grad_a, grad_b};
}

// ============================================
// SumFunction Implementations with Debug Logging
// ============================================

VarPtr SumFunction::apply(const VarPtr &input) {
    cout << "[DEBUG] SumFunction::apply() called for Variable " << input->debug_name << endl;
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
    cout << "[DEBUG] SumFunction::apply() finished. Returning Variable 'Sum_out'" << endl;
    return out;
}

vector<Tensor> SumFunction::backward(const Tensor &grad_output) {
    cout << "[DEBUG] SumFunction::backward() called." << endl;
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
    cout << "[DEBUG] SumFunction::backward() finished." << endl;
    return {grad_input};
}

// ============================================
// MSEFunction Implementations with Debug Logging
// ============================================

VarPtr MSEFunction::apply(const VarPtr &prediction, const Tensor &target) {
    cout << "[DEBUG] MSEFunction::apply() called." << endl;
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
    cout << "[DEBUG] MSEFunction::apply() finished. Returning Variable 'mse_loss'" << endl;
    return mse_loss;
}