#include "autograd.hpp"
#include "tensor.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

// ============================================
// CUDA Kernel Definitions
// ============================================

static __global__ void fill_kernel(float* out, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = value;
    }
}

static __global__ void broadcast_sum_kernel(const float* grad, float* out, size_t total_size, size_t small_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < small_size) {
        float sum_val = 0.0f;
        int repeat = total_size / small_size;
        for (int i = 0; i < repeat; i++){
            sum_val += grad[i * small_size + idx];
        }
        out[idx] = sum_val;
    }
}

// ============================================
// Variable Class Implementations
// ============================================

Variable::Variable(const Tensor& data, bool requires_grad)
    : data(data), requires_grad(requires_grad), grad_initialized(false), pending_count(0)
{
    if (requires_grad) {
        grad = Tensor(data.size(), data.device());
        grad.fill(0.0f);
    }
}

void Variable::set_creator(const FuncPtr& func) {
    creator = func;
}

void Variable::backward(const Tensor& grad_output) {
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

VarPtr Variable::detach() {
    return make_shared<Variable>(data, false);
}

// ============================================
// AddFunction Implementations
// ============================================

VarPtr AddFunction::apply(const VarPtr& a, const VarPtr& b) {
    auto func = make_shared<AddFunction>();
    if (a->requires_grad) a->pending_count++;
    if (b->requires_grad) b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::broadcast_add(a->data, b->data);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad);
    out->set_creator(func);
    func->output = out;
    return out;
}

vector<Tensor> AddFunction::backward(const Tensor& grad_output) {
    auto inp0 = saved_a;
    auto inp1 = saved_b;

    if (inp0->data.size() == inp1->data.size()) {
        return { grad_output, grad_output };
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
        return { grad_output, grad_inp1 };
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
        return { grad_inp0, grad_output };
    }

    return { grad_output, grad_output };
}

// ============================================
// SubtractFunction Implementations
// ============================================

VarPtr SubtractFunction::apply(const VarPtr& a, const VarPtr& b) {
    auto func = make_shared<SubtractFunction>();
    if(a->requires_grad) a->pending_count++;
    if(b->requires_grad) b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::subtract(a->data, b->data);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad);
    out->set_creator(func);
    func->output = out;
    return out;
}

vector<Tensor> SubtractFunction::backward(const Tensor& grad_output) {
    Tensor neg_one(grad_output.size(), grad_output.device());
    neg_one.fill(-1.0f);
    Tensor grad_b = Tensor::multiply(grad_output, neg_one);
    if (!saved_b->requires_grad) {
        grad_b = Tensor(saved_b->data.size(), saved_b->data.device());
        grad_b.fill(0.0f);
    }
    return { grad_output, grad_b };
}

// ============================================
// MultiplyFunction Implementations
// ============================================

VarPtr MultiplyFunction::apply(const VarPtr& a, const VarPtr& b) {
    auto func = make_shared<MultiplyFunction>();
    if(a->requires_grad) a->pending_count++;
    if(b->requires_grad) b->pending_count++;
    func->saved_a = a;
    func->saved_b = b;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::multiply(a->data, b->data);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad);
    out->set_creator(func);
    func->output = out;
    return out;
}

vector<Tensor> MultiplyFunction::backward(const Tensor& grad_output) {
    Tensor grad_a = Tensor::multiply(grad_output, saved_b->data);
    Tensor grad_b = Tensor::multiply(grad_output, saved_a->data);
    return { grad_a, grad_b };
}

// ============================================
// MatMulFunction Implementations
// ============================================

VarPtr MatMulFunction::apply(const VarPtr& a, const VarPtr& b, int M, int K, int N) {
    auto func = make_shared<MatMulFunction>();
    if(a->requires_grad) a->pending_count++;
    if(b->requires_grad) b->pending_count++;
    // Save strong references to inputs.
    func->saved_a = a;
    func->saved_b = b;
    func->M = M; 
    func->K = K; 
    func->N = N;
    func->inputs.push_back(a);
    func->inputs.push_back(b);
    Tensor out_data = Tensor::matmul(a->data, b->data, M, K, N);
    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad);
    out->set_creator(func);
    func->output = out;
    return out;
}

vector<Tensor> MatMulFunction::backward(const Tensor& grad_output) {
    // Use the saved strong references.
    auto inp1 = saved_a;
    auto inp2 = saved_b;
    Tensor b_t = Tensor::transpose(inp2->data, K, N);
    Tensor grad_a = Tensor::matmul(grad_output, b_t, M, N, K);
    Tensor a_t = Tensor::transpose(inp1->data, M, K);
    Tensor grad_b = Tensor::matmul(a_t, grad_output, K, M, N);
    return { grad_a, grad_b };
}

// ============================================
// SumFunction Implementations
// ============================================

VarPtr SumFunction::apply(const VarPtr& input) {
    auto func = make_shared<SumFunction>();
    if(input->requires_grad) input->pending_count++;
    func->saved_input = input;
    func->inputs.push_back(input);
    float total = input->data.sum();
    Tensor out_data(1, input->data.device());
    out_data.fill(total);
    bool req_grad = input->requires_grad;
    auto out = make_shared<Variable>(out_data, req_grad);
    out->set_creator(func);
    func->output = out;
    func->input_size = input->data.size();
    return out;
}

vector<Tensor> SumFunction::backward(const Tensor& grad_output) {
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
    return { grad_input };
}

// ============================================
// MSEFunction Implementations
// ============================================

VarPtr MSEFunction::apply(const VarPtr& prediction, const Tensor& target) {
    auto target_var = make_shared<Variable>(target, false);
    auto diff = SubtractFunction::apply(prediction, target_var);
    auto sq = MultiplyFunction::apply(diff, diff);
    auto sum_loss = SumFunction::apply(sq);
    
    Tensor div_tensor(1, sum_loss->data.device());
    div_tensor.fill(1.0f / prediction->data.size());
    auto scale = make_shared<Variable>(div_tensor, false);
    
    auto mse_loss = MultiplyFunction::apply(sum_loss, scale);
    return mse_loss;
}