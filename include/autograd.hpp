#pragma once
#include "tensor.hpp"
#include <vector>

using namespace std;

class Variable;

class Function{
public:
    vector<Variable*> inputs;

    Variable* output;

    virtual ~Function() {}

    virtual vector<Tensor> backward(const Tensor& grad_output) = 0;
};

// Variable wraps a Tensor and holds gradient information and a pointer to its creator.
class Variable {
public:
    Tensor data;
    Tensor grad;
    bool requires_grad;
    bool grad_initialized;
    Function* creator;

    // Constructor: if requires_grad is true, we initialize the grad tensor to zeros.
    Variable(const Tensor& data, bool requires_grad = false)
        : data(data), requires_grad(requires_grad), grad_initialized(false), creator(nullptr){
        if (requires_grad){
            grad = Tensor(data.size(), data.device());
            grad.fill(0.0f);
        }
    }

    void set_creator(Function* func){
        creator = func;
    }

    void backward(const Tensor& grad_output){
        if (requires_grad){
            if (!grad_initialized){
                grad = grad_output;
                grad_initialized = true;
            }
            else {
                grad = Tensor::add(grad, grad_output);
            }

            if (creator){
                vector<Tensor> input_grads = creator->backward(grad);
                for (size_t i = 0; i < creator->inputs.size(); ++i){
                    if (creator->inputs[i]->requires_grad){
                        creator->inputs[i]->backward(input_grads[i]);
                    }
                }
            }
        }
    }
};

// Addition: forward computes a + b, and backward passes the same grad to both inputs.
class AddFunction : public Function {
public:
    static Variable* apply(Variable* a, Variable* b){
        AddFunction* func = new AddFunction();
        func->inputs.push_back(a);
        func->inputs.push_back(b);
        Tensor out_data = Tensor::add(a->data, b->data);
        
        // The output requires grad if either input does.
        bool req_grad = a->requires_grad || b->requires_grad;
        Variable* out = new Variable(out_data, req_grad);

        out->set_creator(func);
        func->output = out;
        return out;
    }

    // For addition, the derivative with respect to each input is 1.
    vector<Tensor> backward(const Tensor& grad_output) override{
        return {grad_output, grad_output};
    }
};

// Subtraction: forward computes a - b, and backward returns grad_output for a and -grad_output for b.
class SubtractFunction : public Function {
public:
    static Variable* apply(Variable* a, Variable* b) {
        SubtractFunction* func = new SubtractFunction();
        func->inputs.push_back(a);
        func->inputs.push_back(b);
        Tensor out_data = Tensor::subtract(a->data, b->data);

        bool req_grad = a->requires_grad || b->requires_grad;
        Variable* out = new Variable(out_data, req_grad);

        out->set_creator(func);
        func->output = out;
        return out;
    }

    // For subtraction: d(a-b)/da = 1, d(a-b)/db = -1.
    vector<Tensor> backward(const Tensor& grad_output) override {
        Tensor neg_one(grad_output.size(), grad_output.device());
        neg_one.fill(-1.0f);
        Tensor grad_b = Tensor::multiply(grad_output, neg_one);
        return { grad_output, grad_b };
    }
};

// Multiplication: forward computes a * b, and backward multiplies grad_output by the other input’s data.
class MultiplyFunction : public Function {
public:
    static Variable* apply(Variable* a, Variable* b) {
        MultiplyFunction* func = new MultiplyFunction();
        func->inputs.push_back(a);
        func->inputs.push_back(b);

        Tensor out_data = Tensor::multiply(a->data, b->data);
        bool req_grad = a->requires_grad || b->requires_grad;

        Variable* out = new Variable(out_data, req_grad);
        out->set_creator(func);
        func->output = out;
        return out;
    }

    // The gradient for a is grad_output * b and for b is grad_output * a.
    vector<Tensor> backward(const Tensor& grad_output) override {
        Tensor grad_a = Tensor::multiply(grad_output, inputs[1]->data);
        Tensor grad_b = Tensor::multiply(grad_output, inputs[0]->data);
        return {grad_a, grad_b};
    }
};