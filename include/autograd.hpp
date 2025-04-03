#pragma once
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <string>

using namespace std;

// Forward declarations.
class Variable;
class Function;

using VarPtr = shared_ptr<Variable>;
using FuncPtr = shared_ptr<Function>;

// --- Autograd Classes ---

// Base class for functions in the computation graph.
// Inputs are stored as weak pointers to avoid reference cycles.
class Function {
public:
    vector<weak_ptr<Variable>> inputs;
    // Optionally, store output as a weak pointer.
    weak_ptr<Variable> output;

    virtual ~Function() {}
    // Given the gradient of the output, compute the gradients for each input.
    virtual vector<Tensor> backward(const Tensor& grad_output) = 0;
};

// Variable wraps a Tensor and holds gradient information along with a pointer to its creator.
class Variable {
public:
    Tensor data;
    Tensor grad;
    bool requires_grad;
    bool grad_initialized;
    // For shared nodes: number of children (pending backward contributions).
    int pending_count;
    // Pointer to the function that created this variable.
    shared_ptr<Function> creator;

    // Constructor.
    Variable(const Tensor& data, bool requires_grad = false);

    // Set the creator of this variable.
    void set_creator(const FuncPtr& func);

    // Backward pass: accumulate gradient and propagate when all children contributions are received.
    void backward(const Tensor& grad_output);

    // Returns a detached copy (without tracking gradients).
    VarPtr detach();
};

// --- Function Classes Declarations ---

// Addition: computes a + b.
class AddFunction : public Function {
public:
    // Save strong references for backward.
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr& a, const VarPtr& b);
    virtual vector<Tensor> backward(const Tensor& grad_output) override;
};

// Subtraction: computes a - b.
class SubtractFunction : public Function {
public:
    // Save strong references for backward.
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr& a, const VarPtr& b);
    virtual vector<Tensor> backward(const Tensor& grad_output) override;
};

// Multiplication: computes a * b.
class MultiplyFunction : public Function {
public:
    // Save strong references for backward.
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr& a, const VarPtr& b);
    virtual vector<Tensor> backward(const Tensor& grad_output) override;
};

// Matrix multiplication: computes a matrix product.
class MatMulFunction : public Function {
public:
    int M, K, N;
    VarPtr saved_a;
    VarPtr saved_b;
    
    static VarPtr apply(const VarPtr& a, const VarPtr& b, int M, int K, int N);
    virtual vector<Tensor> backward(const Tensor& grad_output) override;
};

// SumFunction: computes the sum of all elements in the input.
class SumFunction : public Function {
public:
    int input_size;
    VarPtr saved_input;
    static VarPtr apply(const VarPtr& input);
    virtual vector<Tensor> backward(const Tensor& grad_output) override;
};

// MSEFunction: computes the mean squared error between a prediction and a target tensor.
class MSEFunction : public Function {
public:
    static VarPtr apply(const VarPtr& prediction, const Tensor& target);
};