#pragma once

#include "tensor.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
    weak_ptr<Variable> output;

    virtual ~Function() {}
    // Given the gradient of the output, compute the gradients for each input.
    virtual vector<Tensor> backward(const Tensor &grad_output) = 0;
};

// Variable wraps a Tensor and holds gradient information along with a pointer to its creator.
class Variable {
  public:
    Tensor data;
    Tensor grad;
    bool requires_grad;
    bool grad_initialized;
    int pending_count;
    shared_ptr<Function> creator;
    string debug_name; // Optional name for tracing.

    // Constructor.
    Variable(const Tensor &data, bool requires_grad = false, const string &name = "");

    // Set the creator of this variable.
    void set_creator(const FuncPtr &func);

    // Backward pass: accumulate gradient and propagate when all children contributions are received.
    void backward(const Tensor &grad_output);

    // Returns a detached copy (without tracking gradients).
    VarPtr detach();
};

// --- Function Classes Declarations ---

class AddFunction : public Function {
  public:
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b);
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class SubtractFunction : public Function {
  public:
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b);
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class MultiplyFunction : public Function {
  public:
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b);
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class MatMulFunction : public Function {
  public:
    int M, K, N;
    VarPtr saved_a;
    VarPtr saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b, int M, int K, int N);
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class SumFunction : public Function {
  public:
    int input_size;
    VarPtr saved_input;

    static VarPtr apply(const VarPtr &input);
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class LogFunction : public Function {
  public:
    VarPtr saved_input;
    static VarPtr apply(const VarPtr &input);

    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

class MSEFunction : public Function {
  public:
    static VarPtr apply(const VarPtr &prediction, const Tensor &target);
};

class CrossEntropyLossFunction : public Function {
  public:
    static VarPtr apply(const VarPtr &prediction, const Tensor &target);
};

class SliceFunction : public Function {
  public:
    VarPtr saved_input;

    // Dimensions for slicing:
    // batch_size: the number of rows (samples) in the input tensor.
    // total_width: the total number of columns (features) in the input tensor.
    // start: the starting index (inclusive) for slicing along the feature dimension.
    // end: the ending index (non-inclusive) for slicing.
    int batch_size;
    int total_width;
    int start;
    int end;

    // Static forward function.
    // Interprets the input tensor as having shape [batch_size, total_width]
    // and extracts the columns in the interval [start, end), resulting in a tensor
    // of shape [batch_size, slice_length] where slice_length = end - start.
    static VarPtr apply(const VarPtr &input, int batch_size, int start, int end);

    // Backward pass: maps the gradients from the sliced output back to the corresponding
    // indices of the input tensor, filling the positions outside the slice with zeros.
    vector<Tensor> backward(const Tensor &grad_output) override;
};