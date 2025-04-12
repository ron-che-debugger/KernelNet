/**
 * @file autograd.hpp
 * @brief Defines core autograd components including Variables and differentiable Functions.
 *
 * This file implements a simple autograd engine with:
 * - The `Variable` class for tracking data and gradients
 * - The `Function` base class for representing differentiable operations
 * - Derived function classes for addition, multiplication, matrix multiplication, etc.
 * - Forward and backward graph construction
 */

#pragma once

#include "tensor.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;

namespace kernelnet {
namespace autograd {
// Forward declarations
class Variable;
class Function;

using VarPtr = shared_ptr<Variable>;
using FuncPtr = shared_ptr<Function>;

/**
 * @brief Abstract base class for all differentiable operations.
 *
 * Each function subclass implements the `backward()` method to return
 * the gradients with respect to its inputs given the output gradient.
 */
class Function {
  public:
    vector<weak_ptr<Variable>> inputs; ///< Weak references to input variables
    weak_ptr<Variable> output;         ///< Weak reference to the output variable

    virtual ~Function() {}

    /**
     * @brief Computes gradients with respect to each input.
     * @param grad_output The gradient flowing into the output.
     * @return A vector of gradients corresponding to each input.
     */
    virtual vector<Tensor> backward(const Tensor &grad_output) = 0;
};

/**
 * @brief A differentiable variable in the computation graph.
 *
 * Wraps a tensor and optionally tracks gradients and a creator function.
 */
class Variable {
  public:
    Tensor data;                  ///< The tensor value.
    Tensor grad;                  ///< The accumulated gradient.
    bool requires_grad;           ///< Whether gradient should be tracked.
    bool grad_initialized;        ///< Whether grad was initialized.
    int pending_count;            ///< Used to handle multiple backward paths.
    shared_ptr<Function> creator; ///< The function that produced this variable.
    string debug_name;            ///< Optional name for debugging.

    /**
     * @brief Constructor for Variable.
     * @param data The tensor to wrap.
     * @param requires_grad Whether this variable should track gradients.
     * @param name Optional debug name.
     */
    Variable(const Tensor &data, bool requires_grad = false, const string &name = "");

    /**
     * @brief Sets the function that created this variable.
     * @param func The creator function.
     */
    void set_creator(const FuncPtr &func);

    /**
     * @brief Initiates the backward pass from this variable.
     * @param grad_output The gradient of the final output w.r.t this variable.
     */
    void backward(const Tensor &grad_output);

    /**
     * @brief Returns a detached copy that does not track gradients.
     * @return A shared pointer to the detached Variable.
     */
    VarPtr detach();
};

// ------------------------------ Function Implementations ------------------------------

/**
 * @brief Element-wise addition: z = a + b
 */
class AddFunction : public Function {
  public:
    VarPtr saved_a, saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b);
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Element-wise subtraction: z = a - b
 */
class SubtractFunction : public Function {
  public:
    VarPtr saved_a, saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b);
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Element-wise multiplication: z = a * b
 */
class MultiplyFunction : public Function {
  public:
    VarPtr saved_a, saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b);
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Matrix multiplication: z = a × b
 */
class MatMulFunction : public Function {
  public:
    int M, K, N;
    VarPtr saved_a, saved_b;

    static VarPtr apply(const VarPtr &a, const VarPtr &b, int M, int K, int N);
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Reduces all elements to a scalar via summation: z = sum(input)
 */
class SumFunction : public Function {
  public:
    int input_size;
    VarPtr saved_input;

    static VarPtr apply(const VarPtr &input);
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Element-wise natural logarithm: z = log(input)
 */
class LogFunction : public Function {
  public:
    VarPtr saved_input;

    static VarPtr apply(const VarPtr &input);
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Mean squared error loss: loss = mean((prediction - target)^2)
 */
class MSEFunction : public Function {
  public:
    /**
     * @brief Computes the MSE loss.
     * @param prediction Model output wrapped in a Variable.
     * @param target Ground truth as a Tensor.
     * @return Scalar loss variable.
     */
    static VarPtr apply(const VarPtr &prediction, const Tensor &target);
};

/**
 * @brief Cross-entropy loss with optional softmax: loss = -target · log(prediction)
 */
class CrossEntropyLossFunction : public Function {
  public:
    /**
     * @brief Computes the cross-entropy loss.
     * @param prediction Logits or softmax predictions.
     * @param target One-hot label vector.
     * @param num_classes Number of classes.
     * @return Scalar loss variable.
     */
    static VarPtr apply(const VarPtr &prediction, const Tensor &target, int num_classes);
};

/**
 * @brief Slices a tensor along its feature dimension.
 *
 * Extracts a sub-tensor from columns [start, end) across all rows.
 */
class SliceFunction : public Function {
  public:
    VarPtr saved_input;

    int batch_size;  ///< Number of rows (samples)
    int total_width; ///< Total number of columns in input
    int start;       ///< Start index for slicing (inclusive)
    int end;         ///< End index for slicing (exclusive)

    /**
     * @brief Forward slicing operation.
     * @param input Input variable with shape [batch_size, total_width]
     * @param batch_size Number of rows.
     * @param start Start index (inclusive).
     * @param end End index (exclusive).
     * @return Sliced output variable.
     */
    static VarPtr apply(const VarPtr &input, int batch_size, int start, int end);

    /**
     * @brief Backward pass for slice: propagates gradient back into correct positions.
     * @param grad_output Gradient flowing into sliced output.
     * @return A vector with a single tensor gradient for the input.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};
} // namespace autograd
} // namespace kernelnet