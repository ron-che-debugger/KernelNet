/**
 * @file module.hpp
 * @brief Defines the abstract base class `Module` for all neural network layers or components.
 *
 * A `Module` provides:
 * - A common interface for forward computation
 * - Parameter access for optimization
 * - Zeroing of gradients for training resets
 */

#pragma once

#include "autograd.hpp"
#include <cassert>
#include <vector>

using namespace std;

/**
 * @brief Abstract base class for all neural network modules.
 *
 * Any custom layer (e.g., Dense, LSTM, Conv2D) should inherit from `Module`
 * and override the `forward` and `parameters` methods.
 */
class Module {
  public:
    /**
     * @brief Applies the module to the input(s).
     *
     * This method must be implemented by subclasses.
     * @param inputs A vector of input Variables.
     * @return A vector of output Variables.
     */
    virtual vector<VarPtr> forward(const vector<VarPtr> &inputs) = 0;

    /**
     * @brief Returns all learnable parameters of the module.
     * @return A vector of parameters (each as `VarPtr`). Defaults to empty.
     */
    virtual vector<VarPtr> parameters() { return {}; }

    /**
     * @brief Zeros out gradients for all parameters.
     *
     * This is typically called before each optimizer step to prevent accumulation.
     */
    virtual void zero_grad() {
        for (auto param : parameters()) {
            param->grad.fill(0.0f);
            param->grad_initialized = false;
        }
    }
};