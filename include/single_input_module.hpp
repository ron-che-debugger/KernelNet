/**
 * @file single_input_module.hpp
 * @brief Defines `SingleInputModule`, an abstract base for modules that take one input.
 *
 * This class simplifies implementation of layers that expect exactly one input
 * and produce exactly one output.
 */

#pragma once

#include "module.hpp"
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;

namespace kernelnet {
namespace nn {
/**
 * @brief Base class for modules that operate on a single input and produce a single output.
 *
 * Provides a unified interface for both vector-based and scalar-input forward calls.
 * Subclasses must override `forward(const VarPtr&)`.
 */
class SingleInputModule : public Module {
  public:
    /**
     * @brief Abstract forward pass with a single input.
     * @param input The input variable.
     * @return The output variable.
     */
    virtual VarPtr forward(const VarPtr &input) = 0;

    /**
     * @brief Wrapper for Module’s multi-input forward, enforcing exactly one input.
     * @param inputs Vector of inputs (must contain exactly one element).
     * @return A vector with a single output.
     */
    vector<VarPtr> forward(const vector<VarPtr> &inputs) override {
        assert(inputs.size() == 1 && "Expected exactly one input");
        return {forward(inputs[0])};
    }
};
} // namespace nn
} // namespace kernelnet