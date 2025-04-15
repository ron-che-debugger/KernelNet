/**
 * @file sequential.hpp
 * @brief Defines the `Sequential` container module for stacking layers linearly.
 */

#pragma once

#include "api_header.hpp"
#include "single_input_module.hpp"
#include <cassert>
#include <initializer_list>
#include <memory>
#include <vector>

using namespace std;

namespace kernelnet {
namespace nn {

/**
 * @brief Container for stacking layers in a forward sequence.
 *
 * Each submodule must be a `SingleInputModule`, accepting and returning a single `VarPtr`.
 */
class KERNELNET_API Sequential : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    vector<shared_ptr<SingleInputModule>> layers; ///< Ordered list of layers to apply
    bool training;                                ///< Mode flag

    /**
     * @brief Constructs an empty Sequential container.
     */
    Sequential();

    /**
     * @brief Constructs a Sequential container from a list of modules.
     * @param modules An initializer list of shared pointers to modules.
     */
    Sequential(initializer_list<shared_ptr<SingleInputModule>> modules);

    /**
     * @brief Forward pass through all layers.
     * @param input Input variable.
     * @return Output variable after applying all layers sequentially.
     */
    virtual VarPtr forward(const VarPtr &input) override;

    /**
     * @brief Gathers all parameters from each submodule.
     * @return Flattened list of learnable parameters.
     */
    virtual vector<VarPtr> parameters() override;

    /**
     * @brief Sets all layers to training mode.
     */
    void train();

    /**
     * @brief Sets all layers to evaluation mode.
     */
    void eval();
};

} // namespace nn
} // namespace kernelnet