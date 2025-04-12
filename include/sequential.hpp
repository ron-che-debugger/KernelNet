/**
 * @file sequential.hpp
 * @brief Defines the `Sequential` container module for stacking layers linearly.
 */

#pragma once

#include "single_input_module.hpp"
#include <cassert>

using namespace std;

/**
 * @brief Container for stacking layers in a forward sequence.
 *
 * Each submodule must be a `SingleInputModule`, accepting and returning a single `VarPtr`.
 */
class Sequential : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    vector<shared_ptr<SingleInputModule>> layers; ///< Ordered list of layers to apply
    bool training;                                ///< Mode flag (used by some layers like dropout/batch norm)

    /**
     * @brief Constructs an empty Sequential container.
     */
    Sequential() : training(true) {}

    /**
     * @brief Constructs a Sequential container from a list of modules.
     * @param modules An initializer list of shared pointers to modules.
     */
    Sequential(initializer_list<shared_ptr<SingleInputModule>> modules)
        : layers(modules), training(true) {}

    /**
     * @brief Forward pass through all layers.
     * @param input Input variable.
     * @return Output variable after applying all layers sequentially.
     */
    VarPtr forward(const VarPtr &input) override {
        VarPtr current = input;
        for (auto &layer : layers) {
            assert(layer && "Encountered a null layer in Sequential.");
            current = layer->forward(current);
        }
        return current;
    }

    /**
     * @brief Gathers all parameters from each submodule.
     * @return Flattened list of learnable parameters.
     */
    vector<VarPtr> parameters() override {
        vector<VarPtr> params;
        for (auto &layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    /**
     * @brief Sets all layers to training mode.
     */
    void train() { training = true; }

    /**
     * @brief Sets all layers to evaluation mode.
     */
    void eval() { training = false; }
};