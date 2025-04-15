#include "sequential.hpp"

namespace kernelnet {
namespace nn {

/**
 * @brief Constructs an empty Sequential container.
 */
Sequential::Sequential() : training(true) {}

/**
 * @brief Constructs a Sequential container from a list of modules.
 * @param modules An initializer list of shared pointers to modules.
 */
Sequential::Sequential(initializer_list<shared_ptr<SingleInputModule>> modules)
    : layers(modules), training(true) {}

/**
 * @brief Forward pass through all layers.
 * @param input Input variable.
 * @return Output variable after applying all layers sequentially.
 */
VarPtr Sequential::forward(const VarPtr &input) {
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
vector<VarPtr> Sequential::parameters() {
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
void Sequential::train() {
    training = true;
}

/**
 * @brief Sets all layers to evaluation mode.
 */
void Sequential::eval() {
    training = false;
}

} // namespace nn
} // namespace kernelnet
