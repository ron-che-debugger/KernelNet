#pragma once
#include "single_input_module.hpp"
#include <cassert>

using namespace std;

class Sequential : public SingleInputModule {
  public:
    using SingleInputModule::forward;
    // Container for submodules.
    vector<shared_ptr<SingleInputModule>> layers;

    // Training flag (useful later for modules like dropout or batch normalization).
    bool training;

    // Default constructor.
    Sequential() : training(true) {}

    // Constructor from an initializer list.
    Sequential(initializer_list<shared_ptr<SingleInputModule>> modules)
        : layers(modules), training(true) {}

    // Override the single-argument forward from SingleInputModule.
    // This will be used by clients that call forward() with a single VarPtr.
    VarPtr forward(const VarPtr &input) override {
        VarPtr current = input;
        for (auto &layer : layers) {
            // Check for non-null layers.
            assert(layer && "Encountered a null layer in Sequential.");
            // Here we expect each layer to return exactly one output.
            current = layer->forward(current);
        }
        return current;
    }

    // Collect parameters from all submodules.
    vector<VarPtr> parameters() override {
        vector<VarPtr> params;
        for (auto &layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    // Set the network in training mode.
    void train() { training = true; }

    // Set the network in evaluation mode.
    void eval() { training = false; }
};