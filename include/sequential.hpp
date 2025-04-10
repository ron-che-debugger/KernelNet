#pragma once
#include "module.hpp"
#include <cassert>

using namespace std;

class Sequential : public Module {
  public:
    // Container for submodules.
    vector<shared_ptr<Module>> layers;

    // Training flag (useful later for modules like dropout or batch normalization).
    bool training;

    // Default constructor.
    Sequential() : training(true) {}

    // Constructor from an initializer list.
    Sequential(initializer_list<shared_ptr<Module>> modules)
        : layers(modules), training(true) {}

    // Unified forward method: expects a vector of inputs.
    // For Sequential we assume that each module produces exactly one output.
    vector<VarPtr> forward(const vector<VarPtr> &inputs) override {
        // Begin with the given inputs.
        vector<VarPtr> current = inputs;
        for (auto &layer : layers) {
            // Check for non-null layers.
            assert(layer && "Encountered a null layer in Sequential.");
            // At each stage, we expect a single output.
            assert(current.size() == 1 && "Sequential layers require single output at each stage");
            current = layer->forward(current);
        }
        return current;
    }

    // Convenience overload: allow passing a single VarPtr.
    VarPtr forward(const VarPtr &input) {
        vector<VarPtr> outputs = forward(vector<VarPtr>{input});
        return outputs.front();
    }

    // Collect parameters from all layers.
    vector<VarPtr> parameters() override {
        vector<VarPtr> params;
        for (auto &layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    // Set the network in training mode.
    void train() {
        training = true;
    }

    // Set the network in evaluation mode.
    void eval() {
        training = false;
    }
};
