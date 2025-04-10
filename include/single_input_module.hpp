#pragma once
#include "module.hpp"

class SingleInputModule : public Module {
  public:
    // This is the interface for modules naturally handling one input.
    virtual VarPtr forward(const VarPtr &input) = 0;

    // The default forward() takes a vector and checks that it is exactly one element.
    vector<VarPtr> forward(const vector<VarPtr> &inputs) override {
        assert(inputs.size() == 1 && "Expected exactly one input");
        return {forward(inputs[0])};
    }
};
