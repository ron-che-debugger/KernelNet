#pragma once
#include "autograd.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

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