/**
 * @file optimizer.hpp
 * @brief Defines the SGD optimizer with optional gradient clipping.
 *
 * Implements stochastic gradient descent:
 *     param = param - lr * grad
 *
 * Supports optional element-wise gradient clipping:
 *     grad = clip(grad, -clip_value, +clip_value)
 */

#pragma once

#include "autograd.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 *
 * Updates parameters in-place using their gradients:
 *     param = param - lr * grad
 *
 * Supports optional per-element gradient clipping.
 */
class SGD {
  public:
    vector<VarPtr> params; ///< List of parameters to update
    float lr;              ///< Learning rate
    float clip_value;      ///< Optional gradient clipping (0 = disabled)

    /**
     * @brief Constructs an SGD optimizer.
     *
     * @param params Parameters to optimize.
     * @param lr Learning rate.
     * @param clip_value Maximum gradient magnitude (set to 0 to disable clipping).
     */
    SGD(const vector<VarPtr> &params, float lr, float clip_value = 0.0f);

    /**
     * @brief Applies one optimization step: updates all parameters.
     */
    void step();

    /**
     * @brief Zeros out gradients of all parameters.
     */
    void zero_grad();
};