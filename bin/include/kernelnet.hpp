#pragma once

/**
 * @file api_header.hpp
 * @brief Umbrella header for the KernelNet library.
 *
 * This header aggregates all public headers of the KernelNet library.
 * It exposes core functionalities such as autograd support, tensor operations,
 * neural network modules (e.g., Dense, Conv2D, LSTM, etc.), optimizers,
 * trainer classes, and benchmark data loaders.
 *
 * By including this header, users have access to the complete public API of KernelNet.
 */

#include "api_header.hpp"
#include "autograd.hpp"
#include "conv2d.hpp"
#include "dense.hpp"
#include "embedding.hpp"
#include "lstm.hpp"
#include "lstm_wrapper.hpp"
#include "maxpool.hpp"
#include "module.hpp"
#include "optimizer.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "sigmoid.hpp"
#include "single_input_module.hpp"
#include "softmax.hpp"
#include "tanh.hpp"
#include "tensor.hpp"
#include "trainer.hpp"

#include "benchmark/cifar10_data_loader.hpp"
#include "benchmark/cifar10_dataset.hpp"
#include "benchmark/ptb_data_loader.hpp"
#include "benchmark/ptb_dataset.hpp"

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;
using namespace kernelnet::optim;
using namespace kernelnet::trainer;
using namespace kernelnet::data;