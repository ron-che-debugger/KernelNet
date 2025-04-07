#include <iostream>
using namespace std;

// #define TEST_TENSORS
// #define TEST_AUTOGRAD
// #define TEST_SINGLEDENSELAYER
// #define TEST_SINGLEMAXPOOLLAYER
// #define TEST_SOFTMAX
// #define TEST_SINGLECONV2DLAYER
// #define TEST_SINGLECONV2DDENSE
// #define TEST_SIMPLECNN
#define TEST_SIGMOID
#define TEST_TANH

#ifdef TEST_TENSORS
#include "tests/tensor_tests.hpp"
#endif

#ifdef TEST_AUTOGRAD
#include "tests/autograd_tests.hpp"
#endif

#ifdef TEST_SINGLEDENSELAYER
#include "tests/single_dense_layer_tests.hpp"
#endif

#ifdef TEST_SINGLEMAXPOOLLAYER
#include "tests/maxpool_tests.hpp"
#endif

#ifdef TEST_SOFTMAX
#include "tests/softmax_tests.hpp"
#endif

#ifdef TEST_SINGLECONV2DLAYER
#include "tests/single_conv2d_tests.hpp"
#endif

#ifdef TEST_SINGLECONV2DDENSE
#include "tests/single_conv2d_dense_test.hpp"
#endif

#ifdef TEST_SIMPLECNN
#include "./tests/simple_cnn_tests.hpp"
#endif

#ifdef TEST_SIGMOID
#include "./tests/sigmoid_tests.hpp"
#endif

#ifdef TEST_TANH
#include "./tests/tanh_tests.hpp"
#endif

int main() {
#ifdef TEST_TENSORS
    cout << "===== Running Tensor Tests =====" << endl;
    runTensorTests();
#endif

#ifdef TEST_AUTOGRAD
    cout << "\n===== Running Autograd Tests =====" << endl;
    runAutogradTests();
#endif

#ifdef TEST_SINGLEDENSELAYER
    cout << "\n===== Running Single Dense Layer Tests =====" << endl;
    runSingleDenseLayerTests();
#endif

#ifdef TEST_SINGLEMAXPOOLLAYER
    cout << "\n===== Running Single Maxpool Layer Tests =====" << endl;
    runSingleMaxpoolTests();
#endif

#ifdef TEST_SOFTMAX
    cout << "\n===== Running Single Softmax Layer Tests =====" << endl;
    runSoftmaxTests();
#endif

#ifdef TEST_SINGLECONV2DLAYER
    cout << "\n===== Running Single Conv2D Tests =====" << endl;
    runSingleConv2DTests();
#endif

#ifdef TEST_SINGLECONV2DDENSE
    cout << "\n===== Running Single Conv2D + Dense Tests =====" << endl;
    runSingleConv2DDenseTests();
#endif

#ifdef TEST_SIMPLECNN
    cout << "\n===== Running Simple CNN Tests =====" << endl;
    runSimpleCnnTests();
#endif

#ifdef TEST_SIGMOID
    cout << "\n===== Running Sigmoid Tests =====" << endl;
    runSigmoidTests();
#endif

#ifdef TEST_TANH
    cout << "\n===== Running Tanh Tests =====" << endl;
    runTanhTests();
#endif
    return 0;
}