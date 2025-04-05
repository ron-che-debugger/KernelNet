#include <iostream>
using namespace std;

#define TEST_TENSORS
#define TEST_AUTOGRAD
#define TEST_SINGLEDENSELAYER
#define TEST_SINGLECONV2DDENSELAYER

#ifdef TEST_TENSORS
#include "tensor_tests.hpp"
#endif

#ifdef TEST_AUTOGRAD
#include "autograd_tests.hpp"
#endif

#ifdef TEST_SINGLEDENSELAYER
#include "single_dense_layer_tests.hpp"
#endif

#ifdef TEST_SINGLECONV2DDENSELAYER
#include "single_conv2d_dense_test.hpp"
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

#ifdef TEST_SINGLECONV2DDENSELAYER
    cout << "\n===== Running Single Conv2D Dense Tests =====" << endl;
    runSingleConv2DDenseTests();
#endif
    return 0;
}