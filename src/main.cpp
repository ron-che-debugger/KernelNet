#include <iostream>
using namespace std;

#define TEST_TENSORS
#define TEST_AUTOGRAD
#define TEST_SINGLEDENSELAYER

#ifdef TEST_TENSORS
#include "tensor_tests.hpp"
#endif

#ifdef TEST_AUTOGRAD
#include "autograd_tests.hpp"
#endif

#ifdef TEST_SINGLEDENSELAYER
#include "single_dense_layer_tests.hpp"
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

    return 0;
}