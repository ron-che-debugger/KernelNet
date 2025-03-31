#include <iostream>
#include "autograd.hpp"
#include "tensor.hpp"

using namespace std;

#define TEST_TENSORS
#define TEST_AUTOGRAD

#ifdef TEST_TENSORS
#include "tensor_tests.hpp"
#endif

#ifdef TEST_AUTOGRAD
#include "autograd_tests.hpp"
#endif

int main() {
#ifdef TEST_TENSORS
    cout << "===== Running Tensor and CUDA Tests =====" << endl;
    runTensorTests();
#endif

#ifdef TEST_AUTOGRAD
    cout << "\n===== Running Autograd Tests =====" << endl;
    runAutogradTests();
#endif

    return 0;
}