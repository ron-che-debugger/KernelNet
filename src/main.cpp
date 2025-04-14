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
// #define TEST_WRAPPERCNN
// #define TEST_WRAPPERCNNCN
// #define TEST_SIGMOID
// #define TEST_TANH
// #define TEST_RELU
// #define TEST_SLICE
// #define TEST_SINGLEEMBEDDING
// #define TEST_SINGLELSTMCELL
// #define TEST_SEQUENTIALLSTM
// #define TEST_WRAPPERLSTM
// #define TEST_CIFAR10
#define TEST_EMBEDLSTM
// #define TEST_PTB

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

#ifdef TEST_WRAPPERCNN
#include "./tests/wrapper_cnn_tests.hpp"
#endif

#ifdef TEST_WRAPPERCNNCN
#include "./tests/wrapper_cnn_cn_tests.hpp"
#endif

#ifdef TEST_SIGMOID
#include "./tests/sigmoid_tests.hpp"
#endif

#ifdef TEST_TANH
#include "./tests/tanh_tests.hpp"
#endif

#ifdef TEST_RELU
#include "./tests/relu_tests.hpp"
#endif

#ifdef TEST_SLICE
#include "./tests/slice_tests.hpp"
#endif

#ifdef TEST_SINGLEEMBEDDING
#include "./tests/single_embedding_layer_tests.hpp"
#endif

#ifdef TEST_SINGLELSTMCELL
#include "./tests/single_lstm_cell_tests.hpp"
#endif

#ifdef TEST_SEQUENTIALLSTM
#include "./tests/sequential_lstm_tests.hpp"
#endif

#ifdef TEST_WRAPPERLSTM
#include "./tests/wrapper_lstm_tests.hpp"
#endif

#ifdef TEST_CIFAR10
#include "./benchmark/cifar10_tests.hpp"
#endif

#ifdef TEST_EMBEDLSTM
#include "./tests/embedding_lstm_tests.hpp"
#endif

#ifdef TEST_PTB
#include "./benchmark/ptb_tests.hpp"
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

#ifdef TEST_WRAPPERCNN
    cout << "\n===== Running Wrapper CNN (Loss: MSE) Tests =====" << endl;
    runWrapperCnnTests();
#endif

#ifdef TEST_WRAPPERCNNCN
    cout << "\n===== Running Wrapper CNN (Loss: Cross-Entropy) Tests =====" << endl;
    runWrapperCnnCnTests();
#endif

#ifdef TEST_SIGMOID
    cout << "\n===== Running Sigmoid Tests =====" << endl;
    runSigmoidTests();
#endif

#ifdef TEST_TANH
    cout << "\n===== Running Tanh Tests =====" << endl;
    runTanhTests();
#endif

#ifdef TEST_RELU
    cout << "\n===== Running Relu Tests =====" << endl;
    runReluTests();
#endif

#ifdef TEST_SLICE
    cout << "\n===== Running Slice Tests =====" << endl;
    runSliceTests();
#endif

#ifdef TEST_SINGLEEMBEDDING
    cout << "\n===== Running Single Embedding Layer Tests =====" << endl;
    runSingleEmbeddingTests();
#endif

#ifdef TEST_SINGLELSTMCELL
    cout << "\n===== Running Single LSTM Cell Tests =====" << endl;
    runSingleLSTMCellTests();
#endif

#ifdef TEST_SEQUENTIALLSTM
    cout << "\n===== Running Sequential LSTM Tests =====" << endl;
    runSeqLSTMTests();
#endif

#ifdef TEST_WRAPPERLSTM
    cout << "\n===== Running Wrapper LSTM Tests =====" << endl;
    runWrapperLSTMTests();
#endif

#ifdef TEST_CIFAR10
    cout << "\n===== Running CIFAR10 Tests =====" << endl;
    runCIFAR10Tests();
#endif

#ifdef TEST_EMBEDLSTM
    cout << "\n===== Running Embedding + LSTM Tests =====" << endl;
    runEmbeddingLSTMTests();
#endif

#ifdef TEST_PTB
    cout << "\n===== Running PTB Tests =====" << endl;
    runPTBTests();
#endif
    return 0;
}