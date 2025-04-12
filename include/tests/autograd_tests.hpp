#pragma once

#include "autograd.hpp"
#include "tensor.hpp"
#include <iostream>
#include <string>
using namespace std;

#define USE_CUDA

// ===== CPU-specific Autograd Tests =====
inline void runAutogradTestsCPU() {
    cout << "====== Running Autograd Tests on CPU ======" << endl;

    // --- Test Addition Autograd ---
    cout << "=== Test Addition Autograd ===" << endl;
    Tensor addTensor1(5, CPU);
    Tensor addTensor2(5, CPU);
    addTensor1.fill(2.0f);
    addTensor2.fill(3.0f);
    auto varAdd1 = make_shared<Variable>(addTensor1, true);
    auto varAdd2 = make_shared<Variable>(addTensor2, true);
    auto varAddResult = AddFunction::apply(varAdd1, varAdd2);
    Tensor gradAdd(varAddResult->data.size(), CPU);
    gradAdd.fill(1.0f);
    varAddResult->backward(gradAdd);
    cout << "Gradient for first addition variable (expected ones):" << endl;
    varAdd1->grad.print();
    cout << "Gradient for second addition variable (expected ones):" << endl;
    varAdd2->grad.print();

    // --- Test Subtraction Autograd ---
    cout << "\n=== Test Subtraction Autograd ===" << endl;
    Tensor subTensor1(5, CPU);
    Tensor subTensor2(5, CPU);
    subTensor1.fill(5.0f);
    subTensor2.fill(2.0f);
    auto varSub1 = make_shared<Variable>(subTensor1, true);
    auto varSub2 = make_shared<Variable>(subTensor2, true);
    auto varSubResult = SubtractFunction::apply(varSub1, varSub2);
    Tensor gradSub(varSubResult->data.size(), CPU);
    gradSub.fill(1.0f);
    varSubResult->backward(gradSub);
    cout << "Gradient for subtracted variable (expected ones):" << endl;
    varSub1->grad.print();
    cout << "Gradient for subtracted-from variable (expected -ones):" << endl;
    varSub2->grad.print();

    // --- Test Multiplication Autograd ---
    cout << "\n=== Test Multiplication Autograd ===" << endl;
    Tensor mulTensor1(5, CPU);
    Tensor mulTensor2(5, CPU);
    mulTensor1.fill(4.0f);
    mulTensor2.fill(3.0f);
    auto varMul1 = make_shared<Variable>(mulTensor1, true);
    auto varMul2 = make_shared<Variable>(mulTensor2, true);
    auto varMulResult = MultiplyFunction::apply(varMul1, varMul2);
    Tensor gradMul(varMulResult->data.size(), CPU);
    gradMul.fill(1.0f);
    varMulResult->backward(gradMul);
    cout << "Gradient for first multiplication variable (expected 3's):" << endl;
    varMul1->grad.print();
    cout << "Gradient for second multiplication variable (expected 4's):" << endl;
    varMul2->grad.print();

    // --- Test SumFunction ---
    cout << "\n===== Testing SumFunction =====" << endl;
    Tensor sumInputTensor(4, CPU);
    sumInputTensor.data()[0] = 1.0f;
    sumInputTensor.data()[1] = 2.0f;
    sumInputTensor.data()[2] = 3.0f;
    sumInputTensor.data()[3] = 4.0f;
    auto varSumInput = make_shared<Variable>(sumInputTensor, true);
    auto varSumOutput = SumFunction::apply(varSumInput);
    cout << "Forward (sum): " << varSumOutput->data.data()[0] << " (expected 10)" << endl;
    Tensor gradSum(varSumOutput->data.size(), CPU);
    gradSum.fill(1.0f);
    varSumOutput->backward(gradSum);
    cout << "Backward (gradients for sum input): ";
    for (size_t i = 0; i < varSumInput->data.size(); i++) {
        cout << varSumInput->grad.data()[i] << " ";
    }
    cout << endl;

    // --- Test Matrix Multiplication Autograd ---
    cout << "\n=== Test Matrix Multiplication Autograd ===" << endl;
    // Create a 2x3 matrix A and a 3x2 matrix B.
    Tensor matATensor(6, CPU);
    Tensor matBTensor(6, CPU);
    // Fill A with 1,2,3,4,5,6 and B with 7,8,9,10,11,12.
    matATensor.data()[0] = 1;
    matATensor.data()[1] = 2;
    matATensor.data()[2] = 3;
    matATensor.data()[3] = 4;
    matATensor.data()[4] = 5;
    matATensor.data()[5] = 6;
    matBTensor.data()[0] = 7;
    matBTensor.data()[1] = 8;
    matBTensor.data()[2] = 9;
    matBTensor.data()[3] = 10;
    matBTensor.data()[4] = 11;
    matBTensor.data()[5] = 12;
    auto varMatA = make_shared<Variable>(matATensor, true);
    auto varMatB = make_shared<Variable>(matBTensor, true);
    // Multiply (2x3)*(3x2) -> 2x2 result.
    auto varMatMulResult = MatMulFunction::apply(varMatA, varMatB, 2, 3, 2);
    cout << "Forward (MatMul result): ";
    for (int i = 0; i < varMatMulResult->data.size(); i++) {
        cout << varMatMulResult->data.data()[i] << " ";
    }
    cout << endl;
    Tensor gradMat(varMatMulResult->data.size(), CPU);
    gradMat.fill(1.0f);
    varMatMulResult->backward(gradMat);
    cout << "Backward (gradients for matrix A): ";
    for (size_t i = 0; i < varMatA->data.size(); i++) {
        cout << varMatA->grad.data()[i] << " ";
    }
    cout << endl;
    cout << "Backward (gradients for matrix B): ";
    for (size_t i = 0; i < varMatB->data.size(); i++) {
        cout << varMatB->grad.data()[i] << " ";
    }
    cout << endl;

    // --- Test LogFunction Autograd ---
    cout << "\n=== Test LogFunction Autograd ===" << endl;
    Tensor logTensor(5, CPU);
    for (int i = 0; i < logTensor.size(); i++) {
        logTensor.data()[i] = 2.0f; // Use a value > 0 (log defined)
    }
    auto varLog = make_shared<Variable>(logTensor, true, "log_input");
    auto varLogResult = LogFunction::apply(varLog);
    cout << "Forward (Log): " << endl;
    varLogResult->data.print(); // Expected: log(2) ~ 0.6931 for each element

    Tensor gradLog(varLogResult->data.size(), CPU);
    gradLog.fill(1.0f); // Upstream gradients set to 1
    varLogResult->backward(gradLog);
    cout << "Backward (gradients for log_input): " << endl;
    varLog->grad.print(); // Expected: 1/2 = 0.5 for each element

    // --- Test MSE Function Autograd ---
    cout << "\n=== Test MSE Function Autograd ===" << endl;
    Tensor msePredTensor(5, CPU);
    Tensor mseTargetTensor(5, CPU);
    msePredTensor.fill(2.0f);
    mseTargetTensor.fill(3.0f);
    auto varMsePred = make_shared<Variable>(msePredTensor, true);
    auto mseLoss = MSEFunction::apply(varMsePred, mseTargetTensor);
    cout << "Forward (MSE loss): " << mseLoss->data.data()[0] << endl;
    Tensor gradMSE(mseLoss->data.size(), CPU);
    gradMSE.fill(1.0f);
    mseLoss->backward(gradMSE);
    cout << "Backward (gradients for MSE prediction): ";
    for (size_t i = 0; i < varMsePred->grad.size(); i++) {
        cout << varMsePred->grad.data()[i] << " ";
    }
    cout << endl;

    // --- Test CrossEntropyLossFunction Autograd (CPU) ---
    cout << "\n=== Test CrossEntropyLossFunction Autograd ===" << endl;

    // Create a tensor for prediction with two elements: [0.1, 0.9]
    Tensor cePredTensor(2, CPU);
    cePredTensor.data()[0] = 0.1f;
    cePredTensor.data()[1] = 0.9f;
    // Create a target tensor (one-hot for class 1): [0, 1]
    Tensor ceTargetTensor(2, CPU);
    ceTargetTensor.data()[0] = 0.0f;
    ceTargetTensor.data()[1] = 1.0f;

    // Wrap the prediction in a Variable (requires grad).
    auto varCePred = make_shared<Variable>(cePredTensor, true, "ce_pred");

    // Apply the cross-entropy loss function.
    auto ceLoss = CrossEntropyLossFunction::apply(varCePred, ceTargetTensor, 0);

    // Print the forward loss.
    cout << "Forward (CrossEntropy loss): "
         << ceLoss->data.data()[0]
         << " (expected ~ " << -logf(0.9f) << ")" << endl;

    // Create a grad output tensor (ones) for backward propagation.
    Tensor gradCELoss(ceLoss->data.size(), CPU);
    gradCELoss.fill(1.0f);

    // Run backward pass.
    ceLoss->backward(gradCELoss);

    // Print the gradient for the prediction.
    cout << "Backward (gradients for ce_pred):" << endl;
    varCePred->grad.print();
}

// ===== CUDA-specific Autograd Tests =====
inline void runAutogradTestsCUDA() {
    cout << "====== Running Autograd Tests on CUDA ======" << endl;

    // For CUDA tests, allocate on CPU then call toCUDA before testing.
    // --- Test Addition Autograd ---
    cout << "=== Test Addition Autograd ===" << endl;
    Tensor addTensor1(5, CPU);
    Tensor addTensor2(5, CPU);
    addTensor1.fill(2.0f);
    addTensor2.fill(3.0f);
    addTensor1.toCUDA();
    addTensor2.toCUDA();
    auto varAdd1 = make_shared<Variable>(addTensor1, true);
    auto varAdd2 = make_shared<Variable>(addTensor2, true);
    auto varAddResult = AddFunction::apply(varAdd1, varAdd2);
    Tensor gradAdd(varAddResult->data.size(), CUDA);
    gradAdd.fill(1.0f);
    varAddResult->backward(gradAdd);
    cout << "Gradient for first addition variable (expected ones):" << endl;
    varAdd1->grad.toCPU();
    varAdd1->grad.print();
    cout << "Gradient for second addition variable (expected ones):" << endl;
    varAdd2->grad.toCPU();
    varAdd2->grad.print();

    // --- Test Subtraction Autograd ---
    cout << "\n=== Test Subtraction Autograd ===" << endl;
    Tensor subTensor1(5, CPU);
    Tensor subTensor2(5, CPU);
    subTensor1.fill(5.0f);
    subTensor2.fill(2.0f);
    subTensor1.toCUDA();
    subTensor2.toCUDA();
    auto varSub1 = make_shared<Variable>(subTensor1, true);
    auto varSub2 = make_shared<Variable>(subTensor2, true);
    auto varSubResult = SubtractFunction::apply(varSub1, varSub2);
    Tensor gradSub(varSubResult->data.size(), CUDA);
    gradSub.fill(1.0f);
    varSubResult->backward(gradSub);
    cout << "Gradient for subtracted variable (expected ones):" << endl;
    varSub1->grad.toCPU();
    varSub1->grad.print();
    cout << "Gradient for subtracted-from variable (expected -ones):" << endl;
    varSub2->grad.toCPU();
    varSub2->grad.print();

    // --- Test Multiplication Autograd ---
    cout << "\n=== Test Multiplication Autograd ===" << endl;
    Tensor mulTensor1(5, CPU);
    Tensor mulTensor2(5, CPU);
    mulTensor1.fill(4.0f);
    mulTensor2.fill(3.0f);
    mulTensor1.toCUDA();
    mulTensor2.toCUDA();
    auto varMul1 = make_shared<Variable>(mulTensor1, true);
    auto varMul2 = make_shared<Variable>(mulTensor2, true);
    auto varMulResult = MultiplyFunction::apply(varMul1, varMul2);
    Tensor gradMul(varMulResult->data.size(), CUDA);
    gradMul.fill(1.0f);
    varMulResult->backward(gradMul);
    cout << "Gradient for first multiplication variable (expected 3's):" << endl;
    varMul1->grad.toCPU();
    varMul1->grad.print();
    cout << "Gradient for second multiplication variable (expected 4's):" << endl;
    varMul2->grad.toCPU();
    varMul2->grad.print();

    // --- Test SumFunction ---
    cout << "\n===== Testing SumFunction =====" << endl;
    Tensor sumInputTensor(4, CPU);
    sumInputTensor.data()[0] = 1.0f;
    sumInputTensor.data()[1] = 2.0f;
    sumInputTensor.data()[2] = 3.0f;
    sumInputTensor.data()[3] = 4.0f;
    sumInputTensor.toCUDA();
    auto varSumInput = make_shared<Variable>(sumInputTensor, true);
    auto varSumOutput = SumFunction::apply(varSumInput);
    varSumOutput->data.toCPU();
    cout << "Forward (sum): " << varSumOutput->data.data()[0] << " (expected 10)" << endl;
    Tensor gradSum(varSumOutput->data.size(), CUDA);
    gradSum.fill(1.0f);
    varSumOutput->backward(gradSum);
    cout << "Backward (gradients for sum input): ";
    varSumInput->grad.toCPU();
    for (size_t i = 0; i < varSumInput->data.size(); i++) {
        cout << varSumInput->grad.data()[i] << " ";
    }
    cout << endl;

    // --- Test Matrix Multiplication Autograd ---
    cout << "\n=== Test Matrix Multiplication Autograd ===" << endl;
    Tensor matATensor(6, CPU);
    Tensor matBTensor(6, CPU);
    matATensor.data()[0] = 1;
    matATensor.data()[1] = 2;
    matATensor.data()[2] = 3;
    matATensor.data()[3] = 4;
    matATensor.data()[4] = 5;
    matATensor.data()[5] = 6;
    matBTensor.data()[0] = 7;
    matBTensor.data()[1] = 8;
    matBTensor.data()[2] = 9;
    matBTensor.data()[3] = 10;
    matBTensor.data()[4] = 11;
    matBTensor.data()[5] = 12;
    matATensor.toCUDA();
    matBTensor.toCUDA();
    auto varMatA = make_shared<Variable>(matATensor, true);
    auto varMatB = make_shared<Variable>(matBTensor, true);
    auto varMatMulResult = MatMulFunction::apply(varMatA, varMatB, 2, 3, 2);
    cout << "Forward (MatMul result): ";
    varMatMulResult->data.toCPU();
    for (int i = 0; i < varMatMulResult->data.size(); i++) {
        cout << varMatMulResult->data.data()[i] << " ";
    }
    cout << endl;
    Tensor gradMat(varMatMulResult->data.size(), CUDA);
    gradMat.fill(1.0f);
    varMatMulResult->backward(gradMat);
    cout << "Backward (gradients for matrix A): ";
    varMatA->grad.toCPU();
    for (size_t i = 0; i < varMatA->data.size(); i++) {
        cout << varMatA->grad.data()[i] << " ";
    }
    cout << endl;
    cout << "Backward (gradients for matrix B): ";
    varMatB->grad.toCPU();
    for (size_t i = 0; i < varMatB->data.size(); i++) {
        cout << varMatB->grad.data()[i] << " ";
    }
    cout << endl;

    // --- Test LogFunction Autograd ---
    cout << "\n=== Test LogFunction Autograd ===" << endl;
    Tensor logTensor(5, CPU);
    for (int i = 0; i < logTensor.size(); i++) {
        logTensor.data()[i] = 2.0f;
    }
    logTensor.toCUDA();
    auto varLog = make_shared<Variable>(logTensor, true, "log_input_cuda");
    auto varLogResult = LogFunction::apply(varLog);
    varLogResult->data.toCPU();
    cout << "Forward (Log): " << endl;
    varLogResult->data.print(); // Expected: ~0.6931 for each element

    Tensor gradLog(varLogResult->data.size(), CUDA);
    gradLog.fill(1.0f);
    varLogResult->backward(gradLog);
    varLog->grad.toCPU();
    cout << "Backward (gradients for log_input): " << endl;
    varLog->grad.print(); // Expected: 0.5 for each element

    // --- Test MSE Function Autograd ---
    cout << "\n=== Test MSE Function Autograd ===" << endl;
    Tensor msePredTensor(5, CPU);
    Tensor mseTargetTensor(5, CPU);
    msePredTensor.fill(2.0f);
    mseTargetTensor.fill(3.0f);
    msePredTensor.toCUDA();
    mseTargetTensor.toCUDA();
    auto varMsePred = make_shared<Variable>(msePredTensor, true);
    auto mseLoss = MSEFunction::apply(varMsePred, mseTargetTensor);
    mseLoss->data.toCPU();
    cout << "Forward (MSE loss): " << mseLoss->data.data()[0] << endl;
    Tensor gradMSE(mseLoss->data.size(), CUDA);
    gradMSE.fill(1.0f);
    mseLoss->backward(gradMSE);
    cout << "Backward (gradients for MSE prediction): ";
    varMsePred->grad.toCPU();
    for (size_t i = 0; i < varMsePred->grad.size(); i++) {
        cout << varMsePred->grad.data()[i] << " ";
    }
    cout << endl;

    // --- Test CrossEntropyLossFunction Autograd ---
    cout << "\n=== Test CrossEntropyLossFunction Autograd ===" << endl;

    // Create a tensor for prediction on CPU and then move to CUDA.
    Tensor cePredTensor(2, CPU);
    cePredTensor.data()[0] = 0.1f;
    cePredTensor.data()[1] = 0.9f;
    cePredTensor.toCUDA();

    // Create a target tensor (one-hot for class 1) on CPU, then move it to CUDA.
    Tensor ceTargetTensor(2, CPU);
    ceTargetTensor.data()[0] = 0.0f;
    ceTargetTensor.data()[1] = 1.0f;
    ceTargetTensor.toCUDA();

    // Wrap the prediction in a Variable (requires grad).
    auto varCePred = make_shared<Variable>(cePredTensor, true, "ce_pred_cuda");

    // Apply the cross-entropy loss function.
    auto ceLoss = CrossEntropyLossFunction::apply(varCePred, ceTargetTensor, 0);

    // Move forward loss back to CPU to print.
    ceLoss->data.toCPU();
    cout << "Forward (CrossEntropy loss): "
         << ceLoss->data.data()[0]
         << " (expected ~ " << -logf(0.9f) << ")" << endl;

    // Create grad output on CUDA.
    Tensor gradCELoss(ceLoss->data.size(), CUDA);
    gradCELoss.fill(1.0f);

    // Run backward pass.
    ceLoss->backward(gradCELoss);

    // Bring gradients back to CPU for printing.
    varCePred->grad.toCPU();
    cout << "Backward (gradients for ce_pred):" << endl;
    varCePred->grad.print();
}

inline void runAutogradTests() {
    runAutogradTestsCPU();
#ifdef USE_CUDA
    runAutogradTestsCUDA();
#endif
}