#ifndef AUTOGRAD_TESTS_HPP
#define AUTOGRAD_TESTS_HPP

#include "tensor.hpp"
#include "autograd.hpp"
#include <iostream>
using namespace std;

inline void runAutogradTests() {
    // ========= Test Addition Autograd =========
    cout << "=== Test Addition Autograd ===" << endl;
    
    // Create two tensors and wrap them as Variables with gradients enabled.
    Tensor addTensor1(5, CPU);
    Tensor addTensor2(5, CPU);
    addTensor1.fill(2.0f);
    addTensor2.fill(3.0f);
    
    Variable* varAdd1 = new Variable(addTensor1, true);
    Variable* varAdd2 = new Variable(addTensor2, true);
    
    // Compute addResult = varAdd1 + varAdd2 using the autograd-enabled addition.
    Variable* varAddResult = AddFunction::apply(varAdd1, varAdd2);
    
    // Create a gradient tensor (ones) for the output.
    Tensor gradAdd(varAddResult->data.size(), CPU);
    gradAdd.fill(1.0f);
    
    // Backpropagate through addResult.
    varAddResult->backward(gradAdd);
    
    cout << "Gradient for first addition variable (expected ones):" << endl;
    varAdd1->grad.print();
    cout << "Gradient for second addition variable (expected ones):" << endl;
    varAdd2->grad.print();
    
    // ========= Test Subtraction Autograd =========
    cout << "\n=== Test Subtraction Autograd ===" << endl;
    
    Tensor subTensor1(5, CPU);
    Tensor subTensor2(5, CPU);
    subTensor1.fill(5.0f);
    subTensor2.fill(2.0f);
    
    Variable* varSub1 = new Variable(subTensor1, true);
    Variable* varSub2 = new Variable(subTensor2, true);
    
    // Compute subResult = varSub1 - varSub2.
    Variable* varSubResult = SubtractFunction::apply(varSub1, varSub2);
    
    Tensor gradSub(varSubResult->data.size(), CPU);
    gradSub.fill(1.0f);
    varSubResult->backward(gradSub);
    
    cout << "Gradient for subtracted variable (expected ones):" << endl;
    varSub1->grad.print();
    cout << "Gradient for subtracted-from variable (expected -ones):" << endl;
    varSub2->grad.print();
    
    // ========= Test Multiplication Autograd =========
    cout << "\n=== Test Multiplication Autograd ===" << endl;
    
    Tensor mulTensor1(5, CPU);
    Tensor mulTensor2(5, CPU);
    mulTensor1.fill(4.0f);
    mulTensor2.fill(3.0f);
    
    Variable* varMul1 = new Variable(mulTensor1, true);
    Variable* varMul2 = new Variable(mulTensor2, true);
    
    // Compute mulResult = varMul1 * varMul2.
    Variable* varMulResult = MultiplyFunction::apply(varMul1, varMul2);
    
    Tensor gradMul(varMulResult->data.size(), CPU);
    gradMul.fill(1.0f);
    varMulResult->backward(gradMul);
    
    cout << "Gradient for first multiplication variable (expected 3's):" << endl;
    varMul1->grad.print();
    cout << "Gradient for second multiplication variable (expected 4's):" << endl;
    varMul2->grad.print();

    // ========= Test SumFunction =========
    cout << "\n===== Testing SumFunction =====" << endl;
    Tensor sumInputTensor(4, CPU);
    sumInputTensor.data()[0] = 1.0f;
    sumInputTensor.data()[1] = 2.0f;
    sumInputTensor.data()[2] = 3.0f;
    sumInputTensor.data()[3] = 4.0f;
    
    Variable* varSumInput = new Variable(sumInputTensor, true);
    Variable* varSumOutput = SumFunction::apply(varSumInput);
    
    cout << "Forward (sum): " << varSumOutput->data.data()[0] << " (expected 10)" << std::endl;
    
    Tensor gradSum(varSumOutput->data.size(), CPU);
    gradSum.fill(1.0f);
    varSumOutput->backward(gradSum);
    
    cout << "Backward (gradients for sum input): ";
    for (size_t i = 0; i < varSumInput->data.size(); i++) {
         cout << varSumInput->grad.data()[i] << " ";
    }
    cout << endl;

    // ========= Test Matrix Multiplication Autograd =========
    cout << "\n=== Test Matrix Multiplication Autograd ===" << endl;
    Tensor matATensor(6, CPU);
    Tensor matBTensor(6, CPU);
    // Fill matATensor with 1,2,3,4,5,6 and matBTensor with 7,8,9,10,11,12.
    matATensor.data()[0] = 1; matATensor.data()[1] = 2; matATensor.data()[2] = 3;
    matATensor.data()[3] = 4; matATensor.data()[4] = 5; matATensor.data()[5] = 6;
    matBTensor.data()[0] = 7; matBTensor.data()[1] = 8; matBTensor.data()[2] = 9;
    matBTensor.data()[3] = 10; matBTensor.data()[4] = 11; matBTensor.data()[5] = 12;
    
    Variable* varMatA = new Variable(matATensor, true);
    Variable* varMatB = new Variable(matBTensor, true);
    
    // Compute matMulResult = varMatA * varMatB; dimensions: (2x3)*(3x2) -> 2x2 matrix.
    Variable* varMatMulResult = MatMulFunction::apply(varMatA, varMatB, 2, 3, 2);
    
    cout << "Forward (MatMul result): ";
    for (int i = 0; i < 4; i++){
        cout << varMatMulResult->data.data()[i] << " ";
    }
    cout << endl;
    
    Tensor gradMat(varMatMulResult->data.size(), CPU);
    gradMat.fill(1.0f);
    varMatMulResult->backward(gradMat);
    
    cout << "Backward (gradients for matrix A): ";
    for (size_t i = 0; i < varMatA->data.size(); i++){
        cout << varMatA->grad.data()[i] << " ";
    }
    cout << endl;
    cout << "Backward (gradients for matrix B): ";
    for (size_t i = 0; i < varMatB->data.size(); i++){
        cout << varMatB->grad.data()[i] << " ";
    }
    cout << endl;
    
    // ========= Test MSE Function Autograd =========
    cout << "\n===== Testing MSEFunction =====" << endl;
    Tensor predTensor(3, CPU);
    predTensor.data()[0] = 2.0f;
    predTensor.data()[1] = 3.0f;
    predTensor.data()[2] = 4.0f;
    
    Variable* varPrediction = new Variable(predTensor, true);
    
    Tensor targetTensor(3, CPU);
    targetTensor.data()[0] = 1.0f;
    targetTensor.data()[1] = 2.0f;
    targetTensor.data()[2] = 3.0f;
    
    // Compute mseLoss = MSEFunction(prediction, target).
    Variable* varMSELoss = MSEFunction::apply(varPrediction, targetTensor);
    
    cout << "Forward (MSE loss): " << varMSELoss->data.data()[0] << endl;
    
    Tensor gradMSE(varMSELoss->data.size(), CPU);
    gradMSE.fill(1.0f);
    varMSELoss->backward(gradMSE);
    
    cout << "Backward (gradients for prediction): ";
    for (size_t i = 0; i < varPrediction->data.size(); i++){
        cout << varPrediction->grad.data()[i] << " ";
    }
    cout << endl;
    
    // Clean up allocated memory.
    delete varAdd1;
    delete varAdd2;
    delete varAddResult;
    delete varSub1;
    delete varSub2;
    delete varSubResult;
    delete varMul1;
    delete varMul2;
    delete varMulResult;
    delete varSumInput;
    delete varSumOutput;
    delete varMatA;
    delete varMatB;
    delete varMatMulResult;
    delete varPrediction;
    delete varMSELoss;
}

#endif // AUTOGRAD_TESTS_HPP