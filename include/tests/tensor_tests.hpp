#ifndef TENSOR_TESTS_HPP
#define TENSOR_TESTS_HPP

#include "tensor.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

inline void runTensorTests() {
    // ========= CPU Tests =========
    cout << "===== Testing on CPU =====" << endl;

    // Create two 1D tensors of size 10.
    Tensor a(10, CPU);
    Tensor b(10, CPU);
    a.fill(2.0f);
    b.fill(3.0f);

    // Test Broadcast Add on CPU
    // Create tensor A of size 6 and tensor B of size 3.
    Tensor A_broadcast(6, CPU);
    Tensor B_broadcast(3, CPU);
    // Fill A_broadcast with values [1, 2, 3, 4, 5, 6]
    for (size_t i = 0; i < A_broadcast.size(); ++i) {
        A_broadcast.data()[i] = i + 1;
    }
    // Fill B_broadcast with values [10, 20, 30]
    B_broadcast.data()[0] = 10;
    B_broadcast.data()[1] = 20;
    B_broadcast.data()[2] = 30;
    // Perform broadcast add: each group of 3 elements in A gets B added.
    auto broadcast_cpu = Tensor::broadcast_add(A_broadcast, B_broadcast);
    cout << "Broadcast Add (CPU) result:" << endl;
    broadcast_cpu.print();
    // Expected output: 11 22 33 14 25 36

    // Test Addition
    auto add_cpu = Tensor::add(a, b);
    cout << "Addition (2+3):" << endl;
    add_cpu.print(); // Expect all elements to be 5.0

    // Test Subtraction (b - a, expect 1.0 each)
    auto sub_cpu = Tensor::subtract(b, a);
    cout << "Subtraction (3-2):" << endl;
    sub_cpu.print();

    // Test Multiplication (2 * 3 = 6)
    auto mul_cpu = Tensor::multiply(a, b);
    cout << "Multiplication (2*3):" << endl;
    mul_cpu.print();

    // Test Sum (for the addition result, 10 * 5 = 50)
    float sum_cpu = add_cpu.sum();
    cout << "Sum of addition result: " << sum_cpu << endl;

    // Test ReLU
    // Create a tensor with alternating positive and negative values.
    Tensor relu_cpu(10, CPU);
    for (size_t i = 0; i < relu_cpu.size(); ++i) {
        // Use even indices as positive, odd as negative.
        relu_cpu.data()[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }
    cout << "Before ReLU:" << endl;
    relu_cpu.print();
    relu_cpu.relu();
    cout << "After ReLU:" << endl;
    relu_cpu.print();

    // Test Matrix Multiplication (CPU)
    // Multiply a 2x3 matrix (A) by a 3x2 matrix (B) to get a 2x2 result.
    int M = 2, K = 3, N = 2;
    Tensor A_mat(M * K, CPU);
    Tensor B_mat(K * N, CPU);
    // Fill A_mat with values 1,2,...,6
    for (int i = 0; i < M * K; i++) {
        A_mat.data()[i] = i + 1;
    }
    // Fill B_mat with values 1,2,...,6
    for (int i = 0; i < K * N; i++) {
        B_mat.data()[i] = i + 1;
    }
    auto matmul_cpu = Tensor::matmul(A_mat, B_mat, M, K, N);
    cout << "Matrix Multiplication (CPU) result:" << endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matmul_cpu.data()[i * N + j] << " ";
        }
        cout << endl;
    }

    // ========= CUDA Tests =========
    cout << "\n===== Testing on CUDA =====" << endl;

    // Create two 1D tensors on CPU then move them to CUDA.
    Tensor a_cuda(10, CPU);
    Tensor b_cuda(10, CPU);
    a_cuda.fill(2.0f);
    b_cuda.fill(3.0f);
    a_cuda.toCUDA();
    b_cuda.toCUDA();

    // Test Broadcast Add on CUDA
    // Create tensor A of size 6 and tensor B of size 3 (initialize on CPU, then move to CUDA).
    Tensor A_broadcast_cuda(6, CPU);
    Tensor B_broadcast_cuda(3, CPU);
    for (size_t i = 0; i < A_broadcast_cuda.size(); ++i) {
        A_broadcast_cuda.data()[i] = i + 1;
    }
    B_broadcast_cuda.data()[0] = 10;
    B_broadcast_cuda.data()[1] = 20;
    B_broadcast_cuda.data()[2] = 30;
    A_broadcast_cuda.toCUDA();
    B_broadcast_cuda.toCUDA();
    // Perform broadcast add on CUDA.
    auto broadcast_cuda = Tensor::broadcast_add(A_broadcast_cuda, B_broadcast_cuda);
    cout << "Broadcast Add (CUDA) result:" << endl;
    broadcast_cuda.print();
    // Expected output (after transferring to host internally in print): 11 22 33 14 25 36

    // Test Addition on CUDA
    auto add_cuda = Tensor::add(a_cuda, b_cuda);
    cout << "Addition (CUDA):" << endl;
    add_cuda.print();

    // Test Subtraction on CUDA (b - a)
    auto sub_cuda = Tensor::subtract(b_cuda, a_cuda);
    cout << "Subtraction (CUDA):" << endl;
    sub_cuda.print();

    // Test Multiplication on CUDA (2 * 3 = 6)
    auto mul_cuda = Tensor::multiply(a_cuda, b_cuda);
    cout << "Multiplication (CUDA):" << endl;
    mul_cuda.print();

    // Test Sum on CUDA
    float sum_cuda = add_cuda.sum();
    cout << "Sum of addition result (CUDA): " << sum_cuda << endl;

    // Test ReLU on CUDA
    Tensor relu_cuda(10, CPU);
    for (size_t i = 0; i < relu_cuda.size(); ++i) {
        relu_cuda.data()[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }
    relu_cuda.toCUDA();
    cout << "Before ReLU (CUDA):" << endl;
    relu_cuda.print();
    relu_cuda.relu();
    cout << "After ReLU (CUDA):" << endl;
    relu_cuda.print();

    // Test Matrix Multiplication on CUDA
    Tensor A_mat_cuda(M * K, CPU);
    Tensor B_mat_cuda(K * N, CPU);
    for (int i = 0; i < M * K; i++) {
        A_mat_cuda.data()[i] = i + 1;
    }
    for (int i = 0; i < K * N; i++) {
        B_mat_cuda.data()[i] = i + 1;
    }
    A_mat_cuda.toCUDA();
    B_mat_cuda.toCUDA();
    auto matmul_cuda = Tensor::matmul(A_mat_cuda, B_mat_cuda, M, K, N);
    cout << "Matrix Multiplication (CUDA) result:" << endl;
    // Transfer the result back to CPU for printing.
    matmul_cuda.toCPU();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matmul_cuda.data()[i * N + j] << " ";
        }
        cout << endl;
    }
}

#endif // TENSOR_TESTS_HPP