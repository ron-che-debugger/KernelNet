/**
 * @file tensor.hpp
 * @brief Defines a simple 1D tensor class with support for CPU and CUDA memory management.
 *
 * This class provides basic operations, memory allocation, device transfers, and
 * element-wise and matrix operations. It also supports deep copying.
 */

#pragma once

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
namespace kernelnet {
namespace tensor {
/**
 * @brief Enum representing the target device.
 */
enum Device {
    CPU, ///< Represents the host (CPU) device.
    CUDA ///< Represents the GPU (CUDA) device.
};

/**
 * @brief A simple tensor class storing a 1D array of floats.
 *
 * The Tensor class manages memory on both host and device. It provides basic operations such as
 * element-wise arithmetic, matrix multiplication, and device transfers.
 */
class Tensor {
  public:
    /**
     * @brief Default constructor.
     *
     * Initializes an empty tensor of size 0 on the CPU.
     */
    Tensor();

    /**
     * @brief Constructs a tensor with a specified number of elements on a given device.
     *
     * @param size Number of elements.
     * @param device The target device (CPU by default).
     */
    Tensor(size_t size, Device device = CPU);

    /**
     * @brief Destructor.
     *
     * Frees allocated memory on the host and, if necessary, on the device.
     */
    ~Tensor();

    /**
     * @brief Deep-copy constructor.
     *
     * Creates a new tensor by deep copying the data from an existing one.
     *
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor &other) {
        _size = 0;
        _data_host = nullptr;
        _data_device = nullptr;
        _device = CPU;
        copyFrom(other);
    }

    /**
     * @brief Deep-copy assignment operator.
     *
     * Frees the existing memory and deep copies data from another tensor.
     *
     * @param other The tensor to copy from.
     * @return Reference to the current tensor.
     */
    Tensor &operator=(const Tensor &other) {
        if (this != &other) {
            free(); // Free existing data.
            copyFrom(other);
        }
        return *this;
    }

    /**
     * @brief Fills the tensor with a constant value.
     *
     * Updates the host memory and, if on CUDA, copies the values to device memory.
     *
     * @param val Value to fill the tensor with.
     */
    void fill(float val);

    /**
     * @brief Prints the tensor contents to standard output.
     *
     * If the tensor is on CUDA, data is copied to the CPU first.
     */
    void print() const;

    /**
     * @brief Transfers tensor data to CUDA (GPU) memory.
     *
     * Allocates device memory and copies the host data.
     */
    void toCUDA();

    /**
     * @brief Transfers tensor data back to CPU memory.
     *
     * Copies device data to host and frees the device memory.
     */
    void toCPU();

    /**
     * @brief Returns a mutable pointer to the tensor's data.
     *
     * @return Pointer to the tensor data.
     */
    float *data();

    /**
     * @brief Returns a constant pointer to the tensor's data.
     *
     * @return Constant pointer to the tensor data.
     */
    const float *data() const;

    /**
     * @brief Returns the total number of elements in the tensor.
     *
     * @return The size of the tensor.
     */
    size_t size() const;

    /**
     * @brief Returns the device where the tensor is stored.
     *
     * @return The device (CPU or CUDA).
     */
    Device device() const;

    // ----------------------- Static Operations -----------------------

    /**
     * @brief Computes element-wise addition of two tensors.
     *
     * Both tensors must have the same size and be on the same device.
     *
     * @param a First input tensor.
     * @param b Second input tensor.
     * @return Tensor resulting from a + b.
     */
    static Tensor add(const Tensor &a, const Tensor &b);

    /**
     * @brief Computes element-wise subtraction of two tensors.
     *
     * Both tensors must have the same size and be on the same device.
     *
     * @param a First input tensor.
     * @param b Second input tensor.
     * @return Tensor resulting from a - b.
     */
    static Tensor subtract(const Tensor &a, const Tensor &b);

    /**
     * @brief Computes element-wise multiplication of two tensors.
     *
     * Both tensors must have the same size and device.
     *
     * @param a First input tensor.
     * @param b Second input tensor.
     * @return Tensor resulting from a * b.
     */
    static Tensor multiply(const Tensor &a, const Tensor &b);

    /**
     * @brief Computes the transpose of a tensor treated as a 2D matrix.
     *
     * The tensor is expected to have rows * cols elements.
     *
     * @param a Input tensor.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return Transposed tensor.
     */
    static Tensor transpose(const Tensor &a, int rows, int cols);

    /**
     * @brief Multiplies each element of a tensor by a scalar.
     *
     * @param a Input tensor.
     * @param scalar Scalar value.
     * @return Output tensor after scalar multiplication.
     */
    static Tensor scalar_multiply(const Tensor &a, float scalar);

    /**
     * @brief Performs broadcast addition between two tensors.
     *
     * If the tensors are of equal size, performs element-wise addition. Otherwise, if one
     * tensor's size divides the other's, the smaller tensor is broadcast accordingly.
     *
     * @param a First tensor.
     * @param b Second tensor.
     * @return Tensor resulting from broadcast addition.
     */
    static Tensor broadcast_add(const Tensor &a, const Tensor &b);

    /**
     * @brief Finds the index of the maximum element in the tensor.
     *
     * Operates on the flattened tensor.
     *
     * @return Index of the maximum element, or -1 if the tensor is empty.
     */
    int argmax() const;

    /**
     * @brief Computes the argmax along the specified axis for a 2D tensor.
     *
     * The tensor is assumed to be of shape (batch_size, dim_size) when flattened.
     *
     * @param axis The axis along which to compute argmax (only axis==1 is supported).
     * @param dim_size The size of the second dimension.
     * @return Vector containing the argmax index for each batch entry.
     */
    vector<int> argmax(int axis, int dim_size) const;

    /**
     * @brief Computes the sum of all elements in the tensor.
     *
     * @return The total sum.
     */
    float sum() const;

    /**
     * @brief Applies ReLU activation in-place.
     *
     * Each element is replaced by max(element, 0).
     */
    void relu();

    /**
     * @brief Performs matrix multiplication on two tensors.
     *
     * Computes the product of two matrices A and B:
     *   C = A x B
     * where A has shape (M, K) and B has shape (K, N). The result C has shape (M, N).
     *
     * @param a First input tensor.
     * @param b Second input tensor.
     * @param M Number of rows in matrix A.
     * @param K Shared dimension.
     * @param N Number of columns in matrix B.
     * @return Output tensor resulting from matrix multiplication.
     */
    static Tensor matmul(const Tensor &a, const Tensor &b, int M, int K, int N);

  private:
    size_t _size;        ///< Total number of elements.
    float *_data_host;   ///< Pointer to host memory.
    float *_data_device; ///< Pointer to device memory.
    Device _device;      ///< Current device (CPU or CUDA).

    /**
     * @brief Allocates host memory for the tensor.
     */
    void alloc_host();

    /**
     * @brief Allocates device memory for the tensor and copies host data.
     */
    void alloc_device();

    /**
     * @brief Frees allocated host and device memory.
     */
    void free();

    /**
     * @brief Helper method to copy data from another tensor.
     *
     * Performs a deep copy from the specified tensor.
     *
     * @param other The source tensor to copy from.
     */
    void copyFrom(const Tensor &other) {
        _size = other._size;
        _device = other._device;
        if (_size > 0) {
            alloc_host();
            if (_device == CUDA) {
                cudaMemcpy(_data_host, other._data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                memcpy(_data_host, other._data_host, _size * sizeof(float));
            }
            if (_device == CUDA) {
                alloc_device();
            }
        }
    }
};
} // namespace tensor
} // namespace kernelnet