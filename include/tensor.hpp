#pragma once
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

enum Device { CPU,
              CUDA };

class Tensor {
  public:
    Tensor();
    Tensor(size_t size, Device device = CPU);
    ~Tensor();

    // Deep-copy constructor
    Tensor(const Tensor &other) {
        _size = 0;
        _data_host = nullptr;
        _data_device = nullptr;
        _device = CPU;
        copyFrom(other);
    }

    // Deep-copy assignment operator
    Tensor &operator=(const Tensor &other) {
        if (this != &other) {
            free(); // free our existing data
            copyFrom(other);
        }
        return *this;
    }

    void fill(float val);
    void print() const;

    void toCUDA();
    void toCPU();

    float *data();
    const float *data() const;
    size_t size() const;
    Device device() const;

    static Tensor add(const Tensor &a, const Tensor &b);
    static Tensor subtract(const Tensor &a, const Tensor &b);
    static Tensor multiply(const Tensor &a, const Tensor &b);
    static Tensor transpose(const Tensor &a, int rows, int cols);
    static Tensor scalar_multiply(const Tensor &a, float scalar);
    static Tensor broadcast_add(const Tensor &a, const Tensor &b);

    int argmax() const;
    float sum() const;
    void relu();
    static Tensor matmul(const Tensor &a, const Tensor &b, int M, int K, int N);

  private:
    size_t _size;
    float *_data_host;
    float *_data_device;
    Device _device;

    void alloc_host();
    void alloc_device();
    void free();

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