#pragma once
#include <iostream>
#include <vector>

enum Device {CPU, CUDA};

class Tensor{
public:
    Tensor(size_t size, Device device = CPU);
    ~Tensor();

    void fill(float val);
    void print() const;

    void toCUDA();
    void toCPU();

    float* data();
    const float* data() const;
    size_t size() const;
    Device device() const;

    static Tensor add(const Tensor& a, const Tensor& b);

private:
    size_t _size;
    float* _data_host;
    float* _data_device;
    Device _device;

    void alloc_host();
    void alloc_device();
    void free();
};