#include "tensor.hpp"
#include <cuda_runtime.h>
#include <cassert>

using namespace std;

Tensor::Tensor(size_t size, Device device) : _size(size), _device(device){
    alloc_host();

    if (_device == CUDA){
        alloc_device();
    }
}

Tensor::~Tensor(){
    free();
}

void Tensor::alloc_host(){
    _data_host = new float[_size]();
}

void Tensor::alloc_device(){
    cudaMalloc(&_data_device, _size * sizeof(float));
    cudaMemcpy(_data_device, _data_host, _size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::fill(float val){
    for (size_t i = 0; i < _size; ++i){
        _data_host[i] = val;
    }

    if (_device == CUDA){
        cudaMemcpy(_data_device, _data_host, _size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Tensor::print() const {
    if (_device == CUDA){
        float* tmp = new float[_size];
        cudaMemcpy(tmp, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < _size; ++i){
            cout << tmp[i] << " ";
        }
        cout << "\n";
        delete[] tmp;
    }
    else {
        for (size_t i = 0; i < _size; ++i){
            cout << _data_host[i] << " ";
        }
        cout << "\n";
    }
}

void Tensor::toCUDA() {
    if (_device == CUDA) return;

    alloc_device();

    _device = CUDA;
}

void Tensor::toCPU() {
    if (_device == CPU) return;

    cudaMemcpy(_data_host, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(_data_device);

    _device = CPU;
}

float* Tensor::data() {
    return (_device == CUDA) ? _data_device : _data_host;
}

const float* Tensor::data() const{
    return (_device == CUDA) ? _data_device : _data_host;
}

size_t Tensor::size() const {
    return _size;
}

void Tensor::free(){
    if (_data_host){
        delete[] _data_host;
    }

    if (_device == CUDA && _data_device) {
        cudaFree(_data_device);
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size){
        out[idx] = a[idx] + b[idx];
    }
}

Tensor Tensor::add(const Tensor& a, const Tensor& b){
    assert(a._size == b._size);
    assert(a._device == b._device);
    Tensor out(a._size, a._device);

    if (a._device == CPU){
        for (size_t i = 0; i < a._size; ++i){
            out._data_host[i] = a._data_host[i] + b._data_host[i];
        }
    }
    else {
        dim3 blockSize(16);
        dim3 gridSize((a._size + blockSize.x - 1) / blockSize.x);
        add_kernel<<<blockSize, gridSize>>>(a._data_device, b._data_device, out._data_device, a._size);
        cudaDeviceSynchronize();
    }

    return out;
}