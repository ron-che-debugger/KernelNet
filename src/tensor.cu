#include "tensor.hpp"
#include <cassert>
#include <cuda_runtime.h>

using namespace std;

Tensor::Tensor() : _size(0), _data_host(nullptr), _data_device(nullptr), _device(CPU) {}

Tensor::Tensor(size_t size, Device device) : _size(size), _device(device) {
    alloc_host();
    if (_device == CUDA) {
        alloc_device();
    }
}

Tensor::~Tensor() {
    free();
}

void Tensor::alloc_host() {
    _data_host = new float[_size]();
}

void Tensor::alloc_device() {
    cudaMalloc(&_data_device, _size * sizeof(float));
    cudaMemcpy(_data_device, _data_host, _size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::fill(float val) {
    for (size_t i = 0; i < _size; ++i) {
        _data_host[i] = val;
    }

    if (_device == CUDA) {
        cudaMemcpy(_data_device, _data_host, _size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Tensor::print() const {
    if (_device == CUDA) {
        float *tmp = new float[_size];
        cudaMemcpy(tmp, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < _size; ++i) {
            cout << tmp[i] << " ";
        }
        cout << "\n";
        delete[] tmp;
    } else {
        for (size_t i = 0; i < _size; ++i) {
            cout << _data_host[i] << " ";
        }
        cout << "\n";
    }
}

void Tensor::toCUDA() {
    if (_device == CUDA)
        return;
    alloc_device();
    _device = CUDA;
}

void Tensor::toCPU() {
    if (_device == CPU)
        return;
    cudaMemcpy(_data_host, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(_data_device);
    _device = CPU;
}

float *Tensor::data() {
    return (_device == CUDA) ? _data_device : _data_host;
}

const float *Tensor::data() const {
    return (_device == CUDA) ? _data_device : _data_host;
}

size_t Tensor::size() const {
    return _size;
}

Device Tensor::device() const {
    return _device;
}

void Tensor::free() {
    if (_data_host) {
        delete[] _data_host;
    }
    if (_device == CUDA && _data_device) {
        cudaFree(_data_device);
    }
}

__global__ void add_kernel(const float *a, const float *b, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    assert(a._size == b._size);
    assert(a._device == b._device);
    Tensor out(a._size, a._device);

    if (a._device == CPU) {
        for (size_t i = 0; i < a._size; ++i) {
            out._data_host[i] = a._data_host[i] + b._data_host[i];
        }
    } else {
        dim3 blockSize(16);
        dim3 gridSize((a._size + blockSize.x - 1) / blockSize.x);
        add_kernel<<<gridSize, blockSize>>>(a._data_device, b._data_device, out._data_device, a._size);
        cudaDeviceSynchronize();
    }

    return out;
}

__global__ void sub_kernel(const float *a, const float *b, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

Tensor Tensor::subtract(const Tensor &a, const Tensor &b) {
    assert(a._size == b._size);
    assert(a._device == b._device);
    Tensor out(a._size, a._device);

    if (a._device == CPU) {
        for (size_t i = 0; i < a._size; ++i) {
            out._data_host[i] = a._data_host[i] - b._data_host[i];
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((a._size + blockSize.x - 1) / blockSize.x);
        sub_kernel<<<gridSize, blockSize>>>(a._data_device, b._data_device, out._data_device, a._size);
        cudaDeviceSynchronize();
    }

    return out;
}

__global__ void mul_kernel(const float *a, const float *b, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

Tensor Tensor::multiply(const Tensor &a, const Tensor &b) {
    assert(a._size == b._size);
    assert(a._device == b._device);
    Tensor out(a._size, a._device);

    if (a._device == CPU) {
        for (size_t i = 0; i < a._size; ++i) {
            out._data_host[i] = a._data_host[i] * b._data_host[i];
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((a._size + blockSize.x - 1) / blockSize.x);
        mul_kernel<<<gridSize, blockSize>>>(a._data_device, b._data_device, out._data_device, a._size);
        cudaDeviceSynchronize();
    }

    return out;
}

__global__ void broadcast_add_kernel(const float *a, const float *b, float *out, size_t total_size, size_t small_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int j = idx % small_size;
        out[idx] = a[idx] + b[j];
    }
}

Tensor Tensor::broadcast_add(const Tensor &a, const Tensor &b) {
    if (a._size == b._size) {
        return Tensor::add(a, b);
    }

    if (a._size > b._size && (a._size % b._size == 0)) {
        Tensor out(a._size, a._device);
        if (a._device == CPU) {
            size_t repeat = a._size / b._size;
            for (size_t i = 0; i < repeat; i++) {
                for (size_t j = 0; j < b._size; j++) {
                    out._data_host[i * b._size + j] = a._data_host[i * b._size + j] + b._data_host[j];
                }
            }
        } else {
            dim3 blockSize(256);
            dim3 gridSize((a._size + blockSize.x - 1) / blockSize.x);
            broadcast_add_kernel<<<gridSize, blockSize>>>(a._data_device, b._data_device, out._data_device, a._size, b._size);
            cudaDeviceSynchronize();
        }
        return out;
    }

    if (b._size > a._size && (b._size % a._size == 0)) {
        return broadcast_add(b, a);
    }

    assert(false && "Incompatible sizes for broadcast addition");
    return Tensor();
}

float Tensor::sum() const {
    float total = 0.0f;

    if (_device == CPU) {
        for (size_t i = 0; i < _size; ++i) {
            total += _data_host[i];
        }
    } else {
        float *tmp = new float[_size];
        cudaMemcpy(tmp, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < _size; ++i) {
            total += tmp[i];
        }
        delete[] tmp;
    }

    return total;
}

__global__ void matrixMul_kernel(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        int tileCol = t * 16 + threadIdx.x;
        int tileRow = t * 16 + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && tileCol < K) ? A[row * K + tileCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tileRow < K && col < N) ? B[tileRow * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < 16; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

Tensor Tensor::matmul(const Tensor &a, const Tensor &b, int M, int K, int N) {
    assert(a._size == M * K);
    assert(b._size == K * N);
    Tensor out(M * N, a._device);

    if (a._device == CPU) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a._data_host[i * K + k] * b._data_host[k * N + j];
                }
                out._data_host[i * N + j] = sum;
            }
        }
    } else {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
        matrixMul_kernel<<<gridSize, blockSize>>>(a._data_device, b._data_device, out._data_device, M, K, N);
        cudaDeviceSynchronize();
    }

    return out;
}

__global__ void relu_kernel(float *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0 ? data[idx] : 0;
    }
}

void Tensor::relu() {
    if (_device == CPU) {
        for (size_t i = 0; i < _size; ++i) {
            _data_host[i] = _data_host[i] > 0 ? _data_host[i] : 0;
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((_size + blockSize.x - 1) / blockSize.x);
        relu_kernel<<<gridSize, blockSize>>>(_data_device, _size);
        cudaDeviceSynchronize();
    }
}

__global__ void transpose_kernel(const float *in, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        out[j * rows + i] = in[i * cols + j];
    }
}

Tensor Tensor::transpose(const Tensor &a, int rows, int cols) {
    assert(rows * cols == a._size);
    Tensor out(a._size, a._device);

    if (a._device == CPU) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out._data_host[j * rows + i] = a._data_host[i * cols + j];
            }
        }
    } else {
        int total = rows * cols;
        dim3 blockSize(256);
        dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
        transpose_kernel<<<gridSize, blockSize>>>(a._data_device, out._data_device, rows, cols);
        cudaDeviceSynchronize();
    }

    return out;
}

__global__ void scalar_mul_kernel(const float *in, float *out, float scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scalar;
    }
}

Tensor Tensor::scalar_multiply(const Tensor &a, float scalar) {
    Tensor out(a._size, a._device);

    if (a._device == CPU) {
        for (size_t i = 0; i < a._size; ++i) {
            out._data_host[i] = a._data_host[i] * scalar;
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize = (a._size + blockSize.x - 1) / blockSize.x;
        scalar_mul_kernel<<<gridSize, blockSize>>>(a._data_device, out._data_device, scalar, a._size);
        cudaDeviceSynchronize();
    }

    return out;
}