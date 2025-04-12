#include "tensor.hpp"

using namespace std;

/**
 * @brief Default constructor for Tensor.
 *
 * Creates a zero-sized tensor on CPU with no allocated memory.
 */
Tensor::Tensor() : _size(0), _data_host(nullptr), _data_device(nullptr), _device(CPU) {}

/**
 * @brief Constructs a Tensor with the specified number of elements on a given device.
 *
 * Allocates host memory immediately and, if the device is CUDA, also allocates device memory.
 *
 * @param size Number of elements in the tensor.
 * @param device Target device (CPU or CUDA).
 */
Tensor::Tensor(size_t size, Device device) : _size(size), _device(device) {
    alloc_host();
    if (_device == CUDA) {
        alloc_device();
    }
}

/**
 * @brief Destructor for Tensor.
 *
 * Frees allocated host and device memory.
 */
Tensor::~Tensor() {
    free();
}

/**
 * @brief Allocates host memory for the tensor.
 *
 * Allocates an array of floats of size _size and initializes the memory to 0.
 */
void Tensor::alloc_host() {
    _data_host = new float[_size]();
}

/**
 * @brief Allocates device memory for the tensor and copies host data to device.
 */
void Tensor::alloc_device() {
    cudaMalloc(&_data_device, _size * sizeof(float));
    cudaMemcpy(_data_device, _data_host, _size * sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * @brief Fills the tensor with a constant value.
 *
 * Updates the host data and, if the tensor resides on CUDA, copies the data
 * to the device.
 *
 * @param val The value with which to fill the tensor.
 */
void Tensor::fill(float val) {
    for (size_t i = 0; i < _size; ++i) {
        _data_host[i] = val;
    }
    if (_device == CUDA) {
        cudaMemcpy(_data_device, _data_host, _size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

/**
 * @brief Prints the tensor values to standard output.
 *
 * If the tensor is on CUDA, its data is first copied to host memory.
 */
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

/**
 * @brief Transfers the tensor to CUDA (GPU) memory.
 *
 * If the tensor is not already on CUDA, allocates device memory and copies
 * the host data to the device.
 */
void Tensor::toCUDA() {
    if (_device == CUDA)
        return;
    alloc_device();
    _device = CUDA;
}

/**
 * @brief Transfers the tensor to CPU memory.
 *
 * Copies device data back to host memory and frees the device memory.
 */
void Tensor::toCPU() {
    if (_device == CPU)
        return;
    cudaMemcpy(_data_host, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(_data_device);
    _device = CPU;
}

/**
 * @brief Returns a mutable pointer to the tensor data.
 *
 * If the tensor resides on CUDA, returns the device pointer.
 *
 * @return Pointer to the tensor's data.
 */
float *Tensor::data() {
    return (_device == CUDA) ? _data_device : _data_host;
}

/**
 * @brief Returns a constant pointer to the tensor data.
 *
 * @return Constant pointer to the tensor's data.
 */
const float *Tensor::data() const {
    return (_device == CUDA) ? _data_device : _data_host;
}

/**
 * @brief Returns the number of elements in the tensor.
 *
 * @return The tensor size.
 */
size_t Tensor::size() const {
    return _size;
}

/**
 * @brief Returns the device on which the tensor is stored.
 *
 * @return The device (CPU or CUDA).
 */
Device Tensor::device() const {
    return _device;
}

/**
 * @brief Frees the memory allocated for the tensor.
 *
 * Deletes host memory and frees device memory if allocated.
 */
void Tensor::free() {
    if (_data_host) {
        delete[] _data_host;
    }
    if (_device == CUDA && _data_device) {
        cudaFree(_data_device);
    }
}

/* --------------------- Element-wise and Matrix Operations --------------------- */

/**
 * @brief CUDA kernel for element-wise addition.
 *
 * Computes out[i] = a[i] + b[i] for each element.
 *
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param out Pointer to the output array.
 * @param size Total number of elements.
 */
__global__ void add_kernel(const float *a, const float *b, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Performs element-wise addition between two tensors.
 *
 * Ensures that both tensors have the same size and reside on the same device.
 *
 * @param a First input tensor.
 * @param b Second input tensor.
 * @return Output tensor containing element-wise sums.
 */
Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    assert(a._size == b._size);
    assert(a._device == b._device);
    Tensor out(a._size, a._device);

    if (a._device == CPU) {
        for (size_t i = 0; i < a._size; ++i) {
            out._data_host[i] = a._data_host[i] + b._data_host[i];
        }
    } else {
        dim3 blockSize(256);
        dim3 gridSize((a._size + blockSize.x - 1) / blockSize.x);
        add_kernel<<<gridSize, blockSize>>>(a._data_device, b._data_device, out._data_device, a._size);
        cudaDeviceSynchronize();
    }
    return out;
}

/**
 * @brief CUDA kernel for element-wise subtraction.
 *
 * Computes out[i] = a[i] - b[i] for each element.
 *
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param out Pointer to the output array.
 * @param size Total number of elements.
 */
__global__ void sub_kernel(const float *a, const float *b, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

/**
 * @brief Performs element-wise subtraction between two tensors.
 *
 * Assumes both tensors have the same size and reside on the same device.
 *
 * @param a First input tensor.
 * @param b Second input tensor.
 * @return Output tensor containing element-wise differences.
 */
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

/**
 * @brief CUDA kernel for element-wise multiplication.
 *
 * Computes out[i] = a[i] * b[i] for every element.
 *
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param out Pointer to the output array.
 * @param size Total number of elements.
 */
__global__ void mul_kernel(const float *a, const float *b, float *out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief Performs element-wise multiplication between two tensors.
 *
 * Assumes both tensors have the same size and device.
 *
 * @param a First input tensor.
 * @param b Second input tensor.
 * @return Output tensor containing element-wise products.
 */
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

/**
 * @brief CUDA kernel for broadcast addition between two tensors.
 *
 * The kernel adds a smaller tensor (of size small_size) to a larger tensor (of size total_size)
 * by replicating the smaller tensor along the corresponding dimension.
 *
 * @param a Pointer to the larger tensor data.
 * @param b Pointer to the smaller tensor data.
 * @param out Pointer to the output array.
 * @param total_size Total number of elements in the larger tensor.
 * @param small_size Size of the smaller tensor.
 */
__global__ void broadcast_add_kernel(const float *a, const float *b, float *out, size_t total_size, size_t small_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int j = idx % small_size;
        out[idx] = a[idx] + b[j];
    }
}

/**
 * @brief Performs broadcast addition between two tensors.
 *
 * If the tensors are of equal size, performs regular addition.
 * If one tensor's size divides the other's, the smaller tensor is broadcast along that dimension.
 *
 * @param a First tensor.
 * @param b Second tensor.
 * @return Resultant tensor after broadcast addition.
 */
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

/**
 * @brief Finds the index of the maximum element in the tensor.
 *
 * Operates on the flattened tensor.
 *
 * @return The index of the maximum element, or -1 if the tensor is empty.
 */
int Tensor::argmax() const {
    if (_size == 0)
        return -1;
    if (_device == CPU) {
        int max_index = 0;
        float max_val = _data_host[0];
        for (size_t i = 1; i < _size; i++) {
            if (_data_host[i] > max_val) {
                max_val = _data_host[i];
                max_index = i;
            }
        }
        return max_index;
    } else {
        float *temp = new float[_size];
        cudaMemcpy(temp, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
        int max_index = 0;
        float max_val = temp[0];
        for (size_t i = 1; i < _size; i++) {
            if (temp[i] > max_val) {
                max_val = temp[i];
                max_index = i;
            }
        }
        delete[] temp;
        return max_index;
    }
}

/**
 * @brief Computes the argmax along axis 1 for a 2D tensor.
 *
 * Assumes the tensor is a flattened 2D tensor with shape (batch_size, dim_size).
 *
 * @param axis The axis along which to compute argmax (only axis 1 is supported).
 * @param dim_size The size of the second dimension.
 * @return A vector containing the argmax index for each batch entry.
 */
vector<int> Tensor::argmax(int axis, int dim_size) const {
    assert(axis == 1 && "Currently only axis==1 is supported in argmax");
    int batch_size = _size / dim_size;
    vector<int> result(batch_size, 0);
    float *data_ptr = new float[_size];
    if (_device == CPU) {
        memcpy(data_ptr, _data_host, _size * sizeof(float));
    } else {
        cudaMemcpy(data_ptr, _data_device, _size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < batch_size; i++) {
        int start = i * dim_size;
        int best_index = 0;
        float best_value = data_ptr[start];
        for (int j = 1; j < dim_size; j++) {
            float cur_val = data_ptr[start + j];
            if (cur_val > best_value) {
                best_value = cur_val;
                best_index = j;
            }
        }
        result[i] = best_index;
    }
    delete[] data_ptr;
    return result;
}

/**
 * @brief Computes the sum of all elements in the tensor.
 *
 * @return The sum of tensor elements.
 */
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

/**
 * @brief CUDA kernel for matrix multiplication.
 *
 * Uses shared memory tiling for efficient matrix multiplication.
 *
 * @param A Pointer to the first input matrix (size: M x K).
 * @param B Pointer to the second input matrix (size: K x N).
 * @param C Pointer to the output matrix (size: M x N).
 * @param M Number of rows in matrix A.
 * @param K Shared dimension.
 * @param N Number of columns in matrix B.
 */
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

/**
 * @brief Performs matrix multiplication on two tensors.
 *
 * Computes the product C = A x B where:
 *   - A is of shape (M, K)
 *   - B is of shape (K, N)
 *   - C is of shape (M, N)
 *
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param M Number of rows in matrix A.
 * @param K Shared dimension.
 * @param N Number of columns in matrix B.
 * @return Output tensor containing the matrix multiplication result.
 */
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

/**
 * @brief CUDA kernel for element-wise ReLU activation.
 *
 * Applies ReLU activation on each element: data[idx] = max(data[idx], 0).
 *
 * @param data Pointer to the data array.
 * @param size Total number of elements.
 */
__global__ void relu_kernel(float *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0 ? data[idx] : 0;
    }
}

/**
 * @brief Applies ReLU activation in-place on the tensor.
 *
 * If the tensor is on CUDA, launches a CUDA kernel; otherwise processes on CPU.
 */
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

/**
 * @brief CUDA kernel for transposing a matrix.
 *
 * Given an input matrix in row-major order, writes its transpose to the output array.
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param rows Number of rows of the input matrix.
 * @param cols Number of columns of the input matrix.
 */
__global__ void transpose_kernel(const float *in, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        out[j * rows + i] = in[i * cols + j];
    }
}

/**
 * @brief Transposes the given tensor, treating it as a 2D matrix.
 *
 * The input tensor must have a total of rows * cols elements.
 *
 * @param a Input tensor.
 * @param rows Number of rows in the input matrix.
 * @param cols Number of columns in the input matrix.
 * @return Transposed tensor.
 */
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

/**
 * @brief CUDA kernel for scalar multiplication.
 *
 * Multiplies each element in the input array by a scalar.
 *
 * @param in Pointer to the input array.
 * @param out Pointer to the output array.
 * @param scalar Scalar value to multiply with.
 * @param size Total number of elements.
 */
__global__ void scalar_mul_kernel(const float *in, float *out, float scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scalar;
    }
}

/**
 * @brief Multiplies each element of the tensor by a scalar.
 *
 * @param a Input tensor.
 * @param scalar Scalar value to multiply.
 * @return Output tensor after scalar multiplication.
 */
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