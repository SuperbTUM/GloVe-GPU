import pickle
import time
from math import ceil

import numpy as np
import pycuda.autoinit
import pycuda.cumath as cumath
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from skcuda import linalg, cublas

kernels = SourceModule(
    """
    #include <stdio.h>
    #include <math.h>
    #define shared_size 1024
    #define TILE_DIM 32
    __global__ void co_occurrence(const int* __restrict__ grams, float* co_occurence_matrix, int size_corpus, int num_grams, int length){
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        int bias = i/(num_grams-1);
        if(i+bias < length - 1){
            int first_pos = grams[i+bias];
            int second_pos = grams[i+bias+1];
            int add_spot = first_pos * size_corpus + second_pos;
            atomicAdd(&co_occurence_matrix[add_spot], 1);
        }

    }

    __global__ void matrix_add(float* matrix1, float* matrix2, int size){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        for(int i = index; i < size; i += stride)
            matrix1[i] += matrix2[i];
    }

    __global__ void matrix_pow(float* matrix, int size){
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int i = index; i < size; i += stride)
            matrix[i] *= matrix[i];
    }
    
    __global__ void matrix_multi(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
    {
        float CValue = 0;
        int ty = threadIdx.x;
        int tx = threadIdx.y;
        int by = blockIdx.y;
        int bx = blockIdx.x;
        int Row = by * TILE_DIM + ty;
        int Col = bx * TILE_DIM + tx;
    
        __shared__ float As[TILE_DIM][TILE_DIM];
        __shared__ float Bs[TILE_DIM][TILE_DIM];
    
        for (int k = 0; k < ((TILE_DIM + ACols - 1) >> log2(TILE_DIM)); k++) {
    
             if (k * TILE_DIM + tx < ACols && Row < ARows)
                 As[ty][tx] = A[Row*ACols + k * TILE_DIM + tx];
             else
                 As[ty][tx] = 0.0;
    
             if (k * TILE_DIM + ty < BRows && Col < BCols)
                 Bs[ty][tx] = B[(k * TILE_DIM + ty) * BCols + Col];
             else
                 Bs[ty][tx] = 0.0;
    
             __syncthreads();
            
             #pragma unroll
             for (int n = 0; n < TILE_DIM; ++n)
                 CValue += As[ty][n] * Bs[n][tx];
    
             __syncthreads();
        }
    
        if (Row < CRows && Col < CCols)
            C[((by * blockDim.y + ty) * CCols) +
               (bx * blockDim.x) + tx] = CValue;
    }
    
    __global__ void sharedABMultiply(float *A, float* B, float *C,
                                     int CCols)
    {
        __shared__ float aTile[TILE_DIM][TILE_DIM],
                         bTile[TILE_DIM][TILE_DIM];
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        aTile[threadIdx.y][threadIdx.x] = A[row*TILE_DIM+threadIdx.x];
        bTile[threadIdx.y][threadIdx.x] = B[threadIdx.y*N+col];
        __syncthreads();
        for (int i = 0; i < TILE_DIM; i++) {
            sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
        }
        C[row*CCols+col] = sum;
    }
    """
)

# define global variables
# define kernels
atomic_add = kernels.get_function("co_occurrence")
matrix_multi = kernels.get_function("matrix_multi")
matrix_add = kernels.get_function("matrix_add")
matrix_pow = kernels.get_function("matrix_pow")
# define streams
stream1 = cuda.Stream(flags=1)
stream2 = cuda.Stream(flags=1)
stream3 = cuda.Stream(flags=1)
stream4 = cuda.Stream(flags=1)
streams = (stream1, stream2, stream3, stream4)
V = 251
linalg.init()
# define indicator
row_vector_indicator = gpuarray.GPUArray(shape=(1, V), dtype=np.float32).fill(1, stream=streams[0])
column_vector_indicator = gpuarray.GPUArray(shape=(V, 1), dtype=np.float32).fill(1, stream=streams[1])
# define handles for cublas
h1 = cublas.cublasCreate()
cublas.cublasSetStream(h1, streams[0].handle)
h2 = cublas.cublasCreate()
cublas.cublasSetStream(h2, streams[1].handle)
h3 = cublas.cublasCreate()
cublas.cublasSetStream(h3, streams[2].handle)
h4 = cublas.cublasCreate()
cublas.cublasSetStream(h4, streams[3].handle)
handles = (h1, h2, h3, h4)

# deprecated
# def matrixMulti(*args):
#     C_list = list()
#     for i, (mat1, mat2) in enumerate(args):
#         ARow, ACol = mat1.shape
#         BRow, BCol = mat2.shape
#         CRow, CCol = ARow, BCol
#         mat3 = gpuarray.zeros(shape=(CRow, CCol), dtype=np.float32)
#         matrix_multi(mat1, mat2, mat3, np.int32(ARow), np.int32(ACol), np.int32(BRow),
#                      np.int32(BCol), np.int32(CRow), np.int32(CCol),
#                      block=(32, 32, 1), grid=(ceil(max(CRow, CCol) / 32), ceil(max(CRow, CCol) / 32), 1),
#                      stream=streams[i])
#         C_list.append(mat3)
#     return C_list


def matrixMulti(*args):
    """
    matrix multiplication cuda version with cublas Sgemm
    :param args: a series of matrix pairs
    :return: list of matrix multiplication results
    """
    alpha = 1
    beta = 0
    transa = 'n'
    transb = 'n'
    C_list = list()
    for i, (mat1, mat2) in enumerate(args):
        m = mat1.shape[0]
        n = mat2.shape[1]
        k = mat1.shape[1]
        C_gpu = gpuarray.zeros((m,n), dtype=np.float32)
        cublas.cublasSgemm(handles[i], transa, transb, n, m, k, alpha, mat2.gpudata, n, mat1.gpudata, k, beta, C_gpu.gpudata, n)
        C_list.append(C_gpu)
    return C_list


def matrixMultiWithSum(*args, bias=None):
    flag = True
    alpha = 1
    beta = 1
    transa = 'n'
    transb = 'n'
    for i, (mat1, mat2) in enumerate(args):
        m = mat1.shape[0]
        n = mat2.shape[1]
        k = mat1.shape[1]
        if bias:
            C_gpu = bias
            bias = None
            flag = False
        elif flag:
            C_gpu = gpuarray.zeros((m,n), dtype=np.float32)
            flag = False
        cublas.cublasSgemm(h1, transa, transb, n, m, k, alpha, mat2.gpudata, n, mat1.gpudata, k, beta, C_gpu.gpudata, n)
    return C_gpu


def load_dataset():
    data_location = 'data.pk'
    data = pickle.load(open(data_location, 'rb'))
    return data


def log_cooccurence(word_data, V):
    """
    Counting co-occurrence in corpus
    :param word_data: corpus in dataset
    :param V: number of vocabs
    :return: log co-occurrence matrix
    """
    cooccurence_matrix_gpu = gpuarray.zeros((V, V), dtype=np.float32)
    n_grams = word_data.shape[1]
    length = word_data.shape[0] * n_grams
    atomic_add(word_data, cooccurence_matrix_gpu, np.int32(V),
               np.int32(n_grams), np.int32(length),
               block=(1024, 1, 1), grid=(ceil(length / 1024), 1, 1))
    smooth = 0.5
    cooccurence_matrix_gpu += smooth
    cooccurence_matrix_gpu = cumath.log(cooccurence_matrix_gpu)
    return cooccurence_matrix_gpu


def init(V, d):
    """
    Initialization of weights and bias of each word
    :param V: number of vocabs
    :param d: number of embedded dims, default: 10
    :return: initial weight and bias matrix
    """
    base = 0.1
    W = base * np.random.normal(size=(V, d)).astype(np.float32)
    W = cuda.register_host_memory(W)
    W_gpu = gpuarray.to_gpu_async(W, stream=streams[0])
    W.base.unregister()

    W_tilde = base * np.random.normal(size=(V, d)).astype(np.float32)
    W_tilde = cuda.register_host_memory(W_tilde)
    W_tilde_gpu = gpuarray.to_gpu_async(W_tilde, stream=streams[1])
    W_tilde.base.unregister()

    b = base * np.random.normal(size=(V, 1)).astype(np.float32)
    b = cuda.register_host_memory(b)
    b_gpu = gpuarray.to_gpu_async(b, stream=streams[2])
    b.base.unregister()

    b_tilde = base * np.random.normal(size=(V, 1)).astype(np.float32)
    b_tilde = cuda.register_host_memory(b_tilde)
    b_tilde_gpu = gpuarray.to_gpu_async(b_tilde, stream=streams[3])
    b_tilde.base.unregister()

    return W_gpu, W_tilde_gpu, b_gpu, b_tilde_gpu


def grad(W, W_tilde, b, b_tilde, co_occurence):
    """
    Calculate gradient of learnable parameters in training
    :param W: weight of each word from left to right
    :param W_tilde: weight of each word from right to left, in asymmetric representation, w and w_tilde are not the same
    :param b: bias of each word from left to right
    :param b_tilde: bias of each word from right to left
    :param co_occurence: the co-occurrence matrix
    :return: the gradient of each parameter
    """
    the_loss = matrixMultiWithSum((W, linalg.transpose(W_tilde)), (b, row_vector_indicator),
                                  (column_vector_indicator, linalg.transpose(b_tilde)), bias=-co_occurence)
    grad_W, grad_W_tilde, grad_b, grad_b_tilde = matrixMulti((the_loss, W_tilde), (linalg.transpose(W), the_loss),
                                                             (row_vector_indicator, the_loss),
                                                             (row_vector_indicator, the_loss))
    grad_W = 2 * grad_W
    grad_W_tilde = 2 * linalg.transpose(grad_W_tilde)
    grad_b = 2 * linalg.transpose(grad_b)
    grad_b_tilde = 2 * linalg.transpose(grad_b_tilde)
    return grad_W, grad_W_tilde, grad_b, grad_b_tilde


def loss(W, W_tilde, b, b_tilde, co_occurence):
    """
    Calculate the loss of the model
    :param W: Same as gradient definition
    :param W_tilde: Same as gradient definition
    :param b: Same as gradient definition
    :param b_tilde: Same as gradient definition
    :param co_occurence: Same as gradient definition
    :return: mean squared loss
    """
    V = co_occurence.shape[0]
    the_loss = matrixMultiWithSum((W, linalg.transpose(W_tilde)), (b, row_vector_indicator),
                                  (column_vector_indicator, linalg.transpose(b_tilde)), bias=-co_occurence)
    matrix_pow(the_loss, np.int32(V * V), block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    loss_sum = gpuarray.sum(the_loss, dtype=np.float32)
    return loss_sum


def lr_scheduler(lr, epoch, drop=0.5, epoch_drop=5):
    return lr * drop ** (epoch // epoch_drop)


def train(W, W_tilde, b, b_tilde, V, d, data):
    """
    Train the model with batch gradient descent
    :param W: Same as gradient definition
    :param W_tilde: Same as gradient definition
    :param b: Same as gradient definition
    :param b_tilde: Same as gradient definition
    :param V: number of vocabs
    :param d: number of embedding dim
    :param data: corpus of the dataset
    :return: final weights of words
    """
    word_data = data['valid_inputs'].astype(np.int32)
    word_data_gpu = gpuarray.to_gpu(word_data)
    co_occurence_valid = log_cooccurence(word_data_gpu, V)
    learning_rate = 0.05 / V
    momentum = 0.9
    epochs = 25
    batch_size = 74500
    train_losses, valid_losses = [], []
    step_w = step_w_tilde = gpuarray.zeros((V, d), dtype=np.float32)
    step_b = step_b_tilde = gpuarray.zeros((V, 1), dtype=np.float32)
    data_inputs = data['train_inputs'].astype(np.int32)
    num_batches = data_inputs.shape[0] // batch_size
    for epoch in range(epochs):
        idxs = np.random.permutation(data_inputs.shape[0])
        data_inputs_random = data_inputs[idxs, :]
        data_inputs_random = gpuarray.to_gpu(data_inputs_random)
        co_occurence_train = log_cooccurence(data_inputs_random, V)
        learning_rate = lr_scheduler(learning_rate, epoch)
        for m in range(num_batches):
            data_inputs_batch = data_inputs_random[m * batch_size:(m + 1) * batch_size, :]
            co_occurence_train_batch = log_cooccurence(data_inputs_batch, V)
            grad_W, grad_W_tilde, grad_b, grad_b_tilde = grad(W, W_tilde, b, b_tilde, co_occurence_train_batch)

            step_w = step_w * momentum + learning_rate * grad_W
            W -= step_w
            step_w_tilde = step_w_tilde * momentum + learning_rate * grad_W_tilde
            W_tilde -= step_w_tilde
            step_b = step_b * momentum + learning_rate * grad_b
            b -= step_b
            step_b_tilde = step_b_tilde * momentum + learning_rate * grad_b_tilde
            b_tilde -= step_b_tilde

        train_loss = loss(W, W_tilde, b, b_tilde, co_occurence_train).get()
        train_losses.append(train_loss)

        valid_loss = loss(W, W_tilde, b, b_tilde, co_occurence_valid).get()
        valid_losses.append(valid_loss)
    final_W = W.get_async(stream=streams[0])
    return final_W


def main(data):
    """
    Call training of the model
    :param data: dataset for training
    :return: final weights
    """
    V = len(data['vocab'])
    d = 10
    W, W_tilde, b, b_tilde = init(V, d)
    return train(W, W_tilde, b, b_tilde, V, d, data)


if __name__ == '__main__':
    data = load_dataset()
    start = time.time()
    final_W = main(data)
    end = time.time()
    print("GPU execution time is {:.2f} seconds.".format(end - start))  # 0.69 seconds on average of 20 times.
    for h in handles:
        cublas.cublasDestroy(h)
