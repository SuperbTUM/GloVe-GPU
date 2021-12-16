import pickle
import time
from math import ceil

import numpy as np
import pycuda.autoinit
import pycuda.cumath as cumath
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from skcuda import linalg

kernels = SourceModule(
    """
    #include <stdio.h>
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
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i < size)  matrix1[i] += matrix2[i];
    }

    __global__ void matrix_pow(float* matrix, int size){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if(i < size)  matrix[i] *= matrix[i];
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
    
        for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
    
             if (k * TILE_DIM + tx < ACols && Row < ARows)
                 As[ty][tx] = A[Row*ACols + k * TILE_DIM + tx];
             else
                 As[ty][tx] = 0.0;
    
             if (k * TILE_DIM + ty < BRows && Col < BCols)
                 Bs[ty][tx] = B[(k * TILE_DIM + ty) * BCols + Col];
             else
                 Bs[ty][tx] = 0.0;
    
             __syncthreads();
    
             for (int n = 0; n < TILE_DIM; ++n)
                 CValue += As[ty][n] * Bs[n][tx];
    
             __syncthreads();
        }
    
        if (Row < CRows && Col < CCols)
            C[((by * blockDim.y + ty) * CCols) +
               (bx * blockDim.x) + tx] = CValue;
    }
    """
)

atomic_add = kernels.get_function("co_occurrence")
matrix_multi = kernels.get_function("matrix_multi")
matrix_add = kernels.get_function("matrix_add")
matrix_pow = kernels.get_function("matrix_pow")
stream1 = cuda.Stream(flags=1)
stream2 = cuda.Stream(flags=1)
stream3 = cuda.Stream(flags=1)
stream4 = cuda.Stream(flags=1)
streams = (stream1, stream2, stream3, stream4)
V = 251
linalg.init()
row_vector_indicator = gpuarray.GPUArray(shape=(1, V), dtype=np.float32).fill(1, stream=streams[0])
column_vector_indicator = gpuarray.GPUArray(shape=(V, 1), dtype=np.float32).fill(1, stream=streams[1])


def matrixMulti(*args):
    C_list = list()
    for i, (mat1, mat2) in enumerate(args):
        ARow, ACol = mat1.shape
        BRow, BCol = mat2.shape
        CRow, CCol = ARow, BCol
        mat3 = gpuarray.zeros(shape=(CRow, CCol), dtype=np.float32)
        matrix_multi(mat1, mat2, mat3, np.int32(ARow), np.int32(ACol), np.int32(BRow),
                     np.int32(BCol), np.int32(CRow), np.int32(CCol),
                     block=(32, 32, 1), grid=(ceil(max(CRow, CCol) / 32), ceil(max(CRow, CCol) / 32), 1),
                     stream=streams[i])
        C_list.append(mat3)
    return C_list


def load_dataset():
    data_location = 'data.pk'
    data = pickle.load(open(data_location, 'rb'))
    return data


def log_cooccurence(word_data, V):
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
    V = co_occurence.shape[0]
    the_loss_components = matrixMulti((W, linalg.transpose(W_tilde)), (b, row_vector_indicator),
                                      (column_vector_indicator, linalg.transpose(b_tilde)))
    for i in range(1, len(the_loss_components)):
        matrix_add(the_loss_components[0], the_loss_components[i], np.int32(V * V),
                   block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    matrix_add(the_loss_components[0], -co_occurence, np.int32(V * V),
               block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    the_loss = the_loss_components[0]
    grad_W, grad_W_tilde, grad_b, grad_b_tilde = matrixMulti((the_loss, W_tilde), (linalg.transpose(W), the_loss),
                                                             (row_vector_indicator, the_loss),
                                                             (row_vector_indicator, the_loss))
    grad_W = 2 * grad_W
    grad_W_tilde = 2 * linalg.transpose(grad_W_tilde)
    grad_b = 2 * linalg.transpose(grad_b)
    grad_b_tilde = 2 * linalg.transpose(grad_b_tilde)
    return grad_W, grad_W_tilde, grad_b, grad_b_tilde


def loss(W, W_tilde, b, b_tilde, co_occurence):
    V = co_occurence.shape[0]
    the_loss_components = matrixMulti((W, linalg.transpose(W_tilde)), (b, row_vector_indicator),
                                      (column_vector_indicator, linalg.transpose(b_tilde)))
    for i in range(1, len(the_loss_components)):
        matrix_add(the_loss_components[0], the_loss_components[i], np.int32(V * V), block=(1024, 1, 1),
                   grid=(ceil(V * V / 1024), 1, 1))
    matrix_add(the_loss_components[0], -co_occurence, np.int32(V * V),
               block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    the_loss = the_loss_components[0]
    matrix_pow(the_loss, np.int32(V * V), block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    loss_sum = gpuarray.sum(the_loss, dtype=np.float32)
    return loss_sum


def lr_scheduler(lr, epoch, drop=0.5, epoch_drop=5):
    return lr * drop ** (epoch // epoch_drop)


def train(W, W_tilde, b, b_tilde, V, d, data):
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
    V = len(data['vocab'])
    d = 10
    W, W_tilde, b, b_tilde = init(V, d)
    return train(W, W_tilde, b, b_tilde, V, d, data)


if __name__ == '__main__':
    data = load_dataset()
    start = time.time()
    final_W = main(data)
    end = time.time()
    print("GPU execution time is {:.2f} seconds.".format(end - start))  # 1.5 seconds.
