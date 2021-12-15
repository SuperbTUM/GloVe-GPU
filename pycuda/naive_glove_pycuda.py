import pycuda.autoinit
from skcuda import cublas, linalg
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from math import ceil
import numpy as np
import time
import pickle

kernels = SourceModule(
    """
    #include <stdio.h>
    #define shared_size 1024
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
    """
)


atomic_add = kernels.get_function("co_occurrence")
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
    transa = 'n'
    transb = 'n'
    alpha = 1
    beta = 0
    handles = list()
    C_list = list()
    for mat1, mat2 in args:
        #  create a stream
        # print("create a stream: cur_stream = cuda.Stream()")
        cur_stream = cuda.Stream(flags=1)
        # print("create a handle: handles.append(cublas.cublasCreate())")
        handles.append(cublas.cublasCreate())
        # print("use handles[-1] to call: cublas.cublasSetStream(handles[-1], cur_stream)")
        h = handles[-1]
        cublas.cublasSetStream(h, cur_stream.handle)
        # cublas.cublasSetStream(cublas_handle, stream.handle)
        # print("cublas call in this stream", args[0], args[1])
        m = mat1.shape[0]
        n = mat2.shape[1]
        k = mat1.shape[1]
        B_gpu = mat1
        A_gpu = mat2
        C_gpu = gpuarray.zeros(shape=(m, n), dtype=np.float32)
        cublas.cublasSgemm(h, transa, transb, n, m, k, alpha, A_gpu.gpudata, n, B_gpu.gpudata, k, beta, C_gpu.gpudata,
                           n)
        # cur_stream.synchronize()
        C_list.append(C_gpu)
    for j in range(len(handles)):
        handle = handles[j]
        # print("destroy handle: cublas.cublasDestroy(handle)")
        cublas.cublasDestroy(handle)
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
    the_loss_components = matrixMulti((W, linalg.transpose(W_tilde)), (b, row_vector_indicator), (column_vector_indicator, linalg.transpose(b_tilde)))
    for i in range(1, len(the_loss_components)):
        matrix_add(the_loss_components[0], the_loss_components[i], np.int32(V * V),
                   block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    matrix_add(the_loss_components[0], -co_occurence, np.int32(V * V),
                   block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
    the_loss = the_loss_components[0]
    grad_W = 2 * matrixMulti((the_loss, W_tilde))[0]
    grad_W_tilde = 2 * linalg.transpose(matrixMulti((linalg.transpose(W), the_loss))[0])
    grad_b = 2 * linalg.transpose(matrixMulti((row_vector_indicator, the_loss))[0])
    grad_b_tilde = 2 * linalg.transpose(matrixMulti((row_vector_indicator, the_loss))[0])
    return grad_W, grad_W_tilde, grad_b, grad_b_tilde


def loss(W, W_tilde, b, b_tilde, co_occurence):
    V = co_occurence.shape[0]
    the_loss_components = matrixMulti((W, linalg.transpose(W_tilde)), (b, row_vector_indicator), (column_vector_indicator, linalg.transpose(b_tilde)))
    for i in range(1, len(the_loss_components)):
        matrix_add(the_loss_components[0], the_loss_components[i], np.int32(V * V), block=(1024, 1, 1), grid=(ceil(V * V / 1024), 1, 1))
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
    print("GPU execution time is {:.2f} seconds.".format(end-start))  # 1.5 seconds.

