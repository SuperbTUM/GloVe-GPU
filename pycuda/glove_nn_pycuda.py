import pycuda.autoinit
from pycuda.gpuarray import *
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np
from math import ceil
from skcuda import linalg, cublas
import time
import pickle

kernels = SourceModule(
    """
__device__ void warpReduce(volatile float* vector, int tid){
    // avoid syncthread
    if(blockDim.x >= 64) vector[tid] += vector[tid+32];
    if(blockDim.x >= 32) vector[tid] += vector[tid+16];
    if(blockDim.x >= 16) vector[tid] += vector[tid+8];
    if(blockDim.x >= 8) vector[tid] += vector[tid+4];
    if(blockDim.x >= 4) vector[tid] += vector[tid+2];
    if(blockDim.x >= 2) vector[tid] += vector[tid+1];
} 

__global__ void matrix_reduction(float* matrix, float* reduction_matrix){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = tid + bid * blockDim.x * 2;
    __shared__ float vector[128];
    vector[tid] = matrix[i] + matrix[i+blockDim.x];
    __syncthreads();
    if(blockDim.x >= 1024){
        if(tid < 512)  vector[tid] += vector[tid + 512];
        __syncthreads();
    }
    if(blockDim.x >= 512){
        if(tid < 256)  vector[tid] += vector[tid + 256];
        __syncthreads(); 
    }
    if(blockDim.x >= 256){
        if(tid < 128)  vector[tid] += vector[tid + 128];
        __syncthreads();
    }
    if(blockDim.x >= 128){
        if(tid < 64)  vector[tid] += vector[tid + 64];
        __syncthreads();
    }
    if(tid < 32)  warpReduce(vector, tid);
    if(tid == 0)  reduction_matrix[bid] = vector[0];
}

__device__ void warpMax(volatile float* vector, int tid){
    if(blockDim.x >= 64) vector[tid] = (vector[tid] > vector[tid+32])? vector[tid]: vector[tid+32];
    if(blockDim.x >= 32) vector[tid] = (vector[tid] > vector[tid+16])? vector[tid]: vector[tid+16];
    if(blockDim.x >= 16) vector[tid] = (vector[tid] > vector[tid+8])? vector[tid]: vector[tid+8];
    if(blockDim.x >= 8) vector[tid] = (vector[tid] > vector[tid+4])? vector[tid]: vector[tid+4];
    if(blockDim.x >= 4) vector[tid] = (vector[tid] > vector[tid+2])? vector[tid]: vector[tid+2];
    if(blockDim.x >= 2) vector[tid] = (vector[tid] > vector[tid+1])? vector[tid]: vector[tid+1];
} 

__global__ void find_max(float* matrix, float* outputs){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = tid + bid * blockDim.x * 2;
    __shared__ float vector[512];
    vector[tid] = (matrix[i] > matrix[i+blockDim.x])? matrix[i]: matrix[i+blockDim.x];
    __syncthreads();
    if(blockDim.x >= 1024){
        if(tid < 512)  vector[tid] = (vector[tid] > vector[tid+512])? vector[tid]: vector[tid+512];
        __syncthreads();
    }
    if(blockDim.x >= 512){
        if(tid < 256)  vector[tid] = (vector[tid] > vector[tid+256])? vector[tid]: vector[tid+256];
        __syncthreads();
    }
    if(blockDim.x >= 256){
        if(tid < 128)  vector[tid] = (vector[tid] > vector[tid+128])? vector[tid]: vector[tid+128];
        __syncthreads(); 
    }
    if(blockDim.x >= 128){
        if(tid < 64)  vector[tid] = (vector[tid] > vector[tid+64])? vector[tid]: vector[tid+64];
        __syncthreads();
    }
    if(tid < 32)  warpMax(vector, tid);
    if(tid == 0)  outputs[bid] = vector[0];    
}

__global__ void matrix_addition(float* matrix, const float* __restrict__ vector, int length){
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int i = bx * blockDim.x + tx;
    if(i < length)  matrix[i] += vector[bx];
}

__global__ void matrix_division(float* matrix_higher, const  float* __restrict__ matrix_lower, int length){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int i = tx + bx * blockDim.x;
    if(i < length)  matrix_higher[i] /= matrix_lower[bx];
}

__global__ void matrix_division_no_reshape(float* high_dim_matrix, float* low_dim_matrix, float* output, int length){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int i = tx + bx * blockDim.x;
    if(i < length)  output[i] = high_dim_matrix[i] / low_dim_matrix[bx];
}
"""
)


def concatenate(arrays, axis=0, allocator=None):
    """
    Join a sequence of arrays along an existing axis.
    :arg arrays: A sequnce of :class:`GPUArray`.
    :arg axis: Index of the dimension of the new axis in the result array.
        Can be -1, for the new axis to be last dimension.
    :returns: :class:`GPUArray`
    """
    # implementation is borrowed from pyopencl.array.concatenate()
    # {{{ find properties of result array

    shape = None

    def shape_except_axis(ary: GPUArray):
        return ary.shape[:axis] + ary.shape[axis + 1:]

    for i_ary, ary in enumerate(arrays):
        allocator = allocator or ary.allocator

        if shape is None:
            # first array
            shape = list(ary.shape)

        else:
            if len(ary.shape) != len(shape):
                raise ValueError("%d'th array has different number of axes "
                                 "(should have %d, has %d)"
                                 % (i_ary, len(ary.shape), len(shape)))

            if (ary.ndim != arrays[0].ndim
                    or shape_except_axis(ary) != shape_except_axis(arrays[0])):
                raise ValueError("%d'th array has residual not matching "
                                 "other arrays" % i_ary)

            shape[axis] += ary.shape[axis]

    # }}}

    shape = tuple(shape)
    dtype = np.find_common_type([ary.dtype for ary in arrays], [])
    result = empty(shape, dtype, allocator=allocator)

    full_slice = (slice(None),) * len(shape)

    base_idx = 0
    for ary in arrays:
        my_len = ary.shape[axis]
        result[full_slice[:axis] + (slice(base_idx, base_idx + my_len),) + full_slice[axis + 1:]] = ary
        base_idx += my_len

    return result


def load_dataset():
    """
    load dataset
    :return: dataset in numpy style
    """
    data_location = 'data.pk'
    data = pickle.load(open(data_location, 'rb'))
    return data


def customize_reduction(matrix):
    """
    This function is designed for matrix reduction along the last dimension
    :param matrix: 3-d matrix waited to be summed in the last dim
    :return: 2-d matrix after reduction
    """
    # This matrix if of shape (B, N, V)
    assert matrix.shape[-1] == 251
    padding = gpuarray.zeros((matrix.shape[0], matrix.shape[1], 5), dtype=np.float32)
    padded_matrix = concatenate((matrix, padding), axis=2)
    results = gpuarray.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float32)
    matrix_reduction(padded_matrix, results, block=(256 // 2, 1, 1), grid=(matrix.shape[0] * matrix.shape[1], 1, 1))
    return results


def customize_max_finder(matrix):
    """
    This function is designed to locate the maximum value of a matrix along the last dimension
    :param matrix: 2-d matrix shape of (B, NV) where B=100, N=4, V=251 in this case
    :return: maximum value vector
    """
    # This matrix is of shape (B, NV)
    padding = gpuarray.zeros((matrix.shape[0], 1024 - matrix.shape[-1]), dtype=np.float32)
    padded_matrix = concatenate((matrix, padding), axis=1)
    results = gpuarray.zeros((matrix.shape[0],), dtype=np.float32)
    find_max(padded_matrix, results, block=(1024 // 2, 1, 1), grid=(matrix.shape[0], 1, 1))
    return results


def customize_matrix_add(matrix, vector):
    """
    This function is designed to do addition of two matrices with different dims (broadcast)
    :param matrix: higher level matrix, normally in shape (*, dim)
    :param vector: lower level vector, normally in shape (*, )
    :return: the addition result in shape of (*, dim)
    """
    matrix_addition(matrix, vector, np.int32(matrix.size), block=(matrix.shape[1], 1, 1), grid=(matrix.shape[0], 1, 1))
    return matrix


def customize_matrix_division(matrix_higher, matrix_lower):
    """
    This function is designed to do division with two different-dims matrix (broadcast)
    :param matrix_higher: higher level matrix, normally in shape (*, dim)
    :param matrix_lower: lower level matrix, normally in shape (*, )
    :return: division result in shape of (*, dim)
    """
    # matrix_division(matrix_higher, matrix_lower, np.int32(matrix_higher.size), block=(matrix_higher.shape[-1], 1, 1),
    #                 grid=(matrix_higher.size // matrix_higher.shape[-1], 1, 1))
    matrix_division_no_reshape(matrix_higher, matrix_lower, output_divided_matrix, np.int32(matrix_higher.size),
                               block=(matrix_higher.shape[-1], 1, 1),
                               grid=(matrix_higher.size // matrix_higher.shape[-1], 1, 1))
    return output_divided_matrix


def matrixMulti(mat1, mat2):
    """
    Same as naive gpu version
    :param mat1: first matrix
    :param mat2: second matrix
    :return: multiplication result
    """
    alpha = 1
    beta = 0
    transa = 'n'
    transb = 'n'
    m = mat1.shape[0]
    n = mat2.shape[1]
    k = mat1.shape[1]
    C_gpu = gpuarray.zeros((m, n), dtype=np.float32)
    cublas.cublasSgemm(handle, transa, transb, n, m, k, alpha, mat2.gpudata, n, mat1.gpudata, k, beta,
                       C_gpu.gpudata, n)
    return C_gpu


def skcudaDot(mat1, mat2, transb="N"):
    """
    A skcuda API for matrix dot product
    :param mat1: first matrix
    :param mat2: second matrix
    :param transb: if needs transpose of second matrix
    :return: dot product result
    """
    return linalg.dot(mat1, mat2, transb=transb, handle=handle)


def copy_non_contiguous(dst, src):
    """
    Copy ``src`` array to ``dst`` array.
    A gpu-array may have a non contiguous block of memory,
    i.e. it may have substancial pitches/strides. However a cpu-array must have a contiguous block of memory.
    All four directions are allowed.
    """
    assert dst.shape == src.shape, \
        "Shapes do not match: " + str(dst.shape) + " <-> " + str(src.shape)

    itemsize = np.dtype(src.dtype).itemsize
    copy = cuda.Memcpy2D()

    copy.set_src_host(src)
    copy.set_dst_device(dst.gpudata)

    if itemsize != dst.strides[1]:
        # arrays have to be copied column by column, because there a two substantial pitches/strides
        # which is not supported by cuda.
        copy.src_pitch = itemsize
        copy.dst_pitch = dst.strides[0]
        copy.width_in_bytes = itemsize
        copy.height = src.shape[0]

        for col in range(src.shape[1]):
            copy.src_x_in_bytes = col * itemsize
            copy.dst_x_in_bytes = col * dst.strides[1]
            copy(aligned=False)
    else:
        # both arrays have a contiguous block of memory for each row
        copy.src_pitch = itemsize * src.shape[1]
        copy.dst_pitch = dst.strides[0]
        copy.width_in_bytes = itemsize * src.shape[1]
        copy.height = src.shape[0]
        copy(aligned=False)


# config settings
TRAIN_CONFIG = {"batch_size": 100,
                "hidden_dim": 128,
                "embedding_dim": 16}
# load trained model
model_location = 'partially_trained.pk'
trained = pickle.load(open(model_location, "rb"))
# load kernel functions
matrix_reduction = kernels.get_function("matrix_reduction")
find_max = kernels.get_function("find_max")
matrix_addition = kernels.get_function("matrix_addition")
# matrix_division = kernels.get_function("matrix_division")
matrix_division_no_reshape = kernels.get_function("matrix_division_no_reshape")
output_divided_matrix = gpuarray.zeros((TRAIN_CONFIG["batch_size"], 1004), dtype=np.float32)
# define streams for asynchronization
stream1 = cuda.Stream(flags=1)
stream2 = cuda.Stream(flags=1)
stream3 = cuda.Stream(flags=1)
stream4 = cuda.Stream(flags=1)
stream5 = cuda.Stream(flags=1)
streams = (stream1, stream2, stream3, stream4, stream5)
# creating handles
handle = cublas.cublasCreate()
linalg.init()


def logistic(y):
    """
    Function for calculating sigmoid activation.
    """
    try:
        return 1. / (1. + cumath.exp(-y))
    except RuntimeWarning:
        raise Exception("Overflow Encountered. Check your model!")


class Model(object):
    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_dim, context_length):
        """
        Definition of the model, including one hidden layer and one output layer, normally these are
        called MLP altogether
        :param batch_size: 100 in this case
        :param vocab_size: 251 in this case
        :param embedding_dim: 16 in this case
        :param hidden_dim: 128 in this case
        :param context_length: 4 in this case
        """
        # definition of hyper-parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        # definition of layers
        self.embedding_layer = gpuarray.zeros((self.batch_size, self.context_length * self.embedding_dim),
                                              dtype=np.float32)
        self.hidden_layer_activated = gpuarray.zeros((self.batch_size, self.hidden_dim), dtype=np.float32)

        # definition of trainable parameters
        embedding_weights = trained["word_embedding_weights"].astype(np.float32)
        self.embedding_weights = cuda.register_host_memory(embedding_weights)
        self.word_embedding_weights = gpuarray.to_gpu_async(self.embedding_weights, stream=streams[0])

        hidden_weights = trained['embed_to_hid_weights'].astype(np.float32)
        hidden_weights = cuda.register_host_memory(hidden_weights)
        self.emb_to_hid_weights = gpuarray.to_gpu_async(hidden_weights, stream=streams[1])

        hid_bias = trained['hid_bias'].astype(np.float32)
        hid_bias = cuda.register_host_memory(hid_bias)
        self.hid_bias = gpuarray.to_gpu_async(hid_bias, stream=streams[2])

        output_weights = trained['hid_to_output_weights'].astype(np.float32)
        output_weights = cuda.register_host_memory(output_weights)
        self.hid_to_out_weights = gpuarray.to_gpu_async(output_weights, stream=streams[3])

        output_bias = trained['output_bias'].astype(np.float32)
        output_bias = cuda.register_host_memory(output_bias)
        self.out_bias = gpuarray.to_gpu_async(output_bias, stream=streams[4])

        self.targets_offset = np.repeat((np.arange(context_length) * self.vocab_size)[np.newaxis, :],
                                        batch_size, axis=0).astype(np.int32)

        self.target_batch = np.zeros((batch_size, context_length * self.vocab_size), dtype=np.float32)
        self.target_batch_gpu = gpuarray.to_gpu((batch_size, context_length * self.vocab_size), dtype=np.float32)

    def _softmax(self, y):
        """
        Calculate softmax activation
        :param y: output of output layer
        :return: activated output
        """
        y = cumath.exp(y)
        y = y.reshape((-1, self.context_length, self.vocab_size))
        sum_y = customize_reduction(y)
        y = customize_matrix_division(y, sum_y)
        return y

    def forward(self, batch_data):
        """
        forward propagation of the model
        :param batch_data: masked inputs
        :return: final weights
        """
        # word_embedding_weights_cpu = self.word_embedding_weights.get()
        for i in range(self.context_length):
            # self.embedding_layer[:, i * self.embedding_dim:(i + 1) * self.embedding_dim] = \
            #     gpuarray.to_gpu(self.embedding_weights[batch_data[:, i], :])  # NEEDS MORE WORK
            copy_non_contiguous(self.embedding_layer[:, i * self.embedding_dim:(i + 1) * self.embedding_dim],
                                self.embedding_weights[batch_data[:, i], :])
        hidden_layer = customize_matrix_add(skcudaDot(self.embedding_layer, self.emb_to_hid_weights, "T"),
                                            self.hid_bias)  # (B, Nd) @ (Nd, H) -> (B, H)
        self.hidden_layer_activated = logistic(hidden_layer)
        output_layer = customize_matrix_add(skcudaDot(self.hidden_layer_activated, self.hid_to_out_weights, "T"),
                                            self.out_bias)  # (B, H) @ (H, NV) -> (B, NV)
        max_output_layer = customize_max_finder(output_layer)
        output_layer = customize_matrix_add(output_layer, -max_output_layer)
        output_layer_activated = self._softmax(output_layer)
        return output_layer_activated

    def indicator_matrix(self, batch_data, mask_zero=True):
        """
        Generate a (B, NV)-shape masked input batch data
        :param mask_zero: if setting zeros in masked places
        :param batch_data: masked inputs
        :return: ground truth of loss function
        """
        self.target_batch[:] = 0
        batch_data += self.targets_offset
        for c in range(self.context_length):
            self.target_batch[np.arange(self.batch_size), batch_data[:, c]] = 1.
            if mask_zero:
                self.target_batch[np.arange(self.batch_size), self.targets_offset[:, c]] = 0.
        copy_non_contiguous(self.target_batch_gpu, self.target_batch)

    @staticmethod
    def compute_loss(target_batch, output_activated):
        """
        Used as verification of the algorithm
        :param target_batch: ground truth
        :param output_activated: outputs of the model
        :return: cross entropy loss
        """
        cross_entropy = -gpuarray.sum(target_batch * cumath.log(output_activated + 1e-5))
        return cross_entropy

    def sample_input_mask(self):
        """Samples a binary mask for the inputs of size batch_size x context_len
        For each row, at most one element will be 1.
        """
        mask_idx = np.random.randint(self.context_length, size=(self.batch_size,))
        mask = np.zeros((self.batch_size, self.context_length),
                        dtype=np.int32)  # Convert to one hot B x N, B batch size, N context len
        mask[np.arange(self.batch_size), mask_idx] = 1
        return mask


def inference():
    """
    Inference with trained model
    :return: average loss
    """
    data = pickle.load(open("data.pk", "rb"))
    data_inputs = data["train_inputs"].astype(np.int32)
    model = Model(TRAIN_CONFIG["batch_size"], len(data["vocab"]), TRAIN_CONFIG['embedding_dim'],
                  TRAIN_CONFIG['hidden_dim'], data_inputs.shape[1])
    batch_size = TRAIN_CONFIG["batch_size"]
    num_batches = data_inputs.shape[0] // batch_size
    train_loss = 0.
    for m in range(num_batches):
        # Data Preparation
        data_inputs_batch = data_inputs[m * batch_size: (m + 1) * batch_size, :]
        mask = model.sample_input_mask()
        input_batch_masked = data_inputs_batch * (1 - mask)  # We only zero out one word per row
        target_batch_masked = data_inputs_batch * mask
        model.indicator_matrix(target_batch_masked)
        # forward
        output_activated = model.forward(input_batch_masked)
        # calculate cross entropy loss
        batch_loss = model.compute_loss(model.target_batch_gpu, output_activated) / batch_size
        train_loss += batch_loss
    return train_loss / num_batches


if __name__ == "__main__":
    times = 10
    start = time.time()
    for i in range(times):
        train_loss = inference()
    end = time.time()
    print("GPU execution time is {:.2f} seconds on average of {} attempts.".format((end - start)/times, times))
    # 7.81 seconds, loss: 3.85
    cublas.cublasDestroy(handle)
