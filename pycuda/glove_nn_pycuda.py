import pycuda.autoinit
from pycuda.gpuarray import *
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np
from math import ceil
from skcuda import linalg
import time
import pickle

kernels = SourceModule(
    """
    #define blockSize blockDim.x
    __device__ void warpReduce(volatile float* vector, int tid){
        if(blockSize >= 32) vector[tid] += vector[tid+32];
        if(blockSize >= 16) vector[tid] += vector[tid+16];
        if(blockSize >= 8) vector[tid] += vector[tid+8];
        if(blockSize >= 4) vector[tid] += vector[tid+4];
        if(blockSize >= 2) vector[tid] += vector[tid+2];
        if(blockSize >= 1) vector[tid] += vector[tid+1];
    } 
    
    __global__ void matrix_reduction(float* matrix, float* reduction_matrix){
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int i = tid + bid * blockSize * 2;
        __shared__ float vector[1024];
        vector[tid] = matrix[i] + matrix[i+blockSize];
        __syncthreads();
        if(blockSize >= 512){
            if(tid < 512){
                vector[tid] += vector[tid + 512];
            }
            __syncthreads();
        }
        if(blockSize >= 256){
            if(tid < 256){
                vector[tid] += vector[tid + 256];
            } 
            __syncthreads(); 
        }
        if(blockSize >= 128){
            if(tid < 128){
                vector[tid] += vector[tid + 128];
            }
            __syncthreads();
        }
        if(blockSize >= 64){
            if(tid < 64){
                vector[tid] += vector[tid + 64];
            } 
            __syncthreads();
        }
        if(tid < 32) warpReduce(vector, tid);
        if(tid == 0){
            reduction_matrix[bid] = vector[0];
        }
    }
    
    """
)

matrix_reduction = kernels.get_function("matrix_reduction")


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
        return ary.shape[:axis] + ary.shape[axis+1:]

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
        result[full_slice[:axis] + (slice(base_idx, base_idx+my_len),) + full_slice[axis+1:]] = ary
        base_idx += my_len

    return result


def load_dataset():
    data_location = 'data.pk'
    data = pickle.load(open(data_location, 'rb'))
    return data


def customize_reduction(matrix):
    # This matrix if of shape (B, N, V)
    assert matrix.shape[-1] == 251
    padding = gpuarray.zeros((matrix.shape[0], matrix.shape[1], 5), dtype=np.float32)
    padded_matrix = concatenate((matrix, padding), axis=2)
    results = gpuarray.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float32)
    matrix_reduction(padded_matrix, results, block=(256 // 2, 1, 1), grid=(matrix.shape[0] * matrix.shape[1], 1, 1))
    return results
