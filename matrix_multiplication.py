#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Gahan Saraiya
GiT: https://github.com/gahan9
StackOverflow: https://stackoverflow.com/users/story/7664524

Multiplies two square matrices together using a *single* block of threads and
global memory only. Each thread computes one element of the resulting matrix.
"""

import numpy
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c) {{
    // P_value is used to store the element of the matrix
    // that is computed by the thread
    float P_value = 0;

    // Each thread loads one row of M and one column of N, 
    //  to produce one element of P.
    for (int k = 0; k < {MATRIX_SIZE}; ++k) {{
        float A_element = a[threadIdx.y * {MATRIX_SIZE} + k];
        float B_element = b[k * {MATRIX_SIZE} + threadIdx.x];
        P_value += A_element * B_element;
    }}

    // Write the matrix to device memory;
    // each thread writes one element
    c[threadIdx.y * {MATRIX_SIZE} + threadIdx.x] = P_value;
}}
"""

# define the (square) matrix size this number (squared) can't exceed max_threads,
# visit http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
# on how to get this number for your device
MATRIX_SIZE = 2
# get the kernel code from the template by specifying the constant MATRIX_SIZE
kernel_code = kernel_code_template.format(MATRIX_SIZE=MATRIX_SIZE)
# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# create two random square matrices
a_cpu = numpy.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(numpy.float32)
b_cpu = numpy.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(numpy.float32)

# calculate result on cpu to verify with GPU result
c_cpu = numpy.dot(a_cpu, b_cpu)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), numpy.float32)

# get the kernel function from the compiled module
matrix_mul = mod.get_function("MatrixMulKernel")

if __name__ == "__main__":
    # call the kernel on the card
    matrix_mul(
        # inputs
        a_gpu, b_gpu,
        # output
        c_gpu,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block=(MATRIX_SIZE, MATRIX_SIZE, 1),
        )

    # print the results
    print("{}\nMatrix A (GPU):\n{}".format("-"*80, a_gpu.get()))
    print("{}\nMatrix B (GPU):\n{}".format("-"*80, b_gpu.get()))
    print("{}\nMatrix C (GPU):\n{}".format("-"*80, c_gpu.get()))
    print("{}\nCPU-GPU difference:\n{}".format("-"*80, c_cpu - c_gpu.get()))

    numpy.allclose(c_cpu, c_gpu.get())
