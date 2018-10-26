#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Gahan Saraiya
GiT: https://github.com/gahan9
StackOverflow: https://stackoverflow.com/users/story/7664524

Vector Addition in CUDA
"""
import numpy
import pycuda.autoinit
from pycuda import driver

from pycuda.compiler import SourceModule

mod = SourceModule("""
__device__ int summation(float *a, int n)
{
    if (n == 1){
        return a[0];
    }
    else if( n == 2 ){
        return a[0] + a[1];
    }
    else
        return summation(a, n/2) * summation(a, n - (n/2));
}

__global__ void vector_add(float *a, float *b)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    a[index] = a[index] + b[index];
}

__global__ void sum(float *result, float *a, float *number_of_elements)
{
    result[0] = summation(a, number_of_elements[0]);
}
""")


class Vector(object):
    def __init__(self, total_elements=512):
        self.total_elements = total_elements
        # self.a = numpy.random.randn(self.total_elements).astype(numpy.float32)
        # self.b = numpy.random.randn(self.total_elements).astype(numpy.float32)
        self.total_elements = 500
        self.a = numpy.array([i for i in range(self.total_elements)]).astype(numpy.float32)
        # self.total_elements = numpy.array([self.total_elements]).astype(numpy.float32)
        self.result = numpy.zeros_like(self.a)
        self.vector_add = mod.get_function("vector_add")

    @property
    def cpu_bottleneck(self):
        """
        bottleneck limit where CPU can perform computation faster (to avoid overhead of
        transferring data to/from GPU)
        """
        return 100

    def sum(self, blocks=None, threads=1):
        # blocks = blocks if blocks else self.total_elements
        # split original array in to equal sub array of length equals to cpu array size
        parallize_equal = self.total_elements - (self.total_elements % self.cpu_bottleneck)
        arrays = numpy.array_split(self.a[:parallize_equal], self.total_elements//self.cpu_bottleneck)
        self.result = arrays[0]
        for i in arrays[1:]:
            self.vector_add(
                driver.Out(self.result), driver.In(self.result), driver.In(i),
                block=(blocks, 1, 1), grid=(512, 512))  # block = (blocks, threads, 1)
        return self.result.sum() + self.a[parallize_equal:].sum()


def test(array_size=400):
    vector_obj = Vector(array_size)
    print("Vector a:---\n{}".format(vector_obj.a))
    vector_obj.sum(blocks=512, threads=1)
    # print("Vector b:---\n{}".format(vector_obj.b))
    print("(device)Result:----\n", vector_obj.result[0])
    print("(host)Result:----\n", sum(vector_obj.a))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        array_size = int(sys.argv[1])
        test(array_size)
    else:
        test()
