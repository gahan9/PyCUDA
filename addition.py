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
__global__ void addition(float *result, float *a, float *b)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    result[index] = a[index] + b[index];
}
""")


class Vector(object):
    def __init__(self, total_elements=512):
        self.total_elements = total_elements
        self.a = numpy.random.randn(array_size).astype(numpy.float32)
        self.b = numpy.random.randn(array_size).astype(numpy.float32)
        self.result = numpy.zeros_like(self.a)

    def addition(self, blocks=None, threads=1):
        blocks = blocks if blocks else self.total_elements
        addition = mod.get_function("addition")
        addition(
            driver.Out(self.result), driver.In(self.a), driver.In(self.b),
            block=(blocks, 1, 1), grid=(512, 512))  # block = (blocks, threads, 1)


def test(array_size=400):
    vector_obj = Vector(array_size)
    vector_obj.addition(blocks=512, threads=1)
    print("Vector a:---\n{}".format(vector_obj.a))
    print("Vector b:---\n{}".format(vector_obj.b))
    print("Result:----\n", vector_obj.result)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        array_size = int(sys.argv[1])
        test(array_size)
    else:
        test()
