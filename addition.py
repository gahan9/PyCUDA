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
    def __init__(self):
        self.a = numpy.random.randn(400).astype(numpy.float32)
        self.b = numpy.random.randn(400).astype(numpy.float32)
        self.result = numpy.zeros_like(self.a)

    def addition(self, blocks=1, threads=1):
        addition = mod.get_function("addition")
        addition(
            driver.Out(self.result), driver.In(self.a), driver.In(self.b),
            block=(400, 1, 1), grid=(1, 1))  # block = (blocks, threads, 1)


def test():
    vector_obj = Vector()
    vector_obj.addition()
    print("Vector a:---\n{}".format(vector_obj.a))
    print("Vector b:---\n{}".format(vector_obj.b))
    print("Result:----\n", vector_obj.result)


if __name__ == "__main__":
    pass