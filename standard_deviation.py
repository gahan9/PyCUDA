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

__global__ void sum(float result, float *a, int number_of_elements)
{
    result = summation(a, number_of_elements);
}
""")


class Vector(object):
    def __init__(self, total_elements=512):
        self.total_elements = total_elements
        # self.a = numpy.random.randn(self.total_elements).astype(numpy.float32)
        # self.b = numpy.random.randn(self.total_elements).astype(numpy.float32)
        self.a = numpy.array([i for i in range(10)]).astype(numpy.float32)
        self.result = numpy.zeros_like(self.a)

    def sum(self, blocks=None, threads=1):
        blocks = blocks if blocks else self.total_elements
        sum_in_cuda = mod.get_function("sum")
        sum_in_cuda(
            driver.Out(self.result), driver.In(self.a), driver.In(self.total_elements),
            block=(blocks, 1, 1), grid=(512, 512))  # block = (blocks, threads, 1)


def test(array_size=400):
    vector_obj = Vector(array_size)
    vector_obj.sum(blocks=512, threads=1)
    print("Vector a:---\n{}".format(vector_obj.a))
    # print("Vector b:---\n{}".format(vector_obj.b))
    print("(device)Result:----\n", vector_obj.result)
    print("(host)Result:----\n", sum(vector_obj.a))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        array_size = int(sys.argv[1])
        test(array_size)
    else:
        test()
