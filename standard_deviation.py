# Import and initialize PyCuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit  # optional use >> reason: initialization, context creation, and cleanup can also be performed manually

import numpy  # utilized to transfer data onto the device; transfer data from numpy arrays on the host


a = numpy.random.randn(4, 4)   # generate random array
# require conversion because variable a consists of double precision numbers,
# but most nVidia devices only support single precision
a = a.astype(numpy.float32)

# allocate memory & transfer data to gpu
a_gpu = cuda.to_device(a)


# write code to double each entry in a_gpu
mod = SourceModule("""
    __global__ void sum(float *a, float b)
    {
        int idx = threadIdx.x;
        a[idx] *= 2;
    }
""")

sum = mod.get_function("doublify")
sum(a_gpu, block=(4, 4, 1))

# fetch the data back from the GPU and display it
a_doubled = cuda.from_device_like(devptr=a_gpu, other_ary=a)  # above two lines of code clubbed in one now
print(a_doubled)
print(a)
