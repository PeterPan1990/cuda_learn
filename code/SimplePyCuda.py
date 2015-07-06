# iterate sin function use GPU acceleration

import numpy as np
import sys


import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel

blocks = 64
block_size = 128
nbr_values = blocks * block_size
n_iter = 100000

print "Using %d threads..." % nbr_values
print "Calculation %d iterations..." % n_iter

# create two timers to test the speed 
start = cuda.Event()
stop = cuda.Event()

############################################################
# SourceModule approach
# write cuda c code and compile it two ptx with SourceModule
mod = SourceModule("""
__global__ void gpusin(float *dest, float *a, int n_iter)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    int n;
    for (n = 0; n < n_iter; ++n)
    {
        a[i] = sin(a[i]);
    }
    dest[i] = a[i];
}
""")

gpusin = mod.get_function("gpusin")

# create an array of 1D
a = np.ones(nbr_values).astype(np.float32)
# create an array that receive the result
dest = np.zeros_like(a)

start.record()
gpusin(cuda.Out(dest), cuda.In(a), np.int32(nbr_values), grid=(blocks, 1), block=(block_size, 1, 1))
stop.record()
stop.synchronize()
secs = start.time_till(stop)*1e-3

print "\n"
print "SourceModule time and first three results:"
print "%f, %s" % (secs, str(dest[:3]))

#################################################################
# Elementwise approach
kernel = ElementwiseKernel(
    "float *a, int n_iter",
    "int n; for (n = 0; n < n_iter; ++n) { a[i] = sin(a[i]); }",
    "gpusin")

a = np.ones(nbr_values).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)

start.record()
kernel(a_gpu, np.int32(n_iter))
stop.record()
stop.synchronize()
secs = start.time_till(stop)*1e-3
print "\n"
print "Elementwise time and first three results:"
print "%f, %s" % (secs, str(a_gpu[:3]))

#################################################################
# GPUArray approach

a = np.ones(nbr_values).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)

start.record()
for i in xrange(n_iter):
    a_gpu = pycuda.cumath.sin(a_gpu)
stop.record()
stop.synchronize()
secs = start.time_till(stop)*1e-3
print "\n"
print "GPUArray time and first three results:"
print "%f, %s" % (secs, str(a_gpu[:3]))
