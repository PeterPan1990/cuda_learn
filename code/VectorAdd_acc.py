# simple python code for vector add use gpu

import numpy as np
from timeit import default_timer as timer
from numbapro import vectorize

@vectorize(["float32(float32, float32)"], target='gpu')
def VectorAdd(a, b):
        return a + b
        
def main():
    N = 32000000
    
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)
    
    start = timer()
    C = VectorAdd(A, B) # numbaoro automatically convert host memory to device memory
    vectoradd_time = timer() - start
    
    print "C[:5] = " + str(C[:5])
    print "C[-5:] = " + str(C[-5:])
    
    print "VectorAdd took %f seconds" % vectoradd_time
    
if __name__ == "__main__":
    main()
    
# shell command:' nvprof python VectorAdd.py ' for time compare
# nvprof: tool that display timeline of your application's CPU and GPU activity