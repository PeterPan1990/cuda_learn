//MatAdd.cu
// author: Pan Yang
// date  : 2015-7-5

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
    
#define M 80  // height of A
#define N 48  // width of A ( == height of B)
#define P 128  // width of B

#define BLOCK_SIZE 16

typedef struct {
    int height;
    int width;
    int stride;
    float *elements;
}Matrix;
    
// cpu code definition 
void MatMulOnHost(const Matrix A, const Matrix B, Matrix C)
{
    int i, j, k;
    for (i = 0; i < A.height; ++i)
    {
        for (j = 0; j < B.width; ++j)
        {
            float sum_ij = 0;
            
            for (k = 0; k < A.width; ++k)
                sum_ij += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            C.elements[i * C.width + j] = sum_ij;
        }
    }
}

// Kernel definition for Matrix multiplication
__global__ void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    /// Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = A.width * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + A.width - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * B.width;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Cvalue = 0;
    
    // shared memory used to store Asub and Bsub
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // loop over all the sub matrices of A and B that are required to compute Csub,
    // multiply each pair of sub matrices together and sum the results
    int a, b, k;
    for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        // load Asub and Bsub from global memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[ty][tx] = A.elements[a + A.width * ty + tx];
        Bs[ty][tx] = B.elements[b + B.width * ty + tx];
        
        // synchronize to make sure the sub-matrices are loaded before starting calculation
        __syncthreads();
        
        // multiply Asub and Bsub together
        for (k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];
        
        // synchronize to make sure that the preceding calculation is done before loading 
        // two new sub-matrices
        __syncthreads();
    }
    
    // write Cvalue to global memory, each thread write one element
    int c = C.width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C.elements[c + C.width * ty + tx] = Cvalue;
}

int main() 
{ 
    // get the basic infomation of GPU device
    printf("\n");
    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    int devID = 0;
    cudaDeviceProp deviceProp;
    cudaError_t error;
    
    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major,      deviceProp.minor);
    }
    
    // load A, B, C on host
    Matrix A;
    A.height = M;
    A.width = N;
    size_t size = A.height * A.width * sizeof(float);
    A.elements = (float *)malloc(size);
    
    Matrix B;
    B.height = N;
    B.width = P;
    size = B.height * B.width * sizeof(float);
    B.elements = (float *)malloc(size);
    
    // initialize A
    srand(rand());
    int i;
    for (i = 0; i < A.height * A.width; ++i)
    {
        A.elements[i] = rand() / (float)RAND_MAX;
        //A.elements[i] = 1.0;
    }
    // initialize B
    srand(rand());
    for (i = 0; i < B.height * B.width; ++i)
    {
        B.elements[i] = rand() / (float)RAND_MAX;
        //B.elements[i] = 0.01;
    }
    
    Matrix C;
    C.height = M;
    C.width = P;
    size = C.height * C.width * sizeof(float);
    C.elements = (float *)malloc(size); // default all zeros
    
    Matrix ref_C; // reference C for result check
    ref_C.height = M;
    ref_C.width = P;
    size = ref_C.height * ref_C.width * sizeof(float);
    ref_C.elements = (float *)malloc(size); // default all zeros
    
    // load a, b, c on device
    Matrix d_A;
    d_A.height = M;
    d_A.width = N;
    size = d_A.height * d_A.width * sizeof(float);
    cudaMalloc(&d_A.elements, size); // alloc memory on device
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); // copy data from host to device
    
    Matrix d_B;
    d_B.height = N;
    d_B.width = P;
    size = d_B.height * d_B.width * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_C;
    d_C.height = M;
    d_C.width = P;
    size = d_C.height * d_C.width * sizeof(float);
    cudaMalloc(&d_C.elements, size); // default are not zeros, maybe random number
    
    // Kernel invocation with m*16*16 threads
    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 GridDim((P + BlockDim.x - 1) / BlockDim.x, (M + BlockDim.x - 1) / BlockDim.y);
    
    cudaEvent_t start_cu, stop_cu;
    float time_gpu = 0.0f;
    cudaEventCreate(&start_cu);
    cudaEventCreate(&stop_cu);
    cudaEventRecord( start_cu, 0);
    
    int nIter = 300;
    
    for (i = 0; i < nIter; ++i)
    {
        MatMul<<<GridDim, BlockDim>>>(d_A, d_B, d_C);
    }
    cudaEventRecord( stop_cu, 0);
    cudaEventSynchronize( stop_cu );
    cudaEventElapsedTime( &time_gpu, start_cu, stop_cu );
    cudaEventDestroy( start_cu );
    cudaEventDestroy( stop_cu );
    
    float msecPerMatrixMul = time_gpu / nIter;
    double flopsPerMatrixMul = 2.0 * (double)A.width * (double)A.height * (double)B.width;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul,
            BlockDim.x * BlockDim.y);
    
    // copy results form device memory to host memory
    cudaMemcpy( C.elements, d_C.elements, size, cudaMemcpyDeviceToHost );
  
    MatMulOnHost(A, B, ref_C);
    printf("GridDim: (%d, %d)\n", GridDim.x, GridDim.y);

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu| / <|x|, |y|>  < eps
    printf("Checking computed result for correctness ... \n");
    bool correct = true;
    
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(M * P); i++)
    {
        double abs_err = fabs(ref_C.elements[i] - C.elements[i]);
        double dot_length = A.width;
        double abs_val = fabs(ref_C.elements[i]);
        double rel_err = abs_err / abs_val / dot_length ;

        if (rel_err > eps)
        {
            //printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, ref_C.elements[i], C.elements[i], eps);
            correct = false;
        }
    }
    
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // free space on host and device
    free(C.elements); 
    free(B.elements); 
    free(A.elements);
    
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
    
    return 0;
}
