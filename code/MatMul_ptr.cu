//MatAdd.cu
// author: Pan Yang
// date  : 2015-7-4

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
    
#define M 4  // height of A
#define N 5  // width of A ( == height of B)
#define P 3  // width of B

#define BLOCK_SIZE 16
    
// cpu code definition 
void MatMulOnHost(const float *A, const float *B, float *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < P; ++j)
        {
            float sum_ij = 0;
            
            for (k = 0; k < N; ++k)
                sum_ij += A[i * N + k] * B[k * P + j];
            C[i * P + j] = sum_ij;
        }
    }
}
  
    
// Kernel definition 
__global__ void MatMul(const float *A, const float *B, float *C)
{
    // Each thread computes one element of C by accumulating results into Cvalue
    float C_ij = 0; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int k;
    for (k = 0; k < N; ++k)
    {
        C_ij += A[i * N + k] * B[k * P + j];
    }
    
    C[i * P + j] = C_ij;
}

int main() 
{ 
    // time for the whole process
    int i, j;
    
    clock_t start, finish; 
    float time;
    start = clock(); 
    
    // load A, B, C on host
    float *A;
    size_t size = M * N * sizeof(float);
    A = (float *)malloc(size);
    
    float *B;
    size = N * P * sizeof(float);
    B = (float *)malloc(size);
    
    // initialize A
    srand((unsigned)time(NULL));
    for (i = 0; i < M * N; ++i)
    {
        A[i] = rand() / (float)RAND_MAX;
        //A[i] = 1.0;
    }
    // initialize B
    srand((unsigned)time(NULL));
    for (i = 0; i < N * P; ++i)
    {
        B[i] = rand() / (float)RAND_MAX;
        //B[i] = 0.01;
    }
    
    float *C;
    size = M * P * sizeof(float);
    C = (float *)malloc(size); // default all zeros
    
    // load a, b, c on device
    float *d_A;
    size = M * N * sizeof(float);
    cudaMalloc((void**)&d_A, size); // alloc memory on device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); // copy data from host to device
    
    float *d_B;
    size = N * P * sizeof(float);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    float *d_C;
    size = M * P * sizeof(float);
    cudaMalloc((void**)&d_C, size); // default are not zeros, maybe random number
    
    
    // Kernel invocation with m*16*16 threads
    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 GridDim((P + BlockDim.x - 1) / BlockDim.x, (M + BlockDim.x - 1) / BlockDim.y);
    MatMul<<<GridDim, BlockDim>>>(d_A, d_B, d_C);
    
    // copy results form device memory to host memory
    cudaMemcpy( C, d_C, size, cudaMemcpyDeviceToHost );
    
    //MatMulOnHost(A, B, C);

    // print A, B, C
    printf("---------------------------------------------------------------------\n");
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            printf(" %9f ", A[i * N +j]);
        printf("...");
        printf("\n");
    }
    printf("---------------------------------------------------------------------\n");
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < P; ++j)
            printf(" %9f ", B[i * P +j]);
        printf("...");
        printf("\n");
    }
    printf("---------------------------------------------------------------------\n");
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < P; ++j)
            printf(" %9f ", C[i * P +j]);
        printf("...");
        printf("\n");
    }
    
    // free space
    free(C); 
    free(B); 
    free(A);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    finish = clock();  
    time = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("calculation time = %fms\n", time);
    return 0;
}
