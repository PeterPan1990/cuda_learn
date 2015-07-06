//MatAdd.cu
// author: Pan Yang
// date  : 2015-7-1

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
    
#define N 32   
#define SIZE (N*N)
#define THREADS_PER_BLOCK 256


// Kernel definition 
/*
__global__ void MatAdd_B(float A[m][m], float B[m][m], float C[m][m]) 
{ 
    int i = threadIdx.x; 
    int j = threadIdx.y; 
    
    if (i < m*m && j < m*m)
        C[i][j] = A[i][j] + B[i][j]; 
} 
*/
    
// Kernel definition 
__global__ void MatAdd(float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    for (k = 0; k < N; ++k)
        C[i*N + j] = A[i*N + j] + B[i*N + j];
}

int main() 
{ 
    // time for the whole process
    clock_t start, finish; 
    float time;
    start = clock(); 
    
    // define a, b, c and alloc memory for them
    float *a, *b, *c;
    int i, j;
    
    a = (float *)malloc(SIZE * sizeof(float));
    b = (float *)malloc(SIZE * sizeof(float));
    c = (float *)malloc(SIZE * sizeof(float));
    
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
            a[i*N + j] = rand() % 10;
    }
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
            b[i*N + j] = rand() % 10;
    }
    
    
    float *d_a, *d_b, *d_c;
    cudaMalloc( &d_a, SIZE * sizeof(float));
    cudaMalloc( &d_b, SIZE * sizeof(float));
    cudaMalloc( &d_c, SIZE * sizeof(float));
    
    
    
    
    // copy data from host memory to device memory
    cudaMemcpy( d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice );
      

    // Kernel invocation with m*16*16 threads
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x -1) / threadsPerBlock.x, (N + threadsPerBlock.y -1) / threadsPerBlock.y);
    MatAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c); 

    // copy results form device memory to host memory
    cudaMemcpy( c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost );
    //------------------------------------------------------------------
    
    for (i = 0; i < 10; ++i)
    {
        for (j = 0; j < 10; ++j)
            printf(" %f ", a[i*N + j]);
        printf("...");
        printf("\n");
    }
    printf("--------------------------------------------------\n");
    for (i = 0; i < 10; ++i)
    {
        for (j = 0; j < 10; ++j)
            printf(" %f ", b[i*N + j]);
        printf("...");
        printf("\n");
    }
    printf("--------------------------------------------------\n");
    for (i = 0; i < 10; ++i)
    {
        for (j = 0; j < 10; ++j)
            printf(" %f ", c[i*N + j]);
        printf("...");
        printf("\n");
    }
    
    //printf("calculation time_gpu = %fms\n", time_gpu);
    
    // free space
    free(a); 
    free(b); 
    free(c);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    finish = clock();  
    time = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("calculation time = %fms\n", time);
    return 0;
}
