//MatAdd.cu
// author: Pan Yang
// date  : 2015-7-3

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>

#define M 4 // height of matrix
#define N 5 // width of matrix
#define SIZE (M*N)
#define BLOCK_SIZE 16


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
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < M && j < N)
        C[i*N + j] = A[i*N + j] + B[i*N + j];
}

int main() 
{ 
    // time for the whole process
    clock_t start, finish; 
    float time;
    start = clock(); 
    
    //----------------------------------1D implementation for 2D array--------------------------------------------------------
    // define a, b, c and alloc memory for them
    
    float *a, *b, *c;
    int i, j;
    
    a = (float *)malloc(SIZE * sizeof(float));
    b = (float *)malloc(SIZE * sizeof(float));
    c = (float *)malloc(SIZE * sizeof(float));
    
    for (i = 0; i < SIZE; ++i)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc( (void**)&d_a, SIZE * sizeof(float));
    cudaMalloc( (void**)&d_b, SIZE * sizeof(float));
    cudaMalloc( (void**)&d_c, SIZE * sizeof(float));
    
    // copy data from host memory to device memory
    cudaMemcpy( d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice );
    //cudaMemcpy( d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice );

    // Kernel invocation with m*16*16 threads
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x -1) / threadsPerBlock.x, (M + threadsPerBlock.y -1) / threadsPerBlock.y);
    MatAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c); 

    // copy results form device memory to host memory
    cudaMemcpy( c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost );
    
    //--------------------------------end of 1D implementation for 2D array----------------------------------------------------
    
    
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            printf(" %9f ", a[i * N +j]);
        printf("...");
        printf("\n");
    }
    printf("--------------------------------------------------\n");
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            printf(" %9f ", b[i * N +j]);
        printf("...");
        printf("\n");
    }
    printf("--------------------------------------------------\n");
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            printf(" %9f ", c[i * N +j]);
        printf("...");
        printf("\n");
    }
    
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
