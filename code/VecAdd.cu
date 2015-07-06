//VecAdd.cu
// author: Pan Yang
// date  : 2015-7-2

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
    
    
#define SIZE 1024

// Kernel definition 
__global__ void VecAdd_T(int *a, int *b, int *c, int n) 
{ 
    int i = threadIdx.x; 
    
    if (i < n)
        c[i] = a[i] + b[i]; 
} 

// Kernel definition 
__global__ void VecAdd_B(int *a, int *b, int *c, int n) 
{ 
    int i = blockIdx.x; 
    
    if (i < n)
        c[i] = a[i] + b[i]; 
} 

// Kernel definition 
__global__ void VecAdd_BT(int *a, int *b, int *c, int n) 
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (i < n)
        c[i] = a[i] + b[i]; 
} 


int main() 
{ 
    // time for the whole process
    clock_t start, finish; 
    float time;
    start = clock(); 
    
    // define a, b, c and alloc memory for them
    int *a, *b, *c;
    
    a = (int *)malloc(SIZE * sizeof(int));
    b = (int *)malloc(SIZE * sizeof(int));
    c = (int *)malloc(SIZE * sizeof(int));
    
    // define d_a, d_b, d_c and alloc memory for them
    int *d_a, *d_b, *d_c;
    
    cudaMalloc( &d_a, SIZE * sizeof(int));
    cudaMalloc( &d_b, SIZE * sizeof(int));
    cudaMalloc( &d_c, SIZE * sizeof(int));
    
    // initialize a, b, c
    int i = 0;
    
    for (i = 0; i < SIZE; ++i)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }
    
    
    // copy data from host memory to device memory
    cudaMemcpy( d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice );
    //cudaMemcpy( d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice );
      
    //----------------------------------------------------------------
    
    /*
    cudaEvent_t start_cu, stop_cu;
    float time_gpu;
    cudaEventCreate(&start_cu);
    cudaEventCreate(&stop_cu);
    cudaEventRecord( start_cu, 0);
    */
    
    // Kernel invocation with N threads 
    VecAdd_T<<<1, SIZE>>>(d_a, d_b, d_c, SIZE);
    
    // Kernel invocation with N blocks 
    //VecAdd_B<<<SIZE, 1>>>(d_a, d_b, d_c, SIZE); 
    
    // Kernel invocation with m blocks and n threads;
    //int threadsPerBlock = 256;
    //int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    //VecAdd_BT<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, SIZE);

    /*
    cudaEventRecord( stop_cu, 0);
    cudaEventSynchronize( stop_cu );
    cudaEventElapsedTime( &time_gpu, start_cu, stop_cu );
    cudaEventDestroy( start_cu );
    cudaEventDestroy( stop_cu );
    */
    
    // copy results form device memory to host memory
    cudaMemcpy( c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost );
    //------------------------------------------------------------------
    
    for (i = 0; i < 10; ++i)
    {
        printf("c[%d] = %d\n", i, c[i]);
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
