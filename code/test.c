// test.c
// author: Pan Yang
// date  : 2015-7-2

#include <stdio.h>
#include <stdlib.h>
    
#define N 10
    
int main()
{
    float **a, **b, **c;
    int i = 0, j = 0;
    
    a = (float **)malloc(N * sizeof(float *));
    b = (float **)malloc(N * sizeof(float *));
    c = (float **)malloc(N * sizeof(float *));
    for (i = 0; i < N; ++i)
    {
        a[i] = (float *)malloc(N * sizeof(float));
        b[i] = (float *)malloc(N * sizeof(float));
        c[i] = (float *)malloc(N * sizeof(float));
    }
    
    // initialize a,b
    
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
            a[i][j] = i * N + j;
            b[i][j] = i * N + j;
    }
    
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
            c[i][j] = a[i][j] + b[i][j];
    }
    
    
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
            printf(" %f ", a[i][j]);
        printf("\n");
    }
    
    return 0;
}