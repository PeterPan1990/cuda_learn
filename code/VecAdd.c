//VecAdd.c
// author: Pan Yang
// date  : 2015-7-2

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>

#define SIZE 1024
    
void VecAdd(int *a, int *b, int *c, int n)
{
    int i;
    
    for (i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    clock_t start, finish; 
    float time;
    start = clock(); 
    
    // define a, b, c and alloc memory for them
    int *a, *b, *c;
    a = (int *)malloc(SIZE * sizeof(int));
    b = (int *)malloc(SIZE * sizeof(int));
    c = (int *)malloc(SIZE * sizeof(int));
    
    // initialize a, b, c
    int i = 0;
    
    for (i = 0; i < SIZE; ++i)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }
    
    
    VecAdd(a, b, c, SIZE);
    
    
    for (i = 0; i < 10; ++i)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }
    
    
    free(a); 
    free(b); 
    free(c);
        
    finish = clock();  
    time = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("calculation time = %fms\n", time);
    return 0;
}