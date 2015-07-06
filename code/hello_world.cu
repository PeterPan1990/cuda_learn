// device code
// author: Pan Yang
// date  : 2015-7-1

#include <stdio.h>
__global__ void mykernel(void)
{

}

int main(void)
{
    mykernel<<<1,1>>>();
    printf("cu: Hello World!\n");
    return 0;
}
