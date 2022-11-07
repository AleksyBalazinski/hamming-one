#include "Array.h"
#include "CudaAllocator.h"

__global__ void printFromGpu(Array<int, CudaAllocator<int>> arr)
{
    for (int i = 0; i < arr.size(); i++)
        printf("%d ", arr[i]);

    printf("\n");
}

int main(int argc, char **argv)
{
    Array<int> h_array(5);
    Array<int, CudaAllocator<int>> d_array(5);

    // init input h_array
    for (int i = 0; i < 5; i++)
        h_array[i] = i * i;

    h_array.copyTo(d_array, cudaMemcpyHostToDevice);
    printFromGpu<<<1, 1>>>(d_array);
}