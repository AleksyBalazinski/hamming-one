#include <string>
#include <iostream>

#include "Array.h"
#include "Utils.h"
#include "CudaAllocator.h"

__global__ void printFromGPU(Array<int, CudaAllocator<int>> *dev_arr, int numOfSequences)
{
    printf("GPU:\n");
    for (int i = 0; i < numOfSequences; i++)
    {
        printf("array size = %d\n", (int)dev_arr[i].size());
        for (int j = 0; j < dev_arr[i].size(); j++)
        {
            printf("%d", dev_arr[i][j]);
        }
        printf("\n");
    }
}

// __global__ void freeSequences(Array<int, CudaAllocator<int>> *dev_sequences, int numOfSequences)
// {
//     for (int i = 0; i < numOfSequences; i++)
//     {
//         dev_sequences[i].freeArray();
//     }
// }

int main(int argc, char **argv)
{
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);

    // load sequences from a file to host's memory
    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);
    std::cout << numOfSequences << ' ' << seqLength << '\n';

    Array<int> *sequences = new Array<int>[numOfSequences];
    for (int i = 0; i < numOfSequences; i++)
    {
        sequences[i] = Array<int>(seqLength);
    }
    readDataFile(pathToData, sequences, numOfSequences);
    printSequences(sequences, numOfSequences, std::cout);

    // copy sequences to device's global memory
    Array<int, CudaAllocator<int>> *temp_dev_sequences =
        new Array<int, CudaAllocator<int>>[numOfSequences];
    for (int i = 0; i < numOfSequences; i++)
    {
        temp_dev_sequences[i] = Array<int, CudaAllocator<int>>(seqLength);
        sequences[i].copyTo(temp_dev_sequences[i], cudaMemcpyHostToDevice);
    }

    Array<int, CudaAllocator<int>> *dev_sequences;
    cudaMalloc(&dev_sequences, numOfSequences * sizeof(Array<int, CudaAllocator<int>>));
    cudaMemcpy(dev_sequences, temp_dev_sequences, numOfSequences * sizeof(Array<int, CudaAllocator<int>>), cudaMemcpyHostToDevice);

    // printFromGPU<<<1, 1>>>(dev_sequences, numOfSequences);

    // start a kernel which will compute hashes and construct matching and own triples

    // for (int i = 0; i < numOfSequences; i++)
    // {
    //     sequences[i].freeArray();
    // }
    // delete[] sequences;

    // freeSequences<<<1, 1>>>(dev_sequences, numOfSequences);
    //  cudaFree(dev_sequences);
}