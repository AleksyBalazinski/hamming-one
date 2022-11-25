#define noDEBUG

#include <string>
#include <iostream>

#include "utils.h"
#include "common/cuda_allocator.hpp"
#include "common/triple.cuh"
#include "hash_table_sc/hash_table_sc.hpp"
#include "hamming/rolling_hash.cuh"
#include "Hash.h"

//#define imin(a, b) ((a) < (b) ? (a) : (b))
#define imin(a, b) (b)
#define PRINTF_FIFO_SIZE (long long int)1e15

int blocksNum(int threads, int threadsPerBlock)
{
    return imin(32, (threads + threadsPerBlock - 1) / threadsPerBlock);
}

__global__ void getHashes(int *sequences, int numOfSequences, int seqLength, size_t *prefixes, size_t *suffixes,
                          Triple<size_t, size_t, int> *matchingHashes, Triple<size_t, size_t, int> *ownHasbes)
{
    int seqId = threadIdx.x + blockDim.x * blockIdx.x;
    if (seqId > numOfSequences - 1)
        return;

    int offset = seqId * seqLength;
    int *sequence = sequences + offset;
    size_t *curPrefixes = prefixes + offset;
    size_t *curSuffixes = suffixes + offset;
    Triple<size_t, size_t, int> *curMatchingHashes = matchingHashes + offset;
    Triple<size_t, size_t, int> *curOwnHashes = ownHasbes + offset;
    hamming::computePrefixHashes(sequence, seqLength, curPrefixes);
    hamming::computeSuffixHashes(sequence, seqLength, curSuffixes);

    for (int i = 0; i < seqLength; i++)
    {
        size_t prefixHash;
        if (i == 0)
            prefixHash = 0;
        else
            prefixHash = curPrefixes[i - 1];

        size_t suffixHash;
        if (i == seqLength - 1)
            suffixHash = 0;
        else
            suffixHash = curSuffixes[seqLength - i - 2];

        int erased = sequence[i];

        curMatchingHashes[i] = Triple<size_t, size_t, int>(prefixHash, suffixHash, erased == 0 ? 1 : 0);
        curOwnHashes[i] = Triple<size_t, size_t, int>(prefixHash, suffixHash, erased);
    }
}

template <class Key, class T, class Hash>
__global__ void findHammingOnePairs(HashTableSC<Key, T, Hash> table, Triple<size_t, size_t, int> *matchingHashes, int numOfSequences, int seqLength)
{
    int elemId = threadIdx.x + blockDim.x * blockIdx.x;
    if (elemId > seqLength * numOfSequences - 1)
        return;

    Entry<Key, T> *cur = table.getBucket(matchingHashes[elemId]);
    while (cur != nullptr)
    {
        if (cur->key == matchingHashes[elemId])
        {
            printf("%d %d\n", cur->value, elemId / seqLength);
        }
        cur = cur->next;
    }
}

int main(int argc, char **argv)
{
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, PRINTF_FIFO_SIZE);

    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);
    const double loadFactor = std::stod(argv[3]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    size_t totalLen = numOfSequences * seqLength;
    // const size_t HASH_ENTRIES = totalLen / loadFactor;
    const size_t HASH_ENTRIES = 2048;
    int *sequences = new int[totalLen];
    readDataFromFile(pathToData, sequences, numOfSequences, seqLength);

    int *dev_sequences;
    cudaMalloc(&dev_sequences, totalLen * sizeof(int));
    cudaMemcpy(dev_sequences, sequences, totalLen * sizeof(int), cudaMemcpyHostToDevice);

    size_t *dev_prefixes;
    size_t *dev_suffixes;
    Triple<size_t, size_t, int> *dev_matchingHashes;
    Triple<size_t, size_t, int> *dev_ownHashes;
    cudaMalloc(&dev_prefixes, totalLen * sizeof(size_t));
    cudaMalloc(&dev_suffixes, totalLen * sizeof(size_t));
    cudaMalloc(&dev_matchingHashes, totalLen * sizeof(Triple<size_t, size_t, int>));
    cudaMalloc(&dev_ownHashes, totalLen * sizeof(Triple<size_t, size_t, int>));

    getHashes<<<blocksNum(numOfSequences, 256), 256>>>(dev_sequences, numOfSequences, seqLength, dev_prefixes, dev_suffixes, dev_matchingHashes, dev_ownHashes);

    HashTableSC<Triple<size_t, size_t, int>, int, TripleHash<size_t, size_t, int>> table(HASH_ENTRIES, totalLen);
    CudaLock *lock = new CudaLock[HASH_ENTRIES];
    CudaLock *dev_lock;

    cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(CudaLock));
    cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(CudaLock), cudaMemcpyHostToDevice);

    addToTable<<<blocksNum(totalLen, 256), 256>>>(dev_ownHashes, table, dev_lock, seqLength);

    findHammingOnePairs<<<blocksNum(totalLen, 256), 256>>>(table, dev_matchingHashes, numOfSequences, seqLength);

    delete[] sequences;
    delete[] lock;
    cudaFree(dev_prefixes);
    cudaFree(dev_suffixes);
    cudaFree(dev_matchingHashes);
    cudaFree(dev_ownHashes);
    cudaFree(dev_lock);
    table.freeTable();
}