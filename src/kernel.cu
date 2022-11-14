#define noDEBUG

#include <string>
#include <iostream>

#include "Array.h"
#include "Utils.h"
#include "CudaAllocator.h"
#include "Tuple.h"
#include "HashTable.h"
#include "Hash.h"

constexpr size_t P = 31;
constexpr size_t M = 1e9 + 9;

#define HASH_ENTRIES 10

int blocksNum(int threads, int threadsPerBlock)
{
    return (threads + threadsPerBlock - 1) / threadsPerBlock;
}

__device__ void computePrefixHashes(int *sequence, int seqLength, size_t *prefixHashes)
{
    size_t hashValue = 0;
    size_t pPow = 1;

    for (int i = 0; i < seqLength; i++)
    {
        hashValue = (hashValue + (size_t(sequence[i]) + 1) * pPow) % M;
        pPow = (pPow * P) % M;

        prefixHashes[i] = hashValue;
    }
}

__device__ void computeSuffixHashes(int *sequence, int seqLength, size_t *suffixHashes)
{
    size_t hashValue = 0;
    size_t pPow = 1;

    for (int i = 0; i < seqLength; i++)
    {
        hashValue = (hashValue + (size_t(sequence[seqLength - i - 1]) + 1) * pPow) % M;
        pPow = (pPow * P) % M;

        suffixHashes[i] = hashValue;
    }
}

__global__ void getHashes(int *sequences, int numOfSequences, int seqLength, size_t *prefixes, size_t *suffixes,
                          Triple<size_t, size_t, int> *matchingHashes, Triple<size_t, size_t, int> *ownHasbes)
{
    int seqId = threadIdx.x + blockDim.x * blockIdx.x;
    if (seqId > numOfSequences)
        return;

    int offset = seqId * seqLength;
    int *sequence = sequences + offset;
    size_t *curPrefixes = prefixes + offset;
    size_t *curSuffixes = suffixes + offset;
    Triple<size_t, size_t, int> *curMatchingHashes = matchingHashes + offset;
    Triple<size_t, size_t, int> *curOwnHashes = ownHasbes + offset;
    computePrefixHashes(sequence, seqLength, curPrefixes);
    computeSuffixHashes(sequence, seqLength, curSuffixes);

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
__global__ void findHammingOnePairs(Table<Key, T, Hash, CudaAllocator> table, Triple<size_t, size_t, int> *matchingHashes, int numOfSequences, int seqLength)
{
    int elemId = threadIdx.x + blockDim.x * blockIdx.x;
    if (elemId > seqLength * numOfSequences)
        return;
    // int seqId;
    //  bool found = table.find(matchingHashes[elemId], seqId);
    Entry<Key, T> *cur = table.getBucket(matchingHashes[elemId]);
    while (cur != nullptr)
    {
        if (cur->key == matchingHashes[elemId])
        {
            printf("%d %d\n", cur->value, elemId / seqLength);
        }
        cur = cur->next;
    }
    // if (found)
    // {
    //     printf("%d %d\n", seqId, elemId / seqLength);
    // }
}

int main(int argc, char **argv)
{
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);
    std::cout << numOfSequences << ' ' << seqLength << '\n'; // TEST
    size_t totalLen = numOfSequences * seqLength;

    int *sequences = new int[totalLen];
    readDataFromFile(pathToData, sequences, numOfSequences, seqLength);
    printSequences(sequences, numOfSequences, seqLength); // TEST

    int blocks = blocksNum(numOfSequences, 4);

    Triple<size_t, size_t, int> *matchingHashes = new Triple<size_t, size_t, int>[totalLen];
    Triple<size_t, size_t, int> *ownHashes = new Triple<size_t, size_t, int>[totalLen];

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

    getHashes<<<blocks, 4>>>(dev_sequences, numOfSequences, seqLength, dev_prefixes, dev_suffixes, dev_matchingHashes, dev_ownHashes);

    cudaMemcpy(matchingHashes, dev_matchingHashes, totalLen * sizeof(Triple<size_t, size_t, int>), cudaMemcpyDeviceToHost);
    cudaMemcpy(ownHashes, dev_ownHashes, totalLen * sizeof(Triple<size_t, size_t, int>), cudaMemcpyDeviceToHost);
#ifdef DEBUG
    std::cout << "matchingHashes:\n";
    for (int seq = 0; seq < numOfSequences; seq++)
    {
        std::cout << seq << ": ";
        for (int j = 0; j < seqLength; j++)
        {
            std::cout << "{" << matchingHashes[seq * seqLength + j].item1 << ", "
                      << matchingHashes[seq * seqLength + j].item2 << ", "
                      << matchingHashes[seq * seqLength + j].item3 << "} ";
        }
        std::cout << "\n";
    }

    std::cout << "ownHashes:\n";
    for (int seq = 0; seq < numOfSequences; seq++)
    {
        std::cout << seq << ": ";
        for (int j = 0; j < seqLength; j++)
        {
            std::cout << "{" << ownHashes[seq * seqLength + j].item1 << ", "
                      << ownHashes[seq * seqLength + j].item2 << ", "
                      << ownHashes[seq * seqLength + j].item3 << "} ";
        }
        std::cout << "\n";
    }
#endif // DEBUG

    Table<Triple<size_t, size_t, int>, int, TripleHash<size_t, size_t, int>, CudaAllocator> table(HASH_ENTRIES, totalLen);
    CudaLock lock[HASH_ENTRIES];
    CudaLock *dev_lock;

    cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(CudaLock));
    cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(CudaLock), cudaMemcpyHostToDevice);

    int *values = new int[totalLen];
    for (int i = 0; i < numOfSequences; i++)
    {
        for (int j = 0; j < seqLength; j++)
            values[i * seqLength + j] = i;
    }
    int *dev_values;
    cudaMalloc(&dev_values, totalLen * sizeof(int));
    cudaMemcpy(dev_values, values, totalLen * sizeof(int), cudaMemcpyHostToDevice);

    addToTable<<<(totalLen + 32 - 1) / 32, 32>>>(dev_ownHashes, dev_values, table, dev_lock);

    Table<Triple<size_t, size_t, int>, int, TripleHash<size_t, size_t, int>, std::allocator> hostTable(HASH_ENTRIES, totalLen);

    copyTableToHost(table, hostTable);
#ifdef DEBUG
    for (int i = 0; i < HASH_ENTRIES; i++)
    {
        printf("bucket %d: ", i);
        auto cur = hostTable.getEntries()[i];
        while (cur != nullptr)
        {
            printf("({%llu %llu %d} -> %d) ", cur->key.item1, cur->key.item2, cur->key.item3, cur->value);
            cur = cur->next;
        }
        printf("\n");
    }
#endif // DEBUG

    findHammingOnePairs<<<(totalLen + 32 - 1) / 32, 32>>>(table, dev_matchingHashes, numOfSequences, seqLength);

    delete[] sequences;
    delete[] matchingHashes;
    delete[] ownHashes;
    delete[] values;
    cudaFree(dev_prefixes);
    cudaFree(dev_suffixes);
    cudaFree(dev_matchingHashes);
    cudaFree(dev_ownHashes);
    cudaFree(dev_values);

    table.freeTable();
    hostTable.freeTable();
}