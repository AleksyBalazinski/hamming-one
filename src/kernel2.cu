#include <string>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <iostream>

#include "utils.h"

constexpr size_t P = 31;
constexpr size_t M = 1e9 + 9;

struct Slot
{
    size_t prefix;
    size_t suffix;
    int erased;
    int seqId;

    __host__ __device__ Slot(size_t prefix, size_t suffix, int erased, int seqId)
        : prefix(prefix), suffix(suffix), erased(erased), seqId(seqId)
    {
    }

    __host__ __device__ Slot() {}
};

__host__ __device__ bool operator<(const Slot &a, const Slot &b)
{
    if (a.prefix != b.prefix)
        return a.prefix < b.prefix;
    if (a.suffix != b.suffix)
        return a.suffix < b.suffix;
    return a.erased < b.erased;
}

__host__ __device__ bool operator>(const Slot &a, const Slot &b)
{
    return (b < a);
}

__host__ __device__ bool operator==(const Slot &a, const Slot &b)
{
    return a.prefix == b.prefix && a.suffix == b.suffix && a.erased == b.erased;
}

__host__ __device__ bool operator!=(const Slot &a, const Slot &b)
{
    return !(a == b);
}

#define imin(a, b) (b)
#define PRINTF_FIFO_SIZE (long long int)1e15

int blocksNum(int threads, int threadsPerBlock)
{
    return imin(32, (threads + threadsPerBlock - 1) / threadsPerBlock);
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
                          Slot *matchingHashes, Slot *ownHasbes)
{
    int seqId = threadIdx.x + blockDim.x * blockIdx.x;
    if (seqId > numOfSequences - 1)
        return;

    int offset = seqId * seqLength;
    int *sequence = sequences + offset;
    size_t *curPrefixes = prefixes + offset;
    size_t *curSuffixes = suffixes + offset;
    Slot *curMatchingHashes = matchingHashes + offset;
    Slot *curOwnHashes = ownHasbes + offset;
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

        curMatchingHashes[i] = Slot(prefixHash, suffixHash, erased == 0 ? 1 : 0, seqId);
        curOwnHashes[i] = Slot(prefixHash, suffixHash, erased, seqId);
    }
}

__global__ void findHammingOnePairs(Slot *ownHashes, Slot *matchingHashes, int numOfSequences, int seqLength)
{
    int elemId = threadIdx.x + blockDim.x * blockIdx.x;
    int totalLen = numOfSequences * seqLength;
    if (elemId > totalLen - 1)
        return;

    Slot curMatchingHash = matchingHashes[elemId];
    int l = 0;
    int r = totalLen - 1;
    int begin = -1;
    while (l <= r)
    {
        int mid = (r - l) / 2 + l;
        if (ownHashes[mid] > curMatchingHash)
            r = mid - 1;
        else if (ownHashes[mid] == curMatchingHash)
        {
            begin = mid;
            r = mid - 1;
        }
        else
            l = mid + 1;
    }

    int end = -1;
    l = 0;
    r = totalLen - 1;
    while (l <= r)
    {
        int mid = (r - l) / 2 + l;
        if (ownHashes[mid] > curMatchingHash)
            r = mid - 1;
        else if (ownHashes[mid] == curMatchingHash)
        {
            end = mid;
            l = mid + 1;
        }
        else
            l = mid + 1;
    }

    if (begin == -1 || end == -1)
        return;

    for (int i = begin; i <= end; i++)
    {
        printf("%d %d\n", ownHashes[i].seqId, elemId / seqLength);
    }
}

int main(int argc, char **argv)
{
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, PRINTF_FIFO_SIZE);

    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    size_t totalLen = numOfSequences * seqLength;
    int *sequences = new int[totalLen];

    readDataFromFile(pathToData, sequences, numOfSequences, seqLength);

    int *dev_sequences;
    cudaMalloc(&dev_sequences, totalLen * sizeof(int));
    cudaMemcpy(dev_sequences, sequences, totalLen * sizeof(int), cudaMemcpyHostToDevice);

    size_t *dev_prefixes;
    size_t *dev_suffixes;
    Slot *dev_matchingHashes;
    Slot *dev_ownHashes;
    cudaMalloc(&dev_prefixes, totalLen * sizeof(size_t));
    cudaMalloc(&dev_suffixes, totalLen * sizeof(size_t));
    cudaMalloc(&dev_matchingHashes, totalLen * sizeof(Slot));
    cudaMalloc(&dev_ownHashes, totalLen * sizeof(Slot));

    getHashes<<<blocksNum(numOfSequences, 256), 256>>>(dev_sequences, numOfSequences, seqLength, dev_prefixes, dev_suffixes, dev_matchingHashes, dev_ownHashes);
    thrust::sort(thrust::device, dev_ownHashes, dev_ownHashes + totalLen);

    findHammingOnePairs<<<blocksNum(totalLen, 256), 256>>>(dev_ownHashes, dev_matchingHashes, numOfSequences, seqLength);

    delete[] sequences;
    cudaFree(dev_prefixes);
    cudaFree(dev_suffixes);
    cudaFree(dev_matchingHashes);
    cudaFree(dev_ownHashes);
}