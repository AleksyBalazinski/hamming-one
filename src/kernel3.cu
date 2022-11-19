#include <string>
#include "Tuple.h"
#include "Utils.h"
#include "Hash.h"
#include "HashTableCpu.h"

constexpr size_t P = 31;
constexpr size_t M = 1e9 + 9;

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

// template <class Key, class T, class Hash>
// __global__ void copyEntriesToDevice(TableCpu<Key, T, Hash> table, Entry<Key, T> **dev_entries, Entry<Key, T> *dev_pool, int totalLen)
// {
//     for (int i = 0; i < table.count; i++)
//     {
//         if (dev_entries[i] != nullptr)
//         {
//             dev_entries[i] = (Entry<Key, T> *)((size_t)dev_entries[i] - (size_t)table.pool + (size_t)dev_pool);
//         }
//     }
//     for (int i = 0; i < table.elements; i++)
//     {
//         if (dev_pool[i].next != nullptr)
//         {
//             dev_pool[i].next = (Entry<Key, T> *)((size_t)dev_pool[i].next - (size_t)table.pool + (size_t)dev_pool);
//         }
//     }
// }

template <class Key, class T, class Hash>
__global__ void findHammingOnePairs(Entry<Key, T> **entries, int count, Hash hasher, Triple<size_t, size_t, int> *matchingHashes, int numOfSequences, int seqLength)
{
    int elemId = threadIdx.x + blockDim.x * blockIdx.x;
    if (elemId > seqLength * numOfSequences - 1)
        return;

    size_t hashValue = hasher(matchingHashes[elemId]) % count;
    Entry<Key, T> *cur = entries[hashValue];
    while (cur != nullptr)
    {
        if (cur->key == matchingHashes[elemId])
        {
            printf("%d %d\n", cur->value, elemId / seqLength);
        }
        cur = cur->next;
    }
}

template <class Key, class T, class Hash>
void findHammingOnePairs(const TableCpu<Key, T, Hash> &table, Triple<size_t, size_t, int> *matchingHashes, int numOfSequences, int seqLength)
{
    for (int elemId = 0; elemId < numOfSequences * seqLength; elemId++)
    {
        size_t hashValue = table.hasher(matchingHashes[elemId]) % table.count;
        Entry<Key, T> *cur = table.entries[hashValue];
        while (cur != nullptr)
        {
            if (cur->key == matchingHashes[elemId])
            {
                printf("%d %d\n", cur->value, elemId / seqLength);
            }
            cur = cur->next;
        }
    }
}

int main(int argc, char **argv)
{
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, PRINTF_FIFO_SIZE);

    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);
    const double loadFactor = std::stod(argv[3]);

    // const int HASH_ENTRIES = std::stoi(argv[3]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    size_t totalLen = numOfSequences * seqLength;
    const int HASH_ENTRIES = totalLen / loadFactor;
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

    TableCpu<Triple<size_t, size_t, int>, int, TripleHash<size_t, size_t, int>> table(HASH_ENTRIES, totalLen);

    Triple<size_t, size_t, int> *ownHashes = new Triple<size_t, size_t, int>[totalLen];
    cudaMemcpy(ownHashes, dev_ownHashes, totalLen * sizeof(Triple<size_t, size_t, int>), cudaMemcpyDeviceToHost);

    addToTable(ownHashes, table, seqLength, numOfSequences);

#ifdef DEBUG
    for (int i = 0; i < HASH_ENTRIES; i++)
    {
        printf("bucket %d:", i);
        Entry<Triple<size_t, size_t, int>, int> *cur = table.entries[i];
        while (cur != nullptr)
        {
            printf("(%llu %llu %d) -> %d, ", cur->key.item1, cur->key.item2, cur->key.item3, cur->value);
            cur = cur->next;
        }
        printf("\n");
    }
#endif
    // copy entries to device
    // Entry<Triple<size_t, size_t, int>, int> **dev_entries;
    // Entry<Triple<size_t, size_t, int>, int> *dev_pool;
    // cudaMalloc(&dev_entries, table.count * sizeof(Entry<Triple<size_t, size_t, int>, int> *));
    // cudaMalloc(&dev_pool, table.elements * sizeof(Entry<Triple<size_t, size_t, int>, int>));

    // cudaMemcpy(dev_entries, table.entries, table.count * sizeof(Entry<Triple<size_t, size_t, int>, int> *), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_pool, table.pool, table.elements * sizeof(Entry<Triple<size_t, size_t, int>, int>), cudaMemcpyHostToDevice);

    // copyEntriesToDevice<<<1, 1>>>(table, dev_entries, dev_pool, totalLen);

    // findHammingOnePairs<<<blocksNum(totalLen, 256), 256>>>(dev_entries, HASH_ENTRIES, TripleHash<size_t, size_t, int>(), dev_matchingHashes, numOfSequences, seqLength);
    Triple<size_t, size_t, int> *matchingHashes = new Triple<size_t, size_t, int>[totalLen];
    cudaMemcpy(matchingHashes, dev_matchingHashes, totalLen * sizeof(Triple<size_t, size_t, int>), cudaMemcpyDeviceToHost);
    findHammingOnePairs(table, matchingHashes, numOfSequences, seqLength);
}