#include <string>
#include "Tuple.h"
#include "utils.h"
#include "Hash.h"
#include "HashTableCpu.h"

constexpr size_t P = 31;
constexpr size_t M1 = 4757501900887;
constexpr size_t M2 = 3348549226339;

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
        hashValue = (hashValue + (size_t(sequence[i]) + 1) * pPow) % M1;
        pPow = (pPow * P) % M1;

        prefixHashes[i] = hashValue;
    }
}

__device__ void computeSuffixHashes(int *sequence, int seqLength, size_t *suffixHashes)
{
    size_t hashValue = 0;
    size_t pPow = 1;

    for (int i = 0; i < seqLength; i++)
    {
        hashValue = (hashValue + (size_t(sequence[seqLength - i - 1]) + 1) * pPow) % M2;
        pPow = (pPow * P) % M2;

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

template <class Key, class T, class Hash>
__global__ void copyEntriesToDevice(TableCpu<Key, T, Hash> table, Entry<Key, T> **dev_entries, Entry<Key, T> *dev_pool)
{
    int entryId = threadIdx.x + blockDim.x * blockIdx.x; // one thread per entry
    if (entryId > table.count - 1)
        return;

    if (dev_entries[entryId] != nullptr)
    {
        dev_entries[entryId] = (Entry<Key, T> *)((size_t)dev_entries[entryId] - (size_t)table.pool + (size_t)dev_pool);
    }
}

template <class Key, class T, class Hash>
__global__ void copyElementsToDevice(TableCpu<Key, T, Hash> table, Entry<Key, T> *dev_pool)
{
    int elementId = threadIdx.x + blockDim.x * blockIdx.x; // one thread per element
    if (elementId > table.elements - 1)
        return;
    if (dev_pool[elementId].next != nullptr)
    {
        dev_pool[elementId].next = (Entry<Key, T> *)((size_t)dev_pool[elementId].next - (size_t)table.pool + (size_t)dev_pool);
    }
}

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

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    size_t totalLen = numOfSequences * seqLength;
    const size_t HASH_ENTRIES = totalLen / loadFactor;
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

    // copy entries to device
    Entry<Triple<size_t, size_t, int>, int> **dev_entries;
    Entry<Triple<size_t, size_t, int>, int> *dev_pool;
    cudaMalloc(&dev_entries, table.count * sizeof(Entry<Triple<size_t, size_t, int>, int> *));
    cudaMemset(dev_entries, 0, table.count * sizeof(Entry<Triple<size_t, size_t, int>, int> *));
    cudaMalloc(&dev_pool, table.elements * sizeof(Entry<Triple<size_t, size_t, int>, int>));

    cudaMemcpy(dev_entries, table.entries, table.count * sizeof(Entry<Triple<size_t, size_t, int>, int> *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pool, table.pool, table.elements * sizeof(Entry<Triple<size_t, size_t, int>, int>), cudaMemcpyHostToDevice);

    copyEntriesToDevice<<<blocksNum(HASH_ENTRIES, 256), 256>>>(table, dev_entries, dev_pool);
    copyElementsToDevice<<<blocksNum(table.elements, 256), 256>>>(table, dev_pool);

    findHammingOnePairs<<<blocksNum(totalLen, 256), 256>>>(dev_entries, HASH_ENTRIES, TripleHash<size_t, size_t, int>(), dev_matchingHashes, numOfSequences, seqLength);
    // Triple<size_t, size_t, int> *matchingHashes = new Triple<size_t, size_t, int>[totalLen];
    // cudaMemcpy(matchingHashes, dev_matchingHashes, totalLen * sizeof(Triple<size_t, size_t, int>), cudaMemcpyDeviceToHost);
    // findHammingOnePairs(table, matchingHashes, numOfSequences, seqLength);
}