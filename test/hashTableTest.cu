#include <stdio.h>
#include "CudaLock.h"
#include "HashTable.h"
#include "Hash.h"
#include "Tuple.h"

constexpr int imin(int a, int b)
{
    return a < b ? a : b;
}

// struct IntegralHash
// {
//     __host__ __device__ size_t operator()(int x)
//     {
//         return x;
//     }
// };

void hashTableTest1()
{
    using Key = unsigned int;
    using Val = float;

    constexpr size_t N = 50;
    constexpr size_t HASH_ENTRIES = 10;
    constexpr int threadsPerBlock = 32;
    constexpr int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

    Key *buffer = (Key *)malloc(N * sizeof(Key));
    for (int i = 0; i < N; i++)
        buffer[i] = (Key)i;
    Val *values;
    values = (Val *)malloc(N * sizeof(Val));
    for (int i = 0; i < N; i++)
        values[i] = (Val)i * 0.01;

    Key *dev_keys;
    Val *dev_values;

    cudaMalloc((void **)&dev_keys, N * sizeof(Key));
    cudaMalloc((void **)&dev_values, N * sizeof(Val));
    cudaMemcpy(dev_keys, buffer, N * sizeof(Key), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values, values, N * sizeof(Val), cudaMemcpyHostToDevice);

    HashTableSC<Key, Val, IntegralHash<Key>, CudaAllocator> table(HASH_ENTRIES, N);

    CudaLock lock[HASH_ENTRIES];
    CudaLock *dev_lock;

    cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(CudaLock));
    cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(CudaLock), cudaMemcpyHostToDevice);

    addToTable<<<blocksPerGrid, threadsPerBlock>>>(dev_keys, dev_values, table, dev_lock);

    HashTableSC<Key, Val, IntegralHash<Key>, std::allocator> hostTable(HASH_ENTRIES, N);

    copyTableToHost(table, hostTable);

    for (int i = 0; i < HASH_ENTRIES; i++)
    {
        printf("bucket %d: ", i);
        auto cur = hostTable.getEntries()[i];
        while (cur != nullptr)
        {
            printf("(%d -> %f) ", cur->key, cur->value);
            cur = cur->next;
        }
        printf("\n");
    }

    table.freeTable();
    hostTable.freeTable();
    cudaFree(dev_lock);
    cudaFree(dev_keys);
    cudaFree(dev_values);
    free(buffer);
}

__global__ void parallelQuery(HashTableSC<Pair<int, int>, int, PairHash<int, int>, CudaAllocator> table, Pair<int, int> *keys, int keysCnt)
{
    int tid = threadIdx.x + blockIdx.x * gridDim.x;
    if (tid >= keysCnt)
        return;
    Pair<int, int> key = keys[tid];
    int value;
    bool found = table.find(key, value);

    if (found == false)
    {
        printf("Key (%d, %d) was not found\n", key.first, key.second);
        return;
    }

    printf("Key (%d, %d) maps to %d\n", key.first, key.second, value);
}

void sequentialQuery(HashTableSC<Pair<int, int>, int, PairHash<int, int>, std::allocator> table, Pair<int, int> *keys, int keysCnt)
{
    for (int i = 0; i < keysCnt; i++)
    {
        Pair<int, int> key = keys[i];
        int value;
        bool found = table.find(key, value);

        if (!found)
            printf("Key (%d, %d) was not found\n", key.first, key.second);
        else
            printf("Key (%d, %d) maps to %d\n", key.first, key.second, value);
    }
}

void hashTableTest2()
{
    constexpr size_t N = 50;
    constexpr size_t HASH_ENTRIES = 10;
    constexpr int threadsPerBlock = 32;
    constexpr int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

    Pair<int, int> keys[N];
    for (int i = 0; i < N; i++)
    {
        keys[i].first = i;
        keys[i].second = i + 1;
    }

    int values[N];
    for (int i = 0; i < N; i++)
    {
        values[i] = 2 * i + 1;
    }

    Pair<int, int> *dev_keys;
    int *dev_values;

    cudaMalloc((void **)&dev_keys, N * sizeof(Pair<int, int>));
    cudaMalloc((void **)&dev_values, N * sizeof(int));
    cudaMemcpy(dev_keys, keys, N * sizeof(Pair<int, int>), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values, values, N * sizeof(int), cudaMemcpyHostToDevice);

    HashTableSC<Pair<int, int>, int, PairHash<int, int>, CudaAllocator> table(HASH_ENTRIES, N);

    CudaLock lock[HASH_ENTRIES];
    CudaLock *dev_lock;

    cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(CudaLock));
    cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(CudaLock), cudaMemcpyHostToDevice);

    addToTable<<<blocksPerGrid, threadsPerBlock>>>(dev_keys, dev_values, table, dev_lock);

    // it should be possible to query the hash table in parallel...
    printf("*** parallel query ***\n");
    parallelQuery<<<blocksPerGrid, threadsPerBlock>>>(table, dev_keys, N);

    // ... as well as sequentially
    HashTableSC<Pair<int, int>, int, PairHash<int, int>, std::allocator> hostTable(HASH_ENTRIES, N);
    copyTableToHost(table, hostTable);
    printf("*** sequential query ***\n");
    sequentialQuery(hostTable, keys, N);

    table.freeTable();
    hostTable.freeTable();
    cudaFree(dev_lock);
    cudaFree(dev_keys);
    cudaFree(dev_values);
}

int main()
{
    hashTableTest1();
    // hashTableTest2();
}