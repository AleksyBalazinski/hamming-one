#include <stdio.h>
#include "CudaLock.h"
#include "HashTable.h"

constexpr int imin(int a, int b)
{
    return a < b ? a : b;
}

using Key = unsigned int;
using Val = float;

constexpr size_t N = 1024 * 1024;
constexpr size_t HASH_ENTRIES = 10;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

struct IntHash
{
    __host__ __device__ size_t operator()(int x)
    {
        return x;
    }
};

void hash_table_test()
{
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

    Table<Key, Val, IntHash, CudaAllocator> table(HASH_ENTRIES, N);

    CudaLock lock[HASH_ENTRIES];
    CudaLock *dev_lock;

    cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(CudaLock));
    cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(CudaLock), cudaMemcpyHostToDevice);

    add_to_table<<<blocksPerGrid, threadsPerBlock>>>(dev_keys, dev_values, table, dev_lock);

    Table<Key, Val, IntHash, std::allocator> hostTable(HASH_ENTRIES, N);

    copy_table_to_host(table, hostTable);
    Entry<Key, Val> *cur = hostTable.entries[0];
    for (int i = 0; i < 10; i++)
    {
        printf("current key: %d, current value: %f\n", cur->key, cur->value);
        cur = cur->next;
    }

    freeTable(table);
    freeTable(hostTable);
    cudaFree(dev_lock);
    cudaFree(dev_keys);
    cudaFree(dev_values);
    free(buffer);
}

int main()
{
    hash_table_test();
}