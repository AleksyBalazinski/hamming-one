#pragma once

#include <functional>
#include "CudaLock.h"
#include "CudaAllocator.h"

template <class Key, class T>
struct Entry
{
    Key key;
    T value;
    Entry<Key, T> *next;
};

template <class Key, class T, class Hash, template <class> class Allocator>
struct Table
{
    Hash hasher;
    Allocator<Entry<Key, T>> entryAllocator;
    Allocator<Entry<Key, T> *> entryPtrAllocator;

    size_t count;
    size_t elements;
    Entry<Key, T> **entries;
    Entry<Key, T> *pool;

    Table() {}

    Table(int nb_entries, int nb_elements)
    {
        count = nb_entries;
        elements = nb_elements;
        entries = entryPtrAllocator.allocate(nb_entries);
        pool = entryAllocator.allocate(nb_elements);
    }
};

template <class Key, class T, class Hash, template <class> class Allocator>
void freeTable(Table<Key, T, Hash, Allocator> &table)
{
    table.entryAllocator.deallocate(table.pool, table.elements);
    table.entryPtrAllocator.deallocate(table.entries, table.count);
}

template <class Key, class T, class Hash, template <class> class Allocator, template <class> class HostAllocator>
void copyTableToHost(const Table<Key, T, Hash, Allocator> &table, Table<Key, T, Hash, HostAllocator> &hostTable)
{
    cudaMemcpy(hostTable.entries, table.entries, table.count * sizeof(Entry<Key, T> *), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostTable.pool, table.pool, table.elements * sizeof(Entry<Key, T>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < table.elements; i++)
    {
        if (hostTable.pool[i].next != NULL)
            hostTable.pool[i].next = (Entry<Key, T> *)((size_t)hostTable.pool[i].next - (size_t)table.pool + (size_t)hostTable.pool);
    }
    for (int i = 0; i < table.count; i++)
    {
        if (hostTable.entries[i] != NULL)
            hostTable.entries[i] = (Entry<Key, T> *)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool);
    }
}

template <class Key, class T, class Hash, template <class> class Allocator>
__global__ void addToTable(unsigned int *keys, float *values, Table<Key, T, Hash, Allocator> table, CudaLock *lock)
{
    int tid = threadIdx.x + blockIdx.x * gridDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < table.elements)
    {
        unsigned int key = keys[tid];
        size_t hashValue = table.hasher(key) % table.count;
        for (int i = 0; i < 32; i++)
        {
            if ((tid % 32) == i)
            {
                Entry<Key, T> *location = &(table.pool[tid]);
                location->key = key;
                location->value = values[tid];
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                lock[hashValue].unlock();
            }
        }
        tid += stride;
    }
}