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
class Table;

template <class Key, class T, class DevHash, template <class> class DevAllocator, class HostHash, template <class> class HostAllocator>
void copyTableToHost(const Table<Key, T, DevHash, DevAllocator> &table, Table<Key, T, HostHash, HostAllocator> &hostTable);

template <class Key, class T, class Hash, template <class> class Allocator>
__global__ void addToTable(Key *keys, T *values, Table<Key, T, Hash, Allocator> table, CudaLock *lock);

template <class Key, class T, class Hash, template <class> class Allocator>
class Table
{
    // template <class K, class U, class DevHash, template <class> class DevAllocator, class HostHash, template <class> class HostAllocator>
    // friend void copyTableToHost<Key, T, DevHash, DevAllocator, HostHash, HostAllocator>(const Table<Key, T, DevHash, DevAllocator> &table, Table<Key, T, HostHash, HostAllocator> &hostTable);

    friend __global__ void addToTable<>(Key *keys, T *values, Table<Key, T, Hash, Allocator> table, CudaLock *lock);

public: // for now
    Hash hasher;
    Allocator<Entry<Key, T>> entryAllocator;
    Allocator<Entry<Key, T> *> entryPtrAllocator;

    size_t count;
    size_t elements;
    Entry<Key, T> **entries;
    Entry<Key, T> *pool;

public:
    Table() {}

    Table(int nb_entries, int nb_elements)
    {
        count = nb_entries;
        elements = nb_elements;
        entries = entryPtrAllocator.allocate(nb_entries);
        pool = entryAllocator.allocate(nb_elements);
    }

    Entry<Key, T> **getEntries() { return entries; }

    __host__ __device__ bool find(Key key, T &out)
    {
        size_t hashValue = hasher(key) % count;
        Entry<Key, T> *cur = entries[hashValue];
        while (cur != nullptr && cur->key != key)
        {
            cur = cur->next;
        }

        if (cur == nullptr)
            return false;

        out = cur->value;
        return true;
    }

    void freeTable()
    {
        entryAllocator.deallocate(pool, elements);
        entryPtrAllocator.deallocate(entries, count);
    }
};

template <class Key, class T, class DevHash, template <class> class DevAllocator, class HostHash, template <class> class HostAllocator>
void copyTableToHost(const Table<Key, T, DevHash, DevAllocator> &table, Table<Key, T, HostHash, HostAllocator> &hostTable)
{
    cudaMemcpy(hostTable.entries, table.entries, table.count * sizeof(Entry<Key, T> *), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostTable.pool, table.pool, table.elements * sizeof(Entry<Key, T>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < table.elements; i++)
    {
        if (hostTable.pool[i].next != nullptr)
            hostTable.pool[i].next = (Entry<Key, T> *)((size_t)hostTable.pool[i].next - (size_t)table.pool + (size_t)hostTable.pool);
    }
    for (int i = 0; i < table.count; i++)
    {
        if (hostTable.entries[i] != nullptr)
            hostTable.entries[i] = (Entry<Key, T> *)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool);
    }
}

template <class Key, class T, class Hash, template <class> class Allocator>
__global__ void addToTable(Key *keys, T *values, Table<Key, T, Hash, Allocator> table, CudaLock *lock)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < table.elements)
    {
        Key key = keys[tid];
        T value = values[tid];
        size_t hashValue = table.hasher(key) % table.count;
        for (int i = 0; i < 32; i++)
        {
            if (i == tid % 32)
            {
                Entry<Key, T> *location = &(table.pool[tid]);
                location->key = key;
                location->value = value;
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                lock[hashValue].unlock();
            }
        }
        tid += stride;
    }
}