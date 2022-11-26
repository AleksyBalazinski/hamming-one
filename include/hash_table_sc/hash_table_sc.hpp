#pragma once

#include <functional>
#include "detail/cuda_lock.hpp"
#include "detail/entry.hpp"
#include "common/cuda_allocator.hpp"
#include "common/cuda_hash.cuh"

template <class Key, class T, class Hash, class Allocator = CudaAllocator<char>, class Lock = CudaLock>
struct HashTableSC
{
    using value_type = Entry<Key, T>;
    using key_type = Key;
    using mapped_type = T;
    using allocator_type = Allocator;
    using hasher = Hash;
    using size_type = std::size_t;
    using lock_type = Lock;

    using pool_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<value_type>;
    using entry_ptr_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<value_type *>;
    using lock_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<lock_type>;

    hasher hf;
    pool_allocator_type entryAllocator;
    entry_ptr_allocator_type entryPtrAllocator;
    lock_allocator_type lock_allocator_;

    size_t count;
    size_t elements;
    value_type **entries;
    value_type *pool;
    lock_type *locks;

    HashTableSC() {}

    HashTableSC(int nb_entries, int nb_elements);

    Entry<Key, T> **getEntries() { return entries; }

    __host__ __device__ bool find(Key key, T &out)
    {
        size_t hashValue = hf(key) % count;
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

    __host__ __device__ Entry<Key, T> *getBucket(Key key)
    {
        size_t hashValue = hf(key) % count;

        return entries[hashValue];
    }

    void freeTable()
    {
        entryAllocator.deallocate(pool, elements);
        entryPtrAllocator.deallocate(entries, count);
    }

    // NEW INTERFACE
    template <typename InputIt>
    bool insert(InputIt first, InputIt last);

    template <typename InputIt, typename OutputIt>
    void find(InputIt first, InputIt last, Output output_begin);

    __device__ bool insert(const value_type &pair);

    __device__ mapped_type find(key_type const &key);
};

template <class Key, class T, class DevHash, class DevAllocator, class HostHash, class HostAllocator>
void copyTableToHost(const HashTableSC<Key, T, DevHash, DevAllocator> &table, HashTableSC<Key, T, HostHash, HostAllocator> &hostTable)
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

template <class Key, class T, class Hash, class Allocator>
__global__ void addToTable(Key *keys, T *values, HashTableSC<Key, T, Hash, Allocator> table, CudaLock *lock)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < table.elements)
    {
        Key key = keys[tid];
        T value = values[tid];
        size_t hashValue = table.hf(key) % table.count;
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

template <class Key, class Hash, class Allocator>
__global__ void addToTable(Key *keys, HashTableSC<Key, int, Hash, Allocator> table, CudaLock *lock, int seqLength)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (tid > table.elements - 1)
        return;
    while (tid < table.elements)
    {
        Key key = keys[tid];
        int value = tid / seqLength;
        size_t hashValue = table.hf(key) % table.count;
        for (int i = 0; i < 32; i++)
        {
            if (i == tid % 32)
            {
                Entry<Key, int> *location = &(table.pool[tid]);
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