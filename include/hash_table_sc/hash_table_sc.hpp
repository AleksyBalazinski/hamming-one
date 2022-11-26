#pragma once

#include <functional>
#include "common/cuda_allocator.hpp"
#include "common/cuda_hash.cuh"
#include "detail/cuda_lock.cuh"
#include "detail/entry.hpp"
#include "common/pair.cuh"
#include "common/cuda_hash.cuh"

template <class Key,
          class T,
          class Hash = CudaHash<Key>,
          class Allocator = CudaAllocator<Key>,
          class Lock = CudaLock>
struct HashTableSC {
    using value_type = Pair<Key, T>;
    using key_type = Key;
    using mapped_type = T;
    using allocator_type = Allocator;
    using hasher = Hash;
    using size_type = std::size_t;
    using lock_type = Lock;
    using entry_type = Entry<Key, T>;

    using pool_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<entry_type>;
    using entry_ptr_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<entry_type*>;
    using lock_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<lock_type>;

    HashTableSC() {}

    HashTableSC(int nb_entries_, int nb_elements_, T sentinel_value);

    ~HashTableSC();

    Entry<Key, T>** getentries_() { return entries_; }

    __host__ __device__ Entry<Key, T>* getBucket(Key key) {
        size_t hashValue = hf_(key) % count_;

        return entries_[hashValue];
    }

    void freeTable() {
        entry_allocator_.deallocate(pool_, elements_);
        entry_ptr_allocator_.deallocate(entries_, count_);
    }

    template <typename InputIt>
    bool insert(InputIt first, InputIt last);

    template <typename InputIt, typename OutputIt>
    void find(InputIt first, InputIt last, OutputIt output_begin);

    __device__ bool insert(const value_type& pair, int thread_id);

    __device__ mapped_type find(key_type const& key);

private:
    mapped_type sentinel_value_;
    hasher hf_;
    pool_allocator_type entry_allocator_;
    entry_ptr_allocator_type entry_ptr_allocator_;
    lock_allocator_type lock_allocator_;

    size_t count_;
    size_t elements_;
    entry_type** entries_;
    entry_type* pool_;
    lock_type* locks_;

};

template <class Key,
          class T,
          class DevHash,
          class DevAllocator,
          class HostHash,
          class HostAllocator>
void copyTableToHost(const HashTableSC<Key, T, DevHash, DevAllocator>& table,
                     HashTableSC<Key, T, HostHash, HostAllocator>& hostTable) {
    cudaMemcpy(hostTable.entries_, table.entries_, table.count_ * sizeof(Entry<Key, T>*),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hostTable.pool_, table.pool_, table.elements_ * sizeof(Entry<Key, T>),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < table.elements_; i++) {
        if (hostTable.pool_[i].next != nullptr)
            hostTable.pool_[i].next = (Entry<Key, T>*)((size_t)hostTable.pool_[i].next -
                                                      (size_t)table.pool_ + (size_t)hostTable.pool_);
    }
    for (int i = 0; i < table.count_; i++) {
        if (hostTable.entries_[i] != nullptr)
            hostTable.entries_[i] = (Entry<Key, T>*)((size_t)hostTable.entries_[i] -
                                                    (size_t)table.pool_ + (size_t)hostTable.pool_);
    }
}

#include "hash_table_sc/detail/hash_table_sc_impl.cuh"