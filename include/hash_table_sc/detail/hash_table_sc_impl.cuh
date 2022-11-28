#pragma once

#include "hash_table_sc/detail/device_side.cuh"

template <class Key, class T, class Hash, class Allocator, class Lock>
HashTableSC<Key, T, Hash, Allocator, Lock>::HashTableSC(int num_entries,
                                                        int num_elements,
                                                        T sentinel_value) {
    count_ = num_entries;
    elements_ = num_elements;
    sentinel_value_ = sentinel_value;
    d_entries_ = entry_ptr_allocator_.allocate(count_);
    entries_ =
        std::shared_ptr<entry_type*>(d_entries_, CudaDeleter<entry_type*>());
    d_pool_ = entry_allocator_.allocate(elements_);
    pool_ = std::shared_ptr<entry_type>(d_pool_, CudaDeleter<entry_type>());
    d_locks_ = lock_allocator_.allocate(count_);
    locks_ = std::shared_ptr<Lock>(d_locks_, CudaDeleter<Lock>());
}

template <class Key, class T, class Hash, class Allocator, class Lock>
HashTableSC<Key, T, Hash, Allocator, Lock>::HashTableSC(
    const HashTableSC& other)
    : count_{other.count_},
      elements_{other.elements_},
      sentinel_value_{other.sentinel_value_},
      d_entries_{other.d_entries_},
      entries_{other.entries_},
      d_pool_{other.d_pool_},
      pool_{other.pool_},
      d_locks_{other.d_locks_},
      locks_{other.locks_},
      hf_{other.hf_},
      entry_allocator_{other.entry_allocator_},
      entry_ptr_allocator_{other.entry_ptr_allocator_},
      lock_allocator_{other.lock_allocator_} {}

template <class Key, class T, class Hash, class Allocator, class Lock>
HashTableSC<Key, T, Hash, Allocator, Lock>::~HashTableSC() {}

template <class Key, class T, class Hash, class Allocator, class Lock>
template <typename InputIt>
bool HashTableSC<Key, T, Hash, Allocator, Lock>::insert(InputIt first,
                                                        InputIt last) {
    const auto num_keys = std::distance(first, last);

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    detail::device_side::insert<<<num_blocks, block_size>>>(first, last, *this);

    return true;
}

template <class Key, class T, class Hash, class Allocator, class Lock>
template <class InputIt, class OutputIt>
void HashTableSC<Key, T, Hash, Allocator, Lock>::find(InputIt first,
                                                      InputIt last,
                                                      OutputIt output_begin) {
    const auto num_keys = std::distance(first, last);
    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;

    detail::device_side::find<<<num_blocks, block_size>>>(first, last,
                                                          output_begin, *this);
}

template <class Key, class T, class Hash, class Allocator, class Lock>
__device__ bool HashTableSC<Key, T, Hash, Allocator, Lock>::insert(
    const value_type& pair,
    int thread_id,
    int lane_id) {
    key_type key = pair.first;
    mapped_type value = pair.second;
    size_t hash_value = hf_(key) % count_;
    if (lane_id == thread_id % 32) {
        auto location = &(d_pool_[thread_id]);
        location->key = key;
        location->value = value;
        d_locks_[hash_value].lock();
        location->next = d_entries_[hash_value];
        d_entries_[hash_value] = location;
        d_locks_[hash_value].unlock();
    }
    return true;
}

template <class Key, class T, class Hash, class Allocator, class Lock>
__device__ typename HashTableSC<Key, T, Hash, Allocator, Lock>::mapped_type
HashTableSC<Key, T, Hash, Allocator, Lock>::find(key_type const& key) {
    size_t hash_value = hf_(key) % count_;
    auto cur = d_entries_[hash_value];

    while (cur != nullptr && cur->key != key) {
        cur = cur->next;
    }

    if (cur == nullptr) {
        return sentinel_value_;
    }
    return cur->value;
}