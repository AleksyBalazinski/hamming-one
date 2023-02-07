/**
 *   Copyright 2023 Aleksy Balazinski
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once

#include <functional>
#include <memory>
#include "common/cuda_allocator.hpp"
#include "common/cuda_deleter.hpp"
#include "common/cuda_hash.cuh"
#include "common/pair.cuh"
#include "detail/cuda_lock.cuh"
#include "detail/entry.hpp"

/**
 * @brief Static associative GPU hash table using separate chaining to resolve
 * collisions.
 *
 * @tparam Key Type of the key values
 * @tparam T Type of the mapped value
 * @tparam Hash Unary function object class that defines the hash function
 * @tparam Allocator The allocator to use for managing GPU device memory
 * @tparam Lock Type of the lock object used to enforce mutual exclusion between
 * GPU threads accessing the same hash table bucket. The class must expose
 * `lock` and `unlock` member functions
 */
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
    using bucket_type = Entry<Key, T>*;

    using pool_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<entry_type>;
    using entry_ptr_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<entry_type*>;
    using lock_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<lock_type>;

    HashTableSC(const HashTableSC& other);

    /**
     * @brief Constructs a new hash table object with the specified capacity and
     * buckets number. Uses the specified sentinel value to represent value that
     * doesn't exist in the hash table.
     *
     * @param num_entries Number of buckets in the hash table
     * @param num_elements Capacity of the hash table, i.e. maximum number of
     * elements the hash table can store
     * @param sentinel_value A reserved value denoting empty value
     */
    HashTableSC(int num_entries, int num_elements, T sentinel_value);

    ~HashTableSC();

    /**
     * @brief Host-side member function for inserting all pairs defined by the
     * input argument iterators. Inserting non-unique keys is allowed, although
     * the find operation on a non-unique key will return only the most recently
     * inserted value associated with that key in that case.
     *
     * @tparam InputIt Device-side iterator that can be dereferenced to
     * `value_type`
     * @param first Beginning of the input pairs to insert
     * @param last End of the input pairs to insert
     * @return Success (true) or failure (false) of the insertion operation
     */
    template <typename InputIt>
    bool insert(InputIt first, InputIt last);

    /**
     * @brief Host-side member function for finding values associated with all
     * keys defined by the input argument iterators.
     *
     * @tparam InputIt Device-side iterator that can be dereferenced to
     * `key_type`
     * @tparam OutputIt Device-side iterator that can be dereferenced to
     * `mapped_type`
     * @param first Beginning of the input keys to find
     * @param last End of the input keys to find
     * @param output_begin Beginning of the output buffer to store the reults
     * into. The size of the buffer must match the number of queries defined by
     * the input iterators
     */
    template <typename InputIt, typename OutputIt>
    void find(InputIt first, InputIt last, OutputIt output_begin);

    /**
     * @brief Device-side member function that inserts a single pair into the
     * hash table.
     *
     * @param pair A key-value pair to be inserted into the hash table
     * @param thread_id The id of the thread performing the insertion
     * @param lane_id The id of the thread performing the insertion relative to
     * the thread's warp
     * @return Success (true) or failure (false) of the insertion operation
     */
    __device__ bool insert(const value_type& pair, int thread_id, int lane_id);

    /**
     * @brief Device-side member function that finds a single pair in the hash
     * table.
     *
     * @param key A key whose associated value is to be found in the hash table
     * @return The value associated with the key if the key exists in the hash
     * table or `sentinel_value` otherwise.
     */
    __device__ mapped_type find(key_type const& key);

    /**
     * @brief Get the bucket that contains a pair with a given key.
     *
     * @param key A key whose associated bucket is to be found in the hash table
     * @return A bucket containing `key` if it exists in the hash table or
     * `nullptr` otherwise
     */
    __host__ __device__ entry_type* getBucket(key_type key);

private:
    mapped_type sentinel_value_;
    hasher hf_;

    pool_allocator_type entry_allocator_;
    entry_ptr_allocator_type entry_ptr_allocator_;
    lock_allocator_type lock_allocator_;

    size_t count_;
    size_t elements_;

    entry_type** d_entries_;
    std::shared_ptr<entry_type*> entries_;

    entry_type* d_pool_;
    std::shared_ptr<entry_type> pool_;

    lock_type* d_locks_;
    std::shared_ptr<lock_type> locks_;
};

#include "hash_table_sc/detail/hash_table_sc_impl.cuh"