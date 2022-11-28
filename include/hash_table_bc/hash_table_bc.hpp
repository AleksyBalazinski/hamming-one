// Based on "Better GPU Hash Tables" by M. A. Awad, S. Ashkiani, S. D. Porumbescu, M. Farach-Colton & J. D. Owens

#pragma once

#include <cuda/atomic>
#include "common/cuda_allocator.hpp"
#include "common/pair.cuh"
#include "common/cuda_helpers.cuh"
#include "common/cuda_hash.cuh"
#include "detail/device_side.cuh"
#include <memory>

template <class Key, class T, class Hash = CudaHash<Key>, class Allocator = CudaAllocator<char>>
struct HashTableBC
{
    using value_type = Pair<Key, T>;
    using key_type = Key;
    using mapped_type = T;
    using atomic_pair_type = cuda::atomic<Pair<Key, T>, cuda::thread_scope_device>;
    using allocator_type = Allocator;
    using hasher = Hash;
    using size_type = std::size_t;

    using atomic_pair_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<atomic_pair_type>;
    using pool_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<bool>;
    using size_type_allocator_type =
        typename std::allocator_traits<Allocator>::rebind_alloc<size_type>;

    static constexpr auto bucket_size = 16;

    HashTableBC(std::size_t capacity,
                Key sentinel_key,
                T sentinel_value,
                Allocator const &allocator = Allocator{});

    ~HashTableBC();

    template <typename InputIt>
    bool insert(InputIt first, InputIt last, cudaStream_t stream = 0);

    template <typename InputIt, typename OutputIt>
    void find(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0);

    template <typename tile_type>
    __device__ bool insert(value_type const &pair, tile_type const &tile);

    template <typename tile_type>
    __device__ mapped_type find(key_type const &key, tile_type const &tile);

private:
    template <typename InputIt, typename HashMap>
    friend __global__ void detail::device_side::cooperativeInsert(InputIt, InputIt, HashMap);

    template <typename InputIt, typename OutputIt, typename HashMap>
    friend __global__ void detail::device_side::cooperativeFind(InputIt, InputIt, OutputIt, HashMap);

    std::size_t capacity_;
    key_type sentinel_key_{};
    mapped_type sentinel_value_{};
    allocator_type allocator_;
    atomic_pair_allocator_type atomic_pairs_allocator_;
    pool_allocator_type pool_allocator_;
    size_type_allocator_type size_type_allocator_;

    atomic_pair_type *d_table_{};
    std::shared_ptr<atomic_pair_type> table_;

    bool *d_build_success_;
    std::shared_ptr<bool> build_success_;

    uint32_t max_cuckoo_chains_;

    Hash hf0_;
    Hash hf1_;
    Hash hf2_;

    std::size_t num_buckets_;
};

#include "detail/hash_table_bc_impl.cuh"