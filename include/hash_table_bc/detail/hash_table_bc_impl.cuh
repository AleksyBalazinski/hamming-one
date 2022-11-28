// Based on "Better GPU Hash Tables" by M. A. Awad, S. Ashkiani, S. D. Porumbescu, M. Farach-Colton & J. D. Owens

#pragma once

#include <random>
#include <iterator>
#include <thrust/fill.h>
#include "detail/bucket.cuh"
#include "detail/device_side.cuh"
#include "detail/rng.hpp"

template <class Key, class T, class Hash, class Allocator>
HashTableBC<Key, T, Hash, Allocator>::HashTableBC(std::size_t capacity,
                                                  Key empty_key_sentinel,
                                                  T empty_value_sentinel,
                                                  Allocator const &allocator)
    : capacity_{std::max(capacity, size_t{1})},
      sentinel_key_{empty_key_sentinel},
      sentinel_value_{empty_value_sentinel},
      allocator_{allocator},
      atomic_pairs_allocator_{allocator},
      pool_allocator_{allocator},
      size_type_allocator_{allocator}
{
    size_t reminder = capacity_ % bucket_size;
    if (reminder)
        capacity_ += bucket_size - reminder;

    num_buckets_ = capacity_ / bucket_size;

    d_table_ = std::allocator_traits<atomic_pair_allocator_type>::allocate(
        atomic_pairs_allocator_, capacity_);
    table_ = std::shared_ptr<atomic_pair_type>(d_table_,
                                               CudaDeleter<atomic_pair_type>());

    d_build_success_ =
        std::allocator_traits<pool_allocator_type>::allocate(pool_allocator_, 1);
    build_success_ = std::shared_ptr<bool>(d_build_success_, CudaDeleter<bool>());

    value_type empty_pair{sentinel_key_, sentinel_value_};
    thrust::fill(thrust::device, d_table_, d_table_ + capacity_, empty_pair);

    double lg_input_size = (float)(log((double)capacity) / log(2.0));
    const unsigned max_iter_const = 7;
    max_cuckoo_chains_ = static_cast<uint32_t>(max_iter_const * lg_input_size);

    std::mt19937 rng(2);
    hf_initializer<hasher, std::mt19937> hf_init{};
    hf0_ = hf_init(rng);
    hf1_ = hf_init(rng);
    hf2_ = hf_init(rng);

    bool success = true;
    CUDA_TRY(
        cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}

template <class Key, class T, class Hash, class Allocator>
HashTableBC<Key, T, Hash, Allocator>::~HashTableBC() {}

template <class Key, class T, class Hash, class Allocator>
template <class InputIt>
bool HashTableBC<Key, T, Hash, Allocator>::insert(InputIt first, InputIt last, cudaStream_t stream)
{
    const auto num_keys = std::distance(first, last);

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    detail::device_side::cooperativeInsert<<<num_blocks, block_size, 0, stream>>>(
        first, last, *this);
    bool success;
    CUDA_TRY(cudaMemcpyAsync(
        &success, d_build_success_, sizeof(bool), cudaMemcpyDeviceToHost, stream));
    return success;
}

template <class Key, class T, class Hash, class Allocator>
template <class InputIt, class OutputIt>
void HashTableBC<Key, T, Hash, Allocator>::find(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream)
{
    const auto num_keys = last - first;
    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;

    detail::device_side::cooperativeFind<<<num_blocks, block_size, 0, stream>>>(
        first, last, output_begin, *this);
}

template <class Key, class T, class Hash, class Allocator>
template <class TileType>
__device__ bool HashTableBC<Key, T, Hash, Allocator>::insert(value_type const &pair,
                                                             TileType const &tile)
{
    auto bucket_id = hf0_(pair.first) % num_buckets_;
    uint32_t cuckoo_counter = 0;
    mars_rng_32 rng; // TODO
    auto lane_id = tile.thread_rank();
    const int elected_lane = 0;
    value_type sentinel_pair{sentinel_key_, sentinel_value_};
    value_type insertion_pair = pair;
    using bucket_type = detail::Bucket<atomic_pair_type, value_type, TileType>;
    do
    {
        bucket_type cur_bucket(&d_table_[bucket_id * bucket_size], tile);
        cur_bucket.load();
        int load = cur_bucket.compute_load(sentinel_pair);

        if (load != bucket_size)
        {
            // bucket is not full
            bool cas_success = false;
            if (lane_id == elected_lane)
            {
                cas_success =
                    cur_bucket.casAtLocation(insertion_pair,
                                             load,
                                             sentinel_pair);
            }
            cas_success = tile.shfl(cas_success, elected_lane);
            if (cas_success)
            {
                return true;
            }
        }
        else
        {
            // cuckoo
            // note that if cuckoo hashing failed we might insert the key
            // but we exchanged another key
            // note that we don't need to shuffle the key since we use the same elected
            // lane for insertion
            if (lane_id == elected_lane)
            {
                auto random_location = rng() % bucket_size;
                auto old_pair = cur_bucket.exchAtLocation(
                    insertion_pair, random_location);

                auto bucket0 = hf0_(old_pair.first) % num_buckets_;
                auto bucket1 = hf1_(old_pair.first) % num_buckets_;
                auto bucket2 = hf2_(old_pair.first) % num_buckets_;

                auto new_bucket_id = bucket0;
                new_bucket_id = bucket_id == bucket1 ? bucket2 : new_bucket_id;
                new_bucket_id = bucket_id == bucket0 ? bucket1 : new_bucket_id;

                bucket_id = new_bucket_id;

                insertion_pair = old_pair;
            }
            bucket_id = tile.shfl(bucket_id, elected_lane);
            cuckoo_counter++;
        }
    } while (cuckoo_counter < max_cuckoo_chains_);
    return false;
}

template <class Key, class T, class Hash, class Allocator>
template <class TileType>
HashTableBC<Key, T, Hash, Allocator>::mapped_type
    __device__
    HashTableBC<Key, T, Hash, Allocator>::find(key_type const &key,
                                               TileType const &tile)
{
    const int num_hfs = 3;
    auto bucket_id = hf0_(key) % num_buckets_;
    value_type sentinel_pair{sentinel_key_, sentinel_value_};
    using bucket_type = detail::Bucket<atomic_pair_type, value_type, TileType>;
    for (int hf = 0; hf < num_hfs; hf++)
    {
        bucket_type cur_bucket(&d_table_[bucket_id * bucket_size], tile);
        cur_bucket.load();
        int key_location = cur_bucket.find_key_location(key);
        if (key_location != -1)
        {
            auto found_value = cur_bucket.get_value_from_lane(key_location);
            return found_value;
        }
        else if (cur_bucket.compute_load(sentinel_pair) < bucket_size)
        {
            return sentinel_value_;
        }
        else
        {
            bucket_id = hf == 0 ? hf1_(key) % num_buckets_ : hf2_(key) % num_buckets_;
        }
    }

    return sentinel_value_;
}