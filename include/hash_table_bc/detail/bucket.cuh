#pragma once

#include <cuda/atomic>
#include <cuda/std/atomic>

namespace detail
{
    template <typename atomic_pair_type, typename pair_type, typename tile_type>
    struct Bucket
    {
        Bucket() = delete;
        __device__ inline Bucket(atomic_pair_type *ptr, const tile_type &tile) : ptr_(ptr), tile_(tile) {}

        __device__ inline Bucket(const Bucket &other) : lane_pair_(other.lane_pair_), ptr_(other.ptr_) {}

        __device__ inline void load()
        {
            lane_pair_ = ptr_[tile_.thread_rank()].load(cuda::memory_order_relaxed);
        }
        __device__ inline int compute_load(const pair_type &sentinel)
        {
            auto load_bitmap = tile_.ballot(lane_pair_.first != sentinel.first);
            int load = __popc(load_bitmap);
            return load;
        }
        // returns -1 if not found
        __device__ inline int find_key_location(const typename pair_type::first_type &key)
        {
            bool key_exist = key == lane_pair_.first;
            auto key_exist_bmap = tile_.ballot(key_exist);
            int key_lane = __ffs(key_exist_bmap);
            return key_lane - 1;
        }
        __device__ inline
            typename pair_type::second_type
            get_value_from_lane(int location)
        {
            return tile_.shfl(lane_pair_.second, location);
        }

        __device__ inline bool casAtLocation(const pair_type &pair,
                                             const int location,
                                             const pair_type &sentinel)
        {
            pair_type expected = sentinel;
            pair_type desired = pair;
            bool cas_success =
                ptr_[location].compare_exchange_strong(expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
            return cas_success;
        }

        __device__ inline pair_type exchAtLocation(const pair_type &pair,
                                                   const int location)
        {
            auto old = ptr_[location].exchange(pair, cuda::memory_order_relaxed);
            return old;
        }

    private:
        pair_type lane_pair_;
        atomic_pair_type *ptr_;
        tile_type tile_;
    };
} // namespace detail