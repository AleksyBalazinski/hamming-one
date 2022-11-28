// Based on "Better GPU Hash Tables" by M. A. Awad, S. Ashkiani, S. D. Porumbescu, M. Farach-Colton & J. D. Owens

#pragma once

#include <cooperative_groups.h>
#include "common/cuda_helpers.cuh"

namespace detail::device_side
{
    template <typename InputIt, typename HashMap>
    __global__ void cooperativeInsert(InputIt first, InputIt last, HashMap map)
    {
        // construct the work tile
        auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        auto thb = cooperative_groups::this_thread_block();
        auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(thb);

        auto count = last - first;
        if (thread_id - tile.thread_rank() >= count)
        {
            return;
        }

        bool to_insert = false;
        typename HashMap::value_type insertion_pair{};

        // load the input
        if (thread_id < count)
        {
            insertion_pair = first[thread_id];
            to_insert = true;
        }

        bool success = true;
        // Perform the insertions
        while (uint32_t work_queue = tile.ballot(to_insert))
        {
            auto cur_lane = __ffs(work_queue) - 1;
            auto cur_pair = tile.shfl(insertion_pair, cur_lane);
            bool cur_result = map.insert(cur_pair, tile);

            if (tile.thread_rank() == cur_lane)
            {
                to_insert = false;
                success = cur_result;
            }
        }

        if (!tile.all(success))
        {
            *map.d_build_success_ = false;
        }
    }

    template <typename InputIt, typename OutputIt, typename HashMap>
    __global__ void cooperativeFind(InputIt first,
                                    InputIt last,
                                    OutputIt output_begin,
                                    HashMap map)
    {
        // construct the tile
        auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        auto thb = cooperative_groups::this_thread_block();
        auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(thb);

        auto count = last - first;
        if (thread_id - tile.thread_rank() >= count)
        {
            return;
        }

        bool to_find = false;
        typename HashMap::key_type find_key;
        typename HashMap::mapped_type result;

        // load the input
        if (thread_id < count)
        {
            find_key = first[thread_id];
            to_find = true;
        }

        // Find
        // auto work_queue = tile.ballot(to_find);
        while (uint32_t work_queue = tile.ballot(to_find))
        {
            auto cur_lane = __ffs(work_queue) - 1;
            auto cur_key = tile.shfl(find_key, cur_lane);

            typename HashMap::mapped_type find_result = map.find(cur_key, tile);

            if (tile.thread_rank() == cur_lane)
            {
                result = find_result;
                to_find = false;
            }
            // work_queue = tile.ballot(to_find);
        }

        if (thread_id < count)
        {
            output_begin[thread_id] = result;
        }
    }
}