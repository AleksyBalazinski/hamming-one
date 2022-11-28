#pragma once

#include <cooperative_groups.h>
#include "hash_table_sc/hash_table_sc.hpp"

namespace detail::device_side {
template <typename InputIt, typename HashTable>
__global__ void insert(InputIt first, InputIt last, HashTable table) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    auto thb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(thb);

    auto count = last - first;
    if (thread_id - tile.thread_rank() >= count) {
        return;
    }

    bool to_insert = false;
    typename HashTable::value_type insertion_pair{};

    if (thread_id < count) {
        insertion_pair = first[thread_id];
        to_insert = true;
    }

    while (uint32_t work_queue = tile.ballot(to_insert)) {
        auto cur_lane = __ffs(work_queue) - 1;
        auto cur_pair = tile.shfl(insertion_pair, cur_lane);
        table.insert(cur_pair, thread_id, cur_lane);

        if (tile.thread_rank() == cur_lane) {
            to_insert = false;
        }
    }
}

template <typename InputIt, typename OutputIt, typename HashTable>
__global__ void find(InputIt first, InputIt last, OutputIt output_begin, HashTable table) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    auto count_ = last - first;
    if (thread_id > count_ - 1)
        return;

    typename HashTable::key_type find_key = first[thread_id];
    typename HashTable::mapped_type result = table.find(find_key);

    output_begin[thread_id] = result;
}
}  // namespace detail::device_side