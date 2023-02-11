/**
 *   Copyright 2021 The Regents of the University of California, Davis
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
__global__ void find(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     HashTable table) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    auto count_ = last - first;
    if (thread_id > count_ - 1)
        return;

    typename HashTable::key_type find_key = first[thread_id];
    typename HashTable::mapped_type result = table.find(find_key);

    output_begin[thread_id] = result;
}
}  // namespace detail::device_side