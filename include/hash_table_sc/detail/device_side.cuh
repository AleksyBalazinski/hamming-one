#pragma once
#include "hash_table_sc/hash_table_sc.hpp"

namespace detail::device_side
{
    template <typename InputIt, typename HashTable>
    __global__ void insert(InputIt first, InputIt last, HashTable table)
    {
        int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
        auto count_ = last - first;
        if (thread_id > count_ - 1)
            return;

        typename HashTable::value_type insertion_pair = first[thread_id];
        table.insert(insertion_pair, thread_id);
    }

    template <typename InputIt, typename OutputIt, typename HashTable>
    __global__ void find(InputIt first, InputIt last, OutputIt output_begin, HashTable table)
    {
        int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
        auto count_ = last - first;
        if (thread_id > count_ - 1)
            return;

        typename HashTable::key_type find_key = first[thread_id];
        typename HashTable::mapped_type result = table.find(find_key);

        output_begin[thread_id] = result;
    }
}