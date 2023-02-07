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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "common/cuda_allocator.hpp"
#include "common/cuda_timer.cuh"
#include "common/triple.cuh"
#include "common/utils.hpp"
#include "hamming/position_hashes.cuh"
#include "hash_table_sc/hash_table_sc.hpp"

int main(int argc, char** argv) {
    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);
    const double load_factor = std::stod(argv[3]);
    std::string path_to_result(argv[4]);

    // GPU timers
    CudaTimer data_copying_timer1;
    CudaTimer data_copying_timer2;
    CudaTimer insert_timer;
    CudaTimer find_timer;

    int num_sequences, seq_length;
    readMetadataFile(path_to_metadata, num_sequences, seq_length);

    size_t total_len = num_sequences * seq_length;
    const size_t HASH_ENTRIES = total_len / load_factor;
    std::vector<int> h_sequences(total_len);
    std::vector<int> seq_ids(total_len);
    readRefinedDataFile(path_to_data, h_sequences, seq_length, seq_ids);

    std::chrono::steady_clock::time_point t_begin_total =
        std::chrono::steady_clock::now();
    thrust::device_vector<int> d_sequences(total_len);

    // copy data to device
    data_copying_timer1.startTimer();
    d_sequences = h_sequences;
    data_copying_timer1.startTimer();
    auto sequences_copy_ms = data_copying_timer1.getElapsedMs();

    thrust::device_vector<size_t> d_prefixes(total_len);
    thrust::device_vector<size_t> d_suffixes(total_len);

    using hash_type = triple<size_t, size_t, int>;
    thrust::device_vector<hash_type> d_matching_hashes(total_len);
    thrust::device_vector<hash_type> d_own_hashes(total_len);

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_sequences + block_size - 1) / block_size;
    hamming::device_side::getHashes<<<num_blocks, block_size>>>(
        seq_length, d_sequences.begin(), d_sequences.end(), d_prefixes.begin(),
        d_suffixes.begin(), d_matching_hashes.begin(), d_own_hashes.begin());

    // create the hash table
    const int empty_value = std::numeric_limits<int>::max();
    HashTableSC<hash_type, int> table(HASH_ENTRIES, total_len, empty_value);

    using pair_type = Pair<hash_type, int>;

    thrust::device_vector<int> d_values(total_len);
    d_values = seq_ids;
    auto toPair = [] __host__ __device__(hash_type a, int b) {
        return pair_type{a, b};
    };

    thrust::device_vector<pair_type> d_pairs(total_len);
    thrust::transform(thrust::device, d_own_hashes.begin(), d_own_hashes.end(),
                      d_values.begin(), d_pairs.begin(), toPair);

    // populate the hash table
    insert_timer.startTimer();
    table.insert(d_pairs.begin(), d_pairs.end());
    insert_timer.stopTimer();
    auto insertion_ms = insert_timer.getElapsedMs();

    CUDA_TRY(cudaDeviceSynchronize());
    thrust::device_vector<int> d_results(total_len);

    // find matches (hamming one pairs)
    find_timer.startTimer();
    table.find(d_matching_hashes.begin(), d_matching_hashes.end(),
               d_results.begin());
    find_timer.stopTimer();
    auto find_ms = find_timer.getElapsedMs();

    CUDA_TRY(cudaDeviceSynchronize());

    // copy results to host
    data_copying_timer2.startTimer();
    thrust::host_vector<int> h_results = d_results;
    data_copying_timer2.stopTimer();
    auto results_copy_ms = data_copying_timer2.getElapsedMs();

    std::chrono::steady_clock::time_point t_end_total =
        std::chrono::steady_clock::now();

    std::ofstream result_out(path_to_result, std::ios::out);
    for (int i = 0; i < total_len; i++) {
        if (h_results[i] != empty_value) {
            result_out << seq_ids[i] << ' ' << h_results[i] << '\n';
        }
    }

    std::cout << "Elapsed total time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     t_end_total - t_begin_total)
                         .count() /
                     1000.0
              << " ms\n";
    std::cout << "Time spent copying data: "
              << sequences_copy_ms + results_copy_ms << " ms\n";
    std::cout << "Insertions took: " << insertion_ms << " ms "
              << "(avg. " << total_len / insertion_ms / 1000.0 << " M/s)\n";
    std::cout << "Finds took: " << find_ms << " ms "
              << "(avg. " << total_len / find_ms / 1000.0 << " M/s)\n";
    result_out.close();
}