#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <limits>
#include "common/cuda_allocator.hpp"
#include "common/triple.cuh"
#include "common/utils.hpp"
#include "hamming/position_hashes.cuh"
#include "hash_table_sc/hash_table_sc.hpp"

void fillWithSeqIds(std::vector<int>& vec, int seq_length) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = i / seq_length;
    }
}

int main(int argc, char** argv) {
    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);
    const double load_factor = std::stod(argv[3]);
    std::string path_to_result(argv[4]);

    int num_sequences, seq_length;
    readMetadataFile(path_to_metadata, num_sequences, seq_length);

    size_t total_len = num_sequences * seq_length;
    const size_t HASH_ENTRIES = total_len / load_factor;
    std::vector<int> h_sequences(total_len);
    readDataFile(path_to_data, h_sequences);

    std::chrono::steady_clock::time_point t_begin_total = std::chrono::steady_clock::now();
    thrust::device_vector<int> d_sequences(total_len);
    d_sequences = h_sequences;

    thrust::device_vector<size_t> d_prefixes(total_len);
    thrust::device_vector<size_t> d_suffixes(total_len);

    using hash_type = triple<size_t, size_t, int>;
    thrust::device_vector<hash_type> d_matching_hashes(total_len);
    thrust::device_vector<hash_type> d_own_hashes(total_len);

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_sequences + block_size - 1) / block_size;
    hamming::device_side::getHashes<<<num_blocks, block_size>>>(
        seq_length, d_sequences.begin(), d_sequences.end(), d_prefixes.begin(), d_suffixes.begin(),
        d_matching_hashes.begin(), d_own_hashes.begin());
    // HASH TABLE
    // dev_own_hashes - keys, seq_id = i / seq_length - values
    // dev_matching_hashes - queries
    const int empty_value = std::numeric_limits<int>::max();
    HashTableSC<hash_type, int> table(HASH_ENTRIES, total_len, empty_value);

    std::vector<int> seq_ids(total_len);
    fillWithSeqIds(seq_ids, seq_length);  // TODO parallelize, generalize
    using pair_type = Pair<hash_type, int>;

    thrust::device_vector<int> d_values(total_len);
    d_values = seq_ids;
    auto toPair = [] __host__ __device__(hash_type a, int b) { return pair_type{a, b}; };

    thrust::device_vector<pair_type> d_pairs(total_len);
    thrust::transform(thrust::device, d_own_hashes.begin(), d_own_hashes.end(), d_values.begin(),
                      d_pairs.begin(), toPair);

    table.insert(d_pairs.begin(), d_pairs.end());
    CUDA_TRY(cudaDeviceSynchronize());
    thrust::device_vector<int> d_results(total_len);
    table.find(d_matching_hashes.begin(), d_matching_hashes.end(), d_results.begin());
    CUDA_TRY(cudaDeviceSynchronize());  // an illegal memory access was encountered

    thrust::host_vector<int> h_results = d_results;
    std::ofstream result_out(path_to_result, std::ios::out);
    for (int i = 0; i < total_len; i++) {
        if (h_results[i] != empty_value)
            result_out << i / seq_length << ' ' << h_results[i] << '\n';
    }
    std::chrono::steady_clock::time_point t_end_total = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_begin_total)
                         .count() /
                     1000.0
              << " ms\n";
    result_out.close();
    table.freeTable();
}