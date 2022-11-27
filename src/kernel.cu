#define noDEBUG

#include <string>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <fstream>
#include <chrono>

#include "common/utils.hpp"
#include "common/cuda_allocator.hpp"
#include "common/triple.cuh"
#include "hash_table_sc/hash_table_sc.hpp"
#include "hamming/rolling_hash.cuh"
#include <limits>

#define PRINTF_FIFO_SIZE (long long int)1e15

__global__ void getHashes(int *sequences, int num_sequences, int seq_length, size_t *prefixes, size_t *suffixes,
                          triple<size_t, size_t, int> *matchingHashes, triple<size_t, size_t, int> *ownHasbes)
{
    int seqId = threadIdx.x + blockDim.x * blockIdx.x;
    if (seqId > num_sequences - 1)
        return;

    int offset = seqId * seq_length;
    int *sequence = sequences + offset;
    size_t *curPrefixes = prefixes + offset;
    size_t *curSuffixes = suffixes + offset;
    triple<size_t, size_t, int> *curMatchingHashes = matchingHashes + offset;
    triple<size_t, size_t, int> *curOwnHashes = ownHasbes + offset;
    hamming::computePrefixHashes(sequence, seq_length, curPrefixes);
    hamming::computeSuffixHashes(sequence, seq_length, curSuffixes);

    for (int i = 0; i < seq_length; i++)
    {
        size_t prefixHash;
        if (i == 0)
            prefixHash = 0;
        else
            prefixHash = curPrefixes[i - 1];

        size_t suffixHash;
        if (i == seq_length - 1)
            suffixHash = 0;
        else
            suffixHash = curSuffixes[seq_length - i - 2];

        int erased = sequence[i];

        curMatchingHashes[i] = triple<size_t, size_t, int>(prefixHash, suffixHash, erased == 0 ? 1 : 0);
        curOwnHashes[i] = triple<size_t, size_t, int>(prefixHash, suffixHash, erased);
    }
}

template<typename SeqIt, typename PartHashIt, typename OutHashIt>
__global__ void getHashes(int seq_length, SeqIt first, SeqIt last, PartHashIt prefixes_first, PartHashIt suffixes_first,
                          OutHashIt matching_hashes_first, OutHashIt own_hashes_first)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    auto total_length = last - first;
    int num_sequences = total_length / seq_length;
    if(thread_id > num_sequences)
        return;
    
    int offset = thread_id * seq_length;
    SeqIt cur_sequence_first = first + offset;
    PartHashIt cur_prefixes_first = prefixes_first + offset;
    PartHashIt cur_suffixes_first = suffixes_first + offset;
    OutHashIt cur_matching_hashes_first = matching_hashes_first + offset;
    OutHashIt cur_own_hashes_first = own_hashes_first + offset;

    hamming::computePrefixHashes(cur_sequence_first, cur_sequence_first + seq_length, cur_prefixes_first);
    hamming::computeSuffixHashes(cur_sequence_first, cur_sequence_first + seq_length, cur_suffixes_first);

    for (int i = 0; i < seq_length; i++)
    {
        size_t prefix_hash;
        if (i == 0)
            prefix_hash = 0;
        else
            prefix_hash = cur_prefixes_first[i - 1];

        size_t suffix_hash;
        if (i == seq_length - 1)
            suffix_hash = 0;
        else
            suffix_hash = cur_suffixes_first[seq_length - i - 2];

        int erased = cur_sequence_first[i];

        cur_matching_hashes_first[i] = triple<size_t, size_t, int>(prefix_hash, suffix_hash, erased == 0 ? 1 : 0);
        cur_own_hashes_first[i] = triple<size_t, size_t, int>(prefix_hash, suffix_hash, erased);
    }
}

void fillWithSeqIds(std::vector<int>& vec, int seq_length)
{
    for(int i=0;i<vec.size();i++)
    {
        vec[i] = i / seq_length;
    }
}

int main(int argc, char **argv)
{
    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, PRINTF_FIFO_SIZE);

    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);
    const double load_factor = std::stod(argv[3]);
    // std::cerr << "load_factor " << load_factor << '\n';
    std::string path_to_result(argv[4]);

    int num_sequences, seq_length;
    readMetadataFile(path_to_metadata, num_sequences, seq_length);

    size_t total_len = num_sequences * seq_length;
    const size_t HASH_ENTRIES = total_len / load_factor;
    std::vector<int> h_sequences(total_len);
    readDataFile(path_to_data, h_sequences);
    // std::cerr << "Files read\n";
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
    // std::cerr << "Before getHashes()\n";
    getHashes<<<num_blocks, block_size>>>(seq_length, d_sequences.begin(), d_sequences.end(), 
            d_prefixes.begin(), d_suffixes.begin(), 
            d_matching_hashes.begin(), d_own_hashes.begin());
    // std::cerr << "After getHashes()\n";
    // HASH TABLE
    // dev_own_hashes - keys, seq_id = i / seq_length - values
    // dev_matching_hashes - queries
    const int empty_value = std::numeric_limits<int>::max();
    HashTableSC<hash_type, int> table(HASH_ENTRIES, total_len, empty_value);

    std::vector<int> seq_ids(total_len);
    fillWithSeqIds(seq_ids, seq_length); // TODO parallelize, generalize
    using pair_type = Pair<hash_type, int>;

    thrust::device_vector<int> d_values(total_len);
    d_values = seq_ids;
    auto toPair = [] __host__ __device__ (hash_type a, int b) {
        return pair_type{a, b};
    };

    thrust::device_vector<pair_type> d_pairs(total_len);
    thrust::transform(thrust::device, d_own_hashes.begin(), d_own_hashes.end(), d_values.begin(), d_pairs.begin(), toPair);

    // std::cerr << "Before insert()\n";
    table.insert(d_pairs.begin(), d_pairs.end());
    // std::cerr << "After insert()\n";
    thrust::device_vector<int> d_results(total_len);
    // std::cerr << "Before find()\n";
    table.find(d_matching_hashes.begin(), d_matching_hashes.end(), d_results.begin());
    // std::cerr << "After find()\n";
    thrust::host_vector<int> h_results = d_results;
    // std::cerr << "After copy to host()\n";
    std::ofstream result_out(path_to_result, std::ios::out);
    // std::cerr << "After result_out\n";
    for(int i=0;i<total_len; i++)
    {
        if(h_results[i] != empty_value)
            result_out << i/seq_length << ' ' << h_results[i] << '\n';
    }
    std::chrono::steady_clock::time_point t_end_total = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_begin_total).count() / 1000.0 << " ms\n";
    result_out.close();
// std::cerr << "1\n";
    table.freeTable();
    // std::cerr << "2\n";
}