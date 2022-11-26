#define noDEBUG

#include <string>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

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

void fillWithSeqIds(std::vector<int>& vec, int seq_length)
{
    for(int i=0;i<vec.size();i++)
    {
        vec[i] = i / seq_length;
    }
}

int main(int argc, char **argv)
{
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, PRINTF_FIFO_SIZE);

    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);
    const double load_factor = std::stod(argv[3]);

    int num_sequences, seq_length;
    readMetadataFile(path_to_metadata, num_sequences, seq_length);

    size_t total_len = num_sequences * seq_length;
    const size_t HASH_ENTRIES = total_len / load_factor;
    int *sequences = new int[total_len];
    readDataFile(path_to_data, sequences, num_sequences, seq_length);

    int *dev_sequences;
    cudaMalloc(&dev_sequences, total_len * sizeof(int));
    cudaMemcpy(dev_sequences, sequences, total_len * sizeof(int), cudaMemcpyHostToDevice);

    size_t *dev_prefixes;
    size_t *dev_suffixes;
    triple<size_t, size_t, int> *dev_matchingHashes;
    triple<size_t, size_t, int> *d_own_hashes;
    cudaMalloc(&dev_prefixes, total_len * sizeof(size_t));
    cudaMalloc(&dev_suffixes, total_len * sizeof(size_t));
    cudaMalloc(&dev_matchingHashes, total_len * sizeof(triple<size_t, size_t, int>));
    cudaMalloc(&d_own_hashes, total_len * sizeof(triple<size_t, size_t, int>));

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_sequences + block_size - 1) / block_size;
    getHashes<<<num_blocks, block_size>>>(dev_sequences, num_sequences, seq_length, dev_prefixes, dev_suffixes, dev_matchingHashes, d_own_hashes);

    // HASH TABLE
    // dev_own_hashes - keys, seq_id = i / seq_length - values
    // dev_matching_hashes - queries
    HashTableSC<triple<size_t, size_t, int>, int> table(HASH_ENTRIES, total_len, std::numeric_limits<int>::max());
    
    thrust::device_vector<triple<size_t, size_t, int>> d_keys(d_own_hashes, d_own_hashes + total_len);

    std::vector<int> seq_ids(total_len);
    fillWithSeqIds(seq_ids, seq_length); // TODO parallelize, generalize
    using pair_type = Pair<triple<size_t, size_t, int>, int>;

    thrust::device_vector<int> d_values(total_len);
    d_values = seq_ids;
    auto toPair = [] __host__ __device__ (triple<size_t, size_t, int> a, int b) {
        return pair_type{a, b};
    };

    thrust::device_vector<pair_type> d_pairs(total_len);
    thrust::transform(thrust::device, d_keys.data().get(), d_keys.data().get() + total_len, d_values.begin(), d_pairs.data().get(), toPair);

    table.insert(d_pairs.data().get(), d_pairs.data().get() + total_len);
    thrust::device_vector<triple<size_t, size_t, int>> d_queries(dev_matchingHashes, dev_matchingHashes + total_len);
    thrust::device_vector<int> d_results(total_len);
    table.find(d_queries.data().get(), d_queries.data().get() + total_len, d_results.data().get());
    
    thrust::host_vector<int> h_results = d_results;


    for(int i=0;i<total_len; i++)
    {
        std::cout << i/seq_length << ' ' << h_results[i] << '\n';
    }

    delete[] sequences;
    cudaFree(dev_prefixes);
    cudaFree(dev_suffixes);
    cudaFree(dev_matchingHashes);
    cudaFree(d_own_hashes);
    table.freeTable();
}