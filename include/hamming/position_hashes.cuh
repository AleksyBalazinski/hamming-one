#pragma once

#include "hamming/rolling_hash.cuh"

namespace hamming::device_side {
template <typename SeqIt, typename PartHashIt, typename OutHashIt>
__global__ void getHashes(int seq_length,
                          SeqIt first,
                          SeqIt last,
                          PartHashIt prefixes_first,
                          PartHashIt suffixes_first,
                          OutHashIt matching_hashes_first,
                          OutHashIt own_hashes_first) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    auto total_length = last - first;
    int num_sequences = total_length / seq_length;
    if (thread_id > num_sequences - 1)
        return;

    int offset = thread_id * seq_length;
    SeqIt cur_sequence_first = first + offset;
    PartHashIt cur_prefixes_first = prefixes_first + offset;
    PartHashIt cur_suffixes_first = suffixes_first + offset;
    OutHashIt cur_matching_hashes_first = matching_hashes_first + offset;
    OutHashIt cur_own_hashes_first = own_hashes_first + offset;

    hamming::computePrefixHashes(cur_sequence_first,
                                 cur_sequence_first + seq_length,
                                 cur_prefixes_first);
    hamming::computeSuffixHashes(cur_sequence_first,
                                 cur_sequence_first + seq_length,
                                 cur_suffixes_first);

    for (int i = 0; i < seq_length; i++) {
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

        cur_matching_hashes_first[i] = triple<size_t, size_t, int>(
            prefix_hash, suffix_hash, erased == 0 ? 1 : 0);
        cur_own_hashes_first[i] =
            triple<size_t, size_t, int>(prefix_hash, suffix_hash, erased);
    }
}
}  // namespace hamming::device_side