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

#pragma once

#include "hamming/rolling_hash.cuh"

namespace hamming::device_side {
/**
 * @brief Calculate the hash object for each position in every sequence.
 * @details The hash object calculated for a given position of a sequence is
 * unique across all positions across all \a unique sequences. In another words,
 * for different sequences s1 through sn, each of length l, each position within
 * every sequence si will be assigned a unique object from a set of l*n objects.
 * This object, for a position k of sequence s, is a triple containing two
 * hashes: one calculated from the subsequence s[0..k-1] - \a prefix hash, and
 * the other from s[k+1..|s|-1] - \a suffix hash, and an integer which is either
 * 1. the bit s[i] or 2. the bit that is the negation of s[i]. The hash objects
 * given by 1. are called \a own \a hashes whereas the objects given by 2.
 * are refered to as \a matching \a hashes.
 * @tparam SeqIt Iterator to values of joined sequences
 * @tparam PartHashIt Iterator to values of partial hashes (suffix or prefix
 * hashes)
 * @tparam OutHashIt Iterator to hash objects
 * @param seq_length length of one sequence
 * @param first Beginning of the values of joined sequences
 * @param last End of the values of joined sequences
 * @param prefixes_first Beginning of the preallocated buffer storing prefix
 * hashes. The buffer shall have size equal to the total length of all joined
 * sequences and need not be initialized
 * @param suffixes_first Beginning of the preallocated buffer storing suffix
 * hashes. The buffer shall have size equal to the total length of all joined
 * sequences and need not be initialized
 * @param matching_hashes_first Beginning of the preallocated buffer of size
 * equal to the total length of all joined sequences. The buffer contains \a own
 * hash objects once the function exits normally.
 * @param own_hashes_first Beginning of the preallocated buffer of size equal to
 * the total length of all joined sequences. The buffer contains \a matching
 * hash objects once the function exits normally.
 */
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