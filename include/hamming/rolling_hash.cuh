#pragma once

namespace hamming
{
    constexpr size_t P = 31;
    constexpr size_t M1 = 4757501900887;
    constexpr size_t M2 = 3348549226339;

    __device__ void computePrefixHashes(int *sequence, int seqLength, size_t *prefixHashes)
    {
        size_t hashValue = 0;
        size_t pPow = 1;

        for (int i = 0; i < seqLength; i++)
        {
            hashValue = (hashValue + (size_t(sequence[i]) + 1) * pPow) % M1;
            pPow = (pPow * P) % M1;

            prefixHashes[i] = hashValue;
        }
    }

    __device__ void computeSuffixHashes(int *sequence, int seqLength, size_t *suffixHashes)
    {
        size_t hashValue = 0;
        size_t pPow = 1;

        for (int i = 0; i < seqLength; i++)
        {
            hashValue = (hashValue + (size_t(sequence[seqLength - i - 1]) + 1) * pPow) % M2;
            pPow = (pPow * P) % M2;

            suffixHashes[i] = hashValue;
        }
    }

    template<typename SeqIt, typename OutPartHashIt>
    __device__ void computePrefixHashes(SeqIt first, SeqIt last, OutPartHashIt prefix_first)
    {
        size_t hash_value = 0;
        size_t p_pow = 1;
        int seq_length = last - first;

        for (int i = 0; i < seq_length; i++)
        {
            hash_value = (hash_value + (size_t(first[i]) + 1) * p_pow) % M1;
            p_pow = (p_pow * P) % M1;

            prefix_first[i] = hash_value;
        }
    }

    template<typename SeqIt, typename OutPartHashIt>
    __device__ void computeSuffixHashes(SeqIt first, SeqIt last, OutPartHashIt prefix_first)
    {
        size_t hash_value = 0;
        size_t p_pow = 1;
        int seq_length = last - first;

        for (int i = 0; i < seq_length; i++)
        {
            hash_value = (hash_value + (size_t(first[seq_length - i - 1]) + 1) * p_pow) % M2;
            p_pow = (p_pow * P) % M2;

            prefix_first[i] = hash_value;
        }
    }
}