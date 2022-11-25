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
}