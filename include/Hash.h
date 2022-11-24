#pragma once

#include "Tuple.h"

template <class T>
struct IntegralHash
{
    __host__ __device__ size_t operator()(T v) { return v; }
};

template <class T1, class T2>
struct PairHash
{
    __host__ __device__ size_t operator()(Pair<T1, T2> pair)
    {
        return pair.first ^ pair.second;
    }
};

template <class T1, class T2, class T3>
struct TripleHash
{
    __host__ __device__ size_t operator()(Triple<T1, T2, T3> triple) const
    {
        const unsigned char *p = reinterpret_cast<const unsigned char *>(&triple);
        size_t h = 2166136261ull;

        for (unsigned int i = 0; i < sizeof(triple); ++i)
            h = (h * 16777619) ^ p[i];

        return h;
    }
};
