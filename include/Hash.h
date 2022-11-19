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
        return triple.item1 ^ triple.item2 ^ triple.item3;
    }
};
