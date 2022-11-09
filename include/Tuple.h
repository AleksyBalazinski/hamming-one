#pragma once

template <class T1, class T2>
struct Pair
{
    T1 first;
    T2 second;
};

template <class T1, class T2>
__host__ __device__ bool operator==(const Pair<T1, T2> &lhs, const Pair<T1, T2> &rhs)
{
    return (lhs.first == rhs.first) && (lhs.second == rhs.second);
}

template <class T1, class T2>
__host__ __device__ bool operator!=(const Pair<T1, T2> &lhs, const Pair<T1, T2> &rhs)
{
    return !(lhs == rhs);
}

template <class T1, class T2, class T3>
struct Triple
{
    T1 item1;
    T2 item2;
    T3 item3;
};
