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

    Triple() {}
    __host__ __device__ Triple(T1 i1, T2 i2, T3 i3) : item1(i1), item2(i2), item3(i3) {}
};

template <class T1, class T2, class T3>
__host__ __device__ bool operator==(const Triple<T1, T2, T3> &lhs, const Triple<T1, T2, T3> &rhs)
{
    return (lhs.item1 == rhs.item1) && (lhs.item2 == rhs.item2) && (lhs.item3 == rhs.item3);
}

template <class T1, class T2, class T3>
__host__ __device__ bool operator!=(const Triple<T1, T2, T3> &lhs, const Triple<T1, T2, T3> &rhs)
{
    return !(lhs == rhs);
}