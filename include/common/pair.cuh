#pragma once

template <class T1, class T2>
struct Pair {
    T1 first;
    T2 second;
    using first_type = T1;
    using second_type = T2;

    Pair() = default;
    ~Pair() = default;
    Pair(Pair const&) = default;
    Pair(Pair&&) = default;
    Pair& operator=(Pair const&) = default;
    Pair& operator=(Pair&&) = default;

    __host__ __device__ inline bool operator==(const Pair& rhs) {
        return (this->first == rhs.first) && (this->second == rhs.second);
    }
    __host__ __device__ inline bool operator!=(const Pair& rhs) {
        return !(*this == rhs);
    }

    __host__ __device__ constexpr Pair(T1 const& t, T2 const& u)
        : first{t}, second{u} {}
};