#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "common/utils.hpp"

template <class T1, class T2, class T3>
struct Triple
{
    T1 item1;
    T2 item2;
    T3 item3;

    Triple() {}
    Triple(T1 i1, T2 i2, T3 i3) : item1(i1), item2(i2), item3(i3) {}
};

template <class T1, class T2, class T3>
bool operator==(const Triple<T1, T2, T3> &lhs, const Triple<T1, T2, T3> &rhs)
{
    return (lhs.item1 == rhs.item1) && (lhs.item2 == rhs.item2) && (lhs.item3 == rhs.item3);
}

template <class T1, class T2, class T3>
bool operator!=(const Triple<T1, T2, T3> &lhs, const Triple<T1, T2, T3> &rhs)
{
    return !(lhs == rhs);
}

template <class T1, class T2, class T3>
struct TripleHash
{
    size_t operator()(Triple<T1, T2, T3> triple) const
    {
        return triple.item1 ^ triple.item2 ^ triple.item3;
    }
};

template <typename Iterator>
std::vector<size_t> computeHashes(Iterator begin, Iterator end, int p, size_t m, int seq_length)
{
    size_t hashValue = 0;
    size_t pPow = 1;
    std::vector<size_t> hashes;
    hashes.reserve(seq_length);

    for (auto &it = begin; it != end; it++)
    {
        hashValue = (hashValue + ((size_t)*it + 1) * pPow) % m;
        pPow = (pPow * p) % m;

        hashes.push_back(hashValue);
    }

    return hashes;
}

void hammingOne(const std::vector<std::vector<int>>& sequences, const std::vector<int>& seq_ids, std::ofstream& result_out)
{
    int seq_length = sequences[0].size();
    using hash_type = Triple<size_t, size_t, int>;
    std::unordered_map<hash_type, int, TripleHash<size_t, size_t, int>> d;
    const int p = 31;
    const size_t m1 = 4757501900887ull;
    constexpr size_t m2 = 3348549226339ull;
    int sId = 0;
    std::vector<hash_type> own_hashes;
    std::vector<hash_type> matching_hashes;
    for (const auto &seq : sequences)
    {
        auto h1 = computeHashes(seq.cbegin(), seq.cend(), p, m1, seq_length);
        auto h2 = computeHashes(seq.crbegin(), seq.crend(), p, m2, seq_length);
        for (int i = 0; i < seq_length; i++)
        {
            size_t prefix_hash;
            if (i == 0)
                prefix_hash = 0;
            else
                prefix_hash = h1[i - 1];

            size_t suffix_hash;
            if (i == seq_length - 1)
                suffix_hash = 0;
            else
                suffix_hash = h2[seq_length - i - 2];

            unsigned char erased = seq[i];

            auto matching_hash = hash_type(prefix_hash, suffix_hash, erased == 0 ? 1 : 0);
            auto own_hash = hash_type(prefix_hash, suffix_hash, erased == 0 ? 0 : 1);

            own_hashes.push_back(own_hash);
            matching_hashes.push_back(matching_hash);
        }
        sId++;
    }

    for (int i = 0; i < own_hashes.size(); i++)
    {
        d.insert({own_hashes[i], seq_ids[i / seq_length]});
    }

    for (int i = 0; i < matching_hashes.size(); i++)
    {
        auto it = d.find(matching_hashes[i]);
        if(it == d.end())
        {
            continue;
        }
            
        result_out << seq_ids[i / seq_length] << ' ' << it->second << '\n';
    }
}

int main(int argc, char **argv)
{
    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);
    std::string path_to_result(argv[3]);

    int num_sequences, seq_length;
    readMetadataFile(path_to_metadata, num_sequences, seq_length);

    std::vector<std::vector<int>> sequences(num_sequences, std::vector<int>(seq_length));
    std::vector<int> seq_ids(num_sequences);
    readRefinedDataFile(path_to_data, sequences, seq_ids);

    std::ofstream result_out(path_to_result, std::ios::out);
    std::chrono::steady_clock::time_point t_begin_total = std::chrono::steady_clock::now();
    hammingOne(sequences, seq_ids, result_out);
    std::chrono::steady_clock::time_point t_end_total = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_begin_total).count() / 1000.0 << " ms\n";
    result_out.close();
}