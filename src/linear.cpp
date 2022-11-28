#define noDEBUG

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <chrono>

void readDataFile(std::string path, std::vector<std::vector<int>> &sequences);
void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength);

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
std::vector<size_t> computeHashes(Iterator begin, Iterator end, int p, size_t m, int seqLength)
{
    size_t hashValue = 0;
    size_t pPow = 1;
    std::vector<size_t> hashes;
    hashes.reserve(seqLength);

    for (auto &it = begin; it != end; it++)
    {
        hashValue = (hashValue + ((size_t)*it + 1) * pPow) % m;
        pPow = (pPow * p) % m;

        hashes.push_back(hashValue);
    }

    return hashes;
}

void hammingOne(std::vector<std::vector<int>> sequences, std::ofstream& result_out)
{
    int seqLength = sequences[0].size();
    std::unordered_multimap<Triple<size_t, size_t, int>, int, TripleHash<size_t, size_t, int>> d;
    const int p = 31;
    const size_t m1 = 4757501900887;
    constexpr size_t m2 = 3348549226339;
    int sId = 0;
    std::vector<Triple<size_t, size_t, int>> ownHashes;
    std::vector<Triple<size_t, size_t, int>> matchingHashes;
    for (const auto &seq : sequences)
    {
        auto h1 = computeHashes(seq.cbegin(), seq.cend(), p, m1, seqLength);
        auto h2 = computeHashes(seq.crbegin(), seq.crend(), p, m2, seqLength);
        for (int i = 0; i < seqLength; i++)
        {
            size_t prefixHash;
            if (i == 0)
                prefixHash = 0;
            else
                prefixHash = h1[i - 1];

            size_t suffixHash;
            if (i == seqLength - 1)
                suffixHash = 0;
            else
                suffixHash = h2[seqLength - i - 2];

            unsigned char erased = seq[i];

            auto matchingHash = Triple<size_t, size_t, int>(prefixHash, suffixHash, erased == 0 ? 1 : 0);
            auto ownHash = Triple<size_t, size_t, int>(prefixHash, suffixHash, seq[i] == 0 ? 0 : 1);

            ownHashes.push_back(ownHash);
            matchingHashes.push_back(matchingHash);
        }
        sId++;
    }

    for (int i = 0; i < ownHashes.size(); i++)
    {
        d.insert({ownHashes[i], i / seqLength});
    }

    for (int i = 0; i < matchingHashes.size(); i++)
    {
        auto iters = d.equal_range(matchingHashes[i]);
        if (iters.first == d.end() && iters.second == d.end())
        {
            continue;
        }

        for (auto it = iters.first; it != iters.second; it++)
        {
            result_out << i / seqLength + 1 << ' ' << it->second + 1 << '\n';
        }
    }
}

int main(int argc, char **argv)
{
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);
    std::string path_to_result_out(argv[3]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    std::vector<std::vector<int>> sequences(numOfSequences, std::vector<int>(seqLength));
    readDataFile(pathToData, sequences);

    std::ofstream result_out(path_to_result_out, std::ios::out);
    std::chrono::steady_clock::time_point t_begin_total = std::chrono::steady_clock::now();
    hammingOne(sequences, result_out);
    std::chrono::steady_clock::time_point t_end_total = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_begin_total).count() / 1000.0 << " ms\n";
    result_out.close();
}

void readDataFile(std::string path, std::vector<std::vector<int>> &sequences)
{
    int numOfSequences = sequences.size();
    int seqLength = sequences[0].size();
    std::ifstream in;
    in.open(path, std::ios::in);

    for (int seq = 0; seq < numOfSequences; seq++)
    {
        for (int j = 0; j < seqLength; j++)
        {
            in >> sequences[seq][j];
        }
    }
    in.close();
}

void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength)
{
    std::ifstream in;
    in.open(path, std::ios::in);
    in >> numOfSequences >> sequenceLength;
}