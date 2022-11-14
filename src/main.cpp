#define noDEBUG

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

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
std::vector<size_t> computeHashes(Iterator begin, Iterator end, int p, int m, int seqLength)
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

void hammingOne(std::vector<std::vector<int>> sequences)
{
    int seqLength = sequences[0].size();
    std::unordered_multimap<Triple<size_t, size_t, int>, int, TripleHash<size_t, size_t, int>> d;
    const int p = 31;
    const int m = 1e9 + 9;
    int sId = 0;
    std::vector<Triple<size_t, size_t, int>> ownHashes;
    std::vector<Triple<size_t, size_t, int>> matchingHashes;
    for (const auto &seq : sequences)
    {
        auto h1 = computeHashes(seq.cbegin(), seq.cend(), p, m, seqLength);
        auto h2 = computeHashes(seq.crbegin(), seq.crend(), p, m, seqLength);
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
#ifdef DEBUG
            std::cout << "seq " << sId << ", pos " << i << ":\n";
            std::cout << "matchingHash: {" << matchingHash.item1 << ", " << matchingHash.item2 << ", " << matchingHash.item3 << "}\n";
            std::cout << "ownHash: {" << ownHash.item1 << ", " << ownHash.item2 << ", " << ownHash.item3 << "}\n";
#endif
            // if (d.find(matchingHash) != d.end())
            // {
            //     for (int matchingSeqId : d[matchingHash])
            //     {
            //         std::cout << matchingSeqId << ' ' << sId << '\n';
            //     }
            // }

            // if (d.find(ownHash) == d.end())
            //     d[ownHash] = std::vector<int>({sId});
            // else
            //     d[ownHash].push_back(sId);
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
            // std::cout << "not found\n";
            continue;
        }

        for (auto it = iters.first; it != iters.second; it++)
        {
            std::cout << i / seqLength << ' ' << it->second << '\n';
        }
    }
}

int main(int argc, char **argv)
{
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);
    std::cout << numOfSequences << ' ' << seqLength << '\n'; // TEST

    std::vector<std::vector<int>> sequences(numOfSequences, std::vector<int>(seqLength));
    readDataFile(pathToData, sequences);

    for (int seq = 0; seq < numOfSequences; seq++)
    {
        std::cout << seq << ": ";
        for (int j = 0; j < seqLength; j++)
        {
            std::cout << sequences[seq][j];
        }
        std::cout << "\n";
    }

    std::cout << "***Solution:***\n";
    hammingOne(sequences);
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