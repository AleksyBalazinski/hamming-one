#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/utils.hpp"
#include <boost/container_hash/hash.hpp>

int main(int argc, char **argv)
{
    std::string pathToMetaData(argv[1]);
    std::string pathToData(argv[2]);
    std::string pathToOut(argv[3]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetaData, numOfSequences, seqLength);

    std::vector<std::vector<int>> sequences(numOfSequences, std::vector<int>(seqLength));
    readDataFile(pathToData, sequences);
    for (int i = 0; i < seqLength; i++)
    {
        std::cout << sequences[0][i] << ' ';
    }

    std::unordered_map<std::vector<int>, std::vector<int>, boost::hash<std::vector<int>>> map{};
    for (int i = 0; i < sequences.size(); i++)
    {
        auto location = map.find(sequences[i]);
        if (location == map.cend())
        {
            map.insert(std::make_pair(std::vector<int>(sequences[i]), std::vector<int>({i + 1})));
        }
        else
        {
            map[sequences[i]].push_back(i + 1);
        }
    }

    std::ofstream out;
    out.open(pathToOut, std::ios::out);
    for (const auto &kv : map)
    {
        for (int x : kv.second)
        {
            out << x << ' ';
        }
        out << '\n';
    }
    out.close();
}