#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/utils.hpp"
#include <boost/container_hash/hash.hpp>

int main(int argc, char **argv)
{
    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);
    std::string path_to_info_out(argv[3]);
    std::string path_to_metadata_out(argv[4]);
    std::string path_to_data_out(argv[5]);

    int num_sequences, seq_length;
    readMetadataFile(path_to_metadata, num_sequences, seq_length);

    std::vector<std::vector<int>> sequences(num_sequences, std::vector<int>(seq_length));
    readDataFile(path_to_data, sequences);

    std::unordered_map<std::vector<int>, std::vector<int>, boost::hash<std::vector<int>>> map{};
    for (int i = 0; i < sequences.size(); i++)
    {
        auto location = map.find(sequences[i]);
        if (location == map.cend())
        {
            map.insert(std::make_pair(std::vector<int>(sequences[i]), std::vector<int>({ i })));
        }
        else
        {
            map[sequences[i]].push_back(i);
        }
    }

//TODO parallelize
    std::ofstream out;
    out.open(path_to_info_out, std::ios::out);
    for (const auto &kv : map)
    {
        for (int x : kv.second)
        {
            out << x + 1 << ' ';
        }
        out << '\n';
    }
    out.close();

    out.open(path_to_data_out);
    for(const auto& kv : map)
    {
        const auto& rep = sequences[kv.second.at(0)];
        out << kv.second.at(0) + 1 << "\t\t\t";
        for (int x : rep)
        {
            out << x << ' ';
        }
        out << '\n';
    }
    out.close();

    out.open(path_to_metadata_out);
    out << map.size() << ' ' << seq_length << '\n';

    out.close();
}