#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common/utils.hpp"

bool isDistOne(const std::vector<int>& s1, const std::vector<int>& s2) {
    int dist = 0;
    for (int i = 0; i < s1.size(); i++) {
        if (s1[i] != s2[i])
            dist++;
        if (dist > 1)
            return false;
    }

    return dist == 1;
}

void hammingOne(const std::vector<std::vector<int>>& sequences,
                std::ofstream& result_out) {
    for (int i = 0; i < sequences.size(); i++) {
        for (int j = 0; j < sequences.size(); j++) {
            if (isDistOne(sequences[i], sequences[j])) {
                result_out << i + 1 << ' ' << j + 1 << '\n';
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);
    std::string path_to_result(argv[3]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    std::vector<std::vector<int>> sequences(numOfSequences,
                                            std::vector<int>(seqLength));
    readDataFile(pathToData, sequences);

    std::ofstream result_out(path_to_result, std::ios::out);
    hammingOne(sequences, result_out);
    result_out.close();
}