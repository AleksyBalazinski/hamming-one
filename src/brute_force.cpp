/**
 *   Copyright 2023 Aleksy Balazinski
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

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