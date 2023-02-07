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

#pragma once

#include <fstream>
#include <string>
#include <vector>

template <typename T>
void readDataFile(std::string path, std::vector<std::vector<T>>& sequences) {
    int numOfSequences = sequences.size();
    int seqLength = sequences[0].size();
    std::ifstream in;
    in.open(path, std::ios::in);

    for (int seq = 0; seq < numOfSequences; seq++) {
        for (int j = 0; j < seqLength; j++) {
            in >> sequences[seq][j];
        }
    }
    in.close();
}

template <typename T>
void readRefinedDataFile(std::string path,
                         std::vector<T>& sequences,
                         int seq_length,
                         std::vector<T>& seq_ids) {
    std::ifstream in;
    in.open(path, std::ios::in);

    int num_sequences = sequences.size() / seq_length;
    for (int i = 0; i < num_sequences; i++) {
        int seq_id;
        in >> seq_id;
        for (int j = 0; j < seq_length; j++) {
            seq_ids[j + i * seq_length] = seq_id;
        }
        for (int j = 0; j < seq_length; j++) {
            in >> sequences[j + i * seq_length];
        }
    }
}

template <typename T>
void readRefinedDataFile(std::string path,
                         std::vector<std::vector<T>>& sequences,
                         std::vector<T>& seq_ids) {
    int num_sequences = sequences.size();
    int seq_length = sequences[0].size();

    std::ifstream in;
    in.open(path, std::ios::in);

    for (int i = 0; i < num_sequences; i++) {
        in >> seq_ids[i];
        for (int j = 0; j < seq_length; j++) {
            in >> sequences[i][j];
        }
    }
}

void readMetadataFile(std::string path,
                      int& numOfSequences,
                      int& sequenceLength) {
    std::ifstream in;
    in.open(path, std::ios::in);
    in >> numOfSequences >> sequenceLength;
}