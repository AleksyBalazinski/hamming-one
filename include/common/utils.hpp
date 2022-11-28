#pragma once

#include <fstream>
#include <string>
#include <vector>

void readDataFile(std::string path, int* sequences, int numOfSequences, int seqLength) {
    std::ifstream in;
    in.open(path, std::ios::in);

    for (size_t i = 0; i < numOfSequences * seqLength; i++) {
        in >> sequences[i];
    }
    in.close();
}

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
void readDataFile(std::string path, std::vector<T>& sequences) {
    std::ifstream in;
    in.open(path, std::ios::in);

    for (size_t i = 0; i < sequences.size(); i++) {
        in >> sequences[i];
    }
    in.close();
}

void readMetadataFile(std::string path, int& numOfSequences, int& sequenceLength) {
    std::ifstream in;
    in.open(path, std::ios::in);
    in >> numOfSequences >> sequenceLength;
}

void printSequences(int* sequences, int numOfSequences, int seqLength) {
    for (int seq = 0; seq < numOfSequences; seq++) {
        printf("%d: ", seq);
        for (int j = 0; j < seqLength; j++) {
            printf("%d", sequences[seq * seqLength + j]);
        }
        printf("\n");
    }
}