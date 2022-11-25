#pragma once

#include <string>
#include <vector>

void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength);

void readDataFile(std::string path, int *sequences, int numOfSequences, int seqLength);

void readDataFile(std::string path, std::vector<std::vector<int>> &sequences);

void printSequences(int *sequences, int numOfSequences, int seqLength);