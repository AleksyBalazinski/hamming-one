#pragma once

#include <string>
#include "Array.h"

void readDataFile(std::string path, Array<bool> *sequences, int numOfSequences);
void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength);
void printSequences(Array<bool> *sequences, size_t numOfSequences, std::ostream &out);
