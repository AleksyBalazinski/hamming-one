#include <string>
#include <vector>
#include <fstream>
#include <iostream>

void readDataFile(std::string path, std::vector<std::vector<int>> &sequences);
void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength);

bool isDistOne(std::vector<int> s1, std::vector<int> s2)
{
    int dist = 0;
    for (int i = 0; i < s1.size(); i++)
    {
        if (s1[i] != s2[i])
            dist++;
        if (dist > 1)
            return false;
    }

    return dist == 1;
}

void hammingOne(std::vector<std::vector<int>> sequences)
{
    for (int i = 0; i < sequences.size(); i++)
    {
        for (int j = 0; j < sequences.size(); j++)
        {
            if (isDistOne(sequences[i], sequences[j]))
            {
                std::cout << i << ' ' << j << '\n';
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);

    int numOfSequences, seqLength;
    readMetadataFile(pathToMetadata, numOfSequences, seqLength);

    std::vector<std::vector<int>> sequences(numOfSequences, std::vector<int>(seqLength));
    readDataFile(pathToData, sequences);
#ifdef DEBUG
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
#endif
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