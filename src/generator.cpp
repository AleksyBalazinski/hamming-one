#include <fstream>
#include <iostream>
#include <string>

void generateDataFile(std::string path, int l, int n) {
    std::ofstream out;
    out.open(path, std::ios::out);
    char val;
    for (long seq = 0; seq < n; seq++) {
        for (int j = 0; j < l; j++) {
            val = rand() % 2;
            out << (int)val << ' ';
        }
        out << '\n';
    }
    out.close();
}

void generateMetadataFile(std::string path, int l, int n) {
    std::ofstream out;
    out.open(path, std::ios::out);
    out << n << ' ' << l;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " [path to metadata file] [path to data file]"
                  << "[sequence length] [# of sequences]\n";
        return -1;
    }
    std::string pathToMetadata(argv[1]);
    std::string pathToData(argv[2]);

    int seqLen = std::stoi(argv[3]);
    long numOfSequences = std::stol(argv[4]);

    generateDataFile(pathToData, seqLen, numOfSequences);
    generateMetadataFile(pathToMetadata, seqLen, numOfSequences);
}