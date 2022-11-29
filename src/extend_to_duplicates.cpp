#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::pair<int, int>> readPairs(std::string path) {
    std::ifstream in;
    in.open(path, std::ios::in);
    std::string line;
    int x, fst, snd;
    int cnt = 0;
    std::vector<std::pair<int, int>> pairs{};
    while (in >> x) {
        if (cnt % 2 == 0)
            fst = x;
        else {
            snd = x;
            pairs.emplace_back(std::make_pair(fst, snd));
        }
        cnt++;
    }
    return pairs;
}

std::unordered_map<int, std::vector<int>> getClasses(std::string path) {
    std::ifstream in;
    in.open(path, std::ios::in);

    std::unordered_map<int, std::vector<int>> classes{};
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        int rep;
        iss >> rep;

        std::vector<int> cur_class{};
        int x;
        while (iss >> x) {
            cur_class.push_back(x);
        }
        classes.insert(std::make_pair(rep, cur_class));
    }
    return classes;
}

std::vector<int> getClass(
    const std::unordered_map<int, std::vector<int>>& classes,
    int representative) {
    auto it = classes.find(representative);
    if (it == classes.cend())
        return {};

    std::vector<int> full_class((*it).second);
    full_class.push_back(representative);

    return full_class;
}

int main(int argc, char** argv) {
    std::string hamm_one_pairs_path(argv[1]);
    std::string refinement_info_path(argv[2]);
    std::string hamm_with_duplicates(argv[3]);

    auto hamm_one_pairs = readPairs(hamm_one_pairs_path);
    auto classes = getClasses(refinement_info_path);

    std::ofstream out(hamm_with_duplicates, std::ios::out);
    for (const auto& p : hamm_one_pairs) {
        auto c1 = getClass(classes, p.first);
        auto c2 = getClass(classes, p.second);
        for (int first : c1) {
            for (int second : c2) {
                out << first << ' ' << second << '\n';
            }
        }
    }
}