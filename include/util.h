#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H

#include <random>
#include <fstream>
#include <iostream>

namespace efanna2e {

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N);

float* data_align(float* data_ori, unsigned point_num, unsigned& dim);

template<typename T>
T* load_data(const char* filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error" << std::endl;
        exit(-1);
    }

    in.read((char*)&dim, 4);

    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);

    T* data = new T[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * sizeof(T));
    }
    in.close();

    return data;
}

}  // namespace efanna2e

#endif  // EFANNA2E_UTIL_H
