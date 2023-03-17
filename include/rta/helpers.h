#ifndef INSTANT_NGP_HELPERS_H
#define INSTANT_NGP_HELPERS_H

#pragma once

#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <sys/types.h>
#include <dirent.h>

static std::vector<float> read_flame_params(std::string path, uint32_t max_params = 75) {
    std::ifstream in(path.c_str());
    if (!in) {
        std::cerr << "Cannot open the File : " << path << std::endl;
        return {};
    }

    std::vector<float> output;
    std::string str;

    while (std::getline(in, str))
        if (str.size() > 0 && output.size() < max_params)
            output.push_back(std::stof(str));

    in.close();
    return output;
}

static Eigen::Matrix3f read_flame_rotation(std::string path) {
    std::ifstream in(path.c_str());
    if (!in) {
        std::cerr << "Cannot open the File : " << path << std::endl;
        return {};
    }

    std::string str;
    Eigen::Matrix3f m;

    int row = 0;
    int col = 0;
    while (std::getline(in, str)) {
        float x = std::stof(str);
        m(row % 3, col) = x;
        ++row;
        if (row % 3 == 0) col += 1;
    }

    in.close();
    return m;
}

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const &vec) {
    std::vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

template<typename T>
std::string to_string_with_precision(const T a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

int static remove_directory(const char *path) {
    DIR *d = opendir(path);
    size_t path_len = strlen(path);
    int r = -1;

    if (d) {
        struct dirent *p;

        r = 0;
        while (!r && (p = readdir(d))) {
            int r2 = -1;
            char *buf;
            size_t len;

            /* Skip the names "." and ".." as we don't want to recurse on them. */
            if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
                continue;

            len = path_len + strlen(p->d_name) + 2;
            buf = static_cast<char *>(malloc(len));

            if (buf) {
                struct stat statbuf;

                snprintf(buf, len, "%s/%s", path, p->d_name);
                if (!stat(buf, &statbuf)) {
                    if (S_ISDIR(statbuf.st_mode))
                        r2 = remove_directory(buf);
                    else
                        r2 = unlink(buf);
                }
                free(buf);
            }
            r = r2;
        }
        closedir(d);
    }

    if (!r)
        r = rmdir(path);

    return r;
}

#endif //INSTANT_NGP_HELPERS_H
