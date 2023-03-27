/*
 -*- coding: utf-8 -*-
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Contact: insta@tue.mpg.de
*/

#ifndef INSTANT_NGP_HELPERS_H
#define INSTANT_NGP_HELPERS_H

#pragma once

#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <stb_image/stb_image_write.h>

static void save_rgba(Eigen::Array4f *rgba_cpu, const char *path, const char *name, Eigen::Vector2i res3d, std::function<float(float)> transform) {
    uint32_t w = res3d.x();
    uint32_t h = res3d.y();

    uint8_t *pngpixels = (uint8_t *) malloc(size_t(w) * size_t(h) * 4);
    uint8_t *dst = pngpixels;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = x + res3d.x() + y * res3d.x();
            float alpha = rgba_cpu[i].w();
            *dst++ = (uint8_t) tcnn::clamp(transform(rgba_cpu[i].x()) * 255.f, 0.f, 255.f);
            *dst++ = (uint8_t) tcnn::clamp(transform(rgba_cpu[i].y()) * 255.f, 0.f, 255.f);
            *dst++ = (uint8_t) tcnn::clamp(transform(rgba_cpu[i].z()) * 255.f, 0.f, 255.f);
            *dst++ = (uint8_t) tcnn::clamp(transform(alpha) * 255.f, 0.f, 255.f);
        }
    }
    // write slice
    filesystem::path output(path);
    output = output / (std::string(name) + ".png");
    stbi_write_png(output.str().c_str(), w, h, 4, pngpixels, w * 4);
    free(pngpixels);
}

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

#endif //INSTANT_NGP_HELPERS_H
