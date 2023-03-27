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

#include "rta/masking.h"

#include <iostream>
#include <fstream>
#include <map>
#include <unordered_map>

#include "filesystem/directory.h"
#include <tinyobjloader/tiny_obj_loader.h>

namespace fs = filesystem;
using namespace Eigen;


rta::Masking::Masking(std::string current_canonical, filesystem::path _data_path) {
    data_path = _data_path;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, current_canonical.c_str())) {
        throw std::runtime_error(warn + err);
    }

    Array3f offset = {0.5f, 0.5f, 0.5f};
    auto num_vertices = attrib.vertices.size();
    for (int i = 0; i < num_vertices; i += 3) {
        Array3f vertex = {attrib.vertices[i + 0] + 0.5f, attrib.vertices[i + 1] + 0.5f, attrib.vertices[i + 2] + 0.5f};
        vertices.emplace_back(vertex.matrix());
    }

    for (size_t s = 0; s < shapes.size(); s++) {
        auto num_faces = shapes[s].mesh.indices.size();
        for (int i = 0; i < num_faces; i += 3) {
            uint32_t a = shapes[s].mesh.indices[i + 0].vertex_index;
            uint32_t b = shapes[s].mesh.indices[i + 1].vertex_index;
            uint32_t c = shapes[s].mesh.indices[i + 2].vertex_index;
            faces.emplace_back(a, b, c);
        }
    }

//    load_flame_masks();
    mesh_to_mask();
    triangle_to_mask();
}

void rta::Masking::load_flame_masks() {
    std::vector<fs::path> paths;
    for (const auto &path: fs::directory{"flame_masks"}) {
        if (path.is_file()) {
            paths.emplace_back(path);
        }
    }

    for (auto path: paths) {
        std::ifstream in(path.str().c_str());
        if (!in) {
            std::cerr << "Cannot open the File : " << path << std::endl;
            return;
        }
        std::vector<uint32_t> output;
        std::string str;
        while (std::getline(in, str))
            output.push_back(std::stoi(str));
        in.close();
        flame_masks[get_type(path.basename())] = output;
    }

    flame_to_smplx_vector = flame_masks[FlameMasks::Mapping];
}

rta::FlameMasks rta::Masking::get_type(std::string name) {
    if (name == "boundary") return FlameMasks::Boundary;
    if (name == "eye_region") return FlameMasks::EyeRegion;
    if (name == "face") return FlameMasks::Face;
    if (name == "forehead") return FlameMasks::Forehead;
    if (name == "left_ear") return FlameMasks::LeftEar;
    if (name == "left_eyeball") return FlameMasks::LeftEyeball;
    if (name == "left_eye_region") return FlameMasks::LeftEyeRegion;
    if (name == "lips") return FlameMasks::Lips;
    if (name == "neck") return FlameMasks::Neck;
    if (name == "nose") return FlameMasks::Nose;
    if (name == "right_ear") return FlameMasks::RightEar;
    if (name == "right_eyeball") return FlameMasks::RightEyeball;
    if (name == "right_eye_region") return FlameMasks::RightEyeRegion;
    if (name == "scalp") return FlameMasks::Scalp;
    if (name == "smplx_flame") return FlameMasks::Mapping;

    return FlameMasks::Torso;
}

uint32_t rta::Masking::get_triangle_mask_id(uint32_t id) {
    return uint32_t(face_masks[id]);
}

uint32_t rta::Masking::get_flame_triangle_mask_id(uint32_t id) {
    return uint32_t(flame_face_masks[id]);
}

void rta::Masking::triangle_to_mask() {
    face_masks.clear();
    flame_face_masks.clear();
    face_masks.resize(faces.size(), FlameMasks::Torso);
    flame_face_masks.resize(faces.size(), FlameMasks::Torso);
    int face_id = 0;
    for (auto face: faces) {
        for (auto &masks_type: {masks}) {
            bool is_flame = masks_type.size() >= 15;
            for (auto const &x: masks_type) {
                auto mask_region = x.first;
                auto flame_mask_vertices = x.second;
                for (auto vertex_id: flame_mask_vertices) {
//                auto vertex_id = flame_to_smplx_vector[flame_id]; // mask in SMPLX topology
                    int hit = 0;
                    for (int i = 0; i < 3; ++i)
                        if (face[i] == vertex_id) hit++;
                    if (hit > 0) {
                        if (is_flame) {
                            flame_face_masks[face_id] = mask_region;
                        } else {
                            face_masks[face_id] = mask_region;
                        }
                        break;
                    }
                }
            }
        }
        face_id++;
    }
}

void rta::Masking::mesh_to_mask() {
    masks[FlameMasks::Lips] = load_mesh_color_mask((data_path / "mesh_masks" / "lips.obj").str());
    masks[FlameMasks::Face] = load_mesh_color_mask((data_path / "mesh_masks" / "face.obj").str());
    masks[FlameMasks::Boundary] = load_mesh_color_mask((data_path / "mesh_masks" / "boundary.obj").str());
    masks[FlameMasks::RightEar] = load_mesh_color_mask((data_path / "mesh_masks" / "right_ear.obj").str());
    masks[FlameMasks::LeftEar] = load_mesh_color_mask((data_path / "mesh_masks" / "left_ear.obj").str());
    masks[FlameMasks::EyeRegion] = load_mesh_color_mask((data_path / "mesh_masks" / "eye_region.obj").str());
}

std::vector<uint32_t> rta::Masking::load_mesh_color_mask(std::string path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
        throw std::runtime_error(warn + err);
    }

    std::vector<uint32_t> mask;
    auto num_colors = attrib.colors.size();
    int id = 0;
    for (int i = 0; i < num_colors; i += 3) {
        Array3f vertex = {attrib.colors[i + 0], attrib.colors[i + 1], attrib.colors[i + 2]};
        if (vertex.x() >= 1.0 && vertex.y() == 0.0 && vertex.z() == 0.0)
            mask.emplace_back(id);
        id++;
    }

    return mask;
}
