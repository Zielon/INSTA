/*
 -*- coding: utf-8 -*-
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2024 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Contact: insta@tue.mpg.de
*/

#ifndef INSTANT_NGP_MASKING_H
#define INSTANT_NGP_MASKING_H

#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "filesystem/directory.h"

#include "common.h"

RTA_NAMESPACE_BEGIN

enum class FlameMasks : int {
    Boundary,
    EyeRegion,
    Face,
    Forehead,
    LeftEar,
    LeftEyeball,
    LeftEyeRegion,
    Lips,
    Neck,
    Nose,
    RightEar,
    RightEyeball,
    RightEyeRegion,
    Scalp,
    Mapping,
    Torso,
    Eyes
};

class Masking {
public:
    Masking(std::string current_canonical, filesystem::path data_path);
    uint32_t get_triangle_mask_id(uint32_t tri);
    uint32_t get_flame_triangle_mask_id(uint32_t tri);
private:
    void load_flame_masks();
    void triangle_to_mask();
    void mesh_to_mask();
    std::vector<uint32_t> load_mesh_color_mask(std::string path);

    filesystem::path data_path;
    std::vector<uint32_t> flame_to_smplx_vector;
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> faces;
    std::vector<FlameMasks> face_masks;
    std::vector<FlameMasks> flame_face_masks;
    FlameMasks get_type(std::string name);
    std::map<FlameMasks, std::vector<uint32_t>> masks;
    std::map<FlameMasks, std::vector<uint32_t>> flame_masks;
};

RTA_NAMESPACE_END

#endif //INSTANT_NGP_MASKING_H
