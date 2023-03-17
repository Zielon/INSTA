#ifndef REAL_TIME_AVATAR_FLAME_H
#define REAL_TIME_AVATAR_FLAME_H

#ifdef USE_TORCH

#include <torch/torch.h>

#include "common.h"


RTA_NAMESPACE_BEGIN

class Flame {
public:
    Flame();
    torch::Tensor eval(const torch::Tensor &shape_params, const torch::Tensor &expression_params, const torch::Tensor &pose_params);
    void export_mesh(const torch::Tensor &vertices);

private:
    torch::Tensor vertices;
    torch::Tensor faces;
    torch::Tensor posedirs;
    torch::Tensor shapedirs;
    torch::Tensor parents;
    torch::Tensor J_regressor;
    torch::Tensor lbs_weights;
    torch::Tensor kid;
    int n_shape = 300;

    void load();
    std::vector<torch::Tensor> lbs(const torch::Tensor &betas, const torch::Tensor &full_pose, const torch::Tensor &template_vertices);
};

RTA_NAMESPACE_END

#endif

#endif //REAL_TIME_AVATAR_FLAME_H
