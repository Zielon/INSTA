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

#include <rta/flame.h>
#include <iostream>
#include <rta/debug.h>

#ifdef USE_TORCH
#include <torch/script.h>

using namespace torch::indexing;

rta::Flame::Flame() {
    load();

    auto shape_params = torch::zeros({1, 300}).cuda();
    auto expression_params = torch::zeros({1, 100}).cuda();
    auto pose_params = torch::zeros({1, 6}).cuda();

    eval(this->kid, expression_params, pose_params);
}

void rta::Flame::load() {
    std::string str = "data/FLAME2020/flame_model.pt";
    torch::jit::script::Module container = torch::jit::load(str);

    this->vertices = container.attr("vertices").toTensor().cuda();
    this->faces = container.attr("faces").toTensor();
    this->posedirs = container.attr("posedirs").toTensor().cuda();
    this->shapedirs = container.attr("shapedirs").toTensor().cuda();
    this->parents = container.attr("parents").toTensor().cuda();
    this->J_regressor = container.attr("J_regressor").toTensor().cuda();
    this->lbs_weights = container.attr("lbs_weights").toTensor().cuda();
    this->kid = container.attr("testshape").toTensor().cuda();
}

torch::Tensor rta::Flame::eval(const torch::Tensor &shape_params, const torch::Tensor &expression_params, const torch::Tensor &pose_params) {
    auto batch_size = shape_params.sizes()[0];
    auto betas = torch::cat({shape_params, expression_params}, 1);
    auto neck_pose_params = torch::zeros({1, 3}).cuda();
    auto eye_pose_params = torch::zeros({1, 6}).cuda();
    auto full_pose = torch::cat({pose_params.index({Slice(), Slice(None, 3)}), neck_pose_params, pose_params.index({Slice(), Slice(3, None)}), eye_pose_params}, 1);
    auto template_vertices = this->vertices.unsqueeze(0).expand({batch_size, -1, -1});

    auto vertices = lbs(betas, full_pose, template_vertices)[0];

//    this->export_mesh(vertices);

    return vertices;
}

torch::Tensor blend_shapes(const torch::Tensor &betas, const torch::Tensor &shape_disps) {
    return torch::einsum("bl,mkl->bmk", {betas, shape_disps});
}

torch::Tensor vertices2joints(const torch::Tensor &J_regressor, const torch::Tensor &vertices) {
    return torch::einsum("bik,ji->bjk", {vertices, J_regressor});
}

torch::Tensor batch_rodrigues(const torch::Tensor &rot_vecs) {

    auto options =
            torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, 1);

    auto batch_size = rot_vecs.sizes()[0];
    auto angle = torch::norm(rot_vecs + 1e-8, 2, 1, true);
    auto rot_dir = rot_vecs / angle;
    auto cos = torch::unsqueeze(torch::cos(angle), 1);
    auto sin = torch::unsqueeze(torch::sin(angle), 1);
    auto r = torch::split(rot_dir, 1, 1);
    auto rx = r[0];
    auto ry = r[1];
    auto rz = r[2];
    auto K = torch::zeros({batch_size, 3, 3}).cuda();
    auto zeros = torch::zeros({batch_size, 1}).cuda();
    K = torch::cat({zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros}, 1).view({batch_size, 3, 3});
    auto ident = torch::eye(3).unsqueeze(0).cuda();
    return ident + sin * K + (1 - cos) * torch::bmm(K, K);
}

torch::Tensor Rt(const torch::Tensor & R, const torch::Tensor & t)
{
    auto options = torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).mode(torch::kConstant);
    return torch::cat({torch::nn::functional::pad(R, options.value(0)), torch::nn::functional::pad(t, options.value(1))}, 2);
}

// x[:,:2].chunk(2,1)
// using namespace torch::indexing;
// auto partial_gates = gates.index({"...", Slice(None, 2)}).chunk(4, 1);

// Python             C++ (assuming `using namespace torch::indexing`)
// -------------------------------------------------------------------
// 0                  0
// None               None
// ...                "..." or Ellipsis
// :                  Slice()
// start:stop:step    Slice(start, stop, step)
// True / False       true / false
// [[1, 2]]           torch::tensor({{1, 2}})

std::vector<torch::Tensor> batch_rigid_transform(const torch::Tensor &rot_mats, const torch::Tensor &joint, const torch::Tensor &parents) {
    auto joints = torch::unsqueeze(joint, -1);
    //
    auto rel_joints = joints.clone();
    rel_joints.index({Slice(), Slice(1, None)}) -= joints.index({Slice(), parents.index({Slice(1, None)})});
    //
    auto transforms_mat = Rt(rot_mats.view({-1, 3, 3}), rel_joints.reshape({-1, 3, 1})).reshape({-1, joints.sizes()[1], 4, 4});

    auto transform_chain = std::vector<torch::Tensor>{transforms_mat.index({Slice(), 0})};
    for(int i = 1; i < parents.sizes()[0]; ++i){
        // Subtract the joint location at the rest pose
        // No need for rotation, since it's identity when at rest
        auto p = parents[i].item().toInt();
        auto curr_res = torch::matmul(transform_chain[p], transforms_mat.index({Slice(), i}));
        transform_chain.push_back(curr_res);
    }

    auto transforms = torch::stack(torch::TensorList(transform_chain), 1);

    //
    //    # The last column of the transformations contains the posed joints
    auto posed_joints = transforms.index({Slice(), Slice(), Slice(None, 3), 3});


    auto options = torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).mode(torch::kConstant).value(0);
    auto joints_homogen = torch::nn::functional::pad(joints, options);

    options = torch::nn::functional::PadFuncOptions({3, 0, 0, 0, 0, 0, 0, 0}).mode(torch::kConstant).value(0);
    auto rel_transforms = transforms - torch::nn::functional::pad(torch::matmul(transforms, joints_homogen), options);
    //

    return {posed_joints, rel_transforms};
}

std::vector<torch::Tensor> rta::Flame::lbs(const torch::Tensor &betas, const torch::Tensor &pose, const torch::Tensor &v_template) {

    auto batch_size = std::max(betas.sizes()[0], betas.sizes()[0]);

    auto pp = torch::einsum("bl,mkl->bmk", {torch::zeros({1, 400}).cuda(), torch::zeros({5023, 3, 400}).cuda()});

    //    # Add shape contribution
    auto v_shaped = v_template + blend_shapes(betas, shapedirs);
    //
    //    # Get the joints
    //    # NxJx3 array
    auto J = vertices2joints(J_regressor, v_shaped);
    //
    //    # 3. Add pose blend shapes
    //    # N x J x 3 x 3
    auto ident = torch::eye(3).cuda();
    //    if pose2rot:
    auto rot_mats = batch_rodrigues(pose.view({-1, 3})).view({batch_size, -1, 3, 3});

    auto pose_feature = (rot_mats.index({Slice(), Slice(1, None), "..."}) - ident).view({batch_size, -1});

    // # (N x P) x (P, V * 3) -> N x V x 3
    auto pose_offsets = torch::matmul(pose_feature, posedirs).view({batch_size, -1, 3});
    //    else:
    //        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
    //        rot_mats = pose.view(batch_size, -1, 3, 3)
    //
    //        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
    //                                    posedirs).view(batch_size, -1, 3)
    //
    auto v_posed = pose_offsets + v_shaped;
    //    # 4. Get the global joint location

    //    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    auto num_joints = J_regressor.sizes()[0];

    auto results = batch_rigid_transform(rot_mats, J, parents);
    auto J_transformed = results[0];
    auto A = results[1];
    //
    //    # 5. Do skinning:
    //    # W is N x V x (J + 1)
    auto W = lbs_weights.unsqueeze(0).expand({batch_size, -1, -1}).contiguous();
    A = A.view({batch_size, num_joints, 16}).contiguous();

    auto T = torch::matmul(W, A);
    T = T.view({batch_size, -1, 4, 4});
    auto homogen_coord = torch::ones({batch_size, v_posed.sizes()[1], 1});
    auto v_posed_homo = torch::cat({v_posed.cpu(), homogen_coord}, 2);
    auto v_homo = torch::matmul(T, torch::unsqueeze(v_posed_homo, -1));

    auto verts = v_homo.index({"...", Slice(None, 3), 0});

    return {verts, J_transformed};
}

void rta::Flame::export_mesh(const torch::Tensor &flame_vertices) {
    auto v = flame_vertices[0].cpu();
    auto f = this->faces.cpu();

    std::ofstream out;
    out.open ("mesh.obj");

    auto numVerts = this->vertices.sizes()[0];
    for (int i = 0; i < numVerts; ++i){
        float x = v.index({i, 0}).item().toFloat() * 1000.f;
        float y = v.index({i, 1}).item().toFloat() * 1000.f;
        float z = v.index({i, 2}).item().toFloat() * 1000.f;
        out << "v " << x << " " << y << " " << z << std::endl;
    }

    auto numFaces = this->faces.sizes()[0];
    for (int i = 0; i < numFaces; ++i){
        int x = f.index({i, 0}).item().toInt() + 1;
        int y = f.index({i, 1}).item().toInt() + 1;
        int z = f.index({i, 2}).item().toInt() + 1;
        out << "f " << x << " " << y << " " << z << std::endl;
    }
    out.close();

    std::cout << "Mesh saved!" << std::endl;
}

#endif