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

#ifndef INSTANT_NGP_AFFINE_CUH
#define INSTANT_NGP_AFFINE_CUH

#include "Eigen/Core"
#include "neural-graphics-primitives/triangle.cuh"

using namespace Eigen;

__host__ __device__ inline Matrix4f tri2projection(Vector3f point, const ngp::Triangle &tri) {
    Matrix3f tbn = Eigen::Matrix3f::Identity();
    Vector4f t = Vector4f::Constant(1.f);
    t.head<3>() = tri.a;
    Matrix4f M = Matrix4f::Identity();
    Matrix3f S = Matrix3f::Identity() * tri.scaling;
    tbn = tri.tbn();
    if (tri.mask_id == 1)
        tbn = tbn * S;
    M.block<3, 3>(0, 0) = tbn;
    M.block<4, 1>(0, 3) = t;

    return M;
}

__host__ __device__ inline Vector4f get_q(Quaternionf q) {
    return Vector4f(q.w(), q.x(), q.y(), q.z());
}

__device__ Eigen::Matrix4f interpolate_transformation(Vector3f point, const int *adjacency, const ngp::Triangle *__restrict__ triangles, const ngp::Triangle &tri, bool invert) {
    auto id = tri.id;
    float sum = 0;
    float factor = 4;
    constexpr int MAX_N = 3;

#pragma unroll
    for (int i = 0; i < MAX_N; ++i) {
        auto edge_tri_id = adjacency[id * MAX_N + i];
        if (edge_tri_id >= 0) {
            ngp::Triangle edge_tri = triangles[edge_tri_id];
            sum += __expf(-factor * (edge_tri.centroid() - point).norm());
        }
    }

    sum += __expf(-1.f * (tri.centroid() - point).norm());

    Matrix4f A = Matrix4f::Zero();
    Vector3f T = Vector3f::Zero();
    Matrix4f M = Matrix4f::Zero();

#pragma unroll
    for (int i = 0; i < MAX_N; ++i) {
        auto edge_tri_id = adjacency[id * MAX_N + i];
        if (edge_tri_id >= 0) {
            ngp::Triangle edge_tri = triangles[edge_tri_id];
            auto a = __expf(-factor * (edge_tri.centroid() - point).norm());
            auto w = a / sum;
//            Vector4f q = get_q(edge_tri.tbn_quat());
//            if (q[0] < 0) q = -q;
//            A = w * q * q.adjoint() + A;
//            T += w * edge_tri.a;
            M += w * tri2projection(point, edge_tri);
        }
    }

    float a = __expf(-1.f * (tri.centroid() - point).norm());
    auto w = a / sum;
//    Vector4f q = get_q(tri.tbn_quat());
//    if (q[0] < 0) q = -q;
//    A = w * q * q.adjoint() + A;
//    T += w * tri.a;
    M += w * tri2projection(point, tri);

//    SelfAdjointEigenSolver<Matrix4f> eig(A);
//    q = eig.eigenvectors().col(3);
//    Matrix3f R = Quaternionf(q.w(), q.x(), q.y(), q.z()).normalized().toRotationMatrix();
//
//    M = Matrix4f::Identity();
//    M.block<3, 3>(0, 0) = R;
//    M.block<3, 1>(0, 3) = T;

    if (invert)
        return M.inverse();
    return M;
}

__host__ __device__ inline Vector3f transform_point(const Vector3f &deform_sample, const Matrix4f &M_world, const Matrix4f &M_canon) {
    Vector4f x_deform = Vector4f::Constant(1.f);
    x_deform.head<3>() = deform_sample;
    Vector4f x_def_local = M_world * x_deform;
    Vector4f x = M_canon * x_def_local;
    return x.head<3>();
}

__host__ __device__ inline Vector3f transform_direction(const Vector3f &deform_dir, const Matrix4f &M_world, const Matrix4f &M_canon) {
    Matrix3f M = M_world.block<3, 3>(0, 0);
    Vector3f x_dir_local = M * deform_dir;
    M = M_canon.block<3, 3>(0, 0);
    Vector3f x = M * x_dir_local;
    return x.normalized();
}

inline void save_point_cloud(const std::string &name, const std::vector<Vector3f> &points) {
    std::ofstream out;
    out.open(name);
    for (auto vert: points) {
        out << vert.x() << " " << vert.y() << " " << vert.z() << std::endl;
    }
    out.close();
}

#endif //INSTANT_NGP_AFFINE_CUH