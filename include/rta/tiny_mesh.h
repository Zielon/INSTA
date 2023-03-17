#ifndef REAL_TIME_AVATAR_TINY_MESH_H
#define REAL_TIME_AVATAR_TINY_MESH_H

#include <tinyobjloader/tiny_obj_loader.h>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/bounding_box.cuh>

#include "common.h"
#include "nn.h"
#include "masking.h"

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <array>

#include "Eigen/Core"

RTA_NAMESPACE_BEGIN

static int mesh_total_count = 0;

class TinyMesh {
public:
    explicit TinyMesh(const std::string& path, cudaStream_t m_inference_stream, std::shared_ptr<Masking> masking, std::shared_ptr<TinyMesh> canonical_mesh);
    ~TinyMesh();
    void print() const;
    std::vector<int> get_triangle_3_neighbours();
    std::vector<int> get_triangle_all_neighbours();
    void save(const std::string& path) const;
    void save_xyz(const std::string& path) const;

    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3i> faces;
    std::map<uint32_t, std::set<uint32_t>> vertex2faces;
    tcnn::GPUMemory<ngp::Triangle> triangles_gpu;
    tcnn::GPUMemory<ngp::Triangle> triangles_gpu_orig_order;
    std::vector<ngp::Triangle> triangles_cpu;
    std::vector<Eigen::Vector3f> samples;
    std::vector<uint32_t> samples_ids;
//    std::shared_ptr<FlannTree> tree;
    std::shared_ptr<ngp::TriangleBvh> triangle_bvh;
    std::shared_ptr<Masking> m_masking;
    Eigen::Vector3f m_centorid{};
    ngp::BoundingBox m_aabb{};
    ngp::BoundingSphere m_bsphere{};

private:
    cudaStream_t m_inference_stream;
//    NearsetNeighbours m_nn;
    int m_max_samples_triangle = 4;

    void load(const std::string& path);
    void find_bounding_sphere();
    void build_triangles(const std::shared_ptr<TinyMesh>& canonical);
    void build_tries();
    void build_flann();
    void sample_points_nn();
    void sample_points_triangle(const ngp::Triangle& t, uint32_t faceId);
    void calculate_normals();
};

RTA_NAMESPACE_END

#endif //REAL_TIME_AVATAR_TINY_MESH_H