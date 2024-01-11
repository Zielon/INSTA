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

#ifndef REAL_TIME_AVATAR_CORE_H
#define REAL_TIME_AVATAR_CORE_H

#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/nerf_network_mapper.h>
#include <neural-graphics-primitives/common.h>
#include "neural-graphics-primitives/thread_pool.h"
#include "neural-graphics-primitives/nerf.h"

#include "common.h"
#include "flame.h"
#include "tiny_mesh.h"
#include "recorder.h"
#include "masking.h"

RTA_NAMESPACE_BEGIN

    enum class MeshType : int {
        Mesh,
        Cavity,
    };

    enum class DepthType : int {
        Flame,
        Cavity,
        Kinect
    };

    class Core : public ngp::Testbed {
    public:
        Core(ngp::ETestbedMode mode) : Testbed(mode) {};
        Core(ngp::ETestbedMode mode, const std::string &data_path) : Testbed(mode) {};
        Core(ngp::ETestbedMode mode, const std::string &data_path, const std::string &network_config_path) : Testbed(mode, data_path) {};
        Core(ngp::ETestbedMode mode, const std::string &data_path, const nlohmann::json &network_config) : Testbed(mode, data_path) {};
        ~Core();
        void load_training_data(const std::string &data_path) override;
        void post_loading() override;
        void reload_training_data(bool force = false, std::string mode = "train") override;
        void update_xforms();
        std::string get_recorder_info();
        bool is_recording();
        void imgui() override;
        bool keyboard_event() override;
        void pre_rendering() override;
        void post_rendering() override;
        virtual void load_meshes(const std::string &data_path, bool init_latent = false);
        virtual tcnn::GPUMatrix<float> render_deformed(const tcnn::GPUMatrix<float> &coords, cudaStream_t stream);
        virtual tcnn::GPUMatrix<float> surface_closest_point(const tcnn::GPUMatrix<float> &coords, cudaStream_t stream);
        virtual tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::RM> point_density_flame_closest_point(const tcnn::GPUMatrix<float> &points, const tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::RM> &density, float radius, cudaStream_t stream);
        virtual void raycast_flame(ngp::CudaRenderBuffer &render_buffer, const Eigen::Vector2i &max_res, const Eigen::Vector2f &focal_length, const Eigen::Matrix<float, 3, 4> &camera_matrix, const Eigen::Vector2f &screen_center, cudaStream_t stream);

    private:
        tcnn::GPUMatrix<float> surface_closest_point_cpu(uint32_t n_elements, float *coords, cudaStream_t stream);
        tcnn::GPUMatrix<float> surface_closest_point_gpu(uint32_t n_elements, float *coords, cudaStream_t stream);
        tcnn::GPUMatrix<float> surface_closest_point_3d_gpu(uint32_t n_elements, float *coords, cudaStream_t stream);
        tcnn::GPUMatrix<float> render_deformed_cpu(uint32_t n_elements, float *coords, cudaStream_t stream);
        tcnn::GPUMatrix<float> render_deformed_gpu(uint32_t n_elements, float *coords, cudaStream_t stream);
        void post_train_data();
        void clean_dataset();
        void update_paths_dataset();
        void test_raycasting();

        ngp::ThreadPool m_pool;
        std::shared_ptr<Recorder> m_recorder;
        SphereTracer m_tracer;
        std::vector<std::shared_ptr<TinyMesh>> m_meshes;
        std::shared_ptr<TinyMesh> m_canonical_shape = nullptr;
        std::shared_ptr<Masking> m_masking;
        tcnn::GPUMemory<float> exp_pca_gpu;
        bool m_raycast_normal = false;
        bool m_use_gpu_for_nn = true;
        bool m_ngp_menu = true;
        bool m_dump_progress = false;
        bool m_optimize_latent_code = false;
        uint32_t n_extra_dims = 0;
        uint32_t n_max_cached_bvh = 4000;
        DepthType m_depth_type = DepthType::Flame;
        MeshType m_mesh_type = MeshType::Cavity;
        std::vector<int> m_adjacency_cpu;
        std::vector<float> m_latent_codes;
        tcnn::GPUMemory<int> m_adjacency_gpu;

        // Cache
        std::map<std::string, std::shared_ptr<TinyMesh>> cached_meshes;
        std::vector<filesystem::path> m_json_paths;
        tcnn::GPUMemory<ngp::Triangle *> m_triangles_ptrs_gpu_topology;

    public:
        // Conditioning
        bool m_use_eyes = false;
        bool m_use_jaw = false;
        bool m_use_exp = true;
        bool m_use_geo = false;
        bool m_use_lips = false;
        bool m_disable_interp = false;

        int N_EXP_PARAMS = 16;
        int N_LIPS_PARAMS = 16;
        int N_GEO_PARAMS = 1;
        int N_JAW_PARAMS = 6;
        int N_EYES_PARAMS = 16;
    };

RTA_NAMESPACE_END

#endif //REAL_TIME_AVATAR_CORE_H
