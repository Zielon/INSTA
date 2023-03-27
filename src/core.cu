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

#include <rta/core.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/bvh.cuh>
#include <neural-graphics-primitives/random_val.cuh>

#include "filesystem/directory.h"

#include "rta/debug.h"
#include "rta/helpers.h"
#include "rta/affine.cuh"
#include "rta/tiny_mesh.h"
#include "neural-graphics-primitives/thread_pool.h"

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

constexpr int MAX_UINT = std::numeric_limits<uint32_t>::max();
static constexpr const char *DepthModeStr = "Flame\0Cavity\0Kinect\0KinectFiltered\0\0";
static constexpr const char *MeshModeStr = "Mesh\0Cavity\0\0";

inline float linear_to_db(float x) {
    return -10.f * logf(x) / logf(10.f);
}

inline static uint32_t fast_mod(const uint32_t input, const uint32_t ceil) {
    return input < ceil ? input : input % ceil;
}

inline __host__ __device__ float srgb_to_linear(float srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    } else {
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

__host__ __device__ Vector3f warp_direction(const Vector3f &dir) {
    return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ Vector3f unwarp_direction(const Vector3f &dir) {
    return dir * 2.0f - Vector3f::Ones();
}

inline __host__ __device__ Eigen::Array3f srgb_to_linear(const Eigen::Array3f &x) {
    return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

__global__ void shade_kernel_sdf(
        const uint32_t n_elements,
        ngp::BoundingBox aabb,
        Vector3f *__restrict__ positions,
        Vector3f *__restrict__ normals,
        ngp::SdfPayload *__restrict__ payloads,
        Array4f *__restrict__ frame_buffer,
        float *__restrict__ depth_buffer,
        bool color_normal,
        Vector3f center,
        bool render_depth
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    ngp::SdfPayload &payload = payloads[i];
    if (!aabb.contains(positions[i])) {
        return;
    }

    Array3f color;
    Vector3f normal = normals[i].normalized();
    Vector3f pos = positions[i];

    Vector3f lights[3];
    lights[0] = center + Vector3f{0, 0, 1.f};
    lights[1] = center + Vector3f{1.5f, 0.25, 0.5f};
    lights[2] = center + Vector3f{-1.5f, 0.25, 0.5f};

    Vector3f cos{0, 0, 0};
    for (const auto &light: lights) {
        cos += 1 / 3.f * fminf(1.f, fmaxf(normal.dot(light), 0)) * Vector3f{0.75, 0.75, 0.75};
    }

    color = color_normal ? srgb_to_linear(0.5f * normal.array() + Array3f::Constant(0.5f)) : srgb_to_linear(cos.array());
    frame_buffer[payload.idx] = {color.x(), color.y(), color.z(), 1.0f};
    depth_buffer[payload.idx] = pos.z();
}

void rta::Core::imgui() {
    bool mapper_network = typeid(*m_nerf_network.get()) == typeid(ngp::NerfNetworkMapper<ngp::precision_t>);

    ImGui::Begin("Instant Volumetric Avatars", nullptr);

    auto str = m_data_path.basename();
    if (m_data_path.is_file())
        str = m_data_path.parent_path().basename();

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);

    ImGui::Text("Actor     = %s", str.c_str());
    ImGui::Text("Loaded    = %d", m_nerf.training.dataset.n_training_images);
    ImGui::Text("Total     = %u", m_nerf.training.dataset.n_all_images);
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
    ImGui::Text("Latent    = %d", n_extra_dims);
    if (m_use_exp)
        ImGui::Text("   Exp    = %d", N_EXP_PARAMS);
    if (m_use_jaw)
        ImGui::Text("   Jaw    = %d", N_JAW_PARAMS);
    if (m_use_eyes)
        ImGui::Text("   Gaze   = %d", N_EYES_PARAMS);
    if (m_use_geo)
        ImGui::Text("   Geo    = %d", N_GEO_PARAMS);
    if (m_use_lips)
        ImGui::Text("   Lips   = %d", N_LIPS_PARAMS);
    if (mapper_network) {
        ImGui::Separator();
        ImGui::Text("   Mapper = %d", m_nerf_network->get_mapper_output());
    }
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 53, 184, 255));
    ImGui::Text("Canonical = %zu", m_canonical_shape->faces.size());
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
    ImGui::Text("Steps     = %d/%d", m_training_step, m_recorder->m_training_steps_wait);
    ImGui::PopStyleColor();
    ImGui::Checkbox("Raycast FLAME", &m_raycast_flame_mesh);
    if (m_raycast_flame_mesh) {
        ImGui::Checkbox("   normal/cosine", &m_raycast_normal);
    }
    ImGui::Checkbox("Render ground truth", &m_render_ground_truth);
    ImGui::Checkbox("Render deformed", &m_render_deformed);
    ImGui::Checkbox("Occ Grid Isosurface", &m_iso_surface_occ_grid);
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
    ImGui::Checkbox("SPEED UP", &m_offscreen_rendering);
    ImGui::PopStyleColor();
    ImGui::PushItemWidth(120);

    reset_accumulation();

    if (m_training_step > 500) {
        m_nerf.training.seg_mask_supervision_lambda = m_network_config["seg_weight"];
        m_nerf.training.penalize_density_alpha = m_network_config["penalize_density_alpha"];
    }

    if (m_training_step > 100) {
        m_nerf.training.beta = m_network_config["beta_loss"];
    }

    ImGui::PushItemWidth(120);
    ImGui::InputInt("Current Mesh", &m_target_deform_frame, 2);

    m_recorder->imgui();

    if (!m_recorder->is_recording())
        m_target_deform_frame = tcnn::clamp<int>(m_target_deform_frame, 0, m_meshes.size() - 1);

    m_nerf.extra_dim_idx_for_inference = m_target_deform_frame;

    if (m_ngp_menu) Testbed::imgui();

    m_recorder->step();

    ImGui::Separator();
    ImGui::Text("Control");
    ImGui::Text("WSDA - camera\nF - raycast FLAME\nK - show cameras\nLeft mouse buttons");
}

void rta::Core::post_rendering() {
    if (m_recorder->is_recording())
        m_recorder->dump_frame_buffer();
}

void rta::Core::load_training_data(const std::string &data_path) {
//    m_nerf.training.optimize_extrinsics = m_render_ngp;
    if (m_network_config.contains("disable_interp")) m_disable_interp = m_network_config["disable_interp"];
    if (m_network_config.contains("dump_progress")) m_dump_progress = m_network_config["dump_progress"];
    if (m_network_config.contains("optimize_latent_code")) m_optimize_latent_code = m_network_config["optimize_latent_code"];
    if (m_network_config.contains("max_cached_bvh")) n_max_cached_bvh = m_network_config["max_cached_bvh"];

    m_nerf.training.depth_supervision_lambda = m_network_config["depth_weight"];
    N_GEO_PARAMS = m_network_config["geo_params"];
    N_EXP_PARAMS = m_network_config["exp_params"];
    N_EYES_PARAMS = N_EXP_PARAMS;

    m_use_eyes = m_network_config["use_eyes"];
    m_use_geo = N_GEO_PARAMS > 0;
    m_use_exp = N_EXP_PARAMS > 0;

    Testbed::load_training_data(data_path);
    load_meshes(data_path);
    reset_network();
    post_train_data();
}

void rta::Core::post_loading() {
    if (!m_recorder) m_recorder.reset(new Recorder(this));
    m_reenact = m_nerf.training.dataset.reenact;
    if (m_network_config.contains("horizontal_angle")) m_recorder->m_horizontal_angle = m_network_config["horizontal_angle"];
    if (m_network_config.contains("horizontal_normals")) m_recorder->m_horizontal_normals = m_network_config["horizontal_normals"];
}

void rta::Core::load_meshes(const std::string &data_path, bool init_latent) {
    auto n_images = m_nerf.training.dataset.n_images;

    if (m_use_exp) n_extra_dims += N_EXP_PARAMS;
    if (m_use_geo) n_extra_dims += N_GEO_PARAMS;

    std::vector<float> exp_pca_cpu(N_EXP_PARAMS * (n_images + 1));

    float *exp_dst = exp_pca_cpu.data();
    for (uint32_t i = 0; i < n_images; ++i) {
        auto path_exp = m_nerf.training.dataset.exp_paths[i];
        auto exp = read_flame_params(path_exp, N_EXP_PARAMS);
        for (int p = 0; p < N_EXP_PARAMS; ++p) exp_dst[p] = exp[p];

        exp_dst += N_EXP_PARAMS;
    }

    std::vector<float> extra_dims_cpu(n_extra_dims * (n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
    float *dst = extra_dims_cpu.data();
    if (n_extra_dims > 0)
        for (uint32_t i = 0; i < n_images; ++i) {
            auto path_exp = m_nerf.training.dataset.exp_paths[i];
            uint32_t j = 0;
            if (m_use_exp) {
                auto exp = read_flame_params(path_exp, N_EXP_PARAMS);
                for (int p = 0; p < N_EXP_PARAMS; ++p) {
                    dst[j] = exp[p];
                    if (init_latent && j >= int(n_extra_dims / 2)) {
                        dst[j] = m_latent_codes[p];
                    }
                    j++;
                }
            }
            if (m_use_geo) {
                for (int p = 0; p < N_GEO_PARAMS; ++p) dst[j++] = 0.f;
            }

            dst += n_extra_dims;
        }

    m_nerf.training.extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);
    exp_pca_gpu.resize_and_copy_from_host(exp_pca_cpu);
    m_nerf.training.dataset.n_extra_learnable_dims = n_extra_dims;
    m_nerf.training.optimize_extra_dims = m_optimize_latent_code;
    m_dataset_paths.optimize_latent_code = m_optimize_latent_code;

    auto progress = tlog::progress(n_images);
    std::atomic<int> n_loaded{0};

    auto current_canonical = m_data_path / "canonical.obj";
    if (m_data_path.is_file())
        current_canonical = m_data_path.parent_path() / "canonical.obj";

    m_masking = std::make_shared<Masking>(current_canonical.str(), filesystem::path("."));
    m_canonical_shape = std::make_shared<TinyMesh>(current_canonical.str(), m_stream.get(), m_masking, nullptr);

    auto face = std::make_shared<TinyMesh>("average.obj", m_stream.get(), m_masking, m_canonical_shape);
    m_render_aabb = face->m_aabb;
    m_render_bsphere = face->m_bsphere;

    if (m_nerf.training.optimize_extra_dims) return;

    std::vector<ngp::TriangleBvhNode *> bvh_ptrs;
    std::vector<ngp::Triangle *> triangle_ptrs;
    std::vector<ngp::Triangle *> triangle_ptrs_topology;

    m_meshes.resize(n_images);
    bvh_ptrs.resize(n_images);
    triangle_ptrs.resize(n_images);
    triangle_ptrs_topology.resize(n_images);

    std::mutex mutex;

    tlog::info() << "Loading meshes!";

    m_adjacency_cpu = m_canonical_shape->get_triangle_3_neighbours();
    m_adjacency_gpu.resize_and_copy_from_host(m_adjacency_cpu);

    ngp::ThreadPool pool;

    pool.parallelFor<size_t>(0, n_images, [&](size_t i) {
        auto path = m_nerf.training.dataset.mesh_paths[i];
        std::shared_ptr<TinyMesh> tiny_mesh = nullptr;
        if (cached_meshes.count(path)) {
            tiny_mesh = cached_meshes[path];
        } else {
            tiny_mesh = std::make_shared<TinyMesh>(path, m_stream.get(), m_masking, m_canonical_shape);
            if (cached_meshes.size() < n_max_cached_bvh) {
                std::lock_guard<std::mutex> guard(mutex);
                cached_meshes[path] = tiny_mesh;
            }
        }

        m_meshes[i] = tiny_mesh;
        bvh_ptrs[i] = tiny_mesh->triangle_bvh->nodes_gpu();
        triangle_ptrs[i] = tiny_mesh->triangles_gpu.data();
        triangle_ptrs_topology[i] = tiny_mesh->triangles_gpu_orig_order.data();
        progress.update(++n_loaded);
    });

    m_bvh_ptrs_gpu.resize_and_copy_from_host(bvh_ptrs);
    m_triangles_ptrs_gpu.resize_and_copy_from_host(triangle_ptrs);
    m_triangles_ptrs_gpu_topology.resize_and_copy_from_host(triangle_ptrs_topology);

    m_canon_bvh_gpu = m_canonical_shape->triangle_bvh->nodes_gpu();
    m_canon_tris_gpu = m_canonical_shape->triangles_gpu.data();

    tlog::success() << "Loaded " << m_meshes.size() << " meshes after " << tlog::durationToString(progress.duration());
}

tcnn::GPUMatrix<float> rta::Core::surface_closest_point(const tcnn::GPUMatrix<float> &coords, cudaStream_t stream) {
    if (m_use_gpu_for_nn && coords.m() == sizeof(ngp::NerfPosition) / sizeof(float))
        return surface_closest_point_3d_gpu(coords.n_elements(), coords.data(), stream);

    return surface_closest_point_gpu(coords.n_elements(), coords.data(), stream);
}

tcnn::GPUMatrix<float> rta::Core::render_deformed(const tcnn::GPUMatrix<float> &coords, cudaStream_t stream) {
    return render_deformed_gpu(coords.n_elements(), coords.data(), stream);
}

void rta::Core::raycast_flame(
        ngp::CudaRenderBuffer &render_buffer,
        const Vector2i &max_res,
        const Vector2f &focal_length,
        const Eigen::Matrix<float, 3, 4> &camera_matrix,
        const Vector2f &screen_center,
        cudaStream_t stream) {

    reset_accumulation();
    m_tracer.enlarge(max_res.x() * max_res.y());
    float plane_z = m_slice_plane_z + m_scale;

    m_tracer.init_rays_from_camera(
            render_buffer.spp(),
            render_buffer.in_resolution(),
            focal_length,
            camera_matrix,
            screen_center,
            get_scaled_parallax_shift(),
            m_snap_to_pixel_centers,
            m_render_aabb,
            get_floor_y(),
            plane_z,
            m_aperture_size,
            m_envmap.envmap->params_inference(),
            m_envmap.resolution,
            render_buffer.frame_buffer(),
            render_buffer.depth_buffer(),
            nullptr,
            0,
            stream
    );

    auto mesh = m_canonical_shape;
    if (m_render_deformed)
        mesh = m_meshes[m_target_deform_frame];

    uint32_t n_hit = m_tracer.trace_bvh(mesh->triangle_bvh.get(), mesh->triangles_gpu.data(), stream);
    ngp::RaysSdfSoa &rays_hit = m_tracer.rays_init();

    linear_kernel(shade_kernel_sdf, 0, stream,
                  n_hit,
                  m_render_aabb,
                  rays_hit.pos.data(),
                  rays_hit.normal.data(),
                  rays_hit.payload.data(),
                  render_buffer.frame_buffer(),
                  render_buffer.depth_buffer(),
                  m_raycast_normal,
                  m_canonical_shape->m_centorid,
                  m_recorder->m_render_train_depth
    );
}

bool rta::Core::keyboard_event() {
    Testbed::keyboard_event();

    int value = m_target_deform_frame;

    if (m_render_ground_truth) {
        m_target_deform_frame = find_best_training_view(m_nerf.training.view);
    }

    if (value != m_target_deform_frame) {
        reset_accumulation();
    }

    if (ImGui::IsKeyPressed('F')) {
        m_raycast_flame_mesh = !m_raycast_flame_mesh;
    }

    if (ImGui::IsKeyPressed('T')) {
        reset_camera();
    }

    if (ImGui::IsKeyPressed('K')) {
        m_nerf.visualize_cameras = !m_nerf.visualize_cameras;
    }

    if (ImGui::IsKeyPressed('H')) {
        m_target_deform_frame = tcnn::clamp<int>(m_target_deform_frame + 5, 0, m_meshes.size() - 1);
        reset_accumulation();
    }

    if (ImGui::IsKeyPressed('J')) {
        m_target_deform_frame = tcnn::clamp<int>(m_target_deform_frame - 5, 0, m_meshes.size() - 1);
        reset_accumulation();
    }

    return false;
}

__global__ void warp(
        uint32_t n_elements,
        float *__restrict__ samples,
        ngp::TriangleBvhNode **__restrict__ bvhnodes,
        ngp::Triangle **__restrict__ tris_bvh,
        ngp::Triangle **__restrict__ tris_deform,
        ngp::Triangle *__restrict__ tris_canon,
        uint32_t offset,
        int32_t frame_id,
        uint32_t n_meshes,
        bool use_geo,
        uint32_t geo_params,
        ngp::BoundingBox render_bbox,
        const ngp::TrainingImageMetadata *__restrict__ metadata,
        const ngp::TrainingXForm *training_xforms,
        default_rng_t rng,
        float *exp_cond,
        uint32_t extra_dims,
        bool use_eye_cond,
        int *adjacency,
        bool disable_interp,
        bool inference
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    uint32_t j = i * offset;
    if (j >= n_elements * offset) return;

    if (samples[j + 4] < 0.f) return;
    if (frame_id == MAX_UINT) frame_id = uint32_t(samples[j + 4]);
    if (frame_id >= n_meshes) return;

    Vector3f point = {samples[j + 0], samples[j + 1], samples[j + 2]};
    Vector3f direction = {samples[j + 8], samples[j + 9], samples[j + 10]};

    if (!render_bbox.contains(point)) return;

    auto nn = closest_triangle(point, bvhnodes[frame_id], tris_bvh[frame_id]);
    auto id = nn.first;

    if (id == -1) {
        printf("[BUG] (warp) No closest triangle found! Frame = %d sample = [%f, %f, %f]\n", frame_id, point.x(), point.y(), point.z());
        return;
    }

    const ngp::Triangle &deformed = tris_bvh[frame_id][id];
    const ngp::Triangle &canonical = tris_canon[deformed.id];

    int mask_id = deformed.mask_id;

    Matrix4f world_to_tri = interpolate_transformation(point, adjacency, tris_deform[frame_id], deformed, true);
    Matrix4f tri_to_canon = interpolate_transformation(point, adjacency, tris_canon, canonical, false);

    if (mask_id != 2 || disable_interp) {
        world_to_tri = tri2projection(point, deformed).inverse();
        tri_to_canon = tri2projection(point, canonical);
    }

    // Transform from deformed position to canonical
    Vector3f canon_point = transform_point(point, world_to_tri, tri_to_canon);

    // Pos
    samples[j + 0] = canon_point.x();
    samples[j + 1] = canon_point.y();
    samples[j + 2] = canon_point.z();

    Vector3f canon_dir = warp_direction(transform_direction(unwarp_direction(direction), world_to_tri, tri_to_canon));

    // Dir
    samples[j + 8] = canon_dir.x();
    samples[j + 9] = canon_dir.y();
    samples[j + 10] = canon_dir.z();

    if (mask_id != 7) {
        for (int m = 0; m < extra_dims; ++m) {
            samples[j + offset - extra_dims + m] = 1.f;
        }
    }

    // Geo
    if (use_geo) {
        float cosine = (point - deformed.centroid()).normalized().dot(deformed.normal());
        samples[j + offset - 1] = cosine;
    }
}

__global__ void warp_random_samples(
        uint32_t n_elements,
        float *__restrict__ samples,
        ngp::TriangleBvhNode **__restrict__ bvhnodes,
        ngp::Triangle **__restrict__ tris_bvh,
        ngp::Triangle **__restrict__ tris_deform,
        ngp::Triangle *__restrict__ tris_canon,
        default_rng_t rng,
        uint32_t n_meshes,
        int *adjacency,
        bool disable_interp
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    uint32_t j = i * 3;
    if (j >= n_elements * 3) return;
    n_meshes -= 1;
    Vector3f point = {samples[j + 0], samples[j + 1], samples[j + 2]};

    rng.advance(i * n_meshes);
    float r = ngp::random_val(rng);
    uint32_t frame_id = uint32_t(r * float(n_meshes));

    auto nn = closest_triangle(point, bvhnodes[frame_id], tris_bvh[frame_id]);
    auto id = nn.first;

    if (id == -1) {
        printf("[BUG] (warp_random_samples) No closest triangle found! Frame = %d sample = [%f, %f, %f]\n", frame_id, point.x(), point.y(), point.z());
        return;
    }

    const ngp::Triangle &deformed = tris_bvh[frame_id][id];
    const ngp::Triangle &canonical = tris_canon[deformed.id];

    Matrix4f world_to_tri = interpolate_transformation(point, adjacency, tris_deform[frame_id], deformed, true);
    Matrix4f tri_to_canon = interpolate_transformation(point, adjacency, tris_canon, canonical, false);

    int mask_id = deformed.mask_id;

    if (mask_id != 2 || disable_interp) {
        world_to_tri = tri2projection(point, deformed).inverse();
        tri_to_canon = tri2projection(point, canonical);
    }

    // Transform from deformed position to canonical
    Vector3f canon = transform_point(point, world_to_tri, tri_to_canon);

    // Pos
    samples[j + 0] = canon.x();
    samples[j + 1] = canon.y();
    samples[j + 2] = canon.z();
}

tcnn::GPUMatrix<float> rta::Core::surface_closest_point_gpu(uint32_t n_elements, float *coords, cudaStream_t stream) {
    reset_accumulation();
    const uint32_t floats_per_coord = sizeof(ngp::NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
    linear_kernel(warp, 0, stream, n_elements / floats_per_coord,
                  coords,
                  m_bvh_ptrs_gpu.data(),
                  m_triangles_ptrs_gpu.data(),
                  m_triangles_ptrs_gpu_topology.data(),
                  m_canonical_shape->triangles_gpu_orig_order.data(),
                  floats_per_coord,
                  MAX_UINT,
                  m_meshes.size(),
                  m_use_geo,
                  N_GEO_PARAMS,
                  m_aabb,
                  m_nerf.training.dataset.metadata_gpu.data(),
                  m_nerf.training.transforms_gpu.data(),
                  m_random_generator,
                  exp_pca_gpu.data(),
                  m_nerf_network->n_extra_dims(),
                  m_use_eyes,
                  m_adjacency_gpu.data(),
                  m_disable_interp,
                  m_recorder->is_recording()
    );

    return tcnn::GPUMatrix<float>(coords, floats_per_coord, n_elements / floats_per_coord);
}

tcnn::GPUMatrix<float> rta::Core::render_deformed_gpu(uint32_t n_elements, float *coords, cudaStream_t stream) {
    reset_accumulation();
    const uint32_t floats_per_coord = sizeof(ngp::NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
    linear_kernel(warp, 0, stream, n_elements / floats_per_coord,
                  coords,
                  m_bvh_ptrs_gpu.data(),
                  m_triangles_ptrs_gpu.data(),
                  m_triangles_ptrs_gpu_topology.data(),
                  m_canonical_shape->triangles_gpu_orig_order.data(),
                  floats_per_coord,
                  (uint32_t) m_target_deform_frame,
                  m_meshes.size(),
                  m_use_geo,
                  N_GEO_PARAMS,
                  m_aabb,
                  m_nerf.training.dataset.metadata_gpu.data(),
                  m_nerf.training.transforms_gpu.data(),
                  m_random_generator,
                  exp_pca_gpu.data(),
                  m_nerf_network->n_extra_dims(),
                  m_use_eyes,
                  m_adjacency_gpu.data(),
                  m_disable_interp,
                  m_recorder->is_recording()
    );

    return tcnn::GPUMatrix<float>(coords, floats_per_coord, n_elements / floats_per_coord);
}

tcnn::GPUMatrix<float> rta::Core::surface_closest_point_3d_gpu(uint32_t n_elements, float *coords, cudaStream_t stream) {
    const uint32_t floats_per_coord = sizeof(ngp::NerfPosition) / sizeof(float);

    linear_kernel(warp_random_samples, 0, stream, n_elements / floats_per_coord,
                  coords,
                  m_bvh_ptrs_gpu.data(),
                  m_triangles_ptrs_gpu.data(),
                  m_triangles_ptrs_gpu_topology.data(),
                  m_canonical_shape->triangles_gpu_orig_order.data(),
                  m_random_generator,
                  m_meshes.size(),
                  m_adjacency_gpu.data(),
                  m_disable_interp
    );

    return tcnn::GPUMatrix<float>(coords, floats_per_coord, n_elements / floats_per_coord);
}

void rta::Core::post_train_data() {
    m_nerf.training.n_images_for_training = m_nerf.training.dataset.n_training_images;
}

std::mt19937 synth_gen(42);

void rta::Core::reload_training_data(bool force, std::string mode) {

    if (m_nerf.training.optimize_extra_dims && force) {
        m_latent_codes.clear();
        m_latent_codes.resize(m_nerf.training.extra_dims_gpu.size());
        std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
        m_nerf.training.extra_dims_gpu.copy_to_host(m_latent_codes);
    }

    auto src = m_data_path;
    if (m_data_path.is_file())
        src = m_data_path.parent_path();

    if ((!m_nerf.training.optimize_extra_dims && ((m_training_step > 0 && fast_mod(m_training_step, 1500) == 0))) || force) {
        std::vector<fs::path> json_paths;
        if (!m_json_paths.empty()) {
            m_data_path = m_json_paths.front().parent_path();
            json_paths = m_json_paths;
        } else {
            for (const auto &path: fs::directory{src}) {
                if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
                    if (path.str().find(mode) != std::string::npos) {
                        json_paths.emplace_back(path);
                    }

                    if (m_nerf.training.dataset.is_synthetic) { // initialized in the first loading
                        if (path.str().find("synthetic") != std::string::npos) {
                            json_paths.emplace_back(path);
                        }
                    }
                }
            }
        }

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        clean_dataset();

        m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen, m_dataset_paths);

        load_nerf_post();
        load_meshes(m_data_path.str(), m_nerf.training.optimize_extra_dims && force);
        post_train_data();
    }
}

void rta::Core::clean_dataset() {
    tlog::info() << "=========== Cleaning the dataset ===========";

//    if (m_dataset_paths.is_training && m_render_ngp)
//        update_xforms();

    m_bvh_ptrs_gpu.free_memory();
    m_triangles_ptrs_gpu.free_memory();
    m_triangles_ptrs_gpu_topology.free_memory();
    m_meshes.clear();

    // ===== Dataset =====
    m_nerf.training.dataset.metadata.clear();
    m_nerf.training.dataset.metadata_gpu.free_memory();
    m_nerf.training.dataset.sharpness_data.free_memory();
    m_nerf.training.dataset.envmap_data.free_memory();

    for (int i = 0; i < m_nerf.training.dataset.n_images; ++i) {
        m_nerf.training.dataset.segmaskmemory[i].free_memory();
        m_nerf.training.dataset.pixelmemory[i].free_memory();
        m_nerf.training.dataset.raymemory[i].free_memory();
        m_nerf.training.dataset.depthmemory[i].free_memory();
    }

    m_nerf.training.dataset = {};

    // ===== Training =====
    m_nerf.training.transforms_gpu.free_memory();
    m_nerf.training.transforms_gpu = {};

    m_nerf.training.transforms.clear();

    m_nerf.training.extra_dims_gpu.free_memory();
    m_nerf.training.extra_dims_gpu = {};
    m_nerf.training.extra_dims_gradient_gpu.free_memory();
    m_nerf.training.extra_dims_gradient_gpu = {};

    m_bvh_ptrs_gpu = {};
    m_triangles_ptrs_gpu = {};
    m_triangles_ptrs_gpu_topology = {};
    n_extra_dims = 0;
    mesh_total_count = 0;
}

rta::Core::~Core() {
    PRINT("Core::~Core");
}

void rta::Core::update_paths_dataset() {

}

void rta::Core::test_raycasting() {
    auto index = 100;
    auto meta = m_nerf.training.dataset.metadata[index];

    Vector2i resolution = {512, 512};
    auto camera = m_nerf.training.transforms[index].start;
    Vector2f focal_length = calc_focal_length(resolution, m_fov_axis, m_zoom);
    Vector2f screen_center = render_screen_center();

    reset_camera();

    resolution = meta.resolution;
    screen_center = meta.principal_point;
    focal_length = meta.focal_length;

    std::vector<Vector3f> hits;
    std::vector<Vector3i> colors;

    for (int u = 0; u < resolution.x(); ++u) {
        for (int v = 0; v < resolution.y(); ++v) {
            const Eigen::Vector2i pixel = {u, v};
            auto ray = ngp::pixel_to_ray_pinhole(1, pixel, resolution, focal_length, camera, screen_center);
            auto origin = ray.o;
            auto dir = ray.d;

            // Comment copy of triangles_cpu in tiny_mesh.cu to enable it
            auto hit = ray_intersect(origin, dir, m_meshes[index]->triangle_bvh->nodes(), m_meshes[index]->triangles_cpu.data());

            if (hit.first != -1) {
                auto point = origin + dir * hit.second;
                auto tri = m_meshes[index]->triangles_cpu[hit.first];

                auto mask_id = tri.mask_id;

                if (mask_id == 0) colors.emplace_back(0, 255, 0);
                else if (mask_id == 3) colors.emplace_back(0, 0, 255);
                else colors.emplace_back(100, 100, 100);
//                colors.emplace_back(mask_id * 10 + 50, 0, 0);

                hits.emplace_back(point);
            }
        }
    }

    std::ofstream out;
    out.open("test.obj");
    for (int i = 0; i < hits.size(); ++i) {
        auto vert = hits[i];
        auto color = colors[i];
        out << "v " << vert.x() << " " << vert.y() << " " << vert.z() << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
    }
    out.close();
    exit(1);
}

void rta::Core::update_xforms() {
    std::vector<fs::path> json_paths;
    for (const auto &path: fs::directory{m_data_path}) {
        if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
            json_paths.emplace_back(path);
        }
    }

    std::string path = json_paths.front().str();
    std::cout << path << std::endl;
    std::ifstream f{path};
    nlohmann::json transforms = nlohmann::json::parse(f, nullptr, true, true);
    auto &frames = transforms["frames"];

    std::sort(frames.begin(), frames.end(), [](const auto &frame1, const auto &frame2) {
        return frame1["mesh_path"] < frame2["mesh_path"];
    });

    auto mapping = m_nerf.training.dataset.current_samples[path];

    for (int i = 0; i < mapping.size(); ++i) {
        int frame_id = mapping[i];
        const Eigen::Matrix<float, 3, 4> xform = m_nerf.training.dataset.xforms[i].start;

        Matrix4f C = Matrix4f::Identity();
        C.block<3, 4>(0, 0) = xform;
        C.block<3, 1>(0, 3) -= Vector3f::Constant(0.5f);
        frames[frame_id]["transform_matrix"] = {C.row(0), C.row(1), C.row(2), C.row(3)};
    }

    std::ofstream file(path);
    file << std::setw(4) << transforms << std::endl;
    file.close();
}

__global__ void point_to_density_flame(
        uint32_t n_elements,
        float *__restrict__ points,
        network_precision_t *__restrict__ density,
        ngp::TriangleBvhNode *__restrict__ bvh_canon,
        ngp::Triangle *__restrict__ tris_canon,
        float radius
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    uint32_t j = i * 3;
    Vector3f point = {points[j + 0], points[j + 1], points[j + 2]};

    auto nn = closest_triangle(point, bvh_canon, tris_canon);
    int id = nn.first;
    float dist = nn.second;
    auto &tri = tris_canon[id];

    // Not inside the mesh
    bool ff = (point - tri.centroid()).dot(tri.normal()) > 0;
    if (!ff) radius = 0.015;

    density[i] = 0;
    if (id != -1 && dist <= radius) // 3 cm radius
        density[i] = fmax(density[i], 1.f);
}

GPUMatrix<network_precision_t, tcnn::RM> rta::Core::point_density_flame_closest_point(const GPUMatrix<float> &points, const GPUMatrix<network_precision_t, RM> &density, float radius, cudaStream_t stream) {
    auto n_elements = density.n();

    linear_kernel(point_to_density_flame, 0, stream, n_elements,
                  points.data(),
                  density.data(),
                  m_canonical_shape->triangle_bvh->nodes_gpu(),
                  m_canonical_shape->triangles_gpu.data(),
                  radius
    );

    return GPUMatrix<network_precision_t, RM>(density.data(), density.m(), density.n());
}
