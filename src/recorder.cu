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

#include <rta/recorder.h>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <stb_image/stb_image_write.h>

#include <rta/debug.h>
#include <rta/core.h>
#include <rta/helpers.h>

static constexpr const char *VideoModeStr = "Floating\0Vertical\0Horizontal\0Overlay\0Normals\0Frontal\0Sweep\0";

using namespace Eigen;
namespace fs = filesystem;

static void save_rgba(const std::vector<Eigen::Array4f> &rgba_cpu, const char *path, const char *name, Eigen::Vector2i res3d, std::function<float(float)> transform) {
    uint32_t w = res3d.x();
    uint32_t h = res3d.y();

    std::vector<uint8_t> pngpixels(size_t(w) * size_t(h) * 4, 0);
    int dst = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = x + res3d.x() + y * res3d.x();
            if (i < rgba_cpu.size()) {
                Eigen::Array4f rgba = rgba_cpu[i];
                pngpixels[dst++] = (uint8_t) tcnn::clamp(transform(rgba.x()) * 255.f, 0.f, 255.f);
                pngpixels[dst++] = (uint8_t) tcnn::clamp(transform(rgba.y()) * 255.f, 0.f, 255.f);
                pngpixels[dst++] = (uint8_t) tcnn::clamp(transform(rgba.z()) * 255.f, 0.f, 255.f);
                pngpixels[dst++] = (uint8_t) tcnn::clamp(transform(rgba.w()) * 255.f, 0.f, 255.f);
            }
        }
    }
    // write slice
    filesystem::path output(path);
    output = output / (std::string(name) + ".png");
    stbi_write_png(output.str().c_str(), w, h, 4, pngpixels.data(), w * 4);
}

rta::Recorder::Recorder(ngp::Testbed *ngp) : m_ngp(ngp) {
    m_training_steps_wait = ngp->m_network_config["max_steps"];
    m_output_path = m_ngp->m_data_path;
    if (m_ngp->m_data_path.is_file())
        m_output_path = m_ngp->m_data_path.parent_path();

    std::string config = m_ngp->m_network_config_path.basename();
    for (std::string str: {"experiments", config.c_str(), "debug"}) {
        m_output_path = m_output_path / str;
        fs::create_directory(m_output_path);
    }
}

void rta::Recorder::save_depth(float *depth_gpu, const char *path, const char *name, Vector2i res3d) {
    uint32_t w = res3d.x();
    uint32_t h = res3d.y();
    uint32_t size = w * h;
    std::vector<float> depth_cpu;
    depth_cpu.resize(size);
    CUDA_CHECK_THROW(cudaMemcpy(depth_cpu.data(), depth_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));

    auto dir = m_current_output / "depth";

    if (m_dst_folder.find("synthetic") == std::string::npos && m_video_mode == VideoType::Overlay) {
        dir = m_current_output / ".." / "depth";
    }

    if (!(dir).exists()) { fs::create_directory(dir); }
    auto file = std::string(name);
    std::ofstream wf(dir.str() + "/" + file + ".bin", std::ios::out | std::ios::binary);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = x + res3d.x() + y * res3d.x();
            if (i < depth_cpu.size()) {
                float d = depth_cpu[i];
                if (d < 0.f || d > 5.f) d = 0;
                uint32_t depth = uint32_t(d * 1000.0f);
                wf.write((char *) &depth, sizeof(uint32_t));
            }
        }
    }
    wf.close();
}

float linear_to_srgb(float linear) {
    if (linear < 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
    }
}

float srgb_to_linear(float srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    } else {
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

void rta::Recorder::imgui() {
    ImGui::Separator();
    ImGui::Text("Record video");
//    if (ImGui::Button("Snapshot")) snapshot();
    if (m_record_all && !m_is_recording && !m_single_step) m_video_mode = VideoType::Floating;
    if (ImGui::Button("Start")) start();
    ImGui::SameLine();
    if (ImGui::Button("Stop")) stop();
    ImGui::SameLine();
    if (ImGui::Button("Reset Camera")) { m_ngp->reset_camera(); };

    if (m_is_recording) {
        ImGui::SameLine();
        auto str = "#" + std::to_string(m_index_frame + 1);
        ImGui::Text(str.c_str());
    }

    ImGui::Combo("Mode ", (int *) &m_video_mode, VideoModeStr);
    ImGui::SameLine();
    ImGui::Checkbox("All", &m_record_all);
    ImGui::Checkbox("Save depth", &m_save_depth);
    ImGui::Checkbox("Resume training", &m_resume_training);

    if (!m_is_recording) {
        ImGui::PushItemWidth(100);
        ImGui::InputInt("# fps", &m_fps, 5);
    }

    if (m_is_recording && m_index_frame >= 0) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start).count();
        float value = float(m_index_frame + 1) / elapsed;
        if (!std::isinf(value))
            m_current_fps = value;
        auto str = "Time " + to_string_with_precision(elapsed) + " [s] " + to_string_with_precision(m_current_fps) + " [fps] " + std::to_string(m_to_record);
        ImGui::Text(str.c_str());
        auto progress = float(m_index_frame) / float(m_to_record);
        ImGui::ProgressBar(progress, ImVec2(-1, 0));
    }
}

void rta::Recorder::start() {
    m_json_cameras = JsonCameras();
    auto *core = (rta::Core *) m_ngp;
    auto root = m_output_path / "snapshot.msgpack";
    if (m_video_mode == VideoType::Overlay)
        m_ngp->save_snapshot(root.str(), false);
    core->m_train = false;
    m_index_frame = 0;
    core->m_dataset_settings.is_training = false;
    core->m_dataset_settings.shuffle = false;
    std::string mode = "test";
    core->m_background_color.w() = 0.f;
    core->reload_training_data(true, mode);
    core->m_offscreen_rendering = false;
    core->m_dynamic_res = false;
    m_is_recording = !m_is_recording;
    m_start = std::chrono::steady_clock::now();
    m_to_record = core->m_nerf.training.dataset.n_all_images;
    if (core->m_dataset_settings.is_training)
        m_to_record = core->m_nerf.training.dataset.n_training_images;
    m_ngp->reset_accumulation();
    m_ngp->reset_camera();

    auto &render_buffer = m_ngp->m_render_surfaces.front();
    render_buffer.clear_frame(core->m_stream.get());

    generate_floating_cameras();
    generate_horizontal_cameras();
    generate_vertical_cameras();
    generate_hemisphere_cameras();
    generate_sweep_cameras();
    tlog::success() << "Recording has started!";
}

void rta::Recorder::stop() {
    auto *core = (rta::Core *) m_ngp;
    core->m_dynamic_res = true;
    m_is_recording = false;
    m_single_step = false;
    m_render_train_depth = false;
    core->m_offscreen_rendering = false;
    core->m_raycast_flame_mesh = false;
    core->m_nerf.training.view = 0;
    core->reset_camera();
    core->reset_accumulation();

    core->m_render_mode = ngp::ERenderMode::Shade;
    core->m_background_color.w() = 1.f;

    std::string cmd = "cp " + m_ngp->m_network_config_path.str() + " " + (m_output_path / "config.json").str();
    system(cmd.c_str());
    dump_cameras_json();

    if (m_resume_training) {
        core->set_train(true);
        core->m_dataset_settings.is_training = true;
        core->m_dataset_settings.shuffle = true;
        core->m_dataset_settings.load_all_training = false;
        core->m_dataset_settings.is_rendering_depth = false;
        core->m_dataset_settings.load_to_gpu = true;
        core->reload_training_data(true);
    }

    m_index_frame = 0;
    m_average_time = 0;
}

void rta::Recorder::video() {
    if (!m_is_recording) return;
    size_t index = m_index_frame;
    m_ngp->m_target_deform_frame = index;
    m_ngp->m_nerf.extra_dim_idx_for_inference = index;
    auto *core = (rta::Core *) m_ngp;

    switch (m_video_mode) {
        case VideoType::Floating:
            m_dst_folder = "floating";
            set_camera_to_novel_view(index);
            break;
        case VideoType::Horizontal:
            m_dst_folder = "horizontal";
            set_camera_to_novel_view(index);
            break;
        case VideoType::Vertical:
            m_dst_folder = "vertical";
            set_camera_to_novel_view(index);
            break;
        case VideoType::Overlay:
            m_dst_folder = "overlay";
            set_camera_to_training_view(index);
            break;
        case VideoType::Sweep:
            m_dst_folder = "sweep";
            set_camera_to_novel_view(index);
            m_ngp->set_fov(18);
            break;
        case VideoType::Normals:
            m_dst_folder = "normals";
            core->m_render_mode = ngp::ERenderMode::Normals;
            if (m_horizontal_normals)
                set_camera_to_novel_view(index);
            else
                set_camera_to_training_view(index);
            break;
        case VideoType::Frontal:
            m_dst_folder = "frontal";
            render_frontal();
            break;
    }
}

void rta::Recorder::render_frontal() {
    create_folder();
    m_ngp->reset_camera();
    m_ngp->set_fov(18);
}

void rta::Recorder::set_camera_to_novel_view(size_t index) {
    create_folder();
    set_floating_camera(index);
}

void rta::Recorder::set_camera_to_training_view(size_t index) {
    create_folder();
    m_ngp->m_nerf.training.view = index;
    auto current_camera = m_ngp->m_camera;
    m_ngp->set_camera_to_training_view(index);
}

void rta::Recorder::dump_frame_buffer(std::string suffix) {
    auto *core = (rta::Core *) m_ngp;
    auto &render_buffer = m_ngp->m_render_surfaces.front();
    Vector2i res = render_buffer.in_resolution();

    if (!core->m_render_window) { // --no-gui option
        render_buffer.resize(res);
        core->render_frame(core->m_smoothed_camera, core->m_smoothed_camera, Eigen::Vector4f::Zero(), render_buffer);
    }

    auto dir = m_output_path / m_dst_folder;
    ngp::BoundingBox aabb = (m_ngp->m_testbed_mode == ngp::ETestbedMode::Nerf) ? m_ngp->m_render_aabb : m_ngp->m_aabb;
    auto rgba = render_buffer.accumulate_buffer();
    auto size = res.x() * res.y();
    auto version = std::string(m_synthetic_version);
    std::string str = std::to_string(m_index_frame);
    Vector2f pp = m_ngp->m_nerf.training.dataset.metadata[0].principal_point;
    if (m_dst_folder.find("synthetic") != std::string::npos) {
        dir = m_current_output / "raw";
//        pp = core->m_screen_center;
    }

    auto id = std::string(5 - std::min(5, int(str.length())), '0') + str + suffix;

    m_json_cameras.w = res.x();
    m_json_cameras.h = res.y();
    m_json_cameras.cx = res.x() * pp.x();
    m_json_cameras.cy = res.y() * pp.y();
    Eigen::Vector2f scale = res.cast<float>();
    scale.x() *= core->m_relative_focal_length.x();
    scale.y() *= core->m_relative_focal_length.y();
    m_json_cameras.fl = scale;

    JsonFrame frame;
    Matrix4f C = Matrix4f::Identity();
    C.block<3, 4>(0, 0) = core->m_camera;
    C.block<3, 1>(0, 3) -= Vector3f::Constant(0.5f);
//    frame.transform_matrix = C.inverse();
    frame.transform_matrix = C;

    frame.exp_path = "flame/exp/" + id + ".txt";
    frame.eyes_path = "flame/eyes/" + id + ".txt";
    frame.mesh_path = "meshes/" + id + ".obj";
    m_json_cameras.frames.push_back(frame);

    std::vector<Array4f> rgba_pred_cpu;
    rgba_pred_cpu.resize(size);
    CUDA_CHECK_THROW(cudaMemcpy(rgba_pred_cpu.data(), rgba, size * sizeof(Array4f), cudaMemcpyDeviceToHost));
    std::function<float(float)> func = [](float c) { return c; };
    if (core->m_render_mode == ngp::ERenderMode::Normals)
        func = srgb_to_linear;
    save_rgba(rgba_pred_cpu, dir.str().c_str(), id.c_str(), res, func);

    if (m_save_depth) {
        save_depth(render_buffer.depth_buffer(), dir.str().c_str(), id.c_str(), render_buffer.in_resolution());
    }

    m_ngp->reset_accumulation();
    m_index_frame++;
    m_average_time += m_current_fps;
}

void rta::Recorder::create_folder() {
    auto root = m_output_path;
    if (!root.exists()) {
        fs::create_directory(root);
    }
    auto version = std::string(m_synthetic_version);
    auto dir = root / m_dst_folder;
    fs::create_directory(dir);
    m_current_output = dir;
}

void rta::Recorder::set_floating_camera(size_t index) {
    auto rt = m_floating_cameras[index];
    if (m_dst_folder == "horizontal") rt = m_horizontal_cameras[index];
    if (m_dst_folder == "normals") rt = m_horizontal_cameras[index];
    if (m_dst_folder == "synthetic_horizontal") rt = m_horizontal_cameras[index];
    if (m_dst_folder == "synthetic_vertical") rt = m_vertical_cameras[index];
    if (m_dst_folder == "synthetic_floating") rt = m_floating_cameras[index];
    if (m_dst_folder == "synthetic_hemisphere" || m_dst_folder == "synthetic_depth") rt = m_hemisphere_cameras[index];
    if (m_dst_folder == "vertical") rt = m_vertical_cameras[index];
    if (m_dst_folder == "sweep") rt = m_sweep_cameras[index];

//    if (m_dst_folder == "synthetic_train_depth") rt = m_average_camera;

    m_ngp->first_training_view();
    m_ngp->m_camera = rt;
    m_ngp->m_smoothed_camera = rt;
}

void rta::Recorder::step() {
    if ((m_ngp->m_training_step == m_training_steps_wait || m_render_from_snapshot) && !m_is_recording) {
        start();
    }

    if (m_single_step) {
        stop();
    };

    if (m_index_frame >= m_to_record && m_is_recording) {
        stop();
        if (m_record_all) {
            next();
        }
    }

    if (m_index_frame < m_to_record) {
        video();
    }
}

float clip(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
}

Eigen::Matrix3f lootAt(const Vector3f &from, const Vector3f &to) {
    auto dir = (to - from).normalized();
    Eigen::Matrix3f camera = Eigen::Matrix3f::Identity();
    Eigen::Vector3f up = {0.0f, 1.0f, 0.0f};
    camera.col(0) = dir.cross(up).normalized();
    camera.col(1) = dir.cross(camera.col(0)).normalized();
    camera.col(2) = dir.normalized();
    return camera;
}

Eigen::Matrix<float, 3, 4> to(Eigen::Matrix<float, 3, 4> camera, const Matrix3f &target) {
    Matrix3f R = camera.block<3, 3>(0, 0);
    Vector3f from = camera.col(3);
    Vector3f to = target * from;

    Matrix3f gt_lookat = lootAt(from, {0.5, 0.5, 0.5});
    Matrix3f novel_lookat = lootAt(to, {0.5, 0.5, 0.5});

    Matrix3f Rt = R * gt_lookat.inverse();
    Rt = Rt * novel_lookat;

    Eigen::Matrix<float, 3, 4> C = Eigen::Matrix<float, 3, 4>::Zero();

    C.block<3, 3>(0, 0) = Rt;
    C.block<3, 1>(0, 3) = to;

    return C;
}

void rta::Recorder::generate_floating_cameras() {
    m_floating_cameras.clear();
    set_neutral_camera();
    Eigen::Matrix<float, 3, 4> start = m_neutral_camera;
    Vector3f center = {0.5f, 0.5f, 0.5f};
    int views = m_to_record;
    for (int i = 0; i < views; ++i) {
        float t = float(i) / float(views);
        float pitch = 0.15f * std::cos(t * 2.f * M_PI);
        float yaw = 0.35f * std::sin(t * 2.f * M_PI);

        AngleAxisf x(pitch, Vector3f::UnitX());
        AngleAxisf y(yaw, Vector3f::UnitY());
        Matrix3f rot = (y * x).toRotationMatrix();

        Eigen::Matrix<float, 3, 4> tmp = start;
        tmp.col(3) = tmp.col(3) - center;
        Eigen::Matrix<float, 3, 4> camera = rot * tmp;
        camera.col(3) = camera.col(3) + center;

        camera.block<3, 3>(0, 0) = lootAt(camera.col(3), {0.5, 0.5, 0.5});
        m_floating_cameras.emplace_back(camera);
    }
}

void rta::Recorder::generate_horizontal_cameras() {
    m_horizontal_cameras.clear();
    set_neutral_camera();
    int views = m_to_record;
    float angle = m_horizontal_angle;
    float anchor = angle * M_PI / 180.f;
    float t = 2.f * angle / float(views);
    float k = 0.f;
    Eigen::Matrix<float, 3, 4> start = m_neutral_camera;
    Vector3f center = {0.5f, 0.5f, 0.5f};
    for (int i = 0; i < views; ++i) {
        float yaw = anchor - k * M_PI / 180.f;
        k += t;
        AngleAxisf y(yaw, Vector3f::UnitY());
        Matrix3f rot = y.toRotationMatrix();

        Eigen::Matrix<float, 3, 4> tmp = start;
        tmp.col(3) = tmp.col(3) - center;
        Eigen::Matrix<float, 3, 4> camera = rot * tmp;
        camera.col(3) = camera.col(3) + center;

        camera.block<3, 3>(0, 0) = lootAt(camera.col(3), {0.5, 0.5, 0.5});
        m_horizontal_cameras.emplace_back(camera);
    }
}

void rta::Recorder::generate_sweep_cameras() {
    m_sweep_cameras.clear();
    m_ngp->reset_camera();
    m_ngp->set_fov(18);
    auto camera = m_ngp->m_camera;
    int views = m_to_record + 1;
    float angle = 5;
    float anchor = 0;
    float t = 2.f * angle / float(views);
    float k = 0.f;
    float yaw = 0;
    Eigen::Matrix<float, 3, 4> start = m_neutral_camera;
    Vector3f center = {0.5f, 0.5f, 0.5f};
    float direction = 1;
    for (int i = 1; i < views; ++i) {
//        if (i == int(views / 3)) {
//            anchor = yaw;
//            k = 0;
//            direction = -1;
//        }
//        yaw = anchor + direction * k * M_PI / 180.f;
//        k += t;
//        AngleAxisf y(yaw, Vector3f::UnitY());
//        Matrix3f rot = y.toRotationMatrix();
        float t = float(i) / float(views);
        float pitch = 0.05f * std::cos(t * 2.f * M_PI) - 0.1;
        float yaw = 0.2f * std::sin(t * 2.f * M_PI);

        AngleAxisf x(pitch, Vector3f::UnitX());
        AngleAxisf y(yaw, Vector3f::UnitY());
        Matrix3f rot = (y * x).toRotationMatrix();

        Eigen::Matrix<float, 3, 4> tmp = start;
        tmp.col(3) = tmp.col(3) - center;
        Eigen::Matrix<float, 3, 4> camera = rot * tmp;
        camera.col(3) = camera.col(3) + center;

        camera.block<3, 3>(0, 0) = lootAt(camera.col(3), {0.5, 0.5, 0.5});
        m_sweep_cameras.emplace_back(camera);
    }
}

void rta::Recorder::generate_vertical_cameras() {
    m_vertical_cameras.clear();
    set_neutral_camera();
    Eigen::Matrix<float, 3, 4> start = m_neutral_camera;
    Vector3f center = {0.5f, 0.5f, 0.5f};
    int views = m_to_record;
    float angle = 10.f;
    float anchor = angle * M_PI / 180.f;
    float t = 2.f * angle / float(views);
    float k = 0.f;
    for (int i = 0; i < views; ++i) {
        float roll = anchor - k * M_PI / 180.f;
        k += t;
        AngleAxisf y(-roll, Vector3f::UnitX());
        Matrix3f rot = y.toRotationMatrix();

        Eigen::Matrix<float, 3, 4> tmp = start;
        tmp.col(3) = tmp.col(3) - center;
        Eigen::Matrix<float, 3, 4> camera = rot * tmp;
        camera.col(3) = camera.col(3) + center;

        camera.block<3, 3>(0, 0) = lootAt(camera.col(3), {0.5, 0.5, 0.5});
        m_vertical_cameras.emplace_back(camera);
    }
}

void rta::Recorder::generate_hemisphere_cameras() {
    m_hemisphere_cameras.clear();
    set_neutral_camera();
    int views = m_to_record;
    int views_x = 50;
    int views_y = views / views_x;
    float angle_x = 20.f;
    float angle_y = 30.f;
    float anchor_x = angle_x * M_PI / 180.f;
    float anchor_y = angle_y * M_PI / 180.f;
    float t_y = 2.f * angle_y / float(views_y);
    float k_y = 0.f;
    Eigen::Matrix<float, 3, 4> start = m_neutral_camera;
    Vector3f center = {0.5f, 0.5f, 0.5f};
    for (int i = 0; i < views_y; ++i) {
        float yaw = anchor_y - k_y * M_PI / 180.f;
        k_y += t_y;
        AngleAxisf y(yaw, Vector3f::UnitY());
        float t_x = 2.f * angle_x / float(views_x);
        float k_x = 0.f;
        for (int j = 0; j < views_x; ++j) {
            float roll = anchor_x - k_x * M_PI / 180.f;
            k_x += t_x;
            AngleAxisf x(-roll, Vector3f::UnitX());
            Matrix3f rot = (x * y).toRotationMatrix();

            Eigen::Matrix<float, 3, 4> camera = Eigen::Matrix<float, 3, 4>::Zero();

            // The origin for rotations has to be 0,0,0
            Vector3f tmp = start.col(3) - center;
            camera.col(3) = rot * tmp;
            camera.col(3) += center;

            camera.block<3, 3>(0, 0) = lootAt(camera.col(3), {0.5, 0.5, 0.5});
            m_hemisphere_cameras.emplace_back(camera);
        }
    }
}


void rta::Recorder::next() {
    if ((m_video_mode == VideoType::Normals) && m_record_all) {
        exit(0);
    }
    m_video_mode = static_cast<VideoType>(static_cast<int>(m_video_mode) + 1);
    start();
}

void rta::Recorder::set_neutral_camera() {
    auto size = m_ngp->m_nerf.training.transforms.size();
    m_neutral_camera = Eigen::Matrix<float, 3, 4>::Zero();
    Vector3f t = Vector3f::Zero();
    for (auto transform: m_ngp->m_nerf.training.transforms) {
        m_neutral_camera += transform.start;
        t += transform.start.col(3);
    }
    m_neutral_camera /= float(size);
    m_average_camera = m_neutral_camera;
    t /= float(size);
    auto z = m_neutral_camera.col(3).z();
    m_ngp->reset_camera();
    m_neutral_camera = m_ngp->m_camera;
    m_neutral_camera.col(3).z() = t.z();
//    std::cout << "Average  " << t.z() - 0.5f << std::endl;
//    m_average_camera = m_ngp->m_nerf.training.transforms[size - 5].start;
}

void rta::Recorder::snapshot() {
    m_dst_folder = "";
    dump_frame_buffer();
}

void rta::Recorder::progress() {
    if (!m_single_step) {
        m_video_mode = VideoType::Overlay;
        m_single_step = true;
        m_dst_folder = "progress";
        create_folder();
        start();
        m_index_frame = 12;
        video();
        m_dst_folder = "progress";
        m_ngp->m_training_step++;
    }
}

void rta::Recorder::dump_cameras_json() {
    nlohmann::json main;
    main.dump(4);
    main["synthetic"] = true;
    main["h"] = m_json_cameras.h;
    main["w"] = m_json_cameras.w;
    main["fl_x"] = m_json_cameras.fl.x();
    main["fl_y"] = m_json_cameras.fl.y();
    main["cx"] = m_json_cameras.cx;
    main["cy"] = m_json_cameras.cy;
    main["integer_depth_scale"] = 0.001f;
    auto data_array = nlohmann::json::array();

    for (auto &frame: m_json_cameras.frames) {
        nlohmann::json data;
        data.dump(4);
        data["depth_path"] = frame.depth_path;
        data["exp_path"] = frame.exp_path;
        data["eyes_path"] = frame.eyes_path;
        data["file_path"] = frame.file_path;
        data["mesh_path"] = frame.mesh_path;
        data["seg_mask_path"] = frame.seg_mask_path;

        auto camera = nlohmann::json::array();
        for (int m = 0; m < 4; ++m) {
            auto row = nlohmann::json::array();
            for (int n = 0; n < 4; ++n) {
                row.push_back(frame.transform_matrix(m, n));
            }
            camera.push_back(row);
        }
        data["transform_matrix"] = camera;
        data_array.push_back(data);
    }

    std::sort(data_array.begin(), data_array.end(), [](const auto &frame1, const auto &frame2) {
        return frame1["mesh_path"] < frame2["mesh_path"];
    });

    main["frames"] = data_array;

    auto version = std::string(m_synthetic_version);
    std::ofstream file(m_current_output.str() + "/../transforms_" + m_dst_folder + ".json");
    file << std::setw(4) << main << std::endl;
    file.close();
    m_json_cameras = JsonCameras();
}
