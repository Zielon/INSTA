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

#ifndef REAL_TIME_AVATAR_RECORDER_H
#define REAL_TIME_AVATAR_RECORDER_H

#include <cstdint>
#include <neural-graphics-primitives/testbed.h>

#include "common.h"

RTA_NAMESPACE_BEGIN

enum class VideoType : int {
    Floating,
    Vertical,
    Horizontal,
    Overlay,
    Normals,
    Frontal,
    Sweep,
    Synthetic
};

struct JsonFrame {
    std::string depth_path = "";
    std::string exp_path = "";
    std::string eyes_path = "";
    std::string file_path = "";
    std::string mesh_path = "";
    std::string seg_mask_path = "";
    Eigen::Matrix<float, 4, 4> transform_matrix = Eigen::Matrix4f::Identity();
};

struct JsonCameras {
    Eigen::Vector2f fl;
    float cx;
    float cy;
    int w;
    int h;
    std::vector<JsonFrame> frames {};
};

class Recorder {
public:
    explicit Recorder(ngp::Testbed* ngp);
    void imgui();
    void stop();
    void start();
    void video();
    void next();

    bool is_recording(){ return m_is_recording; }
    int current_frame() { return m_index_frame; }
    void generate_floating_cameras();
    void generate_horizontal_cameras();
    void generate_vertical_cameras();
    void generate_hemisphere_cameras();
    void generate_sweep_cameras();
    void snapshot();
    void render_frontal();
    void dump_frame_buffer(std::string suffix = "");
    void dump_cameras_json();
    void create_folder();
    void step();
    void progress();
    std::string info() const;

    void set_camera_to_novel_view(size_t index);
    void set_camera_to_training_view(size_t index);
    void set_floating_camera(size_t index);
    void set_neutral_camera();
    void save_depth(float *depth_cpu, const char *path, const char *name, Eigen::Vector2i res3d);

    bool m_is_recording = false;
    bool m_record_all = false;
    bool m_render_from_snapshot = false;
    bool m_single_step = false;
    bool m_save_depth = false;
    bool m_resume_training = false;
    uint32_t m_index_frame = 0;
    bool m_render_train_depth = false;
    int m_to_record = 0;
    int m_fps = 25;
    float m_current_fps = 25;
    float m_average_time = 0;
    int m_horizontal_angle = 30;
    uint32_t m_saved_images_to_load = 0;
    bool m_horizontal_normals = false;
    ngp::Testbed* m_ngp;
    std::string m_dst_folder = "overlay";
    char m_synthetic_version[128] = "v1";
    std::chrono::steady_clock::time_point m_start;
    int m_training_steps_wait = 22500;
    filesystem::path m_output_path;
    filesystem::path m_data_path;
    filesystem::path m_current_output;
    VideoType m_video_mode = VideoType::Overlay;
    Eigen::Matrix<float, 3, 4> m_neutral_camera = Eigen::Matrix<float, 3, 4>::Zero();
    Eigen::Matrix<float, 3, 4> m_average_camera = Eigen::Matrix<float, 3, 4>::Zero();
    Eigen::Vector2f m_neutral_focal_length;
    std::vector<Eigen::Matrix<float, 3, 4>> m_hemisphere_cameras;
    std::vector<Eigen::Matrix<float, 3, 4>> m_floating_cameras;
    std::vector<Eigen::Matrix<float, 3, 4>> m_horizontal_cameras;
    std::vector<Eigen::Matrix<float, 3, 4>> m_vertical_cameras;
    std::vector<Eigen::Matrix<float, 3, 4>> m_sweep_cameras;
    JsonCameras m_json_cameras{};
};

RTA_NAMESPACE_END

#endif //REAL_TIME_AVATAR_RECORDER_H
