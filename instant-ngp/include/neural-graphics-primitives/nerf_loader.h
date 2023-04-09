/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_loader.h
 *  @author Alex Evans, NVIDIA
 *  @brief  Ability to load nerf datasets.
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>

#include <filesystem/path.h>

#include <vector>
#include <map>
#include <mutex>

NGP_NAMESPACE_BEGIN

// how much to scale the scene by vs the original nerf dataset; we want to fit the thing in the unit cube
static constexpr float NERF_SCALE = 1.f;
static constexpr float MESH_SCALE = 1.f;

struct ImageCache {
public:
    uint16_t* get_map_depth(std::string path, Eigen::Vector2i &res) { return get_16bit_map(path, res); };
    uint8_t* get_map_image(std::string path, Eigen::Vector2i &res) { return get_8bit_map(path, res, 1); };
    uint8_t* get_map_seg(std::string path, Eigen::Vector2i &res) { return get_8bit_map(path, res, 2); };
private:
    uint8_t* get_8bit_map(std::string path, Eigen::Vector2i &res, int type);
    uint16_t* get_16bit_map(std::string path, Eigen::Vector2i &res);
    std::map<std::string, uint16_t*> depth_maps;
    std::map<std::string, uint8_t*> image_maps;
    std::map<std::string, uint8_t*> seg_maps;
    std::mutex lock;
    Eigen::Vector2i res;
};

struct TrainingImageMetadata {
	// Camera intrinsics and additional data associated with a NeRF training image
	// the memory to back the pixels and rays is held by GPUMemory objects in the NerfDataset and copied here.
	const void* pixels = nullptr;
	const void* seg_mask = nullptr;
	EImageDataType image_data_type = EImageDataType::Half;

	const float* depth = nullptr;
	const Ray* rays = nullptr;

	CameraDistortion camera_distortion = {};
	Eigen::Vector2i resolution = Eigen::Vector2i::Constant(512);
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	Eigen::Vector2f focal_length = Eigen::Vector2f::Constant(1000.f);
	Eigen::Vector4f rolling_shutter = Eigen::Vector4f::Zero();
	Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.f); // TODO: replace this with more generic float[] of task-specific metadata.
};

inline size_t image_type_size(EImageDataType type) {
	switch (type) {
		case EImageDataType::None: return 0;
		case EImageDataType::Byte: return 1;
		case EImageDataType::Half: return 2;
		case EImageDataType::Float: return 4;
		default: return 0;
	}
}

inline size_t depth_type_size(EDepthDataType type) {
	switch (type) {
		case EDepthDataType::UShort: return 2;
		case EDepthDataType::Float: return 4;
		default: return 0;
	}
}

struct NerfDataset {
	std::vector<tcnn::GPUMemory<Ray>> raymemory;
	std::vector<tcnn::GPUMemory<uint8_t>> pixelmemory;
	std::vector<tcnn::GPUMemory<uint8_t>> segmaskmemory;
	std::vector<tcnn::GPUMemory<float>> depthmemory;

	std::vector<TrainingImageMetadata> metadata;
	tcnn::GPUMemory<TrainingImageMetadata> metadata_gpu;

	void update_metadata(int first = 0, int last = -1);
    std::map<std::string, std::vector<int>> current_samples;
    std::vector<std::string> mesh_paths;
    std::vector<std::string> exp_paths;
	std::vector<TrainingXForm> xforms;
	std::vector<std::string> paths;
	tcnn::GPUMemory<float> sharpness_data;
	Eigen::Vector2i sharpness_resolution = {0, 0};
	tcnn::GPUMemory<float> envmap_data;

	BoundingBox render_aabb = {};
	Eigen::Matrix3f render_aabb_to_local = Eigen::Matrix3f::Identity();
	Eigen::Vector3f up = {0.0f, 1.0f, 0.0f};
	Eigen::Vector3f offset = {0.0f, 0.0f, 0.0f};
	size_t n_images = 0;
	Eigen::Vector2i envmap_resolution = {0, 0};
	float scale = 1.0f;
	int aabb_scale = 1;
	bool from_mitsuba = false;
	bool is_hdr = false;
	bool is_synthetic = false;
	bool wants_importance_sampling = true;
	bool has_rays = false;
    bool reenact = false;
    float density_mesh_distance = 0.025; // 2.5 cm
    uint32_t n_all_images = 0;
    uint32_t n_training_images = 0;
    uint32_t n_extra_learnable_dims = 0;
	bool has_light_dirs = false;

	uint32_t n_extra_dims() const {
		return (has_light_dirs ? 3u : 0u) + n_extra_learnable_dims;
	}

	void set_training_image(int frame_idx, const Eigen::Vector2i& image_resolution, const void* pixels, const void* seg_mask, const void* depth_pixels, float depth_scale, bool image_data_on_gpu, EImageDataType image_type, EDepthDataType depth_type, float sharpen_amount = 0.f, bool white_transparent = false, bool black_transparent = false, uint32_t mask_color = 0, const Ray *rays = nullptr);

	Eigen::Vector3f nerf_direction_to_ngp(const Eigen::Vector3f& nerf_dir) {
		Eigen::Vector3f result = nerf_dir;
		if (from_mitsuba) {
			result *= -1;
		} else {
			result=Eigen::Vector3f(result.y(), result.z(), result.x());
		}
		return result;
	}

    Eigen::Matrix<float, 3, 4> mica_matrix_to_ngp(const Eigen::Matrix<float, 3, 4>& nerf_matrix) {
        Eigen::Matrix<float, 3, 4> result = nerf_matrix;
        result.col(3) = result.col(3) * MESH_SCALE + offset;
        return result;
    }

	Eigen::Matrix<float, 3, 4> nerf_matrix_to_ngp(const Eigen::Matrix<float, 3, 4>& nerf_matrix) {
		Eigen::Matrix<float, 3, 4> result = nerf_matrix;
		result.col(1) *= -1;
		result.col(2) *= -1;
		result.col(3) = result.col(3) * scale + offset;

		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz<-yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(1);
			result.row(1) = (Eigen::Vector4f)result.row(2);
			result.row(2) = tmp;
		}

		return result;
	}

	Eigen::Matrix<float, 3, 4> ngp_matrix_to_nerf(const Eigen::Matrix<float, 3, 4>& ngp_matrix) {
		Eigen::Matrix<float, 3, 4> result = ngp_matrix;
		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz->yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(2);
			result.row(2) = (Eigen::Vector4f)result.row(1);
			result.row(1) = tmp;
		}
		result.col(1) *= -1;
		result.col(2) *= -1;
		result.col(3) = (result.col(3) - offset) / scale;
		return result;
	}

	void nerf_ray_to_ngp(Ray& ray, bool scale_direction = false) {
		ray.o = ray.o * scale + offset;
		if (scale_direction)
			ray.d *= scale;

		float tmp = ray.o[0];
		ray.o[0] = ray.o[1];
		ray.o[1] = ray.o[2];
		ray.o[2] = tmp;

		tmp = ray.d[0];
		ray.d[0] = ray.d[1];
		ray.d[1] = ray.d[2];
		ray.d[2] = tmp;
	}
};

inline std::string replace(const std::string src, const std::string from, const std::string to) {
    auto str = src;
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        throw std::runtime_error("std::string replace()");
    str.replace(start_pos, from.length(), to);
    return str;
}

struct DatasetSettings {
    bool is_training = true;
    bool is_retargeting = false;
    bool is_rendering_depth = false;
    bool load_to_gpu = true;
    bool load_all_training = false;
    bool optimize_latent_code = false;
    bool shuffle = true;
    bool use_dataset_cache = true;
    std::string synthetic_path = "";
    uint32_t images_to_load = 1700;
};

static DatasetSettings default_train_settings = {};
NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount = 0.f, DatasetSettings dataset_settings = default_train_settings);
NerfDataset create_empty_nerf_dataset(size_t n_images, int aabb_scale = 1, bool is_hdr = false);

NGP_NAMESPACE_END
