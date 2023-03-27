#pragma once
#include <fstream>
#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <curand_kernel.h>

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void extract_conditioning(
        const uint32_t n_elements,
        const uint32_t num_exp,
        const uint32_t input_stride,
        const uint32_t output_stride,
        const float* __restrict__ input,
        T* __restrict__ output
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    for (int j = 0; j < num_exp; ++j) {
        output[i + output_stride * j] = input[i * input_stride + j];
    }
}

template <typename T>
__global__ void copy_with_stride(
        const uint32_t n_elements,
        const uint32_t num_exp,
        const uint32_t input_stride,
        const uint32_t output_stride,
        const T* __restrict__ input,
        T* __restrict__ output
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    for (int j = 0; j < num_exp; ++j) {
        output[i + output_stride * j] = input[i + input_stride * j];
    }
}

template <typename T>
class NerfNetworkMapper : public NerfNetwork<T> {
public:
	using json = nlohmann::json;

    NerfNetworkMapper(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network, const json& mapper_network) : NerfNetwork<T>(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset, pos_encoding, dir_encoding, density_network, rgb_network, mapper_network), m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims, dir_encoding, rgb_alignment));
        m_n_extra_dims = 16;
        json local_cond_network_config = mapper_network;
        local_cond_network_config["n_input_dims"] = m_n_extra_dims;

        m_cond_network.reset(tcnn::create_network<T>(local_cond_network_config));

        json local_density_network_config = density_network;
        local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width() + m_cond_network->padded_output_width();
        if (!density_network.contains("n_output_dims")) {
            local_density_network_config["n_output_dims"] = 16;
        }
        m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		m_rgb_network_input_width = tcnn::next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
	}

	virtual ~NerfNetworkMapper() { }

    uint32_t get_mapper_output() override { return m_cond_network->output_width(); }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();
        tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
        tcnn::GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
        tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width() + m_cond_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
	    CUDA_CHECK_THROW(cudaMemsetAsync(density_network_input.data(), 0, density_network_input.n_bytes(), stream));

        auto pos_out = density_network_input.slice_rows(0, m_pos_encoding->padded_output_width());
		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
            pos_out,
			use_inference_params
		);

        if (m_n_extra_dims > 0) {
            auto conditioning = input.slice_rows(input.stride() - m_n_extra_dims, m_n_extra_dims);
            tcnn::GPUMatrixDynamic<T> cond_input = {m_cond_network->input_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
            tcnn::linear_kernel(extract_conditioning<T>, 0, stream, batch_size, m_n_extra_dims, conditioning.stride(), cond_input.stride(), conditioning.data(), cond_input.data());
            tcnn::GPUMatrixDynamic<T> cond_output = {m_cond_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
            m_cond_network->inference_mixed_precision(stream, cond_input, cond_output, use_inference_params);
            auto cond_density = density_network_input.slice_rows(m_pos_encoding->padded_output_width(), m_cond_network->padded_output_width());
            tcnn::linear_kernel(copy_with_stride<T>, 0, stream, batch_size, m_cond_network->padded_output_width(), cond_output.stride(), cond_density.stride(), cond_output.data(), cond_density.data());
        }

//        std::vector<float> input_host(input.n_elements());
//        CUDA_CHECK_THROW(cudaMemcpy(input_host.data(), input.data(), input.n_elements() * sizeof(float), cudaMemcpyDeviceToHost));
//        int n = 200;
//        int frame_id = int(input_host[n * (sizeof(NerfCoordinate) / sizeof(float) + m_n_extra_dims) + 4]);
//        std::ofstream out;
//        out.open("output.txt", std::ios_base::app);
//        if (frame_id > 12) {
//            out << "Frame id = " << frame_id << "\n";
//            std::vector<T> density_input_cpu(density_network_input.n_elements());
//            CUDA_CHECK_THROW(cudaMemcpy(density_input_cpu.data(), density_network_input.data(), density_network_input.n_elements() * sizeof(T), cudaMemcpyDeviceToHost));
//            int line = 0;
//            for (int j = 0; j < density_network_input.m(); ++j) {
//                out << std::to_string(line + 1) << " " << to_string_with_precision(float(density_input_cpu[n + j * density_network_input.stride()]), 10) << "\n";
//                line++;
//                if (j == m_pos_encoding->padded_output_width() - 1){
//                    out << "\nConditioning\n\n";
//                    line = 0;
//                }
//            }
//            out << std::endl;
//            out << std::endl;
//            exit(-1);
//        }
//        out.close();

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		tcnn::linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width() + m_cond_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
        CUDA_CHECK_THROW(cudaMemsetAsync(forward->density_network_input.data(), 0, forward->density_network_input.n_bytes(), stream));
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

        auto out = forward->density_network_input.slice_rows(0, m_pos_encoding->padded_output_width());
		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&out,
			use_inference_params,
			prepare_input_gradients
		);

        if (m_n_extra_dims > 0) {
            auto conditioning = input.slice_rows(input.stride() - m_n_extra_dims, m_n_extra_dims);
            forward->cond_network_input = {m_cond_network->input_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
            tcnn::linear_kernel(extract_conditioning<T>, 0, stream, batch_size, m_n_extra_dims, conditioning.stride(), forward->cond_network_input.stride(), conditioning.data(), forward->cond_network_input.data());
            forward->cond_network_output = {m_cond_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
            forward->cond_network_ctx = m_cond_network->forward(stream, forward->cond_network_input, &forward->cond_network_output, use_inference_params, prepare_input_gradients);
            auto cond_density = forward->density_network_input.slice_rows(m_pos_encoding->padded_output_width(), m_cond_network->padded_output_width());
            tcnn::linear_kernel(copy_with_stride<T>, 0, stream, batch_size, m_cond_network->padded_output_width(), forward->cond_network_output.stride(), cond_density.stride(), forward->cond_network_output.data(), cond_density.data());
        }

		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		if (output) {
			tcnn::linear_kernel(extract_density<T>, 0, stream,
				batch_size, m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1, padded_output_width(), forward->density_network_output.data(), output->data()+3
			);
		}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_density_network->padded_output_width() + m_cond_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width() + m_cond_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(
                stream,
                *forward.density_network_ctx,
                forward.density_network_input,
                forward.density_network_output,
                dL_ddensity_network_output,
                dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr,
                use_inference_params,
                param_gradients_mode
            );

        if (m_n_extra_dims > 0) {
            m_cond_network->backward(
                    stream,
                    *forward.cond_network_ctx,
                    forward.cond_network_input,
                    forward.cond_network_output,
                    dL_ddensity_network_input.slice_rows(m_pos_encoding->padded_output_width(), m_cond_network->padded_output_width()),
                    nullptr,
                    use_inference_params,
                    param_gradients_mode
            );
        }

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input.slice_rows(0, m_pos_encoding->padded_output_width()),
				dL_ddensity_network_input.slice_rows(0, m_pos_encoding->padded_output_width()),
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, default_rng_t rng, bool use_inference_params = true, float* extra_dims_gpu = nullptr, uint32_t num_frames = 0) override {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width() + m_cond_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

        auto out = density_network_input.slice_rows(0, m_pos_encoding->padded_output_width());
		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
            out,
			use_inference_params
		);

        if (m_n_extra_dims > 0) {
            auto exp_output = density_network_input.slice_rows(m_pos_encoding->padded_output_width(), m_cond_network->padded_output_width());
            tcnn::linear_kernel(extract_expressions_random<T>, 0, stream,
                                batch_size,
                                num_frames,
                                m_cond_network->padded_output_width(),
                                extra_dims_gpu,
                                exp_output.stride(),
                                rng,
                                exp_output.data());
        }

        m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);
	}

	std::unique_ptr<tcnn::Context> density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density_forward input must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

        auto out = forward->density_network_input.slice_rows(0, m_pos_encoding->padded_output_width());
		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&out,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->density_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_density_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

		return forward;
	}

	void density_backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) {
		if (input.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
			throw std::runtime_error("NerfNetwork::density_backward input must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		m_density_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_density_network->n_params();

        m_cond_network->set_params(
                params + offset,
                inference_params + offset,
                backward_params + offset,
                gradients + offset
        );
        offset += m_cond_network->n_params();

        m_rgb_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_dir_encoding->n_params();
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_density_network->n_params();

        m_cond_network->initialize_params(
                rnd,
                params_full_precision + offset,
                params + offset,
                inference_params + offset,
                backward_params + offset,
                gradients + offset,
                scale
        );
        offset += m_cond_network->n_params();

		m_rgb_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params() + m_cond_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->padded_output_width();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype",           "NerfNetwork"},
			{"pos_encoding",    m_pos_encoding->hyperparams()},
			{"mapper_network",    m_cond_network->hyperparams()},
			{"dir_encoding",    m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network",     m_rgb_network->hyperparams()},
		};
	}

private:
	std::unique_ptr<tcnn::Network<T>> m_cond_network;
	std::unique_ptr<tcnn::Network<T>> m_density_network;
	std::unique_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

    bool m_rand_initialized = false;

	// // Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrixDynamic<T> density_network_output;

        tcnn::GPUMatrixDynamic<T> cond_network_input;
        tcnn::GPUMatrixDynamic<T> cond_network_output;

		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> cond_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
	};
};

NGP_NAMESPACE_END
