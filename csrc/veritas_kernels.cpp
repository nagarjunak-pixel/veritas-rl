// Native C++20 Fused RLVR (GRPO/DAPO), Schulman GAE + k3 KL, and Online-Softmax Ring-Attention Kernels
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" {

// 1. Group-Relative Policy Optimization (GRPO) Advantage Normalization over (B, G)
void veritas_group_relative_advantages_f32(
    const float* rewards,
    float* advantages_out,
    int32_t num_groups,
    int32_t group_size,
    float eps
) {
    for (int32_t b = 0; b < num_groups; ++b) {
        const float* row = rewards + static_cast<size_t>(b) * group_size;
        float* out = advantages_out + static_cast<size_t>(b) * group_size;

        double sum = 0.0;
        for (int32_t g = 0; g < group_size; ++g) {
            sum += static_cast<double>(row[g]);
        }
        const double mean = sum / static_cast<double>(group_size);

        double sq_sum = 0.0;
        for (int32_t g = 0; g < group_size; ++g) {
            const double diff = static_cast<double>(row[g]) - mean;
            sq_sum += diff * diff;
        }
        const double std_dev = std::sqrt(sq_sum / static_cast<double>(group_size) + static_cast<double>(eps));

        for (int32_t g = 0; g < group_size; ++g) {
            out[g] = static_cast<float>((static_cast<double>(row[g]) - mean) / std_dev);
        }
    }
}

// 2. Generalized Advantage Estimation (GAE-lambda) + Schulman Non-Negative k3 KL + DAPO Clipping Diagnostics
void veritas_gae_and_k3_kl_f32(
    const float* rewards,
    const float* values,
    const float* logp_new,
    const float* logp_old,
    const float* logp_ref,
    int32_t length,
    float gamma,
    float gae_lambda,
    float clip_eps_low,
    float clip_eps_high,
    float* gae_out,
    float* k3_kl_out,
    float* surrogate_out,
    int32_t* clipped_flags_out
) {
    double last_gae = 0.0;
    for (int32_t t = length - 1; t >= 0; --t) {
        const double next_v = (t + 1 < length) ? static_cast<double>(values[t + 1]) : 0.0;
        const double delta = static_cast<double>(rewards[t]) + static_cast<double>(gamma) * next_v - static_cast<double>(values[t]);
        last_gae = delta + static_cast<double>(gamma) * static_cast<double>(gae_lambda) * last_gae;
        gae_out[t] = static_cast<float>(last_gae);
    }

    const double lo_bound = 1.0 - static_cast<double>(clip_eps_low);
    const double hi_bound = 1.0 + static_cast<double>(clip_eps_high);

    for (int32_t t = 0; t < length; ++t) {
        const double log_ratio_ref = static_cast<double>(logp_ref[t]) - static_cast<double>(logp_new[t]);
        const double k3 = std::exp(log_ratio_ref) - log_ratio_ref - 1.0;
        k3_kl_out[t] = static_cast<float>(std::max(0.0, k3));

        const double ratio = std::exp(static_cast<double>(logp_new[t]) - static_cast<double>(logp_old[t]));
        const double clipped_ratio = std::clamp(ratio, lo_bound, hi_bound);
        const double adv = static_cast<double>(gae_out[t]);
        const double unclipped_obj = ratio * adv;
        const double clipped_obj = clipped_ratio * adv;

        surrogate_out[t] = static_cast<float>(std::min(unclipped_obj, clipped_obj));
        clipped_flags_out[t] = (ratio < lo_bound || ratio > hi_bound) ? 1 : 0;
    }
}

// 3. Online-Softmax Ring-Attention Running Accumulator Merge (Milakov-Gimelshein / Dao / Liu)
// Merges block logits `scores` (len B_kv) and `v_block` (B_kv, D) into running `(out_acc, m_state, l_state)`
void veritas_online_softmax_merge_f32(
    const float* scores,
    const float* v_block,
    int32_t block_len,
    int32_t head_dim,
    float* out_acc,
    float* m_state,
    float* l_state
) {
    if (block_len <= 0) {
        return;
    }
    float m_block = scores[0];
    for (int32_t j = 1; j < block_len; ++j) {
        if (scores[j] > m_block) {
            m_block = scores[j];
        }
    }

    const float m_old = *m_state;
    const float l_old = *l_state;
    const float m_new = std::max(m_old, m_block);

    const double alpha_old = (l_old > 0.0f) ? std::exp(static_cast<double>(m_old - m_new)) : 0.0;
    double l_block = 0.0;
    for (int32_t d = 0; d < head_dim; ++d) {
        out_acc[d] = static_cast<float>(static_cast<double>(out_acc[d]) * alpha_old);
    }

    for (int32_t j = 0; j < block_len; ++j) {
        const double w = std::exp(static_cast<double>(scores[j] - m_new));
        l_block += w;
        const float* v_row = v_block + static_cast<size_t>(j) * head_dim;
        for (int32_t d = 0; d < head_dim; ++d) {
            out_acc[d] = static_cast<float>(static_cast<double>(out_acc[d]) + w * static_cast<double>(v_row[d]));
        }
    }

    const double l_new = static_cast<double>(l_old) * alpha_old + l_block;
    *m_state = m_new;
    *l_state = static_cast<float>(l_new);
}

}  // extern "C"
