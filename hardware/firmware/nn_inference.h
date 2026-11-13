/**
 * Minimal, zero-dependency ANSI C inference kernel for ECG_ResNet_SE
 * Optimized for ARM Cortex-M0+ (RP2040 Pico / Pico WH)
 */

#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

#include <math.h>
#include <string.h>
#include "model_data.h"

// Reusable activation buffers (Ping-pong double buffering in scratchpad RAM)
// Peak intermediate size is: 48 channels * 75 length * 4 bytes = 14,400 bytes (~14 KB)
static float ACT_BUF_A[48 * 150];
static float ACT_BUF_B[48 * 150];
static float RESIDUAL_BUF[48 * 75];

static inline float relu_act(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid_act(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * 1D Standard Convolution with stride
 */
static void conv1d(const float* in, int in_c, int in_len,
                   float* out, int out_c, int k, int stride, int pad,
                   const float* weights, const float* bias) {
    int out_len = (in_len + 2 * pad - k) / stride + 1;
    for (int oc = 0; oc < out_c; ++oc) {
        for (int l = 0; l < out_len; ++l) {
            float sum = bias[oc];
            int in_start = l * stride - pad;
            for (int ic = 0; ic < in_c; ++ic) {
                const float* in_ptr = in + ic * in_len;
                const float* w_ptr = weights + (oc * in_c + ic) * k;
                for (int ki = 0; ki < k; ++ki) {
                    int pos = in_start + ki;
                    if (pos >= 0 && pos < in_len) {
                        sum += in_ptr[pos] * w_ptr[ki];
                    }
                }
            }
            out[oc * out_len + l] = sum;
        }
    }
}

/**
 * 1D Depthwise Convolution (groups = channels)
 */
static void depthwise_conv1d(const float* in, float* out, int channels, int len, int k,
                             const float* weights, const float* bias) {
    int pad = k / 2;
    for (int c = 0; c < channels; ++c) {
        const float* in_ptr = in + c * len;
        float* out_ptr = out + c * len;
        const float* w_ptr = weights + c * k;
        float b = bias[c];

        for (int l = 0; l < len; ++l) {
            float sum = b;
            for (int ki = 0; ki < k; ++ki) {
                int pos = l - pad + ki;
                if (pos >= 0 && pos < len) {
                    sum += in_ptr[pos] * w_ptr[ki];
                }
            }
            out_ptr[l] = sum;
        }
    }
}

/**
 * 1D Pointwise Convolution (1x1 projection)
 */
static void pointwise_conv1d(const float* in, float* out, int in_c, int out_c, int len,
                             const float* weights, const float* bias) {
    for (int oc = 0; oc < out_c; ++oc) {
        float b = bias[oc];
        float* out_ptr = out + oc * len;
        for (int l = 0; l < len; ++l) {
            float sum = b;
            for (int ic = 0; ic < in_c; ++ic) {
                sum += in[ic * len + l] * weights[oc * in_c + ic];
            }
            out_ptr[l] = sum;
        }
    }
}

/**
 * MaxPool1D (stride=2, k=2)
 */
static void maxpool1d_s2(const float* in, float* out, int channels, int in_len) {
    int out_len = in_len / 2;
    for (int c = 0; c < channels; ++c) {
        const float* in_ptr = in + c * in_len;
        float* out_ptr = out + c * out_len;
        for (int l = 0; l < out_len; ++l) {
            float v1 = in_ptr[2 * l];
            float v2 = in_ptr[2 * l + 1];
            out_ptr[l] = v1 > v2 ? v1 : v2;
        }
    }
}

/**
 * Squeeze-and-Excitation (SE) Channel Attention
 */
static void se_block(float* x, int channels, int len, int reduction,
                     const float* w_fc1, const float* b_fc1,
                     const float* w_fc2, const float* b_fc2) {
    int mid = channels / reduction > 4 ? channels / reduction : 4;
    float avg[48];
    float mid_act[12];
    float se_weights[48];

    // 1. Global Average Pooling
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        const float* ptr = x + c * len;
        for (int l = 0; l < len; ++l) {
            sum += ptr[l];
        }
        avg[c] = sum / (float)len;
    }

    // 2. FC1 + ReLU
    for (int m = 0; m < mid; ++m) {
        float sum = b_fc1[m];
        for (int c = 0; c < channels; ++c) {
            sum += avg[c] * w_fc1[m * channels + c];
        }
        mid_act[m] = relu_act(sum);
    }

    // 3. FC2 + Sigmoid
    for (int c = 0; c < channels; ++c) {
        float sum = b_fc2[c];
        for (int m = 0; m < mid; ++m) {
            sum += mid_act[m] * w_fc2[c * mid + m];
        }
        se_weights[c] = sigmoid_act(sum);
    }

    // 4. Channel Scaling
    for (int c = 0; c < channels; ++c) {
        float scale = se_weights[c];
        float* ptr = x + c * len;
        for (int l = 0; l < len; ++l) {
            ptr[l] *= scale;
        }
    }
}

/**
 * End-to-End Forward Pass: Extracts 48-dim Latent Biometric Embedding
 */
static void run_embedding(const float* raw_beat_300, float* out_embedding_48) {
    // 1. Stem: Conv1D (1 -> 24, k=15, stride=2, pad=7) on 300 samples -> (24, 150)
    conv1d(raw_beat_300, 1, 300, ACT_BUF_A, 24, 15, 2, 7, W_STEM, B_STEM);
    for (int i = 0; i < 24 * 150; ++i) {
        ACT_BUF_A[i] = relu_act(ACT_BUF_A[i]);
    }

    // MaxPool1d(2) -> (24, 75)
    maxpool1d_s2(ACT_BUF_A, ACT_BUF_B, 24, 150);

    // Save residual for Block 1
    memcpy(RESIDUAL_BUF, ACT_BUF_B, 24 * 75 * sizeof(float));

    // 2. Block 1 (24 channels, length 75)
    depthwise_conv1d(ACT_BUF_B, ACT_BUF_A, 24, 75, 7, W_B1_C1, B_B1_C1);
    pointwise_conv1d(ACT_BUF_A, ACT_BUF_B, 24, 24, 75, W_B1_PW1, B_B1_PW1);
    for (int i = 0; i < 24 * 75; ++i) ACT_BUF_B[i] = relu_act(ACT_BUF_B[i]);

    depthwise_conv1d(ACT_BUF_B, ACT_BUF_A, 24, 75, 7, W_B1_C2, B_B1_C2);
    pointwise_conv1d(ACT_BUF_A, ACT_BUF_B, 24, 24, 75, W_B1_PW2, B_B1_PW2);

    se_block(ACT_BUF_B, 24, 75, 4, W_B1_SE_FC1, B_B1_SE_FC1, W_B1_SE_FC2, B_B1_SE_FC2);

    // Residual Add + ReLU
    for (int i = 0; i < 24 * 75; ++i) {
        ACT_BUF_B[i] = relu_act(ACT_BUF_B[i] + RESIDUAL_BUF[i]);
    }

    // 3. Transition 1: Conv1D (24 -> 48, 1x1) + MaxPool1d(2) -> (48, 37)
    pointwise_conv1d(ACT_BUF_B, ACT_BUF_A, 24, 48, 75, W_TRANS1, B_TRANS1);
    for (int i = 0; i < 48 * 75; ++i) ACT_BUF_A[i] = relu_act(ACT_BUF_A[i]);
    maxpool1d_s2(ACT_BUF_A, ACT_BUF_B, 48, 75); // Length is now 37

    // Save residual for Block 2
    memcpy(RESIDUAL_BUF, ACT_BUF_B, 48 * 37 * sizeof(float));

    // 4. Block 2 (48 channels, length 37)
    depthwise_conv1d(ACT_BUF_B, ACT_BUF_A, 48, 37, 5, W_B2_C1, B_B2_C1);
    pointwise_conv1d(ACT_BUF_A, ACT_BUF_B, 48, 48, 37, W_B2_PW1, B_B2_PW1);
    for (int i = 0; i < 48 * 37; ++i) ACT_BUF_B[i] = relu_act(ACT_BUF_B[i]);

    depthwise_conv1d(ACT_BUF_B, ACT_BUF_A, 48, 37, 5, W_B2_C2, B_B2_C2);
    pointwise_conv1d(ACT_BUF_A, ACT_BUF_B, 48, 48, 37, W_B2_PW2, B_B2_PW2);

    se_block(ACT_BUF_B, 48, 37, 4, W_B2_SE_FC1, B_B2_SE_FC1, W_B2_SE_FC2, B_B2_SE_FC2);

    for (int i = 0; i < 48 * 37; ++i) {
        ACT_BUF_B[i] = relu_act(ACT_BUF_B[i] + RESIDUAL_BUF[i]);
    }

    // 5. Global Adaptive Average Pooling -> 48-dim
    float emb_raw[48];
    for (int c = 0; c < 48; ++c) {
        float sum = 0.0f;
        const float* ptr = ACT_BUF_B + c * 37;
        for (int l = 0; l < 37; ++l) sum += ptr[l];
        emb_raw[c] = sum / 37.0f;
    }

    // 6. LayerNorm(48)
    float mean = 0.0f;
    for (int c = 0; c < 48; ++c) mean += emb_raw[c];
    mean /= 48.0f;
    float var = 0.0f;
    for (int c = 0; c < 48; ++c) {
        float diff = emb_raw[c] - mean;
        var += diff * diff;
    }
    var /= 48.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int c = 0; c < 48; ++c) {
        out_embedding_48[c] = (emb_raw[c] - mean) * inv_std * LN_W[c] + LN_B[c];
    }
}

/**
 * Cosine Similarity between two 48-dim vectors
 */
static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a <= 1e-12f || norm_b <= 1e-12f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

#endif // NN_INFERENCE_H
