/**
 * ecg_dsp.h — Real-Time 500 Hz Causal ECG DSP, R-Peak Segmentation & 5-Point SQI
 *
 * Zero-dependency ANSI C header designed for RP2040 (ARM Cortex-M0+) and x86 host testing.
 * Matches src/mlee/baseline.py preprocessing:
 *   - Sampling rate: 500 Hz (2 ms period)
 *   - Bandpass: 0.5 Hz (2nd-order Butterworth HPF) to 40.0 Hz (2nd-order Butterworth LPF)
 *   - Mains notch: 50.0 Hz (Q = 15, India AC grid; define ECG_NOTCH_60HZ for 60 Hz)
 *   - Beat window: 300 samples (100 pre-R + 200 post-R = 0.60 s), per-beat z-score normalized
 *
 * 5-Point Signal Quality Index (SQI) Gate:
 *   1. 3.0 s (1500-sample) settling hold-off after any LEADS OFF -> LEADS ON transition
 *   2. Zero leads-off flags across all 300 samples of the extracted beat
 *   3. Anti-clipping guard (raw 12-bit ADC strictly inside [50, 4045])
 *   4. Physiological RR-interval sanity (45–150 BPM -> 200..667 samples)
 *   5. QRS variance & crest-factor check (std >= 0.005, |R - mean|/std >= 2.2)
 */

#ifndef ECG_DSP_H
#define ECG_DSP_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define ECG_FS_HZ            500
#define ECG_BEAT_LEN         300
#define ECG_PRE_R            100
#define ECG_POST_R           200
#define ECG_HIST_LEN         512
#define ECG_HIST_MASK        (ECG_HIST_LEN - 1)
#define ECG_ENERGY_WIN       50    /* 100 ms moving average @ 500 Hz */
#define ECG_SETTLE_SAMPLES   1500  /* 3.0 s settling hold-off after leads attach */
#define ECG_REFRACTORY       150   /* 300 ms minimum QRS refractory */
#define ECG_RR_MIN           200   /* 400 ms -> 150 BPM max */
#define ECG_RR_MAX           667   /* 1334 ms -> 45 BPM min */
#define ECG_ADC_CLIP_LO      50
#define ECG_ADC_CLIP_HI      4045
#define ECG_MIN_BEAT_STD     0.004f
#define ECG_MIN_CREST        2.20f

typedef enum {
    DSP_LEADS_OFF = 0,
    DSP_SETTLING  = 1,
    DSP_READY     = 2
} ecg_dsp_state_t;

typedef struct {
    float z1;
    float z2;
} biquad_state_t;

typedef struct {
    /* Filter states */
    biquad_state_t hpf;
    biquad_state_t lpf;
    biquad_state_t notch;

    /* Circular sample histories (512 samples = 1.024 s) */
    float    filt_hist[ECG_HIST_LEN];
    uint16_t raw_hist[ECG_HIST_LEN];
    uint8_t  lo_hist[ECG_HIST_LEN];
    uint32_t sample_idx;

    /* QRS energy moving average (50 samples = 100 ms) */
    float prev_filt;
    float energy_ring[ECG_ENERGY_WIN];
    int   energy_pos;

    /* Adaptive QRS threshold tracking */
    float peak_env;
    float noise_env;
    bool  in_qrs;
    float cand_max_e;
    uint32_t cand_max_idx;
    uint32_t last_cand_r;

    /* Pending beat extraction (waiting for post-R samples to arrive) */
    bool     has_pending;
    uint32_t pending_r_idx;

    /* State, settling, RR/BPM, and SQI telemetry */
    ecg_dsp_state_t state;
    uint32_t settle_count;
    bool     has_prev_r;
    uint32_t last_valid_r;
    float    bpm;
    uint32_t beats_accepted;
    uint32_t sqi_rejects;
    const char *last_reject_reason;
} ecg_dsp_t;

/* Exact scipy.signal bilinear biquad coefficients @ fs = 500 Hz */
static inline float biquad_df2t(biquad_state_t *s, float x,
                                float b0, float b1, float b2,
                                float a1, float a2) {
    float y = b0 * x + s->z1;
    s->z1 = b1 * x - a1 * y + s->z2;
    s->z2 = b2 * x - a2 * y;
    return y;
}

static inline void ecg_dsp_reset_filters(ecg_dsp_t *st) {
    st->hpf.z1 = 0.0f; st->hpf.z2 = 0.0f;
    st->lpf.z1 = 0.0f; st->lpf.z2 = 0.0f;
    st->notch.z1 = 0.0f; st->notch.z2 = 0.0f;
    st->prev_filt = 0.0f;
    memset(st->energy_ring, 0, sizeof(st->energy_ring));
    st->energy_pos = 0;
    st->peak_env = 0.002f;
    st->noise_env = 0.0001f;
    st->in_qrs = false;
    st->cand_max_e = 0.0f;
    st->cand_max_idx = 0;
    st->has_pending = false;
    st->has_prev_r = false;
    st->bpm = 0.0f;
}

static inline void ecg_dsp_init(ecg_dsp_t *st) {
    memset(st, 0, sizeof(*st));
    st->state = DSP_LEADS_OFF;
    st->last_reject_reason = "NONE";
    ecg_dsp_reset_filters(st);
}

static inline float ecg_dsp_filter_sample(ecg_dsp_t *st, float x) {
    /* 1. 2nd-order Butterworth HPF @ 0.5 Hz */
    float y = biquad_df2t(&st->hpf, x,
                          0.9955669720f, -1.9911339440f, 0.9955669720f,
                          -1.9911142922f, 0.9911535959f);
    /* 2. 2nd-order Butterworth LPF @ 40.0 Hz */
    y = biquad_df2t(&st->lpf, y,
                    0.0461318021f, 0.0922636042f, 0.0461318021f,
                    -1.3072850288f, 0.4918122372f);
    /* 3. 2nd-order IIR Notch @ 50.0 Hz (Q = 15, India AC mains) */
    y = biquad_df2t(&st->notch, y,
                    0.9794827610f, -1.5848363987f, 0.9794827610f,
                    -1.5848363987f, 0.9589655220f);
    return y;
}

/**
 * Push one 500 Hz sample into the streaming DSP pipeline.
 *
 * @param st            Pointer to static ecg_dsp_t state
 * @param raw_adc       12-bit ADC sample [0..4095]
 * @param leads_off     true if AD8232 LO+ or LO- is HIGH
 * @param out_filt      Optional pointer to receive filtered float sample
 * @param out_beat_300  Buffer of 300 floats populated when function returns true
 * @return              true iff a quality-verified 300-sample beat was extracted
 */
static inline bool ecg_dsp_push(ecg_dsp_t *st, uint16_t raw_adc, bool leads_off,
                                float *out_filt, float *out_beat_300) {
    uint32_t idx = st->sample_idx;
    uint32_t slot = idx & ECG_HIST_MASK;

    st->raw_hist[slot] = raw_adc;
    st->lo_hist[slot]  = leads_off ? 1 : 0;

    if (leads_off) {
        if (st->state != DSP_LEADS_OFF) {
            ecg_dsp_reset_filters(st);
        }
        st->state = DSP_LEADS_OFF;
        st->settle_count = 0;
        st->filt_hist[slot] = 0.0f;
        if (out_filt) *out_filt = 0.0f;
        st->sample_idx = idx + 1;
        return false;
    }

    /* Convert 12-bit ADC [0..4095] to centered [-1.0 .. +1.0] */
    float x = ((float)raw_adc - 2048.0f) * (1.0f / 2048.0f);
    float y = ecg_dsp_filter_sample(st, x);
    st->filt_hist[slot] = y;
    if (out_filt) *out_filt = y;

    /* Update settling state (3.0 s hold-off after leads attach) */
    if (st->settle_count < ECG_SETTLE_SAMPLES) {
        st->settle_count++;
        st->state = DSP_SETTLING;
    } else {
        st->state = DSP_READY;
    }

    /* Update QRS derivative-squared 100 ms moving average energy */
    float diff = y - st->prev_filt;
    st->prev_filt = y;
    float e = diff * diff;

    st->energy_ring[st->energy_pos] = e;
    st->energy_pos = (st->energy_pos + 1) % ECG_ENERGY_WIN;
    float esum = 0.0f;
    for (int i = 0; i < ECG_ENERGY_WIN; ++i) esum += st->energy_ring[i];
    float energy_ma = esum * (1.0f / (float)ECG_ENERGY_WIN);

    /* Let filters stabilize for first 1.5 s before adapting envelopes */
    if (st->settle_count > (ECG_SETTLE_SAMPLES / 2)) {
        float thr = st->noise_env + 0.25f * (st->peak_env - st->noise_env);
        if (thr < 1e-6f) thr = 1e-6f;

        if (!st->in_qrs) {
            if (energy_ma > thr) {
                st->in_qrs = true;
                st->cand_max_e = energy_ma;
                st->cand_max_idx = idx;
            } else {
                st->noise_env = 0.992f * st->noise_env + 0.008f * energy_ma;
                st->peak_env  = 0.9992f * st->peak_env;
                if (st->peak_env < st->noise_env * 2.5f) {
                    st->peak_env = st->noise_env * 2.5f;
                }
            }
        } else {
            if (energy_ma > st->cand_max_e) {
                st->cand_max_e = energy_ma;
                st->cand_max_idx = idx;
            }
            /* End of QRS energy hump */
            if (energy_ma < 0.45f * st->cand_max_e || (idx - st->cand_max_idx) > 45) {
                st->in_qrs = false;
                st->peak_env = 0.80f * st->peak_env + 0.20f * st->cand_max_e;

                /* Refine true R-peak in filt_hist compensating for ~25-sample MA lag */
                if (st->state == DSP_READY && !st->has_pending &&
                    st->cand_max_idx >= 70 &&
                    (st->cand_max_idx - st->last_cand_r) >= ECG_REFRACTORY) {

                    uint32_t search_lo = st->cand_max_idx - 65;
                    uint32_t search_hi = st->cand_max_idx + 5;
                    if (search_hi > idx) search_hi = idx;

                    uint32_t best_r = search_lo;
                    float best_abs = -1.0f;
                    for (uint32_t k = search_lo; k <= search_hi; ++k) {
                        float a = fabsf(st->filt_hist[k & ECG_HIST_MASK]);
                        if (a > best_abs) {
                            best_abs = a;
                            best_r = k;
                        }
                    }

                    if (best_r >= ECG_PRE_R &&
                        (st->last_cand_r == 0 || (best_r - st->last_cand_r) >= ECG_REFRACTORY)) {
                        st->last_cand_r = best_r;
                        st->pending_r_idx = best_r;
                        st->has_pending = true;
                    }
                }
            }
        }
    }

    /* Check if a pending beat now has all 200 post-R samples available */
    bool beat_ready = false;
    if (st->has_pending && idx >= (st->pending_r_idx + ECG_POST_R - 1)) {
        uint32_t r_idx = st->pending_r_idx;
        st->has_pending = false;

        /* Ensure oldest sample (r_idx - 100) is still inside circular buffer */
        if (idx - (r_idx - ECG_PRE_R) < ECG_HIST_LEN && out_beat_300 != NULL) {
            uint32_t start = r_idx - ECG_PRE_R;

            /* SQI Check 2 & 3: Zero leads-off & anti-clipping across all 300 samples */
            bool sqi_lo_ok = true;
            bool sqi_clip_ok = true;
            float mean = 0.0f;
            for (int i = 0; i < ECG_BEAT_LEN; ++i) {
                uint32_t s = (start + (uint32_t)i) & ECG_HIST_MASK;
                if (st->lo_hist[s]) sqi_lo_ok = false;
                uint16_t raw = st->raw_hist[s];
                if (raw < ECG_ADC_CLIP_LO || raw > ECG_ADC_CLIP_HI) sqi_clip_ok = false;
                float v = st->filt_hist[s];
                out_beat_300[i] = v;
                mean += v;
            }
            mean /= (float)ECG_BEAT_LEN;

            float var = 0.0f;
            for (int i = 0; i < ECG_BEAT_LEN; ++i) {
                float d = out_beat_300[i] - mean;
                var += d * d;
            }
            float std = sqrtf(var / (float)ECG_BEAT_LEN);
            float r_peak_abs = fabsf(out_beat_300[ECG_PRE_R] - mean);
            float crest = (std > 1e-8f) ? (r_peak_abs / std) : 0.0f;

            /* SQI Check 4: Physiological RR-interval sanity (45..150 BPM) */
            bool rr_ok = false;
            if (!st->has_prev_r) {
                /* Anchor first clean beat to establish RR baseline */
                if (sqi_lo_ok && sqi_clip_ok && std >= ECG_MIN_BEAT_STD && crest >= ECG_MIN_CREST) {
                    st->has_prev_r = true;
                    st->last_valid_r = r_idx;
                }
            } else {
                uint32_t rr = r_idx - st->last_valid_r;
                if (rr >= ECG_RR_MIN && rr <= ECG_RR_MAX) {
                    rr_ok = true;
                    st->bpm = 30000.0f / (float)rr;
                    st->last_valid_r = r_idx;
                } else if (rr > ECG_RR_MAX) {
                    /* Missed beat or pause: re-anchor cleanly */
                    st->last_valid_r = r_idx;
                    st->sqi_rejects++;
                    st->last_reject_reason = "RR_SLOW";
                } else {
                    st->sqi_rejects++;
                    st->last_reject_reason = "RR_FAST";
                }
            }

            /* Apply all 5 SQI gates */
            if (!sqi_lo_ok) {
                st->sqi_rejects++;
                st->last_reject_reason = "LEADS_OFF";
            } else if (!sqi_clip_ok) {
                st->sqi_rejects++;
                st->last_reject_reason = "CLIP";
            } else if (std < ECG_MIN_BEAT_STD) {
                st->sqi_rejects++;
                st->last_reject_reason = "FLAT";
            } else if (crest < ECG_MIN_CREST) {
                st->sqi_rejects++;
                st->last_reject_reason = "CREST";
            } else if (rr_ok) {
                float inv_std = 1.0f / std;
                for (int i = 0; i < ECG_BEAT_LEN; ++i) {
                    out_beat_300[i] = (out_beat_300[i] - mean) * inv_std;
                }
                st->beats_accepted++;
                st->last_reject_reason = "OK";
                beat_ready = true;
            }
        }
    }

    st->sample_idx = idx + 1;
    return beat_ready;
}

#endif /* ECG_DSP_H */
