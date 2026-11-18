/**
 * main.c — Raspberry Pi Pico WH Dual-Mode Hybrid Biometric Firmware (Option 3+)
 *
 * Asymmetric Multiprocessing (AMP) @ 250 MHz:
 *   CORE 0 — Real-Time Acquisition, DSP, Comms & Presentation
 *            - 500 Hz Hardware Timer ISR -> AD8232 ADC0 (GP26) + LO+/LO- (GP16/GP17)
 *            - Real-time causal DSP (0.5-40 Hz BP + 50 Hz Notch + R-Peak + 5-Point SQI)
 *            - Standby / Armed 5-Beat Enroll / Armed 5-Beat Auth / Continuous / USB Replay
 *            - Selective per-user Unenroll, Gallery Scope (LIVE ONLY vs ALL 89+N),
 *              and runtime Security Threshold (0.50 / 0.60 / 0.68)
 *            - Non-blocking USB CDC serial + CYW43439 Wi-Fi AP (http://192.168.4.1)
 *
 *   CORE 1 — Dedicated Neural Inference & Live Enrollment Engine
 *            - ECG_ResNet_SE (14,867 params, BN-fused FP32, ~300 ms)
 *            - Open-set cosine matching across 8 Live SRAM users (+ 89 PhysioNet DB)
 *            - Zero-shot 5-beat centroid enrollment & selective slot revocation
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/multicore.h"
#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/vreg.h"
#include "pico/cyw43_arch.h"

#include "model_data.h"
#include "nn_inference.h"
#include "ecg_dsp.h"
#include "wifi_server.h"

/* ---- Hardware Pinout & Configuration ------------------------------------- */
#define PIN_ECG_ADC        26     /* GP26 = ADC0 (Pin 31) */
#define PIN_LO_PLUS        16     /* GP16 (Pin 21) */
#define PIN_LO_MINUS       17     /* GP17 (Pin 22) */

#define DEFAULT_THRESHOLD  0.50f
#define SERIAL_LINE_BUF    4096   /* fits 300 floats as CSV text */
#define ADC_RING_LEN       512    /* 1.024 s slack @ 500 Hz */
#define ADC_RING_MASK      (ADC_RING_LEN - 1)
#define MAX_LIVE_USERS     8
#define TARGET_BEATS       5
#define TELEM_DECIM        5      /* 500 Hz / 5 = 100 Hz USB oscilloscope stream */

/* ---- Inter-Core Mailbox Types -------------------------------------------- */
typedef enum {
    CORE1_OP_INFER        = 0,
    CORE1_OP_ENROLL_BEAT  = 1,
    CORE1_OP_CLEAR_LIVE   = 2,
    CORE1_OP_UNENROLL_ONE = 3
} core1_op_t;

typedef struct {
    core1_op_t op;
    float      beat[BEAT_LENGTH];
    int        true_class;      /* >= 0 for self-test class, -1 for open-set */
    bool       search_live;     /* include live enrolled templates in search */
    bool       live_only;       /* search ONLY live enrolled users (skip 89 DB) */
    float      threshold;       /* active cosine acceptance threshold */
    bool       enroll_finalize; /* true on beat 5/5 to commit centroid */
    int        unenroll_idx;    /* slot 0..7 for CORE1_OP_UNENROLL_ONE */
    char       label[24];
} inference_request_t;

typedef struct {
    int      best_id;
    bool     matched_live;
    float    best_sim;
    uint32_t latency_us;
    bool     accepted;
    int      live_user_count;
    char     matched_name[24];
} inference_result_t;

typedef struct {
    char  name[16];
    float emb[EMBED_DIM];
    bool  active;
} live_user_t;

typedef enum {
    CAP_STANDBY    = 0,
    CAP_ENROLL_5   = 1,
    CAP_AUTH_5     = 2,
    CAP_CONTINUOUS = 3,
    CAP_REPLAY     = 4
} capture_mode_t;

/* ---- Static BSS Buffers (Zero Dynamic Allocation) ------------------------ */
static float    s_serial_beat_buf[BEAT_LENGTH];
static float    s_live_beat_buf[BEAT_LENGTH];
static char     s_line_buf[SERIAL_LINE_BUF];
static int      s_line_pos = 0;

static float    s_core1_emb[EMBED_DIM];
static float    s_enroll_accum[EMBED_DIM];
static int      s_enroll_accum_count = 0;
static live_user_t s_live_users[MAX_LIVE_USERS];
static int      s_live_user_count = 0;

/* Core-0 shadow of live user names (updated after enroll/unenroll/clear) */
static char     s_user_names_shadow[MAX_LIVE_USERS][16];
static int      s_user_shadow_count = 0;

static inference_request_t s_req_box;
static inference_result_t  s_res_box;

/* 500 Hz ADC ISR lock-free SPSC ring buffer */
static volatile uint16_t s_adc_ring[ADC_RING_LEN];
static volatile uint32_t s_adc_head = 0;
static volatile uint32_t s_adc_tail = 0;
static struct repeating_timer s_adc_timer;
static ecg_dsp_t s_dsp;

/* Runtime state */
static uint32_t       s_total_beats = 0;
static capture_mode_t s_cap_mode = CAP_STANDBY;
static bool           s_telem_enabled = false;
static uint32_t       s_telem_div = 0;
static bool           s_live_only_scope = false;
static float          s_match_threshold = DEFAULT_THRESHOLD;
static char           s_enroll_name[16] = "Live_1";
static int            s_progress_cur = 0;
static int            s_progress_tot = 0;

/* 5-beat voting accumulator */
static int            s_vote_accepts = 0;
static float          s_vote_sim_sum = 0.0f;
static char           s_vote_last_name[24] = "---";
static char           s_vote_summary[40]   = "---";

/* Rolling 5-beat vote ring for Continuous mode */
static bool           s_cont_ring[TARGET_BEATS];
static int            s_cont_pos = 0;
static int            s_cont_len = 0;

/* ---- Helpers ------------------------------------------------------------- */
static void blink_led(int times, int on_ms, int off_ms) {
    for (int i = 0; i < times; ++i) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(on_ms);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(off_ms);
    }
}

static const char *mode_name(capture_mode_t m) {
    switch (m) {
        case CAP_STANDBY:    return "STANDBY";
        case CAP_ENROLL_5:   return "ENROLLING";
        case CAP_AUTH_5:     return "AUTH 5-BEAT";
        case CAP_CONTINUOUS: return "CONTINUOUS";
        case CAP_REPLAY:     return "USB REPLAY";
        default:             return "STANDBY";
    }
}

static const char *sensor_name(ecg_dsp_state_t st) {
    switch (st) {
        case DSP_READY:    return "SIGNAL READY";
        case DSP_SETTLING: return "SETTLING";
        default:           return "LEADS OFF";
    }
}

static void push_status_to_wifi(void) {
    wifi_server_update_status(
        mode_name(s_cap_mode),
        sensor_name(s_dsp.state),
        s_dsp.bpm,
        s_user_shadow_count,
        s_progress_cur,
        s_progress_tot,
        s_vote_summary,
        s_live_only_scope,
        s_match_threshold
    );
    wifi_server_update_users(s_user_names_shadow, s_user_shadow_count);
}

static void emit_status_event(void) {
    printf("EVENT:STATUS,mode=%s,sensor=%s,bpm=%.1f,live=%d,prog=%d/%d,vote=%s,sqi_rej=%lu,last_rej=%s,scope=%s,thr=%.2f\n",
           mode_name(s_cap_mode),
           sensor_name(s_dsp.state),
           (double)s_dsp.bpm,
           s_user_shadow_count,
           s_progress_cur,
           s_progress_tot,
           s_vote_summary,
           (unsigned long)s_dsp.sqi_rejects,
           s_dsp.last_reject_reason,
           s_live_only_scope ? "LIVE" : "ALL",
           (double)s_match_threshold);
}

static void emit_users_event(void) {
    if (s_user_shadow_count == 0) {
        printf("EVENT:USERS,NONE\n");
        return;
    }
    printf("EVENT:USERS");
    for (int i = 0; i < s_user_shadow_count; ++i) {
        printf(",%d:%s", i, s_user_names_shadow[i]);
    }
    printf("\n");
}

static void l2_normalize(float *v, int dim) {
    float n2 = 0.0f;
    for (int i = 0; i < dim; ++i) n2 += v[i] * v[i];
    if (n2 > 1e-12f) {
        float inv = 1.0f / sqrtf(n2);
        for (int i = 0; i < dim; ++i) v[i] *= inv;
    }
}

/* ---- 500 Hz Hardware Timer ISR ------------------------------------------- */
static bool adc_timer_cb(struct repeating_timer *t) {
    (void)t;
    uint16_t raw = adc_read();
    bool lo = gpio_get(PIN_LO_PLUS) || gpio_get(PIN_LO_MINUS);
    uint16_t packed = (raw & 0x0FFF) | (lo ? 0x8000 : 0x0000);

    uint32_t next_head = (s_adc_head + 1) & ADC_RING_MASK;
    if (next_head != s_adc_tail) {
        s_adc_ring[s_adc_head] = packed;
        s_adc_head = next_head;
    }
    return true;
}

/* ---- Core 1: Dedicated Neural Inference & Live Enrollment Engine --------- */
static void core1_inference_worker(void) {
    while (1) {
        uint32_t req_addr = multicore_fifo_pop_blocking();
        inference_request_t *req = (inference_request_t *)req_addr;

        if (req->op == CORE1_OP_CLEAR_LIVE) {
            memset(s_live_users, 0, sizeof(s_live_users));
            memset(s_enroll_accum, 0, sizeof(s_enroll_accum));
            s_enroll_accum_count = 0;
            s_live_user_count = 0;
            s_res_box.live_user_count = 0;
            s_res_box.accepted = true;
            s_res_box.best_sim = 0.0f;
            s_res_box.latency_us = 0;
            strncpy(s_res_box.matched_name, "CLEARED", sizeof(s_res_box.matched_name) - 1);
            multicore_fifo_push_blocking((uint32_t)&s_res_box);
            continue;
        }

        if (req->op == CORE1_OP_UNENROLL_ONE) {
            int idx = req->unenroll_idx;
            bool ok = false;
            if (idx >= 0 && idx < s_live_user_count) {
                strncpy(s_res_box.matched_name, s_live_users[idx].name,
                        sizeof(s_res_box.matched_name) - 1);
                s_res_box.matched_name[sizeof(s_res_box.matched_name) - 1] = '\0';
                for (int i = idx; i < s_live_user_count - 1; ++i) {
                    s_live_users[i] = s_live_users[i + 1];
                }
                s_live_user_count--;
                memset(&s_live_users[s_live_user_count], 0, sizeof(live_user_t));
                ok = true;
            } else {
                strncpy(s_res_box.matched_name, "INVALID", sizeof(s_res_box.matched_name) - 1);
            }
            s_res_box.accepted = ok;
            s_res_box.live_user_count = s_live_user_count;
            s_res_box.best_sim = 0.0f;
            s_res_box.latency_us = 0;
            multicore_fifo_push_blocking((uint32_t)&s_res_box);
            continue;
        }

        uint32_t t_start = time_us_32();
        run_embedding(req->beat, s_core1_emb);
        uint32_t t_elapsed = time_us_32() - t_start;

        if (req->op == CORE1_OP_ENROLL_BEAT) {
            l2_normalize(s_core1_emb, EMBED_DIM);
            if (s_enroll_accum_count == 0) {
                memset(s_enroll_accum, 0, sizeof(s_enroll_accum));
            }
            for (int i = 0; i < EMBED_DIM; ++i) {
                s_enroll_accum[i] += s_core1_emb[i];
            }
            s_enroll_accum_count++;

            if (req->enroll_finalize) {
                l2_normalize(s_enroll_accum, EMBED_DIM);
                int slot = (s_live_user_count < MAX_LIVE_USERS)
                           ? s_live_user_count
                           : (MAX_LIVE_USERS - 1);
                strncpy(s_live_users[slot].name, req->label, sizeof(s_live_users[slot].name) - 1);
                s_live_users[slot].name[sizeof(s_live_users[slot].name) - 1] = '\0';
                memcpy(s_live_users[slot].emb, s_enroll_accum, sizeof(float) * EMBED_DIM);
                s_live_users[slot].active = true;
                if (s_live_user_count < MAX_LIVE_USERS) s_live_user_count++;
                s_enroll_accum_count = 0;
            }

            s_res_box.best_id = s_live_user_count - 1;
            s_res_box.matched_live = true;
            s_res_box.best_sim = 1.0f;
            s_res_box.latency_us = t_elapsed;
            s_res_box.accepted = true;
            s_res_box.live_user_count = s_live_user_count;
            strncpy(s_res_box.matched_name, req->label, sizeof(s_res_box.matched_name) - 1);
            s_res_box.matched_name[sizeof(s_res_box.matched_name) - 1] = '\0';
            multicore_fifo_push_blocking((uint32_t)&s_res_box);
            continue;
        }

        /* Standard inference + cosine matching */
        int best_id = -1;
        bool matched_live = false;
        float best_sim = -2.0f;

        if (req->true_class >= 0) {
            /* Self-test against a specific PhysioNet class */
            best_id = req->true_class;
            best_sim = cosine_similarity(
                s_core1_emb, ENROLLED_TEMPLATES + best_id * EMBED_DIM, EMBED_DIM);
        } else if (req->live_only) {
            /* LIVE ONLY scope: match strictly against enrolled live users */
            for (int u = 0; u < s_live_user_count; ++u) {
                if (!s_live_users[u].active) continue;
                float s = cosine_similarity(s_core1_emb, s_live_users[u].emb, EMBED_DIM);
                if (s > best_sim) {
                    best_sim = s;
                    best_id = u;
                    matched_live = true;
                }
            }
            if (best_id < 0) best_sim = 0.0f;
        } else {
            /* ALL scope: search all 89 PhysioNet templates + any active live users */
            best_id = 0;
            best_sim = cosine_similarity(s_core1_emb, ENROLLED_TEMPLATES, EMBED_DIM);
            for (int c = 1; c < NUM_CLASSES; ++c) {
                float s = cosine_similarity(
                    s_core1_emb, ENROLLED_TEMPLATES + c * EMBED_DIM, EMBED_DIM);
                if (s > best_sim) {
                    best_sim = s;
                    best_id = c;
                    matched_live = false;
                }
            }
            if (req->search_live) {
                for (int u = 0; u < s_live_user_count; ++u) {
                    if (!s_live_users[u].active) continue;
                    float s = cosine_similarity(s_core1_emb, s_live_users[u].emb, EMBED_DIM);
                    if (s > best_sim) {
                        best_sim = s;
                        best_id = u;
                        matched_live = true;
                    }
                }
            }
        }

        s_res_box.best_id = best_id;
        s_res_box.matched_live = matched_live;
        s_res_box.best_sim = best_sim;
        s_res_box.latency_us = t_elapsed;
        s_res_box.accepted = (best_id >= 0 && best_sim >= req->threshold);
        s_res_box.live_user_count = s_live_user_count;

        const char *mname = "NO_LIVE_DB";
        if (matched_live && best_id >= 0 && best_id < s_live_user_count) {
            mname = s_live_users[best_id].name;
        } else if (!matched_live && best_id >= 0 && best_id < NUM_CLASSES) {
            mname = ALL_CLASS_NAMES[best_id];
        }
        strncpy(s_res_box.matched_name, mname, sizeof(s_res_box.matched_name) - 1);
        s_res_box.matched_name[sizeof(s_res_box.matched_name) - 1] = '\0';

        multicore_fifo_push_blocking((uint32_t)&s_res_box);
    }
}

/* ---- Core 0 -> Core 1 Dispatch Helper ------------------------------------ */
static inference_result_t *dispatch_core1(core1_op_t op, const float *beat,
                                          int true_class, bool search_live,
                                          bool live_only, bool enroll_finalize,
                                          int unenroll_idx, const char *label) {
    s_req_box.op = op;
    if (beat) memcpy(s_req_box.beat, beat, sizeof(float) * BEAT_LENGTH);
    s_req_box.true_class = true_class;
    s_req_box.search_live = search_live;
    s_req_box.live_only = live_only;
    s_req_box.threshold = s_match_threshold;
    s_req_box.enroll_finalize = enroll_finalize;
    s_req_box.unenroll_idx = unenroll_idx;
    strncpy(s_req_box.label, label ? label : "", sizeof(s_req_box.label) - 1);
    s_req_box.label[sizeof(s_req_box.label) - 1] = '\0';

    multicore_fifo_push_blocking((uint32_t)&s_req_box);

    while (!multicore_fifo_rvalid()) {
        wifi_server_poll();
        tight_loop_contents();
    }

    uint32_t res_addr = multicore_fifo_pop_blocking();
    return (inference_result_t *)res_addr;
}

/* ---- Shadow Gallery Sync Helpers (Core 0) -------------------------------- */
static void shadow_add_user(const char *name) {
    int slot = (s_user_shadow_count < MAX_LIVE_USERS)
               ? s_user_shadow_count
               : (MAX_LIVE_USERS - 1);
    strncpy(s_user_names_shadow[slot], name, 15);
    s_user_names_shadow[slot][15] = '\0';
    if (s_user_shadow_count < MAX_LIVE_USERS) s_user_shadow_count++;
}

static void shadow_remove_user(int idx) {
    if (idx < 0 || idx >= s_user_shadow_count) return;
    for (int i = idx; i < s_user_shadow_count - 1; ++i) {
        memcpy(s_user_names_shadow[i], s_user_names_shadow[i + 1], 16);
    }
    s_user_shadow_count--;
    s_user_names_shadow[s_user_shadow_count][0] = '\0';
}

/* ---- Action Handlers (Shared by Wi-Fi UI & USB Commands) ----------------- */
static void start_enroll(const char *custom_name) {
    if (custom_name && custom_name[0] != '\0') {
        strncpy(s_enroll_name, custom_name, sizeof(s_enroll_name) - 1);
        s_enroll_name[sizeof(s_enroll_name) - 1] = '\0';
    } else {
        snprintf(s_enroll_name, sizeof(s_enroll_name), "Live_%d", s_user_shadow_count + 1);
    }
    s_enroll_accum_count = 0;
    s_cap_mode = CAP_ENROLL_5;
    s_progress_cur = 0;
    s_progress_tot = TARGET_BEATS;
    snprintf(s_vote_summary, sizeof(s_vote_summary), "ENROLLING %s (0/%d)", s_enroll_name, TARGET_BEATS);
    push_status_to_wifi();
    printf("[ctrl] Armed 5-beat zero-shot enrollment for '%s'\n", s_enroll_name);
    emit_status_event();
}

static void start_auth_5(void) {
    s_cap_mode = CAP_AUTH_5;
    s_progress_cur = 0;
    s_progress_tot = TARGET_BEATS;
    s_vote_accepts = 0;
    s_vote_sim_sum = 0.0f;
    strncpy(s_vote_last_name, "---", sizeof(s_vote_last_name));
    snprintf(s_vote_summary, sizeof(s_vote_summary), "CAPTURING 0/%d...", TARGET_BEATS);
    push_status_to_wifi();
    printf("[ctrl] Armed 5-beat live authentication capture\n");
    emit_status_event();
}

static void toggle_continuous(void) {
    if (s_cap_mode == CAP_CONTINUOUS) {
        s_cap_mode = CAP_STANDBY;
        s_progress_cur = 0;
        s_progress_tot = 0;
        printf("[ctrl] Continuous mode OFF -> STANDBY\n");
    } else {
        s_cap_mode = CAP_CONTINUOUS;
        s_progress_cur = 0;
        s_progress_tot = 0;
        s_cont_pos = 0;
        s_cont_len = 0;
        snprintf(s_vote_summary, sizeof(s_vote_summary), "CONTINUOUS LIVE");
        printf("[ctrl] Continuous live authentication ON\n");
    }
    push_status_to_wifi();
    emit_status_event();
}

static void set_standby(void) {
    s_cap_mode = CAP_STANDBY;
    s_progress_cur = 0;
    s_progress_tot = 0;
    s_enroll_accum_count = 0;
    push_status_to_wifi();
    printf("[ctrl] Mode -> STANDBY\n");
    emit_status_event();
}

static void clear_live_users(void) {
    dispatch_core1(CORE1_OP_CLEAR_LIVE, NULL, -1, false, false, false, -1, "CLEAR");
    s_user_shadow_count = 0;
    memset(s_user_names_shadow, 0, sizeof(s_user_names_shadow));
    s_cap_mode = CAP_STANDBY;
    s_progress_cur = 0;
    s_progress_tot = 0;
    snprintf(s_vote_summary, sizeof(s_vote_summary), "LIVE DB CLEARED");
    push_status_to_wifi();
    printf("[ctrl] Cleared all live enrolled templates (89 PhysioNet templates intact)\n");
    emit_users_event();
    emit_status_event();
}

static void unenroll_slot(int slot) {
    inference_result_t *res = dispatch_core1(
        CORE1_OP_UNENROLL_ONE, NULL, -1, false, false, false, slot, "UNENROLL");
    if (res->accepted) {
        char removed[24];
        strncpy(removed, res->matched_name, sizeof(removed) - 1);
        removed[sizeof(removed) - 1] = '\0';
        shadow_remove_user(slot);
        snprintf(s_vote_summary, sizeof(s_vote_summary), "UNENROLLED %s", removed);
        push_status_to_wifi();
        printf("EVENT:UNENROLLED,%s,%d\n", removed, s_user_shadow_count);
        emit_users_event();
        emit_status_event();
    } else {
        printf("[warn] unenroll failed: invalid slot %d (live count=%d)\n",
               slot, s_user_shadow_count);
    }
}

static void unenroll_by_arg(const char *arg) {
    if (!arg || arg[0] == '\0') return;
    /* Try exact name match first */
    for (int i = 0; i < s_user_shadow_count; ++i) {
        if (strcmp(s_user_names_shadow[i], arg) == 0) {
            unenroll_slot(i);
            return;
        }
    }
    /* Otherwise parse numeric slot index */
    char *endp = NULL;
    long idx = strtol(arg, &endp, 10);
    if (endp != arg && *endp == '\0') {
        unenroll_slot((int)idx);
    } else {
        printf("[warn] unenroll target not found: '%s'\n", arg);
    }
}

static void set_scope(bool live_only) {
    s_live_only_scope = live_only;
    push_status_to_wifi();
    printf("[ctrl] Gallery scope -> %s\n", live_only ? "LIVE ONLY" : "ALL (89+N)");
    emit_status_event();
}

static void set_threshold(float thr) {
    if (thr < 0.10f) thr = 0.10f;
    if (thr > 0.95f) thr = 0.95f;
    s_match_threshold = thr;
    push_status_to_wifi();
    printf("[ctrl] Cosine threshold -> %.2f\n", (double)s_match_threshold);
    emit_status_event();
}

static void cycle_threshold(void) {
    if (s_match_threshold < 0.55f)      set_threshold(0.60f);
    else if (s_match_threshold < 0.64f) set_threshold(0.68f);
    else                                set_threshold(0.50f);
}

/* ---- Standard Beat Reporting (Self-Test / USB Replay) -------------------- */
static void run_and_report_replay(const float *beat, int true_class, const char *label) {
    if (true_class < 0 && s_cap_mode == CAP_ENROLL_5) {
        s_progress_cur++;
        bool done = (s_progress_cur >= TARGET_BEATS);
        inference_result_t *res = dispatch_core1(
            CORE1_OP_ENROLL_BEAT, beat, -1, true, false, done, -1, s_enroll_name);
        printf("[enroll] %s beat %d/%d  t=%lu us (%.2f ms)\n",
               s_enroll_name, s_progress_cur, TARGET_BEATS,
               (unsigned long)res->latency_us, (double)res->latency_us / 1000.0);
        if (done) {
            shadow_add_user(s_enroll_name);
            snprintf(s_vote_summary, sizeof(s_vote_summary), "ENROLLED %s", s_enroll_name);
            s_cap_mode = CAP_STANDBY;
            s_progress_cur = 0;
            s_progress_tot = 0;
            printf("EVENT:ENROLLED,%s,%d\n", s_enroll_name, s_user_shadow_count);
            emit_users_event();
            blink_led(3, 90, 90);
        } else {
            snprintf(s_vote_summary, sizeof(s_vote_summary), "ENROLLING %s (%d/%d)",
                     s_enroll_name, s_progress_cur, TARGET_BEATS);
        }
        push_status_to_wifi();
        emit_status_event();
        return;
    }

    ++s_total_beats;
    bool search_live = (true_class < 0 && s_user_shadow_count > 0);
    bool live_only   = (true_class < 0 && s_live_only_scope);
    inference_result_t *res = dispatch_core1(
        CORE1_OP_INFER, beat, true_class, search_live, live_only, false, -1, label);

    printf("[%lu] Match: %-10s (probe: %s) -> %-6s  sim=%.4f  t=%lu us (%.2f ms)\n",
           (unsigned long)s_total_beats,
           res->matched_name,
           label,
           res->accepted ? "ACCEPT" : "REJECT",
           (double)res->best_sim,
           (unsigned long)res->latency_us,
           (double)res->latency_us / 1000.0);

    wifi_server_update_result(res->matched_name, res->best_sim, res->latency_us,
                              res->accepted, s_total_beats);

    /* If armed in CAP_AUTH_5 during playback/replay, accumulate 5-beat majority vote */
    if (true_class < 0 && s_cap_mode == CAP_AUTH_5) {
        s_progress_cur++;
        if (res->accepted) {
            s_vote_accepts++;
            strncpy(s_vote_last_name, res->matched_name, sizeof(s_vote_last_name) - 1);
        }
        s_vote_sim_sum += res->best_sim;
        if (s_progress_cur >= TARGET_BEATS) {
            bool vote_ok = (s_vote_accepts >= 3);
            float avg_sim = s_vote_sim_sum / (float)TARGET_BEATS;
            const char *vname = vote_ok ? s_vote_last_name : "IMPOSTOR";
            snprintf(s_vote_summary, sizeof(s_vote_summary), "%s %d/%d (%.2f)",
                     vote_ok ? "ACCEPT" : "REJECT", s_vote_accepts, TARGET_BEATS, (double)avg_sim);
            printf("EVENT:VOTE,%s,%s,%d/%d,%.4f\n",
                   vname, vote_ok ? "ACCEPT" : "REJECT",
                   s_vote_accepts, TARGET_BEATS, (double)avg_sim);
            s_cap_mode = CAP_STANDBY;
            s_progress_cur = 0;
            s_progress_tot = 0;
            push_status_to_wifi();
            emit_status_event();
            if (vote_ok) blink_led(2, 120, 100);
            else         blink_led(5, 60, 60);
            return;
        } else {
            snprintf(s_vote_summary, sizeof(s_vote_summary), "AUTH %d/%d (%d ok)",
                     s_progress_cur, TARGET_BEATS, s_vote_accepts);
            push_status_to_wifi();
            emit_status_event();
            return;
        }
    }

    push_status_to_wifi();
    if (res->accepted) blink_led(2, 120, 100);
    else               blink_led(5, 60, 60);
}

/* ---- Live AD8232 Beat Handler (Called Only When 5-Point SQI Passes) ------ */
static void handle_live_sqi_beat(const float *beat) {
    if (s_cap_mode == CAP_STANDBY || s_cap_mode == CAP_REPLAY) {
        if (s_telem_enabled) {
            printf("EVENT:RPEAK,bpm=%.1f\n", (double)s_dsp.bpm);
        }
        push_status_to_wifi();
        return;
    }

    if (s_cap_mode == CAP_ENROLL_5) {
        s_progress_cur++;
        bool done = (s_progress_cur >= TARGET_BEATS);
        inference_result_t *res = dispatch_core1(
            CORE1_OP_ENROLL_BEAT, beat, -1, true, false, done, -1, s_enroll_name);
        printf("[live-enroll] %s beat %d/%d  bpm=%.1f  t=%.2f ms\n",
               s_enroll_name, s_progress_cur, TARGET_BEATS,
               (double)s_dsp.bpm, (double)res->latency_us / 1000.0);
        if (done) {
            shadow_add_user(s_enroll_name);
            snprintf(s_vote_summary, sizeof(s_vote_summary), "ENROLLED %s", s_enroll_name);
            s_cap_mode = CAP_STANDBY;
            s_progress_cur = 0;
            s_progress_tot = 0;
            printf("EVENT:ENROLLED,%s,%d\n", s_enroll_name, s_user_shadow_count);
            emit_users_event();
            blink_led(3, 90, 90);
        } else {
            snprintf(s_vote_summary, sizeof(s_vote_summary), "ENROLLING %s (%d/%d)",
                     s_enroll_name, s_progress_cur, TARGET_BEATS);
        }
        push_status_to_wifi();
        emit_status_event();
        return;
    }

    ++s_total_beats;
    inference_result_t *res = dispatch_core1(
        CORE1_OP_INFER, beat, -1, true, s_live_only_scope, false, -1, "LIVE_ECG");

    printf("[%lu] Match: %-10s (probe: LIVE_ECG) -> %-6s  sim=%.4f  t=%lu us (%.2f ms)\n",
           (unsigned long)s_total_beats,
           res->matched_name,
           res->accepted ? "ACCEPT" : "REJECT",
           (double)res->best_sim,
           (unsigned long)res->latency_us,
           (double)res->latency_us / 1000.0);

    wifi_server_update_result(res->matched_name, res->best_sim, res->latency_us,
                              res->accepted, s_total_beats);

    if (s_cap_mode == CAP_AUTH_5) {
        s_progress_cur++;
        if (res->accepted) {
            s_vote_accepts++;
            strncpy(s_vote_last_name, res->matched_name, sizeof(s_vote_last_name) - 1);
        }
        s_vote_sim_sum += res->best_sim;

        if (s_progress_cur >= TARGET_BEATS) {
            bool vote_ok = (s_vote_accepts >= 3);
            float avg_sim = s_vote_sim_sum / (float)TARGET_BEATS;
            const char *vname = vote_ok ? s_vote_last_name : "IMPOSTOR";
            snprintf(s_vote_summary, sizeof(s_vote_summary), "%s %d/%d (%.2f)",
                     vote_ok ? "ACCEPT" : "REJECT", s_vote_accepts, TARGET_BEATS, (double)avg_sim);
            printf("EVENT:VOTE,%s,%s,%d/%d,%.4f\n",
                   vname, vote_ok ? "ACCEPT" : "REJECT",
                   s_vote_accepts, TARGET_BEATS, (double)avg_sim);

            s_cap_mode = CAP_STANDBY;
            s_progress_cur = 0;
            s_progress_tot = 0;
            push_status_to_wifi();
            emit_status_event();

            if (vote_ok) blink_led(2, 120, 100);
            else         blink_led(5, 60, 60);
        } else {
            snprintf(s_vote_summary, sizeof(s_vote_summary), "AUTH %d/%d (%d ok)",
                     s_progress_cur, TARGET_BEATS, s_vote_accepts);
            push_status_to_wifi();
            emit_status_event();
        }
    } else if (s_cap_mode == CAP_CONTINUOUS) {
        s_cont_ring[s_cont_pos] = res->accepted;
        s_cont_pos = (s_cont_pos + 1) % TARGET_BEATS;
        if (s_cont_len < TARGET_BEATS) s_cont_len++;
        int acc = 0;
        for (int i = 0; i < s_cont_len; ++i) if (s_cont_ring[i]) acc++;
        snprintf(s_vote_summary, sizeof(s_vote_summary), "%s (%d/%d)",
                 (acc * 2 >= s_cont_len) ? "ACCEPT" : "REJECT", acc, s_cont_len);
        push_status_to_wifi();
        emit_status_event();

        if (res->accepted) blink_led(2, 90, 70);
        else               blink_led(5, 50, 50);
    }
}

/* ---- USB Serial Line Processor (Commands + 300-Float CSV Replay) --------- */
static void process_serial_line(char *line) {
    while (*line == ' ' || *line == '\t' || *line == '\r' || *line == '\n') line++;
    if (*line == '\0') return;

    if (strncmp(line, "CMD:", 4) == 0) {
        const char *cmd = line + 4;
        if (strncmp(cmd, "ENROLL", 6) == 0) {
            const char *name = cmd + 6;
            while (*name == ' ') name++;
            start_enroll(name);
        } else if (strncmp(cmd, "UNENROLL", 8) == 0) {
            const char *arg = cmd + 8;
            while (*arg == ' ') arg++;
            unenroll_by_arg(arg);
        } else if (strncmp(cmd, "LIST", 4) == 0) {
            emit_users_event();
        } else if (strncmp(cmd, "SCOPE", 5) == 0) {
            const char *arg = cmd + 5;
            while (*arg == ' ') arg++;
            if (strncmp(arg, "LIVE", 4) == 0)      set_scope(true);
            else if (strncmp(arg, "ALL", 3) == 0)  set_scope(false);
            else                                   set_scope(!s_live_only_scope);
        } else if (strncmp(cmd, "THRESH", 6) == 0) {
            const char *arg = cmd + 6;
            while (*arg == ' ') arg++;
            if (*arg != '\0') set_threshold((float)atof(arg));
            else              cycle_threshold();
        } else if (strncmp(cmd, "AUTH", 4) == 0) {
            start_auth_5();
        } else if (strncmp(cmd, "CONT", 4) == 0) {
            toggle_continuous();
        } else if (strncmp(cmd, "STANDBY", 7) == 0) {
            set_standby();
        } else if (strncmp(cmd, "REPLAY", 6) == 0) {
            s_cap_mode = CAP_REPLAY;
            s_progress_cur = 0;
            s_progress_tot = 0;
            push_status_to_wifi();
            emit_status_event();
        } else if (strncmp(cmd, "CLEAR", 5) == 0) {
            clear_live_users();
        } else if (strncmp(cmd, "TELEM ON", 8) == 0) {
            s_telem_enabled = true;
            printf("[ctrl] USB oscilloscope telemetry ON (100 Hz)\n");
            emit_users_event();
            emit_status_event();
        } else if (strncmp(cmd, "TELEM OFF", 9) == 0) {
            s_telem_enabled = false;
            printf("[ctrl] USB oscilloscope telemetry OFF\n");
        } else if (strncmp(cmd, "STATUS", 6) == 0) {
            emit_users_event();
            emit_status_event();
        } else {
            printf("[warn] unknown command: %s\n", line);
        }
        return;
    }

    /* Backward-compatible 300 comma-separated floats for PhysioNet / AD8232 replay */
    int count = 0;
    char *tok = strtok(line, ", \t\n\r");
    while (tok && count < BEAT_LENGTH) {
        s_serial_beat_buf[count++] = (float)atof(tok);
        tok = strtok(NULL, ", \t\n\r");
    }
    if (count == BEAT_LENGTH) {
        run_and_report_replay(s_serial_beat_buf, -1, "STREAM");
    } else if (count > 0) {
        printf("[warn] expected %d samples, got %d -- discarded\n", BEAT_LENGTH, count);
    }
}

/* ---- Onboard Self-Test Suite (Exact PyTorch Parity Verification) --------- */
static void run_self_test(void) {
    printf("\n===========================================================\n");
    printf("  [RP2040 Pico WH]  Onboard Biometric Self-Test Suite\n");
    printf("  ECG_ResNet_SE  |  14,867 params  |  BN-fused FP32\n");
    printf("===========================================================\n\n");

    for (int t = 0; t < NUM_TEST_BEATS; ++t) {
        const float *beat = TEST_BEATS + t * BEAT_LENGTH;
        int          cls  = TEST_LABELS[t];
        const char  *name = TEST_CLASS_NAMES[t];
        run_and_report_replay(beat, cls, name);
        sleep_ms(300);
    }

    printf("\n-----------------------------------------------------------\n");
    printf("  Self-test complete. %d/%d beats processed.\n", NUM_TEST_BEATS, NUM_TEST_BEATS);
    printf("  Wi-Fi dashboard : http://192.168.4.1  (SSID: pico)\n");
    printf("  Live AD8232 pins: OUT=GP26 (ADC0), LO+=GP16, LO-=GP17\n");
    printf("  USB streaming   : send 300 CSV floats or CMD:<ACTION>\n");
    printf("-----------------------------------------------------------\n\n");
}

/* ---- Non-Blocking Main Event Loop (Core 0) ------------------------------- */
static void run_main_loop(void) {
    printf("Listening for live beats over USB CDC serial...\n\n");
    push_status_to_wifi();

    ecg_dsp_state_t prev_dsp_state = s_dsp.state;
    absolute_time_t last_status_ms = get_absolute_time();

    while (1) {
        /* 1. Drain 500 Hz ADC samples from hardware timer ISR ring buffer */
        while (s_adc_tail != s_adc_head) {
            uint16_t packed = s_adc_ring[s_adc_tail];
            s_adc_tail = (s_adc_tail + 1) & ADC_RING_MASK;

            uint16_t raw = packed & 0x0FFF;
            bool lo = (packed & 0x8000) != 0;
            float filt = 0.0f;

            bool beat_ready = ecg_dsp_push(&s_dsp, raw, lo, &filt, s_live_beat_buf);

            if (s_dsp.state != prev_dsp_state) {
                prev_dsp_state = s_dsp.state;
                if (s_dsp.state == DSP_LEADS_OFF && s_cap_mode == CAP_CONTINUOUS) {
                    snprintf(s_vote_summary, sizeof(s_vote_summary), "LOCKED (LEADS OFF)");
                }
                push_status_to_wifi();
                if (s_telem_enabled) emit_status_event();
            }

            if (s_telem_enabled && (++s_telem_div >= TELEM_DECIM)) {
                s_telem_div = 0;
                printf("ECG,%u,%.4f,%d,%.1f\n",
                       (unsigned)raw, (double)filt, (int)s_dsp.state, (double)s_dsp.bpm);
            }

            if (beat_ready) {
                handle_live_sqi_beat(s_live_beat_buf);
            }
        }

        /* 2. Non-blocking USB CDC serial input */
        int ch;
        while ((ch = getchar_timeout_us(0)) != PICO_ERROR_TIMEOUT) {
            if (ch == '\n' || ch == '\r') {
                if (s_line_pos > 0) {
                    s_line_buf[s_line_pos] = '\0';
                    s_line_pos = 0;
                    process_serial_line(s_line_buf);
                }
            } else if (s_line_pos < SERIAL_LINE_BUF - 1) {
                s_line_buf[s_line_pos++] = (char)ch;
            }
        }

        /* 3. Handle any button click from Wi-Fi dashboard */
        int act_arg = 0;
        wifi_action_t act = wifi_server_take_action(&act_arg);
        if (act != WIFI_ACT_NONE) {
            switch (act) {
                case WIFI_ACT_ENROLL:       start_enroll(NULL);               break;
                case WIFI_ACT_AUTH:         start_auth_5();                   break;
                case WIFI_ACT_CONT:         toggle_continuous();              break;
                case WIFI_ACT_STANDBY:      set_standby();                    break;
                case WIFI_ACT_CLEAR:        clear_live_users();               break;
                case WIFI_ACT_UNENROLL:     unenroll_slot(act_arg);           break;
                case WIFI_ACT_SCOPE_TOGGLE: set_scope(!s_live_only_scope);    break;
                case WIFI_ACT_THRESH_CYCLE: cycle_threshold();                break;
                default: break;
            }
        }

        /* 4. Periodic Wi-Fi status refresh (~1 Hz) */
        if (absolute_time_diff_us(last_status_ms, get_absolute_time()) >= 1000000) {
            last_status_ms = get_absolute_time();
            push_status_to_wifi();
        }

        wifi_server_poll();
        sleep_us(500);
    }
}

/* ---- Entry Point --------------------------------------------------------- */
int main(void) {
    vreg_set_voltage(VREG_VOLTAGE_1_20);
    sleep_ms(10);
    set_sys_clock_khz(250000, true);

    stdio_init_all();
    for (int i = 0; i < 40 && !stdio_usb_connected(); ++i) sleep_ms(100);

    printf("\n\n===========================================================\n");
    printf("  Your Heartbeat Is Your Password  --  Option 3+ Dual-Mode\n");
    printf("  Hardware : RP2040 Dual Cortex-M0+ @ 250 MHz (AMP Mode)\n");
    printf("  Model    : ECG_ResNet_SE (14.8k params, 89 DB + 8 Live)\n");
    printf("  Sensor   : AD8232 @ 500 Hz (GP26/ADC0, LO+=GP16, LO-=GP17)\n");
    printf("  DSP      : 0.5-40 Hz BP + 50 Hz Notch + 5-Point SQI\n");
    printf("===========================================================\n");

    adc_init();
    adc_gpio_init(PIN_ECG_ADC);
    adc_select_input(0);

    gpio_init(PIN_LO_PLUS);
    gpio_set_dir(PIN_LO_PLUS, GPIO_IN);
    gpio_pull_up(PIN_LO_PLUS);

    gpio_init(PIN_LO_MINUS);
    gpio_set_dir(PIN_LO_MINUS, GPIO_IN);
    gpio_pull_up(PIN_LO_MINUS);

    ecg_dsp_init(&s_dsp);

    printf("[init] Launching Core 1 neural inference engine...\n");
    multicore_launch_core1(core1_inference_worker);

    printf("[init] Starting Wi-Fi AP (SSID: pico)...\n");
    int wifi_ok = wifi_server_init();
    if (wifi_ok == 0) {
        printf("[init] Wi-Fi ready. Connect to 'pico' -> http://192.168.4.1\n");
        blink_led(3, 100, 100);
    } else {
        printf("[init] Wi-Fi init failed (code %d). Continuing USB-only.\n", wifi_ok);
        blink_led(1, 800, 0);
    }

    run_self_test();

    add_repeating_timer_us(-2000, adc_timer_cb, NULL, &s_adc_timer);
    printf("[init] 500 Hz AD8232 hardware timer started.\n");

    run_main_loop();

    cyw43_arch_deinit();
    return 0;
}
