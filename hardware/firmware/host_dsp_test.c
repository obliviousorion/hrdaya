/**
 * host_dsp_test.c — x86 Host Verification for Real-Time Causal ecg_dsp.h + nn_inference.h
 *
 * Tests:
 *   1. Real continuous 500 Hz PhysioNet ECG recording (Person_01/rec_2.dat) fed sample-by-sample
 *      through causal biquads + R-peak extractor + 5-point SQI -> ECG_ResNet_SE -> Cosine match.
 *   2. Artifact rejection tests:
 *      - Leads-off suppression & 3.0 s settling hold-off
 *      - ADC rail clipping spike rejection
 *      - Pure 50 Hz AC mains sine wave rejection (notch + crest gate)
 *      - Flatline rejection
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "model_data.h"
#include "nn_inference.h"
#include "ecg_dsp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint16_t mv_to_adc(float mv) {
    /* Map typical +/- 2.0 mV ECG to 12-bit ADC around 2048 mid-rail */
    float v = 2048.0f + mv * 700.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 4095.0f) v = 4095.0f;
    return (uint16_t)v;
}

static int test_real_physionet_recording(const char *dat_path, int expected_cls, const char *label) {
    FILE *fp = fopen(dat_path, "rb");
    if (!fp) {
        printf("[FAIL] Could not open %s\n", dat_path);
        return 1;
    }

    int16_t pair[2];
    static ecg_dsp_t dsp;
    ecg_dsp_init(&dsp);

    float beat[BEAT_LENGTH];
    float emb[EMBED_DIM];
    int extracted = 0;
    int accepted = 0;
    float sim_sum = 0.0f;
    float best_sim_seen = -1.0f;

    /* Feed 200 samples of leads-off first to exercise LEADS_OFF -> SETTLING -> READY */
    for (int i = 0; i < 200; ++i) {
        if (ecg_dsp_push(&dsp, 4095, true, NULL, beat)) {
            printf("[FAIL] Beat emitted during LEADS_OFF!\n");
            fclose(fp);
            return 1;
        }
    }

    int sample_n = 0;
    while (fread(pair, sizeof(int16_t), 2, fp) == 2) {
        /* Channel 0 = raw ECG I, gain = 200 ADU/mV */
        float mv = (float)pair[0] / 200.0f;
        /* Add realistic 0.15 mV 50 Hz mains hum + 0.2 Hz baseline wander */
        mv += 0.15f * sinf(2.0f * (float)M_PI * 50.0f * (float)sample_n / 500.0f);
        mv += 0.25f * sinf(2.0f * (float)M_PI * 0.2f * (float)sample_n / 500.0f);
        uint16_t adc = mv_to_adc(mv);

        if (ecg_dsp_push(&dsp, adc, false, NULL, beat)) {
            if (sample_n < ECG_SETTLE_SAMPLES) {
                printf("[FAIL] Beat emitted during 3s settling window (n=%d)!\n", sample_n);
                fclose(fp);
                return 1;
            }
            run_embedding(beat, emb);

            /* Open-set search across all 89 enrolled templates */
            int best_id = 0;
            float best_s = cosine_similarity(emb, ENROLLED_TEMPLATES, EMBED_DIM);
            for (int c = 1; c < NUM_CLASSES; ++c) {
                float s = cosine_similarity(emb, ENROLLED_TEMPLATES + c * EMBED_DIM, EMBED_DIM);
                if (s > best_s) { best_s = s; best_id = c; }
            }
            float true_sim = cosine_similarity(emb, ENROLLED_TEMPLATES + expected_cls * EMBED_DIM, EMBED_DIM);

            extracted++;
            sim_sum += true_sim;
            if (true_sim > best_sim_seen) best_sim_seen = true_sim;
            if (true_sim >= 0.50f && best_id == expected_cls) accepted++;

            printf("  Beat #%02d [%-10s] BPM=%.1f  top=%-10s (%.4f)  true_sim=%.4f  %s\n",
                   extracted, label, dsp.bpm, ALL_CLASS_NAMES[best_id], best_s, true_sim,
                   (best_id == expected_cls && best_s >= 0.50f) ? "ACCEPT [OK]" : "MISS");
        }
        sample_n++;
    }
    fclose(fp);

    float avg_sim = extracted > 0 ? (sim_sum / (float)extracted) : 0.0f;
    printf("-> %s summary: extracted=%d, exact_accept=%d/%d, avg_sim=%.4f, max_sim=%.4f, sqi_rejects=%lu\n\n",
           label, extracted, accepted, extracted, avg_sim, best_sim_seen, (unsigned long)dsp.sqi_rejects);

    if (extracted < 5 || avg_sim < 0.70f || accepted * 2 < extracted) {
        printf("[FAIL] Real ECG streaming DSP accuracy below bar!\n");
        return 1;
    }
    return 0;
}

static int test_artifact_rejections(void) {
    static ecg_dsp_t dsp;
    float beat[BEAT_LENGTH];

    /* 1. Pure 50 Hz mains hum + flatline */
    ecg_dsp_init(&dsp);
    int false_beats = 0;
    for (int i = 0; i < 4000; ++i) {
        float hum = 0.5f * sinf(2.0f * (float)M_PI * 50.0f * (float)i / 500.0f);
        if (ecg_dsp_push(&dsp, mv_to_adc(hum), false, NULL, beat)) false_beats++;
    }
    printf("[SQI Test 1] 8s pure 50 Hz hum -> emitted beats = %d (expected 0)\n", false_beats);
    if (false_beats != 0) return 1;

    /* 2. Flatline */
    ecg_dsp_init(&dsp);
    false_beats = 0;
    for (int i = 0; i < 4000; ++i) {
        if (ecg_dsp_push(&dsp, 2048, false, NULL, beat)) false_beats++;
    }
    printf("[SQI Test 2] 8s flatline       -> emitted beats = %d (expected 0)\n", false_beats);
    if (false_beats != 0) return 1;

    /* 3. ADC Rail Clipping spikes (0 / 4095) */
    ecg_dsp_init(&dsp);
    false_beats = 0;
    for (int i = 0; i < 4000; ++i) {
        uint16_t adc = (i % 400 == 0) ? 4095 : ((i % 400 == 1) ? 0 : 2048);
        if (ecg_dsp_push(&dsp, adc, false, NULL, beat)) false_beats++;
    }
    printf("[SQI Test 3] 8s rail clipping  -> emitted beats = %d (expected 0)\n", false_beats);
    if (false_beats != 0) return 1;

    return 0;
}

int main(void) {
    printf("============================================================\n");
    printf("  Host C Streaming DSP + 5-Point SQI + ECG_ResNet_SE Test   \n");
    printf("============================================================\n\n");

    int rc = 0;
    rc |= test_real_physionet_recording("data/ecg-id-database-1.0.0/Person_01/rec_2.dat", 0, "Person_01");
    rc |= test_real_physionet_recording("data/ecg-id-database-1.0.0/Person_03/rec_2.dat", 2, "Person_03");
    rc |= test_artifact_rejections();

    if (rc == 0) {
        printf("\nALL HOST DSP + SQI + INFERENCE TESTS PASSED!\n");
    } else {
        printf("\nHOST DSP TESTS FAILED!\n");
    }
    return rc;
}
