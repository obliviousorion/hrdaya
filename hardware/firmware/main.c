/**
 * main.c — Raspberry Pi Pico WH Biometric Authentication Firmware
 *
 * Asymmetric Multiprocessing (AMP) & Overclocked Engine:
 *   CORE 0    — I/O, Communications & Presentation (250 MHz)
 *               - USB CDC Serial (/dev/ttyACM0 @ 115200 baud)
 *               - Wi-Fi HTTP dashboard (CYW43439 AP mode: SSID 'pico', http://192.168.4.1)
 *               - Physical LED biometric signaling (2 solid pulses / 5 alert pulses)
 *
 *   CORE 1    — Dedicated Neural Inference Engine (250 MHz)
 *               - ECG_ResNet_SE (14,867 params, BN-fused FP32)
 *               - Full open-set 89-class cosine distance matching
 *               - Inter-core communication via hardware SIO FIFO & static mailbox
 *
 * Hardware: RP2040 Dual Cortex-M0+ @ 250 MHz (1.20V VREG)  |  264 KB SRAM  |  2 MB Flash
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/multicore.h"
#include "hardware/vreg.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "pico/cyw43_arch.h"

#include "model_data.h"
#include "nn_inference.h"
#include "wifi_server.h"

// ---- Configuration ---------------------------------------------------------
#define MATCH_THRESHOLD    0.50f
#define SERIAL_LINE_BUF    4096   /* bytes — fits 300 floats as text */

// ---- Inter-core communication structures -----------------------------------
typedef struct {
    float beat[BEAT_LENGTH];
    int   true_class;
    char  label[32];
} inference_request_t;

typedef struct {
    int      best_id;
    float    best_sim;
    uint32_t latency_us;
    bool     accepted;
    char     matched_name[32];
} inference_result_t;

// ---- Static buffers --------------------------------------------------------
static float serial_beat_buf[BEAT_LENGTH];
static float current_emb[EMBED_DIM];
static uint32_t total_beats = 0;

static inference_request_t s_req_box;
static inference_result_t  s_res_box;

// ---- Helpers ---------------------------------------------------------------
static void blink_led(int times, int on_ms, int off_ms) {
    for (int i = 0; i < times; ++i) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(on_ms);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(off_ms);
    }
}

// ---- Core 1: Dedicated Neural Inference Engine -----------------------------
static void core1_inference_worker(void) {
    while (1) {
        // Wait for Core 0 to dispatch a request mailbox address
        uint32_t req_addr = multicore_fifo_pop_blocking();
        inference_request_t *req = (inference_request_t *)req_addr;

        // High-precision inference timing on Core 1
        uint32_t t_start = time_us_32();
        run_embedding(req->beat, current_emb);
        uint32_t t_elapsed = time_us_32() - t_start;

        // Identification: find best cosine match across enrolled templates
        int best_id = (req->true_class >= 0) ? req->true_class : 0;
        float best_sim = cosine_similarity(current_emb,
                             ENROLLED_TEMPLATES + best_id * EMBED_DIM, EMBED_DIM);

        if (req->true_class < 0) {
            // Open-set: search all 89 enrolled templates
            for (int c = 1; c < NUM_CLASSES; ++c) {
                float s = cosine_similarity(current_emb,
                              ENROLLED_TEMPLATES + c * EMBED_DIM, EMBED_DIM);
                if (s > best_sim) { best_sim = s; best_id = c; }
            }
        }

        s_res_box.best_id = best_id;
        s_res_box.best_sim = best_sim;
        s_res_box.latency_us = t_elapsed;
        s_res_box.accepted = (best_sim >= MATCH_THRESHOLD);

        const char *mname = (best_id >= 0 && best_id < NUM_CLASSES)
                            ? ALL_CLASS_NAMES[best_id]
                            : req->label;
        strncpy(s_res_box.matched_name, mname, sizeof(s_res_box.matched_name) - 1);
        s_res_box.matched_name[sizeof(s_res_box.matched_name) - 1] = '\0';

        // Notify Core 0 that inference is complete
        multicore_fifo_push_blocking((uint32_t)&s_res_box);
    }
}

/**
 * Dispatch inference to Core 1, service Wi-Fi during calculation, update outputs.
 * @param beat        Pointer to BEAT_LENGTH (300) float samples.
 * @param true_class  Enrolled class index to compare against, or -1 for open-set search.
 * @param label       Display name for USB serial output (e.g. "Person_01").
 */
static void run_and_report(const float *beat, int true_class, const char *label) {
    ++total_beats;

    // Populate request mailbox
    memcpy(s_req_box.beat, beat, sizeof(float) * BEAT_LENGTH);
    s_req_box.true_class = true_class;
    strncpy(s_req_box.label, label, sizeof(s_req_box.label) - 1);
    s_req_box.label[sizeof(s_req_box.label) - 1] = '\0';

    // Dispatch request to Core 1 via hardware SIO FIFO
    multicore_fifo_push_blocking((uint32_t)&s_req_box);

    // Non-blocking wait on Core 0: service Wi-Fi AP while Core 1 crunches math
    while (!multicore_fifo_rvalid()) {
        wifi_server_poll();
        tight_loop_contents();
    }

    // Retrieve result pointer from Core 1
    uint32_t res_addr = multicore_fifo_pop_blocking();
    inference_result_t *res = (inference_result_t *)res_addr;

    // --- Primary: USB Serial output ---
    printf("[%lu] Match: %-10s (probe: %s) -> %-6s  sim=%.4f  t=%lu us (%.2f ms)\n",
           total_beats,
           res->matched_name,
           label,
           res->accepted ? "ACCEPT" : "REJECT",
           res->best_sim,
           res->latency_us,
           (float)res->latency_us / 1000.0f);

    // --- Secondary: Wi-Fi dashboard update ---
    wifi_server_update_result(res->matched_name, res->best_sim, res->latency_us, res->accepted, total_beats);

    // --- LED feedback: 2 solid pulses = ACCEPT, 5 rapid alert pulses = REJECT ---
    if (res->accepted) blink_led(2, 120, 100);
    else               blink_led(5, 60, 60);
}

// ---- Onboard self-test suite -----------------------------------------------
static void run_self_test(void) {
    printf("\n===========================================================\n");
    printf("  [RP2040 Pico WH]  Onboard Biometric Self-Test Suite\n");
    printf("  ECG_ResNet_SE  |  14,867 params  |  BN-fused FP32\n");
    printf("===========================================================\n\n");

    for (int t = 0; t < NUM_TEST_BEATS; ++t) {
        const float *beat      = TEST_BEATS + t * BEAT_LENGTH;
        int          cls       = TEST_LABELS[t];
        const char  *name      = TEST_CLASS_NAMES[t];

        run_and_report(beat, cls, name);
        sleep_ms(400);
    }

    printf("\n-----------------------------------------------------------\n");
    printf("  Self-test complete. %d/%d beats processed.\n",
           NUM_TEST_BEATS, NUM_TEST_BEATS);
    printf("  Wi-Fi dashboard: http://192.168.4.1  (SSID: pico)\n");
    printf("  USB streaming: send 300 comma-separated floats + \\n\n");
    printf("-----------------------------------------------------------\n\n");
}

// ---- USB serial streaming mode ---------------------------------------------
static void run_serial_loop(void) {
    printf("Listening for live beats over USB CDC serial...\n\n");

    static char line_buf[SERIAL_LINE_BUF];

    while (1) {
        // Block until a line arrives
        if (!fgets(line_buf, sizeof(line_buf), stdin)) {
            sleep_ms(10);
            wifi_server_poll();
            continue;
        }

        // Parse up to BEAT_LENGTH floats
        int count = 0;
        char *tok = strtok(line_buf, ", \t\n\r");
        while (tok && count < BEAT_LENGTH) {
            serial_beat_buf[count++] = (float)atof(tok);
            tok = strtok(NULL, ", \t\n\r");
        }

        if (count != BEAT_LENGTH) {
            printf("[warn] expected %d samples, got %d — discarded\n",
                   BEAT_LENGTH, count);
            continue;
        }

        // Open-set identification (no claimed identity from host)
        run_and_report(serial_beat_buf, -1, "STREAM");

        wifi_server_poll();
        sleep_ms(5);
    }
}

// ---- Entry point -----------------------------------------------------------
int main(void) {
    // ---- Safe 250 MHz Overclocking & Core Voltage Setup ----
    vreg_set_voltage(VREG_VOLTAGE_1_20);
    sleep_ms(10);
    set_sys_clock_khz(250000, true);

    stdio_init_all();

    // Wait for USB serial terminal to connect (up to 5 s)
    for (int i = 0; i < 50 && !stdio_usb_connected(); ++i) sleep_ms(100);

    printf("\n\n===========================================================\n");
    printf("  Your Heartbeat Is Your Password  --  EEE G513 Project\n");
    printf("  Hardware : RP2040 Dual Cortex-M0+ @ 250 MHz (AMP Mode)\n");
    printf("  Model    : ECG_ResNet_SE  (14.8k params, ~12 KB RAM)\n");
    printf("  Core 0   : Comms (USB CDC), Wi-Fi AP & Web Dashboard\n");
    printf("  Core 1   : Dedicated Neural Inference & Cosine Matching\n");
    printf("===========================================================\n");

    // ---- Launch Core 1 Dedicated Neural Inference Engine ----
    printf("[init] Launching Core 1 neural inference engine...\n");
    multicore_launch_core1(core1_inference_worker);
    printf("[init] Core 1 active on dedicated inference pipeline.\n");

    // ---- Start Wi-Fi secondary channel ----
    printf("[init] Starting Wi-Fi AP (SSID: pico)...\n");
    int wifi_ok = wifi_server_init();
    if (wifi_ok == 0) {
        printf("[init] Wi-Fi ready. Connect to 'pico' -> http://192.168.4.1\n");
        // 3 blinks = Wi-Fi up
        blink_led(3, 100, 100);
    } else {
        printf("[init] Wi-Fi init failed (code %d). Continuing USB-only.\n", wifi_ok);
        blink_led(1, 800, 0);
    }

    // ---- Run onboard self-test ----
    run_self_test();

    // ---- Enter serial streaming loop ----
    run_serial_loop();

    // Never reached
    cyw43_arch_deinit();
    return 0;
}
