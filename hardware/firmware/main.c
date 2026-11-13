/**
 * main.c — Raspberry Pi Pico WH Biometric Authentication Firmware
 *
 * Dual-channel output:
 *   PRIMARY   — USB CDC Serial (/dev/ttyACM0 @ 115200 baud)
 *               Microsecond-accurate inference latency via time_us_32().
 *               Streams 300-sample beats from host laptop; runs onboard self-test.
 *
 *   SECONDARY — Wi-Fi HTTP dashboard (CYW43439 AP mode)
 *               SSID: "pico"  |  Password: "pico"  |  http://192.168.4.1
 *               Same inference result mirrored to a live browser dashboard.
 *               Auto-refreshes every 2 s. Terminal-dark theme.
 *
 * Model: ECG_ResNet_SE  |  14,867 params  |  BN-fused float32 weights
 * Hardware: RP2040 @ 133 MHz  |  264 KB SRAM  |  2 MB Flash
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/gpio.h"
#include "pico/cyw43_arch.h"

#include "model_data.h"
#include "nn_inference.h"
#include "wifi_server.h"

// ---- Configuration ---------------------------------------------------------
#define MATCH_THRESHOLD    0.50f
#define SERIAL_LINE_BUF    4096   /* bytes — fits 300 floats as text */

// ---- Static buffers --------------------------------------------------------
static float serial_beat_buf[BEAT_LENGTH];
static float current_emb[EMBED_DIM];
static uint32_t total_beats = 0;

// ---- Helpers ---------------------------------------------------------------
static void blink_led(int times, int on_ms, int off_ms) {
    for (int i = 0; i < times; ++i) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(on_ms);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(off_ms);
    }
}

/**
 * Run inference on one beat buffer, print USB serial result, update Wi-Fi dashboard.
 * @param beat        Pointer to BEAT_LENGTH (300) float samples.
 * @param true_class  Enrolled class index to compare against, or -1 for open-set search.
 * @param label       Display name for USB serial output (e.g. "Person_01").
 */
static void run_and_report(const float *beat, int true_class, const char *label) {
    ++total_beats;

    // --- Inference ---
    uint32_t t_start = time_us_32();
    run_embedding(beat, current_emb);
    uint32_t t_elapsed = time_us_32() - t_start;

    // --- Identification: best cosine match across all enrolled templates ---
    int   best_id  = (true_class >= 0) ? true_class : 0;
    float best_sim = cosine_similarity(current_emb,
                         ENROLLED_TEMPLATES + best_id * EMBED_DIM, EMBED_DIM);

    if (true_class < 0) {
        // Open-set: search all templates
        for (int c = 1; c < NUM_CLASSES; ++c) {
            float s = cosine_similarity(current_emb,
                          ENROLLED_TEMPLATES + c * EMBED_DIM, EMBED_DIM);
            if (s > best_sim) { best_sim = s; best_id = c; }
        }
    }

    bool accepted = (best_sim >= MATCH_THRESHOLD);
    const char *matched_name = (best_id >= 0 && best_id < NUM_CLASSES)
                               ? ALL_CLASS_NAMES[best_id]
                               : label;

    // --- Primary: USB Serial output ---
    printf("[%lu] Match: %-10s (probe: %s) -> %-6s  sim=%.4f  t=%lu us (%.2f ms)\n",
           total_beats,
           matched_name,
           label,
           accepted ? "ACCEPT" : "REJECT",
           best_sim,
           t_elapsed,
           (float)t_elapsed / 1000.0f);

    // --- Secondary: Wi-Fi dashboard update ---
    wifi_server_update_result(matched_name, best_sim, t_elapsed, accepted, total_beats);

    // --- LED feedback: 2 solid pulses = ACCEPT, 5 rapid alert pulses = REJECT ---
    if (accepted) blink_led(2, 120, 100);
    else          blink_led(5, 60, 60);
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
    stdio_init_all();

    // Wait for USB serial terminal to connect (up to 5 s)
    for (int i = 0; i < 50 && !stdio_usb_connected(); ++i) sleep_ms(100);

    printf("\n\n===========================================================\n");
    printf("  Your Heartbeat Is Your Password  --  EEE G513 Project\n");
    printf("  Hardware : Raspberry Pi Pico WH  (RP2040 @ 133 MHz)\n");
    printf("  Model    : ECG_ResNet_SE  (14.8k params, ~12 KB RAM)\n");
    printf("===========================================================\n");

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
