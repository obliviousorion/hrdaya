/**
 * wifi_server.h — Wi-Fi HTTP dashboard interface (secondary output channel).
 *
 * Starts the Pico WH's CYW43439 chip as a Wi-Fi Access Point named "pico"
 * and serves a minimal terminal-themed live dashboard on http://192.168.4.1
 *
 * Usage:
 *   1. Call wifi_server_init() once at startup.
 *   2. After every inference, call wifi_server_update_result() with the result.
 *   3. Any HTTP client on the "pico" AP will see the latest result on refresh.
 */
#ifndef WIFI_SERVER_H
#define WIFI_SERVER_H

#include <stdbool.h>
#include <stdint.h>

/**
 * Initialise CYW43439 as AP, start DHCP server, open TCP listener on port 80.
 * Blocks for up to 10 s waiting for the Wi-Fi chip to initialise.
 * Returns 0 on success, non-zero on failure.
 */
int wifi_server_init(void);

/**
 * Update the globally cached inference result displayed on the dashboard.
 * Thread-safe: may be called from main loop while lwIP handles HTTP in IRQ.
 *
 * @param person_name   Null-terminated identity string, e.g. "Person_01"
 * @param similarity    Cosine similarity score [0.0, 1.0]
 * @param latency_us    Inference time in microseconds (from time_us_32)
 * @param accepted      true = ACCEPT, false = REJECT
 * @param beat_count    Total beats processed since boot
 */
void wifi_server_update_result(const char *person_name,
                               float       similarity,
                               uint32_t    latency_us,
                               bool        accepted,
                               uint32_t    beat_count);

/**
 * Must be called regularly (e.g. every main-loop iteration) to service
 * the lwIP TCP/IP stack when using poll/background mode.
 * With threadsafe_background this is a no-op but kept for portability.
 */
void wifi_server_poll(void);

#endif /* WIFI_SERVER_H */
