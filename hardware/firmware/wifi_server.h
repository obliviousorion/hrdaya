/**
 * wifi_server.h — Wi-Fi HTTP dashboard & control interface (CYW43439 Soft-AP).
 *
 * Serves a minimalist terminal-dark dashboard on http://192.168.4.1 (SSID: "pico")
 * with live sensor status, 5-beat enrollment/auth triggers, and rolling beat log.
 */
#ifndef WIFI_SERVER_H
#define WIFI_SERVER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    WIFI_ACT_NONE    = 0,
    WIFI_ACT_ENROLL  = 1,
    WIFI_ACT_AUTH    = 2,
    WIFI_ACT_CONT    = 3,
    WIFI_ACT_STANDBY = 4,
    WIFI_ACT_CLEAR   = 5
} wifi_action_t;

int wifi_server_init(void);

void wifi_server_update_result(const char *person_name,
                               float       similarity,
                               uint32_t    latency_us,
                               bool        accepted,
                               uint32_t    beat_count);

void wifi_server_update_status(const char *mode_str,
                               const char *sensor_str,
                               float       bpm,
                               int         live_users,
                               int         progress_cur,
                               int         progress_tot,
                               const char *vote_str);

wifi_action_t wifi_server_take_action(void);

void wifi_server_poll(void);

#endif /* WIFI_SERVER_H */
