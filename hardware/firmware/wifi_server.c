/**
 * wifi_server.c — CYW43439 Wi-Fi AP + minimalist HTTP dashboard & action server.
 *
 * Routes:
 *   GET /          -> Dark-terminal dashboard (auto-refreshes every 2s)
 *   GET /enroll    -> Arm 5-beat zero-shot live enrollment (303 -> /)
 *   GET /auth      -> Arm 5-beat live authentication (303 -> /)
 *   GET /cont      -> Toggle continuous live authentication (303 -> /)
 *   GET /standby   -> Return to live standby/preview (303 -> /)
 *   GET /clear     -> Clear live enrolled SRAM templates (303 -> /)
 *
 * 100% STATIC BUFFERS (zero large stack allocations, zero malloc).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "lwip/tcp.h"
#include "lwip/pbuf.h"
#include "lwip/netif.h"
#include "lwip/ip4_addr.h"

#include "dhcpserver.h"
#include "wifi_server.h"

#ifndef WIFI_SSID
#define WIFI_SSID     "pico"
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "picosama"
#endif

#define AP_IP_ADDR    "192.168.4.1"
#define AP_MASK       "255.255.255.0"
#define HTTP_PORT     80
#define HISTORY_LEN   5

typedef struct {
    char     person_name[16];
    float    similarity;
    uint32_t latency_us;
    bool     accepted;
    uint32_t beat_count;
    bool     has_result;
} infer_result_t;

typedef struct {
    char     person_name[16];
    float    similarity;
    uint32_t latency_us;
    bool     accepted;
    uint32_t beat_count;
} history_entry_t;

typedef struct {
    char  mode_str[20];
    char  sensor_str[20];
    float bpm;
    int   live_users;
    int   progress_cur;
    int   progress_tot;
    char  vote_str[40];
} system_status_t;

static volatile infer_result_t g_result = {
    .person_name = "---",
    .similarity  = 0.0f,
    .latency_us  = 0,
    .accepted    = false,
    .beat_count  = 0,
    .has_result  = false,
};

static volatile system_status_t g_status = {
    .mode_str     = "STANDBY",
    .sensor_str   = "LEADS OFF",
    .bpm          = 0.0f,
    .live_users   = 0,
    .progress_cur = 0,
    .progress_tot = 0,
    .vote_str     = "---",
};

static volatile history_entry_t g_history[HISTORY_LEN];
static volatile int g_history_count = 0;
static volatile wifi_action_t g_pending_action = WIFI_ACT_NONE;

static dhcp_server_t s_dhcp;
static struct tcp_pcb *s_server_pcb = NULL;

/* Static BSS buffers (prevents stack overflow on 2 KB Cortex-M0+ stack) */
static char s_html_buf[5000];
static char s_rows_buf[1200];
static char s_resp_buf[5600];

static const char HTTP_REDIRECT_ROOT[] =
    "HTTP/1.1 303 See Other\r\n"
    "Location: /\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "Cache-Control: no-cache\r\n"
    "\r\n";

static const char HTTP_NO_CONTENT[] =
    "HTTP/1.1 204 No Content\r\n"
    "Connection: close\r\n"
    "\r\n";

static int build_http_response(void) {
    cyw43_arch_lwip_begin();
    infer_result_t r = { 0 };
    memcpy(&r, (const void *)&g_result, sizeof(r));

    system_status_t st = { 0 };
    memcpy(&st, (const void *)&g_status, sizeof(st));

    history_entry_t h[HISTORY_LEN];
    int h_count = g_history_count;
    for (int i = 0; i < h_count; ++i) {
        memcpy(&h[i], (const void *)&g_history[i], sizeof(history_entry_t));
    }
    cyw43_arch_lwip_end();

    float latency_ms = (float)r.latency_us / 1000.0f;
    const char *decision_str  = r.has_result ? (r.accepted ? "ACCEPT" : "REJECT") : "STANDBY";
    const char *decision_col  = r.has_result ? (r.accepted ? "#5af78e" : "#f97583") : "#888";
    const char *sensor_col    = (strcmp(st.sensor_str, "SIGNAL READY") == 0) ? "#5af78e"
                              : (strcmp(st.sensor_str, "SETTLING") == 0)     ? "#f1fa8c"
                                                                              : "#f97583";

    char mode_disp[36];
    if (st.progress_tot > 0) {
        snprintf(mode_disp, sizeof(mode_disp), "%s (%d/%d)", st.mode_str, st.progress_cur, st.progress_tot);
    } else {
        snprintf(mode_disp, sizeof(mode_disp), "%s", st.mode_str);
    }

    char sensor_disp[36];
    if (st.bpm >= 30.0f && st.bpm <= 220.0f) {
        snprintf(sensor_disp, sizeof(sensor_disp), "%s // %.0f BPM", st.sensor_str, (double)st.bpm);
    } else {
        snprintf(sensor_disp, sizeof(sensor_disp), "%s", st.sensor_str);
    }

    s_rows_buf[0] = '\0';
    int offset = 0;
    for (int i = 0; i < h_count && offset < (int)sizeof(s_rows_buf) - 180; ++i) {
        float h_lat_ms = (float)h[i].latency_us / 1000.0f;
        const char *h_dec = h[i].accepted ? "ACCEPT" : "REJECT";
        const char *h_col = h[i].accepted ? "#5af78e" : "#f97583";
        offset += snprintf(s_rows_buf + offset, sizeof(s_rows_buf) - offset,
            "<tr><td>#%02lu</td><td>%s</td><td>%.4f</td><td>%.0fms</td>"
            "<td style=\"color:%s;font-weight:bold\">[%s]</td></tr>",
            (unsigned long)h[i].beat_count, h[i].person_name,
            (double)h[i].similarity, (double)h_lat_ms, h_col, h_dec);
    }
    if (h_count == 0) {
        snprintf(s_rows_buf, sizeof(s_rows_buf),
            "<tr><td colspan=\"5\" style=\"color:#444;text-align:center;padding:6px\">AWAITING BEATS...</td></tr>");
    }

    snprintf(s_html_buf, sizeof(s_html_buf),
        "<!DOCTYPE html><html lang=\"en\"><head>"
        "<meta charset=\"UTF-8\">"
        "<meta http-equiv=\"refresh\" content=\"2\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>ECG Auth</title>"
        "<style>"
        "*{margin:0;padding:0;box-sizing:border-box}"
        "body{background:#0a0a0a;color:#c8c8c8;font-family:'Courier New',monospace;"
        "display:flex;align-items:center;justify-content:center;min-height:100vh;padding:10px}"
        ".box{width:100%%;max-width:420px;border:1px solid #2a2a2a;background:#111}"
        ".hdr{background:#161616;border-bottom:1px solid #2a2a2a;padding:8px 12px;"
        "font-size:10px;color:#777;letter-spacing:2px;display:flex;justify-content:space-between}"
        ".body{padding:12px}"
        ".row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1a1a1a}"
        ".lbl{color:#666;font-size:10px;letter-spacing:1px}"
        ".val{color:#ddd;font-size:11px}"
        ".dec{font-size:13px;font-weight:bold;letter-spacing:2px;color:%s}"
        ".btns{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0 6px}"
        ".btn{display:block;text-align:center;padding:7px 4px;font-size:10px;letter-spacing:1px;"
        "color:#ddd;background:#181818;border:1px solid #333;text-decoration:none}"
        ".btn:active{background:#2a2a2a}"
        ".b1{border-color:#5af78e;color:#5af78e}"
        ".b2{border-color:#8be9fd;color:#8be9fd}"
        ".b3{border-color:#f97583;color:#f97583}"
        ".sub{margin-top:8px;padding:4px 0 2px;font-size:9px;color:#555;letter-spacing:1px;border-top:1px solid #1f1f1f}"
        ".htbl{width:100%%;border-collapse:collapse;font-size:10px;margin-top:2px}"
        ".htbl th{text-align:left;color:#555;font-size:8px;padding:3px 2px;border-bottom:1px solid #1a1a1a}"
        ".htbl td{padding:3px 2px;color:#aaa;border-bottom:1px solid #141414}"
        ".ftr{padding:6px 12px;border-top:1px solid #161616;font-size:9px;color:#3a3a3a;text-align:right}"
        "</style></head><body><div class=\"box\">"
        "<div class=\"hdr\"><span>ECG-AUTH // RP2040</span><span>DB: %d LIVE + 89</span></div>"
        "<div class=\"body\">"
        "<div class=\"row\"><span class=\"lbl\">SENSOR</span><span class=\"val\" style=\"color:%s\">%s</span></div>"
        "<div class=\"row\"><span class=\"lbl\">MODE</span><span class=\"val\">%s</span></div>"
        "<div class=\"row\"><span class=\"lbl\">IDENTITY</span><span class=\"val\">%s</span></div>"
        "<div class=\"row\"><span class=\"lbl\">SIMILARITY</span><span class=\"val\">%.4f / 0.5000</span></div>"
        "<div class=\"row\"><span class=\"lbl\">5-BEAT VOTE</span><span class=\"val\">%s</span></div>"
        "<div class=\"row\"><span class=\"lbl\">LATENCY</span><span class=\"val\">%.1f ms (#%lu)</span></div>"
        "<div class=\"row\" style=\"border:none;padding-top:6px\"><span class=\"lbl\">DECISION</span><span class=\"dec\">[ %s ]</span></div>"
        "<div class=\"btns\">"
        "<a class=\"btn b1\" href=\"/enroll\">[ ENROLL 5B ]</a>"
        "<a class=\"btn b2\" href=\"/auth\">[ AUTH 5B ]</a>"
        "<a class=\"btn\" href=\"/cont\">[ CONTINUOUS ]</a>"
        "<a class=\"btn\" href=\"/standby\">[ STANDBY ]</a>"
        "</div>"
        "<a class=\"btn b3\" style=\"margin-bottom:4px\" href=\"/clear\">[ CLEAR LIVE USERS ]</a>"
        "<div class=\"sub\">[ RECENT INFERENCE LOG ]</div>"
        "<table class=\"htbl\"><thead><tr><th>BEAT</th><th>MATCH</th><th>SIM</th><th>TIME</th><th>STATUS</th></tr></thead>"
        "<tbody>%s</tbody></table>"
        "</div><div class=\"ftr\">refresh 2s // 192.168.4.1 // AD8232+REPLAY</div>"
        "</div></body></html>",
        decision_col,
        st.live_users,
        sensor_col, sensor_disp,
        mode_disp,
        r.person_name,
        (double)r.similarity,
        st.vote_str,
        (double)latency_ms, (unsigned long)r.beat_count,
        decision_str,
        s_rows_buf
    );

    int hlen = (int)strlen(s_html_buf);
    int total_len = snprintf(s_resp_buf, sizeof(s_resp_buf),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n"
        "\r\n"
        "%s",
        hlen, s_html_buf);
    if (total_len >= (int)sizeof(s_resp_buf)) {
        total_len = (int)sizeof(s_resp_buf) - 1;
    }
    return total_len;
}

static err_t tcp_sent_cb(void *arg, struct tcp_pcb *tpcb, u16_t len) {
    (void)arg;
    (void)len;
    if (tpcb && tpcb->snd_queuelen == 0) {
        tcp_arg(tpcb, NULL);
        tcp_sent(tpcb, NULL);
        tcp_recv(tpcb, NULL);
        tcp_err(tpcb, NULL);
        tcp_close(tpcb);
    }
    return ERR_OK;
}

static void tcp_err_cb(void *arg, err_t err) {
    (void)arg;
    (void)err;
}

static void send_static_response(struct tcp_pcb *tpcb, const char *buf, int len) {
    tcp_sent(tpcb, tcp_sent_cb);
    tcp_err(tpcb, tcp_err_cb);
    err_t werr = tcp_write(tpcb, buf, (u16_t)len, 0);
    if (werr == ERR_OK) {
        tcp_output(tpcb);
    } else {
        tcp_close(tpcb);
    }
}

static err_t tcp_recv_cb(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err) {
    (void)arg;
    (void)err;
    if (!p) {
        tcp_close(tpcb);
        return ERR_OK;
    }

    char req[64];
    u16_t copy_len = p->tot_len < (sizeof(req) - 1) ? p->tot_len : (sizeof(req) - 1);
    pbuf_copy_partial(p, req, copy_len, 0);
    req[copy_len] = '\0';

    tcp_recved(tpcb, p->tot_len);
    pbuf_free(p);

    if (strncmp(req, "GET /favicon", 12) == 0) {
        send_static_response(tpcb, HTTP_NO_CONTENT, (int)sizeof(HTTP_NO_CONTENT) - 1);
        return ERR_OK;
    }

    wifi_action_t act = WIFI_ACT_NONE;
    if (strncmp(req, "GET /enroll", 11) == 0)       act = WIFI_ACT_ENROLL;
    else if (strncmp(req, "GET /auth", 9) == 0)     act = WIFI_ACT_AUTH;
    else if (strncmp(req, "GET /cont", 9) == 0)     act = WIFI_ACT_CONT;
    else if (strncmp(req, "GET /standby", 12) == 0) act = WIFI_ACT_STANDBY;
    else if (strncmp(req, "GET /clear", 10) == 0)   act = WIFI_ACT_CLEAR;

    if (act != WIFI_ACT_NONE) {
        g_pending_action = act;
        send_static_response(tpcb, HTTP_REDIRECT_ROOT, (int)sizeof(HTTP_REDIRECT_ROOT) - 1);
        return ERR_OK;
    }

    int resp_len = build_http_response();
    if (resp_len > 0 && resp_len < (int)sizeof(s_resp_buf)) {
        send_static_response(tpcb, s_resp_buf, resp_len);
    } else {
        tcp_close(tpcb);
    }
    return ERR_OK;
}

static err_t tcp_accept_cb(void *arg, struct tcp_pcb *new_pcb, err_t err) {
    (void)arg;
    if (err != ERR_OK || !new_pcb) return ERR_VAL;
    tcp_recv(new_pcb, tcp_recv_cb);
    return ERR_OK;
}

int wifi_server_init(void) {
    if (cyw43_arch_init()) {
        printf("[wifi] cyw43_arch_init failed\n");
        return -1;
    }
    cyw43_arch_enable_ap_mode(WIFI_SSID, WIFI_PASSWORD, CYW43_AUTH_WPA2_AES_PSK);

    ip4_addr_t gw, mask;
    if (!ip4addr_aton(AP_IP_ADDR, &gw) || !ip4addr_aton(AP_MASK, &mask)) {
        return -2;
    }

    struct netif *ap_netif = &cyw43_state.netif[CYW43_ITF_AP];
    netif_set_ipaddr(ap_netif, &gw);
    netif_set_netmask(ap_netif, &mask);
    netif_set_gw(ap_netif, &gw);

    dhcp_server_init(&s_dhcp, &gw, &mask);

    cyw43_arch_lwip_begin();
    s_server_pcb = tcp_new_ip_type(IPADDR_TYPE_V4);
    if (!s_server_pcb) {
        cyw43_arch_lwip_end();
        return -3;
    }
    if (tcp_bind(s_server_pcb, IP4_ADDR_ANY, HTTP_PORT) != ERR_OK) {
        cyw43_arch_lwip_end();
        return -4;
    }
    s_server_pcb = tcp_listen_with_backlog(s_server_pcb, 2);
    tcp_accept(s_server_pcb, tcp_accept_cb);
    cyw43_arch_lwip_end();

    printf("[wifi] AP ready -- SSID: %s  IP: %s  Pass: %s\n",
           WIFI_SSID, AP_IP_ADDR, WIFI_PASSWORD);
    return 0;
}

void wifi_server_update_result(const char *person_name,
                               float       similarity,
                               uint32_t    latency_us,
                               bool        accepted,
                               uint32_t    beat_count) {
    cyw43_arch_lwip_begin();
    strncpy((char *)g_result.person_name, person_name, sizeof(g_result.person_name) - 1);
    g_result.person_name[sizeof(g_result.person_name) - 1] = '\0';
    g_result.similarity  = similarity;
    g_result.latency_us  = latency_us;
    g_result.accepted    = accepted;
    g_result.beat_count  = beat_count;
    g_result.has_result  = true;

    for (int i = HISTORY_LEN - 1; i > 0; --i) {
        memcpy((void *)&g_history[i], (void *)&g_history[i - 1], sizeof(history_entry_t));
    }
    strncpy((char *)g_history[0].person_name, person_name, sizeof(g_history[0].person_name) - 1);
    g_history[0].person_name[sizeof(g_history[0].person_name) - 1] = '\0';
    g_history[0].similarity = similarity;
    g_history[0].latency_us = latency_us;
    g_history[0].accepted   = accepted;
    g_history[0].beat_count = beat_count;
    if (g_history_count < HISTORY_LEN) g_history_count++;
    cyw43_arch_lwip_end();
}

void wifi_server_update_status(const char *mode_str,
                               const char *sensor_str,
                               float       bpm,
                               int         live_users,
                               int         progress_cur,
                               int         progress_tot,
                               const char *vote_str) {
    cyw43_arch_lwip_begin();
    if (mode_str) {
        strncpy((char *)g_status.mode_str, mode_str, sizeof(g_status.mode_str) - 1);
        g_status.mode_str[sizeof(g_status.mode_str) - 1] = '\0';
    }
    if (sensor_str) {
        strncpy((char *)g_status.sensor_str, sensor_str, sizeof(g_status.sensor_str) - 1);
        g_status.sensor_str[sizeof(g_status.sensor_str) - 1] = '\0';
    }
    g_status.bpm          = bpm;
    g_status.live_users   = live_users;
    g_status.progress_cur = progress_cur;
    g_status.progress_tot = progress_tot;
    if (vote_str) {
        strncpy((char *)g_status.vote_str, vote_str, sizeof(g_status.vote_str) - 1);
        g_status.vote_str[sizeof(g_status.vote_str) - 1] = '\0';
    }
    cyw43_arch_lwip_end();
}

wifi_action_t wifi_server_take_action(void) {
    cyw43_arch_lwip_begin();
    wifi_action_t act = g_pending_action;
    g_pending_action = WIFI_ACT_NONE;
    cyw43_arch_lwip_end();
    return act;
}

void wifi_server_poll(void) {
    cyw43_arch_poll();
}
