/**
 * wifi_server.c — CYW43439 Wi-Fi AP + minimal HTTP dashboard server.
 *
 * Architecture:
 *   - CYW43439 chip initialised in Access Point (AP) mode.
 *   - SSID: "pico"  |  Password: "picosama"  |  IP: 192.168.4.1
 *   - Minimal DHCP server assigns 192.168.4.2+ to connecting clients.
 *   - Raw lwIP TCP server on port 80 serves a live HTML dashboard.
 *   - Dashboard auto-refreshes every 2 s via HTTP meta-refresh.
 *   - Design: dark terminal theme, monospace, centered, no emojis, no gradients.
 *   - Rolling history log of the last 5 inferences.
 *   - 100% STATIC BUFFERS (zero stack allocation, zero dynamic malloc).
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "lwip/tcp.h"
#include "lwip/pbuf.h"
#include "lwip/netif.h"
#include "lwip/ip4_addr.h"

#include "dhcpserver.h"
#include "wifi_server.h"

// ---- Configuration ---------------------------------------------------------
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

// ---- Shared result & history state -----------------------------------------
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

static volatile infer_result_t g_result = {
    .person_name = "---",
    .similarity  = 0.0f,
    .latency_us  = 0,
    .accepted    = false,
    .beat_count  = 0,
    .has_result  = false,
};

static volatile history_entry_t g_history[HISTORY_LEN];
static volatile int g_history_count = 0;

static dhcp_server_t s_dhcp;
static struct tcp_pcb *s_server_pcb = NULL;

// Static response buffers in BSS to prevent stack overflow on Cortex-M0+
static char s_html_buf[4096];
static char s_rows_buf[1200];
static char s_resp_buf[4600];

// ---- HTML dashboard builder ------------------------------------------------
static int build_http_response(void) {
    // Snapshot shared state
    cyw43_arch_lwip_begin();
    infer_result_t r = (infer_result_t){ 0 };
    memcpy((void*)&r, (void*)&g_result, sizeof(infer_result_t));

    history_entry_t h[HISTORY_LEN];
    int h_count = g_history_count;
    for (int i = 0; i < h_count; ++i) {
        memcpy(&h[i], (void*)&g_history[i], sizeof(history_entry_t));
    }
    cyw43_arch_lwip_end();

    float latency_ms = (float)r.latency_us / 1000.0f;
    const char *decision_str  = r.accepted ? "ACCEPT" : "REJECT";
    const char *decision_col  = r.accepted ? "#5af78e" : "#f97583";
    const char *status_prefix = r.has_result ? "" : "-- ";

    // Build rolling history rows into static buffer
    s_rows_buf[0] = '\0';
    int offset = 0;
    for (int i = 0; i < h_count && offset < 1000; ++i) {
        float h_lat_ms = (float)h[i].latency_us / 1000.0f;
        const char *h_dec = h[i].accepted ? "ACCEPT" : "REJECT";
        const char *h_col = h[i].accepted ? "#5af78e" : "#f97583";
        offset += snprintf(s_rows_buf + offset, sizeof(s_rows_buf) - offset,
            "<tr>"
              "<td>#%02lu</td>"
              "<td>%s</td>"
              "<td>%.4f</td>"
              "<td>%.1fms</td>"
              "<td style=\"color:%s;font-weight:bold\">[%s]</td>"
            "</tr>",
            (unsigned long)h[i].beat_count,
            h[i].person_name,
            (double)h[i].similarity,
            (double)h_lat_ms,
            h_col,
            h_dec
        );
    }
    if (h_count == 0) {
        snprintf(s_rows_buf, sizeof(s_rows_buf),
            "<tr><td colspan=\"5\" style=\"color:#444;text-align:center;padding:6px\">AWAITING BEATS...</td></tr>");
    }

    snprintf(s_html_buf, sizeof(s_html_buf),
        "<!DOCTYPE html>"
        "<html lang=\"en\">"
        "<head>"
          "<meta charset=\"UTF-8\">"
          "<meta http-equiv=\"refresh\" content=\"2\">"
          "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
          "<title>ECG Auth</title>"
          "<style>"
            "*{margin:0;padding:0;box-sizing:border-box}"
            "body{background:#0a0a0a;color:#c8c8c8;"
                 "font-family:'Courier New',Courier,monospace;"
                 "display:flex;align-items:center;justify-content:center;"
                 "min-height:100vh;padding:12px}"
            ".box{width:100%%;max-width:400px;border:1px solid #2a2a2a;background:#111}"
            ".hdr{background:#161616;border-bottom:1px solid #2a2a2a;"
                 "padding:8px 14px;font-size:10px;color:#555;"
                 "letter-spacing:2px;text-transform:uppercase}"
            ".body{padding:14px}"
            ".row{display:flex;justify-content:space-between;"
                 "padding:5px 0;border-bottom:1px solid #1a1a1a}"
            ".lbl{color:#555;font-size:11px;letter-spacing:1px}"
            ".val{color:#ddd;font-size:11px}"
            ".dec{font-size:13px;font-weight:bold;letter-spacing:2px;color:%s}"
            ".subhdr{margin-top:12px;padding:5px 0 3px;font-size:9px;color:#444;"
                    "letter-spacing:1px;border-top:1px solid #1f1f1f}"
            ".htbl{width:100%%;border-collapse:collapse;font-size:10px;margin-top:2px}"
            ".htbl th{text-align:left;color:#444;font-size:8px;padding:3px 2px;"
                     "letter-spacing:1px;border-bottom:1px solid #1a1a1a}"
            ".htbl td{padding:3px 2px;color:#aaa;border-bottom:1px solid #141414}"
            ".ftr{padding:6px 14px;border-top:1px solid #161616;"
                 "font-size:9px;color:#2e2e2e;text-align:right}"
          "</style>"
        "</head>"
        "<body>"
          "<div class=\"box\">"
            "<div class=\"hdr\">ECG-BIOMETRIC // RP2040</div>"
            "<div class=\"body\">"
              "<div class=\"row\">"
                "<span class=\"lbl\">IDENTITY</span>"
                "<span class=\"val\">%s%s</span>"
              "</div>"
              "<div class=\"row\">"
                "<span class=\"lbl\">SIMILARITY</span>"
                "<span class=\"val\">%.4f / 0.5000</span>"
              "</div>"
              "<div class=\"row\">"
                "<span class=\"lbl\">LATENCY</span>"
                "<span class=\"val\">%.2f ms</span>"
              "</div>"
              "<div class=\"row\">"
                "<span class=\"lbl\">BEATS</span>"
                "<span class=\"val\">%lu</span>"
              "</div>"
              "<div class=\"row\" style=\"border:none;padding-top:8px\">"
                "<span class=\"lbl\">DECISION</span>"
                "<span class=\"dec\">[ %s ]</span>"
              "</div>"
              "<div class=\"subhdr\">[ RECENT INFERENCE LOG ]</div>"
              "<table class=\"htbl\">"
                "<thead>"
                  "<tr>"
                    "<th>BEAT</th>"
                    "<th>MATCH</th>"
                    "<th>SIM</th>"
                    "<th>TIME</th>"
                    "<th>STATUS</th>"
                  "</tr>"
                "</thead>"
                "<tbody>"
                  "%s"
                "</tbody>"
              "</table>"
            "</div>"
            "<div class=\"ftr\">refresh 2s // 192.168.4.1 // ECG-G513</div>"
          "</div>"
        "</body>"
        "</html>",
        decision_col,
        status_prefix, r.person_name,
        (double)r.similarity,
        (double)latency_ms,
        (unsigned long)r.beat_count,
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
        hlen, s_html_buf
    );
    if (total_len >= (int)sizeof(s_resp_buf)) {
        total_len = (int)sizeof(s_resp_buf) - 1;
    }
    return total_len;
}

// ---- lwIP TCP callbacks ----------------------------------------------------
static err_t tcp_sent_cb(void *arg, struct tcp_pcb *tpcb, u16_t len) {
    (void)arg;
    (void)len;
    tcp_arg(tpcb, NULL);
    tcp_sent(tpcb, NULL);
    tcp_recv(tpcb, NULL);
    tcp_err(tpcb, NULL);
    tcp_close(tpcb);
    return ERR_OK;
}

static void tcp_err_cb(void *arg, err_t err) {
    (void)arg;
    (void)err;
}

static err_t tcp_recv_cb(void *arg, struct tcp_pcb *tpcb,
                          struct pbuf *p, err_t err) {
    (void)arg;
    (void)err;
    if (!p) {
        tcp_close(tpcb);
        return ERR_OK;
    }

    // Acknowledge received bytes
    tcp_recved(tpcb, p->tot_len);
    pbuf_free(p);

    // Build HTTP response in static buffer
    int resp_len = build_http_response();
    if (resp_len > 0 && resp_len < (int)sizeof(s_resp_buf)) {
        tcp_sent(tpcb, tcp_sent_cb);
        tcp_err(tpcb, tcp_err_cb);
        tcp_write(tpcb, s_resp_buf, (u16_t)resp_len, TCP_WRITE_FLAG_COPY);
        tcp_output(tpcb);
    } else {
        tcp_close(tpcb);
    }
    return ERR_OK;
}

static err_t tcp_accept_cb(void *arg, struct tcp_pcb *new_pcb, err_t err) {
    if (err != ERR_OK || !new_pcb) return ERR_VAL;
    tcp_recv(new_pcb, tcp_recv_cb);
    return ERR_OK;
}

// ---- Public API ------------------------------------------------------------
int wifi_server_init(void) {
    if (cyw43_arch_init()) {
        printf("[wifi] cyw43_arch_init failed\n");
        return -1;
    }

    // Start Access Point
    cyw43_arch_enable_ap_mode(WIFI_SSID, WIFI_PASSWORD, CYW43_AUTH_WPA2_AES_PSK);

    ip4_addr_t gw, mask;
    if (!ip4addr_aton(AP_IP_ADDR, &gw) || !ip4addr_aton(AP_MASK, &mask)) {
        printf("[wifi] bad IP literals\n");
        return -2;
    }

    // Configure the AP netif with our gateway IP
    struct netif *ap_netif = &cyw43_state.netif[CYW43_ITF_AP];
    netif_set_ipaddr(ap_netif, &gw);
    netif_set_netmask(ap_netif, &mask);
    netif_set_gw(ap_netif, &gw);

    // Start DHCP server
    dhcp_server_init(&s_dhcp, &gw, &mask);

    // Open TCP listener on port 80
    cyw43_arch_lwip_begin();
    s_server_pcb = tcp_new_ip_type(IPADDR_TYPE_V4);
    if (!s_server_pcb) {
        cyw43_arch_lwip_end();
        printf("[wifi] tcp_new failed\n");
        return -3;
    }
    if (tcp_bind(s_server_pcb, IP4_ADDR_ANY, HTTP_PORT) != ERR_OK) {
        cyw43_arch_lwip_end();
        printf("[wifi] tcp_bind failed\n");
        return -4;
    }
    s_server_pcb = tcp_listen_with_backlog(s_server_pcb, 2);
    tcp_accept(s_server_pcb, tcp_accept_cb);
    cyw43_arch_lwip_end();

    printf("[wifi] AP ready — SSID: %s  IP: %s  Pass: %s\n",
           WIFI_SSID, AP_IP_ADDR, WIFI_PASSWORD);
    return 0;
}

void wifi_server_update_result(const char *person_name,
                               float       similarity,
                               uint32_t    latency_us,
                               bool        accepted,
                               uint32_t    beat_count) {
    cyw43_arch_lwip_begin();
    // 1. Update latest single result
    strncpy((char*)g_result.person_name, person_name,
            sizeof(g_result.person_name) - 1);
    g_result.person_name[sizeof(g_result.person_name) - 1] = '\0';
    g_result.similarity  = similarity;
    g_result.latency_us  = latency_us;
    g_result.accepted    = accepted;
    g_result.beat_count  = beat_count;
    g_result.has_result  = true;

    // 2. Shift history ring buffer (index 0 is most recent)
    for (int i = HISTORY_LEN - 1; i > 0; --i) {
        memcpy((void*)&g_history[i], (void*)&g_history[i - 1], sizeof(history_entry_t));
    }
    strncpy((char*)g_history[0].person_name, person_name,
            sizeof(g_history[0].person_name) - 1);
    g_history[0].person_name[sizeof(g_history[0].person_name) - 1] = '\0';
    g_history[0].similarity  = similarity;
    g_history[0].latency_us  = latency_us;
    g_history[0].accepted    = accepted;
    g_history[0].beat_count  = beat_count;
    if (g_history_count < HISTORY_LEN) {
        g_history_count++;
    }
    cyw43_arch_lwip_end();
}

void wifi_server_poll(void) {
    cyw43_arch_poll();
}
