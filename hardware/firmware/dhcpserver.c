/**
 * dhcpserver.c — Minimal DHCP server for Raspberry Pi Pico WH AP mode.
 *
 * Handles DHCP DISCOVER → OFFER and REQUEST → ACK.
 * Assigns one IP per unique MAC address from a fixed pool starting at
 * gateway + 1 (e.g. 192.168.4.2, 192.168.4.3, …).
 *
 * Based on the pico-examples DHCP server (BSD-3-Clause).
 */

#include <string.h>
#include <stdio.h>
#include "pico/stdlib.h"
#include "lwip/udp.h"
#include "lwip/pbuf.h"
#include "lwip/netif.h"
#include "dhcpserver.h"

// ---- DHCP wire-format constants -------------------------------------------
#define DHCP_OP_REQUEST       1
#define DHCP_OP_REPLY         2
#define DHCP_MAGIC_COOKIE     0x63825363

#define DHCP_OPT_SUBNET       1
#define DHCP_OPT_GW           3
#define DHCP_OPT_DNS          6
#define DHCP_OPT_HOSTNAME     12
#define DHCP_OPT_REQIP        50
#define DHCP_OPT_LEASETIME    51
#define DHCP_OPT_MSGTYPE      53
#define DHCP_OPT_SERVERID     54
#define DHCP_OPT_PARAMREQ     55
#define DHCP_OPT_END          255

#define DHCPDISCOVER          1
#define DHCPOFFER             2
#define DHCPREQUEST           3
#define DHCPACK               5

typedef struct __attribute__((packed)) {
    uint8_t  op;
    uint8_t  htype;
    uint8_t  hlen;
    uint8_t  hops;
    uint32_t xid;
    uint16_t secs;
    uint16_t flags;
    uint32_t ciaddr;
    uint32_t yiaddr;
    uint32_t siaddr;
    uint32_t giaddr;
    uint8_t  chaddr[16];
    uint8_t  sname[64];
    uint8_t  file[128];
    uint32_t magic;
    uint8_t  options[308];
} dhcp_msg_t;

// ---- Internal helpers ------------------------------------------------------
static dhcp_server_t *s_server;
static struct udp_pcb *s_pcb;

static int find_lease_by_mac(dhcp_server_t *d, const uint8_t *mac) {
    for (int i = 0; i < DHCP_MAX_LEASES; ++i) {
        if (memcmp(d->leases[i].mac, mac, 6) == 0) return i;
    }
    return -1;
}

static int find_free_or_expired_lease(dhcp_server_t *d) {
    uint32_t now = to_ms_since_boot(get_absolute_time()) / 1000;
    for (int i = 0; i < DHCP_MAX_LEASES; ++i) {
        if (ip4_addr_isany(&d->leases[i].ip)) return i;
        if (d->leases[i].expiry < now)        return i;
    }
    return -1;
}

static uint8_t *opt_put(uint8_t *p, uint8_t tag, uint8_t len, const void *val) {
    *p++ = tag;
    *p++ = len;
    memcpy(p, val, len);
    return p + len;
}

static uint8_t *opt_put_u8(uint8_t *p, uint8_t tag, uint8_t val) {
    return opt_put(p, tag, 1, &val);
}

static uint8_t *opt_put_u32(uint8_t *p, uint8_t tag, uint32_t val) {
    uint32_t v = lwip_htonl(val);
    return opt_put(p, tag, 4, &v);
}

static void send_offer_or_ack(dhcp_server_t *d, struct udp_pcb *pcb,
                               const ip_addr_t *src_addr, uint16_t src_port,
                               dhcp_msg_t *req, uint8_t msg_type, ip4_addr_t *offered_ip) {
    dhcp_msg_t reply;
    memset(&reply, 0, sizeof(reply));

    reply.op     = DHCP_OP_REPLY;
    reply.htype  = 1;
    reply.hlen   = 6;
    reply.xid    = req->xid;
    reply.yiaddr = offered_ip->addr;
    reply.siaddr = d->gw.addr;
    reply.magic  = lwip_htonl(DHCP_MAGIC_COOKIE);
    memcpy(reply.chaddr, req->chaddr, 6);

    uint8_t *p = reply.options;
    p = opt_put_u8 (p, DHCP_OPT_MSGTYPE,   msg_type);
    p = opt_put_u32(p, DHCP_OPT_SERVERID,  d->gw.addr);
    p = opt_put_u32(p, DHCP_OPT_LEASETIME, DHCP_LEASE_TIME_S);
    p = opt_put    (p, DHCP_OPT_SUBNET,  4, &d->mask.addr);
    p = opt_put    (p, DHCP_OPT_GW,      4, &d->gw.addr);
    p = opt_put    (p, DHCP_OPT_DNS,     4, &d->gw.addr);
    *p++ = DHCP_OPT_END;

    struct pbuf *out = pbuf_alloc(PBUF_TRANSPORT, sizeof(dhcp_msg_t), PBUF_RAM);
    if (!out) return;
    memcpy(out->payload, &reply, sizeof(dhcp_msg_t));

    ip_addr_t dst;
    IP4_ADDR(&dst, 255, 255, 255, 255);
    udp_sendto(pcb, out, &dst, DHCP_CLIENT_PORT);
    pbuf_free(out);
}

static void dhcp_recv(void *arg, struct udp_pcb *pcb, struct pbuf *p,
                       const ip_addr_t *addr, u16_t port) {
    dhcp_server_t *d = (dhcp_server_t *)arg;
    if (!p || p->tot_len < sizeof(dhcp_msg_t)) {
        if (p) pbuf_free(p);
        return;
    }

    dhcp_msg_t msg;
    memcpy(&msg, p->payload, sizeof(dhcp_msg_t));
    pbuf_free(p);

    if (msg.op != DHCP_OP_REQUEST) return;
    if (lwip_ntohl(msg.magic) != DHCP_MAGIC_COOKIE) return;

    // Parse message type from options
    uint8_t msg_type = 0;
    for (uint8_t *o = msg.options; *o != DHCP_OPT_END && o < msg.options + 308; ) {
        if (*o == DHCP_OPT_MSGTYPE) { msg_type = *(o + 2); break; }
        o += *(o + 1) + 2;
    }
    if (msg_type != DHCPDISCOVER && msg_type != DHCPREQUEST) return;

    // Find or allocate lease
    int slot = find_lease_by_mac(d, msg.chaddr);
    if (slot < 0) slot = find_free_or_expired_lease(d);
    if (slot < 0) return; // pool exhausted

    dhcp_lease_t *lease = &d->leases[slot];
    if (ip4_addr_isany(&lease->ip)) {
        // Assign next IP: gateway + slot + 1 (e.g. 192.168.4.2 for slot 0)
        uint32_t base = lwip_ntohl(d->gw.addr);
        ip4_addr_set_u32(&lease->ip, lwip_htonl(base + slot + 1));
    }
    memcpy(lease->mac, msg.chaddr, 6);
    lease->expiry = to_ms_since_boot(get_absolute_time()) / 1000 + DHCP_LEASE_TIME_S;

    uint8_t reply_type = (msg_type == DHCPDISCOVER) ? DHCPOFFER : DHCPACK;
    send_offer_or_ack(d, pcb, addr, port, &msg, reply_type, &lease->ip);
}

// ---- Public API -----------------------------------------------------------
void dhcp_server_init(dhcp_server_t *d, ip4_addr_t *gw, ip4_addr_t *mask) {
    s_server = d;
    ip4_addr_copy(d->gw,   *gw);
    ip4_addr_copy(d->mask, *mask);
    memset(d->leases, 0, sizeof(d->leases));

    s_pcb = udp_new();
    if (!s_pcb) return;
    ip_set_option(s_pcb, SOF_BROADCAST);
    udp_bind(s_pcb, IP_ANY_TYPE, DHCP_SERVER_PORT);
    udp_recv(s_pcb, dhcp_recv, d);
}

void dhcp_server_deinit(dhcp_server_t *d) {
    if (s_pcb) {
        udp_remove(s_pcb);
        s_pcb = NULL;
    }
}
