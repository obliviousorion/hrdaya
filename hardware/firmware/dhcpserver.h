/**
 * dhcpserver.h — Minimal DHCP server for Raspberry Pi Pico WH AP mode.
 *
 * Implements DISCOVER → OFFER and REQUEST → ACK for a single /24 subnet.
 * Assigns IPs from 192.168.4.2 upward, gateway = 192.168.4.1.
 *
 * Based on the pico-examples DHCP server pattern (BSD-3-Clause).
 */
#ifndef DHCPSERVER_H
#define DHCPSERVER_H

#include "lwip/ip_addr.h"

#define DHCP_SERVER_PORT      67
#define DHCP_CLIENT_PORT      68
#define DHCP_MAX_LEASES       4
#define DHCP_LEASE_TIME_S     86400  /* 24 hours */

typedef struct {
    ip4_addr_t ip;        /* Leased IP address */
    uint8_t    mac[6];    /* Client MAC address */
    uint32_t   expiry;    /* Lease expiry timestamp (seconds from boot) */
} dhcp_lease_t;

typedef struct {
    ip4_addr_t      gw;
    ip4_addr_t      mask;
    dhcp_lease_t    leases[DHCP_MAX_LEASES];
} dhcp_server_t;

/**
 * Initialise the DHCP server and bind to UDP port 67.
 * @param d       Server state structure (caller owns, must stay alive).
 * @param gw      Gateway IP (Pico's own IP, e.g. 192.168.4.1).
 * @param mask    Subnet mask (e.g. 255.255.255.0).
 */
void dhcp_server_init(dhcp_server_t *d, ip4_addr_t *gw, ip4_addr_t *mask);

/**
 * De-init the DHCP server and close the UDP PCB.
 */
void dhcp_server_deinit(dhcp_server_t *d);

#endif /* DHCPSERVER_H */
