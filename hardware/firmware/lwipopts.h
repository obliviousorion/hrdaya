/**
 * lwipopts.h — lwIP configuration for Raspberry Pi Pico WH (RP2040 + CYW43439)
 *
 * Minimal configuration for a single-threaded AP-mode HTTP server.
 * Based on the official pico-examples lwipopts.h template.
 */
#ifndef _LWIPOPTS_H
#define _LWIPOPTS_H

// ---- Threading model -------------------------------------------------------
// We use cyw43_arch_lwip_threadsafe_background: lwIP runs in IRQ context.
// No RTOS — raw bare-metal.
#define NO_SYS                      1
#define SYS_LIGHTWEIGHT_PROT        0
#define LWIP_SOCKET                 0
#define LWIP_NETCONN                0

// ---- Core protocols --------------------------------------------------------
#define LWIP_ARP                    1
#define LWIP_ETHERNET               1
#define LWIP_ICMP                   1
#define LWIP_RAW                    1
#define LWIP_TCP                    1
#define LWIP_UDP                    1
#define LWIP_DNS                    0
#define LWIP_DHCP                   0   // No DHCP client needed (we are the AP/server)
#define LWIP_AUTOIP                 0

// ---- Memory ----------------------------------------------------------------
#define MEM_ALIGNMENT               4
#define MEM_SIZE                    4000
#define MEMP_NUM_TCP_SEG            32
#define MEMP_NUM_ARP_QUEUE          10
#define PBUF_POOL_SIZE              24
#define PBUF_POOL_BUFSIZE           1500

// ---- TCP -------------------------------------------------------------------
#define TCP_MSS                     1460
#define TCP_WND                     (4 * TCP_MSS)
#define TCP_SND_BUF                 (4 * TCP_MSS)
#define TCP_SND_QUEUELEN            ((2 * (TCP_SND_BUF) + (TCP_MSS - 1)) / (TCP_MSS))
#define LWIP_TCP_KEEPALIVE          0

// ---- Network interface -----------------------------------------------------
#define LWIP_NETIF_STATUS_CALLBACK  1
#define LWIP_NETIF_LINK_CALLBACK    1
#define LWIP_NETIF_HOSTNAME         1
#define LWIP_NETIF_TX_SINGLE_PBUF   1

// ---- DHCP / ARP checks (disable for AP mode where we are the authority) ----
#define DHCP_DOES_ARP_CHECK         0
#define LWIP_DHCP_DOES_ACD_CHECK    0

// ---- Checksum offload (CYW43 handles Ethernet-level checksums) -------------
#define LWIP_CHECKSUM_CTRL_PER_NETIF 0

// ---- Debug (disable for production) ----------------------------------------
#define LWIP_DEBUG                  0
#define LWIP_STATS                  0
#define LWIP_STATS_DISPLAY          0

#endif /* _LWIPOPTS_H */
