cmake_minimum_required(VERSION 3.13)

# Board must be pico_w to unlock CYW43439 Wi-Fi libraries
# Pass at configure time: cmake -DPICO_BOARD=pico_w ...
if(NOT PICO_BOARD)
    set(PICO_BOARD pico_w CACHE STRING "Target Pico board")
endif()

# PICO_SDK_PATH must be set in environment or passed via -DPICO_SDK_PATH=...
include(pico_sdk_import.cmake)

project(pico_ecg_biometrics C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

# ----- Wi-Fi credentials (override at cmake configure time) -----
if(NOT DEFINED WIFI_SSID)
    set(WIFI_SSID "pico")
endif()
if(NOT DEFINED WIFI_PASSWORD)
    set(WIFI_PASSWORD "picosama")
endif()

add_executable(pico_ecg_biometrics
    main.c
    wifi_server.c
    dhcpserver.c
)

target_include_directories(pico_ecg_biometrics PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_compile_definitions(pico_ecg_biometrics PRIVATE
    WIFI_SSID="${WIFI_SSID}"
    WIFI_PASSWORD="${WIFI_PASSWORD}"
    CYW43_HOST_NAME="pico-ecg"
)

target_link_libraries(pico_ecg_biometrics
    pico_stdlib
    pico_time
    hardware_gpio
    pico_cyw43_arch_lwip_threadsafe_background
    pico_lwip_core
    pico_lwip_core4
    pico_lwip_netif
    m
)

# USB CDC for primary serial channel; disable UART
pico_enable_stdio_usb(pico_ecg_biometrics 1)
pico_enable_stdio_uart(pico_ecg_biometrics 0)

# Generate .uf2, .bin, .hex, .map
pico_add_extra_outputs(pico_ecg_biometrics)
