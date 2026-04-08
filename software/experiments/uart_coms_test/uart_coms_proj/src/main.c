#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"

#define TAG             "UART_TEST"

#define TEST_UART       UART_NUM_1
#define TX_PIN          GPIO_NUM_17
#define RX_PIN          GPIO_NUM_16
#define BAUD_RATE       115200

#define BUF_SIZE        256
#define HANDSHAKE       0xFF
#define SEND_DELAY_MS   20

static void uart_init(void)
{
    const uart_config_t uart_cfg = {
        .baud_rate  = BAUD_RATE,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(TEST_UART, BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(TEST_UART, &uart_cfg));
    ESP_ERROR_CHECK(uart_set_pin(TEST_UART, TX_PIN, RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

/* Block until exactly `len` bytes have been received, with a timeout. 
   Returns true on success, false on timeout. */
static bool uart_read_bytes_exact(uint8_t *buf, size_t len, uint32_t timeout_ms)
{
    size_t received = 0;
    TickType_t deadline = xTaskGetTickCount() + pdMS_TO_TICKS(timeout_ms);

    while (received < len) {
        if (xTaskGetTickCount() > deadline) {
            ESP_LOGE(TAG, "Timeout waiting for %d bytes (got %d)", (int)len, (int)received);
            return false;
        }
        int n = uart_read_bytes(TEST_UART, buf + received,
                                len - received, pdMS_TO_TICKS(50));
        if (n > 0) received += n;
    }
    return true;
}

void app_main(void)
{
    uart_init();
    ESP_LOGI(TAG, "=== ESP32 UART Test (ESP-IDF v5.x) ===");

    // ── Phase 1: Receive 0x00–0x0F from Arduino ──────────────
    ESP_LOGI(TAG, "[Phase 1] Waiting to receive 0x00–0x0F from Arduino...");

    bool phase1_pass = true;

    for (uint8_t expected = 0x00; expected <= 0x0F; expected++) {
        uint8_t received = 0;
        if (!uart_read_bytes_exact(&received, 1, 5000)) {
            ESP_LOGE(TAG, "  Timeout waiting for byte 0x%02X", expected);
            phase1_pass = false;
            break;
        }

        if (received == expected) {
            ESP_LOGI(TAG, "  Recv: 0x%02X ✓", received);
        } else {
            ESP_LOGE(TAG, "  ERROR — expected 0x%02X, got 0x%02X", expected, received);
            phase1_pass = false;
        }
    }

    if (phase1_pass) {
        ESP_LOGI(TAG, "[Phase 1] Validation: PASS");
    } else {
        ESP_LOGE(TAG, "[Phase 1] Validation: FAIL");
    }

    // Send result byte + handshake back to Arduino
    uint8_t result_buf[2] = { phase1_pass ? 1 : 0, HANDSHAKE };
    uart_write_bytes(TEST_UART, result_buf, sizeof(result_buf));
    ESP_LOGI(TAG, "[Handshake sent] Starting Phase 2...");

    // ── Phase 2: Send 0x00–0x0F to Arduino ───────────────────
    ESP_LOGI(TAG, "[Phase 2] Sending 0x00–0x0F to Arduino...");

    for (uint8_t i = 0x00; i <= 0x0F; i++) {
        uart_write_bytes(TEST_UART, &i, 1);
        ESP_LOGI(TAG, "  Sent: 0x%02X", i);
        vTaskDelay(pdMS_TO_TICKS(SEND_DELAY_MS));
    }

    ESP_LOGI(TAG, "[Phase 2] Transmission complete. Waiting for Arduino result...");

    // Wait for Arduino's validation result
    uint8_t phase2_result = 0;
    if (uart_read_bytes_exact(&phase2_result, 1, 5000)) {
        if (phase2_result == 1) {
            ESP_LOGI(TAG, "[Phase 2] Arduino validation: PASS");
            ESP_LOGI(TAG, "=== All tests PASSED ===");
        } else {
            ESP_LOGE(TAG, "[Phase 2] Arduino validation: FAIL");
            ESP_LOGE(TAG, "=== Test FAILED ===");
        }
    } else {
        ESP_LOGE(TAG, "Timeout waiting for Arduino Phase 2 result.");
    }
}