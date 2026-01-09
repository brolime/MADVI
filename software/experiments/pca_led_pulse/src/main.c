#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "driver/uart.h"
#include "esp_log.h"

#define I2C_MASTER_NUM       I2C_NUM_0
#define I2C_MASTER_SDA_IO    8
#define I2C_MASTER_SCL_IO    9
#define I2C_MASTER_FREQ_HZ   100000

#define UART_NUM             UART_NUM_0
#define UART_BUF_SIZE        1024
#define PULSE_MS             50

static const char *TAG = "BRAILLE_PULSE";
static uint8_t current_led_state = 0; // Keep track of LED states

// 6-dot Braille mapping
// 1  4
// 2  5
// 3  6
const uint8_t brailleMap[26] = {
    0b000001, // A
    0b000011, // B
    0b000101, // C
    0b000111, // D
    0b000110, // E
    0b001101, // F
    0b001111, // G
    0b001110, // H
    0b001011, // I
    0b001101, // J
    0b010001, // K
    0b010011, // L
    0b010101, // M
    0b010111, // N
    0b010110, // O
    0b011101, // P
    0b011111, // Q
    0b011110, // R
    0b011011, // S
    0b011101, // T
    0b110001, // U
    0b110011, // V
    0b101101, // W
    0b110101, // X
    0b110111, // Y
    0b110110, // Z
};

// --- I2C setup ---
void i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0));
}

// --- PCA9685 register write ---
esp_err_t pca9685_write_reg(uint8_t reg, uint8_t val)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (0x40 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, val, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret;
}

// --- LED pulse helper ---
esp_err_t pca9685_pulse_channel(uint8_t channel)
{
    uint8_t base_reg = 0x06 + 4 * channel;
    // Turn fully ON (ON=4096)
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 0, 0x00)); // ON_L
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 1, 0x10)); // ON_H
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 2, 0x00)); // OFF_L
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 3, 0x00)); // OFF_H
    vTaskDelay(pdMS_TO_TICKS(PULSE_MS)); // Wait 50ms
    // Turn fully OFF (OFF=4096)
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 0, 0x00));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 1, 0x00));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 2, 0x00));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 3, 0x10));
    return ESP_OK;
}

// --- Display Braille with pulsing ---
void braille_display_pulse(char letter)
{
    // Convert letter to index 0-25
    if (letter >= 'a' && letter <= 'z') {
        letter -= 'a';
        ESP_LOGI(TAG, "Received letter: %c", letter + 'a');
    } else if (letter >= 'A' && letter <= 'Z') {
        letter -= 'A';
        ESP_LOGI(TAG, "Received letter: %c", letter + 'A');
    } else {
        ESP_LOGW(TAG, "Unsupported char: %c", letter);
        return;
    }

    uint8_t target = brailleMap[(int)letter]; // Get target braille pattern for letter

    // Loop through each LED channel (0-5)
    // If next state differs from previous, pulse that LED
    // Else, leave it unchanged
    for (uint8_t ch = 0; ch < 6; ch++) {
        bool prev = (current_led_state >> ch) & 0x01;
        bool next = (target >> ch) & 0x01;

        // Only pulse LEDs that need to change
        if (prev != next) {
            ESP_LOGI(TAG, "Pulsing LED %d", ch);
            pca9685_pulse_channel(ch);
        }
    }
    // Update current state
    current_led_state = target;
}

// --- UART task ---
// Read letters from UART and pulse corresponding braille LEDs
void uart_task(void *arg)
{
    uint8_t data[UART_BUF_SIZE];
    while (1) {
        int len = uart_read_bytes(UART_NUM, data, sizeof(data) - 1, pdMS_TO_TICKS(100));
        if (len > 0) {
            data[len] = 0;
            for (int i = 0; i < len; i++) {
                braille_display_pulse(data[i]);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

// --- Main ---
void app_main(void)
{
    // Initialize I2C
    ESP_LOGI(TAG, "Initializing I2C...");
    i2c_master_init();
    vTaskDelay(pdMS_TO_TICKS(100));

    // Wake PCA9685
    uint8_t mode1 = 0x00;
    pca9685_write_reg(0x00, mode1);
    vTaskDelay(pdMS_TO_TICKS(10));

    // Configure UART
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };

    // Set UART parameters and install driver
    uart_param_config(UART_NUM, &uart_config);
    uart_driver_install(UART_NUM, UART_BUF_SIZE * 2, 0, 0, NULL, 0);

    ESP_LOGI(TAG, "UART ready. Type letters to pulse LEDs.");

    xTaskCreate(uart_task, "uart_task", 4096, NULL, 10, NULL); // Start UART task
}