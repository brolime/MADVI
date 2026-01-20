#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "driver/uart.h"
#include "esp_log.h"

// MASTER I2C Configs
#define I2C_MASTER_NUM       I2C_NUM_0
#define I2C_MASTER_SDA_IO    8
#define I2C_MASTER_SCL_IO    9
#define I2C_MASTER_FREQ_HZ   100000
#define PCA9685_ADDR         0x40

// UART Configs
#define UART_NUM             UART_NUM_0
#define UART_BUF_SIZE        1024
#define PULSE_MS             50

// PCA9685 Channel ON/OFF for LED Control
#define CHANNEL_HIGH         0x10
#define CHANNEL_LOW          0x00

static const char *TAG = "BRAILLE_PULSE";

// I2C Master Device and Bus Handles
static i2c_master_bus_handle_t master_i2c_bus = NULL;
static i2c_master_dev_handle_t pca9685_dev = NULL;

// Current LED state (bitmask)
static uint8_t current_led_state = 0x00;


// 6-dot Braille mapping
// 1  4
// 2  5
// 3  6
const uint8_t brailleMap[26] = {
    0b000001, 0b000011, 0b000101, 0b000111, 0b000110,
    0b001101, 0b001111, 0b001110, 0b001011, 0b001101,
    0b010001, 0b010011, 0b010101, 0b010111, 0b010110,
    0b011101, 0b011111, 0b011110, 0b011011, 0b011101,
    0b110001, 0b110011, 0b101101, 0b110101, 0b110111,
    0b110110
};

/* ===================== I2C INIT ===================== */
void i2c_master_init(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_MASTER_NUM,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &master_i2c_bus));

    i2c_device_config_t dev_cfg = {
        .device_address = PCA9685_ADDR,
        .scl_speed_hz = I2C_MASTER_FREQ_HZ,
    };

    ESP_ERROR_CHECK(
        i2c_master_bus_add_device(master_i2c_bus, &dev_cfg, &pca9685_dev)
    );

    ESP_LOGI(TAG, "I2C initialized OK");
}

// PCA9685 I2C Write Value to Register
esp_err_t pca9685_write_reg(uint8_t reg, uint8_t val)
{
    if (pca9685_dev == NULL) {
        ESP_LOGE(TAG, "PCA9685 device handle NULL");
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t buf[2] = { reg, val };
    return i2c_master_transmit(
        pca9685_dev,
        buf,
        sizeof(buf),
        pdMS_TO_TICKS(100)
    );
}

// PCA9685 I2C Read Value from Register
esp_err_t pca9685_read_reg(uint8_t reg, uint8_t *val)
{
    if (pca9685_dev == NULL || val == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    return i2c_master_transmit_receive(
        pca9685_dev,
        &reg,
        1,
        val,
        1,
        pdMS_TO_TICKS(100)
    );
}

// PCA9685 I2C Read Multiple Registers
esp_err_t pca9685_read_regs(uint8_t start_reg, uint8_t *buf, size_t len)
{
    if (pca9685_dev == NULL || buf == NULL || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    return i2c_master_transmit_receive(
        pca9685_dev,
        &start_reg,
        1,
        buf,
        len,
        pdMS_TO_TICKS(100)
    );
}

// PCA9685 Pulse a Single Channel Output (signifying a change in Braille dot state)
esp_err_t pca9685_pulse_channel(uint8_t channel)
{
    uint8_t base_reg = 0x06 + 4 * channel;

    // FULL ON
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 0, CHANNEL_LOW));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 1, CHANNEL_HIGH));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 2, CHANNEL_LOW));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 3, CHANNEL_LOW));

    vTaskDelay(pdMS_TO_TICKS(PULSE_MS));

    // FULL OFF
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 0, CHANNEL_LOW));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 1, CHANNEL_LOW));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 2, CHANNEL_LOW));
    ESP_ERROR_CHECK(pca9685_write_reg(base_reg + 3, CHANNEL_HIGH));

    return ESP_OK;
}

// Takes letter from UART and coordinates with correct pins to pulse
void braille_display_pulse(char letter)
{
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

    uint8_t target = brailleMap[(int)letter];

    for (uint8_t ch = 0; ch < 6; ch++) {
        bool prev = (current_led_state >> ch) & 0x01;
        bool next = (target >> ch) & 0x01;

        if (prev != next) {
            ESP_LOGI(TAG, "Pulsing LED %d", ch);
            pca9685_pulse_channel(ch);
        }
    }

    current_led_state = target;
}

// Constantly waits for keyboard input via UART and reads it to figure out pulsing 
void uart_task(void *arg)
{
    uint8_t data[UART_BUF_SIZE];

    while (1) {
        int len = uart_read_bytes(
            UART_NUM,
            data,
            sizeof(data) - 1,
            pdMS_TO_TICKS(100)
        );

        if (len > 0) {
            data[len] = 0;
            for (int i = 0; i < len; i++) {
                braille_display_pulse(data[i]);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Initializing I2C...");
    i2c_master_init();
    vTaskDelay(pdMS_TO_TICKS(100));

    // Wake PCA9685
    ESP_ERROR_CHECK(pca9685_write_reg(0x00, 0x00));
    vTaskDelay(pdMS_TO_TICKS(10));

    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };

    uart_param_config(UART_NUM, &uart_config);
    uart_driver_install(UART_NUM, UART_BUF_SIZE * 2, 0, 0, NULL, 0);

    ESP_LOGI(TAG, "UART ready. Type letters to pulse LEDs.");

    xTaskCreate(uart_task, "uart_task", 4096, NULL, 10, NULL);
}
