#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_log.h"

#define TAG "BRAILLE"

// I2C slave config (Arduino -> ESP32)
#define I2C_SLAVE_NUM      I2C_NUM_0
#define I2C_SLAVE_SDA      8
#define I2C_SLAVE_SCL      9
#define I2C_SLAVE_ADDR     0x08

// Buffers for incoming/outgoing I2C data
#define SLAVE_RX_BUF       256
#define SLAVE_TX_BUF       256

// I2C master (ESP32 -> PCA9685)
#define I2C_MASTER_NUM     I2C_NUM_1
#define I2C_MASTER_SDA     10
#define I2C_MASTER_SCL     11
#define I2C_MASTER_FREQ    100000

#define PCA9685_ADDR       0x40 // PCA9685 I2C address
#define PULSE_MS           50 // Duration of pulse for actuators

// Button
#define BUTTON_GPIO        GPIO_NUM_4

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

// Circular buffer for storing incoming characters
#define QSIZE 64
static char queue[QSIZE];
static volatile int head = 0;
static volatile int tail = 0;

static uint8_t current_led_state = 0; // Keep track of current LED states

// I2C master setup
static void i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA,
        .scl_io_num = I2C_MASTER_SCL,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ,
    };

    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0));
}

// PCA9685 register write
static esp_err_t pca_write(uint8_t reg, uint8_t val)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();

    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (PCA9685_ADDR << 1) | I2C_MASTER_WRITE, true); // Send address + write bit
    i2c_master_write_byte(cmd, reg, true); // Send register address
    i2c_master_write_byte(cmd, val, true); // Send data
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100)); // Execute command
    i2c_cmd_link_delete(cmd);

    return ret;
}

// Pulse a PCA9685 channel
static void pca_pulse(uint8_t ch)
{
    uint8_t base = 0x06 + 4 * ch; // Each channel has 4 registers starting at 0x06

    // Turn fully ON (ON=4096)
    pca_write(base + 0, 0x00); // ON_L
    pca_write(base + 1, 0x10); // ON_H
    pca_write(base + 2, 0x00); // OFF_L
    pca_write(base + 3, 0x00); // OFF_H

    vTaskDelay(pdMS_TO_TICKS(PULSE_MS)); // Hold pulse for 50ms

    // Turn fully OFF (OFF=4096)
    pca_write(base + 0, 0x00);
    pca_write(base + 1, 0x00);
    pca_write(base + 2, 0x00);
    pca_write(base + 3, 0x10);
}

// // Display Braille with pulsing
// static void braille_display(char c)
// {
//     int idx = -1;

//     if (c >= 'a' && c <= 'z') idx = c - 'a';
//     else if (c >= 'A' && c <= 'Z') idx = c - 'A';
//     else return;

//     uint8_t target = brailleMap[idx];

//     for (int ch = 0; ch < 6; ch++) {
//         bool prev = (current_led_state >> ch) & 1;
//         bool next = (target >> ch) & 1;

//         if (prev != next) {
//             ESP_LOGI(TAG, "Pulse LED %d", ch);
//             pca_pulse(ch);
//         }
//     }

//     current_led_state = target;
// }

// Display Braille with pulsing
void braille_display_pulse(char c)
{
    // Convert letter to index 0-25
    if (c >= 'a' && c <= 'z') {
        c -= 'a';
        ESP_LOGI(TAG, "Received letter: %c", c + 'a');
    } else if (c >= 'A' && c <= 'Z') {
        c -= 'A';
        ESP_LOGI(TAG, "Received letter: %c", c + 'A');
    } else {
        ESP_LOGW(TAG, "Unsupported char: %c", c);
        return;
    }

    uint8_t target = brailleMap[(int)c]; // Get target braille pattern for letter

    // Loop through each LED channel (0-5)
    // If next state differs from previous, pulse that LED
    // Else, leave it unchanged
    for (uint8_t ch = 0; ch < 6; ch++) {
        bool prev = (current_led_state >> ch) & 0x01;
        bool next = (target >> ch) & 0x01;

        // Only pulse LEDs that need to change
        if (prev != next) {
            ESP_LOGI(TAG, "Pulsing LED %d", ch);
            pca_pulse(ch);
        }
    }
    // Update current state
    current_led_state = target;
}

// Buffer operations
// Enqueue character into circular buffer
static void enqueue(char c)
{
    int next = (tail + 1) % QSIZE;

    if (next != head) {
        queue[tail] = c;
        tail = next;
        ESP_LOGI(TAG, "Queued: %c", c);
    } else {
        ESP_LOGW(TAG, "Queue full");
    }
}

// Dequeue character from circular buffer
static int dequeue(char *out)
{
    if (head == tail) return 0;

    *out = queue[head];
    head = (head + 1) % QSIZE;
    return 1;
}

// I2C Rx task
// Continuously read from I2C slave buffer and enqueue received characters
static void i2c_rx_task(void *arg)
{
    uint8_t buf[256];

    while (1) {
        int len = i2c_slave_read_buffer(
            I2C_SLAVE_NUM,
            buf,
            256,
            pdMS_TO_TICKS(20)
        );

        if (len > 0) {
            for (int i = 0; i < len; i++) {
                enqueue(buf[i]);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// Task to display next character on button press
static void button_task(void *arg)
{
    int last_state = 1;

    while (1) {
        int current = gpio_get_level(BUTTON_GPIO);

        // Detect falling edge (button press)
        if (last_state == 1 && current == 0) {

            char c;

            if (dequeue(&c)) {
                ESP_LOGI(TAG, "STEP → %c", c);
                braille_display_pulse(c);
            } else {
                ESP_LOGW(TAG, "Queue empty");
            }

            vTaskDelay(pdMS_TO_TICKS(200)); // debounce
        }

        last_state = current;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// Main
void app_main(void)
{
    ESP_LOGI(TAG, "Starting Braille System");

    // PCA9685 master init
    i2c_master_init();
    pca_write(0x00, 0x00); // wake PCA9685

    // Configure I2C slave
    i2c_config_t conf = {
        .mode = I2C_MODE_SLAVE,
        .sda_io_num = I2C_SLAVE_SDA,
        .scl_io_num = I2C_SLAVE_SCL,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .slave.addr_10bit_en = 0,
        .slave.slave_addr = I2C_SLAVE_ADDR,
    };

    ESP_ERROR_CHECK(i2c_param_config(I2C_SLAVE_NUM, &conf));

    vTaskDelay(pdMS_TO_TICKS(50)); // stabilization delay

    ESP_ERROR_CHECK(i2c_driver_install(
        I2C_SLAVE_NUM,
        I2C_MODE_SLAVE,
        SLAVE_RX_BUF,
        SLAVE_TX_BUF,
        0
    ));

    ESP_LOGI(TAG, "I2C slave ready");

    // Configure button GPIO
    gpio_config_t btn_conf = {
        .pin_bit_mask = (1ULL << BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };

    gpio_config(&btn_conf);

    // Create FreeRTOS tasks
    xTaskCreate(i2c_rx_task, "i2c_rx", 4096, NULL, 10, NULL);
    xTaskCreate(button_task, "button", 4096, NULL, 10, NULL);

    ESP_LOGI(TAG, "System ready — press button to step letters");
}