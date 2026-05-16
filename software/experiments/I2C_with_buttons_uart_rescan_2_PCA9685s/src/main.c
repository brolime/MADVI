
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/uart.h"

#define TAG "BRAILLE"

// I2C slave config (Arduino/FPGA -> ESP32)
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

// PCA9685 I2C addresses
#define PCA9685_ADDR_1       0x40
#define PCA9685_ADDR_2       0x41

// Buttons
#define FWD_BUTTON_GPIO     GPIO_NUM_4
#define BACK_BUTTON_GPIO    GPIO_NUM_5
#define RESET_BUTTON_GPIO   GPIO_NUM_18
#define RESCAN_BUTTON_GPIO  GPIO_NUM_17

// UART handshake to master
#define UART_PORT           UART_NUM_1
#define UART_TX_PIN         GPIO_NUM_16
#define UART_RX_PIN         GPIO_NUM_15
#define UART_BAUD_RATE      115200

// History buffer for received characters
#define HISTORY_SIZE 512
static char history[HISTORY_SIZE];
static int history_len = 0;
static int current_pos = -5;

static bool display_cleared = false;

// Rescan state
static bool rescan_waiting = false;
static int64_t rescan_start_time = 0;
#define RESCAN_TIMEOUT_MS 30000

// 6-dot Braille mapping
// 1  4
// 2  5
// 3  6
const uint8_t brailleMap[26] = {
 // 0b654321
    0b000001, // A
    0b000011, // B
    0b001001, // C
    0b011001, // D
    0b010001, // E
    0b001011, // F
    0b011011, // G
    0b010011, // H
    0b001010, // I
    0b011010, // J
    0b000101, // K
    0b000111, // L
    0b001101, // M
    0b011101, // N
    0b010101, // O
    0b001111, // P
    0b011111, // Q
    0b010111, // R
    0b001110, // S
    0b011110, // T
    0b100101, // U
    0b100111, // V
    0b111010, // W
    0b101101, // X
    0b111101, // Y
    0b110101, // Z
};

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

// UART setup
static void uart_init(void)
{
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(
        UART_PORT,
        256,
        0,
        0,
        NULL,
        0
    ));

    ESP_ERROR_CHECK(uart_param_config(
        UART_PORT,
        &uart_config
    ));

    ESP_ERROR_CHECK(uart_set_pin(
        UART_PORT,
        UART_TX_PIN,
        UART_RX_PIN,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE
    ));
}

// Send rescan request to master
static void send_rescan_request(void)
{
    const char *msg = "RESCAN\n";

    for (int i = 0; i < strlen(msg); i++) {
        ESP_LOGI(TAG, "UART TX: %c", msg[i]);
        uart_write_bytes(
            UART_PORT,
            &msg[i],
            1
        );
    }

    ESP_LOGI(TAG, "UART RESCAN SENT");
}

// PCA9685 register write
static esp_err_t pca_write(uint8_t addr, uint8_t reg, uint8_t val)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();

    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, val, true);
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);

    return ret;
}

// Set the state of a PCA9685 channel
static void pca_set(uint8_t ch, bool on)
{
    uint8_t addr;
    uint8_t local_ch;

    // Determine which PCA9685 to use based on channel number
    if (ch < 16) {
        addr = PCA9685_ADDR_1;
        local_ch = ch;
    } else {
        addr = PCA9685_ADDR_2;
        local_ch = ch - 16;
    }

    uint8_t base = 0x06 + 4 * local_ch; // Each channel has 4 registers starting at 0x06

    if (on) {
        // Turn fully ON (ON=4096)
        pca_write(addr, base + 0, 0x00); // ON_L
        pca_write(addr, base + 1, 0x10); // ON_H
        pca_write(addr, base + 2, 0x00); // OFF_L
        pca_write(addr, base + 3, 0x00); // OFF_H
    } else {
        // Turn fully OFF (OFF=4096)
        pca_write(addr, base + 0, 0x00);
        pca_write(addr, base + 1, 0x00);
        pca_write(addr, base + 2, 0x00);
        pca_write(addr, base + 3, 0x10);
    }
}

// Clear all LEDs
static void clear_display(void)
{
    for (uint8_t i = 0; i < 30; i++) {
        pca_set(i, false);
    }
}

// Display Braille cell
static void display_cell(uint8_t offset, char c)
{
    if (c == ' ') {
        for (int i = 0; i < 6; i++) {
            pca_set(offset + i, false);
        }
        return;
    }

    if (c >= 'a' && c <= 'z') {
        c -= 'a';
    }
    else if (c >= 'A' && c <= 'Z') {
        c -= 'A';
    }
    else {
        for (int i = 0; i < 6; i++) {
            pca_set(offset + i, false);
        }
        return;
    }

    uint8_t pattern = brailleMap[(int)c];

    for (uint8_t i = 0; i < 6; i++) {
        pca_set(offset + i, (pattern >> i) & 0x01);
    }
}

// Display set of characters (5 cells)
static void display_five(int pos)
{
    if (pos < 0 || pos >= history_len) return;

    char c1 = (pos + 0 < history_len) ? history[pos + 0] : ' ';
    char c2 = (pos + 1 < history_len) ? history[pos + 1] : ' ';
    char c3 = (pos + 2 < history_len) ? history[pos + 2] : ' ';
    char c4 = (pos + 3 < history_len) ? history[pos + 3] : ' ';
    char c5 = (pos + 4 < history_len) ? history[pos + 4] : ' ';

    display_cell(0,  c1);
    display_cell(6,  c2);
    display_cell(12, c3);
    display_cell(18, c4);
    display_cell(24, c5);

    ESP_LOGI(TAG, 
        "Display: %c %c %c %c %c", 
        c1, c2, c3, c4, c5
    );
}

// System reset
static void reset_system(void)
{
    ESP_LOGI(TAG, "SYSTEM RESET (RESCAN)");

    history_len = 0;
    current_pos = -5;
    clear_display();
}

// I2C RX task
static void i2c_rx_task(void *arg)
{
    uint8_t buf[256];

    while (1) {

        int len = i2c_slave_read_buffer(
            I2C_SLAVE_NUM,
            buf,
            sizeof(buf),
            pdMS_TO_TICKS(20)
        );

        if (len > 0) {

            for (int i = 0; i < len; i++) {

                char c = buf[i];
                ESP_LOGI(TAG, "Received: %c", c);

                if (history_len < HISTORY_SIZE) {
                    history[history_len++] = c;
                }
            }

            // Rescan complete check
            if (rescan_waiting) {
                ESP_LOGI(TAG, "RESCAN COMPLETE");
                rescan_waiting = false;
            }
        }

        // Rescan timeout check
        if (rescan_waiting) {
            int64_t now = esp_timer_get_time() / 1000;

            if ((now - rescan_start_time) > RESCAN_TIMEOUT_MS) {
                ESP_LOGE(TAG, "RESCAN TIMEOUT");
                rescan_waiting = false;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

// Button tasks
static void button_task(void *arg)
{
    int last_fwd = 1;
    int last_back = 1;
    int last_reset = 1;
    int last_rescan = 1;

    while (1) {

        int fwd = gpio_get_level(FWD_BUTTON_GPIO);
        int back = gpio_get_level(BACK_BUTTON_GPIO);
        int reset = gpio_get_level(RESET_BUTTON_GPIO);
        int rescan = gpio_get_level(RESCAN_BUTTON_GPIO);

        // Forward
        if (last_fwd == 1 && fwd == 0) {

            if (history_len == 0) {
                ESP_LOGW(TAG, "HISTORY EMPTY");
            }
            else if (current_pos + 5 < history_len) {
                current_pos += 5;
                display_five(current_pos);
                display_cleared = false;
            }
            else {
                ESP_LOGW(TAG, "END OF HISTORY");
            }

            vTaskDelay(pdMS_TO_TICKS(200));
        }

        // Back
        if (last_back == 1 && back == 0) {

            if (history_len == 0) {
                ESP_LOGW(TAG, "HISTORY EMPTY");
            }
            else if (current_pos - 5 >= 0) {
                current_pos -= 5;
                display_five(current_pos);
                display_cleared = false;
            }
            else {
                ESP_LOGW(TAG, "BEGINNING OF HISTORY");
            }

            vTaskDelay(pdMS_TO_TICKS(200));
        }

        // Reset
        if (last_reset == 1 && reset == 0) {

            if (!display_cleared) {
                clear_display();
                display_cleared = true;
                ESP_LOGI(TAG, "CLEARED");
            } else {
                display_five(current_pos);
                display_cleared = false;
                ESP_LOGI(TAG, "RESTORED");
            }

            vTaskDelay(pdMS_TO_TICKS(200));
        }

        // Rescan
        if (last_rescan == 1 && rescan == 0) {

            ESP_LOGI(TAG, "RESCAN TRIGGERED");

            reset_system();

            ESP_LOGI(TAG, "SENDING UART RESCAN");
            send_rescan_request();

            rescan_start_time = esp_timer_get_time() / 1000;
            rescan_waiting = true;

            vTaskDelay(pdMS_TO_TICKS(200));
        }

        last_fwd = fwd;
        last_back = back;
        last_reset = reset;
        last_rescan = rescan;

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// Main
void app_main(void)
{
    ESP_LOGI(TAG, "Starting Braille System");

    i2c_master_init(); // I2C master init for PCA9685
    uart_init(); // UART init

    // Wake both PCA9685 devices
    pca_write(PCA9685_ADDR_1, 0x00, 0x00);
    pca_write(PCA9685_ADDR_2, 0x00, 0x00);

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
    ESP_ERROR_CHECK(i2c_driver_install(
        I2C_SLAVE_NUM,
        I2C_MODE_SLAVE,
        SLAVE_RX_BUF,
        SLAVE_TX_BUF,
        0
    ));

    // Configure button GPIOs
    gpio_config_t btn = {
        .pin_bit_mask =
            (1ULL << FWD_BUTTON_GPIO) |
            (1ULL << BACK_BUTTON_GPIO) |
            (1ULL << RESET_BUTTON_GPIO) |
            (1ULL << RESCAN_BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE
    };
    gpio_config(&btn);

    // Create FreeRTOS tasks
    xTaskCreate(i2c_rx_task, "i2c_rx", 4096, NULL, 10, NULL);
    xTaskCreate(button_task, "button", 4096, NULL, 10, NULL);

    ESP_LOGI(TAG, "Ready — forward/back/reset + rescan over UART enabled, leds fully on/off, 5 braille cells over 2 PCA9685s");
}