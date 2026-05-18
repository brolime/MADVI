#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/uart.h"

#define TAG "BRAILLE"

// Buttons
#define FWD_BUTTON_GPIO     GPIO_NUM_4
#define BACK_BUTTON_GPIO    GPIO_NUM_18

// LED GPIOs
// Braille cell 1 GPIOs
#define LED_1_GPIO         GPIO_NUM_20
#define LED_2_GPIO         GPIO_NUM_21
#define LED_3_GPIO         GPIO_NUM_47
#define LED_4_GPIO         GPIO_NUM_48
#define LED_5_GPIO         GPIO_NUM_45
#define LED_6_GPIO         GPIO_NUM_0

// Braille cell 2 GPIOs
#define LED_7_GPIO         GPIO_NUM_35
#define LED_8_GPIO         GPIO_NUM_36
#define LED_9_GPIO         GPIO_NUM_37
#define LED_10_GPIO        GPIO_NUM_38
#define LED_11_GPIO        GPIO_NUM_39
#define LED_12_GPIO        GPIO_NUM_40

// Braille cell 3 GPIOs
#define LED_13_GPIO        GPIO_NUM_41
#define LED_14_GPIO        GPIO_NUM_42
#define LED_15_GPIO        GPIO_NUM_2
#define LED_16_GPIO        GPIO_NUM_1
#define LED_17_GPIO        GPIO_NUM_12
#define LED_18_GPIO        GPIO_NUM_13

// Braille cell 4 GPIOs
#define LED_19_GPIO        GPIO_NUM_6
#define LED_20_GPIO        GPIO_NUM_7
#define LED_21_GPIO        GPIO_NUM_14
#define LED_22_GPIO        GPIO_NUM_17
#define LED_23_GPIO        GPIO_NUM_5
#define LED_24_GPIO        GPIO_NUM_19

// Braille cell 5 GPIOs
#define LED_25_GPIO        GPIO_NUM_8
#define LED_26_GPIO        GPIO_NUM_3
#define LED_27_GPIO        GPIO_NUM_46
#define LED_28_GPIO        GPIO_NUM_9
#define LED_29_GPIO        GPIO_NUM_10
#define LED_30_GPIO        GPIO_NUM_11

// Array of LED GPIOs
static const gpio_num_t led_pins[30] = {
    LED_1_GPIO,
    LED_2_GPIO,
    LED_3_GPIO,
    LED_4_GPIO,
    LED_5_GPIO,
    LED_6_GPIO,
    LED_7_GPIO,
    LED_8_GPIO,
    LED_9_GPIO,
    LED_10_GPIO,
    LED_11_GPIO,
    LED_12_GPIO,
    LED_13_GPIO,
    LED_14_GPIO,
    LED_15_GPIO,
    LED_16_GPIO,
    LED_17_GPIO,
    LED_18_GPIO,
    LED_19_GPIO,
    LED_20_GPIO,
    LED_21_GPIO,
    LED_22_GPIO,
    LED_23_GPIO,
    LED_24_GPIO,
    LED_25_GPIO,
    LED_26_GPIO,
    LED_27_GPIO,
    LED_28_GPIO,
    LED_29_GPIO,
    LED_30_GPIO
};

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

#define LONG_PRESS_MS 1000

static int history_pos = -1;
static int active_cell = -1;

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

// Set the state of GPIO pins for LED
static void led_set(uint8_t index, bool on)
{
    if (index >= 30) {
        return;
    }

    gpio_set_level(
        led_pins[index],
        on ? 1 : 0
    );
}

// Clear all LEDs
static void clear_display(void)
{
    for (uint8_t i = 0; i < 30; i++) {
        led_set(i, false);
    }
}

// Display Braille cell
static void display_cell(uint8_t offset, char c)
{
    if (c == ' ') {
        for (int i = 0; i < 6; i++) {
            led_set(offset + i, false);
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
            led_set(offset + i, false);
        }
        return;
    }

    uint8_t pattern = brailleMap[(int)c];

    for (uint8_t i = 0; i < 6; i++) {
        bool state = (pattern >> i) & 0x01;
        led_set(offset + i, state);
    }
}

// Display one character in one specific cell at a time
static void display_one(void)
{
    clear_display();

    if (history_len == 0) {
        return;
    }

    if (history_pos < 0 || history_pos >= history_len) {
        return;
    }

    char c = history[history_pos];

    // active_cell:
    // 0 -> first cell
    // 1 -> second cell
    // ...
    // 4 -> fifth cell
    display_cell(active_cell * 6, c);

    ESP_LOGI(TAG,
        "Display '%c' in cell %d",
        c,
        active_cell + 1
    );
}

// System reset
static void reset_system(void)
{
    ESP_LOGI(TAG, "SYSTEM RESET (RESCAN)");

    history_len = 0;
    current_pos = -5;
    history_pos = -1;
    active_cell = -1;
    clear_display();
}

static void uart_rx_task(void *arg)
{
    uint8_t buf[128];

    while (1) {

        int len = uart_read_bytes(
            UART_PORT,
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

            if (rescan_waiting) {
                ESP_LOGI(TAG, "RESCAN COMPLETE");
                rescan_waiting = false;
            }
        }

        // Timeout check
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

    int64_t fwd_press_time = 0;
    int64_t back_press_time = 0;

    int64_t last_fwd_event = 0;
    int64_t last_back_event = 0;

    while (1) {

        int fwd = gpio_get_level(FWD_BUTTON_GPIO);
        int back = gpio_get_level(BACK_BUTTON_GPIO);

        // Forward button
        // Pressed
        if (last_fwd == 1 && fwd == 0) {
            fwd_press_time = esp_timer_get_time() / 1000;
        }

        // Released
        if (last_fwd == 0 && fwd == 1) {

            int64_t now = esp_timer_get_time() / 1000;

            // Debounce protection
            if ((now - last_fwd_event) < 300) {
                last_fwd = fwd;
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }

            last_fwd_event = now;

            int64_t press_len =
                now - fwd_press_time;

            // Long Press = Rescan
            if (press_len >= LONG_PRESS_MS) {

                ESP_LOGI(TAG, "FORWARD LONG PRESS -> RESCAN");

                reset_system();

                send_rescan_request();

                rescan_start_time = now;

                rescan_waiting = true;
            }

            // Short Press = Forward
            else {

                ESP_LOGI(TAG, "FORWARD SHORT PRESS");

                if (history_len == 0) {
                    ESP_LOGW(TAG, "HISTORY EMPTY");
                }
                else {

                    // Move forward FIRST
                    if (history_pos < history_len - 1) {

                        history_pos++;

                        active_cell++;

                        if (active_cell >= 5) {
                            active_cell = 0;
                        }
                    }
                    else {

                        ESP_LOGW(TAG, "END OF HISTORY");

                        last_fwd = fwd;

                        vTaskDelay(pdMS_TO_TICKS(10));

                        continue;
                    }

                    display_one();

                    display_cleared = false;
                }
            }

            vTaskDelay(pdMS_TO_TICKS(200));           
        }

        // Back button
        // Pressed
        if (last_back == 1 && back == 0) {
            back_press_time = esp_timer_get_time() / 1000;
        }

        // Released
        if (last_back == 0 && back == 1) {

            int64_t now = esp_timer_get_time() / 1000;

            // Debounce protection
            if ((now - last_back_event) < 300) {
                last_back = back;
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }

            last_back_event = now;

            int64_t press_len =
                now - back_press_time;

            // Long Press = Clear / Restore Display Toggle
            if (press_len >= LONG_PRESS_MS) {

                if (!display_cleared) {

                    clear_display();

                    display_cleared = true;

                    ESP_LOGI(TAG, "DISPLAY CLEARED");
                }
                else {

                    display_one();

                    display_cleared = false;

                    ESP_LOGI(TAG, "DISPLAY RESTORED");
                }
            }

            // Short Press = Back
            else {

                ESP_LOGI(TAG, "BACK SHORT PRESS");

                if (history_len == 0) {
                    ESP_LOGW(TAG, "HISTORY EMPTY");
                }
                else {

                    if (history_pos > 0) {

                        history_pos--;

                        active_cell--;

                        if (active_cell < 0) {
                            active_cell = 4;
                        }

                        display_one();

                        display_cleared = false;
                    }
                    else {
                        ESP_LOGW(TAG, "BEGINNING OF HISTORY");
                    }
                }
            }
        }

        last_fwd = fwd;
        last_back = back;

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// Main
void app_main(void)
{
    ESP_LOGI(TAG, "Starting Braille System");

    uart_init();

    // Configure button GPIOs
    gpio_config_t btn = {
        .pin_bit_mask =
            (1ULL << FWD_BUTTON_GPIO) |
            (1ULL << BACK_BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE
    };
    gpio_config(&btn);

    // Configure LED GPIOs
        gpio_config_t io_conf = {
        .pin_bit_mask =
            (1ULL << LED_1_GPIO) |
            (1ULL << LED_2_GPIO) |
            (1ULL << LED_3_GPIO) |
            (1ULL << LED_4_GPIO) |
            (1ULL << LED_5_GPIO) |
            (1ULL << LED_6_GPIO) |
            (1ULL << LED_7_GPIO) |
            (1ULL << LED_8_GPIO) |
            (1ULL << LED_9_GPIO) |
            (1ULL << LED_10_GPIO) |
            (1ULL << LED_11_GPIO) |
            (1ULL << LED_12_GPIO) |
            (1ULL << LED_13_GPIO) |
            (1ULL << LED_14_GPIO) |
            (1ULL << LED_15_GPIO) |
            (1ULL << LED_16_GPIO) |
            (1ULL << LED_17_GPIO) |
            (1ULL << LED_18_GPIO) |
            (1ULL << LED_19_GPIO) |
            (1ULL << LED_20_GPIO) |
            (1ULL << LED_21_GPIO) |
            (1ULL << LED_22_GPIO) |
            (1ULL << LED_23_GPIO) |
            (1ULL << LED_24_GPIO) |
            (1ULL << LED_25_GPIO) |
            (1ULL << LED_26_GPIO) |
            (1ULL << LED_27_GPIO) |
            (1ULL << LED_28_GPIO) |
            (1ULL << LED_29_GPIO) |
            (1ULL << LED_30_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);

    // Create FreeRTOS tasks
    xTaskCreate(uart_rx_task, "uart_rx", 4096, NULL, 10, NULL);
    xTaskCreate(button_task, "button", 4096, NULL, 10, NULL);

    ESP_LOGI(TAG, "Ready — forward/back/reset + rescan, full UART, leds fully on/off, leds driven by GPIO");
}