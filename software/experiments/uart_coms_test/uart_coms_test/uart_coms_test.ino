// Arduino UNO R4 — UART Test (115200 baud)
// Phase 1: Sends 0x00–0x0F to ESP32
// Phase 2: Receives 0x00–0x0F from ESP32 and validates order

#define BAUD_RATE     115200
#define HANDSHAKE     0xFF
#define SEND_DELAY_MS 20   // gap between each transmitted byte

void setup() {
  Serial.begin(115200);   // USB debug monitor
  Serial1.begin(BAUD_RATE); // Hardware UART (TX=D1, RX=D0)

  while (!Serial);  // Wait for Serial Monitor on R4

  Serial.println("=== Arduino UART Test ===");
  Serial.println("Waiting 2s for ESP32 to boot...");
  delay(2000);

  // ── Phase 1: Arduino → ESP32 ──────────────────────────────
  Serial.println("\n[Phase 1] Sending 0x00–0x0F to ESP32...");

  delay(5000);

  for (uint8_t i = 0; i <= 0x0F; i++) {
    Serial1.write(i);
    Serial.print("  Sent: 0x0");
    Serial.println(i, HEX);
    delay(SEND_DELAY_MS);
  }

  Serial.println("[Phase 1] Transmission complete.");
  Serial.println("Waiting for ESP32 Phase 1 result + handshake...");

  // Wait for ESP32's PASS/FAIL result byte (1 = pass, 0 = fail)
  while (Serial1.available() < 2); // result byte + handshake
  uint8_t phase1_result   = Serial1.read();
  uint8_t handshake_byte  = Serial1.read();

  if (phase1_result == 1) {
    Serial.println("[Phase 1] ESP32 validation: PASS");
  } else {
    Serial.println("[Phase 1] ESP32 validation: FAIL");
  }

  if (handshake_byte != HANDSHAKE) {
    Serial.println("[ERROR] Bad handshake byte from ESP32. Halting.");
    while (1);
  }

  Serial.println("[Handshake received] Starting Phase 2...\n");

  // ── Phase 2: ESP32 → Arduino ──────────────────────────────
  Serial.println("[Phase 2] Receiving 0x00–0x0F from ESP32...");

  uint8_t expected = 0x00;
  bool    pass     = true;

  while (expected <= 0x0F) {
    if (Serial1.available()) {
      uint8_t received = Serial1.read();
      if (received == expected) {
        Serial.print("  Recv: 0x0");
        Serial.print(received, HEX);
        Serial.println(" ✓");
      } else {
        Serial.print("  ERROR — expected 0x0");
        Serial.print(expected, HEX);
        Serial.print(", got 0x0");
        Serial.println(received, HEX);
        pass = false;
      }
      expected++;
    }
  }

  // Send result back to ESP32
  Serial1.write(pass ? 1 : 0);

  Serial.println();
  if (pass) {
    Serial.println("[Phase 2] Arduino validation: PASS");
    Serial.println("=== All tests PASSED ===");
  } else {
    Serial.println("[Phase 2] Arduino validation: FAIL");
    Serial.println("=== Test FAILED ===");
  }
}

void loop() {}