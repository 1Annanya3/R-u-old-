// Balance Test with MPU6050 + MQTT Integration (Multi-Trial Fixed)
// Set DEBUG to 1 for troubleshooting, 0 for production/standalone
#define DEBUG 1  // Change to 0 to disable all serial output
#define DEBUG 1 // Change to 0 to disable all serial output

#if DEBUG
  #define DEBUG_PRINT(x) Serial.print(x)
#include <ArduinoJson.h>

// WiFi Configuration
const char* ssid = "__";
const char* password = "__";
// !!! IMPORTANT: Replace "__" with your actual WiFi credentials !!!
const char* ssid = "ANNANYA";
const char* password = "_______";

// MQTT Configuration
const char* mqtt_broker = "broker.hivemq.com";
const char* TOPIC_DATA = "cs3237/A0277110N/health/data/Balance_Test";
Adafruit_MPU6050 mpu;
const int RESET_BUTTON_PIN = 15;

// Balance detection thresholds - Z-axis modulus (absolute value)
const float Z_AXIS_THRESHOLD = 7.0;  // |Z| >= 7.0 to START, |Z| < 7.0 to STOP
const float X_AXIS_THRESHOLD = 6.0;   // |X| <= 5.0 to START, |X| > 5.0 to STOP
// Balance detection thresholds
// Z_AXIS_THRESHOLD = 0.7 (Z modulus threshold, as requested)
// X_AXIS_THRESHOLD = 0.6 (X modulus threshold, as requested)
const float Z_AXIS_THRESHOLD = 6; // |Z| >= 0.7 to START, |Z| < 0.7 to STOP
const float X_AXIS_THRESHOLD = 5; // |X| <= 0.6 to START, |X| > 0.6 to STOP

// Trial state variables
bool trialRunning = false;
bool waitingForStart = false;
unsigned long trialStartTime = 0;
unsigned long lastMqttCheck = 0;
const unsigned long MQTT_CHECK_INTERVAL = 50;  // Check MQTT every 50ms
const unsigned long MQTT_CHECK_INTERVAL = 50; // Check MQTT every 50ms

// Debugging interval for accelerometer values
#define ACCEL_DEBUG_INTERVAL 3000 // Output accel values every 3000ms (3 seconds), as requested
unsigned long lastAccelDebug = 0;

// Participant metadata (received from server)
String participant_id = "UNKNOWN";
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
      participant_gender = doc["gender"].as<String>();

      DEBUG_PRINTLN("[META] Participant metadata received:");
      DEBUG_PRINT("       ID: ");
      DEBUG_PRINT("        ID: ");
      DEBUG_PRINTLN(participant_id);
      DEBUG_PRINT("       Age: ");
      DEBUG_PRINT("        Age: ");
      DEBUG_PRINTLN(participant_age);
      DEBUG_PRINT("       Gender: ");
      DEBUG_PRINT("        Gender: ");
      DEBUG_PRINTLN(participant_gender);
    }

void mqtt_callback(char* topic, byte* payload, unsigned int length) {

      DEBUG_PRINTLN("\n========================================");
      DEBUG_PRINTLN("[TEST] NEW BALANCE TEST ARMED!");
      DEBUG_PRINTF("[TEST] Waiting for |Z-axis| >= %.1f to start timer...\n", Z_AXIS_THRESHOLD);
      DEBUG_PRINTF("[TEST] Waiting for balance stability break (|Z|>=%.1f OR |X|<=%.1f) to start timer...\n", Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
      DEBUG_PRINTLN("[TEST] (Lift your leg to start)");
      DEBUG_PRINTLN("========================================\n");
    }
void publish_balance_result(unsigned long duration_ms) {
  serializeJson(doc, jsonBuffer);

  DEBUG_PRINTLN("\n[RESULT] Publishing balance test result:");
  DEBUG_PRINT("         Duration: ");
  DEBUG_PRINT("          Duration: ");
  DEBUG_PRINT(duration_ms);
  DEBUG_PRINTLN(" ms");
  DEBUG_PRINT("         JSON: ");
  DEBUG_PRINT("          JSON: ");
  DEBUG_PRINTLN(jsonBuffer);

  // Ensure MQTT is connected before publishing
void setup(void) {
  // Setup MQTT with larger keepalive
  mqtt_client.setServer(mqtt_broker, mqtt_port);
  mqtt_client.setCallback(mqtt_callback);
  mqtt_client.setKeepAlive(60);  // 60 second keepalive
  mqtt_client.setKeepAlive(60); // 60 second keepalive
  DEBUG_PRINTLN("[MQTT] Client configured");

  // Try to initialize MPU6050
void setup(void) {
  DEBUG_PRINTLN("\n========================================");
  DEBUG_PRINTLN("Setup complete - System ready!");
  DEBUG_PRINTF("Threshold: |Z-axis| >= %.1f starts, < %.1f stops\n", 
               Z_AXIS_THRESHOLD, Z_AXIS_THRESHOLD);
  DEBUG_PRINTF("Thresholds: START requires (|Z|>=%.1f OR |X|<=%.1f)\n", 
               Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
  DEBUG_PRINTF("Thresholds: STOP requires (|Z|<%.1f AND |X|>%.1f)\n", 
               Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
  DEBUG_PRINTLN("Waiting for START_BALANCE command...");
  DEBUG_PRINTLN("========================================\n");

void loop() {
  float z_axis_abs = abs(a.acceleration.z);
  float x_axis_abs = abs(a.acceleration.x);

  // *** Debugging: Output accelerometer values every few seconds ***
  if (DEBUG && (millis() - lastAccelDebug >= ACCEL_DEBUG_INTERVAL)) {
      DEBUG_PRINT("[DEBUG ACCEL] Raw X: ");
      DEBUG_PRINT(a.acceleration.x);
      DEBUG_PRINT(", Raw Y: ");
      DEBUG_PRINT(a.acceleration.y);
      DEBUG_PRINT(", Raw Z: ");
      DEBUG_PRINT(a.acceleration.z);
      DEBUG_PRINTF(" | Modulus: |Z|=%.2f, |X|=%.2f\n", z_axis_abs, x_axis_abs);
      lastAccelDebug = millis();
  }

  // Only run balance detection if START_BALANCE command received
  if (waitingForStart && trialRunning) {
  if (waitingForStart || trialRunning) {

    // *** TRIAL START: |Z-axis| >= 7.0 or |X-axis| <= 5.0 ***
    // *** TRIAL START: |Z| >= 0.7 or |X| <= 0.6 ***
    if (!trialRunning && waitingForStart) {
      if ((z_axis_abs >= Z_AXIS_THRESHOLD) || (x_axis_abs <= X_AXIS_THRESHOLD)){
      if ((z_axis_abs >= Z_AXIS_THRESHOLD) && (x_axis_abs <= X_AXIS_THRESHOLD)){
        // start timer!
        trialRunning = true;
        waitingForStart = false;  // No longer waiting
        waitingForStart = false; // No longer waiting
        trialStartTime = millis();

        DEBUG_PRINTLN("[TEST] *** TIMER STARTED ***");
        DEBUG_PRINTF("[TEST] |Z-axis| = %.2f (threshold: >= %.1f)\n", 
                     z_axis_abs, Z_AXIS_THRESHOLD);
        DEBUG_PRINTF("[TEST] |Z-axis| = %.2f, |X-axis| = %.2f\n", z_axis_abs, x_axis_abs);
        DEBUG_PRINTF("[TEST] Start condition met: |Z|=%.2f (>=%.1f) or |X|=%.2f (<=%.1f)\n", 
                     z_axis_abs, Z_AXIS_THRESHOLD, x_axis_abs, X_AXIS_THRESHOLD);
        DEBUG_PRINTF("[TEST] Start time: %lu ms\n", trialStartTime);
      }
    }

    // *** TRIAL STOP: |Z-axis| < 7.0 or |X-axis| > 5.0 ***
    // *** TRIAL STOP: |Z| < 0.7 AND |X| > 0.6 ***
    if (trialRunning) {
      if ((z_axis_abs < Z_AXIS_THRESHOLD) && (x_axis_abs > X_AXIS_THRESHOLD)) {
        // stop timer!
        trialRunning = false;
        unsigned long trialDuration = millis() - trialStartTime;

        DEBUG_PRINTLN("[TEST] *** TIMER STOPPED ***");
        DEBUG_PRINTF("[TEST] |Z-axis| = %.2f, |X-axis| = %.2f\n", z_axis_abs, x_axis_abs);
        DEBUG_PRINTF("[TEST] Stop condition met: |Z|=%.2f (<%.1f) AND |X|=%.2f (>%.1f)\n", 
                     z_axis_abs, Z_AXIS_THRESHOLD, x_axis_abs, X_AXIS_THRESHOLD);
        DEBUG_PRINTF("[TEST] Duration: %lu ms (%.2f seconds)\n", 
                     trialDuration, trialDuration / 1000.0);

void loop() {
    trialRunning = false;
    waitingForStart = false;
    DEBUG_PRINTLN("[BUTTON] Manual reset triggered!");
    delay(200);  // debounce
    delay(200); // debounce
  }

  // Small delay for stability, but not too long to block MQTT
  delay(50);  // 20 readings per second, allows MQTT to process
  delay(50); // 20 readings per second, allows MQTT to process
}
