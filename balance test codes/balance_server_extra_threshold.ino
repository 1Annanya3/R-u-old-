// Balance Test with MPU6050 + MQTT Integration (Multi‑Trial)
// DEBUG: 1 for serial logs, 0 for silent
#define DEBUG 1

#if DEBUG
  #define DEBUG_PRINT(x)   Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
  #define DEBUG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
  #define DEBUG_PRINTF(...)
#endif

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ================== WiFi / MQTT ==================
const char* ssid         = "ANNANYA";
const char* password     = "_______";   // TODO: fill
const char* mqtt_broker  = "broker.hivemq.com";
const int   mqtt_port    = 1883;
const char* mqtt_client_id = "ESP32_Balance_Device";

const char* TOPIC_CMD    = "cs3237/A0277110N/health/cmd";
const char* TOPIC_DATA   = "cs3237/A0277110N/health/data/Balance_Test";

// ================== Hardware ==================
Adafruit_MPU6050 mpu;
const int RESET_BUTTON_PIN = 15;

// ================== Balance thresholds ==================
// Start when |Z| >= 0.7 g OR |X| <= 0.6 g
// Stop  when |Z| <  0.7 g AND |X| >  0.6 g
const float Z_AXIS_THRESHOLD = 0.7f;
const float X_AXIS_THRESHOLD = 0.6f;

// ================== State ==================
bool waitingForStart = false;
bool trialRunning    = false;
unsigned long trialStartTime = 0;
unsigned long lastAccelDebug = 0;
const unsigned long ACCEL_DEBUG_INTERVAL = 3000; // ms
const unsigned long LOOP_DELAY_MS = 50;

// Metadata (must come from server)
String participant_id;         // empty until SET_META
int    participant_age = -1;   // -1 means unknown
String participant_gender;     // empty until SET_META
bool   meta_ready = false;

// Networking
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// ================== MQTT helpers ==================
void reconnect_mqtt() {
  if (mqtt_client.connected()) return;
  DEBUG_PRINT("[MQTT] Connecting... ");
  if (mqtt_client.connect(mqtt_client_id)) {
    DEBUG_PRINTLN("OK");
    mqtt_client.subscribe(TOPIC_CMD);
    DEBUG_PRINT("[MQTT] Subscribed: "); DEBUG_PRINTLN(TOPIC_CMD);
  } else {
    DEBUG_PRINTLN("FAILED");
  }
}

void publish_balance_result(unsigned long duration_ms) {
  StaticJsonDocument<384> doc;
  doc["participant_id"] = participant_id;
  doc["age"]            = participant_age;
  doc["gender"]         = participant_gender;
  doc["duration_ms"]    = duration_ms;
  doc["test_type"]      = "balance";
  doc["timestamp"]      = millis();

  char jsonBuffer[384];
  serializeJson(doc, jsonBuffer);

  DEBUG_PRINTLN("\n[RESULT] Publishing balance test result:");
  DEBUG_PRINT("         Duration: "); DEBUG_PRINT(duration_ms); DEBUG_PRINTLN(" ms");
  DEBUG_PRINT("         JSON: ");     DEBUG_PRINTLN(jsonBuffer);

  if (!mqtt_client.connected()) reconnect_mqtt();
  if (mqtt_client.connected()) {
    bool ok = mqtt_client.publish(TOPIC_DATA, jsonBuffer, false);
    DEBUG_PRINTLN(ok ? "[RESULT] Published" : "[RESULT] Publish failed");
  }
}

// ================== MQTT callback ==================
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    DEBUG_PRINT("[MQTT] JSON error: "); DEBUG_PRINTLN(err.c_str());
    return;
  }

  const char* cmd = doc["cmd"];
  if (!cmd) return;

  if (strcmp(cmd, "SET_META") == 0) {
    participant_id     = doc["participant_id"].as<String>();
    participant_age    = doc["age"].isNull() ? -1 : doc["age"].as<int>();
    participant_gender = doc["gender"].as<String>();
    meta_ready = (participant_id.length() > 0) && (participant_age >= 0) && (participant_gender.length() > 0);

    DEBUG_PRINTLN("[META] Received");
    DEBUG_PRINT("  ID: "); DEBUG_PRINTLN(participant_id);
    DEBUG_PRINT("  Age: "); DEBUG_PRINTLN(participant_age);
    DEBUG_PRINT("  Gender: "); DEBUG_PRINTLN(participant_gender);
    return;
  }

  if (strcmp(cmd, "START_BALANCE") == 0) {
    if (!meta_ready) {
      DEBUG_PRINTLN("[TEST] START blocked: metadata not set");
      return;
    }
    waitingForStart = true;
    trialRunning    = false;
    DEBUG_PRINTLN("\n========================================");
    DEBUG_PRINTLN("[TEST] NEW BALANCE TEST ARMED!");
    DEBUG_PRINTF("[TEST] Start when |Z|>=%.1f OR |X|<=%.1f\n", Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
    DEBUG_PRINTF("[TEST] Stop  when |Z|<%.1f  AND |X|>%.1f\n",  Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
    DEBUG_PRINTLN("========================================\n");
    return;
  }

  if (strcmp(cmd, "RESET") == 0) {
    waitingForStart = false;
    trialRunning    = false;
    DEBUG_PRINTLN("[TEST] System reset");
    return;
  }
}

// ================== Setup / Loop ==================
void setup() {
  #if DEBUG
    Serial.begin(115200); delay(300);
  #endif

  // WiFi
  DEBUG_PRINTLN("[WiFi] Connecting...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(200); }
  DEBUG_PRINT("[WiFi] IP: "); DEBUG_PRINTLN(WiFi.localIP());

  // MQTT
  mqtt_client.setServer(mqtt_broker, mqtt_port);
  mqtt_client.setCallback(mqtt_callback);
  mqtt_client.setKeepAlive(60);
  reconnect_mqtt();

  // Button
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);

  // IMU
  if (!mpu.begin()) {
    DEBUG_PRINTLN("[MPU6050] Not found!");
    while (1) delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  DEBUG_PRINTLN("\n========================================");
  DEBUG_PRINTLN("Setup complete - Awaiting commands");
  DEBUG_PRINTF("Start: |Z|>=%.1f OR |X|<=%.1f\n", Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
  DEBUG_PRINTF("Stop : |Z|<%.1f AND |X|>%.1f\n",  Z_AXIS_THRESHOLD, X_AXIS_THRESHOLD);
  DEBUG_PRINTLN("Send SET_META then START_BALANCE");
  DEBUG_PRINTLN("========================================\n");
}

void loop() {
  if (!mqtt_client.connected()) reconnect_mqtt();
  mqtt_client.loop();

  // Manual reset
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
    trialRunning = false;
    waitingForStart = false;
    DEBUG_PRINTLN("[BUTTON] Manual reset");
    delay(200);
  }

  // Read accelerometer
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float z_abs = fabsf(a.acceleration.z);
  float x_abs = fabsf(a.acceleration.x);

  // Periodic accel debug
  if (DEBUG && (millis() - lastAccelDebug >= ACCEL_DEBUG_INTERVAL)) {
    DEBUG_PRINT("[ACCEL] X: "); DEBUG_PRINT(a.acceleration.x);
    DEBUG_PRINT("  Y: ");       DEBUG_PRINT(a.acceleration.y);
    DEBUG_PRINT("  Z: ");       DEBUG_PRINT(a.acceleration.z);
    DEBUG_PRINTF(" | |Z|=%.2f, |X|=%.2f\n", z_abs, x_abs);
    lastAccelDebug = millis();
  }

  // Run logic only after START_BALANCE
  if (waitingForStart || trialRunning) {
    // Start condition
    if (!trialRunning && waitingForStart) {
      if ((z_abs >= Z_AXIS_THRESHOLD) || (x_abs <= X_AXIS_THRESHOLD)) {
        trialRunning = true;
        waitingForStart = false;
        trialStartTime = millis();
        DEBUG_PRINTLN("[TEST] *** TIMER STARTED ***");
        DEBUG_PRINTF("[TEST] |Z|=%.2f, |X|=%.2f\n", z_abs, x_abs);
      }
    }

    // Stop condition
    if (trialRunning) {
      if ((z_abs < Z_AXIS_THRESHOLD) && (x_abs > X_AXIS_THRESHOLD)) {
        trialRunning = false;
        unsigned long duration = millis() - trialStartTime;
        DEBUG_PRINTLN("[TEST] *** TIMER STOPPED ***");
        DEBUG_PRINTF("[TEST] Duration: %lu ms (%.2f s)\n", duration, duration/1000.0);
        publish_balance_result(duration);
      }
    }
  }

  delay(LOOP_DELAY_MS); // keep MQTT responsive
}
