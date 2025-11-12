// Balance Test with MPU6050 + MQTT Integration (Multi-Trial Fixed)
// Set DEBUG to 1 for troubleshooting, 0 for production/standalone
#define DEBUG 1  // Change to 0 to disable all serial output

#if DEBUG
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
  #define DEBUG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
  #define DEBUG_PRINTF(...)
#endif

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// WiFi Configuration
const char* ssid = "__";
const char* password = "__";

// MQTT Configuration
const char* mqtt_broker = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_Balance_Device";

// MQTT Topics
const char* TOPIC_CMD = "cs3237/A0277110N/health/cmd";
const char* TOPIC_DATA = "cs3237/A0277110N/health/data/Balance_Test";

// Sensor and hardware
Adafruit_MPU6050 mpu;
const int RESET_BUTTON_PIN = 15;

// Balance detection thresholds - Z-axis modulus (absolute value)
const float Z_AXIS_THRESHOLD = 7.0;  // |Z| >= 7.0 to START, |Z| < 7.0 to STOP
const float X_AXIS_THRESHOLD = 6.0;   // |X| <= 5.0 to START, |X| > 5.0 to STOP

// Trial state variables
bool trialRunning = false;
bool waitingForStart = false;
unsigned long trialStartTime = 0;
unsigned long lastMqttCheck = 0;
const unsigned long MQTT_CHECK_INTERVAL = 50;  // Check MQTT every 50ms

// Participant metadata (received from server)
String participant_id = "UNKNOWN";
int participant_age = 0;
String participant_gender = "O";

// WiFi and MQTT clients
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// WiFi connection function
void setup_wifi() {
  delay(10);
  DEBUG_PRINTLN("\n[WiFi] Connecting to WiFi...");
  DEBUG_PRINT("[WiFi] SSID: ");
  DEBUG_PRINTLN(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    DEBUG_PRINT(".");
  }
  
  DEBUG_PRINTLN("\n[WiFi] Connected!");
  DEBUG_PRINT("[WiFi] IP Address: ");
  DEBUG_PRINTLN(WiFi.localIP());
}

// MQTT callback for incoming messages
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  DEBUG_PRINT("\n[MQTT] Message received on topic: ");
  DEBUG_PRINTLN(topic);
  
  // Parse JSON payload
  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, payload, length);

  if (error) {
    DEBUG_PRINT("[MQTT] JSON parse error: ");
    DEBUG_PRINTLN(error.c_str());
    return;
  }

  const char* cmd = doc["cmd"];
  
  if (cmd != nullptr) {
    DEBUG_PRINT("[MQTT] Command received: ");
    DEBUG_PRINTLN(cmd);

    // Handle SET_META command - store participant info
    if (strcmp(cmd, "SET_META") == 0) {
      participant_id = doc["participant_id"].as<String>();
      participant_age = doc["age"];
      participant_gender = doc["gender"].as<String>();
      
      DEBUG_PRINTLN("[META] Participant metadata received:");
      DEBUG_PRINT("       ID: ");
      DEBUG_PRINTLN(participant_id);
      DEBUG_PRINT("       Age: ");
      DEBUG_PRINTLN(participant_age);
      DEBUG_PRINT("       Gender: ");
      DEBUG_PRINTLN(participant_gender);
    }
    
    // Handle START_BALANCE command
    else if (strcmp(cmd, "START_BALANCE") == 0) {
      // Reset state completely for new trial
      trialRunning = false;
      waitingForStart = true;
      trialStartTime = 0;
      
      DEBUG_PRINTLN("\n========================================");
      DEBUG_PRINTLN("[TEST] NEW BALANCE TEST ARMED!");
      DEBUG_PRINTF("[TEST] Waiting for |Z-axis| >= %.1f to start timer...\n", Z_AXIS_THRESHOLD);
      DEBUG_PRINTLN("[TEST] (Lift your leg to start)");
      DEBUG_PRINTLN("========================================\n");
    }
    
    // Handle RESET command
    else if (strcmp(cmd, "RESET") == 0) {
      trialRunning = false;
      waitingForStart = false;
      trialStartTime = 0;
      DEBUG_PRINTLN("[TEST] System reset received");
    }
  }
}

// MQTT reconnection function
void reconnect_mqtt() {
  // Only try reconnecting if not currently connected
  if (!mqtt_client.connected()) {
    DEBUG_PRINT("[MQTT] Attempting connection to ");
    DEBUG_PRINT(mqtt_broker);
    DEBUG_PRINT(":");
    DEBUG_PRINTLN(mqtt_port);
    
    if (mqtt_client.connect(mqtt_client_id)) {
      DEBUG_PRINTLN("[MQTT] Connected successfully!");
      
      // Subscribe to command topic
      mqtt_client.subscribe(TOPIC_CMD);
      DEBUG_PRINT("[MQTT] Subscribed to: ");
      DEBUG_PRINTLN(TOPIC_CMD);
    } else {
      DEBUG_PRINT("[MQTT] Connection failed, rc=");
      DEBUG_PRINTLN(mqtt_client.state());
      DEBUG_PRINTLN("[MQTT] Will retry on next loop...");
    }
  }
}

// Publish balance test results to MQTT
void publish_balance_result(unsigned long duration_ms) {
  StaticJsonDocument<256> doc;
  
  doc["participant_id"] = participant_id;
  doc["age"] = participant_age;
  doc["gender"] = participant_gender;
  doc["duration_ms"] = duration_ms;
  doc["test_type"] = "balance";
  doc["timestamp"] = millis();
  
  char jsonBuffer[256];
  serializeJson(doc, jsonBuffer);
  
  DEBUG_PRINTLN("\n[RESULT] Publishing balance test result:");
  DEBUG_PRINT("         Duration: ");
  DEBUG_PRINT(duration_ms);
  DEBUG_PRINTLN(" ms");
  DEBUG_PRINT("         JSON: ");
  DEBUG_PRINTLN(jsonBuffer);
  
  // Ensure MQTT is connected before publishing
  if (!mqtt_client.connected()) {
    DEBUG_PRINTLN("[RESULT] ERROR: MQTT not connected, reconnecting...");
    reconnect_mqtt();
  }
  
  if (mqtt_client.connected()) {
    bool published = mqtt_client.publish(TOPIC_DATA, jsonBuffer, false);
    
    if (published) {
      DEBUG_PRINTLN("[RESULT] Published successfully!");
    } else {
      DEBUG_PRINTLN("[RESULT] Publish failed!");
    }
  }
  
  // Process any incoming MQTT messages immediately
  mqtt_client.loop();
}

void setup(void) {
  #if DEBUG
    Serial.begin(115200);
    delay(500);
    DEBUG_PRINTLN("\n\n========================================");
    DEBUG_PRINTLN("Balance Test System - Z-Axis Modulus");
    DEBUG_PRINTLN("Multi-Trial Fixed Version");
    DEBUG_PRINTLN("========================================\n");
  #endif

  // Setup WiFi
  setup_wifi();

  // Setup MQTT with larger keepalive
  mqtt_client.setServer(mqtt_broker, mqtt_port);
  mqtt_client.setCallback(mqtt_callback);
  mqtt_client.setKeepAlive(60);  // 60 second keepalive
  DEBUG_PRINTLN("[MQTT] Client configured");

  // Try to initialize MPU6050
  DEBUG_PRINTLN("[MPU6050] Initializing sensor...");
  if (!mpu.begin()) {
    DEBUG_PRINTLN("[MPU6050] ERROR: Failed to find MPU6050 chip!");
    while (1) {
      delay(10);
    }
  }
  DEBUG_PRINTLN("[MPU6050] Sensor found!");

  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);
  DEBUG_PRINT("[GPIO] Reset button configured on pin ");
  DEBUG_PRINTLN(RESET_BUTTON_PIN);
  
  // Configure MPU6050
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  DEBUG_PRINTLN("[MPU6050] Accelerometer range: ±8G");
  
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  DEBUG_PRINTLN("[MPU6050] Gyroscope range: ±500 deg/s");
  
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  DEBUG_PRINTLN("[MPU6050] Filter bandwidth: 21 Hz");

  DEBUG_PRINTLN("\n========================================");
  DEBUG_PRINTLN("Setup complete - System ready!");
  DEBUG_PRINTF("Threshold: |Z-axis| >= %.1f starts, < %.1f stops\n", 
               Z_AXIS_THRESHOLD, Z_AXIS_THRESHOLD);
  DEBUG_PRINTLN("Waiting for START_BALANCE command...");
  DEBUG_PRINTLN("========================================\n");
  
  delay(100);
}

void loop() {
  // Always maintain MQTT connection (critical!)
  if (!mqtt_client.connected()) {
    reconnect_mqtt();
  }
  
  // Process MQTT messages frequently - don't block!
  mqtt_client.loop();

  // Get sensor readings
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Calculate absolute value (modulus) of Z-axis acceleration
  float z_axis_abs = abs(a.acceleration.z);
  float x_axis_abs = abs(a.acceleration.x);

  // Only run balance detection if START_BALANCE command received
  if (waitingForStart && trialRunning) {
    
    // *** TRIAL START: |Z-axis| >= 7.0 or |X-axis| <= 5.0 ***
    if (!trialRunning && waitingForStart) {
      if ((z_axis_abs >= Z_AXIS_THRESHOLD) || (x_axis_abs <= X_AXIS_THRESHOLD)){
        // start timer!
        trialRunning = true;
        waitingForStart = false;  // No longer waiting
        trialStartTime = millis();
        
        DEBUG_PRINTLN("[TEST] *** TIMER STARTED ***");
        DEBUG_PRINTF("[TEST] |Z-axis| = %.2f (threshold: >= %.1f)\n", 
                     z_axis_abs, Z_AXIS_THRESHOLD);
        DEBUG_PRINTF("[TEST] |Z-axis| = %.2f, |X-axis| = %.2f\n", z_axis_abs, x_axis_abs);
        DEBUG_PRINTF("[TEST] Start time: %lu ms\n", trialStartTime);
      }
    }

    // *** TRIAL STOP: |Z-axis| < 7.0 or |X-axis| > 5.0 ***
    if (trialRunning) {
      if ((z_axis_abs < Z_AXIS_THRESHOLD) && (x_axis_abs > X_AXIS_THRESHOLD)) {
        // stop timer!
        trialRunning = false;
        unsigned long trialDuration = millis() - trialStartTime;
        
        DEBUG_PRINTLN("[TEST] *** TIMER STOPPED ***");
        DEBUG_PRINTF("[TEST] |Z-axis| = %.2f, |X-axis| = %.2f\n", z_axis_abs, x_axis_abs);
        DEBUG_PRINTF("[TEST] Duration: %lu ms (%.2f seconds)\n", 
                     trialDuration, trialDuration / 1000.0);
        
        // Publish result to MQTT
        publish_balance_result(trialDuration);
        
        // IMPORTANT: Don't set waitingForStart here!
        // Let the server send START_BALANCE again for next trial
        DEBUG_PRINTLN("\n[TEST] Trial complete. Ready for next START_BALANCE command.");
        DEBUG_PRINTLN("========================================\n");
      }
    }
  }

  // Manual reset button
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
    trialRunning = false;
    waitingForStart = false;
    DEBUG_PRINTLN("[BUTTON] Manual reset triggered!");
    delay(200);  // debounce
  }

  // Small delay for stability, but not too long to block MQTT
  delay(50);  // 20 readings per second, allows MQTT to process
}
