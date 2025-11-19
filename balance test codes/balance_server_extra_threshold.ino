// Reaction Time Test with MQTT Integration + LCD Display
// Set DEBUG to 1 for troubleshooting, 0 for production/standalone
#define DEBUG 0  // Change to 1 to enable serial output

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
#include <LiquidCrystal_I2C.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// WiFi Configuration
const char* ssid = "ANNANYA";
const char* password = ".";

// MQTT Configuration
const char* mqtt_broker   = "broker.hivemq.com";
const int   mqtt_port     = 1883;
const char* mqtt_client_id= "ESP32_Reaction_Device";

// MQTT Topics
const char* TOPIC_CMD  = "cs3237/A0277110N/health/cmd";
const char* TOPIC_DATA = "cs3237/A0277110N/health/data/Reaction_Test";

// Hardware pins
const int LED_PIN  = 2;
const int BUTTON_PIN = 4;
const int I2C_SDA = 21;
const int I2C_SCL = 22;

// LCD Object
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Test configuration
const int TOTAL_TESTS = 5;
const unsigned long DEBOUNCE_US = 50000;  // 50ms debounce

// Timing and state variables
volatile bool led_active = false;
volatile bool button_pressed = false;
volatile unsigned long start_time_us = 0;
volatile unsigned long reaction_time_us = 0;
volatile unsigned long last_isr_us = 0;

// Trial state
bool waitingForStart = false;
bool testRunning = false;
int  test_count = 0;
unsigned long total_reaction_us = 0;
unsigned long test_times[TOTAL_TESTS];

// Participant metadata (must come from server)
String participant_id;         // empty until SET_META received
int    participant_age = -1;   // -1 means unknown
String participant_gender;     // empty until SET_META
bool   meta_ready = false;     // gate to ensure meta received

// WiFi and MQTT clients
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// ISR
void IRAM_ATTR buttonISR() {
  unsigned long now_us = micros();
  if ((now_us - last_isr_us) > DEBOUNCE_US) {
    last_isr_us = now_us;
    if (led_active) {
      reaction_time_us = now_us - start_time_us;
      button_pressed = true;
      digitalWrite(LED_PIN, LOW);
      led_active = false;
    }
  }
}

// WiFi
void setup_wifi() {
  delay(10);
  #if DEBUG
    Serial.println("\n[WiFi] Connecting...");
    Serial.print("[WiFi] SSID: "); Serial.println(ssid);
  #endif
  lcd.clear(); lcd.print("Connecting WiFi");
  WiFi.begin(ssid, password);
  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    lcd.setCursor(dots % 16, 1); lcd.print(".");
    dots++;
  }
  lcd.clear(); lcd.print("WiFi Connected!"); delay(1000);
}

// MQTT callback
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, payload, length);
  if (error) {
    DEBUG_PRINTLN(String("[MQTT] JSON parse error: ") + error.c_str());
    lcd.clear(); lcd.print("JSON Error!"); return;
  }

  const char* cmd = doc["cmd"];
  if (!cmd) return;

  // Receive participant metadata from server
  if (strcmp(cmd, "SET_META") == 0) {
    participant_id     = doc["participant_id"].as<String>();
    participant_age    = doc["age"].isNull() ? -1 : doc["age"].as<int>();
    participant_gender = doc["gender"].as<String>();
    meta_ready = (participant_id.length() > 0) && (participant_age >= 0) && (participant_gender.length() > 0);

    DEBUG_PRINTLN("[META] Received from server");
    DEBUG_PRINT("[META] ID: "); DEBUG_PRINTLN(participant_id);
    DEBUG_PRINT("[META] Age: "); DEBUG_PRINTLN(participant_age);
    DEBUG_PRINT("[META] Gender: "); DEBUG_PRINTLN(participant_gender);

    lcd.clear();
    lcd.print("ID:"); lcd.print(participant_id);
    lcd.setCursor(0,1);
    lcd.print("Age:"); lcd.print(participant_age);
    delay(1200);
    lcd.clear();
    lcd.print(meta_ready ? "Meta OK" : "Meta Missing");
    return;
  }

  // Start command: only if meta_ready
  if (strcmp(cmd, "START_REACTION") == 0) {
    if (!meta_ready) {
      DEBUG_PRINTLN("[TEST] START blocked: meta not set");
      lcd.clear(); lcd.print("Set meta first"); return;
    }
    waitingForStart = true;
    testRunning = false;
    test_count = 0;
    total_reaction_us = 0;
    led_active = false;
    button_pressed = false;

    lcd.clear(); lcd.print("Reaction Test");
    lcd.setCursor(0,1); lcd.print("Starting...");
    delay(800);
    return;
  }

  // Reset
  if (strcmp(cmd, "RESET") == 0) {
    waitingForStart = false;
    testRunning = false;
    test_count = 0;
    total_reaction_us = 0;
    led_active = false;
    button_pressed = false;
    digitalWrite(LED_PIN, LOW);
    lcd.clear(); lcd.print("System Reset");
    delay(800); lcd.clear(); lcd.print(meta_ready ? "Ready" : "Awaiting meta");
    return;
  }
}

// MQTT connect
void reconnect_mqtt() {
  if (mqtt_client.connected()) return;
  lcd.clear(); lcd.print("Connecting MQTT");
  if (mqtt_client.connect(mqtt_client_id)) {
    mqtt_client.subscribe(TOPIC_CMD);
    lcd.setCursor(0,1); lcd.print("MQTT Connected");
  } else {
    lcd.setCursor(0,1); lcd.print("MQTT Failed");
  }
}

// Publish results
void publish_reaction_results() {
  float average_time_us = (float)total_reaction_us / TOTAL_TESTS;
  float average_time_ms = average_time_us / 1000.0;

  StaticJsonDocument<512> doc;
  doc["participant_id"]     = participant_id;
  doc["age"]                = participant_age;
  doc["gender"]             = participant_gender;
  doc["average_reaction_ms"]= average_time_ms;
  doc["test_type"]          = "reaction";
  doc["num_trials"]         = TOTAL_TESTS;
  doc["timestamp"]          = millis();
  JsonArray trials = doc.createNestedArray("trials");
  for (int i=0;i<TOTAL_TESTS;i++) trials.add(test_times[i]/1000.0);

  char jsonBuffer[512];
  serializeJson(doc, jsonBuffer);

  if (!mqtt_client.connected()) reconnect_mqtt();
  if (mqtt_client.connected()) {
    bool ok = mqtt_client.publish(TOPIC_DATA, jsonBuffer, false);
    lcd.clear(); lcd.print(ok ? "Data Sent!" : "Send Failed!");
  }
  delay(1200);
  mqtt_client.loop();
}

void setup() {
  #if DEBUG
    Serial.begin(115200); delay(300);
  #endif
  Wire.begin(I2C_SDA, I2C_SCL);
  lcd.init(); lcd.backlight();
  lcd.clear(); lcd.print("Reaction Test");
  lcd.setCursor(0,1); lcd.print("Initializing...");
  delay(800);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(LED_PIN, LOW);
  attachInterrupt(BUTTON_PIN, buttonISR, FALLING);

  setup_wifi();
  mqtt_client.setServer(mqtt_broker, mqtt_port);
  mqtt_client.setCallback(mqtt_callback);
  mqtt_client.setKeepAlive(60);
  reconnect_mqtt();

  lcd.clear(); lcd.print("Awaiting meta");
}

void loop() {
  if (!mqtt_client.connected()) reconnect_mqtt();
  mqtt_client.loop();

  // Do nothing until metadata is set and a START is received
  if (!(waitingForStart || testRunning)) { delay(10); return; }

  // All tests complete
  if (test_count >= TOTAL_TESTS) {
    float average_ms = ((float)total_reaction_us / TOTAL_TESTS) / 1000.0;
    lcd.clear(); lcd.print("Test Complete!");
    lcd.setCursor(0,1); lcd.printf("Avg: %.1f ms", average_ms);
    delay(1500);

    publish_reaction_results();

    waitingForStart = false; testRunning = false;
    lcd.clear(); lcd.print("Session Done");
    lcd.setCursor(0,1); lcd.print("Awaiting cmd...");
    delay(700);
    return;
  }

  // Start a new trial
  if (!led_active && !button_pressed && !testRunning) {
    lcd.clear(); lcd.printf("Trial %d/%d", test_count + 1, TOTAL_TESTS);
    lcd.setCursor(0,1); lcd.print("Get Ready...");
    unsigned long random_delay = random(1000, 5000);
    delay(random_delay);

    digitalWrite(LED_PIN, HIGH);
    led_active = true;
    start_time_us = micros();
    testRunning = true;
    lcd.setCursor(0,1); lcd.print("PRESS NOW!    ");
  }

  // Handle button press
  if (button_pressed) {
    test_count++;
    test_times[test_count - 1] = reaction_time_us;
    total_reaction_us += reaction_time_us;

    float reaction_ms = reaction_time_us / 1000.0;
    lcd.clear(); lcd.printf("Trial %d Done", test_count);
    lcd.setCursor(0,1); lcd.printf("%.1f ms", reaction_ms);

    testRunning = false;
    button_pressed = false;
    reaction_time_us = 0;
    delay(1200);
  }

  // Timeout after 5s
  if (led_active && (micros() - start_time_us) > 5000000) {
    digitalWrite(LED_PIN, LOW);
    led_active = false;
    testRunning = false;
    lcd.clear(); lcd.print("Trial Timeout");
    lcd.setCursor(0,1); lcd.print("Too slow!");
    test_times[test_count] = 5000000;
    total_reaction_us += 5000000;
    test_count++;
    delay(1200);
  }

  delay(10);
}
