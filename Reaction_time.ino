// Reaction Time Test with MQTT Integration + LCD Display
// Set DEBUG to 1 for troubleshooting, 0 for production/standalone
#define DEBUG 0  // Change to 1 to enable serial output

#if DEBUG
  #define DEBUG_PRINT(x) Serial.print(x)
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
const char* mqtt_broker = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_Reaction_Device";

// MQTT Topics
const char* TOPIC_CMD = "cs3237/A0277110N/health/cmd";
const char* TOPIC_DATA = "cs3237/A0277110N/health/data/Reaction_Test";

// Hardware pins
const int LED_PIN = 2;   
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
int test_count = 0;
unsigned long total_reaction_us = 0;
unsigned long test_times[TOTAL_TESTS];

// Participant metadata (received from server)
String participant_id = "UNKNOWN";
int participant_age = 0;
String participant_gender = "O";

// WiFi and MQTT clients
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// Interrupt Service Routine (ISR)
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

// WiFi connection function
void setup_wifi() {
    delay(10);
    DEBUG_PRINTLN("\n[WiFi] Connecting to WiFi...");
    DEBUG_PRINT("[WiFi] SSID: ");
    DEBUG_PRINTLN(ssid);
    
    lcd.clear();
    lcd.print("Connecting WiFi");
    
    WiFi.begin(ssid, password);
    
    int dots = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        DEBUG_PRINT(".");
        
        // Show progress on LCD
        lcd.setCursor(dots % 16, 1);
        lcd.print(".");
        dots++;
    }
    
    DEBUG_PRINTLN("\n[WiFi] Connected!");
    DEBUG_PRINT("[WiFi] IP Address: ");
    DEBUG_PRINTLN(WiFi.localIP());
    
    lcd.clear();
    lcd.print("WiFi Connected!");
    delay(1500);
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
        lcd.clear();
        lcd.print("JSON Error!");
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
            
            lcd.clear();
            lcd.print("Participant:");
            lcd.setCursor(0, 1);
            lcd.print(participant_id);
            lcd.print(" Age:");
            lcd.print(participant_age);
            delay(2000);
        }
        
        // Handle START_REACTION command
        else if (strcmp(cmd, "START_REACTION") == 0) {
            // Reset state completely for new test session
            waitingForStart = true;
            testRunning = false;
            test_count = 0;
            total_reaction_us = 0;
            led_active = false;
            button_pressed = false;
            
            DEBUG_PRINTLN("\n========================================");
            DEBUG_PRINTLN("[TEST] NEW REACTION TIME TEST ARMED!");
            DEBUG_PRINTF("[TEST] Will run %d trials\n", TOTAL_TESTS);
            DEBUG_PRINTLN("[TEST] Starting first trial...");
            DEBUG_PRINTLN("========================================\n");
            
            lcd.clear();
            lcd.print("Reaction Test");
            lcd.setCursor(0, 1);
            lcd.print("Starting...");
            delay(1000);
        }
        
        // Handle RESET command
        else if (strcmp(cmd, "RESET") == 0) {
            waitingForStart = false;
            testRunning = false;
            test_count = 0;
            total_reaction_us = 0;
            led_active = false;
            button_pressed = false;
            digitalWrite(LED_PIN, LOW);
            
            DEBUG_PRINTLN("[TEST] System reset received");
            
            lcd.clear();
            lcd.print("System Reset");
            delay(1000);
            lcd.clear();
            lcd.print("Ready");
        }
    }
}

// MQTT reconnection function
void reconnect_mqtt() {
    if (!mqtt_client.connected()) {
        DEBUG_PRINT("[MQTT] Attempting connection to ");
        DEBUG_PRINT(mqtt_broker);
        DEBUG_PRINT(":");
        DEBUG_PRINTLN(mqtt_port);
        
        lcd.clear();
        lcd.print("Connecting MQTT");
        
        if (mqtt_client.connect(mqtt_client_id)) {
            DEBUG_PRINTLN("[MQTT] Connected successfully!");
            
            // Subscribe to command topic
            mqtt_client.subscribe(TOPIC_CMD);
            DEBUG_PRINT("[MQTT] Subscribed to: ");
            DEBUG_PRINTLN(TOPIC_CMD);
            
            lcd.setCursor(0, 1);
            lcd.print("MQTT Connected!");
            delay(1000);
        } else {
            DEBUG_PRINT("[MQTT] Connection failed, rc=");
            DEBUG_PRINTLN(mqtt_client.state());
            DEBUG_PRINTLN("[MQTT] Will retry on next loop...");
            
            lcd.setCursor(0, 1);
            lcd.print("MQTT Failed!");
        }
    }
}

// Publish reaction test results to MQTT
void publish_reaction_results() {
    // Calculate average
    float average_time_us = (float)total_reaction_us / TOTAL_TESTS;
    float average_time_ms = average_time_us / 1000.0;
    
    StaticJsonDocument<512> doc;
    
    doc["participant_id"] = participant_id;
    doc["age"] = participant_age;
    doc["gender"] = participant_gender;
    doc["average_reaction_ms"] = average_time_ms;
    doc["test_type"] = "reaction";
    doc["num_trials"] = TOTAL_TESTS;
    doc["timestamp"] = millis();
    
    // Add individual trial times
    JsonArray trials = doc.createNestedArray("trials");
    for (int i = 0; i < TOTAL_TESTS; i++) {
        trials.add(test_times[i] / 1000.0);  // Convert to ms
    }
    
    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer);
    
    DEBUG_PRINTLN("\n[RESULT] Publishing reaction test results:");
    DEBUG_PRINT("         Average: ");
    DEBUG_PRINT(average_time_ms);
    DEBUG_PRINTLN(" ms");
    DEBUG_PRINT("         JSON: ");
    DEBUG_PRINTLN(jsonBuffer);
    
    // Display on LCD
    lcd.clear();
    lcd.print("Sending Data...");
    
    // Ensure MQTT is connected before publishing
    if (!mqtt_client.connected()) {
        DEBUG_PRINTLN("[RESULT] ERROR: MQTT not connected, reconnecting...");
        reconnect_mqtt();
    }
    
    if (mqtt_client.connected()) {
        bool published = mqtt_client.publish(TOPIC_DATA, jsonBuffer, false);
        
        if (published) {
            DEBUG_PRINTLN("[RESULT] Published successfully!");
            lcd.setCursor(0, 1);
            lcd.print("Data Sent!");
        } else {
            DEBUG_PRINTLN("[RESULT] Publish failed!");
            lcd.setCursor(0, 1);
            lcd.print("Send Failed!");
        }
    }
    
    delay(2000);
    
    // Process any incoming MQTT messages immediately
    mqtt_client.loop();
}

void setup() {
    #if DEBUG
        Serial.begin(115200);
        delay(500);
        DEBUG_PRINTLN("\n\n========================================");
        DEBUG_PRINTLN("Reaction Time Test - MQTT Integration");
        DEBUG_PRINTLN("Multi-Trial Fixed Version");
        DEBUG_PRINTLN("========================================\n");
    #endif

    // Setup LCD
    Wire.begin(I2C_SDA, I2C_SCL);
    lcd.init();
    lcd.backlight();
    lcd.clear();
    lcd.print("Reaction Test");
    lcd.setCursor(0, 1);
    lcd.print("Initializing...");
    delay(1000);

    // Setup hardware
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    digitalWrite(LED_PIN, LOW);
    
    // Attach interrupt
    attachInterrupt(BUTTON_PIN, buttonISR, FALLING);
    DEBUG_PRINTLN("[HARDWARE] Button interrupt configured");

    // Setup WiFi
    setup_wifi();

    // Setup MQTT with larger keepalive
    mqtt_client.setServer(mqtt_broker, mqtt_port);
    mqtt_client.setCallback(mqtt_callback);
    mqtt_client.setKeepAlive(60);  // 60 second keepalive
    DEBUG_PRINTLN("[MQTT] Client configured");

    // Initial connection
    reconnect_mqtt();

    DEBUG_PRINTLN("\n========================================");
    DEBUG_PRINTLN("Setup complete - System ready!");
    DEBUG_PRINTLN("Waiting for START_REACTION command...");
    DEBUG_PRINTLN("========================================\n");
    
    lcd.clear();
    lcd.print("System Ready");
    lcd.setCursor(0, 1);
    lcd.print("Awaiting Cmd...");
    
    delay(1000);
}

void loop() {
    // Always maintain MQTT connection (critical!)
    if (!mqtt_client.connected()) {
        reconnect_mqtt();
    }
    
    // Process MQTT messages frequently - don't block!
    mqtt_client.loop();

    // Only run test logic if START_REACTION command received
    if (waitingForStart || testRunning) {
        
        // Check if all tests completed
        if (test_count >= TOTAL_TESTS) {
            // All tests done - show results on LCD
            DEBUG_PRINTLN("\n[TEST] *** ALL TRIALS COMPLETE ***");
            
            float average_ms = ((float)total_reaction_us / TOTAL_TESTS) / 1000.0;
            DEBUG_PRINTF("[TEST] Average reaction time: %.2f ms\n", average_ms);
            
            // Display final results on LCD
            lcd.clear();
            lcd.print("Test Complete!");
            lcd.setCursor(0, 1);
            lcd.printf("Avg: %.1f ms", average_ms);
            delay(3000);
            
            // Display individual results
            DEBUG_PRINTLN("[TEST] Individual trial results:");
            for (int i = 0; i < TOTAL_TESTS; i++) {
                DEBUG_PRINTF("       Trial %d: %.2f ms\n", i+1, test_times[i] / 1000.0);
                
                lcd.clear();
                lcd.printf("Trial %d:", i+1);
                lcd.setCursor(0, 1);
                lcd.printf("%.1f ms", test_times[i] / 1000.0);
                delay(1500);
            }
            
            // Publish results to MQTT
            publish_reaction_results();
            
            // Reset to idle state
            waitingForStart = false;
            testRunning = false;
            
            DEBUG_PRINTLN("\n[TEST] Session complete. Ready for next START_REACTION command.");
            DEBUG_PRINTLN("========================================\n");
            
            lcd.clear();
            lcd.print("Session Done");
            lcd.setCursor(0, 1);
            lcd.print("Awaiting Cmd...");
        }
        
        // Start a new trial if ready
        else if (!led_active && !button_pressed && !testRunning) {
            DEBUG_PRINTF("\n[TEST] === Starting Trial %d of %d ===\n", test_count + 1, TOTAL_TESTS);
            DEBUG_PRINTLN("[TEST] Random delay before LED...");
            
            // Show trial info on LCD
            lcd.clear();
            lcd.printf("Trial %d of %d", test_count + 1, TOTAL_TESTS);
            lcd.setCursor(0, 1);
            lcd.print("Get Ready...");
            
            // Random delay before showing LED
            unsigned long random_delay = random(1000, 5000);
            delay(random_delay);
            
            // Turn on LED and start timing
            digitalWrite(LED_PIN, HIGH);
            led_active = true;
            start_time_us = micros();
            testRunning = true;
            
            DEBUG_PRINTLN("[TEST] LED ON - Press button now!");
            
            // Update LCD
            lcd.setCursor(0, 1);
            lcd.print("PRESS NOW!      ");
        }
        
        // Handle button press
        if (button_pressed) {
            test_count++;
            test_times[test_count - 1] = reaction_time_us;
            total_reaction_us += reaction_time_us;
            
            float reaction_ms = (float)reaction_time_us / 1000.0;
            
            DEBUG_PRINTF("[TEST] Trial %d complete: %.2f ms\n", test_count, reaction_ms);
            
            // Display result on LCD
            lcd.clear();
            lcd.printf("Trial %d Done", test_count);
            lcd.setCursor(0, 1);
            lcd.printf("%.1f ms", reaction_ms);
            
            // Reset state for next trial
            testRunning = false;
            button_pressed = false;
            reaction_time_us = 0;
            
            // Short delay before next trial
            delay(2000);
        }
        
        // Timeout check - if LED has been on for more than 5 seconds, abort trial
        if (led_active && (micros() - start_time_us) > 5000000) {
            DEBUG_PRINTLN("[TEST] Trial timeout - no response!");
            
            digitalWrite(LED_PIN, LOW);
            led_active = false;
            testRunning = false;
            
            // Display timeout on LCD
            lcd.clear();
            lcd.print("Trial Timeout!");
            lcd.setCursor(0, 1);
            lcd.print("Too slow!");
            
            // Record timeout as max value (5000ms)
            test_times[test_count] = 5000000;
            total_reaction_us += 5000000;
            test_count++;
            
            delay(2000);
        }
    }

    delay(10);  // Small delay for stability
}
