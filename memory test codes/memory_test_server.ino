// Memory Test with MQTT Integration (Capacitive Touch Sequence)
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

#include <Wire.h>
#include <Adafruit_MPR121.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#ifndef _BV
#define _BV(bit) (1 << (bit)) 
#endif

// WiFi Configuration
const char* ssid = "ANNANYA";
const char* password = ".";

// MQTT Configuration
const char* mqtt_broker = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_Memory_Device";

// MQTT Topics
const char* TOPIC_CMD = "cs3237/A0277110N/health/cmd";
const char* TOPIC_DATA = "cs3237/A0277110N/health/data/Memory_Test";

// MPR121 Touch Sensor
Adafruit_MPR121 cap = Adafruit_MPR121();
uint16_t lasttouched = 0;
uint16_t currtouched = 0;

// Game Configuration
#define NUM_BUTTONS 10
#define TRIALS_PER_SPAN 3
#define MAX_ERRORS 2

// Hardware Pins
int ledPins[NUM_BUTTONS] = {12, 13, 14, 15, 16, 17, 18, 19, 26, 27};
#define GREEN_LED 33
#define RED_LED   32
#define SIGNAL_LED 4
#define START_BUTTON 23

// Game State Variables
int sequence[10];
int userSeq[10];
int span = 3;
int trialCount = 0;
int errorsAtSpan = 0;
int score = 0;
bool testRunning = false;
bool waitingForStart = false;
volatile bool startButtonPressed = false;

// Participant metadata (received from server)
String participant_id = "UNKNOWN";
int participant_age = 0;
String participant_gender = "O";

// WiFi and MQTT clients
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// Start button interrupt
void IRAM_ATTR toggleTest() {
    startButtonPressed = true;
}

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

        // Handle SET_META command
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
        
        // Handle START_MEMORY command
        else if (strcmp(cmd, "START_MEMORY") == 0) {
            waitingForStart = true;
            testRunning = false;
            score = 0;
            span = 3;
            trialCount = 0;
            errorsAtSpan = 0;
            
            DEBUG_PRINTLN("\n========================================");
            DEBUG_PRINTLN("[TEST] NEW MEMORY TEST ARMED!");
            DEBUG_PRINTLN("[TEST] Press START button to begin");
            DEBUG_PRINTLN("========================================\n");
            
            // Visual feedback - blink signal LED
            for (int i = 0; i < 3; i++) {
                digitalWrite(SIGNAL_LED, HIGH);
                delay(200);
                digitalWrite(SIGNAL_LED, LOW);
                delay(200);
            }
        }
        
        // Handle RESET command
        else if (strcmp(cmd, "RESET") == 0) {
            waitingForStart = false;
            testRunning = false;
            score = 0;
            span = 3;
            trialCount = 0;
            errorsAtSpan = 0;
            
            // Turn off all LEDs
            digitalWrite(GREEN_LED, LOW);
            digitalWrite(RED_LED, LOW);
            digitalWrite(SIGNAL_LED, LOW);
            for (int i = 0; i < NUM_BUTTONS; i++) {
                digitalWrite(ledPins[i], LOW);
            }
            
            DEBUG_PRINTLN("[TEST] System reset received");
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
        
        if (mqtt_client.connect(mqtt_client_id)) {
            DEBUG_PRINTLN("[MQTT] Connected successfully!");
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

// Publish memory test results to MQTT
void publish_memory_results() {
    StaticJsonDocument<256> doc;
    
    doc["participant_id"] = participant_id;
    doc["age"] = participant_age;
    doc["gender"] = participant_gender;
    doc["memory_score"] = score;
    doc["test_type"] = "memory";
    doc["max_span_reached"] = span - 1;  // Last successful span
    doc["timestamp"] = millis();
    
    char jsonBuffer[256];
    serializeJson(doc, jsonBuffer);
    
    DEBUG_PRINTLN("\n[RESULT] Publishing memory test results:");
    DEBUG_PRINT("         Score: ");
    DEBUG_PRINTLN(score);
    DEBUG_PRINT("         Max Span: ");
    DEBUG_PRINTLN(span - 1);
    DEBUG_PRINT("         JSON: ");
    DEBUG_PRINTLN(jsonBuffer);
    
    if (!mqtt_client.connected()) {
        DEBUG_PRINTLN("[RESULT] ERROR: MQTT not connected, reconnecting...");
        reconnect_mqtt();
    }
    
    if (mqtt_client.connected()) {
        bool published = mqtt_client.publish(TOPIC_DATA, jsonBuffer, false);
        
        if (published) {
            DEBUG_PRINTLN("[RESULT] Published successfully!");
            // Visual feedback - blink green LED
            for (int i = 0; i < 3; i++) {
                digitalWrite(GREEN_LED, HIGH);
                delay(150);
                digitalWrite(GREEN_LED, LOW);
                delay(150);
            }
        } else {
            DEBUG_PRINTLN("[RESULT] Publish failed!");
            // Visual feedback - blink red LED
            for (int i = 0; i < 3; i++) {
                digitalWrite(RED_LED, HIGH);
                delay(150);
                digitalWrite(RED_LED, LOW);
                delay(150);
            }
        }
    }
    
    mqtt_client.loop();
}

// Debug function to dump MPR121 registers
void dump_regs() {
    DEBUG_PRINTLN("========================================");
    DEBUG_PRINTLN("CHAN 00 01 02 03 04 05 06 07 08 09 10 11");
    DEBUG_PRINTLN("     -- -- -- -- -- -- -- -- -- -- -- --"); 
    
    // CDC
    DEBUG_PRINT("CDC: ");
    for (int chan=0; chan<12; chan++) {
        uint8_t reg = cap.readRegister8(0x5F+chan);
        if (reg < 10) DEBUG_PRINT(" ");
        DEBUG_PRINT(reg);
        DEBUG_PRINT(" ");
    }
    DEBUG_PRINTLN();
    
    // CDT
    DEBUG_PRINT("CDT: ");
    for (int chan=0; chan<6; chan++) {
        uint8_t reg = cap.readRegister8(0x6C+chan);
        uint8_t cdtx = reg & 0b111;
        uint8_t cdty = (reg >> 4) & 0b111;
        if (cdtx < 10) DEBUG_PRINT(" ");
        DEBUG_PRINT(cdtx);
        DEBUG_PRINT(" ");
        if (cdty < 10) DEBUG_PRINT(" ");
        DEBUG_PRINT(cdty);
        DEBUG_PRINT(" ");
    }
    DEBUG_PRINTLN();
    DEBUG_PRINTLN("========================================");
}

// Generate random sequence
void generateSequence() {
    DEBUG_PRINT("\nSpan number: ");
    DEBUG_PRINTLN(span);
    DEBUG_PRINT("Trial number: ");
    DEBUG_PRINTLN(trialCount + 1);

    for (int i = 0; i < span; i++) {
        sequence[i] = random(NUM_BUTTONS);
    }

    DEBUG_PRINT("Sequence: ");
    for (int i = 0; i < span; i++) {
        DEBUG_PRINT(sequence[i]); 
        DEBUG_PRINT(" ");
    }
    DEBUG_PRINTLN();
}

// Play sequence with LEDs
void playSequence() {
    DEBUG_PRINTLN("Playing Sequence...");
    
    for (int i = 0; i < span; i++) {
        int pad = sequence[i];
        digitalWrite(ledPins[pad], HIGH);
        delay(333);
        digitalWrite(ledPins[pad], LOW);
        delay(333);
    }
}

// Wait for capacitive touch input
int waitForTouch() {
    while (true) {
        currtouched = cap.touched();

        for (int i = 0; i < NUM_BUTTONS; i++) {
            // Detect new touch on pad i
            if ((currtouched & _BV(i)) && !(lasttouched & _BV(i))) {
                DEBUG_PRINT("Pad ");
                DEBUG_PRINT(i);
                DEBUG_PRINTLN(" touched");
                
                digitalWrite(ledPins[i], HIGH);

                // Wait for release of THIS pad
                while (cap.touched() & _BV(i)) {
                    delay(10);
                    currtouched = cap.touched();
                }

                digitalWrite(ledPins[i], LOW);
                lasttouched &= ~_BV(i);

                return i;
            }
        }

        delay(10);
    }
}

// Check if user sequence matches generated sequence
void checkSequence() {
    DEBUG_PRINT("Expected: ");
    for (int i = 0; i < span; i++) {
        DEBUG_PRINT(sequence[i]); 
        DEBUG_PRINT(" ");
    }
    DEBUG_PRINTLN();

    DEBUG_PRINT("User Seq: ");
    for (int i = 0; i < span; i++) {
        DEBUG_PRINT(userSeq[i]); 
        DEBUG_PRINT(" ");
    }
    DEBUG_PRINTLN();

    // Compare sequences
    bool correct = true;
    for (int i = 0; i < span; i++) {
        if (userSeq[i] != sequence[i]) {
            correct = false;
            break;
        }
    }

    trialCount++;

    if (correct) {
        DEBUG_PRINTLN("Correct!");
        digitalWrite(GREEN_LED, HIGH);
        delay(333);
        digitalWrite(GREEN_LED, LOW);
        score++;
    } else {
        DEBUG_PRINTLN("Wrong sequence!");
        digitalWrite(RED_LED, HIGH);
        delay(333);
        digitalWrite(RED_LED, LOW);
        errorsAtSpan++;
        
        if (errorsAtSpan >= MAX_ERRORS) {
            DEBUG_PRINTLN("2 errors at same span! Test ended!");
            DEBUG_PRINT("Final Score: ");
            DEBUG_PRINTLN(score);
            
            testRunning = false;
            waitingForStart = false;
            
            // Publish results
            publish_memory_results();
            
            // Visual feedback - alternating red/green
            for (int i = 0; i < 5; i++) {
                digitalWrite(RED_LED, HIGH);
                delay(200);
                digitalWrite(RED_LED, LOW);
                digitalWrite(GREEN_LED, HIGH);
                delay(200);
                digitalWrite(GREEN_LED, LOW);
            }
            
            return;
        }
    }

    // Check if span is complete
    if (trialCount < TRIALS_PER_SPAN) {
        return;
    } else {
        // Check if max span reached
        if (span >= NUM_BUTTONS) {
            DEBUG_PRINTLN("Max span reached! Test complete!");
            DEBUG_PRINT("Final Score: ");
            DEBUG_PRINTLN(score);
            
            testRunning = false;
            waitingForStart = false;
            
            // Publish results
            publish_memory_results();
            
            // Victory celebration - all green LEDs blink
            for (int i = 0; i < 3; i++) {
                digitalWrite(GREEN_LED, HIGH);
                for (int j = 0; j < NUM_BUTTONS; j++) {
                    digitalWrite(ledPins[j], HIGH);
                }
                delay(300);
                digitalWrite(GREEN_LED, LOW);
                for (int j = 0; j < NUM_BUTTONS; j++) {
                    digitalWrite(ledPins[j], LOW);
                }
                delay(300);
            }
            
            return;
        }

        // Move to next span
        trialCount = 0;
        errorsAtSpan = 0;
        span++;

        DEBUG_PRINT("Next Span: ");
        DEBUG_PRINTLN(span);
        
        // Visual feedback - blink signal LED
        digitalWrite(SIGNAL_LED, HIGH);
        delay(500);
        digitalWrite(SIGNAL_LED, LOW);
    }
}

void setup() {
    #if DEBUG
        Serial.begin(115200);
        delay(1000);
        DEBUG_PRINTLN("\n\n========================================");
        DEBUG_PRINTLN("Memory Test - MQTT Integration");
        DEBUG_PRINTLN("MPR121 Capacitive Touch");
        DEBUG_PRINTLN("========================================\n");
    #endif

    // Initialize I2C
    DEBUG_PRINTLN("[I2C] Initializing Wire...");
    Wire.begin();
    delay(100);
    
    // Initialize MPR121
    DEBUG_PRINTLN("[MPR121] Initializing sensor...");
    if (!cap.begin(0x5A, &Wire)) {
        DEBUG_PRINTLN("[MPR121] ERROR: MPR121 not found!");
        while (1) {
            delay(10);
        }
    }
    DEBUG_PRINTLN("[MPR121] Sensor found!");
    
    delay(100);

    DEBUG_PRINTLN("[MPR121] Initial CDC/CDT values:");
    dump_regs();

    cap.setAutoconfig(true);

    DEBUG_PRINTLN("[MPR121] After autoconfig CDC/CDT values:");
    dump_regs();

    // Setup LED pins
    for (int i = 0; i < NUM_BUTTONS; i++) {
        pinMode(ledPins[i], OUTPUT);
        digitalWrite(ledPins[i], LOW);
    }
    pinMode(GREEN_LED, OUTPUT);
    pinMode(RED_LED, OUTPUT);
    pinMode(SIGNAL_LED, OUTPUT);

    DEBUG_PRINTLN("[HARDWARE] LEDs configured");

    // Setup start button
    pinMode(START_BUTTON, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(START_BUTTON), toggleTest, FALLING);
    DEBUG_PRINTLN("[HARDWARE] Start button configured");

    randomSeed(analogRead(34));

    // Setup WiFi
    setup_wifi();

    // Setup MQTT
    mqtt_client.setServer(mqtt_broker, mqtt_port);
    mqtt_client.setCallback(mqtt_callback);
    mqtt_client.setKeepAlive(60);
    DEBUG_PRINTLN("[MQTT] Client configured");

    // Initial MQTT connection
    reconnect_mqtt();

    DEBUG_PRINTLN("\n========================================");
    DEBUG_PRINTLN("Setup complete - System ready!");
    DEBUG_PRINTLN("Waiting for START_MEMORY command...");
    DEBUG_PRINTLN("========================================\n");
    
    // Startup animation - wave across LEDs
    for (int i = 0; i < NUM_BUTTONS; i++) {
        digitalWrite(ledPins[i], HIGH);
        delay(50);
        digitalWrite(ledPins[i], LOW);
    }
    
    delay(500);
}

void loop() {
    // Maintain MQTT connection
    if (!mqtt_client.connected()) {
        reconnect_mqtt();
    }
    mqtt_client.loop();

    // Get current touch state
    currtouched = cap.touched();

    // Handle start button press
    if (startButtonPressed) {
        startButtonPressed = false;

        if (waitingForStart && !testRunning) {
            // Start test
            DEBUG_PRINTLN("[TEST] Test started!");
            testRunning = true;
            score = 0;
            span = 3;
            trialCount = 0;
            errorsAtSpan = 0;
            
            // Visual confirmation
            digitalWrite(GREEN_LED, HIGH);
            delay(500);
            digitalWrite(GREEN_LED, LOW);
            
            delay(200);
        } else if (testRunning) {
            // Stop test mid-way
            DEBUG_PRINTLN("[TEST] Test stopped by user!");
            testRunning = false;
            waitingForStart = false;
            
            // Turn off all LEDs
            digitalWrite(GREEN_LED, LOW);
            digitalWrite(RED_LED, LOW);
            digitalWrite(SIGNAL_LED, LOW);
            for (int i = 0; i < NUM_BUTTONS; i++) {
                digitalWrite(ledPins[i], LOW);
            }
            
            delay(200);
        }
    }
    
    // Run test if active
    if (testRunning) {
        generateSequence();
        playSequence();

        DEBUG_PRINTLN("[TEST] Waiting for user input...");
        
        // Signal user to start input
        digitalWrite(SIGNAL_LED, HIGH);

        // Collect user input
        for (int i = 0; i < span; i++) {
            userSeq[i] = waitForTouch();
        }
        
        digitalWrite(SIGNAL_LED, LOW);

        // Check if sequence is correct
        checkSequence();
        
        delay(333);
    }

    // Update last touched state
    lasttouched = currtouched;
    
    delay(10);
}
