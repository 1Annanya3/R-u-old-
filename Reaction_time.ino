// Final Integrated Code: Reaction Test Station
#include <Arduino.h>
#include <Wire.h>              
#include <LiquidCrystal_I2C.h>  
#include <WiFi.h>              // Added for Wi-Fi connectivity
#include <HTTPClient.h>         // Added for HTTP requests

// --- NETWORK CONFIGURATION (CRITICAL: UPDATE THESE) ---
const char* ssid = "ANNANYA";           // <-- REPLACE with your WiFi name
const char* password = "hahahaha";   // <-- REPLACE with your WiFi password
const char* server_ip = "10.30.232.95";        // <-- REPLACE with your Laptop's IP
const int server_port = 5000;

// --- PROJECT CONFIGURATION (Hardcoded for testing P001, Age 22) ---
const char* participant_id = "P001";
const int actual_age = 22;

// --- Pin Definitions ---
const int LED_PIN = 2;   
const int BUTTON_PIN = 4;
const int I2C_SDA = 21; 
const int I2C_SCL = 22;

// --- LCD Object ---
LiquidCrystal_I2C lcd(0x27, 16, 2); 

// --- Timing and State Variables ---
volatile bool led_active = false;      
volatile bool button_pressed = false; 
volatile unsigned long start_time_us = 0; 
volatile unsigned long reaction_time_us = 0; 

const unsigned long DEBOUNCE_US = 50000; 
volatile unsigned long last_isr_us = 0;

// --- New Variables for Data Logging ---
const int TOTAL_TESTS = 5;
int test_count = 0;
unsigned long total_reaction_us = 0;

// --- Interrupt Service Routine (ISR) ---
void IRAM_ATTR buttonISR() {
    unsigned long now_us = micros();
    if ((now_us - last_isr_us) > DEBOUNCE_US) {
        last_isr_us = now_us;
        
        if (led_active) {
            reaction_time_us = now_us - start_time_us;
            button_pressed = true;
            digitalWrite(LED_PIN, LOW);
        }
    }
}

// --- Data Transmission Function (NEW) ---
void sendReactionData(float average_reaction_time_ms) {
    HTTPClient http;
    
    // Check if WiFi is connected before attempting HTTP
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("Error: WiFi disconnected. Cannot send data.");
        return;
    }

    // Construct the full URL string 
    // Format: http://<IP>:5000/data?id=<id>&age=<age>&station=reaction&value=<value>
    String url = "http://";
    url += server_ip;
    url += ":" + String(server_port);
    url += "/data?";
    
    url += "id=" + String(participant_id);
    url += "&age=" + String(actual_age); // Send age with the first test
    url += "&station=reaction";
    url += "&value=" + String(average_reaction_time_ms, 2); // Value in ms, 2 decimal places

    Serial.print("Sending Request: ");
    Serial.println(url);
    lcd.clear();
    lcd.print("Sending Data...");

    // Begin connection and send the GET request
    http.begin(url);
    int httpResponseCode = http.GET();

    if (httpResponseCode > 0) {
        // HTTP response code 200 is OK
        String payload = http.getString();
        Serial.printf("Server Response Code: %d\n", httpResponseCode); 
        Serial.println("Server Payload: " + payload);
        
        // Display server status on LCD
        lcd.setCursor(0, 1);
        if (httpResponseCode == 200) {
             lcd.print("Data SENT (200)");
        } else {
             lcd.print("Server Error!");
        }
        
    } else {
        Serial.printf("HTTP request failed. Error Code: %d\n", httpResponseCode);
        lcd.setCursor(0, 1);
        lcd.print("HTTP Fail!");
    }
    
    http.end(); // Close connection
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP); 
    Wire.begin(I2C_SDA, I2C_SCL);

    // --- WI-FI CONNECTION SETUP (NEW) ---
    WiFi.begin(ssid, password);
    lcd.init();
    lcd.backlight();
    lcd.print("Connecting WiFi");
    Serial.print("Connecting to WiFi");
    
    // Wait until connected
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        lcd.print(".");
    }
    
    Serial.println("\nWiFi connected.");
    Serial.print("ESP32 IP address: ");
    Serial.println(WiFi.localIP());
    lcd.clear();
    lcd.print("WiFi Connected!");
    delay(1000);
    // ------------------------------------

    lcd.clear();
    lcd.print("Reaction Time Test");
    lcd.setCursor(0, 1);
    lcd.print("Starting 5 Tests");
    delay(2000);

    attachInterrupt(BUTTON_PIN, buttonISR, FALLING);
    Serial.println("System Initialized. Starting tests...");
}

void loop() {
    // Check if 5 tests have been completed
    if (test_count >= TOTAL_TESTS) {
        // --- State: Final Results & Data Send (MODIFIED) ---
        float average_time_us = (float)total_reaction_us / TOTAL_TESTS;
        float average_time_ms = average_time_us / 1000.0; // Convert to milliseconds
        
        // Display final result on Serial
        Serial.printf("\n--- FINAL RESULTS ---\n");
        Serial.printf("Average Reaction Time: %.2f ms\n", average_time_ms);
        
        // **!!! SEND DATA TO CLOUD !!!**
        sendReactionData(average_time_ms);
        
        Serial.println("Session Complete. Holding.");
        // Stop the loop after displaying final results
        while(1) delay(10); 
    }

    if (!led_active && !button_pressed) {
        // ... (Existing logic for waiting and starting test) ...
        lcd.setCursor(0, 1);
        lcd.printf("Test %d of %d", test_count + 1, TOTAL_TESTS); 

        unsigned long random_delay = random(1000, 5000); 
        delay(random_delay);
        
        // --- Start Test ---
        digitalWrite(LED_PIN, HIGH);
        led_active = true;
        start_time_us = micros();
        
        lcd.setCursor(0, 1);
        lcd.print("CLICK NOW!      ");
    }

    if (button_pressed) {
        // ... (Existing logic for test finished and result display) ...
        test_count++;
        total_reaction_us += reaction_time_us;
        
        float reaction_ms = (float)reaction_time_us / 1000.0;
        
        lcd.clear();
        lcd.printf("Test %d Complete", test_count);
        lcd.setCursor(0, 1);
        lcd.printf("Time: %.2f ms", reaction_ms);

        Serial.printf("[Test %d] Time: %.2f ms\n", test_count, reaction_ms);
        
        // Reset state for the next test
        led_active = false;
        button_pressed = false;
        reaction_time_us = 0;
        
        delay(2000); 
    }
}
