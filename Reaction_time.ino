#include <Arduino.h>
#include <Wire.h>              
#include <LiquidCrystal_I2C.h>  # for the LCD screen

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

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP); 
    Wire.begin(I2C_SDA, I2C_SCL);

    lcd.init();
    lcd.backlight();
    
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
        // --- State: Final Results ---
        float average_time_us = (float)total_reaction_us / TOTAL_TESTS;
        
        lcd.clear();
        lcd.print("AVG Time (us):");
        lcd.setCursor(0, 1);
        char buffer[16];
        sprintf(buffer, "%.0f", average_time_us); // Display in microseconds
        lcd.print(buffer);
        
        Serial.printf("\n--- FINAL RESULTS ---\n");
        Serial.printf("Total Tests: %d\n", TOTAL_TESTS);
        Serial.printf("Total Reaction Time: %lu us\n", total_reaction_us);
        Serial.printf("Average Reaction Time: %.0f us\n", average_time_us);
        
        // Stop the loop after displaying final results
        while(1) delay(10); 
    }

    if (!led_active && !button_pressed) {
        // --- State: Waiting for Test Start ---
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
        // --- State: Test Finished ---
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
