/*********************************************************
This is a library for the MPR121 12-channel Capacitive touch sensor

Designed specifically to work with the MPR121 Breakout in the Adafruit shop 
  ----> https://www.adafruit.com/products/

These sensors use I2C communicate, at least 2 pins are required 
to interface

this sketch demonstrates the auto-config chip-funktionality.
for more information have a look at
https://github.com/adafruit/Adafruit_MPR121/issues/39
https://github.com/adafruit/Adafruit_MPR121/pull/43


based on MPR121test
modified & extended by Carter Nelson/caternuson

MIT license, all text above must be included in any redistribution
**********************************************************/
#include <Wire.h>
#include <Adafruit_MPR121.h>

#ifndef _BV
#define _BV(bit) (1 << (bit)) 
#endif

Adafruit_MPR121 cap = Adafruit_MPR121();

// Keeps track of the last pins touched
// so we know when buttons are 'released'
uint16_t lasttouched = 0;
uint16_t currtouched = 0;

#define NUM_BUTTONS 10
#define TRIALS_PER_SPAN 3
#define MAX_ERRORS 2


int ledPins[NUM_BUTTONS] = {12, 13, 14, 15, 16, 17, 18, 19, 26, 27}; // LEDs for each pad
#define GREEN_LED 33
#define RED_LED   32
#define SIGNAL_LED 4
#define START_BUTTON 23

int sequence[10];
int userSeq[10];

int span = 3;          // starting sequence length
int trialCount = 0;         // trials completed at current span
int errorsAtSpan = 0;       // number of incorrect trials at current span
int score = 0;     // score tracker
bool testRunning = false;
volatile bool startButtonPressed = false;

void IRAM_ATTR toggleTest() {
  startButtonPressed = true;
}

void dump_regs() {
  Serial.println("========================================");
  Serial.println("CHAN 00 01 02 03 04 05 06 07 08 09 10 11");
  Serial.println("     -- -- -- -- -- -- -- -- -- -- -- --"); 
  // CDC
  Serial.print("CDC: ");
  for (int chan=0; chan<12; chan++) {
    uint8_t reg = cap.readRegister8(0x5F+chan);
    if (reg < 10) Serial.print(" ");
    Serial.print(reg);
    Serial.print(" ");
  }
  Serial.println();
  // CDT
  Serial.print("CDT: ");
  for (int chan=0; chan<6; chan++) {
    uint8_t reg = cap.readRegister8(0x6C+chan);
    uint8_t cdtx = reg & 0b111;
    uint8_t cdty = (reg >> 4) & 0b111;
    if (cdtx < 10) Serial.print(" ");
    Serial.print(cdtx);
    Serial.print(" ");
    if (cdty < 10) Serial.print(" ");
    Serial.print(cdty);
    Serial.print(" ");
  }
  Serial.println();
  Serial.println("========================================");
}

//generate random sequence
void generateSequence() {

  Serial.print("\nSpan number: ");
  Serial.println(span);
  Serial.print("Trial number: ");
  Serial.println(trialCount);

  for (int i = 0; i < span; i++) {
    sequence[i] = random(NUM_BUTTONS);
  }

  Serial.print("\n🧩 Sequence: ");
  for (int i = 0; i < span; i++) {
    Serial.print(sequence[i]); Serial.print(" ");

  }
  Serial.println();
}

//Play Sequence
void playSequence() {
  Serial.print("\n🧩 Playing Sequence... ");
  for (int i = 0; i < span; i++) {
    int pad = sequence[i];
    digitalWrite(ledPins[pad], HIGH);
    delay(333);
    digitalWrite(ledPins[pad], LOW);
    delay(333);
  }
  Serial.println();
}

int waitForTouch() {
  while (true) {
      currtouched = cap.touched();

      for (int i = 0; i < NUM_BUTTONS; i++) {
          // Detect new touch on pad i
          if ((currtouched & _BV(i)) && !(lasttouched & _BV(i))) {
              Serial.print("Pad "); Serial.print(i); Serial.println(" touched");
              digitalWrite(ledPins[i], HIGH);

              // Wait for release of THIS pad
              while (cap.touched() & _BV(i)) {
                  delay(10); // small debounce
                  currtouched = cap.touched(); // keep updating
              }

              digitalWrite(ledPins[i], LOW);

              // Clear only the bit for the pad just released
              lasttouched &= ~_BV(i);

              return i; // return the index of the pad released
          }
      }

      delay(10);
  }

}

void checkSequence() {

  Serial.print("Sequence: ");
  for (int i = 0; i < span; i++) Serial.print(sequence[i]); Serial.print(" ");
  Serial.println();

  Serial.print("UserSeq:  ");
  for (int i = 0; i < span; i++) Serial.print(userSeq[i]); Serial.print(" ");
  Serial.println();

  // Compare input to correct sequence
  bool correct = true;
  for (int i = 0; i < span; i++) {
    if (userSeq[i] != sequence[i]) {
      correct = false;
      break;
    }
  }

  trialCount++;

  if(correct) {

    Serial.print("✅ Correct!");
    digitalWrite(GREEN_LED, HIGH);
    delay(333);
    digitalWrite(GREEN_LED, LOW);
    score++;


  } else {
    Serial.println("❌ Wrong sequence!");
    digitalWrite(RED_LED, HIGH);
    delay(333);
    digitalWrite(RED_LED, LOW);
    errorsAtSpan++;
    if (errorsAtSpan >= MAX_ERRORS) {
    Serial.println("❌ 2 errors in same span! Test ended!");
    Serial.print("Your Score is: ");
    Serial.println(score);
    testRunning = false;
    return;
    }

  }

 if (trialCount < TRIALS_PER_SPAN) {
    return;
  } 
  else {

    if (span >= NUM_BUTTONS) {
      Serial.println("🎉 Max span reached! Test complete!");
      Serial.print("Your Score is: ");
      Serial.println(score);
      testRunning = false;
      return;
    }

    trialCount = 0;
    errorsAtSpan = 0; // reset errors for next span
    span++;

    Serial.print("➡️ Next Span: ");
    Serial.println(span);

  }
}

void resetTrial() {
    Serial.println("🔄 Trial reset!");
    testRunning = false;  // stop the trial
    score = 0;
    span = 3;
    trialCount = 0;
    errorsAtSpan = 0;

    // Turn off all LEDs
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, LOW);
    digitalWrite(SIGNAL_LED, LOW);
    for (int i = 0; i < NUM_BUTTONS; i++) digitalWrite(ledPins[i], LOW);
}

void setup() {
  delay(1000);
  Serial.begin(115200);
  while (!Serial);
  Serial.println("MPR121 Autoconfiguration Test. (MPR121-AutoConfig.ino)");
  
  Serial.println("startup `Wire`");
  Wire.begin();
  Serial.println("startup `Wire` done.");
  delay(100);
  
  Serial.println("cap.begin..");
  if (!cap.begin(0x5A, &Wire)) {
    Serial.println("MPR121 not found, check wiring?");
    while (1);
  }
  Serial.println("MPR121 found!");
  
  delay(100);

  Serial.println("Initial CDC/CDT values:");
  dump_regs();

  cap.setAutoconfig(true);

  Serial.println("After autoconfig CDC/CDT values:");
  dump_regs();

  // LED setup
  for (int i = 0; i < NUM_BUTTONS; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(SIGNAL_LED, OUTPUT);

  //Button setup
  pinMode(START_BUTTON, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(START_BUTTON), toggleTest, FALLING);

  randomSeed(analogRead(34));
  delay(500);
 
}


void loop() {
  // Get the currently touched pads
  currtouched = cap.touched();

  if (startButtonPressed) {
    startButtonPressed = false;

    if (!testRunning) {
      // ----------------------------
      // This is where testRunning = true
      // ----------------------------
      Serial.println("🟢 Test started!");
      testRunning = true;
      score = 0;
      span = 3;
      trialCount = 0;
      errorsAtSpan = 0;
      delay(200); // debounce
    } else {
      Serial.println("🔴 Test stopped!");
      testRunning = false;
      resetTrial();
      delay(200); // debounce
    }
  }
  
  if (testRunning) {

    generateSequence();
    playSequence();

    Serial.println("Waiting for input:");
    // signal for user to start input
    digitalWrite(SIGNAL_LED, HIGH);


    for (int i = 0; i < span; i++) {
      userSeq[i] = waitForTouch();

    }
    
    digitalWrite(SIGNAL_LED, LOW);

    checkSequence();
    delay(333);
  }

  // reset our state
  lasttouched = currtouched;
}
