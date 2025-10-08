// Basic demo for accelerometer readings from Adafruit MPU6050

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

const int RESET_BUTTON_PIN = 15;  // Reset button
const float X_ACC_THRESHOLD = -7.0;  // deg
const float LEG_UP_GYRO_THRESHOLD = 30.0;
//const float ACC_STOP_THRESHOLD = 2.0;       // g
const float GYRO_STOP_THRESHOLD = 150.0;    // deg/s
const int LEG_UP_HOLD_MS = 0;             // milliseconds leg must be up

bool trialRunning = false;
unsigned long trialStartTime = 0;
unsigned long legUpDetectedTime = 0;
bool legUpFlag = false;

//helper functions
float getAccelMagnitude(float ax, float ay, float az) {
  return sqrt(ax*ax + ay*ay + az*az);
}

float getGyroMagnitude(float gx, float gy, float gz) {
  return sqrt(gx*gx + gy*gy + gz*gz);
}

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit MPU6050 test!");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP); //reset button
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");
  delay(100);
}

void loop() {

  /* Get new sensor events with the readings */
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  //trial start
  float acc_mag_legup = getAccelMagnitude(a.acceleration.x, a.acceleration.y, a.acceleration.z);
  float gyro_mag_legup = getGyroMagnitude(g.gyro.x, g.gyro.y, g.gyro.z);

  if (!trialRunning) {
    if (a.acceleration.x > X_ACC_THRESHOLD || gyro_mag_legup > LEG_UP_GYRO_THRESHOLD) { // gyro scaled roughly
      if (!legUpFlag) {
        legUpDetectedTime = millis();
        legUpFlag = true;
      } else if (millis() - legUpDetectedTime >= LEG_UP_HOLD_MS) {
        trialRunning = true;
        trialStartTime = millis();
        Serial.println("Trial Started!");
      }
    } else {
      legUpFlag = false;
    }
  }

  //trial stop
  if (trialRunning) {
    mpu.getEvent(&a, &g, &temp); // read again
    float acc_mag_stop = getAccelMagnitude(a.acceleration.x, a.acceleration.y, a.acceleration.z);
    float gyro_mag_stop = getGyroMagnitude(g.gyro.x, g.gyro.y, g.gyro.z);
  
    if (a.acceleration.x < X_ACC_THRESHOLD || gyro_mag_stop > GYRO_STOP_THRESHOLD) {
      trialRunning = false;
      unsigned long trialDuration = millis() - trialStartTime;
      Serial.print("Trial Stopped! Duration (ms): ");
      Serial.println(trialDuration);
    }
  }

//reset button
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
      trialRunning = false;
      legUpFlag = false;
      Serial.println("Trial Reset!");
      delay(200);  // debounce
    }
  // // /* Print out the values */
  //     Serial.print("Acceleration X: ");
  //     Serial.print(a.acceleration.x);
  //     Serial.print(", Y: ");
  //     Serial.print(a.acceleration.y);
  //     Serial.print(", Z: ");
  //     Serial.print(a.acceleration.z);
  //     Serial.println(" m/s^2");
  //     Serial.println(acc_mag_legup);

  //     Serial.print("Rotation X: ");
  //     Serial.print(g.gyro.x);
  //     Serial.print(", Y: ");
  //     Serial.print(g.gyro.y);
  //     Serial.print(", Z: ");
  //     Serial.print(g.gyro.z);
  //     Serial.println(" rad/s");
  //     Serial.println(gyro_mag_legup);

  //     Serial.println("");
  //delay(500);
}