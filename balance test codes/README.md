# 🤸 Balance Test Module

This directory contains all the client-side firmware, training scripts, and pre-trained model assets necessary for the **Balance Assessment** portion of the Health Assessment Cognitive Server.

The test measures a participant's stability by recording the duration they can maintain balance on a sensor-equipped platform.

## 💾 Folder Contents

| File/Directory | Type | Purpose |
| :--- | :--- | :--- |
| **`balance_server.ino`** | Arduino/Firmware | **Primary client code** for the balance test. This file should be uploaded to the microcontroller (e.g., ESP32 or ESP8266) to handle sensor input, MQTT communication, and test timing. |
| **`balance_server_extra_threshold.ino`** | Arduino/Firmware | An **alternative firmware version** that may implement a stricter or different stability threshold logic. |
| **`Balance test model and graph`** | Model/Script Artifacts | Contains the trained machine learning model files (`balance_age_model.joblib`) and metrics/training graphs. These are used by the main Python server to predict cognitive age. |
| **`Synthetic data generation`** | Training Script | Scripts used to create or augment the training dataset for the balance prediction model. |

## ⚙️ Workflow & Interaction

The balance test operates as a dedicated module that sends data to the main Python server.

1.  **Start Command:** The main server's CLI sends an MQTT command: `start balance`.
2.  **Client Execution:** The **`balance_server.ino`** firmware initiates the test sequence on the physical device.
3.  **Data Transmission:** Once the participant loses balance, the client code publishes the result (the duration in seconds, `balance_duration_s`) to the server's dedicated MQTT topic (`/health/data/balance`).
4.  **Server Prediction:** The main server receives the `balance_duration_s`, passes it, along with the **gender**, to the `use_balance_model.py` module.
5.  **Age Estimation:** The module loads the pre-trained model (from `*.joblib`) and returns a predicted "Cognitive Age" based on the duration.
6.  **Logging:** The final predicted age is then logged to the `Predicted_Ages` and `LongTerm` Google Sheets tabs.

## ✨ Model Summary

The age prediction model uses **balance duration** and **gender** as input features. The pre-trained model is used directly for inference.
