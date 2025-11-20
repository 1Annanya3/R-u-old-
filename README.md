# R-u-old- 🩺 Multi-Test Cognitive Assessment Server

This repository contains the full backend and client firmware logic for a multi-modal cognitive assessment platform. The server calculates a "Cognitive Age" from four tests and logs results to Google Sheets.

## ⚙️ Architecture

The core of the system is the Python server (`DEMO/server.py`) which coordinates all activities:

  * **Communication:** Uses **MQTT** to talk to client devices (Arduino/ESP32 firmware) for Reaction, Balance, and Memory tests, and a **Flask API** for Face image uploads.
  * **Persistence:** Logs raw data (Memory) and predicted ages (all tests) to designated Google Sheets tabs (`Predicted_Ages`, `LongTerm`, `Memory Test`).

## ✨ Tests & Prediction Models

| Test Module | Client Code | Prediction Logic |
| :--- | :--- | :--- |
| **Reaction** | `Reaction_time_server_latest.ino` | **Dynamic Model:** Polynomial Regression trained on Google Sheets data at server startup. |
| **Balance** | `balance_server_extra_threshold.ino` | **Pre-trained Model:** Loads `balance_age_model.joblib` to estimate age from balance duration. |
| **Memory** | `memory_test_server.ino` | **Pre-trained Model:** Loads `forward_mem_poly3_ridge*.joblib` to estimate age from memory score. |
| **Face** | Upload via `/upload_face_form` (Flask) | **Pre-trained CNN:** Uses `cnn_predicter.py` (ResNet-based model) to infer age from image. |

## 🚀 Quick Start

1.  **Prerequisites:** You need Python 3.8+ and your Google Sheets `service_account.json` file.
2.  **Run Server:** Navigate to the `DEMO` directory and execute:
    ```bash
    python server.py
    ```
3.  **CLI Commands:** Use the following commands in the running server terminal to control the process:
      * `start <reaction/balance/memory/image>`
      * `combined age` (Calculates weighted average cognitive age)
      * `long term analytics <P_ID>`
      * `quit`
