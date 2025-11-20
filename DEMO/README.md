# 🩺 Health Assessment Cognitive Server

This project is the backend server and machine learning system for a multi-modal cognitive assessment platform. It receives test data (Reaction, Balance, Memory, Face Image) via **MQTT** and a **Flask API**, calculates a "Cognitive Age" for each test, and logs all results to **Google Sheets** for analysis.

## 🚀 Quick Start

### Prerequisites

1.  **Python 3.8+**
2.  **Dependencies:** Install libraries like `gspread`, `paho-mqtt`, `Flask`, `scikit-learn`, `joblib`, `torch`, `pandas`, and `opencv-python`.
3.  **Authentication:** Place your Google Sheets **`service_account.json`** file in the root directory.
4.  **Trained Models:** Ensure all model files (`*.joblib`, `age_cnn.pt`, `*.json`) are in their expected locations (see Architecture below).

### Running the Server

1.  Start the server:
    ```bash
    python server.py
    ```
2.  Follow the prompts to enter the **Participant ID**, **Real Age**, **Gender**, and **Date** for the session.
3.  The **Face API** automatically starts on port `5000` (e.g., `http://<local-IP>:5000/upload_face_form`).
4.  Use the command line interface (CLI) for control:

| CLI Command | Action |
| :--- | :--- |
| `start reaction` | Sends MQTT command to start the Reaction Test. |
| `start image` | Sends MQTT command to start the Image Capture. |
| `combined age` | Calculates and prompts to log the **weighted average** of all available ages (Reaction, Balance, Memory, Face). |
| `long term analytics <P_ID>` | Generates a plot of Cognitive vs. Real Age over time for the specified participant. |
| `quit` | Stops the MQTT loop and server. |

---

## 🏗️ Architecture and Data Flow

**`server.py`** is the core entry point, managing communication and coordinating multiple specialized prediction modules.

### Dependency Flow

The system integrates modules and models to flow data from the client to the persistence layer (Google Sheets).



### Key Modules

| Module | Purpose | Source of Prediction | Persistence |
| :--- | :--- | :--- | :--- |
| **`server.py`** | **Controller** | Handles MQTT/API events, session state, and final logging. | `Predicted_Ages`, `LongTerm` |
| **`face_api.py`** | **Image Endpoint** | Flask API to receive face images, calls **`cnn_predicter.py`** for age inference. | `Predicted_Ages`, `LongTerm` |
| **`reaction_age_predictor.py`** | **Reaction Model** | **Trains a Polynomial Regression** model on the fly using data fetched from Google Sheets. | `Predicted_Ages`, `LongTerm` |
| **`use_balance_model.py`** | **Balance Model** | Loads pre-trained **`balance_age_model.joblib`** to estimate age from balance duration. | `Predicted_Ages`, `LongTerm` |
| **`use_forward_mem_model.py`** | **Memory Model** | Loads pre-trained **`forward_mem_poly3_ridge*.joblib`** to estimate age from memory score. | `Memory Test`, `Predicted_Ages`, `LongTerm` |
| **`long_term_analytics.py`** | **Analytics** | Contains functions for plotting time-series data and performing row **upserts** (updates existing rows or appends new ones) on the `LongTerm` sheet. | `LongTerm` |
