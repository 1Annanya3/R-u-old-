# ⚡ Reaction Test Module

This directory houses the client firmware and model components for the **Reaction Time Assessment**. This module is unique as its prediction model is **trained dynamically** upon server startup using data from Google Sheets.

## 💾 Folder Contents

| File/Directory | Role | Purpose |
| :--- | :--- | :--- |
| **`Reaction_time_server_latest.ino`** | Arduino/Firmware | **Primary client code** uploaded to the microcontroller to run the test and publish the participant's average reaction time (in milliseconds) via MQTT. |
| **`new_reaction test prediction model`** | Model Artifact | Likely the saved model file, though the server-side Python code usually re-trains from the latest sheet data. |
| **`reaction_equation.json`** | Model Reference | A JSON file containing the intercept and terms (coefficients) of the Polynomial Regression model. |

## ⚙️ Workflow & Dynamic Modeling

This module relies on the `reaction_age_predictor.py` script (located in the main server code) for its intelligence:

1.  **Model Training (Server Startup):** When the server starts, **`reaction_age_predictor.py`** connects to Google Sheets, fetches all historical "Reaction Time Test" data, and trains a new **Polynomial Regression Model** immediately.
2.  **Server Command:** The server initiates the test via MQTT: `start reaction`.
3.  **Data Sent:** The client firmware (**.ino**) publishes the `average_reaction_ms` via MQTT.
4.  **Prediction:** The server uses the dynamically trained model in `reaction_age_predictor.py` to calculate the **Cognitive Age** based on the reaction time and gender.
5.  **Logging:** The result is logged to Google Sheets.
