# 🧠 Memory Test Module

This directory contains the client firmware and the pre-trained model assets for the **Memory Assessment**. This test typically yields a numerical score, which the server uses to predict the participant's cognitive age.

## 💾 Folder Contents

| File/Directory | Role | Purpose |
| :--- | :--- | :--- |
| **`memory_test_server.ino`** | Firmware | **Primary client code** for running the test and publishing the raw score via MQTT. |
| **`corsi_block_memory_test.ino`** | Firmware | An alternative or component firmware, likely implementing the Corsi Block memory task. |
| **`*.joblib`** | Pre-trained Model | The **pre-trained Ridge Regression model** used by the server's prediction logic to estimate age from the memory score. |
| **`Age_Prediction_memory_training.py`** | Training Script | The script used to train and save the `.joblib` model file. |

## ⚙️ Workflow

1.  **Server Command:** The main server initiates the test via MQTT command: `start memory`.
2.  **Data Sent:** The client publishes the raw **`memory_score`** via MQTT to the server.
3.  **Logging:** The server logs the raw score into the **`Memory Test`** Google Sheet tab.
4.  **Prediction:** The server loads the **`.joblib`** model and estimates the **Cognitive Age** based on the score and gender.
5.  **Persistence:** The predicted age is logged to the `Predicted_Ages` and `LongTerm` Google Sheets tabs.
