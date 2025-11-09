Overview

This server receives health test results over MQTT and logs them to Google Sheets. It also predicts cognitive age from Reaction and Memory tests and maintains a LongTerm sheet for longitudinal tracking. 🙂

Supported tests: Reaction, Balance, Memory, Image.

Key sheets:

Memory Test: raw memory rows.

Predicted_Ages: per‑event predicted ages (Reaction/Balance/Memory/Face/Combined).

LongTerm: one row per participant_id per date with reaction_time_ms, balance_duration_s, memory_score, cognitive_age, real_age, date. 📈

Folder contents

test.py: main server (MQTT, Sheets logging, LongTerm upserts, CLI).

long_term_analytics.py: LongTerm sheet creation, upsert by (participant_id, date), plotting cognitive vs real age over time.

use_forward_mem_model.py: loads the NCPT Forward memory polynomial Ridge model and provides predict_score(...) and estimate_age_from_score(...).

reaction_age_predictor.py: reaction time → cognitive age formula.

service_account.json: Google service account credentials (keep secret!). 🔐

forward_mem_poly3_ridge.joblib (or similar): saved memory model.

train_forward_mem_model_poly3_ridge_save.py (optional): training script for the memory model.

Prerequisites

Python 3.10+ on Linux/WSL/macOS.

A Google Cloud service account with Sheets API enabled; JSON key saved as service_account.json and shared edit access to your spreadsheet.

A spreadsheet named CS3237_Health_Assessment_Data (or adjust in code).

Install

Create/activate a virtual environment:

python3 -m venv venv

source venv/bin/activate (Windows/WSL: source venv/bin/activate)

Upgrade tooling:

python -m pip install -U pip setuptools wheel

Install dependencies:

pip install gspread oauth2client paho-mqtt numpy pandas scikit-learn joblib matplotlib

Place service_account.json alongside test.py.

Put your trained model file (e.g., forward_mem_poly3_ridge.joblib) next to use_forward_mem_model.py.

Run

Start the server:

python test.py

Provide prompts:

Participant ID (e.g., P001)

Real age (optional; stored in LongTerm)

Gender (M/F/O)

Date (YYYY-MM-DD; used as the key with participant_id)

Commands:

start reaction | start balance | start memory | start image

long term analytics <P_ID> → shows a plot of cognitive vs real age over time

quit

Data flow

On each incoming MQTT message:

Reaction

Logs predicted age to Predicted_Ages (if reaction predictor is enabled).

Upserts LongTerm with reaction_time_ms and cognitive_age (if computed).

Balance

Upserts LongTerm with balance_duration_s.

Memory

Appends raw to Memory Test.

Computes cognitive age via memory model, logs to Predicted_Ages, and upserts LongTerm with memory_score and cognitive_age.

Upsert rule: if the same participant_id and date exist, the row is updated in place; otherwise a new row is appended. ✅
