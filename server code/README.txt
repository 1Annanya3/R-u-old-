# CS3237 Health Assessment Server — Quick Start

Run Reaction, Balance, Memory, and Image tests over MQTT, log to Google Sheets, and estimate cognitive age. 🚀

## Requirements
- Python 3.10+
- Files in one folder: service_account.json, test.py, long_term_analytics.py, use_forward_mem_model.py, reaction_age_predictor.py, forward_mem_poly3_ridge.joblib
- Spreadsheet: “CS3237_Health_Assessment_Data” shared with your service account (Editor)

## Install
- python3 -m venv venv  
- source venv/bin/activate  
- python -m pip install -U pip setuptools wheel  
- pip install gspread oauth2client paho-mqtt numpy pandas scikit-learn==1.6.1 joblib matplotlib

## Run
- python test.py  
- Enter: **Participant ID**, optional **Real age**, **Gender (M/F/O)**, **Date (YYYY‑MM‑DD)**  
- Commands:
  - start memory | start reaction | start balance | start image
  - long term analytics <P_ID>
  - quit

## What gets logged
- “Memory Test”: raw memory rows  
- “Predicted_Ages”: predicted ages (Reaction/Balance/Memory/Face/Combined)  
- “LongTerm”: one row per **participant_id + date** with reaction_time_ms, balance_duration_s, memory_score, cognitive_age, real_age, date (rows are updated if same day) 📈

## Tips
- Always type **SEND** to commit an event.  
- If the model won’t load, match the scikit‑learn version to how the joblib was saved (1.6.1 here) or re‑train and re‑save.  
- Keep service_account.json private.
