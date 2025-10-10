# backend_server.py

from flask import Flask, request, jsonify
import pandas as pd
import threading
import os
import gspread # Library to interact with Google Sheets
import time

app = Flask(__name__)

# --- CONFIGURATION (UPDATE THESE BEFORE RUNNING) ---
# NOTE: Ensure 'service_account.json' is in the same directory as this script.
SERVICE_ACCOUNT_FILE = 'service_account.json' 
SPREADSHEET_NAME = 'CS3237_Health_Assessment_Data' 
# ---------------------------------------------------

# --- Global Data Storage and Synchronization ---
# SESSION_DATA holds incomplete session data for consolidation
SESSION_DATA = {} 
# Lock to ensure thread-safe operations when reading/writing global data
DATA_LOCK = threading.Lock() 
# We expect 3 main features: reaction_time, balance_duration, memory_score
TOTAL_FEATURES_TO_COLLECT = 3 
# We expect an additional 'actual_age', so total keys check is +1
TOTAL_KEYS_CHECK = TOTAL_FEATURES_TO_COLLECT + 1 

# Global Sheets Client and Worksheet instance
gc = None
worksheet = None

# --- Utility Functions ---

def setup_google_sheets():
    """Initializes the connection to Google Sheets using the Service Account."""
    global gc, worksheet
    
    try:
        # Authenticate using the service account file
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        
        # Open the specific spreadsheet
        spreadsheet = gc.open(SPREADSHEET_NAME)
        
        # Open the first worksheet (Sheet1 is the default)
        worksheet = spreadsheet.sheet1
        
        # Optional: Verify the header row structure
        expected_header = ['participant_id', 'actual_age', 'reaction_time_ms', 'balance_duration_s', 'memory_score']
        current_header = worksheet.row_values(1)
        if current_header != expected_header:
             print(f"WARNING: Sheet header mismatch. Expected: {expected_header}, Found: {current_header}")
        
        print(f"Successfully connected to Google Sheet: {SPREADSHEET_NAME}")
        
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Service account file '{SERVICE_ACCOUNT_FILE}' not found.")
        print("Ensure the JSON key is in the script directory.")
        exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to Google Sheets. Error: {e}")
        exit(1)


def log_session_complete(participant_id):
    """Writes the final consolidated data to the Google Sheet and cleans up the session."""
    
    # Retrieve the final data
    final_data = SESSION_DATA.pop(participant_id)
    
    # Define the order of columns and values (MUST match Google Sheet order)
    ordered_data = {
        'participant_id': participant_id,
        'actual_age': final_data.get('actual_age'),
        'reaction_time_ms': final_data.get('reaction_time_ms'),
        'balance_duration_s': final_data.get('balance_duration_s'),
        'memory_score': final_data.get('memory_score')
    }
    
    data_row = list(ordered_data.values())

    try:
        # Append the new row to the worksheet
        worksheet.append_row(data_row)
        print(f"\n--- LOGGED TO CLOUD: {participant_id} ---")
        print(f"Data: {ordered_data}")

        return True

    except Exception as e:
        print(f"CRITICAL ERROR saving data to Google Sheets for {participant_id}: {e}")
        return False

# --- Flask Routes ---

@app.route('/data', methods=['GET'])
def receive_data():
    """Endpoint to receive sensor data from ESP32 devices via HTTP GET requests."""
    
    # 1. Receive and Parse Data from the URL parameters
    participant_id = request.args.get('id')
    station = request.args.get('station')
    value = request.args.get('value')
    age = request.args.get('age') 

    if not all([participant_id, station, value]):
        return jsonify({"status": "error", "message": "Missing required parameters (id, station, value)"}), 400

    try:
        # Map incoming station names to standardized dictionary keys
        station_map = {
            'reaction': 'reaction_time_ms',
            'balance': 'balance_duration_s',
            'memory': 'memory_score'
        }
        
        feature_key = station_map.get(station.lower())
        if not feature_key:
            return jsonify({"status": "error", "message": f"Unknown station: {station}"}), 400

        # --- 2. Store and Consolidate Data ---
        with DATA_LOCK:
            if participant_id not in SESSION_DATA:
                SESSION_DATA[participant_id] = {}

            # Store age and feature value
            if age and 'actual_age' not in SESSION_DATA[participant_id]:
                SESSION_DATA[participant_id]['actual_age'] = int(age) # Age is an integer
            
            SESSION_DATA[participant_id][feature_key] = float(value) # Value is a float

            print(f"Received: ID={participant_id}, Station={station}, Value={value}")

            # Check for completion (4 total keys: id, age, reaction, balance, memory)
            if len(SESSION_DATA[participant_id]) >= TOTAL_KEYS_CHECK:
                success = log_session_complete(participant_id)
                if success:
                    return jsonify({"status": "SUCCESS", "message": f"Session {participant_id} finished. Data saved to Google Sheets."})
                else:
                    return jsonify({"status": "ERROR", "message": f"Session {participant_id} finished, but failed to save data."})
            else:
                return jsonify({"status": "RECEIVED", "message": f"Data from {station} received for {participant_id}. Waiting for {TOTAL_FEATURES_TO_COLLECT - (len(SESSION_DATA[participant_id]) - 1)} more test(s)."}), 200
                
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid format for age or value. Must be numeric."}), 400
    except Exception as e:
        print(f"Server Error during processing: {e}")
        return jsonify({"status": "error", "message": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    setup_google_sheets()
    # Run the server on the entire local network (0.0.0.0) so the ESP32 can connect
    print("\n--- Starting Flask Backend Server ---")
    print(f"Accessible at: http://<LAPTOP_IP_ADDRESS>:5000/data")
    print("-------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
