# Multi-Test Health Assessment Server with Google Sheets logging
# - Memory_Test: raw row -> 'Memory Test'; predicted age -> 'Predicted_Ages'.Memory
# - Reaction_Test: predicted age -> 'Predicted_Ages'.Reaction (via JSON equation)
# - LongTerm: per (participant_id, date) row with reaction/balance/memory/cognitive/real age
# - Face upload: HTTP endpoint (/upload_face, /upload_face_form) for CNN age; logged to Predicted_Ages + LongTerm
# - CLI: start <test>, long term analytics <P_ID>, quit

import ssl
import json
import re
import gspread
import paho.mqtt.client as mqtt
from oauth2client.service_account import ServiceAccountCredentials
from datetime import date as _date

# External helpers/models
from long_term_analytics import (
    ensure_longterm_sheet,
    upsert_longterm_row,
    plot_cognitive_vs_real,
)

# Face HTTP API (runs alongside MQTT loop)
from face_api import FaceAPI

from use_forward_mem_model import pipe  # for readiness flag

# ======================
# MQTT CONFIG
# ======================
BROKER = "broker.hivemq.com"
PORT = 8883
TOPIC_CMD = "cs3237/A0277110N/health/cmd"

DATA_TOPICS = {
    "reaction": "cs3237/A0277110N/health/data/Reaction_Test",
    "balance":  "cs3237/A0277110N/health/data/Balance_Test",
    "memory":   "cs3237/A0277110N/health/data/Memory_Test",
    "image":    "cs3237/A0277110N/health/data/image"
}
SUBSCRIPTION_TOPICS = [
    (DATA_TOPICS["reaction"], 1),
    (DATA_TOPICS["balance"], 1),
    (DATA_TOPICS["memory"], 1),
    (DATA_TOPICS["image"], 1)
]

# ======================
# Google Sheets CONFIG
# ======================
SERVICE_ACCOUNT_FILE = "service_account.json"
SPREADSHEET_NAME = "CS3237_Health_Assessment_Data"
MEMORY_SHEET_NAME = "Memory Test"
PRED_AGES_SHEET_NAME = "Predicted_Ages"

# Globals for Sheets
gc = None
SH = None
WS_MEMORY = None
WS_PRED = None
WS_LONG = None
worksheet = None  # optional legacy default

# Session state (used to store PID, Gender, and individual predicted ages)
current_participant = {}

# Globals for Face Event Handling
FACE_PRED_QUEUE = []

# ======================
# External models
# ======================
# Memory forward model (joblib pipeline)
try:
    from use_forward_mem_model import estimate_age_from_score
    FORWARD_MODEL_READY = pipe is not None
    print("Forward memory model loaded" if FORWARD_MODEL_READY else "Forward memory model not loaded")
except Exception as e:
    FORWARD_MODEL_READY = False
    print(f"WARNING: Forward memory model not available: {e}")

# Reaction model (loads/trains from Sheets on import)
try:
    from reaction_age_predictor import predict_age_from_reaction
    REACTION_EQ_READY = True
except Exception as e:
    REACTION_EQ_READY = False
    print(f"WARNING: Reaction equation not available: {e}")
    
# Balance model
try:
    from use_balance_model import estimate_age_from_balance
    BALANCE_MODEL_READY = True
except Exception as e:
    BALANCE_MODEL_READY = False
    print(f"WARNING: Balance model not available: {e}")

# ======================
# Sheets setup
# ======================
def setup_google_sheets():
    """Connect and ensure tabs + headers exist."""
    global gc, SH, WS_MEMORY, WS_PRED, WS_LONG, worksheet
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    gc = gspread.authorize(creds)
    SH = gc.open(SPREADSHEET_NAME)

    # Memory Test sheet
    try:
        WS_MEMORY = SH.worksheet(MEMORY_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        WS_MEMORY = SH.add_worksheet(title=MEMORY_SHEET_NAME, rows=1000, cols=10)
    mem_headers = ["participant_id", "participant_gender", "memory_score", "actual_age"]
    if WS_MEMORY.row_values(1) != mem_headers:
        WS_MEMORY.update("A1", [mem_headers])

    # Predicted_Ages sheet
    try:
        WS_PRED = SH.worksheet(PRED_AGES_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        WS_PRED = SH.add_worksheet(title=PRED_AGES_SHEET_NAME, rows=1000, cols=10)
    pred_headers = ["P_Number", "Reaction", "Balance", "Memory", "Face", "Combined"]
    if WS_PRED.row_values(1) != pred_headers:
        WS_PRED.update("A1", [pred_headers])

    # LongTerm sheet
    WS_LONG = ensure_longterm_sheet(SH)

    worksheet = SH.sheet1
    print(f"Connected Sheets: '{MEMORY_SHEET_NAME}', '{PRED_AGES_SHEET_NAME}', 'LongTerm'")

# ======================
# Predicted_Ages helper
# ======================
def append_predicted_age_to_pred_sheet(participant_id: str,
                                       reaction_age: float | None = None,
                                       balance_age: float | None = None,
                                       memory_age: float | None = None,
                                       face_age: float | None = None,
                                       combined_age: float | None = None) -> bool:
    """Append one row to Predicted_Ages with only provided columns filled."""
    try:
        row = [
            str(participant_id),
            "" if reaction_age is None else round(float(reaction_age), 2),
            "" if balance_age  is None else round(float(balance_age),  2),
            "" if memory_age   is None else round(float(memory_age),   2),
            "" if face_age     is None else round(float(face_age),     2),
            "" if combined_age is None else round(float(combined_age), 2),
        ]
        WS_PRED.append_row(row, value_input_option="USER_ENTERED", table_range="A1")
        print(f"Predicted_Ages row: {row}")
        return True
    except Exception as e:
        print(f"Predicted_Ages append failed: {e}")
        return False

# ======================
# Face age logging hook for Flask API
# ======================
def on_face_age_ready(predicted_age: float, form_participant_id: str, form_gender: str):
    """
    Called by the Flask API. Puts the prediction into a queue to be processed
    in the main thread, allowing for the SEND/DELETE prompt.
    """
    global FACE_PRED_QUEUE
    
    # Prioritize the session ID if the form gave UNKNOWN
    pid = form_participant_id
    if pid == "UNKNOWN":
        pid = current_participant.get("id", "UNKNOWN")

    # Prioritize the session Gender if the form gave UNKNOWN
    gender = form_gender
    if gender == "UNKNOWN":
        gender = current_participant.get("gender", "O")

    # Use session age as the actual age
    age = current_participant.get("age", 0)
    
    # Queue the event for processing in the main loop
    FACE_PRED_QUEUE.append({
        "pid": pid,
        "age": age,
        "gender": gender,
        "predicted_age": predicted_age,
        "date_str": current_participant.get("date", _date.today().isoformat())
    })
    print(f"\n[Face API received] Age: {predicted_age:.2f} for ID: {pid}. Pending SEND/DELETE.")


# ======================
# Combined age with weights
# ======================
def calculate_weighted_age(data: dict) -> float | None:
    """Calculates combined age based on saved session data."""
    weights = {
        "face_age": 0.25,
        "memory_age": 0.3,
        "balance_age": 0.25,
        "reaction_age": 0.2
    }
    
    # Filter for ages that are available (not None and numeric)
    available_ages = {}
    total_weight = 0.0
    
    for key, weight in weights.items():
        age_value = data.get(key)
        if age_value is not None and isinstance(age_value, (float, int)):
            available_ages[key] = age_value
            total_weight += weight

    if not available_ages:
        print("Error: No predicted ages available for combination.")
        return None

    weighted_sum = 0.0
    
    # Normalize weights based on available tests
    if total_weight > 0:
        for key, age_value in available_ages.items():
            normalized_weight = weights[key] / total_weight
            weighted_sum += age_value * normalized_weight

        return round(weighted_sum, 2)
    
    return None

def handle_weighted_age_calc():
    """Calculate weighted age, show it, and optionally append to Predicted_Ages."""
    # Use the individual age predictions stored in the current_participant session
    session_data = {
        "face_age": current_participant.get("face_age"),
        "memory_age": current_participant.get("memory_age"),
        "balance_age": current_participant.get("balance_age"),
        "reaction_age": current_participant.get("reaction_age")
    }

    pid = current_participant.get("id", "UNKNOWN")
    
    weighted_age = calculate_weighted_age(session_data)

    if weighted_age is None:
        print("Weighted age could not be calculated. Need at least one predicted age.")
        return

    print(f"\nCalculated weighted age: {weighted_age}")

    while True:
        action = input("Action (SEND / DELETE): ").strip().upper()
        if action == "SEND":
            # Append combined age to Predicted_Ages (Combined column)
            try:
                append_predicted_age_to_pred_sheet(pid, combined_age=weighted_age)
                print(f"Combined age {weighted_age} appended for {pid}")
            except Exception as e:
                print(f"Combined age append failed: {e}")
            break
        elif action == "DELETE":
            print("Discarded weighted age")
            break
        else:
            print("Type SEND or DELETE")


# ======================
# Unified logger
# ======================
def log_session_data(participant_id: str, age: int, gender: str, test_type: str, value: str) -> tuple[bool, dict]:
    """
    reaction -> Predicted_Ages.Reaction & current_participant['reaction_age']
    memory   -> Memory Test raw row + Predicted_Ages.Memory & current_participant['memory_age']
    balance  -> Predicted_Ages.Balance & current_participant['balance_age']
    
    Returns (ok, context) where context may include {'cognitive_age': float}
    """
    overall_ok = True
    context = {}

    try:
        t = test_type.lower()
        if t == "reaction":
            try:
                avg_ms = float(re.sub(r"[^0-9.+-]", "", str(value)))
            except:
                print(f"Reaction value not numeric: {value}")
                return False, context

            pred_age_rxn = None
            if REACTION_EQ_READY:
                try:
                    pred_age_rxn = predict_age_from_reaction(gender, avg_ms)
                    print(f"Predicted Reaction Age: {pred_age_rxn:.2f}")
                    context['cognitive_age'] = pred_age_rxn
                    current_participant['reaction_age'] = pred_age_rxn # Save to session
                except Exception as e:
                    print(f"Reaction predictor error: {e}")

            ok = append_predicted_age_to_pred_sheet(participant_id, reaction_age=pred_age_rxn)
            overall_ok = overall_ok and ok
        
        elif t == "balance":
            try:
                duration_s = float(re.sub(r"[^0-9.+-]", "", str(value)))
            except:
                print(f"Balance value not numeric: {value}")
                return False, context

            pred_age_bal = None
            if BALANCE_MODEL_READY:
                try:
                    pred_age_bal, _ = estimate_age_from_balance(gender, duration_s)
                    print(f"Predicted Balance Age: {pred_age_bal:.2f}")
                    context['cognitive_age'] = pred_age_bal
                    current_participant['balance_age'] = pred_age_bal # Save to session
                except Exception as e:
                    print(f"Balance predictor error: {e}")

            ok = append_predicted_age_to_pred_sheet(participant_id, balance_age=pred_age_bal)
            overall_ok = overall_ok and ok

        elif t == "memory":
            # Raw row to 'Memory Test'
            mem_raw = str(value)
            row_memory = [str(participant_id), str(gender), mem_raw, str(age)]
            try:
                WS_MEMORY.append_row(row_memory, value_input_option="USER_ENTERED", table_range="A1")
                print(f"Memory row -> '{MEMORY_SHEET_NAME}': {row_memory}")
            except Exception as e:
                print(f"Memory sheet append failed: {e}")
                overall_ok = False

            # Predicted age to Predicted_Ages.Memory
            pred_age_mem = None
            try:
                mem_num = float(re.sub(r"[^0-9.+-]", "", mem_raw))
            except:
                mem_num = None
            if FORWARD_MODEL_READY and mem_num is not None:
                try:
                    gmap = {"M": "Male", "F": "Female", "O": "Other"}
                    g_for_model = gmap.get(str(gender).upper(), "Other")
                    est_age, _ = estimate_age_from_score(g_for_model, float(mem_num))
                    pred_age_mem = est_age
                    context['cognitive_age'] = pred_age_mem
                    current_participant['memory_age'] = pred_age_mem # Save to session
                    print(f"Predicted Memory Age: {pred_age_mem:.2f}")
                except Exception as e:
                    print(f"Memory model estimation failed: {e}")

            ok = append_predicted_age_to_pred_sheet(participant_id, memory_age=pred_age_mem)
            overall_ok = overall_ok and ok

        elif t == "image":
            # Image is processed via Flask/FACE_PRED_QUEUE
            print(f"Image event acknowledged: {value}")
        else:
            print(f"Unknown test_type '{test_type}'")
            return False, context

        return overall_ok, context

    except Exception as e:
        print(f"Logger error: {type(e).__name__}: {e}")
        return False, context

# ======================
# MQTT callbacks
# ======================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("MQTT connected")
        for topic, qos in SUBSCRIPTION_TOPICS:
            r, _ = client.subscribe(topic, qos=qos)
            print(f" Subscribed {topic} -> {r}")
    else:
        print(f"MQTT connect failed rc={rc}")

def on_message(client, userdata, msg):
    topic = msg.topic
    raw = msg.payload.decode("utf-8", errors="replace").strip()
    print("\n" + "="*60)
    print(f"{topic} :: {raw}")

    m = re.search(r'/health/data/(.+)', topic)
    if not m:
        print("Unexpected topic")
        return
    test_suffix = m.group(1)
    test_type = test_suffix.lower().replace("_test", "").replace("_", "")

    try:
        obj = json.loads(raw)
        
        pid = current_participant.get("id", "UNKNOWN")
        age = current_participant.get("age", 0)
        gender = current_participant.get("gender", "O")
        date_for_row = current_participant.get("date", _date.today().isoformat())
        
        if obj.get("participant_id"):
             pid = str(obj["participant_id"])
        if obj.get("age"):
             try: age = int(obj["age"])
             except: pass
        if obj.get("gender"):
             gender = str(obj["gender"])

        rxn_ms = ""
        bal_s  = ""
        mem_sc = ""

        if test_type == "reaction":
            v = obj.get("average_reaction_ms", 0)
            rxn_ms = round(float(v), 3)
            value = str(rxn_ms)
        elif test_type == "balance":
            duration_ms = obj.get("duration_ms", 0)
            duration_s = obj.get("duration_s", 0)
            if duration_s:
                 bal_s = round(float(duration_s), 3)
            elif duration_ms:
                 bal_s = round(float(duration_ms)/1000.0, 3)
            value = str(bal_s) if bal_s != "" else ""
        elif test_type == "memory":
            v = obj.get("memory_score", 0)
            mem_sc = float(v)
            value = str(mem_sc)
        elif test_type == "image":
            # Image is handled by the Flask API/FACE_PRED_QUEUE for confirmation
            value = str(obj.get("image_path", obj.get("status", "captured")))
            if value == "captured":
                 print("Image captured message received. Awaiting face API upload.")
            return # Exit function for image type

        print(f"Parsed -> type={test_type} pid={pid} gender={gender} value={value}")

        while True:
            action = input("Action (SEND / DELETE): ").strip().upper()
            if action == "SEND":
                ok, ctx = log_session_data(pid, age, gender, test_type, value)

                # Cognitive age if computed for this message
                lt_cog  = ctx.get("cognitive_age", "")
                lt_real = current_participant.get("age", "")

                # Upsert LongTerm with the cleaned numeric fields
                try:
                    upsert_longterm_row(
                        WS_LONG,
                        participant_id=pid,
                        gender=gender,
                        reaction_time_ms=rxn_ms,
                        balance_duration_s=bal_s,
                        memory_score=mem_sc,
                        cognitive_age=lt_cog,
                        real_age=lt_real,
                        date_str=date_for_row
                    )
                    print("LongTerm updated")
                except Exception as e:
                    print(f"LongTerm upsert failed: {e}")

                print("Logged" if ok else "Log failed")
                break
            elif action == "DELETE":
                print("Discarded")
                break
            else:
                print("Type SEND or DELETE")

    except Exception as e:
        print(f"on_message error: {type(e).__name__}: {e}")

# ======================
# Command publish
# ======================
def publish_command(client, test_type: str):
    cmd_map = {
        "reaction": "START_REACTION",
        "balance":  "START_BALANCE",
        "memory":   "START_MEMORY",
        "image":    "START_IMAGE"
    }
    if test_type not in cmd_map:
        print(f"Invalid test type: {test_type}")
        return False
    payload = json.dumps({"cmd": cmd_map[test_type]})
    r, _ = client.publish(TOPIC_CMD, payload, qos=1)
    print("Publish OK" if r == mqtt.MQTT_ERR_SUCCESS else f"Publish failed: {r}")
    return r == mqtt.MQTT_ERR_SUCCESS

# ======================
# Main
# ======================
def main():
    print("Starting server...")
    setup_google_sheets()

    print("\nENTER PARTICIPANT INFORMATION")
    participant_id = input("Participant ID (e.g., P001): ").strip() or "UNKNOWN"

    # Optional real age (used in LongTerm)
    try:
        age = int(input("Real age (years, optional): ") or "0")
    except:
        age = 0

    gender_input = input("Gender (M/F/O for Other): ").strip().upper()
    gender = gender_input if gender_input in ["M","F","O"] else "O"

    date_input = input("Date (YYYY-MM-DD, default=today): ").strip()
    date_str = date_input or _date.today().isoformat()

    current_participant["id"] = participant_id
    current_participant["age"] = age
    current_participant["gender"] = gender
    current_participant["date"] = date_str

    # Start the face upload API (runs in background thread)
    face_api = FaceAPI(on_face_age_ready)
    face_api.run_async(host="0.0.0.0", port=5000)
    print("Open from phone (same Wi‑Fi): http://<laptop-local-IP>:5000/upload_face_form")

    client = mqtt.Client(transport="tcp")
    client.on_connect = on_connect
    client.on_message = on_message

    if PORT == 8883:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
        print("TLS enabled")

    print(f"Connecting to {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()

    meta = {"cmd": "SET_META", "participant_id": participant_id, "age": age, "gender": gender}
    client.publish(TOPIC_CMD, json.dumps(meta), qos=1)

    print("\nCommands: start reaction | start balance | start memory | start image | combined age | long term analytics <P_ID> | quit")
    try:
        while True:
            # Process face prediction events from queue
            if FACE_PRED_QUEUE:
                face_event = FACE_PRED_QUEUE.pop(0)
                pid = face_event["pid"]
                pred_age = face_event["predicted_age"]
                gender = face_event["gender"]
                date_for_row = face_event["date_str"]
                real_age = face_event["age"]

                # Save face age to session for combined calculation
                current_participant['face_age'] = pred_age

                print("\n" + "="*60)
                print(f"[Face Prediction] ID: {pid}, Age: {pred_age:.2f}")

                while True:
                    action = input("Action (SEND / DELETE): ").strip().upper()
                    if action == "SEND":
                        # Log to Predicted_Ages
                        try:
                            append_predicted_age_to_pred_sheet(pid, face_age=pred_age)
                        except Exception as e:
                            print(f"Predicted_Ages face append failed: {e}")

                        # Upsert LongTerm (Face age is the cognitive age)
                        try:
                            upsert_longterm_row(
                                WS_LONG,
                                participant_id=pid,
                                gender=gender,
                                reaction_time_ms="",
                                balance_duration_s="",
                                memory_score="",
                                cognitive_age=pred_age,
                                real_age=real_age,
                                date_str=date_for_row
                            )
                            print(f"LongTerm face age updated: {pid} {date_for_row} -> {pred_age:.2f}")
                        except Exception as e:
                            print(f"LongTerm face upsert failed: {e}")

                        print("Logged Face data")
                        break
                    elif action == "DELETE":
                        print("Discarded Face data")
                        # Clear age from session if discarded
                        current_participant['face_age'] = None 
                        break
                    else:
                        print("Type SEND or DELETE")
                
                # Immediately prompt for the next command after processing the event
                print("\nCommands: start reaction | start balance | start memory | start image | combined age | long term analytics <P_ID> | quit")
                continue
            # -------------------------------------------------------------

            cmd = input("> ").strip().lower()
            if cmd == "quit":
                break
            if cmd.startswith("start "):
                t = cmd.split(" ", 1)[1]
                publish_command(client, t)
                continue
            if cmd.startswith("combined age"):
                handle_weighted_age_calc()
                continue
            if cmd.startswith("long term analytics"):
                parts = cmd.split()
                pid = parts[-1] if len(parts) >= 4 else current_participant.get("id","UNKNOWN")
                plot_cognitive_vs_real(WS_LONG, pid)
                continue
            print("Unknown command")
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        print("Stopped.")

if __name__ == "__main__":
    main()
