import ssl
import json
import gspread
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT   = 8883

TOPIC_CMD  = "cs3237/A0277110N/balance/cmd"   # to ESP32
TOPIC_DATA = "cs3237/A0277110N/balance/data"  # from ESP32

SERVICE_ACCOUNT_FILE = "service_account.json"
SPREADSHEET_NAME     = "CS3237_Health_Assessment_Data"

gc = None
worksheet = None

def setup_google_sheets():
    global gc, worksheet
    try:
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        sh = gc.open(SPREADSHEET_NAME)
        worksheet = sh.sheet1
        print(f"Connected to Google Sheet: {SPREADSHEET_NAME}")
        return True
    except Exception as e:
        print(f"[CRITICAL] Google Sheets setup failed: {e}")
        return False

def log_session_data(participant_id: str, age: int, gender: str, balance_duration_s: float) -> bool:
    ordered = {
        "participant_id": participant_id,
        "actual_age": age,
        "gender": gender,
        "reaction_time_ms": 0.0,
        "balance_duration_s": balance_duration_s,
        "memory_score": 0
    }
    try:
        worksheet.append_row(list(ordered.values()), value_input_option="USER_ENTERED")
        print(f"[GSPREAD] Logged row: {ordered}")
        return True
    except Exception as e:
        print(f"[CRITICAL] Append failed: {e}")
        return False

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("MQTT connected.")
        client.subscribe(TOPIC_DATA, qos=1)
        print(f"Subscribed: {TOPIC_DATA}")
    else:
        print(f"MQTT connect failed rc={rc}")

def on_message(client, userdata, msg):
    if msg.topic != TOPIC_DATA:
        return
    raw = msg.payload.decode("utf-8", errors="replace").strip()
    print("=" * 50)
    print("DATA RECEIVED")
    print(raw)
    try:
        obj = json.loads(raw)
        if obj.get("test_type") != "balance":
            print("[SKIP] Non-balance message ignored")
            print("=" * 50)
            return
        pid    = str(obj.get("participant_id", "UNKNOWN"))
        age    = int(obj.get("age", 0))
        gender = str(obj.get("gender", "O"))
        trial  = int(obj.get("trial_index", 1))
        if "duration_s" in obj:
            duration_s = float(obj["duration_s"])
        elif "duration_ms" in obj:
            duration_s = float(obj["duration_ms"]) / 1000.0
        else:
            print("[ERROR] Missing duration field")
            print("=" * 50)
            return

        print(f"Balance -> ID={pid}, age={age}, gender={gender}, trial={trial}, duration_s={duration_s:.3f}")
        while True:
            action = input("Action (SEND to log / DELETE to discard): ").strip().upper()
            if action == "SEND":
                ok = log_session_data(pid, age, gender, duration_s)
                print(f"[LOG] {'OK' if ok else 'FAILED'}")
                break
            elif action == "DELETE":
                print("[LOG] Deleted locally; not appended")
                break
            else:
                print("Invalid input. Type SEND or DELETE.")

        while True:
            nxt = input("Next trial? (YES to proceed / NO to hold): ").strip().upper()
            if nxt == "YES":
                client.publish(TOPIC_CMD, json.dumps({"cmd":"NEXT"}), qos=1)
                print("[CMD] NEXT sent")
                break
            elif nxt == "NO":
                client.publish(TOPIC_CMD, json.dumps({"cmd":"HOLD"}), qos=1)
                print("[CMD] HOLD sent")
                break
            else:
                print("Invalid input. Type YES or NO.")
    except json.JSONDecodeError:
        print("[ERROR] Payload is not JSON; ignoring")
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
    print("=" * 50)

if __name__ == "__main__":
    if not setup_google_sheets():
        raise SystemExit(1)

    # Gather operator inputs and send to device
    participant_id = input("Participant ID (e.g., P001): ").strip() or "UNKNOWN"
    age_str = input("Age (number): ").strip() or "0"
    gender   = input("Gender (M/F/Other): ").strip().upper() or "O"
    age = int(age_str) if age_str.isdigit() else 0

    client = mqtt.Client(transport="tcp")
    client.on_connect = on_connect
    client.on_message = on_message
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
    client.connect(BROKER, PORT, keepalive=60)

    # Start network loop
    client.loop_start()

    # Send metadata and start command
    meta = {"cmd":"SET_META", "participant_id": participant_id, "age": age, "gender": gender}
    client.publish(TOPIC_CMD, json.dumps(meta), qos=1)
    client.publish(TOPIC_CMD, json.dumps({"cmd":"START"}), qos=1)
    print("[CMD] SET_META and START sent")

    # Keep the script running for operator interaction
    try:
        while True:
            pass
    except KeyboardInterrupt:
        client.loop_stop()
        client.disconnect()
