# Python 3.x: Fixed Multi-Test Health Assessment Server with Robust Google Sheets Logging
# Supports Reaction_Test, Balance_Test, Memory_Test, Image via specific MQTT topics
# Fixed append_row issues with table_range and error handling

import ssl
import json
import gspread
import paho.mqtt.client as mqtt
import re
from oauth2client.service_account import ServiceAccountCredentials
import sys

# --- MQTT CONFIGURATION ---
BROKER = "broker.hivemq.com"
PORT = 8883  # Use 1883 if TLS causes issues

TOPIC_CMD = "cs3237/A0277110N/health/cmd"  # Unified commands to devices

# Specific data topics (devices publish to these)
DATA_TOPICS = {
    "reaction": "cs3237/A0277110N/health/data/Reaction_Test",
    "balance": "cs3237/A0277110N/health/data/Balance_Test",
    "memory": "cs3237/A0277110N/health/data/Memory_Test",
    "image": "cs3237/A0277110N/health/data/image"
}

# Subscription list for all topics
SUBSCRIPTION_TOPICS = [
    (DATA_TOPICS["reaction"], 1),
    (DATA_TOPICS["balance"], 1),
    (DATA_TOPICS["memory"], 1),
    (DATA_TOPICS["image"], 1)
]

# --- GOOGLE SHEETS CONFIGURATION ---
SERVICE_ACCOUNT_FILE = "service_account.json"
SPREADSHEET_NAME = "CS3237_Health_Assessment_Data"

# --- GLOBAL VARIABLES ---
gc = None
worksheet = None
current_participant = {}

def setup_google_sheets():
    """Initialize Google Sheets connection with error handling"""
    global gc, worksheet

    try:
        # Scope for Google Sheets API
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        # Authenticate with service account
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            SERVICE_ACCOUNT_FILE, scope
        )
        gc = gspread.authorize(creds)

        # Open spreadsheet
        sh = gc.open(SPREADSHEET_NAME)
        worksheet = sh.sheet1  # First worksheet

        print(f"Successfully connected to Google Sheet: '{SPREADSHEET_NAME}'")
        print(f"Worksheet title: '{worksheet.title}'")
        print(f"Current row count: {len(worksheet.get_all_values())}")

        # Ensure headers exist (optional - create if sheet is empty)
        headers = ["participant_id", "actual_age", "gender", "reaction_time_ms",
                  "balance_duration_s", "memory_score"]
        current_headers = worksheet.row_values(1)
        if len(current_headers) == 0 or current_headers[0] != "participant_id":
            worksheet.append_row(headers)
            print("Added header row to sheet")

        return True

    except FileNotFoundError:
        print(f"ERROR: Service account file '{SERVICE_ACCOUNT_FILE}' not found!")
        print("Make sure 'service_account.json' is in the same directory")
        return False
    except gspread.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet '{SPREADSHEET_NAME}' not found!")
        print("Share the sheet with your service account email")
        return False
    except Exception as e:
        print(f"CRITICAL: Google Sheets setup failed: {e}")
        print(f"Full error: {type(e).__name__}: {str(e)}")
        return False

def log_session_data(participant_id: str, age: int, gender: str, test_type: str, value: str) -> bool:
    """
    Log data to Google Sheet with fixed column alignment and error handling
    All values converted to strings for consistent formatting
    """
    try:
        # Map test type to appropriate sheet column
        if test_type == "reaction":
            ordered_values = [
                str(participant_id),      # A: participant_id
                str(age),                 # B: actual_age
                str(gender),              # C: gender
                str(value),       # D: reaction_time_ms
                "NUL",                    # E: balance_duration_s
                "NUL"                     # F: memory_score
            ]
            print(f"Logging REACTION: {participant_id}, Age {age}, {value}ms")

        elif test_type == "balance":
            ordered_values = [
                str(participant_id),
                str(age),
                str(gender),
                "NUL",
                str(value),        # E: balance_duration_s
                "NUL"
            ]
            print(f"Logging BALANCE: {participant_id}, Age {age}, {value}s")

        elif test_type == "memory":
            ordered_values = [
                str(participant_id),
                str(age),
                str(gender),
                "NUL",
                "NUL",
                str(value)    # F: memory_score
            ]
            print(f"Logging MEMORY: {participant_id}, Age {age}, {value} points")

        elif test_type == "image":
            ordered_values = [
                str(participant_id),
                str(age),
                str(gender),
                "NUL",
                "NUL",
                str(value)                # F: image status
            ]
            print(f"Logging IMAGE: {participant_id}, Age {age}, {value}")

        else:
            print(f"ERROR: Unknown test_type '{test_type}'")
            return False

        # Append row with explicit table range to ensure column alignment
        result = worksheet.append_row(
            ordered_values,
            value_input_option="USER_ENTERED",  # Preserve formatting
            table_range="A1"  # Start from A1 to maintain column structure
        )

        # Verify success
        if result and hasattr(result, 'updatedRows'):
            row_number = len(worksheet.get_all_values())
            print(f"SUCCESS: Data appended to row {row_number}")
            print(f"Logged values: {ordered_values}")
            return True
        else:
            print(f"WARNING: Append returned unexpected result: {result}")
            return False

    except gspread.exceptions.APIError as e:
        print(f"API Error during append: {e}")
        print(f"Response: {e.response}")
        return False
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet not found in spreadsheet")
        return False
    except Exception as e:
        print(f"Unexpected error during append: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def on_connect(client, userdata, flags, rc, properties=None):
    """Handle MQTT connection and subscribe to all test topics"""
    if rc == 0:
        print("MQTT Connected to broker")

        # Subscribe to all data topics
        for topic, qos in SUBSCRIPTION_TOPICS:
            result, mid = client.subscribe(topic, qos=qos)
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"   Subscribed to {topic} (QoS {qos})")
            else:
                print(f"   Failed to subscribe to {topic}: {result}")

        print(f"Monitoring {len(SUBSCRIPTION_TOPICS)} test topics")
        print("Ready for test data...")

    else:
        print(f"MQTT Connection failed with return code: {rc}")
        print("Error codes: 1=Unspecified, 2=Invalid Protocol, 3=Invalid Client ID, 4=Bad Username/Password, 5=Timeout")

def on_message(client, userdata, msg):
    """Process incoming test data from devices"""
    full_topic = msg.topic
    print("\n" + "="*60)
    print(f"MESSAGE RECEIVED")
    print(f"Topic: {full_topic}")
    print(f"QoS: {msg.qos}")
    print(f"Retain: {msg.retain}")

    # Decode payload
    raw = msg.payload.decode("utf-8", errors="replace").strip()
    print(f"Raw payload: {raw}")

    # Extract test type from topic
    match = re.search(r'/health/data/(.+)', full_topic)
    if not match:
        print("Topic doesn't match expected pattern: /health/data/<test>")
        print("="*60)
        return

    test_suffix = match.group(1)
    test_type = test_suffix.lower().replace("_test", "").replace("_", "")

    print(f"Parsed test type: '{test_type}' from '{test_suffix}'")

    # Validate test type
    if test_type not in ["reaction", "balance", "memory", "image"]:
        print(f"Unknown test type: {test_type}")
        print("="*60)
        return

    try:
        # Parse JSON payload
        obj = json.loads(raw)
        print(f"JSON parsed successfully")

        # Extract participant info (fallback to stored data)
        pid = str(obj.get("participant_id", current_participant.get("id", "UNKNOWN")))
        age = int(obj.get("age", current_participant.get("age", 0)))
        gender = str(obj.get("gender", current_participant.get("gender", "O")))

        # Extract test-specific value
        value = "NUL"
        if test_type == "reaction":
            value = str(obj.get("average_reaction_ms", 0))
            print(f"REACTION DATA: ID={pid}, Age={age}, Gender={gender}, Time={value}ms")
        elif test_type == "balance":
            # Handle seconds or milliseconds
            duration_ms = obj.get("duration_ms", 0)
            duration_s = obj.get("duration_s", 0)
            if duration_ms != 0:
                value = str(round(float(duration_ms) / 1000, 3))
            else:
                value = str(round(float(duration_s), 3))
            print(f"BALANCE DATA: ID={pid}, Age={age}, Gender={gender}, Duration={value}s")
        elif test_type == "memory":
            value = str(obj.get("memory_score", "NUL"))
            print(f"MEMORY DATA: ID={pid}, Age={age}, Gender={gender}, Score={value}")
        elif test_type == "image":
            value = str(obj.get("image_path", obj.get("status", "captured")))
            print(f"IMAGE DATA: ID={pid}, Age={age}, Gender={gender}, Status={value}")

        # Operator confirmation (SEND/DELETE)
        print(f"\nDATA RECEIVED FOR {test_type.upper()}")
        print(f"Participant: {pid}, Age: {age}, Gender: {gender}")
        print(f"Value: {value}")
        print(f"Topic: {full_topic}")

        while True:
            action = input("\nAction (SEND to log / DELETE to discard): ").strip().upper()

            if action == "SEND":
                print("Attempting to log data...")
                success = log_session_data(pid, age, gender, test_type, value)

                if success:
                    print("Data successfully logged to Google Sheets!")
                else:
                    print("Failed to log data - check console for errors")
                break

            elif action == "DELETE":
                print("Data discarded - not logged to sheet")
                break

            else:
                print("Invalid input. Please type 'SEND' or 'DELETE'")

        print("="*60)

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw payload was: {raw[:100]}...")
        print("="*60)

    except Exception as e:
        print(f"Unexpected error processing message: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("="*60)

def publish_command(client, test_type: str):
    """Publish command to start specific test"""
    cmd_map = {
        "reaction": "START_REACTION",
        "balance": "START_BALANCE",
        "memory": "START_MEMORY",
        "image": "START_IMAGE"
    }

    if test_type not in cmd_map:
        print(f"Invalid test type: {test_type}")
        return False

    cmd = cmd_map[test_type]
    payload = json.dumps({"cmd": cmd})

    result, mid = client.publish(TOPIC_CMD, payload, qos=1)
    if result == mqtt.MQTT_ERR_SUCCESS:
        print(f"Command '{cmd}' published successfully (MID: {mid})")
        print(f"Sent to topic: {TOPIC_CMD}")
        print(f"Payload: {payload}")
        return True
    else:
        print(f"Failed to publish command '{cmd}': {result}")
        return False

def main():
    """Main program entry point"""
    print("Starting Multi-Test Health Assessment Server")
    print("="*60)

    # Setup Google Sheets
    if not setup_google_sheets():
        print("\nCannot continue without Google Sheets access")
        input("Press Enter to exit...")
        return

    # Get participant information once
    print("\nENTER PARTICIPANT INFORMATION")
    print("-" * 30)

    participant_id = input("Participant ID (e.g., P001): ").strip()
    if not participant_id:
        participant_id = "UNKNOWN"

    while True:
        try:
            age_input = input("Age (number): ").strip()
            age = int(age_input) if age_input.isdigit() else 0
            break
        except ValueError:
            print("Please enter a valid number for age")

    gender_input = input("Gender (M/F/O for Other): ").strip().upper()
    gender = gender_input if gender_input in ["M", "F", "O"] else "O"

    # Store participant data
    current_participant["id"] = participant_id
    current_participant["age"] = age
    current_participant["gender"] = gender

    print(f"\nPARTICIPANT REGISTERED")
    print(f"ID: {participant_id}")
    print(f"Age: {age}")
    print(f"Gender: {gender}")
    print("-" * 30)

    # Setup MQTT client
    client = mqtt.Client(transport="tcp")
    client.on_connect = on_connect
    client.on_message = on_message

    # Enable TLS if using port 8883
    if PORT == 8883:
        client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        print("TLS security enabled (port 8883)")
    else:
        print("Non-secure connection (port 1883)")

    # Connect to broker
    print(f"\nConnecting to MQTT broker {BROKER}:{PORT}...")
    try:
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()  # Start background network loop
        print("MQTT client started")
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        return

    # Send initial metadata to devices
    print("\nSending participant metadata to all devices...")
    meta_payload = {
        "cmd": "SET_META",
        "participant_id": participant_id,
        "age": age,
        "gender": gender,
        "timestamp": "2025-11-06T15:00:00+08:00"
    }

    result, mid = client.publish(TOPIC_CMD, json.dumps(meta_payload), qos=1)
    if result == mqtt.MQTT_ERR_SUCCESS:
        print(f"Metadata sent successfully (MID: {mid})")
    else:
        print(f"Failed to send metadata: {result}")

    # Main command loop
    print(f"\nSERVER READY - Command Interface")
    print("="*60)
    print("Available commands:")
    print("  'start reaction'   - Start reaction time test")
    print("  'start balance'    - Start balance test")
    print("  'start memory'     - Start memory test")
    print("  'start image'      - Start image capture")
    print("  'quit'             - Shutdown server")
    print("-" * 60)

    try:
        while True:
            cmd_input = input("\nCommand: ").strip().lower()

            if cmd_input == "quit":
                print("\nShutting down server...")
                client.publish(TOPIC_CMD, json.dumps({"cmd": "RESET"}), qos=1)
                client.loop_stop()
                client.disconnect()
                print("Goodbye!")
                break

            elif cmd_input.startswith("start "):
                test_name = cmd_input[6:].strip()
                if test_name in ["reaction", "balance", "memory", "image"]:
                    print(f"\nInitiating {test_name} test...")
                    success = publish_command(client, test_name)
                    if success:
                        print(f"{test_name.upper()} command sent to devices")
                        print(f"Devices should publish results to: {DATA_TOPICS[test_name]}")
                    else:
                        print(f"Failed to send {test_name} command")
                else:
                    print(f"Invalid test: '{test_name}'")
                    print("Valid tests: reaction, balance, memory, image")

            else:
                print("Unknown command. Type 'start <test>' or 'quit'")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("MQTT connections closed")

if __name__ == "__main__":
    main()
