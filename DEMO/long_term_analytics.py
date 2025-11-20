# long_term_analytics.py
import datetime as dt
import gspread
import pandas as pd
import matplotlib.pyplot as plt

LONG_SHEET_NAME = "LongTerm"

LONG_HEADERS = [
    "participant_id",
    "participant_gender",
    "reaction_time_ms",
    "balance_duration_s",
    "memory_score",
    "cognitive_age",
    "real_age",
    "date"
]

def ensure_longterm_sheet(SPREADSHEET):
    try:
        ws = SPREADSHEET.worksheet(LONG_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = SPREADSHEET.add_worksheet(title=LONG_SHEET_NAME, rows=2000, cols=12)
    if ws.row_values(1) != LONG_HEADERS:
        ws.update("A1", [LONG_HEADERS])
    return ws

def _normalize_date_str(date_str: str) -> str:
    # Accepts 'YYYY-MM-DD' or 'DD/MM/YYYY' or 'YYYY/MM/DD'
    s = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            d = dt.datetime.strptime(s, fmt).date()
            return d.isoformat()  # YYYY-MM-DD
        except:
            pass
    return dt.date.today().isoformat()

def upsert_longterm_row(ws_long, participant_id: str, gender: str,
                        reaction_time_ms: str|float|None,
                        balance_duration_s: str|float|None,
                        memory_score: str|float|None,
                        cognitive_age: str|float|None,
                        real_age: str|float|None,
                        date_str: str):
    """
    If (participant_id, date) exists, update the row's columns.
    Else, append a new row.
    """
    pid = str(participant_id)
    g   = str(gender)
    d   = _normalize_date_str(date_str)

    rt  = "" if reaction_time_ms in [None, ""] else str(reaction_time_ms)
    bd  = "" if balance_duration_s in [None, ""] else str(balance_duration_s)
    ms  = "" if memory_score in [None, ""] else str(memory_score)
    ca  = "" if cognitive_age in [None, ""] else str(round(float(cognitive_age), 2))
    ra  = "" if real_age in [None, ""] else str(real_age)

    # Try to find existing row with same pid + date
    vals = ws_long.get_all_values()
    header = vals[0] if vals else LONG_HEADERS
    rows = vals[1:] if len(vals) > 1 else []

    col_idx = {name: i for i, name in enumerate(header)}
    found_row_idx = None
    for r_i, r in enumerate(rows, start=2):  # sheet rows start at 1; data at 2
        if len(r) < len(header):
            r = r + [""]*(len(header)-len(r))
        if r[col_idx["participant_id"]] == pid and r[col_idx["date"]] == d:
            found_row_idx = r_i
            break

    new_row = [""] * len(header)
    new_row[col_idx["participant_id"]]      = pid
    new_row[col_idx["participant_gender"]]  = g
    new_row[col_idx["reaction_time_ms"]]    = rt
    new_row[col_idx["balance_duration_s"]]  = bd
    new_row[col_idx["memory_score"]]        = ms
    new_row[col_idx["cognitive_age"]]       = ca
    new_row[col_idx["real_age"]]            = ra
    new_row[col_idx["date"]]                = d

    if found_row_idx is None:
        ws_long.append_row(new_row, value_input_option="USER_ENTERED", table_range="A1")
        return ("append", d)
    else:
        # merge with existing: fetch old row and fill only provided fields if blanks
        old = ws_long.row_values(found_row_idx)
        old = old + [""]*(len(header)-len(old))
        for name, val in zip(header, new_row):
            if val != "":
                old[col_idx[name]] = val
        rng = f"A{found_row_idx}:{chr(ord('A')+len(header)-1)}{found_row_idx}"
        ws_long.update(rng, [old], value_input_option="USER_ENTERED")
        return ("update", d)

def plot_cognitive_vs_real(ws_long, participant_id: str):
    """Fetch participant rows and plot cognitive vs real age over time."""
    vals = ws_long.get_all_records(expected_headers=LONG_HEADERS)
    df = pd.DataFrame(vals)
    if df.empty:
        print("No long-term data yet.")
        return

    df = df[df['participant_id'] == participant_id].copy()
    if df.empty:
        print(f"No rows for {participant_id}.")
        return

    # Normalize & sort dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    # Convert numeric cols
    for col in ['cognitive_age', 'real_age']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Days since first
    t0 = df['date'].min()
    df['days_since_first'] = (df['date'] - t0).dt.days

    # Plot cognitive vs real over time
    plt.figure(figsize=(9,5))
    plt.plot(df['date'], df['cognitive_age'], marker='o', label='Cognitive age')
    plt.plot(df['date'], df['real_age'], marker='s', label='Real age')
    plt.xlabel("Date")
    plt.ylabel("Age (years)")
    plt.title(f"Long‑term: {participant_id} cognitive vs real age over time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df[['date','days_since_first','cognitive_age','real_age']]
