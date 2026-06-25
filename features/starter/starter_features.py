from collections import deque
import pandas as pd

def safe_div(a, b, default=0):
    if b == 0:
        return default
    return a / b

def normalize_pitcher_id(pitcher_id):
    if pd.isna(pitcher_id) or pitcher_id is None:
        return "UNKNOWN"
    return pitcher_id

def create_pitcher_state():
    return {
        "starts": 0,
        "ip": 0.0,
        "hits": 0,
        "er": 0,
        "bb": 0,
        "so": 0,
        "hr": 0,
        "pitches": 0,
        "last_start_date": None,
        "recent_starts": deque(maxlen=3),
    }

def starter_snapshot(state, current_date):
    starts = state["starts"]
    if starts < 1 or state["ip"] <= 0:
        return {
            "sp_starts": starts,
            "sp_era": 4.40,
            "sp_whip": 1.30,
            "sp_kbb": 2.40,
            "sp_ip_per_start": 4.8,
            "sp_rest_days": 5,
            "sp_recent_era": 4.40,
        }

    era = 9 * state["er"] / state["ip"]
    whip = (state["hits"] + state["bb"]) / state["ip"]
    kbb = safe_div(state["so"], max(state["bb"], 1), default=2.4)
    ip_per_start = safe_div(state["ip"], starts, default=4.8)

    if state["last_start_date"] is None:
        rest_days = 5
    else:
        rest_days = max(0, (current_date - state["last_start_date"]).days)

    recent = list(state["recent_starts"])
    if recent:
        recent_ip = sum(x[0] for x in recent)
        recent_er = sum(x[1] for x in recent)
        recent_era = 9 * recent_er / recent_ip if recent_ip > 0 else era
    else:
        recent_era = era

    return {
        "sp_starts": starts,
        "sp_era": era,
        "sp_whip": whip,
        "sp_kbb": kbb,
        "sp_ip_per_start": ip_per_start,
        "sp_rest_days": rest_days,
        "sp_recent_era": recent_era,
    }

def build_starter_features(home_sp, away_sp):
    return {
        "home_sp_starts": home_sp["sp_starts"],
        "away_sp_starts": away_sp["sp_starts"],
        "sp_starts_diff": home_sp["sp_starts"] - away_sp["sp_starts"],
        "home_sp_era": home_sp["sp_era"],
        "away_sp_era": away_sp["sp_era"],
        "sp_era_diff": away_sp["sp_era"] - home_sp["sp_era"],
        "home_sp_whip": home_sp["sp_whip"],
        "away_sp_whip": away_sp["sp_whip"],
        "sp_whip_diff": away_sp["sp_whip"] - home_sp["sp_whip"],
        "home_sp_kbb": home_sp["sp_kbb"],
        "away_sp_kbb": away_sp["sp_kbb"],
        "sp_kbb_diff": home_sp["sp_kbb"] - away_sp["sp_kbb"],
        "home_sp_ip_per_start": home_sp["sp_ip_per_start"],
        "away_sp_ip_per_start": away_sp["sp_ip_per_start"],
        "sp_ip_per_start_diff": home_sp["sp_ip_per_start"] - away_sp["sp_ip_per_start"],
        "home_sp_rest_days": home_sp["sp_rest_days"],
        "away_sp_rest_days": away_sp["sp_rest_days"],
        "sp_rest_diff": home_sp["sp_rest_days"] - away_sp["sp_rest_days"],
        "home_sp_recent_era": home_sp["sp_recent_era"],
        "away_sp_recent_era": away_sp["sp_recent_era"],
        "sp_recent_era_diff": away_sp["sp_recent_era"] - home_sp["sp_recent_era"],
    }

def update_pitcher_state(state, game_row, prefix, current_date):
    ip = float(game_row.get(f"{prefix}_starter_ip", 0) or 0)
    if ip <= 0:
        return

    er = int(game_row.get(f"{prefix}_starter_er", 0) or 0)
    hits = int(game_row.get(f"{prefix}_starter_hits", 0) or 0)
    bb = int(game_row.get(f"{prefix}_starter_bb", 0) or 0)
    so = int(game_row.get(f"{prefix}_starter_so", 0) or 0)

    state["starts"] += 1
    state["ip"] += ip
    state["hits"] += hits
    state["er"] += er
    state["bb"] += bb
    state["so"] += so
    state["hr"] += int(game_row.get(f"{prefix}_starter_hr", 0) or 0)
    state["pitches"] += int(game_row.get(f"{prefix}_starter_pitches", 0) or 0)
    state["last_start_date"] = current_date
    state["recent_starts"].append((ip, er, hits, bb, so))
