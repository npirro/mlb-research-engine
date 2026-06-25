from collections import deque


def create_bullpen_state():
    return {
        "games": 0,
        "relief_ip": 0.0,
        "relief_er": 0,
        "relief_hits": 0,
        "relief_bb": 0,
        "relief_so": 0,
        "relief_pitches": 0,
        "recent_games": deque(maxlen=7),
    }


def bullpen_snapshot(state):
    if state["relief_ip"] <= 0:
        return {
            "bp_era": 4.30,
            "bp_whip": 1.30,
            "bp_kbb": 2.40,
            "bp_recent_pitches": 0,
            "bp_recent_ip": 0.0,
            "bp_fatigue": 50,
        }

    bp_era = 9 * state["relief_er"] / state["relief_ip"]
    bp_whip = (state["relief_hits"] + state["relief_bb"]) / state["relief_ip"]
    bp_kbb = state["relief_so"] / max(state["relief_bb"], 1)

    recent_pitches = sum(g["pitches"] for g in state["recent_games"])
    recent_ip = sum(g["ip"] for g in state["recent_games"])

    fatigue = min(100, recent_pitches * 0.35 + recent_ip * 6)

    return {
        "bp_era": bp_era,
        "bp_whip": bp_whip,
        "bp_kbb": bp_kbb,
        "bp_recent_pitches": recent_pitches,
        "bp_recent_ip": recent_ip,
        "bp_fatigue": fatigue,
    }


def build_bullpen_features(home_bp, away_bp):
    return {
        "home_bp_era": home_bp["bp_era"],
        "away_bp_era": away_bp["bp_era"],
        "bp_era_diff": away_bp["bp_era"] - home_bp["bp_era"],

        "home_bp_whip": home_bp["bp_whip"],
        "away_bp_whip": away_bp["bp_whip"],
        "bp_whip_diff": away_bp["bp_whip"] - home_bp["bp_whip"],

        "home_bp_kbb": home_bp["bp_kbb"],
        "away_bp_kbb": away_bp["bp_kbb"],
        "bp_kbb_diff": home_bp["bp_kbb"] - away_bp["bp_kbb"],

        "home_bp_recent_pitches": home_bp["bp_recent_pitches"],
        "away_bp_recent_pitches": away_bp["bp_recent_pitches"],
        "bp_recent_pitches_diff": away_bp["bp_recent_pitches"] - home_bp["bp_recent_pitches"],

        "home_bp_recent_ip": home_bp["bp_recent_ip"],
        "away_bp_recent_ip": away_bp["bp_recent_ip"],
        "bp_recent_ip_diff": away_bp["bp_recent_ip"] - home_bp["bp_recent_ip"],

        "home_bp_fatigue": home_bp["bp_fatigue"],
        "away_bp_fatigue": away_bp["bp_fatigue"],
        "bp_fatigue_diff": away_bp["bp_fatigue"] - home_bp["bp_fatigue"],
    }


def update_bullpen_state(state, relief_ip, relief_er, relief_hits, relief_bb, relief_so, relief_pitches):
    relief_ip = float(relief_ip or 0)

    state["games"] += 1
    state["relief_ip"] += relief_ip
    state["relief_er"] += int(relief_er or 0)
    state["relief_hits"] += int(relief_hits or 0)
    state["relief_bb"] += int(relief_bb or 0)
    state["relief_so"] += int(relief_so or 0)
    state["relief_pitches"] += int(relief_pitches or 0)

    state["recent_games"].append({
        "ip": relief_ip,
        "er": int(relief_er or 0),
        "hits": int(relief_hits or 0),
        "bb": int(relief_bb or 0),
        "so": int(relief_so or 0),
        "pitches": int(relief_pitches or 0),
    })