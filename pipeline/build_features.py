import pandas as pd
from pathlib import Path
from collections import deque

def safe_div(a, b, default=0):
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default

def build_features(games):
    out_path = Path("data/processed/features_v3_starting_pitchers.parquet")

    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    team_state = {}
    pitcher_state = {}
    rows = []

    def get_team_state(team_id):
        if team_id not in team_state:
            team_state[team_id] = {
                "games": 0, "wins": 0, "runs_for": 0, "runs_against": 0,
                "recent": deque(maxlen=10)
            }
        return team_state[team_id]

    def get_pitcher_state(pitcher_id):
        if pd.isna(pitcher_id) or pitcher_id is None:
            pitcher_id = "UNKNOWN"
        if pitcher_id not in pitcher_state:
            pitcher_state[pitcher_id] = {
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
        return pitcher_state[pitcher_id]

    def recent_team_stats(state):
        recent = list(state["recent"])
        if not recent:
            return {"recent_win_pct": 0.5, "recent_rpg": 4.3, "recent_rapg": 4.3}
        n = len(recent)
        return {
            "recent_win_pct": sum(x[0] for x in recent) / n,
            "recent_rpg": sum(x[1] for x in recent) / n,
            "recent_rapg": sum(x[2] for x in recent) / n,
        }

    def starter_features(state, current_date):
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

    for _, g in games.iterrows():
        current_date = g["date"]

        home_id = g["home_team_id"]
        away_id = g["away_team_id"]

        hs = get_team_state(home_id)
        aw = get_team_state(away_id)

        home_ps = get_pitcher_state(g.get("home_starter_id"))
        away_ps = get_pitcher_state(g.get("away_starter_id"))

        home_sp = starter_features(home_ps, current_date)
        away_sp = starter_features(away_ps, current_date)

        if hs["games"] >= 10 and aw["games"] >= 10:
            home_win_pct = hs["wins"] / hs["games"]
            away_win_pct = aw["wins"] / aw["games"]
            home_rpg = hs["runs_for"] / hs["games"]
            away_rpg = aw["runs_for"] / aw["games"]
            home_rapg = hs["runs_against"] / hs["games"]
            away_rapg = aw["runs_against"] / aw["games"]
            home_run_diff_pg = home_rpg - home_rapg
            away_run_diff_pg = away_rpg - away_rapg
            home_recent = recent_team_stats(hs)
            away_recent = recent_team_stats(aw)

            rows.append({
                "game_pk": g["game_pk"],
                "date": g["date"],
                "season": g["season"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_starter_name": g.get("home_starter_name"),
                "away_starter_name": g.get("away_starter_name"),

                "win_pct_diff": home_win_pct - away_win_pct,
                "rpg_diff": home_rpg - away_rpg,
                "rapg_diff": away_rapg - home_rapg,
                "run_diff_per_game_diff": home_run_diff_pg - away_run_diff_pg,

                "recent_win_pct_diff": home_recent["recent_win_pct"] - away_recent["recent_win_pct"],
                "recent_rpg_diff": home_recent["recent_rpg"] - away_recent["recent_rpg"],
                "recent_rapg_diff": away_recent["recent_rapg"] - home_recent["recent_rapg"],

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

                "home_field": 1,
                "home_win": g["home_win"],
            })

        home_win = int(g["home_win"] == 1)
        away_win = 1 - home_win

        hs["games"] += 1
        hs["wins"] += home_win
        hs["runs_for"] += g["home_score"]
        hs["runs_against"] += g["away_score"]
        hs["recent"].append((home_win, g["home_score"], g["away_score"]))

        aw["games"] += 1
        aw["wins"] += away_win
        aw["runs_for"] += g["away_score"]
        aw["runs_against"] += g["home_score"]
        aw["recent"].append((away_win, g["away_score"], g["home_score"]))

        def update_pitcher_state(ps, prefix):
            ip = float(g.get(f"{prefix}_starter_ip", 0) or 0)
            if ip <= 0:
                return
            ps["starts"] += 1
            ps["ip"] += ip
            ps["hits"] += int(g.get(f"{prefix}_starter_hits", 0) or 0)
            ps["er"] += int(g.get(f"{prefix}_starter_er", 0) or 0)
            ps["bb"] += int(g.get(f"{prefix}_starter_bb", 0) or 0)
            ps["so"] += int(g.get(f"{prefix}_starter_so", 0) or 0)
            ps["hr"] += int(g.get(f"{prefix}_starter_hr", 0) or 0)
            ps["pitches"] += int(g.get(f"{prefix}_starter_pitches", 0) or 0)
            ps["last_start_date"] = current_date
            ps["recent_starts"].append((
                ip,
                int(g.get(f"{prefix}_starter_er", 0) or 0),
                int(g.get(f"{prefix}_starter_hits", 0) or 0),
                int(g.get(f"{prefix}_starter_bb", 0) or 0),
                int(g.get(f"{prefix}_starter_so", 0) or 0),
            ))

        update_pitcher_state(home_ps, "home")
        update_pitcher_state(away_ps, "away")

    features = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    print(f"Saved {len(features)} feature rows to {out_path}")
    return features
