from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import sys
import time

import pandas as pd
import requests


MLB_BASE = "https://statsapi.mlb.com/api/v1"

EXPORT_DIR = Path("exports")
ENV_FEATURES_PATH = Path("exports/features_with_environment_lab.parquet")

CANDIDATE_FEATURES = [
    "vs_hand_games_scaled_diff",
    "env_temp",
    "opp_adj_offense_diff",
]


# =========================
# Helpers
# =========================

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_hand(value):
    if value is None or pd.isna(value):
        return "U"

    value = str(value).upper().strip()

    if value in {"L", "LEFT", "LEFTY"}:
        return "L"

    if value in {"R", "RIGHT", "RIGHTY"}:
        return "R"

    if value in {"S", "SWITCH"}:
        return "S"

    return "U"


def get_target_date():
    if len(sys.argv) >= 2:
        return datetime.strptime(sys.argv[1], "%Y-%m-%d").date()

    return datetime.now(ZoneInfo("America/New_York")).date()


def regular_season_start_for_year(year):
    return date(year, 3, 1)


def get_game_status(game):
    return game.get("status", {}).get("detailedState", "")


def get_game_time_et(game):
    game_datetime = game.get("gameDate")

    if not game_datetime:
        return "TBD"

    try:
        dt = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        et = dt.astimezone(ZoneInfo("America/New_York"))

        try:
            return et.strftime("%-I:%M %p")
        except Exception:
            return et.strftime("%I:%M %p").lstrip("0")

    except Exception:
        return "TBD"


def get_team_id(game, side):
    return safe_int(
        game.get("teams", {}).get(side, {}).get("team", {}).get("id")
    )


def get_team_name(game, side):
    return (
        game.get("teams", {})
        .get(side, {})
        .get("team", {})
        .get("name", side.title())
    )


def get_team_abbr(game, side):
    team = game.get("teams", {}).get(side, {}).get("team", {})
    return team.get("abbreviation") or team.get("teamName") or team.get("name", side.title())


def get_score(game, side):
    return safe_int(game.get("teams", {}).get(side, {}).get("score"), default=None)


def get_probable_pitcher_id(game, side):
    pitcher = game.get("teams", {}).get(side, {}).get("probablePitcher")

    if not pitcher:
        return 0

    return safe_int(pitcher.get("id"))


def get_probable_pitcher_name(game, side):
    pitcher = game.get("teams", {}).get(side, {}).get("probablePitcher")

    if not pitcher:
        return "TBD"

    return pitcher.get("fullName", "TBD")


def is_final_game(game):
    status = get_game_status(game)
    home_score = get_score(game, "home")
    away_score = get_score(game, "away")

    return status in {"Final", "Game Over", "Completed Early"} and home_score is not None and away_score is not None


# =========================
# MLB API
# =========================

def fetch_schedule_range(start_day, end_day):
    if pd.to_datetime(start_day).date() > pd.to_datetime(end_day).date():
        return {"dates": []}

    params = {
        "sportId": 1,
        "startDate": str(start_day),
        "endDate": str(end_day),
        "gameTypes": "R",
        "hydrate": "team,linescore,probablePitcher",
    }

    r = requests.get(f"{MLB_BASE}/schedule", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def flatten_games(schedule_json):
    games = []

    for date_block in schedule_json.get("dates", []):
        for game in date_block.get("games", []):
            if game.get("gameType") != "R":
                continue
            games.append(game)

    return games


def fetch_people(player_ids):
    player_ids = sorted(set(safe_int(pid) for pid in player_ids if safe_int(pid) > 0))

    if not player_ids:
        return {}

    print(f"\nFetching pitcher handedness for {len(player_ids)} unique pitchers...")

    people = {}
    batch_size = 100

    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i + batch_size]

        params = {
            "personIds": ",".join(str(pid) for pid in batch),
        }

        r = requests.get(f"{MLB_BASE}/people", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for person in data.get("people", []):
            player_id = safe_int(person.get("id"))
            people[player_id] = {
                "full_name": person.get("fullName"),
                "pitch_hand": normalize_hand(
                    person.get("pitchHand", {}).get("code")
                ),
                "bat_side": normalize_hand(
                    person.get("batSide", {}).get("code")
                ),
            }

        print(f"Fetched {min(i + batch_size, len(player_ids))}/{len(player_ids)}")
        time.sleep(0.10)

    return people


def fetch_game_feed(game_pk):
    urls = [
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live",
    ]

    errors = []

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            errors.append(f"{url} -> {type(e).__name__}")

    raise RuntimeError("All live feed attempts failed: " + " | ".join(errors))


# =========================
# Environment / Temperature
# =========================

def load_env_temp_default():
    if ENV_FEATURES_PATH.exists():
        try:
            df = pd.read_parquet(ENV_FEATURES_PATH)

            if "env_temp" in df.columns:
                temp = pd.to_numeric(df["env_temp"], errors="coerce").dropna()

                if len(temp):
                    return float(temp.median())
        except Exception:
            pass

    return 70.0


def parse_temp_from_value(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        temp = float(value)

        if -20 <= temp <= 130:
            return temp

        return None

    text = str(value)

    match = re.search(r"(-?\d{1,3})\s*(?:degrees|degree|deg|°)", text, re.I)

    if match:
        temp = float(match.group(1))

        if -20 <= temp <= 130:
            return temp

    return None


def extract_temperature_from_feed(feed):
    weather = feed.get("gameData", {}).get("weather", {})

    for key in ["temp", "temperature"]:
        temp = parse_temp_from_value(weather.get(key))

        if temp is not None:
            return temp, "game_feed_weather"

    for value in weather.values():
        temp = parse_temp_from_value(value)

        if temp is not None:
            return temp, "game_feed_weather_text"

    info_items = (
        feed.get("liveData", {})
        .get("boxscore", {})
        .get("info", [])
    )

    for item in info_items:
        label = str(item.get("label", "")).lower()
        value = item.get("value", "")

        if "weather" in label:
            temp = parse_temp_from_value(value)

            if temp is not None:
                return temp, "boxscore_weather"

    return None, "missing"


def get_live_env_temp(game_pk, default_temp):
    try:
        feed = fetch_game_feed(game_pk)
        temp, source = extract_temperature_from_feed(feed)

        if temp is not None:
            return temp, source

    except Exception as e:
        return default_temp, f"default_after_error:{type(e).__name__}"

    return default_temp, "default_missing_live_weather"


# =========================
# Opponent-Adjusted Offense
# =========================

def create_team_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "recent": deque(maxlen=10),
        "opponents": [],
        "recent_opponents": deque(maxlen=10),
    }


def update_team_state(state, runs_for, runs_against, opponent_id):
    runs_for = safe_int(runs_for)
    runs_against = safe_int(runs_against)

    win = 1 if runs_for > runs_against else 0

    state["games"] += 1
    state["wins"] += win
    state["runs_for"] += runs_for
    state["runs_against"] += runs_against

    state["recent"].append(
        {
            "win": win,
            "runs_for": runs_for,
            "runs_against": runs_against,
            "run_diff": runs_for - runs_against,
        }
    )

    state["opponents"].append(opponent_id)
    state["recent_opponents"].append(opponent_id)


def snapshot_team_state(state):
    games = state["games"]

    if games <= 0:
        return {
            "win_pct": 0.500,
            "rpg": 4.50,
            "rapg": 4.50,
            "run_diff_pg": 0.00,
            "games_scaled": 0.00,
        }

    rpg = state["runs_for"] / games
    rapg = state["runs_against"] / games

    return {
        "win_pct": state["wins"] / games,
        "rpg": rpg,
        "rapg": rapg,
        "run_diff_pg": rpg - rapg,
        "games_scaled": min(games / 50.0, 1.0),
    }


def schedule_strength_snapshot(team_states, opponent_ids):
    opponent_ids = list(opponent_ids)

    if not opponent_ids:
        return {
            "opp_win_pct": 0.500,
            "opp_rpg": 4.50,
            "opp_rapg": 4.50,
            "opp_run_diff_pg": 0.00,
            "opponents_faced_scaled": 0.00,
        }

    snaps = [snapshot_team_state(team_states[opp_id]) for opp_id in opponent_ids]

    return {
        "opp_win_pct": sum(s["win_pct"] for s in snaps) / len(snaps),
        "opp_rpg": sum(s["rpg"] for s in snaps) / len(snaps),
        "opp_rapg": sum(s["rapg"] for s in snaps) / len(snaps),
        "opp_run_diff_pg": sum(s["run_diff_pg"] for s in snaps) / len(snaps),
        "opponents_faced_scaled": min(len(opponent_ids) / 50.0, 1.0),
    }


def build_team_states_from_history(history_games):
    team_states = defaultdict(create_team_state)

    final_games = [g for g in history_games if is_final_game(g)]

    final_games = sorted(
        final_games,
        key=lambda g: (
            g.get("officialDate", ""),
            g.get("gamePk", 0),
        ),
    )

    for game in final_games:
        home_team_id = get_team_id(game, "home")
        away_team_id = get_team_id(game, "away")

        home_score = get_score(game, "home")
        away_score = get_score(game, "away")

        if home_score is None or away_score is None:
            continue

        update_team_state(
            team_states[home_team_id],
            home_score,
            away_score,
            opponent_id=away_team_id,
        )

        update_team_state(
            team_states[away_team_id],
            away_score,
            home_score,
            opponent_id=home_team_id,
        )

    return team_states


def compute_opp_adj_offense_diff(team_states, home_team_id, away_team_id):
    home = snapshot_team_state(team_states[home_team_id])
    away = snapshot_team_state(team_states[away_team_id])

    home_sos = schedule_strength_snapshot(
        team_states,
        team_states[home_team_id]["opponents"],
    )

    away_sos = schedule_strength_snapshot(
        team_states,
        team_states[away_team_id]["opponents"],
    )

    home_adj_offense = home["rpg"] - home_sos["opp_rapg"]
    away_adj_offense = away["rpg"] - away_sos["opp_rapg"]

    return home_adj_offense - away_adj_offense


# =========================
# Team vs Starter Hand Sample Feature
# =========================

def create_hand_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "recent": deque(maxlen=10),
    }


def create_team_hand_state():
    return {
        "L": create_hand_state(),
        "R": create_hand_state(),
    }


def update_hand_state(state, runs_for, runs_against):
    runs_for = safe_int(runs_for)
    runs_against = safe_int(runs_against)

    win = 1 if runs_for > runs_against else 0

    state["games"] += 1
    state["wins"] += win
    state["runs_for"] += runs_for
    state["runs_against"] += runs_against

    state["recent"].append(
        {
            "win": win,
            "runs_for": runs_for,
            "runs_against": runs_against,
            "run_diff": runs_for - runs_against,
        }
    )


def snapshot_hand_state(state):
    games = state["games"]

    return {
        "games": games,
        "games_scaled": min(games / 50.0, 1.0) if games > 0 else 0.0,
    }


def build_team_hand_states_from_history(history_games, pitcher_hands):
    team_hand_states = defaultdict(create_team_hand_state)

    final_games = [g for g in history_games if is_final_game(g)]

    final_games = sorted(
        final_games,
        key=lambda g: (
            g.get("officialDate", ""),
            g.get("gamePk", 0),
        ),
    )

    usable = 0
    total = 0

    for game in final_games:
        total += 1

        home_team_id = get_team_id(game, "home")
        away_team_id = get_team_id(game, "away")

        home_score = get_score(game, "home")
        away_score = get_score(game, "away")

        if home_score is None or away_score is None:
            continue

        home_starter_id = get_probable_pitcher_id(game, "home")
        away_starter_id = get_probable_pitcher_id(game, "away")

        home_starter_hand = pitcher_hands.get(home_starter_id, "U")
        away_starter_hand = pitcher_hands.get(away_starter_id, "U")

        game_usable = False

        # Home offense faced away starter hand.
        if away_starter_hand in {"L", "R"}:
            update_hand_state(
                team_hand_states[home_team_id][away_starter_hand],
                home_score,
                away_score,
            )
            game_usable = True

        # Away offense faced home starter hand.
        if home_starter_hand in {"L", "R"}:
            update_hand_state(
                team_hand_states[away_team_id][home_starter_hand],
                away_score,
                home_score,
            )
            game_usable = True

        if game_usable:
            usable += 1

    return team_hand_states, usable, total


def compute_vs_hand_games_scaled_diff(
    team_hand_states,
    home_team_id,
    away_team_id,
    home_starter_hand,
    away_starter_hand,
):
    home_starter_hand = normalize_hand(home_starter_hand)
    away_starter_hand = normalize_hand(away_starter_hand)

    if away_starter_hand in {"L", "R"}:
        home_state = team_hand_states[home_team_id][away_starter_hand]
    else:
        home_state = create_hand_state()

    if home_starter_hand in {"L", "R"}:
        away_state = team_hand_states[away_team_id][home_starter_hand]
    else:
        away_state = create_hand_state()

    home_snapshot = snapshot_hand_state(home_state)
    away_snapshot = snapshot_hand_state(away_state)

    return {
        "home_vs_hand_games_scaled": home_snapshot["games_scaled"],
        "away_vs_hand_games_scaled": away_snapshot["games_scaled"],
        "vs_hand_games_scaled_diff": (
            home_snapshot["games_scaled"] - away_snapshot["games_scaled"]
        ),
        "home_vs_hand_games": home_snapshot["games"],
        "away_vs_hand_games": away_snapshot["games"],
    }


# =========================
# Main Feasibility Lab
# =========================

def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    target_date = get_target_date()
    season_start = regular_season_start_for_year(target_date.year)
    history_end = target_date - timedelta(days=1)

    print("=== Candidate v1.1 Live Feature Feasibility Lab ===")
    print(f"Target date: {target_date}")
    print("\nCandidate features to generate live:")
    for feature in CANDIDATE_FEATURES:
        print(f"- {feature}")

    print("\nFetching MLB schedule data...")

    history_json = fetch_schedule_range(
        season_start.isoformat(),
        history_end.isoformat(),
    )

    target_json = fetch_schedule_range(
        target_date.isoformat(),
        target_date.isoformat(),
    )

    history_games = flatten_games(history_json)
    target_games = flatten_games(target_json)

    print(f"Historical games loaded: {len(history_games)}")
    print(f"Target date games loaded: {len(target_games)}")

    if not target_games:
        print("\nNo regular-season MLB games found for target date.")
        return

    all_pitcher_ids = []

    for game in history_games + target_games:
        all_pitcher_ids.append(get_probable_pitcher_id(game, "home"))
        all_pitcher_ids.append(get_probable_pitcher_id(game, "away"))

    people = fetch_people(all_pitcher_ids)

    pitcher_hands = {
        pid: info.get("pitch_hand", "U")
        for pid, info in people.items()
    }

    pitcher_names = {
        pid: info.get("full_name", "")
        for pid, info in people.items()
    }

    known_pitcher_ids = [
        pid for pid in set(all_pitcher_ids)
        if pid > 0 and pitcher_hands.get(pid, "U") in {"L", "R"}
    ]

    print(f"\nKnown pitcher hand IDs: {len(known_pitcher_ids)} / {len(set([pid for pid in all_pitcher_ids if pid > 0]))}")

    print("\nBuilding live-style historical states...")

    team_states = build_team_states_from_history(history_games)

    team_hand_states, usable_hand_history, total_hand_history = (
        build_team_hand_states_from_history(history_games, pitcher_hands)
    )

    print(f"Final historical games usable for hand-state feature: {usable_hand_history} / {total_hand_history}")

    env_temp_default = load_env_temp_default()
    print(f"\nDefault env_temp fallback: {env_temp_default:.1f}")

    rows = []

    print("\nGenerating candidate features for target slate...")

    for game in target_games:
        game_pk = safe_int(game.get("gamePk"))

        home_team_id = get_team_id(game, "home")
        away_team_id = get_team_id(game, "away")

        home_team = get_team_name(game, "home")
        away_team = get_team_name(game, "away")

        home_abbr = get_team_abbr(game, "home")
        away_abbr = get_team_abbr(game, "away")

        home_starter_id = get_probable_pitcher_id(game, "home")
        away_starter_id = get_probable_pitcher_id(game, "away")

        home_starter_hand = pitcher_hands.get(home_starter_id, "U")
        away_starter_hand = pitcher_hands.get(away_starter_id, "U")

        env_temp, env_temp_source = get_live_env_temp(game_pk, env_temp_default)

        opp_adj_offense_diff = compute_opp_adj_offense_diff(
            team_states,
            home_team_id,
            away_team_id,
        )

        hand_features = compute_vs_hand_games_scaled_diff(
            team_hand_states=team_hand_states,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_starter_hand=home_starter_hand,
            away_starter_hand=away_starter_hand,
        )

        both_starter_hands_known = (
            home_starter_hand in {"L", "R"}
            and away_starter_hand in {"L", "R"}
        )

        row = {
            "target_date": target_date.isoformat(),
            "game_pk": game_pk,
            "game": f"{away_abbr} @ {home_abbr}",
            "game_time_et": get_game_time_et(game),
            "status": get_game_status(game),
            "away_team": away_team,
            "home_team": home_team,
            "away_probable_pitcher": get_probable_pitcher_name(game, "away"),
            "home_probable_pitcher": get_probable_pitcher_name(game, "home"),
            "away_starter_id": away_starter_id,
            "home_starter_id": home_starter_id,
            "away_starter_hand": away_starter_hand,
            "home_starter_hand": home_starter_hand,
            "both_starter_hands_known": both_starter_hands_known,
            "env_temp": env_temp,
            "env_temp_source": env_temp_source,
            "opp_adj_offense_diff": opp_adj_offense_diff,
            **hand_features,
        }

        row["feature_complete_numeric"] = all(
            pd.notna(row.get(feature)) for feature in CANDIDATE_FEATURES
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    out_path = EXPORT_DIR / f"candidate_v1_1_live_feature_feasibility_{target_date.isoformat()}.csv"
    latest_path = EXPORT_DIR / "candidate_v1_1_live_feature_feasibility_latest.csv"

    out.to_csv(out_path, index=False)
    out.to_csv(latest_path, index=False)

    print("\n=== Live Candidate Feature Output ===")
    display_cols = [
        "game",
        "game_time_et",
        "status",
        "away_probable_pitcher",
        "away_starter_hand",
        "home_probable_pitcher",
        "home_starter_hand",
        "env_temp",
        "env_temp_source",
        "opp_adj_offense_diff",
        "vs_hand_games_scaled_diff",
        "home_vs_hand_games",
        "away_vs_hand_games",
        "feature_complete_numeric",
    ]

    print(out[display_cols].to_string(index=False))

    print("\n=== Feasibility Summary ===")

    total_games = len(out)
    complete_numeric = int(out["feature_complete_numeric"].sum())
    both_hand_known = int(out["both_starter_hands_known"].sum())
    actual_env_temp = int(~out["env_temp_source"].astype(str).str.contains("default", case=False, na=False).sum())

    env_default_count = int(out["env_temp_source"].astype(str).str.contains("default", case=False, na=False).sum())

    print(f"Games evaluated: {total_games}")
    print(f"Numeric candidate feature rows: {complete_numeric} / {total_games}")
    print(f"Both starter hands known: {both_hand_known} / {total_games}")
    print(f"Actual live env_temp found: {total_games - env_default_count} / {total_games}")
    print(f"Default env_temp used: {env_default_count} / {total_games}")

    print("\nSaved:")
    print(f"- {out_path}")
    print(f"- {latest_path}")

    print("\n=== Feasibility Verdict ===")

    if complete_numeric == total_games and both_hand_known == total_games and env_default_count == 0:
        print("PASS: All candidate features can be generated live with actual values.")
        print("Next step: implement candidate v1.1 live feature generation in the Streamlit app.")

    elif complete_numeric == total_games and both_hand_known == total_games:
        print("PARTIAL PASS: Candidate features are numeric for all games and starter hands are known.")
        print("Warning: env_temp required fallback/default for at least one game.")
        print("Next step: decide whether default env_temp is acceptable or add a real weather source before shipping.")

    elif complete_numeric == total_games:
        print("PARTIAL PASS: Candidate features are numeric for all games.")
        print("Warning: at least one starter hand was unknown, so vs_hand_games_scaled_diff may be neutral/defaulted.")
        print("Next step: improve probable pitcher hand coverage before shipping candidate v1.1.")

    else:
        print("FAIL: At least one candidate feature could not be generated numerically.")
        print("Do not attempt to ship candidate v1.1 yet.")


if __name__ == "__main__":
    main()