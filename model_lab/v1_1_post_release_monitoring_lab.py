from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import re
import sys
import time

import joblib
import pandas as pd
import requests


MLB_BASE = "https://statsapi.mlb.com/api/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

EXPORT_DIR = Path("exports")

MODEL_V1_0_PATH = Path("model_artifacts/mlb_logistic_model.joblib")
MODEL_V1_1_PATH = Path("model_artifacts/mlb_candidate_v1_1_logistic_model.joblib")

FINAL_STATES = {"Final", "Game Over", "Completed Early"}

V1_0_DEFAULT_FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
    "home_field",
]

V1_1_DEFAULT_FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
    "home_field",
    "vs_hand_games_scaled_diff",
    "env_temp",
    "opp_adj_offense_diff",
]

MANUAL_VENUE_COORDS = {
    "yankee stadium": (40.8296, -73.9262),
    "fenway park": (42.3467, -71.0972),
    "oriole park at camden yards": (39.2840, -76.6217),
    "rogers centre": (43.6414, -79.3894),
    "comerica park": (42.3390, -83.0485),
    "progressive field": (41.4962, -81.6852),
    "target field": (44.9817, -93.2776),
    "kauffman stadium": (39.0517, -94.4803),
    "rate field": (41.8300, -87.6339),
    "guaranteed rate field": (41.8300, -87.6339),
    "pnc park": (40.4469, -80.0057),
    "great american ball park": (39.0978, -84.5066),
    "citi field": (40.7571, -73.8458),
    "wrigley field": (41.9484, -87.6553),
    "american family field": (43.0280, -87.9712),
    "busch stadium": (38.6226, -90.1928),
    "coors field": (39.7561, -104.9942),
    "chase field": (33.4455, -112.0667),
    "tropicana field": (27.7682, -82.6534),
    "loandepot park": (25.7781, -80.2197),
    "angel stadium": (33.8003, -117.8827),
    "petco park": (32.7073, -117.1573),
    "dodger stadium": (34.0739, -118.2400),
    "oracle park": (37.7786, -122.3893),
    "truist park": (33.8908, -84.4678),
    "tmobile park": (47.5914, -122.3325),
    "globe life field": (32.7473, -97.0842),
    "minute maid park": (29.7573, -95.3555),
    "daikin park": (29.7573, -95.3555),
    "oakland coliseum": (37.7516, -122.2005),
    "sutter health park": (38.5804, -121.5139),
    "citizens bank park": (39.9061, -75.1665),
    "nationals park": (38.8730, -77.0074),
}


# =========================
# Helpers
# =========================

def normalize_key(value):
    value = str(value or "").lower().strip()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
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


def get_game_utc_datetime(game):
    game_datetime = game.get("gameDate")

    if not game_datetime:
        return None

    try:
        return datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
    except Exception:
        return None


def get_game_time_et(game):
    dt = get_game_utc_datetime(game)

    if dt is None:
        return "TBD"

    et = dt.astimezone(ZoneInfo("America/New_York"))

    try:
        return et.strftime("%-I:%M %p")
    except Exception:
        return et.strftime("%I:%M %p").lstrip("0")


def get_team_id(game, side):
    return safe_int(game.get("teams", {}).get(side, {}).get("team", {}).get("id"))


def get_team_name(game, side):
    return game.get("teams", {}).get(side, {}).get("team", {}).get("name", side.title())


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

    return status in FINAL_STATES and home_score is not None and away_score is not None


def clip_prob(p):
    return min(max(float(p), 1e-6), 1 - 1e-6)


def binary_log_loss(y, p_home):
    p = clip_prob(p_home)
    y = int(y)

    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def brier_score(y, p_home):
    return (float(p_home) - int(y)) ** 2


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.1%}"


# =========================
# Model Loading
# =========================

def infer_model_features(model, default_features):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return default_features.copy()


def load_model_artifact(path, default_features, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} model artifact missing: {path}"
        )

    artifact = joblib.load(path)

    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("pipeline") or artifact.get("estimator")
        features = (
            artifact.get("features")
            or artifact.get("feature_names")
            or artifact.get("selected_features")
        )

        if model is None:
            raise ValueError(f"{label} artifact is a dict but no model was found.")

        if features is None:
            features = infer_model_features(model, default_features)

        return model, list(features), artifact

    model = artifact
    features = infer_model_features(model, default_features)

    return model, features, {}


def predict_home_prob(model, features, feature_dict):
    X = pd.DataFrame([feature_dict])

    for f in features:
        if f not in X.columns:
            X[f] = 0.0

    X = X[features].astype(float)

    return float(model.predict_proba(X)[0][1])


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
        "hydrate": "team,venue,linescore,probablePitcher",
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
                "pitch_hand": normalize_hand(person.get("pitchHand", {}).get("code")),
                "bat_side": normalize_hand(person.get("batSide", {}).get("code")),
            }

        time.sleep(0.05)

    return people


def fetch_game_feed(game_pk):
    urls = [
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue

    return None


# =========================
# Team State Features
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
            "run_diff_per_game": 0.00,
            "run_diff_pg": 0.00,
            "recent_win_pct": 0.500,
            "recent_rpg": 4.50,
            "recent_rapg": 4.50,
            "recent_run_diff_pg": 0.00,
            "games_scaled": 0.00,
        }

    win_pct = state["wins"] / games
    rpg = state["runs_for"] / games
    rapg = state["runs_against"] / games
    run_diff_pg = rpg - rapg

    recent = list(state["recent"])

    if recent:
        recent_win_pct = sum(g["win"] for g in recent) / len(recent)
        recent_rpg = sum(g["runs_for"] for g in recent) / len(recent)
        recent_rapg = sum(g["runs_against"] for g in recent) / len(recent)
        recent_run_diff_pg = sum(g["run_diff"] for g in recent) / len(recent)
    else:
        recent_win_pct = win_pct
        recent_rpg = rpg
        recent_rapg = rapg
        recent_run_diff_pg = run_diff_pg

    return {
        "win_pct": win_pct,
        "rpg": rpg,
        "rapg": rapg,
        "run_diff_per_game": run_diff_pg,
        "run_diff_pg": run_diff_pg,
        "recent_win_pct": recent_win_pct,
        "recent_rpg": recent_rpg,
        "recent_rapg": recent_rapg,
        "recent_run_diff_pg": recent_run_diff_pg,
        "games_scaled": min(games / 50.0, 1.0),
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


def build_base_team_features_for_game(game, team_states):
    home_team_id = get_team_id(game, "home")
    away_team_id = get_team_id(game, "away")

    home_snapshot = snapshot_team_state(team_states[home_team_id])
    away_snapshot = snapshot_team_state(team_states[away_team_id])

    return {
        "win_pct_diff": home_snapshot["win_pct"] - away_snapshot["win_pct"],
        "rpg_diff": home_snapshot["rpg"] - away_snapshot["rpg"],
        "rapg_diff": away_snapshot["rapg"] - home_snapshot["rapg"],
        "run_diff_per_game_diff": (
            home_snapshot["run_diff_per_game"] - away_snapshot["run_diff_per_game"]
        ),
        "recent_win_pct_diff": (
            home_snapshot["recent_win_pct"] - away_snapshot["recent_win_pct"]
        ),
        "recent_rpg_diff": home_snapshot["recent_rpg"] - away_snapshot["recent_rpg"],
        "recent_rapg_diff": (
            away_snapshot["recent_rapg"] - home_snapshot["recent_rapg"]
        ),
        "home_field": 1.0,
    }


# =========================
# Hand Feature
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

    for game in final_games:
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

        if away_starter_hand in {"L", "R"}:
            update_hand_state(
                team_hand_states[home_team_id][away_starter_hand],
                home_score,
                away_score,
            )

        if home_starter_hand in {"L", "R"}:
            update_hand_state(
                team_hand_states[away_team_id][home_starter_hand],
                away_score,
                home_score,
            )

    return team_hand_states


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
# Weather Feature
# =========================

def get_venue_id(game):
    return safe_int(game.get("venue", {}).get("id"))


def get_venue_name(game):
    return game.get("venue", {}).get("name", "Unknown Venue")


def extract_coords_from_venue_obj(venue):
    if not isinstance(venue, dict):
        return None, None, None

    possible_locations = [
        venue.get("location", {}),
        venue.get("venues", [{}])[0].get("location", {}) if venue.get("venues") else {},
    ]

    for loc in possible_locations:
        if not isinstance(loc, dict):
            continue

        possible_coord_objs = [
            loc.get("defaultCoordinates", {}),
            loc.get("coordinates", {}),
            loc,
        ]

        for coord_obj in possible_coord_objs:
            if not isinstance(coord_obj, dict):
                continue

            lat = coord_obj.get("latitude") or coord_obj.get("lat") or coord_obj.get("y")
            lon = coord_obj.get("longitude") or coord_obj.get("lng") or coord_obj.get("lon") or coord_obj.get("x")

            lat = safe_float(lat)
            lon = safe_float(lon)

            if lat is not None and lon is not None:
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon, "mlb_venue_coordinates"

    return None, None, None


def fetch_venue_details(venue_id):
    if not venue_id:
        return None

    urls = [
        f"{MLB_BASE}/venues/{venue_id}",
        f"{MLB_BASE}/venues?venueIds={venue_id}",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue

    return None


def get_venue_coordinates(game):
    venue = game.get("venue", {})
    venue_name = get_venue_name(game)
    venue_id = get_venue_id(game)

    lat, lon, source = extract_coords_from_venue_obj(venue)

    if lat is not None and lon is not None:
        return lat, lon, source

    details = fetch_venue_details(venue_id)

    if details:
        lat, lon, source = extract_coords_from_venue_obj(details)

        if lat is not None and lon is not None:
            return lat, lon, source

    manual = MANUAL_VENUE_COORDS.get(normalize_key(venue_name))

    if manual:
        return manual[0], manual[1], "manual_venue_coordinates"

    return None, None, "missing_coordinates"


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


def extract_temperature_from_mlb_feed(feed):
    if not feed:
        return None, "mlb_feed_missing"

    weather = feed.get("gameData", {}).get("weather", {})

    for key in ["temp", "temperature"]:
        temp = parse_temp_from_value(weather.get(key))

        if temp is not None:
            return temp, "mlb_game_feed_weather"

    for value in weather.values():
        temp = parse_temp_from_value(value)

        if temp is not None:
            return temp, "mlb_game_feed_weather_text"

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
                return temp, "mlb_boxscore_weather"

    return None, "mlb_temp_unavailable"


def fetch_open_meteo_hourly_temp(lat, lon, game_utc_dt):
    if game_utc_dt is None:
        return None, None, None, "missing_game_time"

    target_et_date = game_utc_dt.astimezone(ZoneInfo("America/New_York")).date()

    start_date = target_et_date - timedelta(days=1)
    end_date = target_et_date + timedelta(days=1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    r = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return None, None, None, "open_meteo_missing_hourly"

    utc_offset_seconds = safe_int(data.get("utc_offset_seconds"), default=0)
    game_local_naive = (game_utc_dt + timedelta(seconds=utc_offset_seconds)).replace(tzinfo=None)

    best = None

    for t, temp in zip(times, temps):
        if temp is None:
            continue

        try:
            forecast_dt = datetime.fromisoformat(str(t))
        except Exception:
            continue

        diff_minutes = abs((forecast_dt - game_local_naive).total_seconds()) / 60.0

        if best is None or diff_minutes < best["diff_minutes"]:
            best = {
                "temp": float(temp),
                "forecast_hour": str(t),
                "diff_minutes": diff_minutes,
            }

    if best is None:
        return None, None, None, "open_meteo_no_matching_temp"

    return (
        best["temp"],
        best["forecast_hour"],
        best["diff_minutes"],
        "open_meteo_forecast",
    )


def get_env_temp_for_game(game):
    game_pk = safe_int(game.get("gamePk"))
    game_utc_dt = get_game_utc_datetime(game)

    feed = fetch_game_feed(game_pk)
    mlb_temp, mlb_source = extract_temperature_from_mlb_feed(feed)

    if mlb_temp is not None:
        return {
            "env_temp": mlb_temp,
            "env_temp_source": mlb_source,
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    lat, lon, coord_source = get_venue_coordinates(game)

    if lat is None or lon is None or game_utc_dt is None:
        return {
            "env_temp": 74.0,
            "env_temp_source": "default_missing_coordinates_or_time",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    try:
        forecast_temp, forecast_hour, diff_minutes, forecast_source = (
            fetch_open_meteo_hourly_temp(float(lat), float(lon), game_utc_dt)
        )

        if forecast_temp is not None:
            return {
                "env_temp": forecast_temp,
                "env_temp_source": forecast_source,
                "forecast_hour": forecast_hour,
                "forecast_hour_diff_minutes": round(diff_minutes, 1),
            }

        return {
            "env_temp": 74.0,
            "env_temp_source": f"default_after_{forecast_source}",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    except Exception as e:
        return {
            "env_temp": 74.0,
            "env_temp_source": f"default_after_open_meteo_error:{type(e).__name__}",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }


# =========================
# Monitoring
# =========================

def build_model_prediction(prefix, model, features, feature_dict, home_abbr, away_abbr):
    home_prob = predict_home_prob(model, features, feature_dict)
    away_prob = 1 - home_prob

    if home_prob >= away_prob:
        pick_side = "Home"
        pick_team = home_abbr
        model_prob = home_prob
    else:
        pick_side = "Away"
        pick_team = away_abbr
        model_prob = away_prob

    return {
        f"{prefix}_home_prob": home_prob,
        f"{prefix}_away_prob": away_prob,
        f"{prefix}_pick_side": pick_side,
        f"{prefix}_pick_team": pick_team,
        f"{prefix}_model_prob": model_prob,
    }


def add_actual_metrics(row, prefix):
    if not row["is_final"]:
        row[f"{prefix}_correct"] = pd.NA
        row[f"{prefix}_log_loss"] = pd.NA
        row[f"{prefix}_brier"] = pd.NA
        return row

    y = int(row["actual_home_win"])
    p_home = float(row[f"{prefix}_home_prob"])

    row[f"{prefix}_correct"] = row[f"{prefix}_pick_side"] == row["actual_side"]
    row[f"{prefix}_log_loss"] = binary_log_loss(y, p_home)
    row[f"{prefix}_brier"] = brier_score(y, p_home)

    return row


def summarize_model(df, prefix, scope):
    final_df = df[df["is_final"] == True].copy()

    row = {
        "scope": scope,
        "model": prefix,
        "final_games": len(final_df),
        "accuracy": pd.NA,
        "avg_log_loss": pd.NA,
        "avg_brier": pd.NA,
        "top1_accuracy": pd.NA,
        "top3_accuracy": pd.NA,
        "top5_accuracy": pd.NA,
    }

    if final_df.empty:
        return row

    final_df[f"{prefix}_correct"] = final_df[f"{prefix}_correct"].astype(bool)

    row["accuracy"] = float(final_df[f"{prefix}_correct"].mean())
    row["avg_log_loss"] = float(pd.to_numeric(final_df[f"{prefix}_log_loss"], errors="coerce").mean())
    row["avg_brier"] = float(pd.to_numeric(final_df[f"{prefix}_brier"], errors="coerce").mean())

    ranked = final_df.sort_values(f"{prefix}_model_prob", ascending=False).copy()

    for n in [1, 3, 5]:
        top = ranked.head(min(n, len(ranked)))

        if len(top):
            row[f"top{n}_accuracy"] = float(top[f"{prefix}_correct"].mean())

    return row


def build_comparison_summary(df, scope):
    final_df = df[df["is_final"] == True].copy()

    row = {
        "scope": scope,
        "final_games": len(final_df),
        "changed_picks": pd.NA,
        "same_picks": pd.NA,
        "v1_1_accuracy_when_changed": pd.NA,
        "v1_0_accuracy_when_changed": pd.NA,
    }

    if final_df.empty:
        return row

    changed = final_df[final_df["changed_pick"] == True].copy()
    same = final_df[final_df["changed_pick"] == False].copy()

    row["changed_picks"] = len(changed)
    row["same_picks"] = len(same)

    if len(changed):
        changed["v1_1_correct"] = changed["v1_1_correct"].astype(bool)
        changed["v1_0_correct"] = changed["v1_0_correct"].astype(bool)

        row["v1_1_accuracy_when_changed"] = float(changed["v1_1_correct"].mean())
        row["v1_0_accuracy_when_changed"] = float(changed["v1_0_correct"].mean())

    return row


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    target_date = get_target_date()
    season_start = regular_season_start_for_year(target_date.year)
    history_end = target_date - timedelta(days=1)

    run_time_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M:%S %p ET")

    print("=== v1.1 Post-Release Monitoring Lab ===")
    print(f"Target date: {target_date}")
    print(f"Run time: {run_time_et}")
    print("\nPurpose:")
    print("- Track v1.1 picks vs actual winners")
    print("- Compare v1.1 against old production v1.0")
    print("- Monitor accuracy, log loss, Brier, and top-ranked pick performance")

    print("\nLoading models...")

    model_v1_0, features_v1_0, artifact_v1_0 = load_model_artifact(
        MODEL_V1_0_PATH,
        V1_0_DEFAULT_FEATURES,
        "v1_0",
    )

    model_v1_1, features_v1_1, artifact_v1_1 = load_model_artifact(
        MODEL_V1_1_PATH,
        V1_1_DEFAULT_FEATURES,
        "v1_1",
    )

    print(f"v1.0 feature count: {len(features_v1_0)}")
    print(f"v1.1 feature count: {len(features_v1_1)}")

    print("\nFetching schedule data...")

    if history_end >= season_start:
        history_json = fetch_schedule_range(
            season_start.isoformat(),
            history_end.isoformat(),
        )
        history_games = flatten_games(history_json)
    else:
        history_games = []

    target_json = fetch_schedule_range(
        target_date.isoformat(),
        target_date.isoformat(),
    )
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

    print(f"\nFetching pitcher handedness for {len(set([p for p in all_pitcher_ids if p]))} unique pitchers...")

    people = fetch_people(all_pitcher_ids)

    pitcher_hands = {
        pid: info.get("pitch_hand", "U")
        for pid, info in people.items()
    }

    print("Building historical states...")

    team_states = build_team_states_from_history(history_games)
    team_hand_states = build_team_hand_states_from_history(history_games, pitcher_hands)

    rows = []

    print("\nGenerating predictions...")

    for game in target_games:
        game_pk = safe_int(game.get("gamePk"))

        home_team_id = get_team_id(game, "home")
        away_team_id = get_team_id(game, "away")

        home_name = get_team_name(game, "home")
        away_name = get_team_name(game, "away")

        home_abbr = get_team_abbr(game, "home")
        away_abbr = get_team_abbr(game, "away")

        home_score = get_score(game, "home")
        away_score = get_score(game, "away")
        final = is_final_game(game)

        if final:
            actual_home_win = 1 if home_score > away_score else 0
            actual_side = "Home" if actual_home_win == 1 else "Away"
            actual_winner = home_abbr if actual_home_win == 1 else away_abbr
        else:
            actual_home_win = pd.NA
            actual_side = pd.NA
            actual_winner = pd.NA

        home_starter_id = get_probable_pitcher_id(game, "home")
        away_starter_id = get_probable_pitcher_id(game, "away")

        home_starter_hand = pitcher_hands.get(home_starter_id, "U")
        away_starter_hand = pitcher_hands.get(away_starter_id, "U")

        feature_dict = build_base_team_features_for_game(game, team_states)

        env = get_env_temp_for_game(game)

        hand_features = compute_vs_hand_games_scaled_diff(
            team_hand_states=team_hand_states,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_starter_hand=home_starter_hand,
            away_starter_hand=away_starter_hand,
        )

        feature_dict.update(
            {
                "env_temp": env["env_temp"],
                "opp_adj_offense_diff": compute_opp_adj_offense_diff(
                    team_states,
                    home_team_id,
                    away_team_id,
                ),
                **hand_features,
            }
        )

        row = {
            "prediction_run_at_et": run_time_et,
            "target_date": target_date.isoformat(),
            "game_pk": game_pk,
            "game": f"{away_abbr} @ {home_abbr}",
            "game_time_et": get_game_time_et(game),
            "status": get_game_status(game),
            "away_team": away_name,
            "home_team": home_name,
            "away_abbr": away_abbr,
            "home_abbr": home_abbr,
            "away_probable_pitcher": get_probable_pitcher_name(game, "away"),
            "home_probable_pitcher": get_probable_pitcher_name(game, "home"),
            "away_starter_hand": away_starter_hand,
            "home_starter_hand": home_starter_hand,
            "away_score": away_score,
            "home_score": home_score,
            "is_final": final,
            "actual_home_win": actual_home_win,
            "actual_side": actual_side,
            "actual_winner": actual_winner,
            "env_temp": env["env_temp"],
            "env_temp_source": env["env_temp_source"],
            "opp_adj_offense_diff": feature_dict["opp_adj_offense_diff"],
            "vs_hand_games_scaled_diff": feature_dict["vs_hand_games_scaled_diff"],
            "home_vs_hand_games": feature_dict["home_vs_hand_games"],
            "away_vs_hand_games": feature_dict["away_vs_hand_games"],
        }

        row.update(
            build_model_prediction(
                "v1_0",
                model_v1_0,
                features_v1_0,
                feature_dict,
                home_abbr,
                away_abbr,
            )
        )

        row.update(
            build_model_prediction(
                "v1_1",
                model_v1_1,
                features_v1_1,
                feature_dict,
                home_abbr,
                away_abbr,
            )
        )

        row["changed_pick"] = row["v1_1_pick_team"] != row["v1_0_pick_team"]
        row["v1_1_prob_minus_v1_0_prob"] = row["v1_1_model_prob"] - row["v1_0_model_prob"]

        row = add_actual_metrics(row, "v1_0")
        row = add_actual_metrics(row, "v1_1")

        rows.append(row)

    out = pd.DataFrame(rows)

    out["v1_0_rank"] = out["v1_0_model_prob"].rank(ascending=False, method="first").astype(int)
    out["v1_1_rank"] = out["v1_1_model_prob"].rank(ascending=False, method="first").astype(int)

    out = out.sort_values("v1_1_rank").copy()

    current_summary_rows = [
        summarize_model(out, "v1_0", f"target_date_{target_date.isoformat()}"),
        summarize_model(out, "v1_1", f"target_date_{target_date.isoformat()}"),
    ]

    current_summary = pd.DataFrame(current_summary_rows)
    comparison_summary = pd.DataFrame(
        [build_comparison_summary(out, f"target_date_{target_date.isoformat()}")]
    )

    prediction_path = EXPORT_DIR / f"v1_1_monitoring_predictions_{target_date.isoformat()}.csv"
    latest_path = EXPORT_DIR / "v1_1_monitoring_latest.csv"
    summary_path = EXPORT_DIR / f"v1_1_monitoring_summary_{target_date.isoformat()}.csv"
    comparison_path = EXPORT_DIR / f"v1_1_monitoring_comparison_{target_date.isoformat()}.csv"
    log_path = EXPORT_DIR / "v1_1_monitoring_log.csv"
    cumulative_summary_path = EXPORT_DIR / "v1_1_monitoring_cumulative_summary.csv"

    out.to_csv(prediction_path, index=False)
    out.to_csv(latest_path, index=False)
    current_summary.to_csv(summary_path, index=False)
    comparison_summary.to_csv(comparison_path, index=False)

    if log_path.exists():
        old_log = pd.read_csv(log_path)
        old_log = old_log[old_log["target_date"].astype(str) != target_date.isoformat()].copy()
        log_df = pd.concat([old_log, out], ignore_index=True, sort=False)
    else:
        log_df = out.copy()

    log_df.to_csv(log_path, index=False)

    cumulative_summary = pd.DataFrame(
        [
            summarize_model(log_df, "v1_0", "cumulative"),
            summarize_model(log_df, "v1_1", "cumulative"),
        ]
    )

    cumulative_comparison = pd.DataFrame(
        [build_comparison_summary(log_df, "cumulative")]
    )

    cumulative_full_summary = pd.concat(
        [
            cumulative_summary,
            cumulative_comparison,
        ],
        ignore_index=True,
        sort=False,
    )

    cumulative_full_summary.to_csv(cumulative_summary_path, index=False)

    display = out.copy()

    display["score"] = display.apply(
        lambda r: f"{r['away_score']}-{r['home_score']}" if r["is_final"] else "—",
        axis=1,
    )

    display["v1_0_prob"] = display["v1_0_model_prob"].apply(fmt_pct)
    display["v1_1_prob"] = display["v1_1_model_prob"].apply(fmt_pct)

    display_cols = [
        "v1_1_rank",
        "game",
        "game_time_et",
        "status",
        "score",
        "actual_winner",
        "v1_1_pick_team",
        "v1_1_prob",
        "v1_1_correct",
        "v1_0_pick_team",
        "v1_0_prob",
        "v1_0_correct",
        "changed_pick",
        "env_temp",
        "env_temp_source",
    ]

    print("\n=== Monitoring Predictions ===")
    print(display[display_cols].to_string(index=False))

    print("\n=== Same-Day Model Summary ===")
    print(current_summary.to_string(index=False))

    print("\n=== Same-Day v1.1 vs v1.0 Pick Comparison ===")
    print(comparison_summary.to_string(index=False))

    print("\n=== Cumulative Monitoring Summary ===")
    print(cumulative_full_summary.to_string(index=False))

    print("\nSaved:")
    print(f"- {prediction_path}")
    print(f"- {latest_path}")
    print(f"- {summary_path}")
    print(f"- {comparison_path}")
    print(f"- {log_path}")
    print(f"- {cumulative_summary_path}")

    final_count = int(out["is_final"].sum())

    print("\n=== Monitoring Verdict ===")

    if final_count == 0:
        print("PREGAME ONLY: Picks were logged, but no final outcomes are available yet.")
        print("Run this again after the slate finishes to score v1.1 vs v1.0.")
    else:
        v1_0_acc = current_summary.loc[current_summary["model"] == "v1_0", "accuracy"].iloc[0]
        v1_1_acc = current_summary.loc[current_summary["model"] == "v1_1", "accuracy"].iloc[0]

        v1_0_ll = current_summary.loc[current_summary["model"] == "v1_0", "avg_log_loss"].iloc[0]
        v1_1_ll = current_summary.loc[current_summary["model"] == "v1_1", "avg_log_loss"].iloc[0]

        print(f"Final games scored: {final_count} / {len(out)}")
        print(f"v1.0 accuracy: {fmt_pct(v1_0_acc)}")
        print(f"v1.1 accuracy: {fmt_pct(v1_1_acc)}")
        print(f"v1.0 log loss: {v1_0_ll:.6f}")
        print(f"v1.1 log loss: {v1_1_ll:.6f}")

        if v1_1_ll < v1_0_ll:
            print("GOOD DAY: v1.1 beat v1.0 on log loss for this slate.")
        else:
            print("NORMAL VARIANCE: v1.1 did not beat v1.0 on log loss for this slate.")

        print("Keep tracking. Do not overreact to one slate.")


if __name__ == "__main__":
    main()