from collections import defaultdict, deque
from pathlib import Path
import sys
import time

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.registry.feature_sets import (  # noqa: E402
    TEAM_FEATURES,
    STARTER_FEATURES,
    SCHEDULE_FEATURES,
)


MLB_BASE = "https://statsapi.mlb.com/api/v1"

RAW_PATH = Path("data/raw/games_with_starters_2023_2025.parquet")
FEATURES_PATH = Path("data/processed/features_v4_feature_factory.parquet")
LINEUPS_PATH = Path("data/raw/game_lineups_2023_2025.parquet")
PLAYER_HAND_CACHE_PATH = Path("data/raw/player_handedness_cache.parquet")
EXPORT_DIR = Path("exports")


LINEUP_PLATOON_FEATURES = [
    "home_lineup_platoon_adv_pct",
    "away_lineup_platoon_adv_pct",
    "lineup_platoon_adv_pct_diff",
    "home_lineup_opposite_hand_pct",
    "away_lineup_opposite_hand_pct",
    "lineup_opposite_hand_pct_diff",
    "home_lineup_switch_pct",
    "away_lineup_switch_pct",
    "lineup_switch_pct_diff",
    "home_lineup_known_bats_scaled",
    "away_lineup_known_bats_scaled",
    "lineup_known_bats_scaled_diff",
    "home_faces_lhp",
    "away_faces_lhp",
    "starter_hand_matchup_same",
]

TEAM_VS_HAND_FEATURES = [
    "team_vs_hand_win_pct_diff",
    "team_vs_hand_rpg_diff",
    "team_vs_hand_rapg_diff",
    "team_vs_hand_run_diff_pg_diff",
    "team_vs_hand_recent_win_pct_diff",
    "team_vs_hand_recent_rpg_diff",
    "team_vs_hand_recent_rapg_diff",
    "team_vs_hand_recent_run_diff_pg_diff",
    "home_vs_hand_games_scaled",
    "away_vs_hand_games_scaled",
    "vs_hand_games_scaled_diff",
]

PLATOON_FEATURES = LINEUP_PLATOON_FEATURES + TEAM_VS_HAND_FEATURES


# =========================
# General Helpers
# =========================

def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
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


def find_col(df, candidates, required=True, label="column"):
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        print(f"\nCould not find required {label}.")
        print("Tried:")
        for c in candidates:
            print(f"- {c}")

        print("\nAvailable columns:")
        for c in df.columns:
            print(f"- {c}")

        raise ValueError(f"Missing required {label}")

    return None


def find_starter_columns(raw_games):
    home_candidates = [
        "home_starter_id",
        "home_starting_pitcher_id",
        "home_starter_player_id",
        "home_probable_pitcher_id",
        "home_pitcher_id",
        "home_sp_id",
        "home_starter_mlbam_id",
        "home_starting_pitcher_mlbam",
        "home_starting_pitcher_mlbam_id",
        "home_probable_pitcher_mlbam_id",
    ]

    away_candidates = [
        "away_starter_id",
        "away_starting_pitcher_id",
        "away_starter_player_id",
        "away_probable_pitcher_id",
        "away_pitcher_id",
        "away_sp_id",
        "away_starter_mlbam_id",
        "away_starting_pitcher_mlbam",
        "away_starting_pitcher_mlbam_id",
        "away_probable_pitcher_mlbam_id",
    ]

    home_col = find_col(
        raw_games,
        home_candidates,
        required=False,
        label="home starter pitcher id column",
    )

    away_col = find_col(
        raw_games,
        away_candidates,
        required=False,
        label="away starter pitcher id column",
    )

    if home_col is None or away_col is None:
        print("\nCould not automatically detect starter pitcher ID columns.")
        print("Columns containing 'starter' or 'pitcher':")
        for c in raw_games.columns:
            lc = c.lower()
            if "starter" in lc or "pitcher" in lc:
                print(f"- {c}")

        raise ValueError(
            "Starter pitcher columns not found. "
            "Tell me the printed pitcher/starter columns and I’ll adjust the script."
        )

    return home_col, away_col


# =========================
# Player Handedness
# =========================

def load_existing_handedness_cache():
    if PLAYER_HAND_CACHE_PATH.exists():
        cache = pd.read_parquet(PLAYER_HAND_CACHE_PATH)

        if "player_id" in cache.columns:
            cache["player_id"] = cache["player_id"].astype("Int64")

        return cache

    return pd.DataFrame(
        columns=[
            "player_id",
            "full_name",
            "bat_side",
            "pitch_hand",
        ]
    )


def fetch_player_handedness_from_mlb(player_ids):
    player_ids = [safe_int(pid) for pid in player_ids if safe_int(pid) > 0]
    player_ids = sorted(set(player_ids))

    if not player_ids:
        return pd.DataFrame(
            columns=[
                "player_id",
                "full_name",
                "bat_side",
                "pitch_hand",
            ]
        )

    rows = []
    batch_size = 100

    print(f"\nDownloading handedness for {len(player_ids)} missing players...")

    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i + batch_size]

        params = {
            "personIds": ",".join(str(x) for x in batch),
        }

        r = requests.get(f"{MLB_BASE}/people", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for person in data.get("people", []):
            rows.append(
                {
                    "player_id": safe_int(person.get("id")),
                    "full_name": person.get("fullName"),
                    "bat_side": normalize_hand(
                        person.get("batSide", {}).get("code")
                    ),
                    "pitch_hand": normalize_hand(
                        person.get("pitchHand", {}).get("code")
                    ),
                }
            )

        print(f"Fetched {min(i + batch_size, len(player_ids))}/{len(player_ids)}")
        time.sleep(0.15)

    return pd.DataFrame(rows)


def ensure_player_handedness(player_ids):
    PLAYER_HAND_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    player_ids = sorted(set(safe_int(pid) for pid in player_ids if safe_int(pid) > 0))

    cache = load_existing_handedness_cache()

    if len(cache):
        cached_ids = set(cache["player_id"].dropna().astype(int).tolist())
    else:
        cached_ids = set()

    missing_ids = [pid for pid in player_ids if pid not in cached_ids]

    if missing_ids:
        downloaded = fetch_player_handedness_from_mlb(missing_ids)
        cache = pd.concat([cache, downloaded], ignore_index=True)
        cache = cache.drop_duplicates(subset=["player_id"], keep="last")
        cache.to_parquet(PLAYER_HAND_CACHE_PATH, index=False)

    cache["player_id"] = cache["player_id"].astype("Int64")
    cache["bat_side"] = cache["bat_side"].apply(normalize_hand)
    cache["pitch_hand"] = cache["pitch_hand"].apply(normalize_hand)

    return cache


# =========================
# Lineup Normalization
# =========================

def normalize_lineups(lineups):
    df = lineups.copy()

    game_col = find_col(
        df,
        ["game_pk", "gamePk", "game_id"],
        label="lineup game_pk column",
    )

    team_col = find_col(
        df,
        ["team_id", "teamId", "mlb_team_id"],
        label="lineup team_id column",
    )

    player_col = find_col(
        df,
        ["player_id", "person_id", "batter_id", "mlbam_id", "id"],
        label="lineup player_id column",
    )

    batting_order_col = find_col(
        df,
        ["batting_order", "battingOrder", "order", "lineup_slot"],
        required=False,
        label="lineup batting order column",
    )

    starter_col = find_col(
        df,
        ["is_starter", "starter", "isStarting", "is_starting"],
        required=False,
        label="lineup starter flag column",
    )

    out = pd.DataFrame(
        {
            "game_pk": df[game_col].apply(safe_int),
            "team_id": df[team_col].apply(safe_int),
            "player_id": df[player_col].apply(safe_int),
        }
    )

    if batting_order_col is not None:
        out["batting_order_raw"] = df[batting_order_col]
        out["batting_order_num"] = df[batting_order_col].apply(safe_int)

        # MLB battingOrder usually looks like 100, 200, ..., 900.
        # Some exports may use 1..9.
        out["lineup_slot"] = out["batting_order_num"].apply(
            lambda x: x // 100 if x >= 100 else x
        )

        out = out[(out["lineup_slot"] >= 1) & (out["lineup_slot"] <= 9)].copy()

    elif starter_col is not None:
        starter_values = df[starter_col].astype(str).str.lower().isin(
            ["1", "true", "yes", "y"]
        )
        out = out[starter_values].copy()
        out["lineup_slot"] = 0

    else:
        print(
            "\nNo batting order or starter flag found in lineup file. "
            "Assuming lineup file already contains starters only."
        )
        out["lineup_slot"] = 0

    out = out.dropna(subset=["game_pk", "team_id", "player_id"])
    out = out[out["player_id"] > 0]
    out = out.drop_duplicates(subset=["game_pk", "team_id", "player_id"])

    return out


# =========================
# Platoon Feature Logic
# =========================

def batter_has_platoon_advantage(bat_side, pitcher_hand):
    bat_side = normalize_hand(bat_side)
    pitcher_hand = normalize_hand(pitcher_hand)

    if pitcher_hand not in {"L", "R"}:
        return None

    if bat_side == "S":
        return 1

    if pitcher_hand == "R" and bat_side == "L":
        return 1

    if pitcher_hand == "L" and bat_side == "R":
        return 1

    if bat_side in {"L", "R"}:
        return 0

    return None


def batter_is_opposite_hand(bat_side, pitcher_hand):
    bat_side = normalize_hand(bat_side)
    pitcher_hand = normalize_hand(pitcher_hand)

    if pitcher_hand not in {"L", "R"}:
        return None

    if bat_side == "S":
        return 0

    if pitcher_hand == "R" and bat_side == "L":
        return 1

    if pitcher_hand == "L" and bat_side == "R":
        return 1

    if bat_side in {"L", "R"}:
        return 0

    return None


def lineup_platoon_summary(lineup_rows, pitcher_hand):
    pitcher_hand = normalize_hand(pitcher_hand)

    if lineup_rows is None or len(lineup_rows) == 0:
        return {
            "platoon_adv_pct": 0.50,
            "opposite_hand_pct": 0.50,
            "switch_pct": 0.00,
            "known_bats_scaled": 0.00,
        }

    known = lineup_rows[
        lineup_rows["bat_side"].apply(normalize_hand).isin(["L", "R", "S"])
    ].copy()

    known_count = len(known)

    if known_count == 0 or pitcher_hand not in {"L", "R"}:
        return {
            "platoon_adv_pct": 0.50,
            "opposite_hand_pct": 0.50,
            "switch_pct": 0.00,
            "known_bats_scaled": 0.00,
        }

    platoon_values = []
    opposite_values = []
    switch_count = 0

    for _, row in known.iterrows():
        bat_side = normalize_hand(row["bat_side"])

        if bat_side == "S":
            switch_count += 1

        adv = batter_has_platoon_advantage(bat_side, pitcher_hand)
        opp = batter_is_opposite_hand(bat_side, pitcher_hand)

        if adv is not None:
            platoon_values.append(adv)

        if opp is not None:
            opposite_values.append(opp)

    if not platoon_values:
        platoon_adv_pct = 0.50
    else:
        platoon_adv_pct = sum(platoon_values) / len(platoon_values)

    if not opposite_values:
        opposite_hand_pct = 0.50
    else:
        opposite_hand_pct = sum(opposite_values) / len(opposite_values)

    return {
        "platoon_adv_pct": platoon_adv_pct,
        "opposite_hand_pct": opposite_hand_pct,
        "switch_pct": switch_count / known_count,
        "known_bats_scaled": min(known_count / 9.0, 1.0),
    }


def build_lineup_platoon_features(
    lineup_by_game_team,
    game_pk,
    home_team_id,
    away_team_id,
    home_starter_hand,
    away_starter_hand,
):
    home_lineup = lineup_by_game_team.get((game_pk, home_team_id))
    away_lineup = lineup_by_game_team.get((game_pk, away_team_id))

    # Home hitters face away starter.
    home_summary = lineup_platoon_summary(home_lineup, away_starter_hand)

    # Away hitters face home starter.
    away_summary = lineup_platoon_summary(away_lineup, home_starter_hand)

    home_faces_lhp = 1 if normalize_hand(away_starter_hand) == "L" else 0
    away_faces_lhp = 1 if normalize_hand(home_starter_hand) == "L" else 0

    starter_hand_matchup_same = (
        1
        if normalize_hand(home_starter_hand) in {"L", "R"}
        and normalize_hand(home_starter_hand) == normalize_hand(away_starter_hand)
        else 0
    )

    return {
        "home_lineup_platoon_adv_pct": home_summary["platoon_adv_pct"],
        "away_lineup_platoon_adv_pct": away_summary["platoon_adv_pct"],
        "lineup_platoon_adv_pct_diff": (
            home_summary["platoon_adv_pct"] - away_summary["platoon_adv_pct"]
        ),
        "home_lineup_opposite_hand_pct": home_summary["opposite_hand_pct"],
        "away_lineup_opposite_hand_pct": away_summary["opposite_hand_pct"],
        "lineup_opposite_hand_pct_diff": (
            home_summary["opposite_hand_pct"] - away_summary["opposite_hand_pct"]
        ),
        "home_lineup_switch_pct": home_summary["switch_pct"],
        "away_lineup_switch_pct": away_summary["switch_pct"],
        "lineup_switch_pct_diff": (
            home_summary["switch_pct"] - away_summary["switch_pct"]
        ),
        "home_lineup_known_bats_scaled": home_summary["known_bats_scaled"],
        "away_lineup_known_bats_scaled": away_summary["known_bats_scaled"],
        "lineup_known_bats_scaled_diff": (
            home_summary["known_bats_scaled"] - away_summary["known_bats_scaled"]
        ),
        "home_faces_lhp": home_faces_lhp,
        "away_faces_lhp": away_faces_lhp,
        "starter_hand_matchup_same": starter_hand_matchup_same,
    }


# =========================
# Team vs Starter Hand Logic
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

    if games <= 0:
        return {
            "win_pct": 0.500,
            "rpg": 4.50,
            "rapg": 4.50,
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
        "run_diff_pg": run_diff_pg,
        "recent_win_pct": recent_win_pct,
        "recent_rpg": recent_rpg,
        "recent_rapg": recent_rapg,
        "recent_run_diff_pg": recent_run_diff_pg,
        "games_scaled": min(games / 50.0, 1.0),
    }


def build_team_vs_hand_features(
    team_hand_states,
    home_team_id,
    away_team_id,
    home_starter_hand,
    away_starter_hand,
):
    home_starter_hand = normalize_hand(home_starter_hand)
    away_starter_hand = normalize_hand(away_starter_hand)

    # Home team bats against away starter hand.
    if away_starter_hand in {"L", "R"}:
        home_state = team_hand_states[home_team_id][away_starter_hand]
    else:
        home_state = create_hand_state()

    # Away team bats against home starter hand.
    if home_starter_hand in {"L", "R"}:
        away_state = team_hand_states[away_team_id][home_starter_hand]
    else:
        away_state = create_hand_state()

    home = snapshot_hand_state(home_state)
    away = snapshot_hand_state(away_state)

    return {
        "team_vs_hand_win_pct_diff": home["win_pct"] - away["win_pct"],
        "team_vs_hand_rpg_diff": home["rpg"] - away["rpg"],

        # Positive means away has allowed more in this split than home.
        "team_vs_hand_rapg_diff": away["rapg"] - home["rapg"],

        "team_vs_hand_run_diff_pg_diff": (
            home["run_diff_pg"] - away["run_diff_pg"]
        ),
        "team_vs_hand_recent_win_pct_diff": (
            home["recent_win_pct"] - away["recent_win_pct"]
        ),
        "team_vs_hand_recent_rpg_diff": (
            home["recent_rpg"] - away["recent_rpg"]
        ),

        # Positive means away has recently allowed more in this split than home.
        "team_vs_hand_recent_rapg_diff": (
            away["recent_rapg"] - home["recent_rapg"]
        ),

        "team_vs_hand_recent_run_diff_pg_diff": (
            home["recent_run_diff_pg"] - away["recent_run_diff_pg"]
        ),
        "home_vs_hand_games_scaled": home["games_scaled"],
        "away_vs_hand_games_scaled": away["games_scaled"],
        "vs_hand_games_scaled_diff": (
            home["games_scaled"] - away["games_scaled"]
        ),
    }


# =========================
# Feature Frame Builder
# =========================

def build_lineup_lookup(lineups_with_hands):
    lookup = {}

    for keys, group in lineups_with_hands.groupby(["game_pk", "team_id"]):
        lookup[keys] = group.copy()

    return lookup


def build_platoon_feature_frame(
    raw_games,
    lineups_with_hands,
    starter_hands,
    home_starter_col,
    away_starter_col,
):
    rows = []
    lineup_by_game_team = build_lineup_lookup(lineups_with_hands)

    df = raw_games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["game_date"] = df["date"].dt.date
    df = df.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)

    team_hand_states = defaultdict(create_team_hand_state)

    current_season = None

    for season in sorted(df["season"].dropna().unique()):
        season_df = df[df["season"] == season].copy()
        team_hand_states = defaultdict(create_team_hand_state)
        current_season = season

        for game_date, day_games in season_df.groupby("game_date", sort=True):
            # Create features for every game on this date first.
            # Then update after all same-day games to avoid doubleheader leakage.
            for _, game in day_games.iterrows():
                game_pk = safe_int(game["game_pk"])
                home_team_id = safe_int(game["home_team_id"])
                away_team_id = safe_int(game["away_team_id"])

                home_starter_id = safe_int(game[home_starter_col])
                away_starter_id = safe_int(game[away_starter_col])

                home_starter_hand = starter_hands.get(home_starter_id, "U")
                away_starter_hand = starter_hands.get(away_starter_id, "U")

                lineup_features = build_lineup_platoon_features(
                    lineup_by_game_team=lineup_by_game_team,
                    game_pk=game_pk,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_starter_hand=home_starter_hand,
                    away_starter_hand=away_starter_hand,
                )

                team_vs_hand_features = build_team_vs_hand_features(
                    team_hand_states=team_hand_states,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_starter_hand=home_starter_hand,
                    away_starter_hand=away_starter_hand,
                )

                rows.append(
                    {
                        "game_pk": game_pk,
                        **lineup_features,
                        **team_vs_hand_features,
                    }
                )

            # Update after the full day is processed.
            for _, game in day_games.iterrows():
                home_team_id = safe_int(game["home_team_id"])
                away_team_id = safe_int(game["away_team_id"])

                home_score = safe_int(game.get("home_score", 0))
                away_score = safe_int(game.get("away_score", 0))

                home_starter_id = safe_int(game[home_starter_col])
                away_starter_id = safe_int(game[away_starter_col])

                home_starter_hand = starter_hands.get(home_starter_id, "U")
                away_starter_hand = starter_hands.get(away_starter_id, "U")

                # Home offense faced away starter hand.
                if away_starter_hand in {"L", "R"}:
                    update_hand_state(
                        team_hand_states[home_team_id][away_starter_hand],
                        home_score,
                        away_score,
                    )

                # Away offense faced home starter hand.
                if home_starter_hand in {"L", "R"}:
                    update_hand_state(
                        team_hand_states[away_team_id][home_starter_hand],
                        away_score,
                        home_score,
                    )

    return pd.DataFrame(rows)


# =========================
# Evaluation
# =========================

def evaluate_holdout(df, features, test_season=None):
    if test_season is None:
        test_season = int(df["season"].max())

    features = unique_features(features)

    data = df.dropna(subset=features + ["home_win"]).copy()

    train_df = data[data["season"] < test_season]
    test_df = data[data["season"] == test_season]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(train_df[features], train_df["home_win"])

    probs = model.predict_proba(test_df[features])[:, 1]
    preds = (probs >= 0.50).astype(int)

    return {
        "test_season": test_season,
        "games": len(test_df),
        "accuracy": accuracy_score(test_df["home_win"], preds),
        "log_loss": log_loss(test_df["home_win"], probs),
        "brier": brier_score_loss(test_df["home_win"], probs),
    }


def run_feature_set_comparison(df):
    feature_sets = {
        "team_only": TEAM_FEATURES + SCHEDULE_FEATURES,
        "lineup_platoon_only": LINEUP_PLATOON_FEATURES + SCHEDULE_FEATURES,
        "team_vs_hand_only": TEAM_VS_HAND_FEATURES + SCHEDULE_FEATURES,
        "platoon_all_only": PLATOON_FEATURES + SCHEDULE_FEATURES,
        "team_plus_lineup_platoon": (
            TEAM_FEATURES + LINEUP_PLATOON_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_team_vs_hand": (
            TEAM_FEATURES + TEAM_VS_HAND_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_platoon_all": (
            TEAM_FEATURES + PLATOON_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_platoon_all": (
            TEAM_FEATURES
            + STARTER_FEATURES
            + PLATOON_FEATURES
            + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Platoon Matchup Feature Set Comparison: 2025 Holdout ===")

    for name, features in feature_sets.items():
        result = evaluate_holdout(df, features)

        rows.append(
            {
                "feature_set": name,
                **result,
            }
        )

        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.3f}, "
            f"LogLoss={result['log_loss']:.4f}, "
            f"Brier={result['brier']:.4f}"
        )

    out = pd.DataFrame(rows).sort_values("log_loss")
    out.to_csv(
        EXPORT_DIR / "platoon_feature_set_comparison.csv",
        index=False,
    )

    return out, feature_sets


def run_platoon_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Platoon Matchup Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in PLATOON_FEATURES:
        trial_features = baseline_features + [feature]
        result = evaluate_holdout(df, trial_features)

        row = {
            "feature": feature,
            **result,
            "log_loss_delta": result["log_loss"] - baseline["log_loss"],
            "brier_delta": result["brier"] - baseline["brier"],
        }

        rows.append(row)

        print(
            f"add_{feature}: "
            f"Accuracy={result['accuracy']:.3f}, "
            f"LogLoss={result['log_loss']:.4f} "
            f"({row['log_loss_delta']:+.4f}), "
            f"Brier={result['brier']:.4f} "
            f"({row['brier_delta']:+.4f})"
        )

    out = pd.DataFrame(rows).sort_values("log_loss_delta")
    out.to_csv(
        EXPORT_DIR / "platoon_ablation.csv",
        index=False,
    )

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Platoon Matchup Cross-Validation by Season ===")

    for name, features in feature_sets.items():
        for season in test_seasons:
            result = evaluate_holdout(df, features, test_season=season)

            rows.append(
                {
                    "feature_set": name,
                    **result,
                }
            )

            print(
                f"{name} | Test {season}: "
                f"Accuracy={result['accuracy']:.3f}, "
                f"LogLoss={result['log_loss']:.4f}, "
                f"Brier={result['brier']:.4f}"
            )

    by_season = pd.DataFrame(rows)

    summary = (
        by_season.groupby("feature_set")
        .agg(
            seasons_tested=("test_season", "nunique"),
            avg_accuracy=("accuracy", "mean"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier=("brier", "mean"),
            worst_log_loss=("log_loss", "max"),
            best_log_loss=("log_loss", "min"),
        )
        .reset_index()
        .sort_values("avg_log_loss")
    )

    by_season.to_csv(
        EXPORT_DIR / "platoon_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "platoon_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Platoon Matchup Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


# =========================
# Main
# =========================

def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Platoon Matchup Research Lab ===")
    print("Testing whether handedness / lineup platoon context beats Production v1.0.")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw games file: {RAW_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

    if not LINEUPS_PATH.exists():
        raise FileNotFoundError(
            f"Missing lineup file: {LINEUPS_PATH}\n"
            "Run the lineup lab first so game_lineups_2023_2025.parquet exists."
        )

    raw_games = pd.read_parquet(RAW_PATH)
    base_features = pd.read_parquet(FEATURES_PATH)
    raw_lineups = pd.read_parquet(LINEUPS_PATH)

    raw_games["date"] = pd.to_datetime(raw_games["date"])

    home_starter_col, away_starter_col = find_starter_columns(raw_games)

    print(f"\nDetected home starter column: {home_starter_col}")
    print(f"Detected away starter column: {away_starter_col}")

    lineups = normalize_lineups(raw_lineups)

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Raw lineup rows: {len(raw_lineups)}")
    print(f"Normalized lineup starter rows: {len(lineups)}")

    starter_ids = pd.concat(
        [
            raw_games[home_starter_col].apply(safe_int),
            raw_games[away_starter_col].apply(safe_int),
        ],
        ignore_index=True,
    )

    batter_ids = lineups["player_id"].apply(safe_int)

    all_player_ids = sorted(
        set(
            [pid for pid in starter_ids.tolist() if pid > 0]
            + [pid for pid in batter_ids.tolist() if pid > 0]
        )
    )

    handedness = ensure_player_handedness(all_player_ids)

    hand_map = handedness.set_index("player_id").to_dict("index")

    starter_hands = {
        int(pid): normalize_hand(info.get("pitch_hand"))
        for pid, info in hand_map.items()
    }

    bat_sides = {
        int(pid): normalize_hand(info.get("bat_side"))
        for pid, info in hand_map.items()
    }

    lineups["bat_side"] = lineups["player_id"].map(bat_sides).fillna("U")
    lineups["bat_side"] = lineups["bat_side"].apply(normalize_hand)

    known_bat_rate = (
        lineups["bat_side"].isin(["L", "R", "S"]).mean()
        if len(lineups)
        else 0
    )

    known_home_starter_hand_rate = (
        raw_games[home_starter_col]
        .apply(lambda x: starter_hands.get(safe_int(x), "U"))
        .isin(["L", "R"])
        .mean()
    )

    known_away_starter_hand_rate = (
        raw_games[away_starter_col]
        .apply(lambda x: starter_hands.get(safe_int(x), "U"))
        .isin(["L", "R"])
        .mean()
    )

    print(f"\nUnique players needing handedness: {len(all_player_ids)}")
    print(f"Known batter hand rate: {known_bat_rate:.1%}")
    print(f"Known home starter hand rate: {known_home_starter_hand_rate:.1%}")
    print(f"Known away starter hand rate: {known_away_starter_hand_rate:.1%}")

    print("\nBatter hand counts:")
    print(lineups["bat_side"].value_counts(dropna=False).to_string())

    platoon_features = build_platoon_feature_frame(
        raw_games=raw_games,
        lineups_with_hands=lineups,
        starter_hands=starter_hands,
        home_starter_col=home_starter_col,
        away_starter_col=away_starter_col,
    )

    enriched = base_features.merge(
        platoon_features,
        on="game_pk",
        how="left",
    )

    for feature in PLATOON_FEATURES:
        if feature not in enriched.columns:
            enriched[feature] = 0.0

    enriched[PLATOON_FEATURES] = enriched[PLATOON_FEATURES].fillna(0.0)

    enriched.to_parquet(
        EXPORT_DIR / "features_with_platoon_lab.parquet",
        index=False,
    )

    print(f"\nPlatoon feature rows: {len(platoon_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nPlatoon feature preview:")
    print(enriched[PLATOON_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_platoon_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- exports/features_with_platoon_lab.parquet")
    print("- exports/platoon_feature_set_comparison.csv")
    print("- exports/platoon_ablation.csv")
    print("- exports/platoon_cross_validation_by_season.csv")
    print("- exports/platoon_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Platoon matchup only promotes if team_plus_platoon_all")
    print("or a narrower platoon set improves avg log loss/Brier")
    print("versus team_only across cross-validation.")


if __name__ == "__main__":
    main()