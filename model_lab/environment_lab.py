from collections import defaultdict, deque
from pathlib import Path
import re
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


MLB_BASE = "https://statsapi.mlb.com/api/v1.1"

RAW_PATH = Path("data/raw/games_with_starters_2023_2025.parquet")
FEATURES_PATH = Path("data/processed/features_v4_feature_factory.parquet")

ENV_CACHE_PATH = Path("data/raw/game_environment_2023_2025.parquet")
ENV_CHECKPOINT_PATH = Path("data/raw/game_environment_2023_2025_checkpoint.parquet")

EXPORT_DIR = Path("exports")


ENVIRONMENT_FEATURES = [
    "env_day_game",
    "env_temp",
    "env_wind_speed",
    "env_wind_out",
    "env_wind_in",
    "env_wind_cross",
    "env_weather_clear",
    "env_weather_cloudy",
    "env_weather_precip",
    "park_home_win_rate_entering",
    "park_runs_per_game_entering",
    "park_recent_runs_per_game_entering",
    "park_games_entering_scaled",
]


def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def fetch_game_feed(game_pk):
    r = requests.get(
        f"{MLB_BASE}/game/{game_pk}/feed/live",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def parse_temperature(temp_value):
    if temp_value is None:
        return 70.0

    text = str(temp_value)

    match = re.search(r"-?\d+", text)
    if not match:
        return 70.0

    temp = float(match.group())

    # Guard against bad values.
    if temp < 20 or temp > 120:
        return 70.0

    return temp


def parse_wind_speed(wind_value):
    if wind_value is None:
        return 0.0

    text = str(wind_value).lower()

    match = re.search(r"(\d+)", text)
    if not match:
        return 0.0

    speed = float(match.group())

    if speed < 0 or speed > 60:
        return 0.0

    return speed


def parse_wind_direction_flags(wind_value):
    text = str(wind_value or "").lower()

    wind_out = 0
    wind_in = 0
    wind_cross = 0

    if "out" in text:
        wind_out = 1

    if "in" in text:
        wind_in = 1

    if "l to r" in text or "r to l" in text or "left to right" in text or "right to left" in text:
        wind_cross = 1

    return wind_out, wind_in, wind_cross


def parse_weather_flags(condition_value):
    text = str(condition_value or "").lower()

    clear_words = ["clear", "sunny", "fair"]
    cloudy_words = ["cloud", "overcast", "partly"]
    precip_words = ["rain", "drizzle", "snow", "shower", "storm"]

    weather_clear = 1 if any(w in text for w in clear_words) else 0
    weather_cloudy = 1 if any(w in text for w in cloudy_words) else 0
    weather_precip = 1 if any(w in text for w in precip_words) else 0

    return weather_clear, weather_cloudy, weather_precip


def collect_environment_for_game(game_row):
    game_pk = game_row["game_pk"]

    feed = fetch_game_feed(game_pk)

    game_data = feed.get("gameData", {})

    venue = game_data.get("venue", {}) or {}
    weather = game_data.get("weather", {}) or {}
    datetime_data = game_data.get("datetime", {}) or {}

    venue_id = venue.get("id")
    venue_name = venue.get("name")

    day_night = datetime_data.get("dayNight")
    if day_night is None:
        day_night = game_row.get("dayNight", "")

    day_night_text = str(day_night or "").lower()
    env_day_game = 1 if day_night_text == "day" else 0

    condition = weather.get("condition")
    temp = parse_temperature(weather.get("temp"))
    wind = weather.get("wind")

    wind_speed = parse_wind_speed(wind)
    wind_out, wind_in, wind_cross = parse_wind_direction_flags(wind)
    weather_clear, weather_cloudy, weather_precip = parse_weather_flags(condition)

    return {
        "game_pk": game_pk,
        "date": game_row["date"],
        "season": game_row["season"],
        "venue_id": venue_id,
        "venue_name": venue_name,
        "day_night": day_night,
        "weather_condition": condition,
        "weather_temp": temp,
        "weather_wind": wind,
        "env_day_game": env_day_game,
        "env_temp": temp,
        "env_wind_speed": wind_speed,
        "env_wind_out": wind_out,
        "env_wind_in": wind_in,
        "env_wind_cross": wind_cross,
        "env_weather_clear": weather_clear,
        "env_weather_cloudy": weather_cloudy,
        "env_weather_precip": weather_precip,
    }


def download_environment_cache(raw_games):
    if ENV_CACHE_PATH.exists():
        print(f"Using cached environment file: {ENV_CACHE_PATH}")
        return pd.read_parquet(ENV_CACHE_PATH)

    ENV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()

    if ENV_CHECKPOINT_PATH.exists():
        print(f"Using environment checkpoint: {ENV_CHECKPOINT_PATH}")
        existing = pd.read_parquet(ENV_CHECKPOINT_PATH)

    done_games = set(existing["game_pk"].unique()) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    raw_games = raw_games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    print("\nDownloading historical game environment data...")
    print("This may take a while the first time.")

    for _, game in raw_games.iterrows():
        game_pk = game["game_pk"]

        if game_pk in done_games:
            continue

        try:
            row = collect_environment_for_game(game)
            rows.append(row)
            done_games.add(game_pk)

            if len(done_games) % 250 == 0:
                checkpoint = pd.DataFrame(rows)
                checkpoint.to_parquet(ENV_CHECKPOINT_PATH, index=False)
                print(f"Checkpoint saved. Games processed: {len(done_games)}")

            time.sleep(0.025)

        except Exception as e:
            print(f"Environment download failed for game {game_pk}: {e}")
            continue

    env_df = pd.DataFrame(rows)

    if env_df.empty:
        raise RuntimeError("No environment rows were downloaded.")

    env_df = env_df.drop_duplicates("game_pk")
    env_df["date"] = pd.to_datetime(env_df["date"])
    env_df = env_df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    env_df.to_parquet(ENV_CACHE_PATH, index=False)
    env_df.to_parquet(ENV_CHECKPOINT_PATH, index=False)

    print(f"Saved environment cache: {ENV_CACHE_PATH}")
    print(f"Rows: {len(env_df)}")

    return env_df


def create_park_state():
    return {
        "games": 0,
        "home_wins": 0,
        "total_runs": 0,
        "recent_total_runs": deque(maxlen=50),
    }


def park_snapshot(state):
    games = state["games"]

    if games <= 0:
        return {
            "park_home_win_rate_entering": 0.540,
            "park_runs_per_game_entering": 9.00,
            "park_recent_runs_per_game_entering": 9.00,
            "park_games_entering_scaled": 0.0,
        }

    home_win_rate = state["home_wins"] / games
    runs_per_game = state["total_runs"] / games

    recent = list(state["recent_total_runs"])
    if recent:
        recent_rpg = sum(recent) / len(recent)
    else:
        recent_rpg = runs_per_game

    return {
        "park_home_win_rate_entering": home_win_rate,
        "park_runs_per_game_entering": runs_per_game,
        "park_recent_runs_per_game_entering": recent_rpg,
        "park_games_entering_scaled": min(games / 100.0, 1.0),
    }


def update_park_state(state, home_score, away_score):
    home_score = safe_int(home_score)
    away_score = safe_int(away_score)

    state["games"] += 1
    state["home_wins"] += 1 if home_score > away_score else 0
    state["total_runs"] += home_score + away_score
    state["recent_total_runs"].append(home_score + away_score)


def build_environment_feature_frame(raw_games, env_cache):
    park_states = defaultdict(create_park_state)
    rows = []

    raw_games = raw_games.copy()
    raw_games["date"] = pd.to_datetime(raw_games["date"])
    raw_games = raw_games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    env_cache = env_cache.copy()
    env_cache["date"] = pd.to_datetime(env_cache["date"])

    env_by_game = {
        row["game_pk"]: row
        for _, row in env_cache.iterrows()
    }

    for _, game in raw_games.iterrows():
        game_pk = game["game_pk"]

        env_row = env_by_game.get(game_pk)

        if env_row is None:
            venue_id = -1
            base_env = {
                "env_day_game": 0,
                "env_temp": 70.0,
                "env_wind_speed": 0.0,
                "env_wind_out": 0,
                "env_wind_in": 0,
                "env_wind_cross": 0,
                "env_weather_clear": 0,
                "env_weather_cloudy": 0,
                "env_weather_precip": 0,
            }
        else:
            venue_id = env_row.get("venue_id", -1)

            if pd.isna(venue_id):
                venue_id = -1

            base_env = {
                "env_day_game": safe_int(env_row.get("env_day_game", 0)),
                "env_temp": safe_float(env_row.get("env_temp", 70.0), 70.0),
                "env_wind_speed": safe_float(env_row.get("env_wind_speed", 0.0), 0.0),
                "env_wind_out": safe_int(env_row.get("env_wind_out", 0)),
                "env_wind_in": safe_int(env_row.get("env_wind_in", 0)),
                "env_wind_cross": safe_int(env_row.get("env_wind_cross", 0)),
                "env_weather_clear": safe_int(env_row.get("env_weather_clear", 0)),
                "env_weather_cloudy": safe_int(env_row.get("env_weather_cloudy", 0)),
                "env_weather_precip": safe_int(env_row.get("env_weather_precip", 0)),
            }

        park_state = park_states[venue_id]
        park_features = park_snapshot(park_state)

        rows.append(
            {
                "game_pk": game_pk,
                **base_env,
                **park_features,
            }
        )

        # Update park state AFTER feature creation.
        # This prevents current-game result leakage.
        update_park_state(
            park_state,
            game.get("home_score", 0),
            game.get("away_score", 0),
        )

    return pd.DataFrame(rows)


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
        "environment_only": ENVIRONMENT_FEATURES + SCHEDULE_FEATURES,
        "team_plus_environment": (
            TEAM_FEATURES + ENVIRONMENT_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_environment": (
            TEAM_FEATURES
            + STARTER_FEATURES
            + ENVIRONMENT_FEATURES
            + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Environment Feature Set Comparison: 2025 Holdout ===")

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
    out.to_csv(EXPORT_DIR / "environment_feature_set_comparison.csv", index=False)

    return out, feature_sets


def run_environment_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Environment Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in ENVIRONMENT_FEATURES:
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
    out.to_csv(EXPORT_DIR / "environment_ablation.csv", index=False)

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Environment Cross-Validation by Season ===")

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
        EXPORT_DIR / "environment_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "environment_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Environment Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Environment Research Lab ===")
    print("Testing whether weather/park/game environment beats Production v1.0.")

    raw_games = pd.read_parquet(RAW_PATH)
    raw_games["date"] = pd.to_datetime(raw_games["date"])

    base_features = pd.read_parquet(FEATURES_PATH)

    env_cache = download_environment_cache(raw_games)

    print("\nEnvironment cache quality:")
    print(f"Environment rows: {len(env_cache)}")
    print(f"Unique venues: {env_cache['venue_id'].nunique()}")
    print("Weather condition sample:")
    print(env_cache["weather_condition"].value_counts(dropna=False).head(10).to_string())

    environment_features = build_environment_feature_frame(raw_games, env_cache)

    enriched = base_features.merge(
        environment_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(
        EXPORT_DIR / "features_with_environment_lab.parquet",
        index=False,
    )

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nEnvironment feature preview:")
    print(enriched[ENVIRONMENT_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_environment_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- data/raw/game_environment_2023_2025.parquet")
    print("- exports/features_with_environment_lab.parquet")
    print("- exports/environment_feature_set_comparison.csv")
    print("- exports/environment_ablation.csv")
    print("- exports/environment_cross_validation_by_season.csv")
    print("- exports/environment_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Environment only promotes if team_plus_environment improves")
    print("avg log loss/Brier versus team_only across cross-validation.")


if __name__ == "__main__":
    main()