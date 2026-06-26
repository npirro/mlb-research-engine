from collections import defaultdict, deque
from pathlib import Path
import sys

import pandas as pd
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


RAW_PATH = Path("data/raw/games_with_starters_2023_2025.parquet")
FEATURES_PATH = Path("data/processed/features_v4_feature_factory.parquet")
EXPORT_DIR = Path("exports")

SCHEDULE_SPOT_FEATURES = [
    "rest_days_diff",
    "days_since_last_game_diff",
    "games_last_3_days_diff",
    "games_last_7_days_diff",
    "no_rest_diff",
    "travel_switch_diff",
    "home_stand_len",
    "away_road_trip_len",
    "home_stand_vs_away_road_trip_diff",
]


def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def create_schedule_state():
    return {
        "last_date": None,
        "last_side": None,
        "consecutive_side_games": 0,
        "recent_game_dates": deque(maxlen=20),
    }


def days_between(current_date, last_date):
    if last_date is None:
        return 3

    return max((current_date - last_date).days, 0)


def count_recent_games(recent_dates, current_date, window_days):
    cutoff = current_date - pd.Timedelta(days=window_days)

    return sum(1 for d in recent_dates if d >= cutoff and d < current_date)


def snapshot_schedule_state(state, current_date, current_side):
    days_since_last_game = days_between(current_date, state["last_date"])

    # Rest days = off days between games.
    # Played yesterday = 0 rest.
    # Last game 2 days ago = 1 rest day.
    rest_days = max(days_since_last_game - 1, 0)
    rest_days = min(rest_days, 5)

    games_last_3 = count_recent_games(
        state["recent_game_dates"],
        current_date,
        3,
    )

    games_last_7 = count_recent_games(
        state["recent_game_dates"],
        current_date,
        7,
    )

    no_rest = 1 if days_since_last_game <= 1 else 0

    travel_switch = 0
    if state["last_side"] is not None and state["last_side"] != current_side:
        travel_switch = 1

    consecutive_same_side_entering = 0
    if state["last_side"] == current_side:
        consecutive_same_side_entering = state["consecutive_side_games"]

    return {
        "days_since_last_game": days_since_last_game,
        "rest_days": rest_days,
        "games_last_3": games_last_3,
        "games_last_7": games_last_7,
        "no_rest": no_rest,
        "travel_switch": travel_switch,
        "consecutive_same_side_entering": consecutive_same_side_entering,
    }


def update_schedule_state(state, current_date, current_side):
    if state["last_side"] == current_side:
        state["consecutive_side_games"] += 1
    else:
        state["consecutive_side_games"] = 1

    state["last_date"] = current_date
    state["last_side"] = current_side
    state["recent_game_dates"].append(current_date)


def build_schedule_diffs(home_snapshot, away_snapshot):
    home_stand_len = home_snapshot["consecutive_same_side_entering"]
    away_road_trip_len = away_snapshot["consecutive_same_side_entering"]

    return {
        # Positive should generally mean home has the schedule advantage.
        "rest_days_diff": home_snapshot["rest_days"] - away_snapshot["rest_days"],
        "days_since_last_game_diff": (
            home_snapshot["days_since_last_game"]
            - away_snapshot["days_since_last_game"]
        ),
        "games_last_3_days_diff": (
            away_snapshot["games_last_3"] - home_snapshot["games_last_3"]
        ),
        "games_last_7_days_diff": (
            away_snapshot["games_last_7"] - home_snapshot["games_last_7"]
        ),
        "no_rest_diff": away_snapshot["no_rest"] - home_snapshot["no_rest"],
        "travel_switch_diff": (
            away_snapshot["travel_switch"] - home_snapshot["travel_switch"]
        ),
        "home_stand_len": home_stand_len,
        "away_road_trip_len": away_road_trip_len,
        "home_stand_vs_away_road_trip_diff": (
            home_stand_len + away_road_trip_len
        ),
    }


def build_schedule_spot_feature_frame(raw_games):
    schedule_states = defaultdict(create_schedule_state)
    rows = []

    df = raw_games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    for _, game in df.iterrows():
        current_date = game["date"]

        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        home_state = schedule_states[home_team_id]
        away_state = schedule_states[away_team_id]

        home_snapshot = snapshot_schedule_state(
            home_state,
            current_date,
            "home",
        )

        away_snapshot = snapshot_schedule_state(
            away_state,
            current_date,
            "away",
        )

        row = {
            "game_pk": game["game_pk"],
            **build_schedule_diffs(home_snapshot, away_snapshot),
        }

        rows.append(row)

        # Update only AFTER feature creation.
        # This prevents current-game leakage.
        update_schedule_state(home_state, current_date, "home")
        update_schedule_state(away_state, current_date, "away")

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
        "schedule_spot_only": SCHEDULE_SPOT_FEATURES + SCHEDULE_FEATURES,
        "team_plus_schedule_spot": (
            TEAM_FEATURES + SCHEDULE_SPOT_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_schedule_spot": (
            TEAM_FEATURES
            + STARTER_FEATURES
            + SCHEDULE_SPOT_FEATURES
            + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Schedule Spot Feature Set Comparison: 2025 Holdout ===")

    for name, features in feature_sets.items():
        result = evaluate_holdout(df, features)

        row = {
            "feature_set": name,
            **result,
        }

        rows.append(row)

        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.3f}, "
            f"LogLoss={result['log_loss']:.4f}, "
            f"Brier={result['brier']:.4f}"
        )

    out = pd.DataFrame(rows).sort_values("log_loss")
    out.to_csv(EXPORT_DIR / "schedule_spot_feature_set_comparison.csv", index=False)

    return out, feature_sets


def run_schedule_spot_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Schedule Spot Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in SCHEDULE_SPOT_FEATURES:
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
    out.to_csv(EXPORT_DIR / "schedule_spot_ablation.csv", index=False)

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Schedule Spot Cross-Validation by Season ===")

    for name, features in feature_sets.items():
        for season in test_seasons:
            result = evaluate_holdout(df, features, test_season=season)

            row = {
                "feature_set": name,
                **result,
            }

            rows.append(row)

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
        EXPORT_DIR / "schedule_spot_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "schedule_spot_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Schedule Spot Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Schedule Spot Research Lab ===")
    print("Testing whether rest/travel/schedule features beat Production v1.0.")

    raw_games = pd.read_parquet(RAW_PATH)
    base_features = pd.read_parquet(FEATURES_PATH)

    schedule_features = build_schedule_spot_feature_frame(raw_games)

    enriched = base_features.merge(
        schedule_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(
        EXPORT_DIR / "features_with_schedule_spot_lab.parquet",
        index=False,
    )

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nSchedule spot feature preview:")
    print(enriched[SCHEDULE_SPOT_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_schedule_spot_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- exports/features_with_schedule_spot_lab.parquet")
    print("- exports/schedule_spot_feature_set_comparison.csv")
    print("- exports/schedule_spot_ablation.csv")
    print("- exports/schedule_spot_cross_validation_by_season.csv")
    print("- exports/schedule_spot_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Schedule spot only promotes if team_plus_schedule_spot improves")
    print("avg log loss/Brier versus team_only across cross-validation.")


if __name__ == "__main__":
    main()