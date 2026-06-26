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

BULLPEN_FEATURES = [
    "bp_era_diff",
    "bp_whip_diff",
    "bp_kbb_diff",
    "bp_recent_pitches_diff",
    "bp_recent_ip_diff",
    "bp_fatigue_diff",
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
            "bp_fatigue": 50.0,
        }

    bp_era = 9 * state["relief_er"] / state["relief_ip"]
    bp_whip = (state["relief_hits"] + state["relief_bb"]) / state["relief_ip"]
    bp_kbb = state["relief_so"] / max(state["relief_bb"], 1)

    recent_pitches = sum(g["pitches"] for g in state["recent_games"])
    recent_ip = sum(g["ip"] for g in state["recent_games"])

    fatigue = min(100.0, recent_pitches * 0.35 + recent_ip * 6)

    return {
        "bp_era": bp_era,
        "bp_whip": bp_whip,
        "bp_kbb": bp_kbb,
        "bp_recent_pitches": recent_pitches,
        "bp_recent_ip": recent_ip,
        "bp_fatigue": fatigue,
    }


def build_bullpen_diffs(home_bp, away_bp):
    return {
        # Positive means home bullpen has been better.
        "bp_era_diff": away_bp["bp_era"] - home_bp["bp_era"],
        "bp_whip_diff": away_bp["bp_whip"] - home_bp["bp_whip"],
        "bp_kbb_diff": home_bp["bp_kbb"] - away_bp["bp_kbb"],

        # Positive means away bullpen is more recently worked/fatigued.
        "bp_recent_pitches_diff": (
            away_bp["bp_recent_pitches"] - home_bp["bp_recent_pitches"]
        ),
        "bp_recent_ip_diff": away_bp["bp_recent_ip"] - home_bp["bp_recent_ip"],
        "bp_fatigue_diff": away_bp["bp_fatigue"] - home_bp["bp_fatigue"],
    }


def update_bullpen_state(state, row, side):
    relief_ip = safe_float(row.get(f"{side}_bullpen_ip", 0))
    relief_er = safe_int(row.get(f"{side}_bullpen_er", 0))
    relief_hits = safe_int(row.get(f"{side}_bullpen_hits", 0))
    relief_bb = safe_int(row.get(f"{side}_bullpen_bb", 0))
    relief_so = safe_int(row.get(f"{side}_bullpen_so", 0))
    relief_pitches = safe_int(row.get(f"{side}_bullpen_pitches", 0))

    state["games"] += 1
    state["relief_ip"] += relief_ip
    state["relief_er"] += relief_er
    state["relief_hits"] += relief_hits
    state["relief_bb"] += relief_bb
    state["relief_so"] += relief_so
    state["relief_pitches"] += relief_pitches

    state["recent_games"].append(
        {
            "ip": relief_ip,
            "er": relief_er,
            "hits": relief_hits,
            "bb": relief_bb,
            "so": relief_so,
            "pitches": relief_pitches,
        }
    )


def build_bullpen_feature_frame(raw_games):
    bullpen_states = defaultdict(create_bullpen_state)
    rows = []

    df = raw_games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    for _, game in df.iterrows():
        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        home_state = bullpen_states[home_team_id]
        away_state = bullpen_states[away_team_id]

        home_snapshot = bullpen_snapshot(home_state)
        away_snapshot = bullpen_snapshot(away_state)

        row = {
            "game_pk": game["game_pk"],
            **build_bullpen_diffs(home_snapshot, away_snapshot),
        }

        rows.append(row)

        # Update only AFTER creating the row.
        # This prevents current-game leakage.
        update_bullpen_state(home_state, game, "home")
        update_bullpen_state(away_state, game, "away")

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
        "bullpen_only": BULLPEN_FEATURES + SCHEDULE_FEATURES,
        "team_plus_bullpen": TEAM_FEATURES + BULLPEN_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_bullpen": (
            TEAM_FEATURES + STARTER_FEATURES + BULLPEN_FEATURES + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Bullpen Feature Set Comparison: 2025 Holdout ===")

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
    out.to_csv(EXPORT_DIR / "bullpen_feature_set_comparison.csv", index=False)

    return out, feature_sets


def run_bullpen_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Bullpen Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in BULLPEN_FEATURES:
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
    out.to_csv(EXPORT_DIR / "bullpen_ablation.csv", index=False)

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Bullpen Cross-Validation by Season ===")

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

    by_season.to_csv(EXPORT_DIR / "bullpen_cross_validation_by_season.csv", index=False)
    summary.to_csv(EXPORT_DIR / "bullpen_cross_validation_summary.csv", index=False)

    print("\n=== Bullpen Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Bullpen Research Lab ===")
    print("Testing whether bullpen features beat Production v1.0.")

    raw_games = pd.read_parquet(RAW_PATH)
    base_features = pd.read_parquet(FEATURES_PATH)

    bullpen_features = build_bullpen_feature_frame(raw_games)

    enriched = base_features.merge(
        bullpen_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(EXPORT_DIR / "features_with_bullpen_lab.parquet", index=False)

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Enriched rows: {len(enriched)}")
    print("\nBullpen feature preview:")
    print(enriched[BULLPEN_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_bullpen_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- exports/features_with_bullpen_lab.parquet")
    print("- exports/bullpen_feature_set_comparison.csv")
    print("- exports/bullpen_ablation.csv")
    print("- exports/bullpen_cross_validation_by_season.csv")
    print("- exports/bullpen_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Bullpen only promotes if team_plus_bullpen improves avg log loss/Brier")
    print("versus team_only across cross-validation.")


if __name__ == "__main__":
    main()