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


HOME_ROAD_FEATURES = [
    "home_road_win_pct_diff",
    "home_road_rpg_diff",
    "home_road_rapg_diff",
    "home_road_run_diff_pg_diff",
    "home_road_recent_win_pct_diff",
    "home_road_recent_rpg_diff",
    "home_road_recent_rapg_diff",
    "home_road_recent_run_diff_pg_diff",
    "home_home_games_scaled",
    "away_road_games_scaled",
    "home_road_games_scaled_diff",
]


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


def create_split_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "recent": deque(maxlen=10),
    }


def create_team_split_state():
    return {
        "home": create_split_state(),
        "away": create_split_state(),
    }


def update_split_state(state, runs_for, runs_against):
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


def snapshot_split_state(state):
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


def build_home_road_diffs(home_home, away_road):
    return {
        # Positive means the home team has the split advantage.
        "home_road_win_pct_diff": home_home["win_pct"] - away_road["win_pct"],
        "home_road_rpg_diff": home_home["rpg"] - away_road["rpg"],

        # Positive means away road pitching/defense allows more than home home pitching/defense.
        "home_road_rapg_diff": away_road["rapg"] - home_home["rapg"],

        "home_road_run_diff_pg_diff": (
            home_home["run_diff_pg"] - away_road["run_diff_pg"]
        ),

        "home_road_recent_win_pct_diff": (
            home_home["recent_win_pct"] - away_road["recent_win_pct"]
        ),

        "home_road_recent_rpg_diff": (
            home_home["recent_rpg"] - away_road["recent_rpg"]
        ),

        # Positive means away has recently allowed more on road than home has recently allowed at home.
        "home_road_recent_rapg_diff": (
            away_road["recent_rapg"] - home_home["recent_rapg"]
        ),

        "home_road_recent_run_diff_pg_diff": (
            home_home["recent_run_diff_pg"] - away_road["recent_run_diff_pg"]
        ),

        "home_home_games_scaled": home_home["games_scaled"],
        "away_road_games_scaled": away_road["games_scaled"],
        "home_road_games_scaled_diff": (
            home_home["games_scaled"] - away_road["games_scaled"]
        ),
    }


def build_home_road_feature_frame(raw_games):
    team_split_states = defaultdict(create_team_split_state)
    rows = []

    current_season = None

    df = raw_games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    for _, game in df.iterrows():
        season = int(game["season"])

        if current_season is None or season != current_season:
            team_split_states = defaultdict(create_team_split_state)
            current_season = season

        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        home_home_state = team_split_states[home_team_id]["home"]
        away_road_state = team_split_states[away_team_id]["away"]

        home_home_snapshot = snapshot_split_state(home_home_state)
        away_road_snapshot = snapshot_split_state(away_road_state)

        rows.append(
            {
                "game_pk": game["game_pk"],
                **build_home_road_diffs(home_home_snapshot, away_road_snapshot),
            }
        )

        # Update AFTER creating features to prevent current-game leakage.
        home_score = safe_int(game.get("home_score", 0))
        away_score = safe_int(game.get("away_score", 0))

        update_split_state(
            team_split_states[home_team_id]["home"],
            home_score,
            away_score,
        )

        update_split_state(
            team_split_states[away_team_id]["away"],
            away_score,
            home_score,
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
        "home_road_only": HOME_ROAD_FEATURES + SCHEDULE_FEATURES,
        "team_plus_home_road": (
            TEAM_FEATURES + HOME_ROAD_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_home_road": (
            TEAM_FEATURES
            + STARTER_FEATURES
            + HOME_ROAD_FEATURES
            + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Home/Road Split Feature Set Comparison: 2025 Holdout ===")

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
    out.to_csv(EXPORT_DIR / "home_road_feature_set_comparison.csv", index=False)

    return out, feature_sets


def run_home_road_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Home/Road Split Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in HOME_ROAD_FEATURES:
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
    out.to_csv(EXPORT_DIR / "home_road_ablation.csv", index=False)

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Home/Road Split Cross-Validation by Season ===")

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
        EXPORT_DIR / "home_road_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "home_road_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Home/Road Split Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Home/Road Split Research Lab ===")
    print("Testing whether team-specific home/road splits beat Production v1.0.")

    raw_games = pd.read_parquet(RAW_PATH)
    raw_games["date"] = pd.to_datetime(raw_games["date"])

    base_features = pd.read_parquet(FEATURES_PATH)

    home_road_features = build_home_road_feature_frame(raw_games)

    enriched = base_features.merge(
        home_road_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(
        EXPORT_DIR / "features_with_home_road_lab.parquet",
        index=False,
    )

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nHome/road feature preview:")
    print(enriched[HOME_ROAD_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_home_road_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- exports/features_with_home_road_lab.parquet")
    print("- exports/home_road_feature_set_comparison.csv")
    print("- exports/home_road_ablation.csv")
    print("- exports/home_road_cross_validation_by_season.csv")
    print("- exports/home_road_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Home/road only promotes if team_plus_home_road improves")
    print("avg log loss/Brier versus team_only across cross-validation.")


if __name__ == "__main__":
    main()