from collections import defaultdict
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

from lineup_lab import (  # noqa: E402
    RAW_PATH,
    FEATURES_PATH,
    LINEUP_CACHE_PATH,
    EXPORT_DIR,
    create_player_state,
    update_player_state,
    aggregate_lineup,
)


LINEUP_DELTA_FEATURES = [
    "lineup_ops_delta_diff",
    "lineup_obp_delta_diff",
    "lineup_slg_delta_diff",
    "lineup_iso_delta_diff",
    "lineup_top4_ops_delta_diff",
    "lineup_bottom5_ops_delta_diff",
    "lineup_recent_tb_pg_delta_diff",
    "lineup_bb_rate_delta_diff",
    "lineup_k_rate_delta_diff",
    "lineup_power_rate_delta_diff",
]


LINEUP_METRICS = [
    "lineup_ops",
    "lineup_obp",
    "lineup_slg",
    "lineup_iso",
    "lineup_top4_ops",
    "lineup_bottom5_ops",
    "lineup_recent_tb_pg",
    "lineup_bb_rate",
    "lineup_k_rate",
    "lineup_power_rate",
]


DEFAULT_LINEUP = {
    "lineup_ops": 0.730,
    "lineup_obp": 0.320,
    "lineup_slg": 0.410,
    "lineup_iso": 0.165,
    "lineup_top4_ops": 0.760,
    "lineup_bottom5_ops": 0.710,
    "lineup_recent_tb_pg": 1.35,
    "lineup_bb_rate": 0.085,
    "lineup_k_rate": 0.225,
    "lineup_power_rate": 0.032,
}


def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def create_team_lineup_state():
    return {
        "lineups_seen": 0,
        "sums": {metric: 0.0 for metric in LINEUP_METRICS},
    }


def team_normal_lineup_snapshot(state):
    count = state["lineups_seen"]

    if count <= 0:
        return DEFAULT_LINEUP.copy()

    return {
        metric: state["sums"][metric] / count
        for metric in LINEUP_METRICS
    }


def update_team_lineup_state(state, lineup_snapshot):
    state["lineups_seen"] += 1

    for metric in LINEUP_METRICS:
        state["sums"][metric] += float(lineup_snapshot.get(metric, DEFAULT_LINEUP[metric]))


def lineup_delta(current_lineup, normal_lineup):
    return {
        metric: float(current_lineup[metric]) - float(normal_lineup[metric])
        for metric in LINEUP_METRICS
    }


def build_lineup_delta_diffs(home_delta, away_delta):
    return {
        "lineup_ops_delta_diff": home_delta["lineup_ops"] - away_delta["lineup_ops"],
        "lineup_obp_delta_diff": home_delta["lineup_obp"] - away_delta["lineup_obp"],
        "lineup_slg_delta_diff": home_delta["lineup_slg"] - away_delta["lineup_slg"],
        "lineup_iso_delta_diff": home_delta["lineup_iso"] - away_delta["lineup_iso"],
        "lineup_top4_ops_delta_diff": (
            home_delta["lineup_top4_ops"] - away_delta["lineup_top4_ops"]
        ),
        "lineup_bottom5_ops_delta_diff": (
            home_delta["lineup_bottom5_ops"] - away_delta["lineup_bottom5_ops"]
        ),
        "lineup_recent_tb_pg_delta_diff": (
            home_delta["lineup_recent_tb_pg"] - away_delta["lineup_recent_tb_pg"]
        ),
        "lineup_bb_rate_delta_diff": (
            home_delta["lineup_bb_rate"] - away_delta["lineup_bb_rate"]
        ),

        # Positive means home lineup is striking out less than its normal lineup
        # compared with the away lineup's strikeout change.
        "lineup_k_rate_delta_diff": (
            away_delta["lineup_k_rate"] - home_delta["lineup_k_rate"]
        ),

        "lineup_power_rate_delta_diff": (
            home_delta["lineup_power_rate"] - away_delta["lineup_power_rate"]
        ),
    }


def build_lineup_delta_feature_frame(raw_games, lineup_cache):
    rows = []

    player_states = defaultdict(create_player_state)
    team_lineup_states = defaultdict(create_team_lineup_state)

    current_season = None

    raw_games = raw_games.copy()
    raw_games["date"] = pd.to_datetime(raw_games["date"])
    raw_games = raw_games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    lineup_cache = lineup_cache.copy()
    lineup_cache["date"] = pd.to_datetime(lineup_cache["date"])

    by_game = {
        game_pk: group.copy()
        for game_pk, group in lineup_cache.groupby("game_pk")
    }

    for _, game in raw_games.iterrows():
        season = int(game["season"])

        if current_season is None or season != current_season:
            player_states = defaultdict(create_player_state)
            team_lineup_states = defaultdict(create_team_lineup_state)
            current_season = season

        game_pk = game["game_pk"]
        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        game_lineups = by_game.get(game_pk, pd.DataFrame())

        home_starters = pd.DataFrame()
        away_starters = pd.DataFrame()

        if not game_lineups.empty:
            home_starters = game_lineups[
                (game_lineups["side"] == "home") & (game_lineups["is_starter"] == 1)
            ]

            away_starters = game_lineups[
                (game_lineups["side"] == "away") & (game_lineups["is_starter"] == 1)
            ]

        # Current lineup quality using player stats entering the game.
        home_current = aggregate_lineup(home_starters, player_states)
        away_current = aggregate_lineup(away_starters, player_states)

        # Team's normal lineup quality entering the game.
        home_normal = team_normal_lineup_snapshot(team_lineup_states[home_team_id])
        away_normal = team_normal_lineup_snapshot(team_lineup_states[away_team_id])

        home_delta = lineup_delta(home_current, home_normal)
        away_delta = lineup_delta(away_current, away_normal)

        rows.append(
            {
                "game_pk": game_pk,
                **build_lineup_delta_diffs(home_delta, away_delta),
            }
        )

        # Update team normal lineup using only the pre-game lineup snapshot.
        # This avoids using same-game batting results.
        update_team_lineup_state(team_lineup_states[home_team_id], home_current)
        update_team_lineup_state(team_lineup_states[away_team_id], away_current)

        # Update player states AFTER feature creation.
        # This prevents same-game performance leakage.
        if not game_lineups.empty:
            for _, batter_row in game_lineups.iterrows():
                update_player_state(player_states[batter_row["player_id"]], batter_row)

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
        "lineup_delta_only": LINEUP_DELTA_FEATURES + SCHEDULE_FEATURES,
        "team_plus_lineup_delta": (
            TEAM_FEATURES + LINEUP_DELTA_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_lineup_delta": (
            TEAM_FEATURES
            + STARTER_FEATURES
            + LINEUP_DELTA_FEATURES
            + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Lineup Delta Feature Set Comparison: 2025 Holdout ===")

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
    out.to_csv(EXPORT_DIR / "lineup_delta_feature_set_comparison.csv", index=False)

    return out, feature_sets


def run_lineup_delta_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Lineup Delta Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in LINEUP_DELTA_FEATURES:
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
    out.to_csv(EXPORT_DIR / "lineup_delta_ablation.csv", index=False)

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Lineup Delta Cross-Validation by Season ===")

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
        EXPORT_DIR / "lineup_delta_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "lineup_delta_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Lineup Delta Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Lineup Delta Research Lab ===")
    print("Testing whether today's lineup vs normal lineup beats Production v1.0.")

    if not LINEUP_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Missing lineup cache: {LINEUP_CACHE_PATH}. "
            "Run model_lab/lineup_lab.py first."
        )

    raw_games = pd.read_parquet(RAW_PATH)
    raw_games["date"] = pd.to_datetime(raw_games["date"])

    base_features = pd.read_parquet(FEATURES_PATH)
    lineup_cache = pd.read_parquet(LINEUP_CACHE_PATH)

    lineup_delta_features = build_lineup_delta_feature_frame(raw_games, lineup_cache)

    enriched = base_features.merge(
        lineup_delta_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(
        EXPORT_DIR / "features_with_lineup_delta_lab.parquet",
        index=False,
    )

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nLineup delta feature preview:")
    print(enriched[LINEUP_DELTA_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_lineup_delta_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- exports/features_with_lineup_delta_lab.parquet")
    print("- exports/lineup_delta_feature_set_comparison.csv")
    print("- exports/lineup_delta_ablation.csv")
    print("- exports/lineup_delta_cross_validation_by_season.csv")
    print("- exports/lineup_delta_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Lineup delta only promotes if team_plus_lineup_delta improves")
    print("avg log loss/Brier versus team_only across cross-validation.")


if __name__ == "__main__":
    main()