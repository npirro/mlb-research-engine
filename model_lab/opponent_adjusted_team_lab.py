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


OPP_ADJ_FEATURES = [
    "opp_adj_win_pct_diff",
    "opp_adj_run_diff_pg_diff",
    "opp_adj_offense_diff",
    "opp_adj_defense_diff",
    "opp_adj_recent_win_pct_diff",
    "opp_adj_recent_run_diff_pg_diff",
    "sos_win_pct_diff",
    "sos_run_diff_pg_diff",
    "sos_rpg_diff",
    "sos_rapg_diff",
    "home_opponents_faced_scaled",
    "away_opponents_faced_scaled",
    "opponents_faced_scaled_diff",
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


def build_opponent_adjusted_features(
    home_state,
    away_state,
    team_states,
):
    home = snapshot_team_state(home_state)
    away = snapshot_team_state(away_state)

    home_sos = schedule_strength_snapshot(
        team_states,
        home_state["opponents"],
    )

    away_sos = schedule_strength_snapshot(
        team_states,
        away_state["opponents"],
    )

    home_recent_sos = schedule_strength_snapshot(
        team_states,
        home_state["recent_opponents"],
    )

    away_recent_sos = schedule_strength_snapshot(
        team_states,
        away_state["recent_opponents"],
    )

    # Opponent-adjusted win strength:
    # "How good has this team been compared with who it has faced?"
    home_adj_win = home["win_pct"] - home_sos["opp_win_pct"]
    away_adj_win = away["win_pct"] - away_sos["opp_win_pct"]

    # Opponent-adjusted run differential:
    # "Is this run differential good relative to schedule strength?"
    home_adj_run_diff = home["run_diff_pg"] - home_sos["opp_run_diff_pg"]
    away_adj_run_diff = away["run_diff_pg"] - away_sos["opp_run_diff_pg"]

    # Opponent-adjusted offense:
    # "Does this team score more than its opponents usually allow?"
    home_adj_offense = home["rpg"] - home_sos["opp_rapg"]
    away_adj_offense = away["rpg"] - away_sos["opp_rapg"]

    # Opponent-adjusted defense:
    # "Does this team allow fewer runs than its opponents usually score?"
    home_adj_defense = home_sos["opp_rpg"] - home["rapg"]
    away_adj_defense = away_sos["opp_rpg"] - away["rapg"]

    home_recent_adj_win = (
        home["recent_win_pct"] - home_recent_sos["opp_win_pct"]
    )
    away_recent_adj_win = (
        away["recent_win_pct"] - away_recent_sos["opp_win_pct"]
    )

    home_recent_adj_run_diff = (
        home["recent_run_diff_pg"] - home_recent_sos["opp_run_diff_pg"]
    )
    away_recent_adj_run_diff = (
        away["recent_run_diff_pg"] - away_recent_sos["opp_run_diff_pg"]
    )

    return {
        # Positive means the home team has the advantage.
        "opp_adj_win_pct_diff": home_adj_win - away_adj_win,
        "opp_adj_run_diff_pg_diff": home_adj_run_diff - away_adj_run_diff,
        "opp_adj_offense_diff": home_adj_offense - away_adj_offense,
        "opp_adj_defense_diff": home_adj_defense - away_adj_defense,
        "opp_adj_recent_win_pct_diff": (
            home_recent_adj_win - away_recent_adj_win
        ),
        "opp_adj_recent_run_diff_pg_diff": (
            home_recent_adj_run_diff - away_recent_adj_run_diff
        ),

        # Pure strength-of-schedule gaps.
        "sos_win_pct_diff": home_sos["opp_win_pct"] - away_sos["opp_win_pct"],
        "sos_run_diff_pg_diff": (
            home_sos["opp_run_diff_pg"] - away_sos["opp_run_diff_pg"]
        ),
        "sos_rpg_diff": home_sos["opp_rpg"] - away_sos["opp_rpg"],
        "sos_rapg_diff": home_sos["opp_rapg"] - away_sos["opp_rapg"],

        # Sample-size awareness.
        "home_opponents_faced_scaled": home_sos["opponents_faced_scaled"],
        "away_opponents_faced_scaled": away_sos["opponents_faced_scaled"],
        "opponents_faced_scaled_diff": (
            home_sos["opponents_faced_scaled"]
            - away_sos["opponents_faced_scaled"]
        ),
    }


def build_opponent_adjusted_feature_frame(raw_games):
    rows = []

    df = raw_games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["game_date"] = df["date"].dt.date
    df = df.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)

    for season in sorted(df["season"].dropna().unique()):
        team_states = defaultdict(create_team_state)
        season_df = df[df["season"] == season].copy()

        for game_date, day_games in season_df.groupby("game_date", sort=True):
            # First create features for every game on this date.
            # Then update states after the full date is processed.
            # This prevents same-day/doubleheader leakage.
            for _, game in day_games.iterrows():
                home_team_id = game["home_team_id"]
                away_team_id = game["away_team_id"]

                home_state = team_states[home_team_id]
                away_state = team_states[away_team_id]

                rows.append(
                    {
                        "game_pk": game["game_pk"],
                        **build_opponent_adjusted_features(
                            home_state=home_state,
                            away_state=away_state,
                            team_states=team_states,
                        ),
                    }
                )

            # Update AFTER creating that day's features.
            for _, game in day_games.iterrows():
                home_team_id = game["home_team_id"]
                away_team_id = game["away_team_id"]

                home_score = safe_int(game.get("home_score", 0))
                away_score = safe_int(game.get("away_score", 0))

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
        "opponent_adjusted_only": OPP_ADJ_FEATURES + SCHEDULE_FEATURES,
        "team_plus_opponent_adjusted": (
            TEAM_FEATURES + OPP_ADJ_FEATURES + SCHEDULE_FEATURES
        ),
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_opponent_adjusted": (
            TEAM_FEATURES
            + STARTER_FEATURES
            + OPP_ADJ_FEATURES
            + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Opponent-Adjusted Team Feature Set Comparison: 2025 Holdout ===")

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
        EXPORT_DIR / "opponent_adjusted_feature_set_comparison.csv",
        index=False,
    )

    return out, feature_sets


def run_opponent_adjusted_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Opponent-Adjusted Team Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in OPP_ADJ_FEATURES:
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
        EXPORT_DIR / "opponent_adjusted_ablation.csv",
        index=False,
    )

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Opponent-Adjusted Team Cross-Validation by Season ===")

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
        EXPORT_DIR / "opponent_adjusted_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "opponent_adjusted_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Opponent-Adjusted Team Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Opponent-Adjusted Team Research Lab ===")
    print("Testing whether schedule-adjusted team quality beats Production v1.0.")

    raw_games = pd.read_parquet(RAW_PATH)
    raw_games["date"] = pd.to_datetime(raw_games["date"])

    base_features = pd.read_parquet(FEATURES_PATH)

    opponent_adjusted_features = build_opponent_adjusted_feature_frame(raw_games)

    enriched = base_features.merge(
        opponent_adjusted_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(
        EXPORT_DIR / "features_with_opponent_adjusted_lab.parquet",
        index=False,
    )

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Opponent-adjusted feature rows: {len(opponent_adjusted_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nOpponent-adjusted feature preview:")
    print(enriched[OPP_ADJ_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_opponent_adjusted_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- exports/features_with_opponent_adjusted_lab.parquet")
    print("- exports/opponent_adjusted_feature_set_comparison.csv")
    print("- exports/opponent_adjusted_ablation.csv")
    print("- exports/opponent_adjusted_cross_validation_by_season.csv")
    print("- exports/opponent_adjusted_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Opponent-adjusted team strength only promotes if")
    print("team_plus_opponent_adjusted improves avg log loss/Brier")
    print("versus team_only across cross-validation.")


if __name__ == "__main__":
    main()