import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
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
    SCHEDULE_FEATURES,
)


# =========================
# Paths
# =========================

BASE_FEATURES_PATH = Path("data/processed/features_v4_feature_factory.parquet")

ENV_FEATURES_PATH = Path("exports/features_with_environment_lab.parquet")
OPP_ADJ_FEATURES_PATH = Path("exports/features_with_opponent_adjusted_lab.parquet")
PLATOON_FEATURES_PATH = Path("exports/features_with_platoon_lab.parquet")

EXPORT_DIR = Path("exports")
MODEL_DIR = Path("model_artifacts")

CANDIDATE_MODEL_PATH = MODEL_DIR / "mlb_candidate_v1_1_logistic_model.joblib"


# =========================
# Frozen Candidate Definition
# =========================

CANDIDATE_VERSION = "candidate_v1_1"

CANDIDATE_ADDED_FEATURES = [
    "vs_hand_games_scaled_diff",
    "env_temp",
    "opp_adj_offense_diff",
]

EXTRA_FEATURE_SOURCES = {
    "env_temp": ENV_FEATURES_PATH,
    "opp_adj_offense_diff": OPP_ADJ_FEATURES_PATH,
    "vs_hand_games_scaled_diff": PLATOON_FEATURES_PATH,
}


# =========================
# Helpers
# =========================

def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def build_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )


def load_base_features():
    if not BASE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing base feature file: {BASE_FEATURES_PATH}")

    df = pd.read_parquet(BASE_FEATURES_PATH)

    required = ["game_pk", "season", "home_win"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Base feature file missing required columns: {missing}")

    return df


def merge_candidate_features(base_df):
    df = base_df.copy()

    for feature, path in EXTRA_FEATURE_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required feature source for {feature}: {path}\n"
                "Run the prior lab that created this parquet export."
            )

        extra = pd.read_parquet(path)

        if "game_pk" not in extra.columns:
            raise ValueError(f"{path} does not contain game_pk")

        if feature not in extra.columns:
            raise ValueError(
                f"{path} does not contain required feature: {feature}"
            )

        slim = extra[["game_pk", feature]].copy()

        if feature in df.columns:
            df = df.drop(columns=[feature])

        df = df.merge(
            slim,
            on="game_pk",
            how="left",
        )

        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)

    return df


def evaluate_holdout(df, features, test_season=None):
    if test_season is None:
        test_season = int(df["season"].max())

    features = unique_features(features)

    data = df.dropna(subset=features + ["home_win"]).copy()

    train_df = data[data["season"] < test_season]
    test_df = data[data["season"] == test_season]

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Not enough rows for test season {test_season}. "
            f"Train={len(train_df)}, Test={len(test_df)}"
        )

    model = build_model()
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


def evaluate_cross_validation(df, features, label):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    for season in test_seasons:
        result = evaluate_holdout(df, features, test_season=season)

        rows.append(
            {
                "feature_set": label,
                **result,
            }
        )

    by_season = pd.DataFrame(rows)

    summary = {
        "feature_set": label,
        "seasons_tested": int(by_season["test_season"].nunique()),
        "avg_accuracy": float(by_season["accuracy"].mean()),
        "avg_log_loss": float(by_season["log_loss"].mean()),
        "avg_brier": float(by_season["brier"].mean()),
        "worst_log_loss": float(by_season["log_loss"].max()),
        "best_log_loss": float(by_season["log_loss"].min()),
    }

    return by_season, summary


def compare_candidate_to_baseline(baseline_by_season, candidate_by_season):
    merged = candidate_by_season.merge(
        baseline_by_season[
            [
                "test_season",
                "accuracy",
                "log_loss",
                "brier",
            ]
        ].rename(
            columns={
                "accuracy": "baseline_accuracy",
                "log_loss": "baseline_log_loss",
                "brier": "baseline_brier",
            }
        ),
        on="test_season",
        how="left",
    )

    merged["accuracy_delta"] = merged["accuracy"] - merged["baseline_accuracy"]
    merged["log_loss_delta"] = merged["log_loss"] - merged["baseline_log_loss"]
    merged["brier_delta"] = merged["brier"] - merged["baseline_brier"]

    return merged


def promotion_verdict(baseline_summary, candidate_summary, comparison_by_season):
    improves_avg_log_loss = (
        candidate_summary["avg_log_loss"] < baseline_summary["avg_log_loss"]
    )

    improves_avg_brier = (
        candidate_summary["avg_brier"] < baseline_summary["avg_brier"]
    )

    improves_every_season_log_loss = (
        comparison_by_season["log_loss_delta"].max() < 0
    )

    improves_every_season_brier = (
        comparison_by_season["brier_delta"].max() < 0
    )

    strict_promote = (
        improves_avg_log_loss
        and improves_avg_brier
        and improves_every_season_log_loss
        and improves_every_season_brier
    )

    return {
        "improves_avg_log_loss": bool(improves_avg_log_loss),
        "improves_avg_brier": bool(improves_avg_brier),
        "improves_every_season_log_loss": bool(improves_every_season_log_loss),
        "improves_every_season_brier": bool(improves_every_season_brier),
        "strict_promote": bool(strict_promote),
    }


def train_final_candidate_model(df, features):
    features = unique_features(features)

    data = df.dropna(subset=features + ["home_win"]).copy()

    model = build_model()
    model.fit(data[features], data["home_win"])

    return model, data


def save_coefficients(model, features):
    logistic = model.named_steps["logistic"]

    coef_df = pd.DataFrame(
        {
            "feature": features,
            "coefficient_scaled": logistic.coef_[0],
        }
    )

    coef_df["abs_coefficient_scaled"] = coef_df["coefficient_scaled"].abs()

    coef_df = coef_df.sort_values(
        "abs_coefficient_scaled",
        ascending=False,
    )

    coef_df.to_csv(
        EXPORT_DIR / "candidate_v1_1_coefficients.csv",
        index=False,
    )

    return coef_df


def main():
    EXPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=== Candidate v1.1 Promotion Lab ===")
    print("Freezing and validating the first challenger to Production v1.0.")
    print("\nCandidate added features:")
    for f in CANDIDATE_ADDED_FEATURES:
        print(f"- {f}")

    base = load_base_features()
    enriched = merge_candidate_features(base)

    baseline_features = unique_features(TEAM_FEATURES + SCHEDULE_FEATURES)
    candidate_features = unique_features(
        TEAM_FEATURES + SCHEDULE_FEATURES + CANDIDATE_ADDED_FEATURES
    )

    missing_baseline = [f for f in baseline_features if f not in enriched.columns]
    missing_candidate = [f for f in candidate_features if f not in enriched.columns]

    if missing_baseline:
        raise ValueError(f"Missing baseline features: {missing_baseline}")

    if missing_candidate:
        raise ValueError(f"Missing candidate features: {missing_candidate}")

    print(f"\nRows: {len(enriched)}")
    print(f"Baseline feature count: {len(baseline_features)}")
    print(f"Candidate feature count: {len(candidate_features)}")

    print("\nCandidate feature preview:")
    print(enriched[CANDIDATE_ADDED_FEATURES].head().to_string(index=False))

    print("\n=== 2025 Holdout Comparison ===")

    baseline_2025 = evaluate_holdout(enriched, baseline_features, test_season=2025)
    candidate_2025 = evaluate_holdout(enriched, candidate_features, test_season=2025)

    print(
        f"team_only 2025: "
        f"Accuracy={baseline_2025['accuracy']:.6f}, "
        f"LogLoss={baseline_2025['log_loss']:.6f}, "
        f"Brier={baseline_2025['brier']:.6f}"
    )

    print(
        f"candidate_v1_1 2025: "
        f"Accuracy={candidate_2025['accuracy']:.6f}, "
        f"LogLoss={candidate_2025['log_loss']:.6f}, "
        f"Brier={candidate_2025['brier']:.6f}"
    )

    print("\n=== Cross-Validation ===")

    baseline_by_season, baseline_summary = evaluate_cross_validation(
        enriched,
        baseline_features,
        "team_only",
    )

    candidate_by_season, candidate_summary = evaluate_cross_validation(
        enriched,
        candidate_features,
        CANDIDATE_VERSION,
    )

    comparison_by_season = compare_candidate_to_baseline(
        baseline_by_season,
        candidate_by_season,
    )

    print("\nBaseline by season:")
    print(baseline_by_season.to_string(index=False))

    print("\nCandidate by season:")
    print(candidate_by_season.to_string(index=False))

    print("\nCandidate vs baseline by season:")
    print(
        comparison_by_season[
            [
                "test_season",
                "accuracy_delta",
                "log_loss_delta",
                "brier_delta",
            ]
        ].to_string(index=False)
    )

    summary_df = pd.DataFrame([baseline_summary, candidate_summary])

    print("\nCross-validation summary:")
    print(summary_df.to_string(index=False))

    verdict = promotion_verdict(
        baseline_summary,
        candidate_summary,
        comparison_by_season,
    )

    print("\n=== Promotion Verdict ===")

    for key, value in verdict.items():
        print(f"{key}: {value}")

    if verdict["strict_promote"]:
        print("\nVERDICT: CANDIDATE v1.1 PASSES RESEARCH PROMOTION CHECK.")
        print("Next step: build live feature generation before shipping.")
    else:
        print("\nVERDICT: REJECT. Do not promote.")

    print("\n=== Training Final Candidate Artifact ===")

    final_model, training_data = train_final_candidate_model(
        enriched,
        candidate_features,
    )

    coef_df = save_coefficients(final_model, candidate_features)

    artifact = {
        "model": final_model,
        "features": candidate_features,
        "model_version": CANDIDATE_VERSION,
        "model_family": "logistic_regression_scaled",
        "feature_set_name": "team_plus_env_temp_opp_adj_offense_vs_hand_games_scaled",
        "production_parent": "production_v1_team_only",
        "candidate_added_features": CANDIDATE_ADDED_FEATURES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(training_data)),
        "training_seasons": sorted(
            int(s) for s in training_data["season"].dropna().unique()
        ),
        "holdout_2025": {
            "baseline": baseline_2025,
            "candidate": candidate_2025,
        },
        "cross_validation": {
            "baseline_summary": baseline_summary,
            "candidate_summary": candidate_summary,
            "verdict": verdict,
        },
    }

    joblib.dump(artifact, CANDIDATE_MODEL_PATH)

    baseline_by_season.to_csv(
        EXPORT_DIR / "candidate_v1_1_baseline_cv_by_season.csv",
        index=False,
    )

    candidate_by_season.to_csv(
        EXPORT_DIR / "candidate_v1_1_candidate_cv_by_season.csv",
        index=False,
    )

    comparison_by_season.to_csv(
        EXPORT_DIR / "candidate_v1_1_vs_baseline_by_season.csv",
        index=False,
    )

    summary_df.to_csv(
        EXPORT_DIR / "candidate_v1_1_cv_summary.csv",
        index=False,
    )

    with open(EXPORT_DIR / "candidate_v1_1_feature_list.json", "w") as f:
        json.dump(
            {
                "model_version": CANDIDATE_VERSION,
                "baseline_features": baseline_features,
                "candidate_added_features": CANDIDATE_ADDED_FEATURES,
                "candidate_features": candidate_features,
                "verdict": verdict,
            },
            f,
            indent=2,
        )

    with open(EXPORT_DIR / "candidate_v1_1_model_card.md", "w") as f:
        f.write("# Candidate v1.1 Model Card\n\n")
        f.write("## Status\n\n")
        f.write(
            "Research promotion passed.\n\n"
            if verdict["strict_promote"]
            else "Research promotion failed.\n\n"
        )
        f.write("## Added Features\n\n")
        for feature in CANDIDATE_ADDED_FEATURES:
            f.write(f"- `{feature}`\n")
        f.write("\n## Cross-Validation Summary\n\n")
        f.write("```text\n"); f.write(summary_df.to_string(index=False)); f.write("\n```")
        f.write("\n\n## Verdict\n\n")
        for key, value in verdict.items():
            f.write(f"- `{key}`: `{value}`\n")

    print(f"Saved candidate model artifact:")
    print(f"- {CANDIDATE_MODEL_PATH}")

    print("\nSaved reports:")
    print("- exports/candidate_v1_1_baseline_cv_by_season.csv")
    print("- exports/candidate_v1_1_candidate_cv_by_season.csv")
    print("- exports/candidate_v1_1_vs_baseline_by_season.csv")
    print("- exports/candidate_v1_1_cv_summary.csv")
    print("- exports/candidate_v1_1_coefficients.csv")
    print("- exports/candidate_v1_1_feature_list.json")
    print("- exports/candidate_v1_1_model_card.md")

    print("\nTop coefficients:")
    print(
        coef_df[
            [
                "feature",
                "coefficient_scaled",
                "abs_coefficient_scaled",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nImportant:")
    print("This creates a candidate artifact only.")
    print("Do not replace the production app model until live feature generation exists.")


if __name__ == "__main__":
    main()
