from itertools import combinations
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
    SCHEDULE_FEATURES,
)


FEATURES_PATH = Path("data/processed/features_v4_feature_factory.parquet")
EXPORT_DIR = Path("exports")


EXTRA_FEATURE_FILES = [
    Path("exports/features_with_environment_lab.parquet"),
    Path("exports/features_with_opponent_adjusted_lab.parquet"),
    Path("exports/features_with_platoon_lab.parquet"),
    Path("exports/features_with_home_road_lab.parquet"),
    Path("exports/features_with_lineup_lab.parquet"),
    Path("exports/features_with_lineup_delta_lab.parquet"),
]


# These are the individual "maybe" signals we saw inside rejected labs.
# The bakeoff tests whether any individual needle, or small combo of needles,
# can beat production without bringing the whole noisy feature group.
CANDIDATE_FEATURE_ALIASES = {
    "opp_adj_offense_diff": [
        "opp_adj_offense_diff",
    ],
    "env_temp": [
        "env_temp",
        "temperature",
        "game_temp",
        "temp",
    ],
    "lineup_platoon_adv_pct_diff": [
        "lineup_platoon_adv_pct_diff",
    ],
    "away_lineup_platoon_adv_pct": [
        "away_lineup_platoon_adv_pct",
    ],
    "home_faces_lhp": [
        "home_faces_lhp",
    ],
    "away_faces_lhp": [
        "away_faces_lhp",
    ],
    "vs_hand_games_scaled_diff": [
        "vs_hand_games_scaled_diff",
    ],
    "away_vs_hand_games_scaled": [
        "away_vs_hand_games_scaled",
    ],
    "home_road_rapg_diff": [
        "home_road_rapg_diff",
    ],
}


MAX_PAIR_FEATURES = 12
MAX_TRIPLE_FEATURES = 12


def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def safe_feature_list(df, features):
    return [f for f in unique_features(features) if f in df.columns]


def load_base_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing base feature file: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)

    required = ["game_pk", "season", "home_win"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Base feature file missing required columns: {missing}")

    return df


def find_alias_column(df, aliases):
    lower_map = {c.lower(): c for c in df.columns}

    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    return None


def merge_candidate_features(base_df):
    df = base_df.copy()
    found = {}
    source_map = {}
    missing = []

    for canonical_name, aliases in CANDIDATE_FEATURE_ALIASES.items():
        found[canonical_name] = False

    for path in EXTRA_FEATURE_FILES:
        if not path.exists():
            continue

        try:
            extra = pd.read_parquet(path)
        except Exception as e:
            print(f"Skipping unreadable file {path}: {e}")
            continue

        if "game_pk" not in extra.columns:
            print(f"Skipping {path}: no game_pk column.")
            continue

        add_cols = ["game_pk"]
        rename_map = {}

        for canonical_name, aliases in CANDIDATE_FEATURE_ALIASES.items():
            if found[canonical_name]:
                continue

            actual_col = find_alias_column(extra, aliases)

            if actual_col is not None:
                add_cols.append(actual_col)
                rename_map[actual_col] = canonical_name
                found[canonical_name] = True
                source_map[canonical_name] = str(path)

        add_cols = unique_features(add_cols)

        if len(add_cols) <= 1:
            continue

        slim = extra[add_cols].copy()
        slim = slim.rename(columns=rename_map)

        # Avoid merge collisions if a canonical column already exists.
        merge_cols = ["game_pk"] + [
            c for c in slim.columns if c != "game_pk" and c not in df.columns
        ]

        if len(merge_cols) <= 1:
            continue

        df = df.merge(
            slim[merge_cols],
            on="game_pk",
            how="left",
        )

    for canonical_name in CANDIDATE_FEATURE_ALIASES:
        if canonical_name not in df.columns:
            missing.append(canonical_name)

    available = [
        c for c in CANDIDATE_FEATURE_ALIASES.keys()
        if c in df.columns
    ]

    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df, available, missing, source_map


def evaluate_holdout(df, features, test_season=None):
    if test_season is None:
        test_season = int(df["season"].max())

    features = unique_features(features)

    data = df.dropna(subset=features + ["home_win"]).copy()

    train_df = data[data["season"] < test_season]
    test_df = data[data["season"] == test_season]

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Not enough data for test_season={test_season}. "
            f"Train rows={len(train_df)}, test rows={len(test_df)}"
        )

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
        "seasons_tested": by_season["test_season"].nunique(),
        "avg_accuracy": by_season["accuracy"].mean(),
        "avg_log_loss": by_season["log_loss"].mean(),
        "avg_brier": by_season["brier"].mean(),
        "worst_log_loss": by_season["log_loss"].max(),
        "best_log_loss": by_season["log_loss"].min(),
    }

    return by_season, summary


def evaluate_combo(df, base_features, combo, baseline_summary, baseline_by_season):
    combo = list(combo)
    label = "team_plus_" + "__".join(combo)

    features = base_features + combo

    by_season, summary = evaluate_cross_validation(df, features, label)

    summary["num_added_features"] = len(combo)
    summary["added_features"] = ", ".join(combo)

    summary["avg_log_loss_delta"] = (
        summary["avg_log_loss"] - baseline_summary["avg_log_loss"]
    )

    summary["avg_brier_delta"] = (
        summary["avg_brier"] - baseline_summary["avg_brier"]
    )

    summary["avg_accuracy_delta"] = (
        summary["avg_accuracy"] - baseline_summary["avg_accuracy"]
    )

    merged = by_season.merge(
        baseline_by_season[
            [
                "test_season",
                "log_loss",
                "brier",
                "accuracy",
            ]
        ].rename(
            columns={
                "log_loss": "baseline_log_loss",
                "brier": "baseline_brier",
                "accuracy": "baseline_accuracy",
            }
        ),
        on="test_season",
        how="left",
    )

    merged["season_log_loss_delta"] = (
        merged["log_loss"] - merged["baseline_log_loss"]
    )

    merged["season_brier_delta"] = (
        merged["brier"] - merged["baseline_brier"]
    )

    summary["beats_avg_log_loss"] = summary["avg_log_loss_delta"] < 0
    summary["beats_avg_brier"] = summary["avg_brier_delta"] < 0
    summary["beats_every_season_log_loss"] = (
        merged["season_log_loss_delta"].max() < 0
    )
    summary["beats_every_season_brier"] = (
        merged["season_brier_delta"].max() < 0
    )

    summary["strict_promotion_candidate"] = (
        summary["beats_avg_log_loss"]
        and summary["beats_avg_brier"]
        and summary["beats_every_season_log_loss"]
        and summary["beats_every_season_brier"]
    )

    return by_season, summary


def run_bakeoff(df, available_candidates):
    EXPORT_DIR.mkdir(exist_ok=True)

    base_features = unique_features(TEAM_FEATURES + SCHEDULE_FEATURES)

    base_features = safe_feature_list(df, base_features)

    if not base_features:
        raise ValueError("No baseline features found in dataframe.")

    print("\n=== Baseline Cross-Validation ===")

    baseline_by_season, baseline_summary = evaluate_cross_validation(
        df,
        base_features,
        "team_only",
    )

    print(
        f"team_only: "
        f"AvgAccuracy={baseline_summary['avg_accuracy']:.6f}, "
        f"AvgLogLoss={baseline_summary['avg_log_loss']:.6f}, "
        f"AvgBrier={baseline_summary['avg_brier']:.6f}, "
        f"WorstLogLoss={baseline_summary['worst_log_loss']:.6f}"
    )

    all_by_season = [baseline_by_season]
    all_summaries = []

    print("\n=== Single Feature Bakeoff ===")

    single_summaries = []

    for feature in available_candidates:
        by_season, summary = evaluate_combo(
            df=df,
            base_features=base_features,
            combo=[feature],
            baseline_summary=baseline_summary,
            baseline_by_season=baseline_by_season,
        )

        all_by_season.append(by_season)
        all_summaries.append(summary)
        single_summaries.append(summary)

        print(
            f"+ {feature}: "
            f"AvgLogLoss={summary['avg_log_loss']:.6f} "
            f"({summary['avg_log_loss_delta']:+.6f}), "
            f"AvgBrier={summary['avg_brier']:.6f} "
            f"({summary['avg_brier_delta']:+.6f}), "
            f"StrictPromote={summary['strict_promotion_candidate']}"
        )

    single_results = (
        pd.DataFrame(single_summaries)
        .sort_values(["avg_log_loss", "avg_brier"])
        .reset_index(drop=True)
    )

    # Use the best singles as ingredients for pair/triple search.
    combo_pool = (
        single_results["added_features"]
        .head(MAX_PAIR_FEATURES)
        .tolist()
    )

    print("\n=== Pair Feature Bakeoff ===")
    pair_summaries = []

    for combo in combinations(combo_pool, 2):
        by_season, summary = evaluate_combo(
            df=df,
            base_features=base_features,
            combo=combo,
            baseline_summary=baseline_summary,
            baseline_by_season=baseline_by_season,
        )

        all_by_season.append(by_season)
        all_summaries.append(summary)
        pair_summaries.append(summary)

    pair_results = (
        pd.DataFrame(pair_summaries)
        .sort_values(["avg_log_loss", "avg_brier"])
        .reset_index(drop=True)
        if pair_summaries
        else pd.DataFrame()
    )

    if len(pair_results):
        print(pair_results.head(15)[
            [
                "added_features",
                "avg_accuracy",
                "avg_log_loss",
                "avg_log_loss_delta",
                "avg_brier",
                "avg_brier_delta",
                "strict_promotion_candidate",
            ]
        ].to_string(index=False))
    else:
        print("Not enough available candidates for pair testing.")

    triple_pool = (
        single_results["added_features"]
        .head(MAX_TRIPLE_FEATURES)
        .tolist()
    )

    print("\n=== Triple Feature Bakeoff ===")
    triple_summaries = []

    for combo in combinations(triple_pool, 3):
        by_season, summary = evaluate_combo(
            df=df,
            base_features=base_features,
            combo=combo,
            baseline_summary=baseline_summary,
            baseline_by_season=baseline_by_season,
        )

        all_by_season.append(by_season)
        all_summaries.append(summary)
        triple_summaries.append(summary)

    triple_results = (
        pd.DataFrame(triple_summaries)
        .sort_values(["avg_log_loss", "avg_brier"])
        .reset_index(drop=True)
        if triple_summaries
        else pd.DataFrame()
    )

    if len(triple_results):
        print(triple_results.head(15)[
            [
                "added_features",
                "avg_accuracy",
                "avg_log_loss",
                "avg_log_loss_delta",
                "avg_brier",
                "avg_brier_delta",
                "strict_promotion_candidate",
            ]
        ].to_string(index=False))
    else:
        print("Not enough available candidates for triple testing.")

    all_results = (
        pd.DataFrame(all_summaries)
        .sort_values(["avg_log_loss", "avg_brier"])
        .reset_index(drop=True)
    )

    all_cv_by_season = pd.concat(all_by_season, ignore_index=True)

    single_results.to_csv(
        EXPORT_DIR / "promising_bakeoff_single_results.csv",
        index=False,
    )

    pair_results.to_csv(
        EXPORT_DIR / "promising_bakeoff_pair_results.csv",
        index=False,
    )

    triple_results.to_csv(
        EXPORT_DIR / "promising_bakeoff_triple_results.csv",
        index=False,
    )

    all_results.to_csv(
        EXPORT_DIR / "promising_bakeoff_all_results.csv",
        index=False,
    )

    all_cv_by_season.to_csv(
        EXPORT_DIR / "promising_bakeoff_cross_validation_by_season.csv",
        index=False,
    )

    print("\n=== Overall Bakeoff Top 20 ===")

    print(all_results.head(20)[
        [
            "added_features",
            "num_added_features",
            "avg_accuracy",
            "avg_accuracy_delta",
            "avg_log_loss",
            "avg_log_loss_delta",
            "avg_brier",
            "avg_brier_delta",
            "worst_log_loss",
            "strict_promotion_candidate",
        ]
    ].to_string(index=False))

    promotion_candidates = all_results[
        all_results["strict_promotion_candidate"] == True
    ].copy()

    print("\n=== Strict Promotion Candidates ===")

    if len(promotion_candidates):
        print(promotion_candidates[
            [
                "added_features",
                "num_added_features",
                "avg_accuracy",
                "avg_log_loss",
                "avg_log_loss_delta",
                "avg_brier",
                "avg_brier_delta",
                "worst_log_loss",
            ]
        ].to_string(index=False))
    else:
        print("None.")

    print("\nSaved:")
    print("- exports/promising_bakeoff_single_results.csv")
    print("- exports/promising_bakeoff_pair_results.csv")
    print("- exports/promising_bakeoff_triple_results.csv")
    print("- exports/promising_bakeoff_all_results.csv")
    print("- exports/promising_bakeoff_cross_validation_by_season.csv")

    print("\nDecision Rule:")
    print("A candidate can only be considered for promotion if it improves")
    print("avg log loss AND avg Brier versus team_only across cross-validation.")
    print("Strict promotion also requires improvement in every tested season.")

    return all_results, all_cv_by_season


def main():
    print("=== Promising Feature Bakeoff Lab ===")
    print("Testing individual 'maybe' features from rejected labs.")
    print("Goal: find small stable signal without importing noisy full feature groups.")

    base = load_base_features()

    enriched, available, missing, source_map = merge_candidate_features(base)

    print(f"\nBase rows: {len(base)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nAvailable candidate features:")

    for feature in available:
        source = source_map.get(feature, "base dataframe")
        print(f"- {feature}  | source: {source}")

    print("\nMissing candidate features:")

    if missing:
        for feature in missing:
            print(f"- {feature}")
    else:
        print("None.")

    if not available:
        raise ValueError(
            "No candidate features found. "
            "Make sure the prior lab parquet exports still exist locally."
        )

    print("\nCandidate feature preview:")
    print(enriched[available].head().to_string(index=False))

    run_bakeoff(enriched, available)


if __name__ == "__main__":
    main()