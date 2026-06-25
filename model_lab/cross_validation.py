import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from features.registry.feature_sets import FEATURE_SETS


def run_cross_validation(df):
    print("")
    print("=== Cross-Validation by Season ===")

    rows = []
    seasons = sorted(df["season"].unique())

    for feature_set_name, features in FEATURE_SETS.items():
        for test_season in seasons:
            train_df = df[df["season"] < test_season].copy()
            test_df = df[df["season"] == test_season].copy()

            if train_df.empty or test_df.empty:
                continue

            train_df = train_df.dropna(subset=features + ["home_win"])
            test_df = test_df.dropna(subset=features + ["home_win"])

            if train_df.empty or test_df.empty:
                continue

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=1000))
            ])

            model.fit(train_df[features], train_df["home_win"])

            probs = model.predict_proba(test_df[features])[:, 1]
            preds = (probs >= 0.50).astype(int)

            row = {
                "feature_set": feature_set_name,
                "test_season": int(test_season),
                "train_seasons": ",".join(str(s) for s in sorted(train_df["season"].unique())),
                "games_tested": len(test_df),
                "accuracy": accuracy_score(test_df["home_win"], preds),
                "log_loss": log_loss(test_df["home_win"], probs),
                "brier_score": brier_score_loss(test_df["home_win"], probs),
            }

            rows.append(row)

            print(
                f"{feature_set_name} | Test {test_season}: "
                f"Accuracy={row['accuracy']:.3f}, "
                f"LogLoss={row['log_loss']:.3f}, "
                f"Brier={row['brier_score']:.3f}"
            )

    results = pd.DataFrame(rows)

    summary = (
        results
        .groupby("feature_set")
        .agg(
            seasons_tested=("test_season", "count"),
            avg_accuracy=("accuracy", "mean"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier_score=("brier_score", "mean"),
            worst_log_loss=("log_loss", "max"),
            best_log_loss=("log_loss", "min"),
        )
        .reset_index()
        .sort_values("avg_log_loss")
    )

    Path("exports").mkdir(exist_ok=True)
    results.to_csv("exports/cross_validation_by_season.csv", index=False)
    summary.to_csv("exports/cross_validation_summary.csv", index=False)

    print("")
    print("=== Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    print("")
    print("Saved:")
    print("- exports/cross_validation_by_season.csv")
    print("- exports/cross_validation_summary.csv")

    return results, summary