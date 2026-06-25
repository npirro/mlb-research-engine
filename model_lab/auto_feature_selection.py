import json
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from features.registry.feature_sets import (
    TEAM_FEATURES,
    STARTER_FEATURES,
    SCHEDULE_FEATURES,
)


def evaluate_features(df, features):
    latest = df["season"].max()

    data = df.dropna(subset=features + ["home_win"]).copy()

    train_df = data[data["season"] < latest]
    test_df = data[data["season"] == latest]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000))
    ])

    model.fit(train_df[features], train_df["home_win"])

    probs = model.predict_proba(test_df[features])[:, 1]
    preds = (probs >= 0.50).astype(int)

    return {
        "accuracy": accuracy_score(test_df["home_win"], preds),
        "log_loss": log_loss(test_df["home_win"], probs),
        "brier": brier_score_loss(test_df["home_win"], probs),
    }


def run_auto_feature_selection(df):

    baseline = TEAM_FEATURES + SCHEDULE_FEATURES
    selected = baseline.copy()
    remaining = STARTER_FEATURES.copy()

    print("\n=== Auto Feature Selection ===")

    current = evaluate_features(df, selected)

    print(
        f"Baseline: "
        f"Accuracy={current['accuracy']:.3f} "
        f"LogLoss={current['log_loss']:.4f}"
    )

    while remaining:

        best_feature = None
        best_result = current

        for feature in remaining:

            trial = selected + [feature]
            result = evaluate_features(df, trial)

            if result["log_loss"] < best_result["log_loss"]:
                best_result = result
                best_feature = feature

        if best_feature is None:
            break

        print(f"Accepted: {best_feature}")

        selected.append(best_feature)
        remaining.remove(best_feature)
        current = best_result

    payload = {
        "selected_features": selected,
        "accuracy": current["accuracy"],
        "log_loss": current["log_loss"],
        "brier": current["brier"],
    }

    Path("exports").mkdir(exist_ok=True)

    with open(
        "exports/selected_features.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2)

    print("\n=== Final Feature Set ===")
    print(selected)

    return payload