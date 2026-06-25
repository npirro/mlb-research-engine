import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from features.registry.feature_sets import TEAM_FEATURES, STARTER_FEATURES, SCHEDULE_FEATURES


def score_feature_set(df, features, label):
    latest = df["season"].max()
    data = df.dropna(subset=features + ["home_win"]).copy()

    train_df = data[data["season"] < latest].copy()
    test_df = data[data["season"] == latest].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000))
    ])

    model.fit(train_df[features], train_df["home_win"])

    probs = model.predict_proba(test_df[features])[:, 1]
    preds = (probs >= 0.50).astype(int)

    return {
        "test_name": label,
        "features": ", ".join(features),
        "feature_count": len(features),
        "test_season": int(latest),
        "games_tested": len(test_df),
        "accuracy": accuracy_score(test_df["home_win"], preds),
        "log_loss": log_loss(test_df["home_win"], probs),
        "brier_score": brier_score_loss(test_df["home_win"], probs),
    }


def run_starter_ablation(df):
    """
    Tests starter features one-by-one and in selected groups.

    Purpose:
    Starter features do NOT get into production because they sound smart.
    They get in only if they improve holdout log loss/Brier versus team_only.
    """
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    rows = []

    print("")
    print("=== Starter Feature Ablation ===")

    baseline = score_feature_set(df, baseline_features, "baseline_team_only")
    rows.append(baseline)

    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.3f}, "
        f"Brier={baseline['brier_score']:.3f}"
    )

    for starter_feature in STARTER_FEATURES:
        features = baseline_features + [starter_feature]
        row = score_feature_set(df, features, f"add_{starter_feature}")
        rows.append(row)

        delta_log_loss = row["log_loss"] - baseline["log_loss"]
        delta_brier = row["brier_score"] - baseline["brier_score"]

        print(
            f"add_{starter_feature}: "
            f"Accuracy={row['accuracy']:.3f}, "
            f"LogLoss={row['log_loss']:.3f} ({delta_log_loss:+.4f}), "
            f"Brier={row['brier_score']:.3f} ({delta_brier:+.4f})"
        )

    useful_candidates = [
        "sp_starts_diff",
        "sp_ip_per_start_diff",
        "sp_whip_diff",
        "sp_rest_diff",
    ]

    row = score_feature_set(df, baseline_features + useful_candidates, "team_plus_useful_starter_candidates")
    rows.append(row)
    print(
        f"team_plus_useful_starter_candidates: "
        f"Accuracy={row['accuracy']:.3f}, "
        f"LogLoss={row['log_loss']:.3f}, "
        f"Brier={row['brier_score']:.3f}"
    )

    row = score_feature_set(df, baseline_features + STARTER_FEATURES, "team_plus_all_starter_features")
    rows.append(row)
    print(
        f"team_plus_all_starter_features: "
        f"Accuracy={row['accuracy']:.3f}, "
        f"LogLoss={row['log_loss']:.3f}, "
        f"Brier={row['brier_score']:.3f}"
    )

    result = pd.DataFrame(rows).sort_values(["log_loss", "brier_score"])
    Path("exports").mkdir(exist_ok=True)
    result.to_csv("exports/starter_ablation.csv", index=False)
    print("Saved starter ablation to exports/starter_ablation.csv")

    return result
