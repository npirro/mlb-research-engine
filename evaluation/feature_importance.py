FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "home_recent_win_pct",
    "away_recent_win_pct",
    "recent_win_pct_diff",
    "home_recent_rpg",
    "away_recent_rpg",
    "recent_rpg_diff",
    "home_field",
]

import pandas as pd
from pathlib import Path

def export_feature_importance(model):
    try:
        logistic = model.named_steps["logistic"]
        coefs = logistic.coef_[0]
        df = pd.DataFrame({
            "feature": FEATURES,
            "coefficient": coefs,
            "abs_importance": abs(coefs),
            "direction": ["Favors Home" if c > 0 else "Favors Away" for c in coefs],
        }).sort_values("abs_importance", ascending=False)

        Path("exports").mkdir(exist_ok=True)
        df.to_csv("exports/feature_importance.csv", index=False)
        print("")
        print("=== Feature Importance ===")
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"Could not export feature importance: {e}")
        return pd.DataFrame()
