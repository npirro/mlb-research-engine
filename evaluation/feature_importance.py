FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
    "sp_starts_diff",
    "sp_era_diff",
    "sp_whip_diff",
    "sp_kbb_diff",
    "sp_ip_per_start_diff",
    "sp_rest_diff",
    "sp_recent_era_diff",
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
