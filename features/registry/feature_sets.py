TEAM_FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
]

STARTER_FEATURES = [
    "sp_starts_diff",
    "sp_era_diff",
    "sp_whip_diff",
    "sp_kbb_diff",
    "sp_ip_per_start_diff",
    "sp_rest_diff",
    "sp_recent_era_diff",
]

SCHEDULE_FEATURES = ["home_field"]

FEATURE_SETS = {
    "team_only": TEAM_FEATURES + SCHEDULE_FEATURES,
    "starter_only": STARTER_FEATURES + SCHEDULE_FEATURES,
    "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
}

PRODUCTION_FEATURE_SET = "team_only"

def get_features(feature_set_name=PRODUCTION_FEATURE_SET):
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    return FEATURE_SETS[feature_set_name]
