# Candidate v1.1 Model Card

## Status

Research promotion passed.

## Added Features

- `vs_hand_games_scaled_diff`
- `env_temp`
- `opp_adj_offense_diff`

## Cross-Validation Summary

```text
   feature_set  seasons_tested  avg_accuracy  avg_log_loss  avg_brier  worst_log_loss  best_log_loss
     team_only               2      0.558575      0.681618   0.244303        0.682139       0.681098
candidate_v1_1               2      0.561458      0.680840   0.243930        0.682090       0.679590
```

## Verdict

- `improves_avg_log_loss`: `True`
- `improves_avg_brier`: `True`
- `improves_every_season_log_loss`: `True`
- `improves_every_season_brier`: `True`
- `strict_promote`: `True`
