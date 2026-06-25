# MLB Research Engine v3 — Starting Pitchers

This version adds the first real baseball intelligence module: actual historical starting pitchers.

## What changed

The pipeline now:

- downloads MLB historical regular-season games
- pulls boxscores for each game
- identifies the actual starting pitchers
- tracks pitcher stats chronologically
- creates no-lookahead starter features before each game

## New starter features

- `sp_era_diff`
- `sp_whip_diff`
- `sp_kbb_diff`
- `sp_ip_per_start_diff`
- `sp_rest_diff`
- `sp_recent_era_diff`
- `sp_starts_diff`

Positive diff values generally favor the home team.

## Run

```powershell
py run_pipeline.py
```

The first run will take longer than v2A because it downloads boxscores for every historical game.

## Benchmark to beat from v2A

- Accuracy: 0.554
- Log Loss: 0.681
- Brier Score: 0.244

## After running

Review:

- `exports/feature_importance.csv`
- `exports/model_comparison.csv`
- `exports/calibration_buckets.csv`

Then commit:

```powershell
git add .
git commit -m "Add starting pitcher module"
git push
```
