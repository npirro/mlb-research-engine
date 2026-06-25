# MLB Research Engine v4B — Feature Control + Starter Ablation

This version enforces the thesis:

The numbers decide. Not opinions.

## Key change

Production feature set is now:

team_only

Why?

The v4 backtest showed:

- team_only beat team_plus_starter
- starter features currently add noise
- starter features stay in the lab until they earn their way back

## New output

exports/starter_ablation.csv

This tests:
- baseline team_only
- team_only + each starter feature individually
- team_only + useful starter candidates
- team_only + all starter features

## Run

```powershell
py run_pipeline.py
```

## After running

Send the output from:

=== Starter Feature Ablation ===

and:

=== Holdout Test Results ===

## Commit

```powershell
git add .
git commit -m "Add feature control and starter ablation"
git push
```
