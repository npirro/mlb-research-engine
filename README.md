# MLB Research Engine v4 — Feature Factory

This version restructures the project around the thesis:

The numbers decide. Not opinions.

## What changed

Feature logic is now modular:

features/
- team/team_features.py
- starter/starter_features.py
- schedule/schedule_features.py
- registry/feature_sets.py

## New output

exports/feature_set_comparison.csv

This compares:
- team_only
- starter_only
- team_plus_starter

## Run

py run_pipeline.py

## Commit after running

git add .
git commit -m "Refactor into feature factory"
git push
