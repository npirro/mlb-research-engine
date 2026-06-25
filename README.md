# MLB Research Engine v2A

Adds:
- recent form features
- run differential features
- feature importance report
- model comparison report

Run:

```powershell
py run_pipeline.py
```

If packages are missing:

```powershell
py -m pip install -r requirements.txt
```

Review:
- exports/feature_importance.csv
- exports/model_comparison.csv
- exports/calibration_buckets.csv

Benchmark from v1:
- Accuracy: 0.552
- Log Loss: 0.680
- Brier Score: 0.244
