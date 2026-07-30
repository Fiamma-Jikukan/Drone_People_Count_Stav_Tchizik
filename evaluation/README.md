# Evaluation — reviewer's guide

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Everything used to **validate** the pipeline: the scoring code, the input sample, three
evaluation runs, and the written analysis. This page is the map.

## ▶ Start here

- **Headline numbers, all runs + both modalities in one table:**
  [`docs/results_comparison.md`](docs/results_comparison.md)
- **Baseline result + FP/FN example overlays:** [`first_run/results/`](first_run/results/)
- **Full analysis of settings (CLAHE, confidence):** [`docs/ablation_results.md`](docs/ablation_results.md)
  and [`third_run/run3_analysis.md`](third_run/run3_analysis.md)

## Layout

```
evaluation/
├── README.md                  ← this guide
│
├── metrics.py                 ← metric functions (point-in-box matching, P/R/F1, MAE)
├── evaluate.py                ← scores saved predictions → CSV + summary + plots + FP/FN examples
├── plots.py                   ← result charts (shared styling)
├── confidence_sweep.py        ← re-thresholds a low-floor run (no model re-run)
├── clahe_sweep.py             ← re-runs thermal detection over a CLAHE clip×tile grid
│
│                                (the 8 eval images are listed in ground_truth/evaluation_sample.txt,
│                                 not duplicated here — main.py accepts that .txt manifest as --input)
├── clahe_sweep/               ← output of clahe_sweep.py (clahe_sweep.csv)
│
├── docs/                      ← written analysis & references
│   ├── results_comparison.md  ← ★ consolidated comparison table (start here)
│   ├── ablation_results.md    ← CLAHE on/off + clip×tile grid findings
│   └── ablation_plan.md       ← the experiment design
│
├── first_run/    ← Run 1: BASELINE
├── second_run/   ← Run 2: CLAHE off
└── third_run/    ← Run 3: confidence sweep + best operating point
```

## The three runs

Ground truth is the same 8 point-annotated images throughout (RGB 263 people /
thermal 160). Only one factor changes per run.

| Run | Folder | Config vs baseline | What it shows |
|-----|--------|--------------------|---------------|
| **1 — baseline** | `first_run/` | CLAHE **on**, RGB 0.25 / thermal 0.20 | The shipped default result (RGB F1 0.885 / thermal 0.385). |
| **2 — CLAHE off** | `second_run/` | CLAHE **off** | CLAHE mildly *hurts* thermal (F1 0.385 → 0.462); RGB unchanged. |
| **3 — operating point** | `third_run/` | CLAHE off, thermal **0.075** | Lowering the thermal threshold cuts counting error to a quarter (MAE 28 → 7.25). |

Each run folder holds `run_config.json` (the exact parameters), `json/` (per-image
predictions), `annotated/` (annotated images), and `results/` (metrics CSV, summary,
plots, FP/FN example overlays).

### ⚠️ Note on `third_run/`
`third_run/sweep_run/json/` and `third_run/sweep_run/annotated/` are a **low-floor (0.05) capture** used only
as the *source data for the confidence sweep* — their counts are intentionally **inflated**
(many weak detections) and are **not** a meaningful result. The reviewable, baseline-
faithful result at the recommended thresholds lives in
[`third_run/after_sweep_run/results/`](third_run/after_sweep_run/results/); the sweep
charts are in `third_run/sweep_run/sweep/`. Full explanation:
[`third_run/run3_analysis.md`](third_run/run3_analysis.md).

## Reproduce

```bash
# 1. Baseline predictions for the 8 evaluation images
python main.py --input ground_truth/evaluation_sample.txt --output evaluation/first_run/predictions --weights yolo11x.pt

# 2. Score them → metrics, plots, FP/FN examples
python evaluation/evaluate.py --pred-dir evaluation/first_run/predictions/json --output evaluation/first_run/results

# (optional) confidence sweep and CLAHE grid
python evaluation/confidence_sweep.py --pred-dir evaluation/second_run/json --output evaluation/second_run/sweep
python evaluation/clahe_sweep.py
```
