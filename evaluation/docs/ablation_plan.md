# Ablation Plan

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

The first evaluation run used a single fixed configuration, so it tells us *how good*
the pipeline is but not *why*, or *which knobs matter*. This document defines the
ablation: controlled experiments that change **one factor at a time** from the
baseline and re-measure, to quantify the "effect of settings" required by item #6 and
to answer questions the single run left open — starting with **"did CLAHE help?"**

---

## Method

- **Change one factor at a time** from the baseline; keep everything else fixed.
- Score each run with the **same ground truth** and the **same metrics**
  (`evaluation/metrics.py` / `evaluation/evaluate.py`), **per modality**.
- Compare against the baseline run's numbers (RGB F1 0.885 / MAE 5.5;
  thermal F1 0.385 / MAE 28).

### Baseline (run 1, already done)
YOLO11x, SAHI **on**, CLAHE **on**, confidence RGB 0.25 / thermal 0.20, NMS IoU 0.5 —
outputs in `first_run/predictions/`.

### Planned experiments

| Factor | Baseline | Variation | Question it answers |
|--------|----------|-----------|---------------------|
| **CLAHE (thermal)** | on | **off** *(this run)* | Does contrast normalisation actually help thermal, or is domain mismatch the ceiling? |
| Confidence threshold | 0.25 / 0.20 | lower / higher | Precision ↔ recall trade-off (esp. thermal recall). |
| NMS IoU | 0.5 | 0.4 / 0.6 | Merging vs splitting people in dense clusters. |
| SAHI tiling | on | off (single pass) | How much of the small-object recall comes from tiling. |

---

## This experiment: CLAHE off

**Hypothesis.** CLAHE was applied to stabilise the thermal camera's per-frame
auto-gain (doc 1). But the baseline thermal recall was only **0.25**, which suggests
the real bottleneck is **domain mismatch** (the COCO-pretrained detector has never
seen `WhiteHot` thermal), not contrast. If that is right, **removing CLAHE should
change thermal results only marginally** — and could even help slightly if CLAHE was
amplifying edge/noise artefacts into false structure.

**What changes.** CLAHE only affects **thermal** preprocessing; **RGB is a pass-through
and is unaffected**. So this doubles as a **sanity check**: the RGB metrics in this run
should be **identical** to the baseline, and any difference must come from thermal
alone.

**What we measure.** Thermal **precision / recall / MAE** with CLAHE off vs on
(baseline). A meaningful thermal change would show CLAHE matters; a negligible change
would confirm that fixing thermal needs a **thermal-appropriate model**, not better
contrast (docs 6–7).

**How to compare after the run.**

```bash
# Score the no-CLAHE predictions against the same ground truth
python evaluation/evaluate.py --pred-dir evaluation/second_run/json --output evaluation/second_run/results
```

Then compare `evaluation/second_run/results/summary.json` (CLAHE off) against
`first_run/results/summary.json` (CLAHE on), focusing on the **thermal** row.

---

## Run command

Run the model on all 8 evaluation images **without CLAHE**, writing to `second_run`
(same weights and thresholds as the baseline, so CLAHE is the only changed factor):

```bash
python main.py --input evaluation/eval_images --output evaluation/second_run --weights yolo11x.pt --no-clahe
```

Outputs land in `evaluation/second_run/json/` (for the metrics) and
`evaluation/second_run/annotated/` (visual check). First run downloads no new weights
(`yolo11x.pt` is already cached); expect a few minutes on CPU.

**Result (measured).** Removing CLAHE **improved** thermal (recall 0.25 → 0.32,
F1 0.385 → 0.462, MAE 28 → 24.75) while RGB stayed **identical** (the control) — so
CLAHE was mildly *hurting* thermal. The gain is small: even at its best thermal
setting the pipeline stays far behind RGB, which points to **domain mismatch** (a
COCO-pretrained detector on `WhiteHot` thermal) as the real ceiling, not contrast
normalisation. CLAHE **off** is therefore the recommended thermal configuration.

---

## CLAHE clip/tile sweep (run)

The on/off test used only the default CLAHE (clip 2.0, tile 8). To check whether a
*different* CLAHE setting could beat "off" — i.e. whether CLAHE was inherently harmful
or just badly parameterised — `clahe_sweep.py` re-runs thermal detection across the
full **clip × tile grid**. CLAHE is *pre-detection*, so each setting changes the
detector input and must be re-run; the runner loads the model once and loops the grid
(clip {1, 2, 4} × tile {4, 8, 16}, plus a CLAHE-off reference) over the thermal images,
at a fixed thermal conf 0.20 / NMS 0.5 so CLAHE is the only variable.

```bash
python evaluation/clahe_sweep.py
```

**Result.** **No CLAHE setting beat off** — the full table and consistency checks are
in [ablation_results.md](ablation_results.md) §3. This makes the "CLAHE off" decision
robust to parameterisation, not an artefact of one setting.

---

## Confidence sweep (run — run 3)

The confidence threshold is applied *after* detection, so it can be swept **without
re-running the model** per value: capture one run at a low `--base-conf`, then re-apply
higher thresholds to the saved detections. This is **exact** because both SAHI's tile
merge and our NMS are greedy by descending score, so filtering the saved detections at
any threshold T equals running the whole pipeline at T.

`confidence_sweep.py` was run on a **0.05-floor capture** (CLAHE off), then a
baseline-faithful **operating-point run** at the recommended thresholds (RGB 0.25 /
thermal 0.10) was scored for inspectable per-image results and FP/FN examples.

```bash
python main.py --input evaluation/eval_images --output evaluation/third_run/sweep_run --no-clahe --base-conf 0.05 --rgb-conf 0.05 --thermal-conf 0.05
python evaluation/confidence_sweep.py --pred-dir evaluation/third_run/sweep_run/json --output evaluation/third_run/sweep_run/sweep --thresholds 0.05 0.075 0.10 0.15 0.20 0.25 0.30 0.35
```

**Result.** Thermal is **under-confident, not blind** — lowering thermal to ~0.10 roughly
halves counting error. Full table, charts, operating-point evaluation, and caveats are in
[`third_run/run3_analysis.md`](../third_run/run3_analysis.md).

---

## Further experiments (not run here)

**NMS IoU** and **SAHI on/off** remain as candidate factors. They need model re-runs
(both are applied before the JSON is saved, so neither can be swept post-hoc). `main.py`
carries `--nms-iou`, `--no-sahi`, `--base-conf`, and `--clahe-clip` / `--clahe-tile` for
ad-hoc single runs.
