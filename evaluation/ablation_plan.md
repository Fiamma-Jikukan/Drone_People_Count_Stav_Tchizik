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
outputs in `evaluation/predictions/`.

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
`evaluation/results/summary.json` (CLAHE on), focusing on the **thermal** row.

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
CLAHE was mildly *hurting* thermal, and the confidence sweep below uses **CLAHE off**.

---

## Confidence sweep (third run)

**Why one run is enough.** Unlike CLAHE (a pre-detection step), the confidence
threshold is applied *after* detection. So instead of re-running the model per
threshold, we run it **once at a low floor** and re-apply higher thresholds to the
saved detections. This is **exact**: both SAHI's tile merge and our NMS are greedy by
descending score, so a detection can only be suppressed by a *higher*-scoring one —
filtering the saved (already-merged) detections at any threshold T gives the identical
result to running the whole pipeline at T. One run therefore covers the entire sweep.

**Config.** Best settings so far — CLAHE **off**, SAHI **on**, `yolo11x` — run at a low
confidence floor of **0.10** for both modalities, so the saved JSON contains every
detection down to 0.10.

**Sweep.** Re-threshold the saved detections at **0.10, 0.15, 0.20, 0.25, 0.30, 0.35**
(per modality) and recompute metrics.

**What we measure / expect.** Precision / recall / F1 / MAE vs threshold, per modality.
Raising the threshold should trade recall for precision. The key question: does
**lowering the thermal threshold meaningfully recover recall**, or does thermal stay
low — confirming the bottleneck is the model not firing at all (domain mismatch), not a
threshold that is merely too strict?

### Run command

```bash
python main.py --input evaluation/eval_images --output evaluation/third_run --weights yolo11x.pt --no-clahe --rgb-conf 0.10 --thermal-conf 0.10
```

This writes detections down to confidence **0.10** (CLAHE off) into
`evaluation/third_run/json/`. Note the per-image counts in this run are **inflated** (a
low floor keeps many weak detections) — that is expected; the sweep re-applies the real
thresholds to these saved detections, with no further model runs.

---

## CLAHE clip/tile sweep

The CLAHE on/off test used only the default clip=2.0 / tile=8. This sweep asks whether a
**different CLAHE setting** (gentler clip, coarser/finer grid) would beat "off" — i.e.
whether CLAHE was inherently harmful or just badly parameterised.

**Why a dedicated runner (not one main.py run per setting).** CLAHE is a *preprocessing*
step, so each setting changes the detector input and must be re-run (unlike the
confidence sweep, which re-thresholds cached detections). `evaluation/clahe_sweep.py`
loads the model **once** and loops the grid over the 4 thermal images, so it is far
cheaper than re-launching `main.py` per setting. Thermal-only (RGB is unaffected by CLAHE),
at a **fixed** thermal confidence so CLAHE is the only variable.

**Grid.** clip ∈ {1.0, 2.0, 4.0} × tile ∈ {4, 8, 16}, plus a CLAHE-**off** reference row.
Consistency check: "off" should reproduce the CLAHE-off result and "clip=2.0 tile=8" the
CLAHE-on baseline (both at the same threshold).

`main.py` also gained `--clahe-clip` / `--clahe-tile` flags for ad-hoc single runs.

### Run command

```bash
python evaluation/clahe_sweep.py
```

Writes `evaluation/clahe_sweep/clahe_sweep.csv` (per-setting thermal P / R / F1 / MAE) and
prints the table. Defaults: `yolo11x`, thermal conf 0.20, NMS 0.5, the grid above.

**Result (measured).** Thermal wants a much **lower** threshold: dropping thermal
0.20 → 0.10 lifted recall 0.32 → 0.52, F1 0.46 → 0.61, and **halved MAE (24.75 → 11.75)**,
while RGB was flat with an optimum around 0.20–0.25. Two consistency checks passed
exactly (RGB @ 0.25 = baseline; thermal @ 0.20 = second_run), confirming the
re-thresholding is exact. But thermal recall was **still climbing at the 0.10 edge**, and
0.10 was the model's detection floor — hence the next run.

---

## Lower thermal floor (fourth run)

**Why.** Thermal recall had not plateaued at 0.10, and 0.10 was the model's
**detection floor** (`base_confidence`), so anything weaker was never captured. This run
lowers the floor to **0.03** to see whether more real thermal people are recoverable
below 0.10 — and where precision finally collapses.

**Scope.** **Thermal only** (RGB is settled at ~0.20–0.25), on the 4 thermal images in
`evaluation/eval_images_thermal/`. CLAHE **off**, SAHI **on**, `yolo11x`. A new
`--base-conf` flag sets the model's detection floor; `--thermal-conf 0.03` saves
detections down to 0.03.

**Sweep.** Re-threshold at **0.03, 0.05, 0.075, 0.10, 0.15, 0.20** and find where thermal
F1 / MAE bottom out — the practical thermal operating point.

### Run command

```bash
python main.py --input evaluation/eval_images_thermal --output evaluation/fourth_run --weights yolo11x.pt --no-clahe --base-conf 0.03 --thermal-conf 0.03
```

Then, after the run — compute the metrics, then the threshold sweep:

```bash
# Metrics + FP/FN examples (scored at the 0.03 floor)
python evaluation/evaluate.py --pred-dir evaluation/fourth_run/json --output evaluation/fourth_run/results

# Threshold sweep (the main analysis for this run)
python evaluation/confidence_sweep.py --pred-dir evaluation/fourth_run/json --output evaluation/fourth_run/sweep --thresholds 0.03 0.05 0.075 0.10 0.15 0.20
```
