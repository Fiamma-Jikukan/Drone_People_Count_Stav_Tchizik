# Ablation Results

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Results of the two evaluation runs defined in [ablation_plan.md](ablation_plan.md),
plus a **CLAHE clip/tile grid sweep** (§3). The ablation quantifies the **effect of
settings** (item #6): the CLAHE factor, evaluated against the same point ground truth
with the same metrics, per modality. **Only the setting under test changes between the
runs.** The **confidence-threshold** factor is covered separately by run 3 — see
[`third_run/run3_analysis.md`](../third_run/run3_analysis.md).

---

## 1. The two runs at a glance

| Run | What changed | Config | Output |
|-----|--------------|--------|--------|
| **1 — baseline** | — | CLAHE **on**, thermal conf 0.20 | `first_run/predictions/` |
| **2 — CLAHE off** | thermal preprocessing | CLAHE **off**, thermal conf 0.20 | `evaluation/second_run/` |

Both runs: YOLO11x, SAHI on, confidence RGB 0.25 / thermal 0.20, NMS IoU 0.5. CLAHE is
the **only** factor that differs, so any change is attributable to it.

---

## 2. Run 1 → 2: CLAHE **hurt** thermal

| Metric (thermal) | Run 1 (CLAHE on) | Run 2 (CLAHE off) |
|---|---:|---:|
| Recall | 0.250 | **0.319** |
| F1 | 0.385 | **0.462** |
| MAE | 28.0 | **24.75** |

RGB was **byte-identical** between the two runs (CLAHE only touches thermal), which is
a clean control — the only thing that moved was thermal, and it moved **up**. So CLAHE,
added to *help* thermal by normalising the camera's per-frame auto-gain, was mildly
*hurting* it: piling local contrast enhancement onto the already contrast-processed
`WhiteHot` JPEGs distorts the warm blobs away from what the detector expects.
**CLAHE is off from here on.**

---

## 3. CLAHE clip/tile sweep: no setting beats "off"

The on/off test used only the default CLAHE (clip 2.0, tile 8). To rule out that CLAHE
was merely *badly parameterised* rather than inherently harmful, `clahe_sweep.py` re-runs
thermal detection across the full **clip × tile grid** (one model load, thermal-only,
fixed thermal conf 0.20 / NMS 0.5) and scores each setting against the same ground truth.

Thermal metrics, sorted by F1 (**off** is the reference row):

| setting | Precision | Recall | F1 | MAE |
|---|---:|---:|---:|---:|
| **off** | 0.836 | **0.319** | **0.462** | **24.75** |
| clip 2.0 / tile 16 | 0.852 | 0.287 | 0.430 | 26.50 |
| clip 1.0 / tile 16 | 0.880 | 0.275 | 0.419 | 27.50 |
| clip 4.0 / tile 16 | 0.878 | 0.269 | 0.411 | 27.75 |
| clip 1.0 / tile 8 | 0.843 | 0.269 | 0.408 | 27.25 |
| clip 1.0 / tile 4 | 0.840 | 0.263 | 0.400 | 27.50 |
| clip 2.0 / tile 8 | 0.833 | 0.250 | 0.385 | 28.00 |
| clip 2.0 / tile 4 | 0.867 | 0.244 | 0.380 | 28.75 |
| clip 4.0 / tile 8 | 0.844 | 0.237 | 0.371 | 28.75 |
| clip 4.0 / tile 4 | 0.905 | 0.237 | 0.376 | 29.50 |

- **Off wins outright.** Every one of the nine CLAHE settings has **lower recall and
  worse MAE** than off; the best of them (clip 2.0 / tile 16, F1 0.430) still trails off
  (0.462). So CLAHE was not badly tuned — it is **inherently unhelpful** on this thermal
  data, robust to parameterisation.
- **Consistency checks pass exactly.** `clip 2.0 / tile 8` (the run-1 configuration)
  reproduces run 1's thermal to the digit (F1 0.385, MAE 28.00), and `off` reproduces
  run 2 (F1 0.462, MAE 24.75). The sweep is the same computation as the main pipeline,
  not a separate code path.
- **The pattern confirms the mechanism.** CLAHE consistently trades a little precision
  *up* for a lot of recall *down* — it pushes the already-processed `WhiteHot` blobs
  further from the model's learned "person", so fewer clear the detection floor. Coarser
  tiles (16, closest to global) hurt least; the aggressive clip 4.0 hurts most. None
  recovers what it costs.

Table: `evaluation/clahe_sweep/clahe_sweep.csv`.

---

## 4. What this tells us

- **The gain is real but small.** Turning CLAHE off is a free improvement, but even at
  its better setting thermal (F1 0.462, MAE 24.75) stays far behind RGB
  (F1 0.885, MAE 5.5). Contrast normalisation is not the bottleneck.
- **The bottleneck is domain mismatch.** A COCO-pretrained detector has never seen
  `WhiteHot` thermal, so warm blobs only weakly resemble a "person". That is a
  *model* limitation, which no preprocessing knob can close — the clip/tile sweep (§3)
  makes this concrete: **none of nine CLAHE settings** helps. It needs a
  **thermal-appropriate (fine-tuned) model** (docs 6–7).
- **RGB is robust.** It is a pass-through (CLAHE never applies), and its strong numbers
  hold regardless. The two modalities behave very differently, which is why the pipeline
  keeps **separate per-modality settings** by design.

---

## 5. Caveats

- **Small, single-condition sample** (8 images / 4 thermal, one scene, one dusk
  session): the CLAHE-off advantage is indicative and could shift on new data.
- The measured effect is modest (a few points of thermal F1); it should be read as
  "CLAHE did not help here", not as a large tuning win.
- The best *fix* for thermal remains a **thermal-appropriate model**, not preprocessing
  (docs 6–7). The **confidence threshold** was swept in run 3
  ([`third_run/run3_analysis.md`](../third_run/run3_analysis.md)); **NMS IoU** and **SAHI
  on/off** remain future work — see [ablation_plan.md](ablation_plan.md).
