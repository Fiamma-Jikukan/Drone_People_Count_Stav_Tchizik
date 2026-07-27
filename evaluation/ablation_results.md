# Ablation Results

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

Results of the two evaluation runs defined in [ablation_plan.md](ablation_plan.md).
The ablation quantifies the **effect of settings** (item #6): the CLAHE on/off factor,
evaluated against the same point ground truth with the same metrics, per modality.
**Only the setting under test changes between the runs.**

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

## 3. What this tells us

- **The gain is real but small.** Turning CLAHE off is a free improvement, but even at
  its better setting thermal (F1 0.462, MAE 24.75) stays far behind RGB
  (F1 0.885, MAE 5.5). Contrast normalisation is not the bottleneck.
- **The bottleneck is domain mismatch.** A COCO-pretrained detector has never seen
  `WhiteHot` thermal, so warm blobs only weakly resemble a "person". That is a
  *model* limitation, which no preprocessing knob can close — it needs a
  **thermal-appropriate (fine-tuned) model** (docs 6–7).
- **RGB is robust.** It is a pass-through (CLAHE never applies), and its strong numbers
  hold regardless. The two modalities behave very differently, which is why the pipeline
  keeps **separate per-modality settings** by design.

---

## 4. Caveats

- **Small, single-condition sample** (8 images / 4 thermal, one scene, one dusk
  session): the CLAHE-off advantage is indicative and could shift on new data.
- The measured effect is modest (a few points of thermal F1); it should be read as
  "CLAHE did not help here", not as a large tuning win.
- The best *fix* for thermal remains a **thermal-appropriate model**, not preprocessing
  (docs 6–7). The remaining candidate factors (confidence threshold, NMS IoU, SAHI
  on/off) are left as future work — see [ablation_plan.md](ablation_plan.md) and the
  `confidence_sweep.py` / `clahe_sweep.py` helpers in this folder.
