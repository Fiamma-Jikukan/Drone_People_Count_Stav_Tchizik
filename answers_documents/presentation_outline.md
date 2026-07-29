# Technical Presentation — Outline (~15 min)

*Person Counting in RGB and Thermal Drone Images — Take-Home Assignment (Stav)*

A slide-by-slide plan for the ~15-minute walkthrough (required deliverable #4). For each
slide: **talking points**, a **time budget**, the **figure/artefact to show**, and the
**one thing to land**. The Q&A-prep appendix maps to the review's "explain / modify /
diagnose" probes.

**Narrative spine:** a reliable RGB baseline → an honest ablation (CLAHE hurts, threshold
recovers thermal) → the real ceiling is domain mismatch, not tuning → a clear, evidence-
backed next step. Every claim is backed by a number and a figure.

---

## Slide 1 — Title & problem (1 min)

- **Task:** detect *people* and **count** them, per image, in aerial **RGB + thermal**
  drone pairs. Location outputs (boxes/points) are only for validation and de-duplication.
- **Not** identity, tracking, or face recognition — explicitly out of scope.
- Framing: an **engineering prototype** — reliable baseline, meaningful validation, clean
  code, clear conclusions.
- **Land:** "counting is the metric that matters; boxes are a means, not the goal."

*Show:* one RGB+thermal pair side by side (the provided example, or `0006_V`/`0006_T`).

---

## Slide 2 — Data challenges (1.5 min)

- **Small people** — a person is ~10–30 px in a 12 MP frame; downscaling to 640 px erases
  them. → drives the SAHI decision.
- **Dense clusters / occlusion** — seated groups overlap → merged boxes, under-count. →
  drives NMS thinking.
- **RGB vs thermal domain gap** — thermal is `WhiteHot`, contrast-processed, per-frame
  auto-gain; a COCO-trained detector has never seen it. → the headline limitation.
- **Resolution / FOV / timing offsets** between the two sensors → pairs aren't pixel-aligned.
- **Land:** "each challenge maps to a specific design decision — resolution→SAHI,
  clusters→NMS, thermal→preprocessing + the domain-mismatch risk we flagged up front."

*Show:* a zoomed crop of tiny/clustered people; the thermal `WhiteHot` frame.

---

## Slide 3 — Model research & selection (2 min)

- Compared two families: **YOLO11 + SAHI** (one-stage CNN) vs **RT-DETR / RF-DETR**
  (detection transformer), across: small-object suitability, RGB+thermal, pretrained
  weights, speed/hardware, fine-tuning, licensing, edge/cloud.
- **Chose YOLO11 + SAHI** — mature pretrained `person` weights, SAHI tiling for small
  aerial objects, CPU-viable, first-class fine-tuning API, tunable NMS.
- **Honest trade-off:** RT-DETR's Apache-2.0 licensing is cleaner; pipeline is written
  **detector-agnostic** so swapping is cheap.
- **Land:** "SAHI is the single biggest small-object lever, and it applies to whichever
  detector we pick."

*Show:* the comparison table from [doc 2](2_model_research_and_selection.md).

---

## Slide 4 — Pipeline architecture (2 min)

- **9 function-based steps**, one module each, orchestrated by `main.py`:
  `load → modality → preprocess → detect (YOLO11+SAHI) → keep people → confidence/NMS →
  count → annotate → save JSON`.
- **One pipeline, both modalities** — only preprocessing + thresholds differ. Modality
  inferred from `_V`/`_T` naming (overridable).
- **Config not hard-coding** — every knob is a CLI flag; `run_config.json` records each run.
- **Output:** annotated image + JSON `{image_name, modality, people_count, detections[...]}`.
- **Land:** "clean separation of responsibilities; reproducible; swappable."

*Show:* the pipeline diagram (README) and a sample JSON record.

---

## Slide 5 — Key parameters & preprocessing (1 min)

- **RGB:** near pass-through (SAHI resolution does the work).
- **Thermal:** grayscale → CLAHE → 3-channel (baseline default).
- **SAHI:** 640-px tiles, 0.2 overlap. **Post:** keep `person` → per-modality confidence
  (RGB 0.25 / thermal 0.20) → NMS IoU 0.5 → count.
- **Land:** "per-modality thresholds are deliberate — the two modalities have different
  optimal operating points (proven later)."

*Show:* the parameters table from [doc 1](1_data_analysis.md) / `parameters_reference.md`.

---

## Slide 6 — Ground truth & validation method (1.5 min)

- **Point annotations** (one point per person) on **8 images** (4 RGB + 4 thermal),
  hand-made with a **custom annotation tool** (zoom/pan, keyboard-driven).
- **Why points, not boxes:** counting needs *presence*, not exact extent; points are faster
  and unambiguous on tiny/occluded people.
- **Matching rule:** a detection is a **TP** if its box contains an unmatched GT point
  (greedy by confidence, one-to-one) → precision / recall / F1; `|pred − GT|` → MAE.
- **Land:** "the evaluation is detection-level *and* count-level, per modality — human GT,
  automated scoring."

*Show:* a rendered annotation overlay; the matching diagram.

---

## Slide 7 — Results: RGB is strong (1.5 min)

- **RGB: F1 0.885, MAE 5.5, precision 0.93** across 25–111-person frames. Small error even
  on the densest (111 → 100).
- Errors are **dense-cluster** members and the smallest/occluded people — under-counting,
  not false alarms.
- **Land:** "RGB is production-adjacent already; no training needed."

*Show:* `counts_per_image.png` + the `0006_V` annotated example (mostly green).

---

## Slide 8 — Results: thermal is the hard case (1.5 min)

- **Thermal (baseline): F1 0.385, MAE 28, recall 0.25.** One frame (`0005_T`) → **zero
  detections**.
- The gap is almost entirely **recall**, not precision — the model *misses* people, doesn't
  invent them.
- **Cause:** domain mismatch — COCO-trained model on `WhiteHot` thermal.
- **Land:** "this isn't a bug — it's the predicted domain-mismatch risk, now measured."

*Show:* the `0005_T` zero-detection overlay + `detection_outcomes.png` (tall thermal FN bar).

---

## Slide 9 — The ablation: what actually moves thermal (2 min)

*This is the "command of the solution" slide — the effect-of-settings analysis.*

- **CLAHE:** swept the full clip × tile grid — **no setting beats "off."** CLAHE mildly
  *hurts* thermal. (Kept on as the conservative baseline; documented the better setting.)
- **Confidence — the big lever:** a low-floor sweep shows thermal is **under-confident, not
  blind** — it fires on **74 %** of thermal people at a low threshold. Dropping thermal
  **0.20 → 0.10** (CLAHE off) roughly **halves counting error (MAE ≈ 24.75 → 11.75)**.
- **RGB wants the opposite** (higher threshold) — the two optima are on opposite ends,
  validating per-modality thresholds.
- **The ceiling is precision:** recovering recall floods in false positives → thermal F1
  caps ~0.67. Tuning makes thermal *usable*, not *good*.
- **Land:** "cheap settings take thermal a long way, but there's a wall only training moves."

*Show:* `mae_vs_threshold.png` + `recall_vs_threshold.png` (note: sweep = *shape*; the
operating-point run gives the faithful 11.75 / 5.5).

---

## Slide 10 — RGB vs thermal comparison (1 min)

- **Best-vs-best:** RGB (0.25) F1 0.885 / MAE 5.5 vs thermal (CLAHE off, 0.10) F1 0.61 /
  MAE 11.75 — **RGB wins on every pair**, even with thermal tuned.
- **When would thermal win?** Darkness / no visible light — *not* this dusk scene — and only
  with a thermal-appropriate detector.
- **Fusion (proposed, not built):** rigid mount → fixed homography → use strong RGB to
  **confirm/seed** thermal detections, recovering thermal FNs.
- **Land:** "the pair's real value is cross-validation; fusion is the natural extension."

*Show:* operating-point `detection_scores.png` (best-vs-best); a paired RGB/thermal frame.

---

## Slide 11 — Limitations (1 min)

- **Small, single-condition sample** — 8 images, one location, one dusk session → results
  indicative, not generalisable.
- **Thermal domain mismatch** is the dominant limitation; threshold tuning is a mitigation,
  not a fix.
- **No fine-tuning** — a *deliberate, evidence-backed* choice (points not boxes, 8 images
  would overfit) — see [doc 8](8_finetuning_future_work.md).
- **Land:** "we know exactly where the prototype is weak and why."

---

## Slide 12 — Recommended next steps (1 min)

- **Fine-tune a thermal detector** (FLIR ADAS / aerial thermal) — the single highest-impact
  step; lifts the whole precision–recall curve that tuning can't.
- **Implement RGB↔thermal fusion** via the fixed homography (cross-validation + FN recovery).
- **Scale + diversify** the evaluation set (lighting, density, altitude); add per-size and
  inference-time metrics.
- **Land:** "clear, prioritised, and justified by the evidence shown."

---

## Slide 13 — Close & demo readiness (0.5 min)

- One-line recap: **reliable RGB baseline, honest thermal analysis, evidence-backed roadmap.**
- Offer a **live demo**: run on a new image, change a threshold and predict the effect,
  open a JSON/annotated output.
- **AI-assistance disclosed** in the README; all ground truth is human-made.

---

## Appendix — Q&A prep (the review will probe these)

**"Modify a threshold — what happens?"**
- Lower thermal 0.20 → 0.10: recall ↑ (more people found), precision ↓ (more FPs), MAE ↓
  (better count). Raise it: the reverse. RGB is flat 0.20–0.30, so barely moves.

**"Why per-modality thresholds?"**
- The sweep: RGB optimum is *high* (recall-saturated), thermal optimum is *low*
  (under-confident). Opposite ends → one shared threshold would hurt one modality.

**"Why did CLAHE stay on if it hurts?"**
- It's the shipped *baseline* we chose to keep; the ablation *documents* that off is better
  (grid-proven) without changing the default. Turning it off is one flag.

**"Why not just lower the thermal threshold to fix thermal?"**
- It halves MAE but precision becomes the ceiling (F1 caps ~0.67). The wall is domain
  mismatch — only a thermal-trained model moves it.

**"Sweep says thermal MAE 7.5 at 0.10, but the run says 11.75 — which is right?"**
- The run (11.75). The sweep re-thresholds a 0.05-floor capture where SAHI switches its
  tile-merge, so read the sweep for *shape*, the operating-point run for *absolute* numbers.
  (RGB @ 0.25 reproduces the baseline byte-for-byte — proof the run is faithful.)

**"Why points, not boxes, for ground truth?"**
- Counting needs presence, not extent; points are unambiguous on tiny/occluded people and
  far faster to annotate 8 images.

**"How does SAHI help / what's the cost?"**
- Tiles the frame so small people are large *within* a tile → big recall gain. Cost: N×
  slower (~17 s/image on CPU). It's the biggest small-object lever.

**"Diagnose: why zero detections on `0005_T`?"**
- Per-frame auto-gain normalised that scene so people barely stand out; the COCO model
  didn't fire. Even at threshold 0.10 it recovers only 3/25 — domain mismatch, not threshold.

**Be ready to:** open `main.py` and trace the 9 steps; show where confidence/NMS is applied;
run on one image; point to a TP/FP/FN overlay and read it.

---

## Timing summary (~15 min)

| Slides | Segment | Time |
|---|---|---:|
| 1–2 | Problem + data challenges | 2.5 min |
| 3 | Model selection | 2 min |
| 4–6 | Implementation + validation method | 4.5 min |
| 7–10 | Results + ablation + comparison | 5 min |
| 11–13 | Limitations, next steps, close | 2.5 min |
| — | Buffer / Q&A | flexible |
