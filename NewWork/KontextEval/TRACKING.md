# FLUX.1-Kontext-dev Incremental Editing — Phase Tracker

> Fill in **Results** sections as you run each phase. Paste terminal output directly into the code blocks.

| Phase | Name | Status | Script |
|-------|------|--------|--------|
| 1 | Environment & Baseline | ⬜ Pending | `phase1_baseline.py` |
| 2 | Architecture Inspection | ⬜ Pending | `phase2_architecture.py` |
| 3 | Attention Hooking | ⬜ Pending | `phase3_hooking.py` |
| 4 | Attention Cache | ⬜ Pending | `phase4_cache.py` |
| 5 | Injection Prototype | ⬜ Pending | `phase5_injection.py` |
| 6 | Layer Ablation | ⬜ Pending | `phase6_ablation.py` |
| 7 | Incremental Pipeline | ⬜ Pending | `phase7_pipeline.py` |
| 8 | Adaptive α | ⬜ Pending | `phase8_adaptive.py` |
| 9 | Full Evaluation | ⬜ Pending | `phase9_eval.py` |

**Status key:** ⬜ Pending · 🔄 In Progress · ✅ Done · ❌ Failed

---

## Setup

```bash
cd /content/cherry_on_top_exp
pip install -r NewWork/KontextEval/requirements.txt
```

---

## Phase 1 — Environment & Baseline

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase1_baseline.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase1
```

### Checklist
- [ ] Google Colab A100 configured
- [ ] FLUX.1-Kontext-dev loads without error
- [ ] Baseline edits produce plausible output images
- [ ] Deterministic generation verified (same seed → same pixels)

### Results

```
# Paste terminal output here
```

| Edit | Category | Deterministic | Hash (first 8) |
|------|----------|---------------|----------------|
| color_change | color_change | — | — |
| object_addition | object_addition | — | — |
| object_replacement | object_replacement | — | — |
| style_change | style_change | — | — |

**All deterministic:** —

### Notes
_Observations, unexpected behaviour, GPU memory usage_

---

## Phase 2 — Architecture Inspection

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase2_architecture.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase2
```

### Checklist
- [ ] Total parameter count documented
- [ ] Double-stream block count confirmed
- [ ] Single-stream block count confirmed
- [ ] Q/K/V projection shapes recorded
- [ ] Expected K/V tensor shape at 1024×1024 computed
- [ ] `architecture_summary.json` saved

### Results

```
# Paste terminal output here
```

| Property | Value |
|----------|-------|
| Total params (B) | — |
| Double-stream blocks | — |
| Single-stream blocks | — |
| Hidden dim | — |
| Attention heads | — |
| Head dim | — |
| Image tokens / image | — |
| Kontext total img tokens | — |
| Text tokens (T5) | — |
| Joint seq len (double) | — |
| K/V shape (double, example) | — |
| K/V size MB (bfloat16, per block) | — |

### Notes

---

## Phase 3 — Attention Hooking

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase3_hooking.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase3
```

### Checklist
- [ ] CaptureProcessor attaches to all blocks without error
- [ ] Output pixel-identical to baseline (same seed, same prompt)
- [ ] Q, K, V shapes extracted and documented
- [ ] Inference overhead measured

### Results

```
# Paste terminal output here
```

| Metric | Value |
|--------|-------|
| Pixel-identical | — |
| Time without hooks (s) | — |
| Time with hooks (s) | — |
| Overhead (%) | — |
| Total tensors captured | — |
| Total capture size (MB) | — |
| `double_0_K` shape | — |
| `double_0_V` shape | — |
| `single_0_K` shape | — |

### Notes

---

## Phase 4 — Attention Cache Construction

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase4_cache.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase4 \
    --prompt "a modern living room with a sofa"
```

### Checklist
- [ ] K/V captured for all blocks
- [ ] Cache saved to disk (one `.pt` file per tensor)
- [ ] Cache reloaded successfully
- [ ] Numerical equality verified (all keys pass)

### Results

```
# Paste terminal output here
```

| Property | Value |
|----------|-------|
| Tensors saved | — |
| Verification passed | — |
| Total cache size (MB) | — |
| `block_0_K` shape | — |
| `block_0_V` shape | — |

**Sample cache layout:**

| Key | Shape | MB |
|-----|-------|----|
| double_0_K | — | — |
| double_0_V | — | — |
| double_18_K | — | — |
| single_0_K | — | — |

### Notes

---

## Phase 5 — Attention Injection Prototype

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase5_injection.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase5 \
    --base_prompt "a modern living room with a sofa and a coffee table" \
    --edit_prompt "add a yellow bicycle leaning against the wall"
```

### Checklist
- [ ] Experiment A: K-only injection at α = 0.25 / 0.50 / 0.75 / 1.00
- [ ] Experiment B: V-only injection at α = 0.25 / 0.50 / 0.75 / 1.00
- [ ] Experiment C: K+V injection at α = 0.25 / 0.50 / 0.75 / 1.00
- [ ] Comparison grid saved

### Results

```
# Paste terminal output here
```

#### Visual observations

| Experiment | α | Content preserved? | Edit applied? | Artefacts? |
|-----------|---|-------------------|---------------|------------|
| K_only | 0.25 | — | — | — |
| K_only | 0.50 | — | — | — |
| K_only | 0.75 | — | — | — |
| K_only | 1.00 | — | — | — |
| V_only | 0.25 | — | — | — |
| V_only | 0.50 | — | — | — |
| V_only | 0.75 | — | — | — |
| V_only | 1.00 | — | — | — |
| K_and_V | 0.25 | — | — | — |
| K_and_V | 0.50 | — | — | — |
| K_and_V | 0.75 | — | — | — |
| K_and_V | 1.00 | — | — | — |

**Best experiment / α:** —

### Notes

---

## Phase 6 — Layer Ablation

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase6_ablation.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase6 \
    --alpha_k 0.5 --alpha_v 0.5
```

### Checklist
- [ ] Early layers (double 0–6) tested
- [ ] Middle layers (double 7–13) tested
- [ ] Late double layers (14–18) tested
- [ ] Late single-stream layers tested
- [ ] All double blocks tested
- [ ] All blocks tested
- [ ] CSV saved

### Results

```
# Paste terminal output here
```

| Layer Group | Blocks | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DINOv2 ↑ |
|-------------|--------|--------|--------|---------|---------|
| early (0–6) | 7 dbl | — | — | — | — |
| middle (7–13) | 7 dbl | — | — | — | — |
| late_double (14–18) | 5 dbl | — | — | — | — |
| late_single | 38 sgl | — | — | — | — |
| all_double | 19 dbl | — | — | — | — |
| all | 19+38 | — | — | — | — |
| **baseline (no inj)** | — | — | — | — | — |

**Best layer group:** — (highest DINOv2 / lowest LPIPS)

### Hypothesis vs Finding
| Hypothesis | Confirmed? |
|------------|------------|
| Early → layout/global structure | — |
| Middle → object identity | — |
| Late double → fine details | — |
| Late single → texture | — |

### Notes

---

## Phase 7 — Incremental Editing Pipeline

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase7_pipeline.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase7 \
    --alpha_k 0.5 --alpha_v 0.5
```

### Edit Sequence
| Step | Prompt |
|------|--------|
| 0 | A modern living room |
| 1 | Add a bicycle |
| 2 | Add a vase on the table |
| 3 | Replace bicycle with a car |
| 4 | Change car color to red |
| 5 | Remove vase |

### Checklist
- [ ] Baseline run (no injection) completed
- [ ] Method run (K+V injection) completed
- [ ] DINOv2 computed for all steps
- [ ] results.json saved

### Results

```
# Paste terminal output here
```

| Step | Prompt | Baseline DINOv2 | Method DINOv2 | Baseline PSNR | Method PSNR |
|------|--------|----------------|---------------|---------------|-------------|
| 1 | Add a bicycle | — | — | — | — |
| 2 | Add a vase | — | — | — | — |
| 3 | Replace bicycle → car | — | — | — | — |
| 4 | Change car → red | — | — | — | — |
| 5 | Remove vase | — | — | — | — |

**Questions to answer from images:**
- [ ] Does the living room remain consistent across steps?
- [ ] Does the bicycle disappear correctly at step 3?
- [ ] Does the car colour change at step 4?
- [ ] Do errors accumulate noticeably?

### Notes

---

## Phase 8 — Adaptive Attention Preservation

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase8_adaptive.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase8 \
    --fixed_alpha 0.5 \
    --adaptive_base 0.5 \
    --adaptive_method cosine
```

### Checklist
- [ ] Fixed-α baseline completed
- [ ] Cosine-adaptive run completed
- [ ] Per-layer α values logged
- [ ] Final metrics computed

### Results

```
# Paste terminal output here
```

| Method | PSNR ↑ | LPIPS ↓ | DINOv2 ↑ |
|--------|--------|---------|---------|
| Baseline (no inj) | — | — | — |
| Fixed α = 0.5 | — | — | — |
| Adaptive (cosine) | — | — | — |

**Adaptive α per step (mean over layers):**

| Step | Prompt | Mean α |
|------|--------|--------|
| 1 | Add a bicycle | — |
| 2 | Add a vase | — |
| 3 | Replace bicycle → car | — |
| 4 | Change car → red | — |
| 5 | Remove vase | — |

**Adaptive vs Fixed: improvement?** —

### Notes

---

## Phase 9 — Full Evaluation

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase9_eval.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase9
```

### Checklist
- [ ] All 5 methods run on full sequence
- [ ] LPIPS computed for all steps/methods
- [ ] DINOv2 computed for all steps/methods
- [ ] PSNR computed for all steps/methods
- [ ] CLIP direction similarity computed
- [ ] CSV exported
- [ ] Final comparison table produced

### Content Preservation (DINOv2 ↑ vs base image)

| Step | Native | K only | V only | K+V fixed | K+V adaptive |
|------|--------|--------|--------|-----------|--------------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

### Edit Success (CLIP Direction Similarity ↑)

| Step | Native | K only | V only | K+V fixed | K+V adaptive |
|------|--------|--------|--------|-----------|--------------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

### Content Preservation (LPIPS ↓ vs base image)

| Step | Native | K only | V only | K+V fixed | K+V adaptive |
|------|--------|--------|--------|-----------|--------------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

### Summary Findings
| Claim | Evidence | Confirmed? |
|-------|----------|------------|
| K injection preserves content better than native | DINOv2 gap at step 5 | — |
| V injection helps less than K injection | DINOv2 K vs V | — |
| K+V beats K-only | DINOv2 K+V vs K | — |
| Adaptive α outperforms fixed α | DINOv2 adaptive vs fixed | — |
| Edit alignment degrades with injection (trade-off) | CLIP dir. similarity | — |

### Results Terminal Output
```
# Paste final terminal output here
```

### Notes

---

## File Structure

```
NewWork/KontextEval/
├── TRACKING.md             ← this file
├── requirements.txt
├── utils/
│   ├── model_utils.py      — pipeline loading, generate()
│   ├── attention_utils.py  — CaptureProcessor, InjectProcessor, AdaptiveInjectProcessor
│   ├── cache_utils.py      — save/load/verify K/V cache
│   └── metrics.py          — LPIPS, PSNR, DINOv2, CLIP
├── phase1_baseline.py
├── phase2_architecture.py
├── phase3_hooking.py
├── phase4_cache.py
├── phase5_injection.py
├── phase6_ablation.py
├── phase7_pipeline.py
├── phase8_adaptive.py
└── phase9_eval.py
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CaptureProcessor mirrors FluxAttnProcessor2_0 exactly | Guarantees zero-impact hooking (Phase 3 sanity check) |
| Cache stores K/V on CPU | Avoids occupying GPU VRAM between generations |
| Injection formula: `(1-α)K_curr + α K_cache` | α=0 → no injection, α=1 → full replacement; continuous sweep |
| Adaptive α via cosine similarity | High sim. → small α (allow edit); low sim. → large α (preserve) |
| DINOv2 [CLS] for semantic preservation | Captures object identity better than pixel metrics |
| CLIP direction for edit alignment | Standard metric from InstructPix2Pix, measures edit direction |
