# CLIP for Satellite Images - Few-Shot & Prompts (EuroSAT)

**Goal:** A compact, portfolio-ready project showing zero-shot CLIP baselines, few-shot fine-tuning, and prompt optimization on **EuroSAT (RGB)** — plus optional extensions (Domain Transfer, Grad-CAM, LoRA).

- 🔎 **Zero-shot** with OpenCLIP + prompt ensembles
- 🧪 **Few-shot** (linear probe, partial unfreeze)
- 📝 **Prompt ablations** (+ learnable prompt weights)
- 🌍 **Domain transfer** (RESISC45) - *planned*
- 🔍 **Grad-CAM style** interpretability - *planned*
- 🧩 **LoRA** partial-unfreeze alternative - *planned*

---

## TL;DR results (validation)

| Method                            | Shots | Top-1 | Top-5 | Notes |
|----------------------------------|:-----:|:-----:|:-----:|------|
| Zero-shot (best prompt/ensemble) |   0   |  55%  |  85%  | ViT-L/14 example |
| Linear probe (frozen encoder)    |   5   |  18%  |  64%  | (check config if low) |
| Partial unfreeze (last block)    |   5   |  75%  |  98%  | ✅ solid gain |

Artifacts live under `results/`.

---

## Repo structure

```
.
├─ notebooks/
│  ├─ 01_zero_shot_CLIP.ipynb         # zero-shot eval + prompt ensembles
│  ├─ 02_few_shot_learning.ipynb      # linear probe + partial unfreeze
│  ├─ 03_prompt_ablations.ipynb       # prompt variants, synonyms, learnable weights
│  ├─ 04_resisc45_transfer.ipynb      # (planned) domain transfer
│  ├─ 05_grad_cam.ipynb               # (planned) interpretability
│  └─ 06_lora_partial_unfreeze.ipynb  # (planned) PEFT alternative to unfreeze
├─ results/                           # saved metrics/plots
├─ report.md                          # short research-style report (template provided)
├─ README.md
└─ requirements.txt / environment.yml

````

---

## Setup

> Python **3.11** recommended.

```bash
# Option A: pip
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install open_clip_torch torch torchvision datasets pyarrow scikit-learn matplotlib tqdm pillow

# Option B: conda
conda create -n clip311 python=3.11 -y
conda activate clip311
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install open_clip_torch datasets pyarrow scikit-learn matplotlib tqdm pillow
````

GPU is optional but strongly recommended (8–12GB VRAM is plenty for ViT-B/32; ViT-L/14 benefits from 12–16GB).

---

## Quickstart

1. **Week 1 — Zero-shot**

   * Open `notebooks/01_zero_shot_CLIP.ipynb`
   * Model default: `ViT-B-32` (`laion2b_s34b_b79k`)
   * Outputs: accuracy table, confusion matrix, per-class bars

2. **Week 2 — Few-shot**

   * Open `notebooks/02_few_shot_learning.ipynb`
   * Set `K_SHOTS=5` (or 10)
   * Run **Linear Probe** and **Partial Unfreeze** (last ViT block)

3. **Week 3 — Prompts**

   * Open `notebooks/03_prompt_ablations.ipynb`
   * Run single prompts + ensemble; optionally enable **synonyms**

---

## Datasets

* **EuroSAT (RGB)** is loaded from **Hugging Face Datasets**:

  * We stratify the single `train` split into **train/val** (e.g., 80/20).
* **RESISC45** (planned extension) via HF Datasets or `torchvision`.

> The notebooks auto-download to `~/.cache` (HF) or `./data/` (torchvision variant).

---

## Reproduce our splits

We fix a seed and **stratify** by label to avoid class imbalance in val. To reproduce:

* Keep `SEED=42` and `HOLDOUT_FRACTION=0.2` in all notebooks.
* Log configs in the `metadata.json`/`summary.json` saved per run.

---

## Common tweaks

* **Better accuracy quickly**

  * Switch to `ViT-L-14` with `PRETRAINED="laion2b_s32b_b82k"`.
  * Try **K_SHOTS=10**.
  * Use **prompt synonyms** or **learnable prompt weights**.
* **Faster iteration**

  * Start with `ViT-B-32`, smaller batches, CPU ok (slower).

---

## Planned extensions

* **04_resisc45_transfer.ipynb**

  * Evaluate zero-shot & few-shot on **RESISC45**; compare to EuroSAT.
  * Nice “generalization” story for the report.

* **05_grad_cam.ipynb**

  * Visualize attention/importance for successes and failures.
  * Use Grad-CAM-like maps on the ViT image encoder features.

* **06_lora_partial_unfreeze.ipynb**

  * Replace last-block unfreeze with **LoRA** adapters (fewer trainable params).
  * Compare accuracy vs. params vs. runtime.

---

## Results to include in the report

* **Zero-shot vs Few-shot vs Partial Unfreeze** (Top-1/Top-5)
* **Prompt sensitivity** (best single vs ensemble; per-class heatmap)
* **Confusion matrix** (best method)
* (**Optional**) Domain transfer & LoRA comparisons

---

## Tips & gotchas

* Make sure **image/text features are L2-normalized** before dot-product.
* For the **linear probe**, force features to `float32` to avoid dtype issues.
* Keep eval preprocessing **clean** (resize → center-crop → normalize).
* Verify your **val split** is stratified and seed-fixed.

---

## Cite / Acknowledge

This project uses:

* **OpenCLIP** (`open_clip_torch`) for CLIP models/checkpoints
* **EuroSAT** dataset
* **Hugging Face Datasets** tooling

---

## License

MIT. Datasets and pretrained weights follow their original licenses.
