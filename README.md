# MIQ-Net: Maximal-Invariant Quotient Network for Brain-Tumour MRI Classification

Reference implementation and full evaluation protocol (`MIQ_main_v2.py`) for lightweight brain-tumour MRI classification (glioma / meningioma / no tumour / pituitary) with an input front-end that is invariant to a defined MRI intensity-nuisance group.

> **Status:** research code accompanying a manuscript. All claims below are limited to what the code measures. Anything not measured is listed under [Non-claims](#3-non-claims-and-known-limitations).

---

## 1. What is claimed

MRI intensity nuisance is modelled on the log-image `z = log I` as

```
z  ->  gamma * z + b ,   gamma > 0 ,   b in U
```

where `U` is a K-dimensional smooth subspace that includes the constant (global gain). The class label is assumed invariant to this group. The classifier is forced to factor through the quotient `X/G` by a fixed front-end.

| ID | Statement | Status in code |
|----|-----------|----------------|
| **T1** | `Phi(z) = P_perp z / ||P_perp z||_mask` is invariant to `z -> gamma z + b`, and `Phi(z1) = Phi(z2)` iff `z2 = c z1 + b`, `c > 0`, `b in U`. | Exact on the masked domain up to the ridge term (`RIDGE = 1e-3`). Checked numerically in `--smoke`. |
| **T2** | The top-K Karhunen-Loeve (KL) subspace minimises `E||P_perp b||^2` for a bias distribution with covariance `C` (Eckart-Young). | Checked in `--smoke` (KL <= DCT < random leakage). |
| **T3** | `||Phi(z') - Phi(z)|| <= 2||P_perp b_tail|| / (gamma ||P_perp z||)`. | Bound checked in `--smoke`; leakage and empirical logit/feature ratios are **measured** per condition (`theorem3_diag_*.csv`). It is a diagnostic, not a certificate. |

**Circularity warning (T2).** The KL basis is estimated from the same Gaussian-random-field (GRF) family used for the `bias_in` test. `bias_in` is therefore an **in-family sanity check, not independent evidence**. Independent evidence is `bias_out` (unseen spatial frequency, grid 12) and `radial` (structurally different parametric gain field). Neither is ever used in training (`P_OUT_TRAIN = 0.0`).

**What the study contributes.** Log-domain bias removal (homomorphic filtering, N4), gain/contrast normalisation, spectral-norm heads and consistency regularisation are all known. The contribution is the *combination* with a provable maximal-invariant quotient, leakage diagnostics, and a protocol whose controls separate "quotient" from "augmentation", "consistency loss", "classical homomorphic filtering" and "plain log + normalise (K = 1)".

---

## 2. Models and controls

Three backbones (timm, `features_only=True`, ImageNet-pretrained by default):

| Key | timm name | Nominal size* |
|-----|-----------|---------------|
| `EdgeNeXtXXS` | `edgenext_xx_small.in1k` | ~1.3 M |
| `FastViTT8` | `fastvit_t8.apple_in1k` | ~3.6 M |
| `MobileNetV3L` | `mobilenetv3_large_100.ra_in1k` | ~4.2 M |

\*Nominal figures only. **Reported sizes are the measured `params_M` in `all_results.csv`** (feature-extractor + head + front-end buffers excluded/included as counted by `model.parameters()`).

Per backbone `b`, five model kinds:

| Model | Front-end | Training recipe | Role |
|-------|-----------|-----------------|------|
| `BASE_b` | raw (ImageNet norm.) | ERM | plain baseline |
| `AUG_b` | raw | ERM + nuisance augmentation (p = 0.5) | augmentation control |
| `HOMO_b` | homomorphic filter (log - masked Gaussian low-pass, sigma = 16) | ERM | classical bias-removal control |
| `CONS_b` | raw | ERM + orbit-consistency loss | **recipe control** (same loss as MIQ, no quotient) |
| `MIQ_b` | quotient (K = 12, KL basis) | orbit-consistency + spectral-norm head | **proposed** |

Leave-one-out / sensitivity ablations of `MIQ_b` (run on the first backbone only unless `MIQ_ABL_ALL=1`):
`no_inq`, `no_gamma`, `no_mask`, `dct_basis`, `rand_basis`, `k1`, `k6`, `k24`, `no_cons`, `no_lip`.

Default run size: `(5 kinds x 3 backbones) + 10 ablations = 25 models`, x 5 seeds = **125 training runs**.

---

## 3. Non-claims and known limitations

Read these before citing any number produced by this code.

1. **Public brain-MRI test accuracy is near ceiling.** Clean-accuracy differences of a few tenths of a percent are not the evidence. The headline evidence is robustness under shift, calibration (ECE), de-duplicated accuracy and cost.
2. **Invariance is to the modelled group only.** Additive/Rician noise and out-of-span bias are **not** removed; the sweeps expose this. Stated limits, not bugs.
3. **The brain mask is threshold-based** (`MASK_THR = 0.02` on grey intensity) and is computed *after* the nuisance, so it is only approximately invariant under strong nuisance. T1 holds on a fixed mask.
4. **Lipschitz control is on the linear head only** (spectral norm x a learnable scale `exp(logit_scale)`, logged as `head_L`). The backbone is **not** Lipschitz-constrained. No end-to-end Lipschitz guarantee is claimed. `Lhat_*` in the T3 diagnostics are empirical ratios, not bounds.
5. **Near-duplicate detection is heuristic** (16x16 average hash, Hamming <= 4). It reduces, but does not prove absence of, patient/slice leakage. The dataset provides no patient IDs.
6. **Site-shortcut probe uses image resolution as a site proxy.** It is a proxy, not a scanner/site label.
7. **Statistical caveats.**
   - Per-seed McNemar tests reuse the same test images; combining seeds with Fisher's method assumes independence that does not strictly hold.
   - With n = 5 seeds the exact two-sided Wilcoxon p-value cannot go below 0.0625. Paired t-test, Cohen's d_z, Fisher-McNemar and the pooled tests carry the inference.
   - Pooled family tests pair over (backbone x seed); pairs sharing a backbone are not fully independent. They complement, not replace, per-backbone tests.
   - The verdict "SIGNIFICANT WIN" requires numerical improvement **and** Holm-adjusted t-test p < 0.05.
8. **Bitwise reproducibility is not guaranteed.** `cudnn.deterministic=True` and fixed seeds are set, but bf16 autocast and some CUDA kernels (e.g. interpolation backward) are non-deterministic. Expect small run-to-run variation.
9. **Generalisation beyond this dataset is untested** unless an external set is supplied via `MIQ_EXT`.

---

## 4. Environment

Tested interface: Python 3, PyTorch with CUDA (bf16 autocast used automatically if supported; CPU fallback exists but is impractically slow for the full run).

Required packages (minimum features used):

| Package | Why it matters |
|---------|----------------|
| `torch` >= 2.0 | `nn.utils.parametrizations.spectral_norm`, bf16 autocast |
| `torchvision` >= 0.16 | `torchvision.transforms.v2` |
| `timm` >= 0.9 | `fastvit_t8.apple_in1k`, `edgenext_xx_small.in1k` |
| `scikit-learn` >= 1.0 | `StratifiedGroupKFold` |
| `numpy`, `pandas`, `scipy`, `Pillow`, `matplotlib`, `seaborn` | metrics, statistics, plots |

```bash
pip install torch torchvision timm scikit-learn numpy pandas scipy pillow matplotlib seaborn
```

**Record your exact versions.** Every run writes `run_config.json` with Python / torch / timm versions, device, AMP dtype, seeds and all key hyper-parameters. Cite that file in the paper, not this README.

Pretrained weights are downloaded by timm on first use (internet required unless cached). Set `MIQ_PRETRAINED=0` to train from scratch.

---

## 5. Data

Expected `ImageFolder` layout (4 class folders per split):

```
<MIQ_TRAIN>/{glioma,meningioma,notumor,pituitary}/*.{jpg,png}
<MIQ_TEST>/{glioma,meningioma,notumor,pituitary}/*.{jpg,png}
```

Dataset is **not** redistributed here. Obtain it from its original source and comply with its licence and terms of use. State the source, version and licence in the manuscript.

Optional external set (`MIQ_EXT`): an `ImageFolder` whose class folder names match a subset of the four class names. Only matching classes are scored; unmatched folders are ignored.

**The default paths and GPU index in the script point to the author's cluster. You must override them (below).**

---

## 6. Usage

Run in this order.

```bash
# 1. Theory + architecture self-test. No data, no pretrained weights needed.
python MIQ_main_v2.py --smoke

# 2. Verify timm names and features_only for the 3 backbones.
python MIQ_main_v2.py --check-backbones

# 3. Near-duplicate audit (train vs test) only.
MIQ_TRAIN=/path/Training MIQ_TEST=/path/Test python MIQ_main_v2.py --audit

# 4. Full multi-seed run.
MIQ_TRAIN=/path/Training MIQ_TEST=/path/Test MIQ_SAVE=MIQ_v2_final GPU_ID=0 \
python MIQ_main_v2.py
```

`--smoke` asserts: basis orthonormality; T2 ordering (KL <= DCT < random); T1 invariance error < 2e-2 and >10x smaller than a different-content distance; T3 bound; ECE of a perfect predictor = 0; forward/backward/Grad-CAM for every architecture. **If `--smoke` fails, do not run the full experiment.**

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIQ_TRAIN` | author's cluster path | training `ImageFolder` |
| `MIQ_TEST` | author's cluster path | test `ImageFolder` |
| `MIQ_SAVE` | `MIQ_v2_final` | output directory |
| `MIQ_EXT` | empty | optional external `ImageFolder` |
| `MIQ_EPOCHS` | `20` | max epochs (early stopping, patience 8) |
| `MIQ_SEEDS` | `42,123,456,789,2024` | comma-separated seeds |
| `MIQ_BACKBONES` | all three | comma-separated subset of registry keys |
| `MIQ_PRETRAINED` | `1` | `0` = random init |
| `MIQ_ABL_ALL` | `0` | `1` = ablations on all backbones (45 models) |
| `MIQ_AUDIT` | `1` | `0` = skip the leakage audit |
| `GPU_ID` | `5` | CUDA device index (falls back to `cuda:0` / CPU) |

**Resume:** a `(model, seed)` pair is skipped if it is in `all_results.csv` **and** its `preds_seed{seed}.npz` exists. Delete both to force a re-run. Changing hyper-parameters without changing `MIQ_SAVE` will silently reuse old results; use a new `MIQ_SAVE` for every configuration.

---

## 7. Protocol (what makes the comparison fair)

- **Validation split is group-aware.** Near-duplicate clusters in the training set (union-find on average-hash) never straddle train/validation (`StratifiedGroupKFold`, 7 splits, seed-dependent; about 1/7 held out).
- **Test set is never used for model selection.** Early stopping and checkpoint choice use validation loss only.
- **Test audit.** Near-duplicates of training images in the test set are indexed (`dup_test_idx.npy`); accuracy is also reported on the de-duplicated test subset (`acc_dedup`).
- **Training nuisance** is low-frequency GRF bias (grids 3-8) with scale in [0.3, 1.2] and gamma in [0.6, 1.6] only.
- **Held out from training:** high-frequency bias (grid 12), radial bias, gamma outside [0.6, 1.6], Rician noise.
- **Same optimisation for all kinds:** AdamW, base LR 3e-4 (backbone x0.1), 2 warm-up epochs then cosine, weight decay 1e-2, focal loss (gamma 2) with label smoothing 0.08, CutMix/Mixup schedule, gradient clipping 2.0, batch 32, image size 224.
- **Consistency branch** (CONS, MIQ): with probability 0.5 per batch, loss = mean CE on clean and nuisance views + symmetric KL (`CONS_LAMBDA = 1.0`); otherwise CutMix/Mixup as for other models.

### Robustness conditions (fixed generator seeds)

| Kind | Levels | Training exposure |
|------|--------|-------------------|
| `bias_in` | s = 0.5, 1.0, 1.5, 2.0 (grid 5) | in-family (s up to 1.2 trained) |
| `bias_out` | s = 0.5 ... 2.0 (grid 12) | **held out** |
| `radial` | s = 0.5 ... 2.0 | **held out** |
| `gamma` | 0.5, 0.75, 1.5, 2.0 | 0.75 and 1.5 inside range; 0.5 and 2.0 extrapolated |
| `noise` (Rician) | sigma = 0.02, 0.05 | **held out** |

Headline columns: `acc_bias_in` (s = 1), `acc_bias_out` (s = 1), `acc_radial` (s = 1), `acc_gamma` (gamma = 0.5), `acc_noise` (sigma = 0.05).

---

## 8. Metrics and statistics

- **Per run:** accuracy, macro F1 / precision / recall / AUC, Cohen's kappa, ECE (15 bins), per-class precision / recall / F1 / AUC / specificity, bootstrap 95% CIs (2000 resamples, fixed RNG), confusion matrix, params (M), latency (ms/img, front-end included, batch 32, after one warm-up call).
- **Pairwise (all model pairs):** image-wise McNemar per seed (chi-square with continuity correction and exact binomial), Fisher-combined across seeds; Wilcoxon, paired t-test and Cohen's d_z across seeds.
- **Superiority table** (`MIQ_<first backbone>` vs every other model, per metric): Holm correction across comparisons within each metric; lower-is-better metrics (`ece`, `site_excess`) are sign-flipped.
- **Pooled family tests:** `MIQ` vs each of `BASE`, `AUG`, `HOMO`, `CONS`, paired over (backbone x seed), with bootstrap CI of the mean difference and Holm correction across families.
- **Latency** is hardware-specific; report the device from `run_config.json`.

---

## 9. Output layout

```
<MIQ_SAVE>/
  run_config.json                  versions, device, all key hyper-parameters
  audit_leakage.csv, dup_test_idx.npy, train_groups.npy
  all_results.csv                  one row per (model, seed)
  summary_mean_std.csv
  mcnemar_seedwise.csv, mcnemar_aggregated.csv
  wilcoxon_ttest_results.csv
  superiority_table.csv
  pooled_family_tests.csv
  robust_all.csv, theorem3_diag_all.csv, site_probe_all.csv
  barplot_<metric>.png, robustness_<backbone>.png
  ablation_delta_<backbone>.csv / .png
  <model>/
    role.txt
    metrics/   history, test_metrics, bootstrap_ci, robust, theorem3_diag (quotient / homomorphic
               models only), site_probe, external (if MIQ_EXT), confusion matrix (csv + png)
    curves/    training curves
    gradcam/   seed*/<class>/*.png   (15 main models only; 30 images per class)
    preds/     preds_seed*.npz       (y_true, y_pred, y_prob)
```

---

## 10. Reproducibility checklist (for reviewers)

- [ ] `python MIQ_main_v2.py --smoke` passes
- [ ] `run_config.json` archived with results
- [ ] Dataset source, version and licence stated
- [ ] Audit result (`audit_leakage.csv`) reported, including `acc_dedup`
- [ ] Number of seeds and their values reported (default 5)
- [ ] `bias_in` reported as in-family only; `bias_out` / `radial` / `noise` reported as held-out
- [ ] Ablation backbone stated (default: first backbone only)
- [ ] Multiple-comparison correction stated (Holm)
- [ ] Hardware and latency measurement conditions stated
- [ ] Limitations in Section 3 reproduced in the manuscript

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Cannot build '<name>'` | timm version lacks the model or weights tag; run `--check-backbones`, upgrade timm |
| Pretrained download fails | no internet; pre-cache weights or set `MIQ_PRETRAINED=0` (changes results) |
| `group split unavailable` printed | `train_groups()` failed; falls back to a plain stratified split. **This weakens the protocol; fix before reporting** |
| Results identical after changing a setting | resume logic reused old files; use a fresh `MIQ_SAVE` |
| CUDA OOM | lower `BATCH` / `EVAL_BATCH` in the config block (changes the recipe; re-run all models) |
| Wrong GPU used | set `GPU_ID` (default is 5) |

---

## 12. Citation and licence

> TODO before release: add the manuscript citation (BibTeX), the code licence, and the dataset licence/attribution. These are intentionally not pre-filled.

## 13. Contact

> TODO: corresponding author and issue tracker.
