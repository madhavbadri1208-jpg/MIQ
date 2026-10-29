# MIQ-Net: Maximal-Invariant Quotient Network for Brain-Tumour MRI Classification


Everything lives in one script: `MIQ.py`.

> **Research code. Not a medical device. Not validated for clinical use.**

---

## Table of contents

1. [Overview](#1-overview)
2. [Method and theory (T1, T2, T3)](#2-method-and-theory-t1-t2-t3)
3. [Models and controls](#3-models-and-controls)
4. [Installation](#4-installation)
5. [Data](#5-data)
6. [Quick start](#6-quick-start)
7. [Configuration](#7-configuration)
8. [Experimental protocol](#8-experimental-protocol)
9. [Metrics and statistics](#9-metrics-and-statistics)
10. [Outputs](#10-outputs)
11. [Reading the T3 diagnostics](#11-reading-the-t3-diagnostics)
12. [Reproducibility](#12-reproducibility)
13. [Limitations and non-claims](#13-limitations-and-non-claims)
14. [Troubleshooting](#14-troubleshooting)
15. [Citation, licence, disclaimer](#15-citation-licence-disclaimer)

---

## 1. Overview

MRI intensity nuisance acts on the log-image `z = log I` as

```
z  ->  gamma * z + b ,      gamma > 0 ,      b in U
```

where `U` is a K-dimensional smooth subspace that includes the constant (global gain). The class label is assumed invariant to this group, so the classifier is constrained to factor through the quotient space `X / G`.

**Scope of the claims is deliberately narrow, so each can be checked:**

- the front-end is a maximal invariant of the stated group (T1);
- the bias subspace is optimal in a stated sense (T2);
- residual leakage from out-of-span bias is bounded and measured (T3);
- evidence is robustness, calibration, cross-source generalisation and cost, **not** the last 0.2% of clean accuracy, which is near ceiling on public brain-MRI benchmarks.

**What is not new.** Log-domain bias removal (homomorphic filtering, N4), gain/contrast normalisation, spectral-norm Lipschitz heads and consistency regularisation are all established. The contribution is their combination with a provable maximal-invariant quotient, leakage diagnostics, and a protocol with controls that separate the quotient from augmentation, from the consistency loss, from classical homomorphic filtering, and from plain log + normalise (`K = 1`).

---

## 2. Method and theory (T1, T2, T3)

Front-end (`QuotientInput`): grey image `g` -> brain mask `m` (`g > 0.02`) -> `z = log g` on the mask -> project out the span of `U` (ridge-regularised, masked least squares) -> divide by the masked RMS -> `phi`. Output channels fed to the backbone: `[phi, |grad phi|, tanh(phi)]`, all functions of `phi` only.

| ID | Statement | Status |
|----|-----------|--------|
| **T1** Invariance and maximality | `Phi(z) = P_perp z / ||P_perp z||_mask` is invariant to `z -> gamma z + b`. Moreover `Phi(z1) = Phi(z2)` if and only if `z2 = c z1 + b`, `c > 0`, `b in U`. | Exact on a fixed mask, up to the ridge term (`RIDGE = 1e-3`). Verified numerically by `--smoke`. |
| **T2** Optimal subspace (Eckart-Young) | For a bias distribution with covariance `C`, the top-K Karhunen-Loeve (KL) subspace minimises `E||P_perp b||^2`. | Verified by `--smoke` (KL <= DCT < random leakage). |
| **T3** Leakage bound | If the true bias has a component `b_tail` outside `U`, then `||Phi(z') - Phi(z)|| <= 2 ||P_perp b_tail|| / (gamma ||P_perp z||)`. | Inequality verified on a synthetic image by `--smoke`. In the full run, leakage is **measured**, not bounded (see Section 11). |

### T3 in plain terms

T1 is exact only for bias inside `U`. Real bias fields also have components the K modes do not cover (for example higher spatial frequency). T3 states that the resulting change in the quotient `Phi` is controlled: it grows with the size of the uncovered bias component and shrinks with the signal energy `||P_perp z||` and the contrast factor `gamma`. The vectors are compared after normalisation to the same RMS, so the left-hand side is a distance between unit-RMS images.

The bound follows from the standard inequality `|| a/||a|| - b/||b|| || <= 2 ||a - b|| / ||a||`.

**What T3 does and does not give you:**

- It explains *why* performance degrades gracefully under held-out bias (`bias_out`, `radial`) rather than failing outright.
- It is a statement about the front-end `Phi` only. It says nothing about the backbone's response to a change in `Phi`; that is why the code also reports empirical logit-to-`Phi` ratios (`Lhat_*`).
- Measured leakage values are descriptive statistics, not certificates.

### T2 circularity warning

The KL basis is estimated from the same Gaussian-random-field family used for the `bias_in` test. **`bias_in` is therefore an in-family sanity check and is not independent evidence.** Independent evidence is `bias_out` (unseen spatial frequency, grid 12) and `radial` (structurally different parametric gain field). Neither is ever used in training (`P_OUT_TRAIN = 0.0`).

---

## 3. Models and controls

Backbones (timm, `features_only=True`, ImageNet-pretrained by default):

| Key | timm identifier |
|-----|-----------------|
| `EdgeNeXtXXS` | `edgenext_xx_small.in1k` |
| `FastViTT8` | `fastvit_t8.apple_in1k` |
| `MobileNetV3L` | `mobilenetv3_large_100.ra_in1k` |

Parameter counts and latency are **measured** and written to `all_results.csv` (`params_M`, `ms_per_img`, front-end included). Do not quote nominal sizes.

Five model kinds per backbone `b`:

| Model | Front-end | Training | Purpose |
|-------|-----------|----------|---------|
| `BASE_b` | raw (ImageNet normalisation) | ERM | plain baseline |
| `AUG_b` | raw | ERM + nuisance augmentation (p = 0.5) | augmentation control |
| `HOMO_b` | homomorphic filter (log minus masked Gaussian low-pass, sigma = 16) | ERM | classical bias-removal control |
| `CONS_b` | raw | ERM + orbit-consistency loss | **recipe control**: same loss as MIQ, no quotient |
| `MIQ_b` | quotient (`K = 12`, KL basis) | orbit-consistency + spectral-norm head | **proposed** |

.

---

## 4. Installation

```bash
git clone <your-repository-url>
cd <repository-directory>
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision timm scikit-learn numpy pandas scipy pillow matplotlib seaborn
```

| Package | Minimum | Needed for |
|---------|---------|------------|
| `torch` | 2.0 | `nn.utils.parametrizations.spectral_norm`, bf16 autocast |
| `torchvision` | 0.16 | `torchvision.transforms.v2` |
| `timm` | 0.9 | `fastvit_t8.apple_in1k`, `edgenext_xx_small.in1k` |
| `scikit-learn` | 1.0 | `StratifiedGroupKFold` |

A CUDA GPU is strongly recommended (bf16 autocast is enabled automatically when supported). A CPU fallback exists but is impractical for the full run. Pretrained weights are downloaded by timm on first use. **Exact versions of every run are written to `run_config.json`; cite that file, not the minimums above.**

---

## 5. Data

The protocol is **cross-source**: train and validate on one public dataset, test on a different one.

| Role | Source | Folder used | Variable |
|------|--------|-------------|----------|
| Training and validation | Kaggle brain-tumour MRI (4 classes) | `Training/` only; validation is a group-aware split of it | `MIQ_TRAIN` |
| Test | Mendeley Data brain-tumour MRI | entire set; never used for model selection | `MIQ_TEST` |

| Item | Kaggle (train) | Mendeley (test) |
|------|----------------|-----------------|
| Dataset title | _add_ | _add_ |
| Authors / owner | _add_ | _add_ |
| URL / DOI | _add_ | _add_ |
| Version / access date | _add_ | _add_ |
| Licence | _add_ | _add_ |
| Images per class | _add_ | _add_ |

The datasets are **not redistributed** here. Obtain both from their original sources and comply with their licences.

Required `ImageFolder` layout. Class folder names must be identical in both sources:

```
<MIQ_TRAIN>/{glioma,meningioma,notumor,pituitary}/*.{jpg,png}
<MIQ_TEST>/{glioma,meningioma,notumor,pituitary}/*.{jpg,png}
```

`ImageFolder` assigns labels by sorted folder name. If the test folders are named differently (for example `no_tumor`), rename or symlink them, otherwise labels are silently wrong.

Notes on the cross-source design:

- The Kaggle `Test/` folder is **not used** and must not be merged into the test set.
- Test accuracy is not comparable to papers that evaluate on the same-source Kaggle test split.
- Overlap between public brain-MRI collections is possible (shared upstream sources). The built-in audit (`--audit`, `acc_dedup`) measures near-duplicates between train and test. A low duplicate rate supports independence; it does not prove it.
- Class definitions may differ between sources. Report any difference you find.

Optional extra evaluation set (`MIQ_EXT`): an `ImageFolder` whose class names match a subset of the four classes. Only matching classes are scored.

---

## 6. Quick start

Run in this order.

```bash
# 1. Theory + architecture self-test. No data or pretrained weights needed.
python MIQ_main_v2.py --smoke

# 2. Verify the three timm backbones build with features_only.
python MIQ_main_v2.py --check-backbones

# 3. Near-duplicate audit (train vs test) only.
MIQ_TRAIN=/path/Training MIQ_TEST=/path/Test python MIQ_main_v2.py --audit

# 4. Full multi-seed experiment.
MIQ_TRAIN=/path/Training MIQ_TEST=/path/Test MIQ_SAVE=results GPU_ID=0 \
python MIQ_main_v2.py
```

`--smoke` asserts: basis orthonormality; T2 ordering; T1 invariance error `< 2e-2` and at least 10x smaller than a different-content distance; the T3 inequality; ECE of a perfect predictor is 0; forward, backward and Grad-CAM pass for every architecture. **If `--smoke` fails, do not run the full experiment.**

---

## 7. Configuration

All set through environment variables.

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIQ_TRAIN` | cluster-specific path | training `ImageFolder`. **Always override.** |
| `MIQ_TEST` | cluster-specific path | test `ImageFolder`. **Always override.** |
| `MIQ_SAVE` | `MIQ_v2_final` | output directory |
| `MIQ_EXT` | empty | optional extra `ImageFolder` |
| `MIQ_EPOCHS` | `20` | maximum epochs (early stopping, patience 8) |
| `MIQ_SEEDS` | `42,123,456,789,2024` | comma-separated seeds |
| `MIQ_BACKBONES` | all three | comma-separated subset of registry keys |
| `MIQ_PRETRAINED` | `1` | `0` = random initialisation |
| `MIQ_AUDIT` | `1` | `0` = skip the near-duplicate audit |
| `GPU_ID` | `5` | CUDA index; falls back to `cuda:0`, then CPU. **Override.** |

**Resume:** a `(model, seed)` pair is skipped if it appears in `all_results.csv` **and** `preds_seed{seed}.npz` exists. Any change of configuration requires a **new `MIQ_SAVE`**, otherwise old results are silently reused.

---

## 8. Experimental protocol

**Splits**
- Validation split is group-aware: near-duplicate clusters in the training set (union-find on a 16x16 average hash, Hamming <= 4) never straddle train/validation (`StratifiedGroupKFold`, 7 splits, seed-dependent, about 1/7 held out).
- The test set is never used for model selection. Early stopping and checkpointing use validation loss only.
- Test near-duplicates of training images are indexed (`dup_test_idx.npy`) and accuracy is also reported on the de-duplicated test subset (`acc_dedup`).

**Training nuisance** (used by `AUG`, `CONS`, `MIQ`): low-frequency Gaussian-field bias (grids 3-8), scale in [0.3, 1.2], gamma in [0.6, 1.6].

**Held out of training:** high-frequency bias (grid 12), radial bias, gamma outside [0.6, 1.6], Rician noise.

**Identical optimisation for every kind:** AdamW, base LR 3e-4 (backbone x0.1), 2 warm-up epochs then cosine, weight decay 1e-2, focal loss (gamma 2) with label smoothing 0.08, CutMix/Mixup schedule, gradient clipping 2.0, batch 32, input 224x224.

**Consistency branch** (`CONS`, `MIQ`): with probability 0.5 per batch, loss = mean cross-entropy on clean and nuisance views + symmetric KL (`CONS_LAMBDA = 1.0`); otherwise CutMix/Mixup as for the other kinds.

**Robustness conditions** (fixed generator seeds, identical across models):

| Kind | Levels | Relation to training |
|------|--------|----------------------|
| `bias_in` | s = 0.5, 1.0, 1.5, 2.0 (grid 5) | in-family; s <= 1.2 seen |
| `bias_out` | s = 0.5, 1.0, 1.5, 2.0 (grid 12) | **held out** |
| `radial` | s = 0.5, 1.0, 1.5, 2.0 | **held out** |
| `gamma` | 0.5, 0.75, 1.5, 2.0 | 0.75 and 1.5 inside range; 0.5 and 2.0 extrapolated |
| `noise` (Rician) | sigma = 0.02, 0.05 | **held out** |

Headline columns: `acc_bias_in` (s = 1), `acc_bias_out` (s = 1), `acc_radial` (s = 1), `acc_gamma` (gamma = 0.5), `acc_noise` (sigma = 0.05).

---

## 9. Metrics and statistics

**Per run:** accuracy, macro F1 / precision / recall / AUC, Cohen's kappa, ECE (15 bins), per-class precision / recall / F1 / AUC / specificity, bootstrap 95% CIs (2000 resamples), confusion matrix, parameter count, latency (ms/img, batch 32, one warm-up call, front-end included), site-shortcut probe, de-duplicated accuracy, optional external accuracy.

**Site-shortcut probe:** a logistic-regression probe predicts image resolution (site proxy) from penultimate embeddings. `site_excess` = probe balanced accuracy minus a label-only baseline; lower is better.

**Inference**
- Per seed: image-wise McNemar (continuity-corrected chi-square and exact binomial), Fisher-combined across seeds.
- Across seeds: Wilcoxon signed-rank, paired t-test, Cohen's d_z for every model pair and metric.
- Superiority table: proposed model vs every other model, Holm-corrected within each metric; lower-is-better metrics (`ece`, `site_excess`) are sign-flipped.
- Pooled family tests: `MIQ` vs each of `BASE`, `AUG`, `HOMO`, `CONS`, paired over (backbone x seed), with bootstrap CI of the mean difference and Holm correction across families.
- A result is labelled **SIGNIFICANT WIN** only if it is a numerical improvement **and** the Holm-adjusted t-test p-value is below 0.05.

**Statistical caveats (apply when reporting)**
- Per-seed McNemar tests reuse the same test images; Fisher combination assumes independence that does not strictly hold.
- With 5 seeds the exact two-sided Wilcoxon p-value cannot go below 0.0625.
- Pooled pairs sharing a backbone are not fully independent. Pooled tests complement, not replace, per-backbone tests.

---

## 10. Outputs

```
<MIQ_SAVE>/
  run_config.json                    versions, device, AMP dtype, seeds, hyper-parameters
  audit_leakage.csv  dup_test_idx.npy  train_groups.npy
  all_results.csv                    one row per (model, seed)
  summary_mean_std.csv
  mcnemar_seedwise.csv  mcnemar_aggregated.csv
  wilcoxon_ttest_results.csv
  superiority_table.csv
  pooled_family_tests.csv
  robust_all.csv  theorem3_diag_all.csv  site_probe_all.csv
  barplot_<metric>.png  robustness_<backbone>.png
  ablation_delta_<backbone>.csv / .png
  <model>/
    role.txt
    metrics/   history, test_metrics, bootstrap_ci, robust, theorem3_diag*,
               site_probe, external**, confusion matrix (csv + png)
    curves/    training curves
    gradcam/   seed*/<class>/*.png     (15 main models; 30 images per class)
    preds/     preds_seed*.npz         (y_true, y_pred, y_prob)
```

\* only for models whose front-end exposes `phi` (`MIQ_*` and `HOMO_*`). \*\* only if `MIQ_EXT` is set.

---

## 11. Reading the T3 diagnostics

Files: `<model>/metrics/theorem3_diag_seed<seed>.csv` and the aggregate `theorem3_diag_all.csv`. One row per perturbation condition (`kind`, `level`).

| Column | Definition |
|--------|------------|
| `mean_leak_d` | mean over test images of `||phi' - phi|| / sqrt(n_mask)`: RMS per-pixel change of the quotient under the perturbation. Smaller = more invariant. |
| `mean_dlogit` | mean `||logits' - logits||` over test images. |
| `Lhat_p50`, `Lhat_p95`, `Lhat_max` | per-image ratio `||dlogit|| / ||dphi||` (images with `||dphi|| > 1e-6`): empirical sensitivity of the network to a change in the quotient. **Descriptive, not a Lipschitz bound.** |
| `spearman_leakproxy_vs_dlogit` | Spearman correlation between an out-of-span energy proxy and `||dlogit||`. The proxy is the fraction of `phi` energy in the next K basis modes (`U2`). **Quotient front-end only.** |

Expected behaviour of the files:

- Written only for `MIQ_*` and `HOMO_*`. `BASE`, `AUG`, `CONS` and `no_inq` have no `phi` and produce no T3 file. This is correct.
- No row for the `clean` condition (no perturbation).
- For `HOMO_*` the Spearman column is `NaN` (no K-mode subspace, hence no proxy).
- If a run finished but a file is absent, the robustness step did not complete; re-run that `(model, seed)` after deleting its entry in `all_results.csv`.

How to interpret:

- In-span nuisance (`gamma`, in-family `bias_in`) should give `mean_leak_d` near zero for `MIQ`. Values clearly above zero indicate leakage through the mask, ridge term, or interpolation.
- Held-out nuisance (`bias_out`, `radial`) should give larger `mean_leak_d`, increasing with `level`, in line with T3. A positive Spearman value indicates that residual out-of-span energy predicts how much the logits move.
- `Lhat` shows how much the backbone amplifies residual leakage. Large `Lhat_p95` with small `mean_leak_d` means the front-end is fine but the backbone is sensitive.
- The T3 *inequality* itself is checked only by `--smoke` on a synthetic image. The full run reports measured leakage; it does not evaluate the bound per test image.

---

## 12. Reproducibility

- All hyper-parameters, seeds and library versions are saved in `run_config.json`.
- `cudnn.deterministic = True`, `cudnn.benchmark = False` and per-run seeding are set. **Bitwise reproducibility is not guaranteed**: bf16 autocast and some CUDA kernels (for example interpolation backward) are non-deterministic. Expect small run-to-run variation; report mean and standard deviation over seeds.
- Latency is hardware-specific; report the device and batch size.

Checklist for any report that uses this code:

- [ ] `--smoke` passes
- [ ] `run_config.json` archived with results
- [ ] Both dataset sources cited with DOI/URL, version and licence
- [ ] Class-name mapping between train and test sources verified
- [ ] Kaggle `Test/` folder confirmed unused
- [ ] Audit result (`audit_leakage.csv`) and `acc_dedup` reported
- [ ] Seeds and their number reported (default 5)
- [ ] `bias_in` described as in-family; `bias_out`, `radial`, `noise` as held-out
- [ ] Ablation backbone stated (default: first only)
- [ ] Holm correction and test choice stated
- [ ] Hardware and latency conditions stated
- [ ] Section 13 limitations reproduced

---

## 13. Limitations and non-claims

1. **Near-ceiling benchmarks.** Small differences in clean accuracy are not evidence of superiority.
2. **Scope of invariance.** Only the modelled group is handled. Additive/Rician noise and out-of-span bias are not removed; the sweeps expose this.
3. **Mask.** The brain mask is a threshold (`0.02`) on grey intensity computed after the nuisance, so it is only approximately invariant under strong nuisance. T1 holds for a fixed mask.
4. **Lipschitz control is head-only.** Spectral norm times a learnable scale (`head_L`) applies to the linear classifier. The backbone is unconstrained and **no end-to-end Lipschitz guarantee is claimed**.
5. **Duplicate detection is heuristic** (average hash, Hamming <= 4) and no patient identifiers are available. Leakage cannot be ruled out, only reduced and measured.
6. **Site proxy.** Image resolution is a proxy for acquisition site, not a site label.
7. **Single cross-source pair.** Results cover one Kaggle-to-Mendeley evaluation unless `MIQ_EXT` is used. They do not establish performance on other scanners, sites or populations.
8. **Statistics.** See the caveats in Section 9.
9. **Not clinical evidence.** No prospective, multi-centre or radiologist-reader validation is provided.

---

## 14. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Cannot build '<name>'` | timm version lacks the model or weight tag. Run `--check-backbones`, upgrade timm. |
| Pretrained download fails | No internet. Pre-cache weights, or set `MIQ_PRETRAINED=0` (changes results). |
| `group split unavailable` printed | Group clustering failed; the code fell back to a plain stratified split, which **weakens the protocol**. Fix and re-run before reporting. |
| Results unchanged after a config change | Resume logic reused old files. Use a new `MIQ_SAVE`. |
| Labels look scrambled on test | Test class-folder names differ from training; fix names (Section 5). |
| CUDA out of memory | Lower `BATCH` / `EVAL_BATCH` in the configuration block. This changes the recipe; re-run all models. |
| Wrong GPU | Set `GPU_ID`. |
| No T3 file for a model | Expected for models without a `phi` front-end (Section 11). |

---

## 15. Citation, licence, disclaimer

**Citation.** To be added upon publication.

**Licence.** To be added. Third-party datasets and pretrained weights are governed by their own licences.

**Disclaimer.** This software is provided for research purposes only, without warranty. It is not a medical device and must not be used for diagnosis or treatment decisions.
