MIQ-Net: Maximal-Invariant Quotient Network

Brain-tumour MRI classification (glioma | meningioma | notumor | pituitary) with lightweight backbones, built around an input front-end that makes the classifier invariant to a clearly defined family of MRI intensity nuisances.

Everything lives in a single script, MIQ_main_v2.py. It contains the theory core (the quotient front-end), five model families plus ten ablations, a hardened evaluation protocol, robustness and leakage diagnostics, statistical testing and plotting.

Table of contents
Overview and claims
Method
Models and ablations
Requirements and installation
Dataset layout
Usage
Configuration reference
Training protocol
Evaluation protocol
Statistics
Output files
Smoke test
Reproducibility and resuming
Practical notes and known caveats
Troubleshooting
Code map
1. Overview and claims

MRI intensity nuisance is modelled as acting on the log-image z = log I:

z  ->  gamma * z + b,        gamma > 0,   b in U

where U is a K-dimensional smooth subspace that includes the constant (global gain). The tumour label does not change under this group, so the classifier can be forced to factor through the quotient X / G.

The claims are intentionally narrow so each one is checkable:

ID	Statement	How it is checked
T1	`Phi(z) = P_perp z /	
T2 (Eckart-Young)	For a bias distribution with covariance C, the top-K Karhunen-Loeve (KL) subspace minimises `E	
T3 (leakage bound)	`	
What is not new

Log-domain bias removal (homomorphic filtering, N4), gain/contrast normalisation, spectral-norm Lipschitz heads and consistency regularisation are all known. The contribution is their combination with a provable maximal-invariant quotient, the leakage diagnostics, and a protocol with controls that separate:

"quotient" from "augmentation" (AUG)
"quotient" from "consistency loss" (CONS)
"quotient" from "classical homomorphic filtering" (HOMO)
"quotient" from "plain log + normalise" (k1 ablation)
Honest limits
Additive / Rician noise and out-of-span bias are not removed exactly. The robustness sweeps expose this.
The KL basis is computed from the same Gaussian-random-field family used for the bias_in test, so bias_in is an in-family sanity check, not independent evidence. Independent evidence is bias_out (unseen spatial frequency) and radial (structurally different parametric gain field); neither is ever used in training.
Public brain-MRI test accuracy is near ceiling. The headline evidence is robustness, shift behaviour, calibration and cost, backed by significance tests, not the last 0.2% of clean accuracy.
2. Method
2.1 Quotient front-end (QuotientInput)

For an RGB input x in [0, 1]:

g = mean(x, channel) (grayscale).
Brain mask m = (g > MASK_THR) with MASK_THR = 0.02.
z = log(clamp(g, 1e-3)) * m.
Masked ridge projection onto the nuisance subspace: zp = (z - c U) * m, where c solves (U_m U^T + ridge * I) c = U_m z and U_m = U * m. The ridge is 1e-3.
Scale normalisation: phi = zp / RMS_mask(zp).
Output three channels, all functions of phi only: [phi, |grad phi| (Sobel), tanh(phi)].

The basis U has K = 12 orthonormal rows. Row 0 is the constant (gain). The other rows are, depending on basis:

kl (default): top KL modes of the low-frequency Gaussian-field bias family (SVD of 500 samples per grid in GRF_LOW_GRIDS, computed at 56x56 and bilinearly upsampled), then orthonormalised by QR.
dct: lowest-order 2-D cosine modes.
rand: random directions (control).

The basis is built with 2K rows and is nested; the first K rows are the projector (U), and the next K (U2) are used only to compute a leakage proxy leak_last = ||U2 zp|| / ||zp|| for diagnostics.

Invariance is exact up to the ridge term. The front-end runs in fp32 under torch.no_grad(), so no gradients flow into it.

2.2 Orbit-consistency training

With probability CONS_P = 0.5 per batch, the model sees x and a nuisance-transformed copy x' and minimises:

0.5 * (FocalCE(l1, y) + FocalCE(l2, y))  +  CONS_LAMBDA * SymKL(l1, l2)
2.3 Lipschitz head

The linear classifier head is wrapped in spectral_norm and multiplied by a learnable scalar exp(logit_scale) (initialised to 5). The head's Lipschitz constant is therefore exp(logit_scale), reported as head_L.

2.4 Classical baseline (HomomorphicInput)

log -> subtract a mask-normalised Gaussian low-pass (sigma = 16) -> RMS-normalise, with the same three-channel output as the quotient. It removes smooth bias but with a fixed filter instead of the projection onto a fitted subspace.

3. Models and ablations

Model names follow <KIND>_<Backbone> or MIQ_<Backbone>_<ablation>.

Kind	Front-end	Nuisance augmentation	Consistency loss	Lipschitz head	Role
BASE	raw (ImageNet norm)	no	no	no	Plain ERM
AUG	raw	yes (p = 0.5 per batch)	no	no	ERM + nuisance augmentation
HOMO	homomorphic filter	no	no	no	Classical bias removal + ERM
CONS	raw	via consistency pairs	yes	no	Control: MIQ's training recipe without the quotient
MIQ	quotient	via consistency pairs	yes	yes	Proposed
Backbones (all pretrained on ImageNet-1k by default)
Key	timm name	Approx. size
EdgeNeXtXXS	edgenext_xx_small.in1k	~1.3M
FastViTT8	fastvit_t8.apple_in1k	~3.6M
MobileNetV3L	mobilenetv3_large_100.ra_in1k	~4.2M

The proposed model is MIQ_<first backbone in BACKBONE_NAMES>, i.e. MIQ_EdgeNeXtXXS unless you filter with MIQ_BACKBONES.

Ablations (leave-one-out / sensitivity, applied to full MIQ)
Suffix	Change
no_inq	Raw front-end (keeps consistency loss and Lipschitz head)
no_gamma	No scale (RMS) normalisation
no_mask	No brain mask
dct_basis	DCT basis instead of KL
rand_basis	Random basis instead of KL
k1	K = 1: log + mask + centre + scale only (no smooth-bias removal)
k6	K = 6
k24	K = 24
no_cons	No orbit-consistency loss
no_lip	Plain linear head

By default ablations run on the first backbone only. Set MIQ_ABL_ALL=1 for all backbones.

Run count: 5 kinds x 3 backbones = 15 models, plus 10 ablations = 25 models x 5 seeds = 125 training runs by default. With MIQ_ABL_ALL=1: 45 models x 5 seeds = 225 runs.

4. Requirements and installation

Python 3.9+ and a CUDA GPU are recommended (CPU works but is slow). Main dependencies:

torch
torchvision        # needs torchvision.transforms.v2 (>= 0.16)
timm               # recent enough to provide the model/tag names above
numpy
pandas
scipy
scikit-learn       # StratifiedGroupKFold (>= 1.0)
matplotlib
seaborn
Pillow
bash
pip install torch torchvision timm numpy pandas scipy scikit-learn matplotlib seaborn pillow

Pretrained weights are downloaded by timm on first use, so the machine needs internet access (or a populated timm/Hugging Face cache). Set MIQ_PRETRAINED=0 to train from scratch with no downloads. The exact library versions used for a run are saved to run_config.json.

If your GPU supports bf16, the backbone runs under bf16 autocast; the quotient front-end and the head stay in fp32. Otherwise everything runs in fp32. No GradScaler is used.

5. Dataset layout

Two torchvision.datasets.ImageFolder directories, one sub-folder per class:

Training/
  glioma/  meningioma/  notumor/  pituitary/
Test/
  glioma/  meningioma/  notumor/  pituitary/

This matches the common public four-class brain-tumour MRI dataset. Images are converted to RGB and resized to 224x224. Class order is whatever ImageFolder yields (alphabetical).

Optional external dataset (MIQ_EXT): another ImageFolder. Only class names that also exist in the training set are scored; the result is reported as acc_ext.

Set paths via environment variables (the defaults in the script point at a specific cluster path and must be overridden):

bash
export MIQ_TRAIN="/path/to/Training/"
export MIQ_TEST="/path/to/Test/"
6. Usage
bash
# 1. Self-test of theory and architectures (no dataset, no pretrained weights needed)
python MIQ_main_v2.py --smoke

# 2. Verify the timm backbone names build with features_only=True
python MIQ_main_v2.py --check-backbones

# 3. Near-duplicate audit between train and test only
python MIQ_main_v2.py --audit

# 4. Full multi-seed experiment (all models x all seeds)
python MIQ_main_v2.py

Examples:

bash
# Quick trial: one backbone, two seeds, few epochs
MIQ_BACKBONES=EdgeNeXtXXS MIQ_SEEDS=42,123 MIQ_EPOCHS=5 python MIQ_main_v2.py

# Ablations on every backbone, specific GPU
GPU_ID=0 MIQ_ABL_ALL=1 python MIQ_main_v2.py

# Train from scratch, with an external test set
MIQ_PRETRAINED=0 MIQ_EXT=/data/external_mri python MIQ_main_v2.py
7. Configuration reference
Environment variables
Variable	Default	Meaning
MIQ_TRAIN	cluster path	Training ImageFolder
MIQ_TEST	cluster path	Test ImageFolder
MIQ_EXT	""	Optional external ImageFolder
MIQ_SAVE	MIQ_v2_final	Output directory
MIQ_EPOCHS	20	Max epochs per run
MIQ_SEEDS	42,123,456,789,2024	Comma-separated seeds
MIQ_BACKBONES	all	Comma-separated subset of EdgeNeXtXXS,FastViTT8,MobileNetV3L
MIQ_PRETRAINED	1	1 = ImageNet weights, 0 = random init
MIQ_ABL_ALL	0	1 = run ablations on every backbone
MIQ_AUDIT	1	1 = run near-duplicate audit if not already cached
GPU_ID	5	CUDA device index (falls back to cuda:0, then CPU)
Key constants (edit in the script)
Constant	Value	Notes
IMG_SIZE	224	
LR / WARMUP_EPOCHS	3e-4 / 2	Cosine schedule after warm-up, min LR 1e-6
BATCH / EVAL_BATCH	32 / 64	
PATIENCE	8	Early stopping on validation loss
LABEL_SMOOTH	0.08	Focal loss with gamma = 2
CUTMIX_ALPHA / MIXUP_ALPHA	1.0 / 0.4	
K_MODES	12	Dimension of nuisance subspace
MASK_THR	0.02	Brain mask threshold
RIDGE	1e-3	Projector regulariser
GRF_LOW_GRIDS	(3,4,5,6,7,8)	Training bias scales
OUT_GRID	12	Held-out higher-frequency bias
P_OUT_TRAIN	0.0	Fraction of training nuisance using OUT_GRID (0 = truly unseen)
CONS_P / CONS_LAMBDA	0.5 / 1.0	Consistency probability / weight
DUP_HAMMING	4	Near-duplicate threshold (average hash)
N_BOOTSTRAP	2000	Bootstrap resamples for CIs
ALPHA_STAT	0.05	Significance level
GRADCAM_MAX_PER_CLASS	30	Grad-CAM images per class
8. Training protocol
Optimiser: AdamW, weight decay 1e-2. Backbone LR multiplier 0.1, head multiplier 1.0.
Schedule: linear warm-up (2 epochs) then cosine decay.
Loss: focal loss (gamma 2) with label smoothing 0.08.
Regularisation: gradient clipping at 2.0, dropout 0.3 before the head.
Spatial augmentation: RandomResizedCrop (scale 0.7-1.0), horizontal flip, vertical flip (p 0.2), rotation 30 degrees, shear 10.
CutMix / MixUp: alternate by epoch (CutMix on even epochs, MixUp on odd), with the CutMix alpha ramped up over the first 20 epochs. They are applied only to batches that do not take the consistency branch.
Model selection: best validation loss; the test set is never used for selection.
Validation split: StratifiedGroupKFold(n_splits=7), taking one fold (~1/7 of the data). Groups are near-duplicate clusters in the training set (union-find over average-hash Hamming distance <= 4), so a cluster never straddles train and validation. If grouping fails the script falls back to a stratified 85/15 split and prints a warning.
Training nuisance (used by AUG, CONS, MIQ): x' = renorm(x^gamma * exp(s * f)), with gamma log-uniform in [0.6, 1.6], s in [0.3, 1.2], and f a low-frequency Gaussian random field (grid from GRF_LOW_GRIDS). Background pixels stay 0.
9. Evaluation protocol

For every (model, seed) the script records:

Clean test performance

Accuracy, macro F1, macro precision, macro recall, macro AUC, Cohen's kappa
Per-class precision / recall / F1 / AUC / specificity
Bootstrap 95% CIs (2000 resamples)
Confusion matrix (CSV + PNG)
ECE (15 bins)

Held-out robustness (deterministic seeds, never used in training)

Kind	Perturbation	Levels
bias_in	GRF bias, grid 5 (in-family sanity check)	s = 0.5, 1.0, 1.5, 2.0
bias_out	GRF bias, grid 12 (unseen frequency)	s = 0.5, 1.0, 1.5, 2.0
radial	Smooth parametric radial gain with random centre (unseen family)	s = 0.5, 1.0, 1.5, 2.0
gamma	Gamma change, no bias	0.5, 0.75, 1.5, 2.0
noise	Rician noise	sigma = 0.02, 0.05

Note that gamma 0.75 and 1.5 lie inside the training range [0.6, 1.6]; 0.5 and 2.0 are extrapolations. The summary column acc_gamma uses gamma = 0.5.

For each condition the script stores accuracy, prediction flip rate versus the clean prediction, and mean logit change.

Theorem-3 diagnostics (models with a quotient/homomorphic front-end): the mean change in phi (normalised by sqrt of mask size), empirical logit-to-phi ratio percentiles (Lhat_p50, Lhat_p95, Lhat_max), and the Spearman correlation between the leakage proxy and the logit change.

Shortcut / site probe: image dimensions of the test images are used as a crude site proxy (sizes with fewer than 15 images are pooled into other). A logistic regression on penultimate embeddings predicts the site; site_excess is its balanced accuracy minus a label-only baseline. Lower means less site information beyond what the label explains.

Leakage audit: each test image is compared with all training images by a 16x16 average hash. Test images within Hamming distance 4 of any training image are flagged, saved to dup_test_idx.npy, and acc_dedup reports accuracy on the remaining de-duplicated test subset.

Cost: parameter count (M) and latency in ms/img over 64 test images (front-end included, batches of 32).

Interpretability: Grad-CAM on the five main kinds (not ablations), up to 30 images per class per seed. For quotient/homomorphic models the figure also shows phi.

10. Statistics

All tests are paired over seeds on a shared test set.

Test	Scope	Output
McNemar (continuity-corrected chi-square and exact binomial)	Image-wise, per seed, all model pairs	mcnemar_seedwise.csv
Fisher combination of per-seed exact p-values	Across seeds	mcnemar_aggregated.csv
Wilcoxon signed-rank + paired t-test + Cohen d_z	Per model pair over seeds, every metric	wilcoxon_ttest_results.csv
Superiority table (proposed vs. every other model, Holm-adjusted t-test)	Per metric	superiority_table.csv
Pooled family tests (MIQ vs BASE / AUG / HOMO / CONS paired over backbone x seed; bootstrap CI on the mean difference; Holm across families)	Per metric	pooled_family_tests.csv

Verdicts are SIGNIFICANT WIN (numerical win and Holm-adjusted p < 0.05), numerical win, or no win. For ece and site_excess, lower is better.

Things to keep in mind:

With n = 5 seeds, the exact two-sided Wilcoxon test cannot go below p = 0.0625. The paired t-test, d_z, Fisher-combined McNemar and pooled tests carry the inference.
Pooled tests treat (backbone, seed) pairs as units; pairs sharing a backbone are not fully independent, so the pooled test complements, rather than replaces, per-backbone tests.
Pooled tests need at least 6 pairs. A single-backbone run with 5 seeds has only 5 pairs and is skipped; use at least 2 backbones (or more seeds).
Per-pair Wilcoxon/t-tests need at least 3 matched seeds.
11. Output files

Everything is written under MIQ_SAVE (default MIQ_v2_final/).

MIQ_v2_final/
├── run_config.json               # versions, hyper-parameters, paths
├── audit_leakage.csv             # train/test near-duplicate summary
├── dup_test_idx.npy              # indices of near-duplicate test images
├── train_groups.npy              # near-duplicate cluster ids (training set)
├── all_results.csv               # one row per (model, seed)
├── summary_mean_std.csv          # mean/std per model
├── mcnemar_seedwise.csv
├── mcnemar_aggregated.csv
├── wilcoxon_ttest_results.csv
├── superiority_table.csv
├── pooled_family_tests.csv
├── robust_all.csv                # all robustness sweeps concatenated
├── theorem3_diag_all.csv         # leakage diagnostics concatenated
├── site_probe_all.csv
├── barplot_<metric>.png          # mean +/- std for every metric
├── robustness_<backbone>.png     # accuracy vs perturbation level per kind
├── ablation_delta_<backbone>.csv / .png   # (ablation - full MIQ)
└── <model_name>/
    ├── role.txt
    ├── metrics/
    │   ├── history_seed*.csv
    │   ├── test_metrics_seed*.csv        # per-class metrics
    │   ├── bootstrap_ci_seed*.csv
    │   ├── cm_seed*.csv / .png
    │   ├── robust_seed*.csv
    │   ├── theorem3_diag_seed*.csv       # quotient / homomorphic models only
    │   ├── site_probe_seed*.csv
    │   └── external_seed*.csv            # only if MIQ_EXT is set
    ├── curves/curves_seed*.png           # loss (blue, left) / accuracy (red, right)
    ├── gradcam/seed*/<class>/img_*.png   # main kinds only
    └── preds/preds_seed*.npz             # y_true, y_pred, y_prob
Columns of all_results.csv
Column	Meaning
model, seed	Identifiers
acc, f1_macro, precision_macro, recall_macro, auc_macro, kappa	Clean test metrics
acc_bias_in, acc_bias_out, acc_radial	Accuracy at bias scale s = 1.0
acc_gamma	Accuracy at gamma = 0.5
acc_noise	Accuracy at Rician sigma = 0.05
site_excess	Site-probe excess (lower is better)
ece	Expected calibration error (lower is better)
acc_dedup	Accuracy on de-duplicated test subset
acc_ext	Accuracy on external dataset (NaN if unset)
head_L	Lipschitz scale of the head (NaN when no Lipschitz head)
params_M, ms_per_img	Cost
12. Smoke test

python MIQ_main_v2.py --smoke needs no data and no weights. It asserts:

T2: bases are orthonormal and, on held-out in-family fields, leakage satisfies KL <= DCT < random.
T1: a synthetic brain pushed through gamma = 1.37 and an in-span bias gives a nearly identical phi (error < 2e-2), while a different image is more than 10x further away.
T3: measured leakage for an out-of-span (grid 12) bias is within the stated bound.
Nuisance functions preserve shape; ECE of a perfect predictor is 0.
Every architecture (all kinds x backbones, plus all ablations on the first backbone) builds, does a forward and backward pass, produces finite logits of the right shape, and supports Grad-CAM. For the full MIQ models it also prints the logit gap under an in-span nuisance.

Run this first after any change.

13. Reproducibility and resuming
set_seed seeds Python, NumPy and PyTorch, and sets cuDNN to deterministic (benchmark off). bf16 autocast and some GPU kernels can still cause small run-to-run differences.
Robustness perturbations use fixed generator seeds (777 + condition index), so every model sees the same perturbations.
The experiment is resumable. A (model, seed) is skipped if it is already in all_results.csv and its preds_seed<seed>.npz exists. If a run is interrupted, re-run the same command.
To force a rerun, delete the relevant row in all_results.csv and/or the model's preds/ file. To redo the audit or grouping, delete dup_test_idx.npy / train_groups.npy.
If MIQ_SEEDS, MIQ_BACKBONES or ablation settings change between invocations, the final tables are built from whatever is in all_results.csv, so keep one output directory per experimental configuration.
14. Practical notes and known caveats
Hard-coded defaults: the dataset paths and GPU_ID = 5 default to a specific cluster. Override them with environment variables.
Batches on the consistency branch (probability 0.5 for CONS, MIQ and their ablations) do not use CutMix/MixUp, so the effective mixing rate differs between those models and BASE/AUG/HOMO. Keep this in mind when comparing training recipes.
no_inq replaces only the front-end; it keeps the consistency loss and Lipschitz head, so it is the right comparison for "what does the quotient add", and differs from CONS only in the Lipschitz head.
Near-duplicate detection is a heuristic (16x16 average hash, Hamming <= 4). It will miss heavily transformed duplicates and may flag near-identical but distinct slices.
Site proxy is image size only. It is a coarse shortcut probe, not a real scanner/site label.
Latency is measured on 64 images at batch size 32 on the current device, so it is hardware-dependent.
Grad-CAM is generated only for BASE, AUG, HOMO, CONS, MIQ per backbone; ablations are skipped.
Sample-level failures: if a training batch raises a RuntimeError (e.g., OOM), the batch is skipped with a printed message rather than aborting the run. Check logs for repeated skips.
Runtime: the full default experiment is 125 training runs plus evaluation, robustness sweeps (17 conditions per run), bootstrap CIs and Grad-CAM. Use MIQ_BACKBONES, MIQ_SEEDS and MIQ_EPOCHS for development.
15. Troubleshooting
Symptom	Likely cause / fix
Cannot build '<name>'	The timm version does not provide that model/tag. Run --check-backbones; the error lists similar model names to edit BACKBONE_REGISTRY.
Weights fail to download	No internet or blocked hub. Pre-populate the cache or use MIQ_PRETRAINED=0.
group split unavailable warning	Hashing the training set failed (bad path or unreadable image); the stratified fallback is used, which can let near-duplicates straddle train/val.
audit skipped	Same cause; check MIQ_TRAIN / MIQ_TEST.
CUDA out of memory	Lower BATCH / EVAL_BATCH; the front-end solves a small K x K system per image, which is cheap, but Grad-CAM and robustness sweeps hold full batches.
Pooled tests are empty	Fewer than 6 (backbone, seed) pairs. Run more backbones or seeds.
acc_ext is NaN	MIQ_EXT is unset, or none of its class names match the training classes.
Very slow on CPU	Expected; use a GPU or reduce backbones, seeds and epochs.
Tables mix old and new runs	all_results.csv is reused across invocations. Use a fresh MIQ_SAVE per configuration.
16. Code map
Section of MIQ_main_v2.py	Contents
Configuration	Paths, hyper-parameters, backbone registry, model and ablation lists
Model-name to config	variant_cfg, get_dirs, set_seed, save_run_config
Theory core	build_basis (KL / DCT / random), Sobel, RawInput, QuotientInput, HomomorphicInput
Nuisance	bias_field, apply_nuisance, apply_radial, apply_rician, synth_random, perturb, ROBUST_CONDS
Models	Net (front-end + timm backbone + dropout + optional spectral-norm head), GradCAM
Augmentation / loss	CutMix, MixUp, LR schedule, FocalLoss, sym_kl, parameter groups
Metrics	Macro AUC, ECE, bootstrap CIs, save_test_metrics
Dataset audit	Average-hash audit, near-duplicate grouping, group-aware split, site proxy and probe, latency
Robustness / T3 diagnostics	robustness_eval
Training	run_one_seed (train, select, test, robustness, probe, external, Grad-CAM, save)
Statistics	McNemar, Fisher combination, Holm, Wilcoxon / t-test / d_z, pooled family tests, superiority table
Plots	Bar plots, robustness curves, ablation deltas
Self-tests	smoke_test, check_backbones
Main	Pipeline orchestration
License and citation

Add your licence and citation details here.
