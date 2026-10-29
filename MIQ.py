# -*- coding: utf-8 -*-
"""
MIQ_main_v2.py
==============
MIQ-Net : Maximal-Invariant Quotient Network
Brain-tumour MRI classification (glioma | meningioma | notumor | pituitary)
Light backbones: EdgeNeXt-XXS (1.3M), FastViT-T8 (3.6M), MobileNetV3-Large (4.2M)

-------------------------------------------------------------------------------
WHAT IS CLAIMED  (deliberately narrow, so every claim is checkable)
-------------------------------------------------------------------------------
MRI nuisance acts on the LOG-image z = log I as   z -> gamma*z + b,  gamma>0,
b in U (K-dim smooth subspace incl. constant gain).  The label is invariant to
this group, so the classifier can be forced to factor through the quotient X/G.

 T1 (exact on the masked domain, up to the ridge term)
     Phi(z) = P_perp z / ||P_perp z||_mask is invariant to z->gamma z+b and
     Phi(z1)=Phi(z2)  <=>  z2 = c z1 + b, c>0, b in U.
 T2 (Eckart-Young)  For a bias distribution with covariance C, the top-K KL
     subspace minimises E||P_perp b||^2.  NOTE: the KL basis here is computed
     from the SAME Gaussian-random-field family used for the 'bias_in' test, so
     'bias_in' is an in-family sanity check, NOT independent evidence.
     Independent evidence = 'bias_out' (unseen frequency) and 'radial'
     (structurally different parametric gain field), both NEVER used in training.
 T3 (leakage bound)  ||Phi(z')-Phi(z)|| <= 2||P_perp b_tail|| / (gamma ||P_perp z||)
     (standard inequality ||a/|a| - b/|b||| <= 2||a-b||/||a||). Measured, not assumed.

POSITIONING (what is NOT new):  log-domain bias removal (homomorphic filtering,
N4), gain/contrast normalisation, spectral-norm Lipschitz heads and consistency
regularisation are all known.  The study's contribution is the COMBINATION with a
provable maximal-invariant quotient, the leakage diagnostics, and a protocol with
controls that separate "quotient" from "augmentation", "consistency loss" and
"classical homomorphic filtering".

LIMITS (stated honestly): additive/Rician noise and out-of-span bias are not
removed exactly; the sweeps expose this.  Public brain-MRI test accuracy is near
ceiling, so the headline evidence is robustness / shift / calibration / cost, with
significance tests, NOT the last 0.2% of clean accuracy.  No component ablation is
run in this script, so no claim is made about which individual component of MIQ
is responsible for any gain.

-------------------------------------------------------------------------------
MODELS  (5 kinds x 3 backbones = 15 models)
-------------------------------------------------------------------------------
 BASE_b  raw ERM
 AUG_b   raw + nuisance augmentation
 HOMO_b  homomorphic filter + ERM
 CONS_b  raw + orbit-consistency loss (CONTROL: same recipe as MIQ, no quotient)
 MIQ_b   quotient + orbit-consistency + Lipschitz head (proposed)

PROTOCOL HARDENING (v2)
 * validation split is GROUP-aware (near-duplicate clusters never straddle train/val)
 * test set never used for model selection; test near-duplicates audited, accuracy
   also reported on the de-duplicated test subset
 * training nuisance = low-frequency GRF bias + gamma in [0.6,1.6] ONLY;
   bias_out (grid 12), radial bias, gamma 0.5 / 2.0, Rician noise are held out
 * metrics: acc/F1/prec/rec/AUC, kappa, ECE, bootstrap CIs, per-seed McNemar,
   Wilcoxon + paired t + Cohen d_z + Holm, POOLED family tests (backbone x seed)
 * params and ms/img (front-end included) for the "lightweight" claim
 * site-shortcut probe, optional external dataset (MIQ_EXT)

USAGE
  python MIQ_main_v2.py --smoke            # theory + architecture self-test (no data)
  python MIQ_main_v2.py --check-backbones  # verify timm names / features_only
  python MIQ_main_v2.py --audit            # near-duplicate audit only
  python MIQ_main_v2.py                    # full multi-seed run
Env: MIQ_TRAIN MIQ_TEST MIQ_SAVE MIQ_EXT MIQ_EPOCHS MIQ_SEEDS MIQ_BACKBONES
     MIQ_PRETRAINED MIQ_AUDIT GPU_ID
"""

import os
import sys
import gc
import json
import math
import time
import random
import itertools
import warnings
import platform
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import chi2, binom, wilcoxon, ttest_rel, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2
from PIL import Image

import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, classification_report, cohen_kappa_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict, StratifiedGroupKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 10, "figure.titlesize": 15,
    "figure.titleweight": "bold", "axes.linewidth": 1.4, "lines.linewidth": 2.2,
})

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRAIN_PATH = os.environ.get("MIQ_TRAIN", "/nfsshare/users/P127163008/brainz/Braintumor dataset/Training/")
TEST_PATH = os.environ.get("MIQ_TEST", "/nfsshare/users/P127163008/brainz/Braintumor dataset/Test/")
EXT_PATH = os.environ.get("MIQ_EXT", "")          # optional external ImageFolder
SAVE_DIR = Path(os.environ.get("MIQ_SAVE", "MIQ_v2_final"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED = os.environ.get("MIQ_PRETRAINED", "1") == "1"
RUN_AUDIT = os.environ.get("MIQ_AUDIT", "1") == "1"

_gpu_id = int(os.environ.get("GPU_ID", 5))
if torch.cuda.device_count() > _gpu_id:
    DEVICE = torch.device(f"cuda:{_gpu_id}")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# bf16 autocast for the BACKBONE only (quotient front-end stays fp32); no GradScaler needed
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else None

IMG_SIZE = 224
LR = 3e-4
WARMUP_EPOCHS = 2
EPOCHS = int(os.environ.get("MIQ_EPOCHS", 20))
PATIENCE = 8
LABEL_SMOOTH = 0.08
CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.4
EPS = 1e-8
ALPHA_STAT = 0.05
BATCH = 32
EVAL_BATCH = 64
N_BOOTSTRAP = 2000
GRADCAM_MAX_PER_CLASS = 30

# quotient / nuisance hyper-parameters
K_MODES = 12
MASK_THR = 0.02
RIDGE = 1e-3
GRF_LOW_GRIDS = (3, 4, 5, 6, 7, 8)   # in-family bias scales (training + bias_in test)
OUT_GRID = 12                         # HELD-OUT higher-frequency bias (test only)
P_OUT_TRAIN = 0.0                     # fraction of training nuisance using OUT_GRID (0 => truly unseen)
CONS_P = 0.5
CONS_LAMBDA = 1.0
DETERMINISTIC_SYNTH_SEED = 777
DUP_HAMMING = 4

SEEDS = [int(s) for s in os.environ.get("MIQ_SEEDS", "42,123,456,789,2024").split(",")]

BACKBONE_REGISTRY = {
    "EdgeNeXtXXS":  "edgenext_xx_small.in1k",
    "FastViTT8":    "fastvit_t8.apple_in1k",
    "MobileNetV3L": "mobilenetv3_large_100.ra_in1k",
}
_bsel = os.environ.get("MIQ_BACKBONES")
BACKBONE_NAMES = [b for b in BACKBONE_REGISTRY if (not _bsel or b in _bsel.split(","))]

KINDS = ("BASE", "AUG", "HOMO", "CONS", "MIQ")

MODELS = [f"{k}_{b}" for k in KINDS for b in BACKBONE_NAMES]
GRADCAM_MODELS = list(MODELS)
PROPOSED = f"MIQ_{BACKBONE_NAMES[0]}"

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
EXTRA = ["acc_bias_in", "acc_bias_out", "acc_radial", "acc_gamma", "acc_noise",
         "site_excess", "ece", "acc_dedup", "acc_ext"]
LOWER_BETTER = {"site_excess", "ece"}
METRIC_LABELS = {
    "acc": "Accuracy", "f1_macro": "Macro F1", "precision_macro": "Macro Precision",
    "recall_macro": "Macro Recall", "auc_macro": "Macro AUC", "kappa": "Cohen kappa",
    "acc_bias_in": "Acc | in-family bias s=1", "acc_bias_out": "Acc | held-out high-freq bias s=1",
    "acc_radial": "Acc | held-out radial bias s=1",
    "acc_gamma": "Acc | gamma=0.5 (extrapolated)", "acc_noise": "Acc | Rician sigma=0.05",
    "site_excess": "Site-probe excess (lower=better)", "ece": "ECE (lower=better)",
    "acc_dedup": "Acc | de-duplicated test", "acc_ext": "Acc | external dataset",
}
ALL_COLS = METRICS + ["kappa"] + EXTRA


def short(m):
    return (m.replace("BASE_", "B-").replace("AUG_", "A-").replace("HOMO_", "H-")
             .replace("CONS_", "C-").replace("MIQ_", "Q-"))


# --------------------------------------------------------------------------
# Model-name -> config
# --------------------------------------------------------------------------
def variant_cfg(name):
    kind, bname = name.split("_", 1)
    if bname not in BACKBONE_REGISTRY:
        raise ValueError(f"Bad model name: {name}")
    cfg = dict(front="raw", aug_nuis=False, cons=False, lip=False)
    if kind == "AUG":
        cfg["aug_nuis"] = True
    elif kind == "HOMO":
        cfg["front"] = "homo"
    elif kind == "CONS":
        cfg["cons"] = True
    elif kind == "MIQ":
        cfg.update(front="quotient", cons=True, lip=True)
    elif kind != "BASE":
        raise ValueError(f"Unknown kind {kind}")
    return BACKBONE_REGISTRY[bname], cfg


def get_dirs(model_name):
    root = SAVE_DIR / model_name
    for d in [root / "metrics", root / "curves", root / "gradcam", root / "preds"]:
        d.mkdir(parents=True, exist_ok=True)
    note = root / "role.txt"
    if not note.exists():
        if model_name.startswith("MIQ_"):
            kind = "Maximal-Invariant Quotient Network - proposed"
        elif model_name.startswith("HOMO_"):
            kind = "Classical homomorphic bias removal + ERM baseline"
        elif model_name.startswith("CONS_"):
            kind = "Raw input + orbit-consistency loss (control for MIQ training recipe)"
        elif model_name.startswith("AUG_"):
            kind = "ERM + nuisance augmentation baseline"
        else:
            kind = "Plain ERM baseline"
        note.write_text(f"Model : {model_name}\nKind  : {kind}\n")
    return {"root": root, "metrics": root / "metrics", "curves": root / "curves",
            "gradcam": root / "gradcam", "preds": root / "preds"}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_config():
    cfg = dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__,
               device=str(DEVICE), amp=str(AMP_DTYPE), seeds=SEEDS, epochs=EPOCHS, batch=BATCH,
               lr=LR, patience=PATIENCE, k_modes=K_MODES, ridge=RIDGE, mask_thr=MASK_THR,
               grf_low_grids=list(GRF_LOW_GRIDS), out_grid=OUT_GRID, p_out_train=P_OUT_TRAIN,
               cons_p=CONS_P, cons_lambda=CONS_LAMBDA, backbones=BACKBONE_REGISTRY,
               models=MODELS, proposed=PROPOSED, train_path=TRAIN_PATH, test_path=TEST_PATH,
               ext_path=EXT_PATH)
    (SAVE_DIR / "run_config.json").write_text(json.dumps(cfg, indent=2))


# ==========================================================================
# THEORY CORE
# ==========================================================================
def _gauss_field_vectors(n_per_grid, grids, out_size, gen):
    xs = []
    for g in grids:
        nz = torch.randn(n_per_grid, 1, g, g, generator=gen)
        xs.append(F.interpolate(nz, size=(out_size, out_size), mode="bilinear",
                                align_corners=True).flatten(1))
    return torch.cat(xs)


def build_basis(kind, k_total, n, seed=0):
    """Orthonormal basis, rows = modes, shape (k_total, n*n); row 0 = constant (gain).
    Nested: the first k rows span the same space for any k <= k_total.
      'kl'   top (k_total-1) Karhunen-Loeve modes of the in-family bias [Theorem 2]
      'dct'  lowest-order cosine modes      'rand' random directions
    The model uses 'kl'; 'dct' and 'rand' exist only for the Theorem-2 smoke check."""
    P = n * n
    gen = torch.Generator().manual_seed(seed)
    if kind == "kl":
        m = 56
        X = _gauss_field_vectors(500, GRF_LOW_GRIDS, m, gen)
        X = X - X.mean(1, keepdim=True)
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        modes = Vh[:k_total - 1].reshape(-1, 1, m, m)
        modes = F.interpolate(modes, size=(n, n), mode="bilinear",
                              align_corners=True).flatten(1)
    elif kind == "dct":
        ax = (torch.arange(n).float() + 0.5) / n
        order = sorted([(p + q, p, q) for p in range(12) for q in range(12)])
        order = [o for o in order if o[0] > 0][:k_total - 1]
        modes = torch.stack([(torch.cos(math.pi * p * ax)[:, None] *
                              torch.cos(math.pi * q * ax)[None, :]).flatten()
                             for _, p, q in order])
    elif kind == "rand":
        modes = torch.randn(k_total - 1, P, generator=gen)
    else:
        raise ValueError(kind)
    M = torch.cat([torch.ones(1, P), modes])
    Q, _ = torch.linalg.qr(M.T)
    return Q.T.contiguous()


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("k", torch.stack([kx, kx.T])[:, None])

    def forward(self, t):
        g = F.conv2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), self.k)
        return torch.sqrt((g ** 2).sum(1, keepdim=True) + 1e-6)


class RawInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


class QuotientInput(nn.Module):
    """Maximal-invariant quotient front-end (Theorem 1).
    g = gray(x); m = brain mask; z = log g; zp = P_perp z; phi = zp / RMS(zp).
    Output channels [phi, |grad phi|, tanh(phi)] (all functions of phi only).
    Invariance is exact up to the ridge term (RIDGE) in the projector."""

    def __init__(self, n=IMG_SIZE, K=K_MODES, basis="kl", thr=MASK_THR, ridge=RIDGE):
        super().__init__()
        self.n, self.K = n, K
        self.thr, self.ridge = thr, ridge
        U_all = build_basis(basis, 2 * K, n)
        self.register_buffer("U", U_all[:K].contiguous())
        self.register_buffer("U2", U_all[K:].contiguous())
        self.register_buffer("I_K", torch.eye(K))
        self.edge = Sobel()
        self.phi_last = None
        self.leak_last = None
        self.nm_last = None

    @torch.no_grad()
    def project(self, z, m):
        Um = self.U[None] * m[:, None, :]
        G = Um @ self.U.T + self.ridge * self.I_K
        r = torch.einsum("bkp,bp->bk", Um, z)
        c = torch.linalg.solve(G, r.unsqueeze(-1)).squeeze(-1)
        return (z - c @ self.U) * m

    @torch.no_grad()
    def log_and_mask(self, x):
        g = x.float().mean(1, keepdim=True)
        m = (g > self.thr).float()
        z = torch.log(g.clamp_min(1e-3)) * m
        return z.flatten(1), m.flatten(1)

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.shape
        z, m = self.log_and_mask(x)
        zp = self.project(z, m)
        nm = m.sum(1).clamp_min(1.0)
        zp = zp / torch.sqrt((zp ** 2).sum(1) / nm + 1e-6)[:, None]
        self.leak_last = (zp @ self.U2.T).norm(dim=1) / (zp.norm(dim=1) + 1e-6)
        self.nm_last = nm
        phi = zp.view(B, 1, H, W)
        self.phi_last = phi
        return torch.cat([phi, self.edge(phi), torch.tanh(phi)], dim=1)


class HomomorphicInput(nn.Module):
    """Classical baseline: log -> subtract mask-normalised Gaussian low-pass -> std-normalise."""

    def __init__(self, sigma=16.0, thr=MASK_THR):
        super().__init__()
        r = int(3 * sigma)
        ax = torch.arange(-r, r + 1).float()
        k = torch.exp(-ax ** 2 / (2 * sigma ** 2)); k = k / k.sum()
        self.r, self.thr = r, thr
        self.register_buffer("kx", k.view(1, 1, 1, -1))
        self.register_buffer("ky", k.view(1, 1, -1, 1))
        self.edge = Sobel()
        self.phi_last = None
        self.leak_last = None
        self.nm_last = None

    def blur(self, t):
        t = F.pad(t, (self.r,) * 4, mode="reflect")
        return F.conv2d(F.conv2d(t, self.kx), self.ky)

    @torch.no_grad()
    def forward(self, x):
        g = x.float().mean(1, keepdim=True)
        m = (g > self.thr).float()
        z = torch.log(g.clamp_min(1e-3)) * m
        low = self.blur(z * m) / (self.blur(m) + 1e-3)
        zp = (z - low) * m
        nm = m.flatten(1).sum(1).clamp_min(1.0)
        s = torch.sqrt((zp ** 2).flatten(1).sum(1) / nm + 1e-6).view(-1, 1, 1, 1)
        phi = zp / s
        self.phi_last, self.nm_last = phi, nm
        return torch.cat([phi, self.edge(phi), torch.tanh(phi)], dim=1)


# ---------------------------------------------------------------- nuisance
def _bc(v, x):
    v = torch.as_tensor(v, dtype=x.dtype, device=x.device)
    if v.dim() == 0:
        v = v.expand(x.size(0))
    return v.view(-1, 1, 1, 1)


def bias_field(B, n, grid, device, gen=None):
    nz = torch.randn(B, 1, grid, grid, generator=gen, device=device)
    f = F.interpolate(nz, size=(n, n), mode="bilinear", align_corners=True)
    f = f - f.mean((2, 3), keepdim=True)
    return f / (f.std((2, 3), keepdim=True) + 1e-6)


def _renorm(xp):
    return xp / xp.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)


def apply_nuisance(x, s, gamma, grid, gen=None):
    """x' = rescale( x^gamma * exp(s * GRF bias) ).  Background 0 stays 0."""
    f = bias_field(x.size(0), x.size(-1), grid, x.device, gen)
    return _renorm(x.clamp_min(0) ** _bc(gamma, x) * torch.exp(_bc(s, x) * f))


def apply_radial(x, s, gen=None):
    """HELD-OUT bias family: smooth parametric radial (coil-sensitivity / vignetting-like)
    gain with random centre; structurally different from the Gaussian-field training family."""
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                            torch.linspace(-1, 1, W, device=x.device), indexing="ij")
    cy = (torch.rand(B, 1, 1, device=x.device, generator=gen) - 0.5) * 0.8
    cx = (torch.rand(B, 1, 1, device=x.device, generator=gen) - 0.5) * 0.8
    f = -((yy[None] - cy) ** 2 + (xx[None] - cx) ** 2).unsqueeze(1)
    f = f - f.mean((2, 3), keepdim=True)
    f = f / (f.std((2, 3), keepdim=True) + 1e-6)
    return _renorm(x.clamp_min(0) * torch.exp(_bc(s, x) * f))


def apply_rician(x, sigma, gen=None):
    n1 = torch.randn(x.shape, generator=gen, device=x.device)
    n2 = torch.randn(x.shape, generator=gen, device=x.device)
    return _renorm(torch.sqrt((x + sigma * n1) ** 2 + (sigma * n2) ** 2))


def synth_random(x, p_apply=1.0, p_out=P_OUT_TRAIN):
    """Training nuisance (AUG baseline, CONS and MIQ orbit-consistency).
    gamma in [0.6,1.6], bias scale in [0.3,1.2], LOW-frequency grids only by default."""
    B = x.size(0)
    gamma = torch.exp(torch.empty(B, device=x.device).uniform_(math.log(0.6), math.log(1.6)))
    s = torch.empty(B, device=x.device).uniform_(0.3, 1.2)
    grid = OUT_GRID if random.random() < p_out else random.choice(GRF_LOW_GRIDS)
    xp = apply_nuisance(x, s, gamma, grid)
    if p_apply >= 1.0:
        return xp
    sel = (torch.rand(B, device=x.device) < p_apply).view(-1, 1, 1, 1)
    return torch.where(sel, xp, x)


def perturb(x, kind, level, gen):
    if kind == "clean":
        return x
    if kind == "bias_in":
        return apply_nuisance(x, level, 1.0, 5, gen)
    if kind == "bias_out":
        return apply_nuisance(x, level, 1.0, OUT_GRID, gen)
    if kind == "radial":
        return apply_radial(x, level, gen)
    if kind == "gamma":
        return apply_nuisance(x, 0.0, level, 5, gen)
    if kind == "noise":
        return apply_rician(x, level, gen)
    raise ValueError(kind)


ROBUST_CONDS = ([("bias_in", s) for s in (0.5, 1.0, 1.5, 2.0)] +
                [("bias_out", s) for s in (0.5, 1.0, 1.5, 2.0)] +
                [("radial", s) for s in (0.5, 1.0, 1.5, 2.0)] +
                [("gamma", g) for g in (0.5, 0.75, 1.5, 2.0)] +
                [("noise", s) for s in (0.02, 0.05)])
ROBUST_KINDS = ["bias_in", "bias_out", "radial", "gamma", "noise"]


# ==========================================================================
# Models
# ==========================================================================
def _build_backbone(timm_name):
    try:
        bb = timm.create_model(timm_name, pretrained=PRETRAINED, features_only=True)
    except Exception as e:
        cands = timm.list_models(timm_name.split(".")[0].split("_")[0] + "*")[:12]
        raise RuntimeError(f"Cannot build '{timm_name}': {e}\nSimilar timm models: {cands}")
    return bb, int(bb.feature_info.channels()[-1])


class Net(nn.Module):
    def __init__(self, timm_name, nc, cfg):
        super().__init__()
        if cfg["front"] == "quotient":
            self.front = QuotientInput()
        elif cfg["front"] == "homo":
            self.front = HomomorphicInput()
        else:
            self.front = RawInput()
        self.bb, ch = _build_backbone(timm_name)
        self.drop = nn.Dropout(0.3)
        lin = nn.Linear(ch, nc)
        if cfg["lip"]:
            self.head = nn.utils.parametrizations.spectral_norm(lin)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        else:
            self.head = lin
            self.logit_scale = None
        self.last_feat = None
        self.last_embed = None

    def forward(self, x):
        z = self.front(x)                                    # fp32 front-end
        use_amp = AMP_DTYPE is not None and x.is_cuda
        with torch.autocast(device_type="cuda" if x.is_cuda else "cpu",
                            dtype=AMP_DTYPE or torch.bfloat16, enabled=use_amp):
            feat = self.bb(z)[-1]
        feat = feat.float()
        self.last_feat = feat
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        self.last_embed = pooled
        out = self.head(self.drop(pooled))
        if self.logit_scale is not None:
            out = out * torch.exp(self.logit_scale)
        return out,

    @torch.no_grad()
    def head_lipschitz(self):
        if self.logit_scale is None:
            return float("nan")
        return float(torch.exp(self.logit_scale))


def create_model(name, nc):
    timm_name, cfg = variant_cfg(name)
    return Net(timm_name, nc, cfg)


class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)[0]
        feat = self.model.last_feat
        if cls is None:
            cls = int(logits.argmax(1).item())
        grad = torch.autograd.grad(logits[:, cls].sum(), feat)[0]
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feat.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        mn, mx = cam.min(), cam.max()
        return ((cam - mn) / (mx - mn + EPS)).detach().cpu().numpy()[0, 0], logits.detach()


def to_img(t):
    return (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ==========================================================================
# Augmentation / schedule / loss
# ==========================================================================
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_r = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_r), int(W * cut_r)
    cy, cx = random.randint(0, H), random.randint(0, W)
    return (max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H),
            max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W))


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y1, y2, x1, x2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (x.size(2) * x.size(3) + EPS)
    return mixed, y, y[idx], lam


def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def get_lr(epoch, total, warmup, base, min_lr=1e-6):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smooth=LABEL_SMOOTH):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def sym_kl(l1, l2):
    p, q = F.log_softmax(l1, 1), F.log_softmax(l2, 1)
    return 0.5 * (F.kl_div(q, p, log_target=True, reduction="batchmean") +
                  F.kl_div(p, q, log_target=True, reduction="batchmean"))


def build_param_groups(model):
    bb_ids = {id(p) for p in model.bb.parameters()}
    bb = [p for p in model.parameters() if id(p) in bb_ids]
    head = [p for p in model.parameters() if id(p) not in bb_ids]
    return [{"params": bb, "lr_mult": 0.1, "weight_decay": 1e-2},
            {"params": head, "lr_mult": 1.0, "weight_decay": 1e-2}]


# ==========================================================================
# Metrics / bootstrap
# ==========================================================================
def _macro_auc(yt, ypr):
    vals = []
    for ci in range(ypr.shape[1]):
        b = (yt == ci).astype(int)
        if b.min() == b.max():
            continue
        fpr, tpr, _ = roc_curve(b, ypr[:, ci]); vals.append(auc(fpr, tpr))
    return float(np.mean(vals)) if vals else float("nan")


def ece_score(y_true, y_prob, n_bins=15):
    conf = y_prob.max(1); corr = (y_prob.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP, alpha=0.05):
    n = len(y_true); rng = np.random.RandomState(42)
    acc = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        acc["acc"].append(accuracy_score(yt, yp))
        acc["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        acc["precision_macro"].append(precision_score(yt, yp, average="macro", zero_division=0))
        acc["recall_macro"].append(recall_score(yt, yp, average="macro", zero_division=0))
        acc["auc_macro"].append(_macro_auc(yt, ypr))
    return {m: tuple(float(v) for v in np.nanpercentile(acc[m], [100 * alpha / 2, 100 * (1 - alpha / 2)]))
            for m in METRICS}


def save_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob, out_dir):
    rows, per_auc = [], []
    report = classification_report(y_true, y_pred, target_names=classes,
                                   labels=list(range(len(classes))), output_dict=True, zero_division=0)
    for ci, cls in enumerate(classes):
        r = report[cls]
        fpr, tpr, _ = roc_curve((y_true == ci).astype(int), y_prob[:, ci])
        ca = auc(fpr, tpr); per_auc.append(ca)
        cmb = confusion_matrix((y_true == ci).astype(int), (y_pred == ci).astype(int), labels=[0, 1])
        tn, fp = cmb[0, 0], cmb[0, 1]
        rows.append({"model": model_name, "seed": seed, "class": cls,
                     "precision": round(r["precision"], 4), "recall": round(r["recall"], 4),
                     "f1_score": round(r["f1-score"], 4), "support": int(r["support"]),
                     "auc": round(ca, 4), "specificity": round(tn / (tn + fp + EPS), 4)})
    acc = accuracy_score(y_true, y_pred)
    mprec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mrec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mauc = float(np.mean(per_auc)); kappa = cohen_kappa_score(y_true, y_pred)
    cis = bootstrap_ci(y_true, y_pred, y_prob)
    vm = {"acc": acc, "f1_macro": mf1, "precision_macro": mprec, "recall_macro": mrec, "auc_macro": mauc}
    pd.DataFrame([{"model": model_name, "seed": seed, "metric": m, "value": round(vm[m], 4),
                   "ci_lo": round(cis[m][0], 4), "ci_hi": round(cis[m][1], 4)} for m in METRICS]
                 ).to_csv(Path(out_dir) / f"bootstrap_ci_seed{seed}.csv", index=False)
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"test_metrics_seed{seed}.csv", index=False)
    return acc, mf1, mprec, mrec, mauc, kappa


# ==========================================================================
# Dataset audit / group split / site proxy
# ==========================================================================
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _ahash(path):
    im = Image.open(path).convert("L").resize((16, 16), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return np.packbits((a > a.mean()).flatten())            # 32 bytes


def audit_leakage(max_hamming=DUP_HAMMING):
    out_csv = SAVE_DIR / "audit_leakage.csv"
    idx_npy = SAVE_DIR / "dup_test_idx.npy"
    tr = datasets.ImageFolder(TRAIN_PATH); te = datasets.ImageFolder(TEST_PATH)
    print(f"[audit] hashing {len(tr.samples)} train + {len(te.samples)} test images ...")
    H_tr = np.stack([_ahash(p) for p, _ in tr.samples])
    H_te = np.stack([_ahash(p) for p, _ in te.samples])
    dup, same_label, mind = [], [], []
    for i0 in range(0, len(H_te), 128):
        blk = H_te[i0:i0 + 128]
        d = _POP[np.bitwise_xor(blk[:, None, :], H_tr[None, :, :])].sum(-1)
        j = d.argmin(1); dm = d.min(1)
        for k in range(len(blk)):
            mind.append(int(dm[k]))
            if dm[k] <= max_hamming:
                dup.append(i0 + k)
                same_label.append(te.targets[i0 + k] == tr.targets[j[k]])
    pd.DataFrame([{"n_train": len(H_tr), "n_test": len(H_te), "max_hamming": max_hamming,
                   "n_test_near_dup_of_train": len(dup),
                   "pct_test_near_dup": round(100 * len(dup) / len(H_te), 3),
                   "n_dup_same_label": int(np.sum(same_label)) if same_label else 0,
                   "median_min_hamming": float(np.median(mind))}]).to_csv(out_csv, index=False)
    np.save(idx_npy, np.array(dup, dtype=np.int64))
    print(f"[audit] {len(dup)} / {len(H_te)} test images are near-duplicates of training images "
          f"(Hamming<={max_hamming}). Saved {out_csv.name}, {idx_npy.name}")
    return dup


def train_groups(max_hamming=DUP_HAMMING):
    """Near-duplicate clusters inside the training set (union-find on average-hash).
    Used so that a cluster never straddles the train / validation split."""
    path = SAVE_DIR / "train_groups.npy"
    if path.exists():
        return np.load(path)
    tr = datasets.ImageFolder(TRAIN_PATH)
    H = np.stack([_ahash(p) for p, _ in tr.samples]); n = len(H)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i0 in range(0, n, 128):
        blk = H[i0:i0 + 128]
        d = _POP[np.bitwise_xor(blk[:, None, :], H[None, :, :])].sum(-1)
        ii, jj = np.where(d <= max_hamming)
        for a, b in zip(ii + i0, jj):
            if a != b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
    groups = np.array([find(i) for i in range(n)])
    np.save(path, groups)
    print(f"[groups] {n} train images -> {len(np.unique(groups))} near-duplicate clusters")
    return groups


def make_split(targets, seed):
    targets = np.asarray(targets)
    try:
        groups = train_groups()
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        tr, va = next(iter(sgkf.split(np.zeros(len(targets)), targets, groups)))
        return tr, va
    except Exception as e:
        print(f"  ! group split unavailable ({e}); falling back to stratified split")
        return train_test_split(np.arange(len(targets)), test_size=0.15,
                                stratify=targets, random_state=seed)


def site_groups(samples, min_count=15):
    sizes = []
    for p, _ in samples:
        try:
            sizes.append(Image.open(p).size)
        except Exception:
            sizes.append((0, 0))
    keys, counts = np.unique(np.array([f"{w}x{h}" for w, h in sizes]), return_counts=True)
    keep = {k for k, c in zip(keys, counts) if c >= min_count}
    return np.array([f"{w}x{h}" if f"{w}x{h}" in keep else "other" for w, h in sizes])


def site_probe(emb, labels, sites):
    """Excess site-predictability of penultimate features beyond what the label explains."""
    u = np.unique(sites)
    if len(u) < 2 or min(np.unique(sites, return_counts=True)[1]) < 5:
        return dict(site_bacc=np.nan, label_only_bacc=np.nan, site_excess=np.nan, n_sites=len(u))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=300))
    pred = cross_val_predict(clf, emb, sites, cv=skf)
    b_probe = balanced_accuracy_score(sites, pred)
    base = np.empty(len(sites), dtype=object)
    for tr, te in skf.split(emb, sites):
        for c in np.unique(labels):
            sub = sites[tr][labels[tr] == c]
            maj = pd.Series(sub).mode().iloc[0] if len(sub) else u[0]
            base[te[labels[te] == c]] = maj
    b_lab = balanced_accuracy_score(sites, base.astype(str))
    return dict(site_bacc=float(b_probe), label_only_bacc=float(b_lab),
                site_excess=float(b_probe - b_lab), n_sites=len(u))


def measure_latency(model, xu8, n=64):
    model.eval()
    x = xu8[:n].to(DEVICE).float() / 255.0
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    with torch.no_grad():
        model(x[:8]); sync()
        t = time.perf_counter()
        for i in range(0, len(x), 32):
            model(x[i:i + 32])
        sync()
    return 1000.0 * (time.perf_counter() - t) / len(x)


# ==========================================================================
# Robustness + Theorem-3 diagnostics
# ==========================================================================
@torch.no_grad()
def robustness_eval(model, xu8, y_true):
    model.eval()
    conds = [("clean", 0.0)] + ROBUST_CONDS
    gens = []
    for i in range(len(conds)):
        g = torch.Generator(device=DEVICE); g.manual_seed(DETERMINISTIC_SYNTH_SEED + i); gens.append(g)
    N = xu8.size(0)
    acc = {c: 0 for c in conds}; flip = {c: 0 for c in conds}
    dl = {c: [] for c in conds}; dd = {c: [] for c in conds}; lk = {c: [] for c in conds}
    has_phi = hasattr(model.front, "phi_last")
    for i in range(0, N, EVAL_BATCH):
        x = xu8[i:i + EVAL_BATCH].to(DEVICE).float() / 255.0
        y = torch.from_numpy(y_true[i:i + EVAL_BATCH]).to(DEVICE)
        lg0 = model(x)[0]
        phi0 = model.front.phi_last if has_phi else None
        p0 = lg0.argmax(1)
        for ci, (kind, level) in enumerate(conds):
            c = (kind, level)
            if kind == "clean":
                lg, phi = lg0, phi0
            else:
                lg = model(perturb(x, kind, level, gens[ci]))[0]
                phi = model.front.phi_last if has_phi else None
            acc[c] += int((lg.argmax(1) == y).sum()); flip[c] += int((lg.argmax(1) != p0).sum())
            dl[c].append((lg - lg0).norm(dim=1).cpu())
            if has_phi and kind != "clean":
                nm = model.front.nm_last
                dd[c].append(((phi - phi0).flatten(1).norm(dim=1) / torch.sqrt(nm)).cpu())
                if getattr(model.front, "leak_last", None) is not None:
                    lk[c].append(model.front.leak_last.cpu())
    rows, diag = [], []
    for c in conds:
        kind, level = c
        d_l = torch.cat(dl[c]).numpy()
        rows.append({"kind": kind, "level": level, "acc": acc[c] / N, "flip_rate": flip[c] / N,
                     "mean_dlogit": float(d_l.mean())})
        if dd[c]:
            d_d = torch.cat(dd[c]).numpy(); ok = d_d > 1e-6
            lh = d_l[ok] / d_d[ok] if ok.any() else np.array([np.nan])
            sp = np.nan
            if lk[c]:
                lkv = torch.cat(lk[c]).numpy()
                if np.std(lkv) > 0 and np.std(d_l) > 0:
                    sp = float(spearmanr(lkv, d_l).correlation)
            diag.append({"kind": kind, "level": level, "mean_leak_d": float(d_d.mean()),
                         "mean_dlogit": float(d_l.mean()), "Lhat_p50": float(np.nanpercentile(lh, 50)),
                         "Lhat_p95": float(np.nanpercentile(lh, 95)), "Lhat_max": float(np.nanmax(lh)),
                         "spearman_leakproxy_vs_dlogit": sp})
    return pd.DataFrame(rows), pd.DataFrame(diag)


# ==========================================================================
# Plot helper (colour-coded dual-axis training curves)
# ==========================================================================
def plot_curves(hist, title, path):
    LOSS_C, ACC_C = "#1f4e9c", "#c0392b"     # blue = loss (left axis), red = accuracy (right axis)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(hist["epoch"], hist["train_loss"], "-", color=LOSS_C, marker="o", ms=3, label="Train loss")
    ax1.plot(hist["epoch"], hist["val_loss"], "--", color=LOSS_C, marker="s", ms=3, alpha=0.65, label="Val loss")
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["train_acc"], "-", color=ACC_C, marker="^", ms=3, label="Train acc")
    ax2.plot(hist["epoch"], hist["val_acc"], "--", color=ACC_C, marker="v", ms=3, alpha=0.65, label="Val acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=LOSS_C)
    ax2.set_ylabel("Accuracy", color=ACC_C)
    ax1.tick_params(axis="y", colors=LOSS_C)
    ax2.tick_params(axis="y", colors=ACC_C)
    ax1.spines["left"].set_color(LOSS_C); ax1.spines["left"].set_linewidth(2.2)
    ax2.spines["right"].set_color(ACC_C); ax2.spines["right"].set_linewidth(2.2)
    ax1.grid(alpha=0.3)
    ln = ax1.get_lines() + ax2.get_lines()
    ax1.legend(ln, [l.get_label() for l in ln], loc="center right")
    ax1.set_title(title); plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()


# ==========================================================================
# Train + evaluate one (model, seed)
# ==========================================================================
def run_one_seed(seed, model_name):
    done_csv = SAVE_DIR / "all_results.csv"
    if done_csv.exists():
        ex = pd.read_csv(done_csv)
        done = ex[(ex["model"] == model_name) & (ex["seed"] == seed)]
        if not done.empty and (SAVE_DIR / model_name / "preds" / f"preds_seed{seed}.npz").exists():
            print(f"  =>  {model_name} seed={seed} already done - skipping")
            return done.iloc[0].to_dict()

    set_seed(seed)
    dirs = get_dirs(model_name)
    _, cfg = variant_cfg(model_name)
    print(f"\n{'=' * 72}\n  Seed {seed}  |  {model_name}\n  cfg={cfg}\n{'=' * 72}")

    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(30),
        v2.RandomAffine(degrees=0, shear=10),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    ])
    full_train = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds = datasets.ImageFolder(TEST_PATH, transform=val_tf)
    classes = full_train.classes; nc = len(classes)

    train_idx, val_idx = make_split(full_train.targets, seed)
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(full_val_ref, val_idx), batch_size=EVAL_BATCH,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=2)

    model = create_model(model_name, nc).to(DEVICE)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = optim.AdamW(build_param_groups(model), lr=LR)
    criterion = FocalLoss()
    best_loss, best_state, pat = float("inf"), None, 0
    hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}

    for epoch in range(EPOCHS):
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        cur_lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, LR)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr * g["lr_mult"]
        use_cutmix = (epoch % 2 == 0)
        cm_alpha = CUTMIX_ALPHA * 0.5 * (1 - math.cos(math.pi * min(epoch / 20.0, 1.0)))
        apply_aug = cm_alpha > 0.05

        model.train()
        tl = tc = tt = 0
        for bidx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if cfg["aug_nuis"]:
                    x = synth_random(x, p_apply=0.5)
                if cfg["cons"] and x.size(0) > 1 and random.random() < CONS_P:
                    l1 = model(x)[0]; l2 = model(synth_random(x))[0]
                    loss = 0.5 * (criterion(l1, y) + criterion(l2, y)) + CONS_LAMBDA * sym_kl(l1, l2)
                    logits = l1
                elif apply_aug and x.size(0) > 1:
                    if use_cutmix:
                        xa, ya, yb, lam = cutmix_data(x, y, cm_alpha)
                    else:
                        xa, ya, yb, lam = mixup_data(x, y, MIXUP_ALPHA)
                    logits = model(xa)[0]
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    logits = model(x)[0]; loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                tl += loss.item() * x.size(0)
                tc += (logits.argmax(1) == y).sum().item(); tt += y.size(0)
            except RuntimeError as e:
                print(f"    ! batch {bidx} skipped: {e}")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
        train_loss, train_acc = tl / max(tt, 1), tc / max(tt, 1)

        model.eval(); vl = vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                lg = model(x)[0]
                vl += criterion(lg, y).item() * x.size(0)
                vc += (lg.argmax(1) == y).sum().item(); vt += y.size(0)
        val_loss, val_acc = vl / max(vt, 1), vc / max(vt, 1)
        for k, v in zip(hist, [epoch + 1, round(train_loss, 6), round(train_acc, 6),
                               round(val_loss, 6), round(val_acc, 6)]):
            hist[k].append(v)
        print(f"  [{epoch + 1:2d}/{EPOCHS}]  Tr {train_loss:.4f}/{train_acc:.4f}  "
              f"Va {val_loss:.4f}/{val_acc:.4f}  lr={cur_lr:.2e}")
        if val_loss < best_loss:
            best_loss, best_state, pat = val_loss, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                print("  => Early stopping."); break

    model.load_state_dict(best_state); model.eval()
    pd.DataFrame(hist).to_csv(dirs["metrics"] / f"history_seed{seed}.csv", index=False)

    # ---- clean test inference (+ embeddings + uint8 cache)
    yt_l, ypr_l, emb_l, xu8_l = [], [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            xd = x.to(DEVICE)
            lg = model(xd)[0]
            ypr_l.append(F.softmax(lg, 1).cpu().numpy()); yt_l.append(y.numpy())
            emb_l.append(model.last_embed.cpu().numpy())
            xu8_l.append((x * 255).round().to(torch.uint8))
    y_true = np.concatenate(yt_l); y_prob = np.concatenate(ypr_l); y_pred = y_prob.argmax(1)
    emb = np.concatenate(emb_l); xu8 = torch.cat(xu8_l)
    np.savez(dirs["preds"] / f"preds_seed{seed}.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    ece = ece_score(y_true, y_prob)
    ms_img = measure_latency(model, xu8)

    # ---- robustness + theorem-3 diagnostics
    rob, diag = robustness_eval(model, xu8, y_true)
    rob.insert(0, "seed", seed); rob.insert(0, "model", model_name)
    rob.to_csv(dirs["metrics"] / f"robust_seed{seed}.csv", index=False)
    if not diag.empty:
        diag.insert(0, "seed", seed); diag.insert(0, "model", model_name)
        diag.to_csv(dirs["metrics"] / f"theorem3_diag_seed{seed}.csv", index=False)

    def racc(kind, lvl):
        r = rob[(rob.kind == kind) & (np.isclose(rob.level, lvl))]
        return float(r["acc"].iloc[0]) if len(r) else np.nan

    # ---- shortcut probe
    sites = site_groups(test_ds.samples)
    probe = site_probe(emb, y_true, sites)
    pd.DataFrame([dict(model=model_name, seed=seed, **probe)]).to_csv(
        dirs["metrics"] / f"site_probe_seed{seed}.csv", index=False)

    # ---- de-duplicated accuracy
    acc_dedup = np.nan
    dp = SAVE_DIR / "dup_test_idx.npy"
    if dp.exists():
        dup = np.load(dp); keep = np.setdiff1d(np.arange(len(y_true)), dup)
        if len(keep):
            acc_dedup = float(accuracy_score(y_true[keep], y_pred[keep]))

    # ---- optional external dataset
    acc_ext = np.nan
    if EXT_PATH and Path(EXT_PATH).exists():
        ext = datasets.ImageFolder(EXT_PATH, transform=val_tf)
        cmap = {i: classes.index(c) for i, c in enumerate(ext.classes) if c in classes}
        ys, ps = [], []
        with torch.no_grad():
            for x, y in DataLoader(ext, batch_size=EVAL_BATCH, shuffle=False, num_workers=2):
                lg = model(x.to(DEVICE))[0].argmax(1).cpu().numpy()
                for yy, pp in zip(y.numpy(), lg):
                    if int(yy) in cmap:
                        ys.append(cmap[int(yy)]); ps.append(int(pp))
        if ys:
            acc_ext = float(np.mean(np.array(ys) == np.array(ps)))
            pd.DataFrame([{"model": model_name, "seed": seed, "acc_ext": acc_ext, "n": len(ys)}]
                         ).to_csv(dirs["metrics"] / f"external_seed{seed}.csv", index=False)

    # ---- Grad-CAM
    if model_name in GRADCAM_MODELS:
        base = dirs["gradcam"] / f"seed{seed}"
        for c in classes:
            (base / c).mkdir(parents=True, exist_ok=True)
        tg = np.array(test_ds.targets); sel = []
        for ci in range(nc):
            ids = np.where(tg == ci)[0]
            sel.extend(ids[:GRADCAM_MAX_PER_CLASS] if GRADCAM_MAX_PER_CLASS else ids)
        eng = GradCAM(model)
        for i in sel:
            try:
                x, y = test_ds[int(i)]
                xb = x.unsqueeze(0).to(DEVICE)
                with torch.enable_grad():
                    cam, lg = eng.generate(xb, cls=int(y))
                pred = int(lg.argmax(1).item()); orig = to_img(x)
                panels = [("Original", orig)]
                if hasattr(model.front, "phi_last") and model.front.phi_last is not None:
                    ph = model.front.phi_last[0, 0].detach().cpu().numpy()
                    panels.append(("Quotient phi", ph))
                fig, axes = plt.subplots(1, len(panels) + 2, figsize=(4.2 * (len(panels) + 2), 4.5))
                for a, (t, im) in zip(axes, panels):
                    a.imshow(im, cmap=None if im.ndim == 3 else "gray"); a.set_title(t); a.axis("off")
                axes[-2].imshow(cam, cmap="jet", vmin=0, vmax=1); axes[-2].set_title("GradCAM"); axes[-2].axis("off")
                axes[-1].imshow(orig); axes[-1].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
                axes[-1].set_title(f"T:{classes[y]} P:{classes[pred]} ({'ok' if pred == y else 'wrong'})")
                axes[-1].axis("off")
                fig.suptitle(f"{model_name} - Seed {seed}", y=1.02); plt.tight_layout()
                plt.savefig(base / classes[y] / f"img_{int(i):05d}.png", dpi=100, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    ! gradcam img {i} skipped: {e}")
        model.zero_grad(set_to_none=True)

    acc, mf1, mprec, mrec, mauc, kappa = save_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob, dirs["metrics"])
    cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(nc))),
                         index=classes, columns=classes)
    cm_df.to_csv(dirs["metrics"] / f"cm_seed{seed}.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"CM - {model_name} s{seed}"); plt.tight_layout()
    plt.savefig(dirs["metrics"] / f"cm_seed{seed}.png", dpi=130, bbox_inches="tight"); plt.close()

    plot_curves(hist, f"{model_name} - Seed {seed}", dirs["curves"] / f"curves_seed{seed}.png")

    print(f"  [ok] Acc={acc:.4f} F1={mf1:.4f} AUC={mauc:.4f} ECE={ece:.4f} | "
          f"bias_in={racc('bias_in', 1.0):.3f} bias_out={racc('bias_out', 1.0):.3f} "
          f"radial={racc('radial', 1.0):.3f} gamma={racc('gamma', 0.5):.3f} "
          f"noise={racc('noise', 0.05):.3f} | site_excess={probe['site_excess']:.3f} | "
          f"{params_m:.2f}M params, {ms_img:.2f} ms/img")
    return {"seed": seed, "model": model_name, "acc": round(acc, 4), "f1_macro": round(mf1, 4),
            "precision_macro": round(mprec, 4), "recall_macro": round(mrec, 4),
            "auc_macro": round(mauc, 4), "kappa": round(kappa, 4),
            "acc_bias_in": racc("bias_in", 1.0), "acc_bias_out": racc("bias_out", 1.0),
            "acc_radial": racc("radial", 1.0),
            "acc_gamma": racc("gamma", 0.5), "acc_noise": racc("noise", 0.05),
            "site_excess": probe["site_excess"], "ece": ece, "acc_dedup": acc_dedup,
            "acc_ext": acc_ext, "head_L": model.head_lipschitz(),
            "params_M": round(params_m, 3), "ms_per_img": round(ms_img, 3)}


# ==========================================================================
# Statistics
# ==========================================================================
def load_preds(model_name, seed):
    d = np.load(SAVE_DIR / model_name / "preds" / f"preds_seed{seed}.npz")
    return d["y_true"], d["y_pred"]


def mcnemar_test_pair(y_true, pa, pb):
    ca, cb = (pa == y_true), (pb == y_true)
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb)); n = b + c
    if n == 0:
        return np.nan, np.nan, np.nan, b, c
    chi2_v = (abs(b - c) - 1.0) ** 2 / n
    return (chi2_v, float(chi2.sf(chi2_v, df=1)),
            float(min(2 * binom.cdf(min(b, c), n, 0.5), 1.0)), b, c)


def fisher_combine(pvals):
    p = np.asarray([x for x in pvals if not np.isnan(x)], dtype=float)
    if len(p) == 0:
        return np.nan
    return float(chi2.sf(-2.0 * np.sum(np.log(np.clip(p, 1e-300, 1.0))), df=2 * len(p)))


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float); out = np.full_like(p, np.nan)
    ok = ~np.isnan(p); m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj; out[ok] = tmp
    return out


def run_mcnemar_all_seeds():
    print("\n=== McNemar (image-wise, all seeds) ===")
    rows = []
    for seed in SEEDS:
        preds, y_ref = {}, None
        for m in MODELS:
            try:
                yt, yp = load_preds(m, seed); preds[m] = yp
                y_ref = yt if y_ref is None else y_ref
            except Exception:
                pass
        for m1, m2 in itertools.combinations(list(preds), 2):
            c2, pc, pe, b, c = mcnemar_test_pair(y_ref, preds[m1], preds[m2])
            rows.append({"seed": seed, "model_A": m1, "model_B": m2, "b": b, "c": c, "chi2": c2,
                         "p_chi2": pc, "p_exact": pe,
                         "winner": m1 if b > c else (m2 if c > b else "tie")})
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_DIR / "mcnemar_seedwise.csv", index=False)
    if not df.empty:
        agg = df.groupby(["model_A", "model_B"]).agg(mean_p_exact=("p_exact", "mean"),
                                                     b_sum=("b", "sum"), c_sum=("c", "sum")).reset_index()
        agg["fisher_p_exact"] = [fisher_combine(df[(df.model_A == a) & (df.model_B == b)]["p_exact"].values)
                                 for a, b in zip(agg.model_A, agg.model_B)]
        agg.to_csv(SAVE_DIR / "mcnemar_aggregated.csv", index=False)
    return df


def run_paired_stats(df):
    """n seeds paired per model pair. With n=5 exact Wilcoxon cannot go below p=0.0625;
    the paired t-test, d_z, Fisher-McNemar and the POOLED tests below carry the inference."""
    print("\n=== Wilcoxon + paired t-test + Cohen d_z ===")
    rows = []
    for metric in ALL_COLS:
        if metric not in df.columns:
            continue
        for m1, m2 in itertools.combinations(MODELS, 2):
            s1 = df[df.model == m1].sort_values("seed")[metric].values.astype(float)
            s2 = df[df.model == m2].sort_values("seed")[metric].values.astype(float)
            if len(s1) < 3 or len(s1) != len(s2) or np.isnan(s1).any() or np.isnan(s2).any():
                continue
            d = s1 - s2
            try:
                _, w_p = wilcoxon(s1, s2, zero_method="wilcox")
            except ValueError:
                w_p = np.nan
            try:
                _, t_p = ttest_rel(s1, s2)
            except Exception:
                t_p = np.nan
            dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
            rows.append({"metric": metric, "model_A": m1, "model_B": m2, "mean_A": float(s1.mean()),
                         "mean_B": float(s2.mean()), "delta_A_minus_B": float(d.mean()),
                         "cohen_dz": dz, "wilcoxon_p": w_p, "ttest_p": t_p})
    res = pd.DataFrame(rows)
    res.to_csv(SAVE_DIR / "wilcoxon_ttest_results.csv", index=False)
    return res


def run_pooled_tests(df):
    """MIQ vs each baseline FAMILY pooled over (backbone x seed) pairs (n = #backbones*#seeds).
    Caveat: pairs sharing a backbone are not fully independent; this is a power-gaining
    complement to the per-backbone tests, not a replacement. Bootstrap CI of mean diff included."""
    print("\n=== Pooled family tests (MIQ vs family, paired over backbone x seed) ===")
    d = df.copy()
    d["kind"] = d.model.str.split("_").str[0]
    d["bb"] = d.model.str.split("_").str[1]
    rows = []
    for metric in ALL_COLS:
        if metric not in d.columns:
            continue
        piv = d.pivot_table(index=["bb", "seed"], columns="kind", values=metric)
        if "MIQ" not in piv.columns:
            continue
        sub = []
        for fam in ("BASE", "AUG", "HOMO", "CONS"):
            if fam not in piv.columns:
                continue
            pr = piv[["MIQ", fam]].dropna()
            if len(pr) < 6:
                continue
            a, b = pr["MIQ"].values, pr[fam].values; diff = a - b
            try:
                w_p = float(wilcoxon(a, b)[1])
            except ValueError:
                w_p = np.nan
            try:
                t_p = float(ttest_rel(a, b)[1])
            except Exception:
                t_p = np.nan
            sd = diff.std(ddof=1)
            dz = float(diff.mean() / sd) if sd > 0 else np.nan
            rng = np.random.RandomState(0)
            bs = [diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            win = (-diff.mean() if metric in LOWER_BETTER else diff.mean()) > 0
            sub.append({"metric": metric, "family": fam, "n_pairs": len(pr),
                        "MIQ_mean": float(a.mean()), "family_mean": float(b.mean()),
                        "delta": float(diff.mean()), "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
                        "cohen_dz": dz, "wilcoxon_p": w_p, "ttest_p": t_p, "numerical_win": bool(win)})
        if sub:
            adj = holm_adjust([s["ttest_p"] for s in sub])
            for s, a_ in zip(sub, adj):
                s["ttest_p_holm"] = float(a_)
                s["verdict"] = ("SIGNIFICANT WIN" if (s["numerical_win"] and a_ < ALPHA_STAT)
                                else ("numerical win" if s["numerical_win"] else "no win"))
            rows.extend(sub)
    res = pd.DataFrame(rows)
    res.to_csv(SAVE_DIR / "pooled_family_tests.csv", index=False)
    if not res.empty:
        print(res.round(4).to_string(index=False))
    return res


def build_superiority_table(df, mc_df, wt_df):
    mc = {}
    if not mc_df.empty:
        for (a, b), g in mc_df.groupby(["model_A", "model_B"]):
            mc[(a, b)] = mc[(b, a)] = fisher_combine(g["p_exact"].values)
    wt = {}
    for r in wt_df.itertuples():
        wt[(r.metric, r.model_A, r.model_B)] = (r.wilcoxon_p, r.ttest_p, r.cohen_dz, r.delta_A_minus_B)
        wt[(r.metric, r.model_B, r.model_A)] = (r.wilcoxon_p, r.ttest_p,
                                                -r.cohen_dz if not pd.isna(r.cohen_dz) else np.nan,
                                                -r.delta_A_minus_B)
    rows = []
    for metric in ALL_COLS:
        sub = []
        lower_better = metric in LOWER_BETTER
        for m in MODELS:
            if m == PROPOSED or (metric, PROPOSED, m) not in wt:
                continue
            w_p, t_p, dz, delta = wt[(metric, PROPOSED, m)]
            win_delta = -delta if lower_better else delta
            sub.append({"Metric": METRIC_LABELS[metric], "Compared_to": m,
                        "Proposed_mean": df[df.model == PROPOSED][metric].mean(),
                        "Other_mean": df[df.model == m][metric].mean(), "Delta": delta,
                        "Cohen_dz": dz, "Wilcoxon_p": w_p, "t_test_p": t_p,
                        "McNemar_fisher_p": mc.get((PROPOSED, m), np.nan) if metric == "acc" else np.nan,
                        "_win": win_delta})
        if not sub:
            continue
        ta = holm_adjust([s["t_test_p"] for s in sub])
        for s, t in zip(sub, ta):
            s["t_test_p_holm"] = t
            s["Verdict"] = ("SIGNIFICANT WIN" if (s["_win"] > 0 and not np.isnan(t) and t < ALPHA_STAT)
                            else ("numerical win" if s["_win"] > 0 else "no win"))
            del s["_win"]
        rows.extend(sub)
    sup = pd.DataFrame(rows)
    sup.to_csv(SAVE_DIR / "superiority_table.csv", index=False)
    print(sup.round(4).to_string(index=False))
    return sup


def collect_robust_and_diag():
    R, D, S = [], [], []
    for m in MODELS:
        for seed in SEEDS:
            for pref, lst in (("robust", R), ("theorem3_diag", D), ("site_probe", S)):
                p = SAVE_DIR / m / "metrics" / f"{pref}_seed{seed}.csv"
                if p.exists():
                    lst.append(pd.read_csv(p))
    out = {}
    for name, lst in (("robust_all", R), ("theorem3_diag_all", D), ("site_probe_all", S)):
        if lst:
            out[name] = pd.concat(lst, ignore_index=True)
            out[name].to_csv(SAVE_DIR / f"{name}.csv", index=False)
    return out


# ==========================================================================
# Plots
# ==========================================================================
def save_plots(df, extra):
    pal = sns.color_palette("tab20", max(len(MODELS), 3))
    for metric in ALL_COLS:
        if metric not in df.columns:
            continue
        ms, ss, lb = [], [], []
        for m in MODELS:
            r = df[df.model == m]
            if not r.empty:
                ms.append(r[metric].mean()); ss.append(r[metric].std() if len(r) > 1 else 0.0); lb.append(short(m))
        if not ms or np.all(np.isnan(ms)):
            continue
        fig, ax = plt.subplots(figsize=(14, 6.5))
        x = np.arange(len(ms))
        ax.bar(x, ms, yerr=np.nan_to_num(ss), color=pal[:len(ms)], edgecolor="k", capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(lb, rotation=55, ha="right", fontsize=9)
        ax.set_title(f"{METRIC_LABELS[metric]} (mean +/- std, n={len(SEEDS)} seeds)")
        ax.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(SAVE_DIR / f"barplot_{metric}.png", dpi=150, bbox_inches="tight"); plt.close()

    if "robust_all" in extra:
        R = extra["robust_all"]
        for b in BACKBONE_NAMES:
            fig, axes = plt.subplots(1, len(ROBUST_KINDS), figsize=(5.2 * len(ROBUST_KINDS), 5))
            for ax, kind in zip(axes, ROBUST_KINDS):
                for k in KINDS:
                    sub = R[(R.model == f"{k}_{b}") & (R.kind.isin([kind, "clean"]))]
                    if sub.empty:
                        continue
                    clean = sub[sub.kind == "clean"]["acc"].mean()
                    g = sub[sub.kind == kind].groupby("level")["acc"].agg(["mean", "std"]).reset_index()
                    lv = [0.0 if kind != "gamma" else 1.0] + list(g["level"])
                    ax.errorbar(lv, [clean] + list(g["mean"]), yerr=[0] + list(g["std"].fillna(0)),
                                marker="o", capsize=3, label=k)
                ax.set_title(kind); ax.set_xlabel("level"); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3)
            axes[0].legend(); fig.suptitle(f"Robustness - {b}"); plt.tight_layout()
            plt.savefig(SAVE_DIR / f"robustness_{b}.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  [ok] plots saved")


# ==========================================================================
# Smoke test (no dataset, no pretrained weights)
# ==========================================================================
def smoke_test():
    global PRETRAINED
    PRETRAINED = False
    torch.manual_seed(0); random.seed(0)
    print("== MIQ smoke test ==")
    n = 56
    # ---- T2: KL <= DCT << random leakage on held-out in-family fields
    leak = {}
    for kind in ("kl", "dct", "rand"):
        U = build_basis(kind, K_MODES, n)
        assert torch.allclose(U @ U.T, torch.eye(K_MODES), atol=1e-4), "basis not orthonormal"
        g = torch.Generator().manual_seed(123)
        Bf = _gauss_field_vectors(60, GRF_LOW_GRIDS, n, g)
        Bf = Bf - Bf.mean(1, keepdim=True)
        res = Bf - (Bf @ U.T) @ U
        leak[kind] = float((res.norm(dim=1) ** 2 / Bf.norm(dim=1) ** 2).mean())
    print(f"  [T2] mean leakage  KL={leak['kl']:.4f}  DCT={leak['dct']:.4f}  random={leak['rand']:.4f}")
    assert leak["kl"] <= leak["dct"] * 1.02 and leak["dct"] < leak["rand"], "Theorem-2 check failed"

    # ---- T1: exact invariance / maximality on a synthetic 'brain'
    q = QuotientInput(n=n, K=K_MODES, basis="kl")
    yy, xx = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(), indexing="ij")
    mask = ((((yy - n / 2) / (0.42 * n)) ** 2 + ((xx - n / 2) / (0.36 * n)) ** 2) < 1).float()
    img = (0.4 + 0.5 * torch.rand(1, 1, n, n)) * mask
    x1 = img.expand(1, 3, n, n).contiguous()
    c = torch.randn(K_MODES) * 0.25
    b = (c @ q.U).view(1, 1, n, n); gam = 1.37
    x2 = (img ** gam * torch.exp(b)); x2 = (x2 / x2.max()).expand(1, 3, n, n).contiguous()
    q(x1); p1 = q.phi_last.clone(); q(x2); p2 = q.phi_last.clone()
    err = float((p1 - p2).abs().max())
    img3 = (0.4 + 0.5 * torch.rand(1, 1, n, n)) * mask
    q(img3.expand(1, 3, n, n).contiguous()); p3 = q.phi_last.clone()
    diff = float((p1 - p3).abs().max())
    print(f"  [T1] invariance err={err:.2e}   different-content distance={diff:.3f}")
    assert err < 2e-2 and diff > 10 * err, "Theorem-1 check failed"

    # ---- T3: bound  d <= 2 ||P b_tail|| / (gamma ||P z||)
    z1, m1 = q.log_and_mask(x1)
    bt = bias_field(1, n, 12, torch.device("cpu")).flatten(1) * 0.3
    zt = (gam * z1 + bt) * m1
    u = q.project(z1, m1); v = q.project(bt, m1)
    ph_a = u / torch.sqrt((u ** 2).sum() / m1.sum())
    up = q.project(zt, m1); ph_b = up / torch.sqrt((up ** 2).sum() / m1.sum())
    d = float((ph_a - ph_b).norm() / torch.sqrt(m1.sum()))
    bound = float(2 * v.norm() / (gam * u.norm()))
    print(f"  [T3] leakage d={d:.3f} <= bound={bound:.3f}")
    assert d <= bound + 1e-6, "Theorem-3 bound violated"

    # ---- nuisance / ECE / split utilities
    xb = torch.rand(2, 3, 64, 64)
    for k, lv in ROBUST_CONDS[::3]:
        assert perturb(xb, k, lv, None).shape == xb.shape
    assert perturb(xb, "radial", 1.0, None).shape == xb.shape
    assert synth_random(xb).shape == xb.shape
    pr = np.eye(4)[np.random.randint(0, 4, 200)]
    assert abs(ece_score(np.argmax(pr, 1), pr)) < 1e-9, "ECE of perfect predictor must be 0"

    # ---- architectures
    names = []
    for bn in BACKBONE_REGISTRY:
        names += [f"{k}_{bn}" for k in KINDS]
    for nm in names:
        m = create_model(nm, 4)
        npar = sum(p.numel() for p in m.parameters()) / 1e6
        m.train(); x = torch.rand(4, 3, IMG_SIZE, IMG_SIZE)
        out = m(x)[0]; (out.logsumexp(1).mean()).backward()
        m.eval()
        with torch.no_grad():
            o1 = m(x)[0]
        cam = GradCAM(m).generate(x[:1], cls=0)[0]
        assert o1.shape == (4, 4) and torch.isfinite(o1).all() and cam.shape == (IMG_SIZE, IMG_SIZE)
        note = ""
        if nm.startswith("MIQ_"):
            with torch.no_grad():
                yy2, xx2 = torch.meshgrid(torch.arange(IMG_SIZE).float(), torch.arange(IMG_SIZE).float(), indexing="ij")
                mk = ((((yy2 - 112) / 90) ** 2 + ((xx2 - 112) / 80) ** 2) < 1).float()
                base = (0.4 + 0.5 * torch.rand(2, 1, IMG_SIZE, IMG_SIZE)) * mk
                cc = torch.randn(2, K_MODES) * 0.2
                bb_ = torch.einsum("bk,kp->bp", cc, m.front.U).view(2, 1, IMG_SIZE, IMG_SIZE)
                xa = base.expand(2, 3, IMG_SIZE, IMG_SIZE).contiguous()
                xb2 = base ** 1.3 * torch.exp(bb_)
                xb2 = (xb2 / xb2.amax((1, 2, 3), keepdim=True)).expand(2, 3, IMG_SIZE, IMG_SIZE).contiguous()
                gap = float((m(xa)[0] - m(xb2)[0]).abs().max())
            note = f"  logit gap under in-span nuisance={gap:.2e}"
        print(f"  {nm:34s} params={npar:6.2f}M out={tuple(o1.shape)}{note}")
    print("== smoke test PASSED ==")


def check_backbones():
    for k, v in BACKBONE_REGISTRY.items():
        try:
            m = timm.create_model(v, pretrained=False, features_only=True)
            x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            print(f"  OK  {k:14s} {v:36s} channels={m.feature_info.channels()} "
                  f"last_map={tuple(m(x)[-1].shape[2:])} params={sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
        except Exception as e:
            print(f"  FAIL {k}: {e}")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke_test(); sys.exit(0)
    if "--check-backbones" in sys.argv:
        check_backbones(); sys.exit(0)
    if "--audit" in sys.argv:
        audit_leakage(); sys.exit(0)

    print("\n" + "=" * 72)
    print("  MIQ-Net v2: Maximal-Invariant Quotient Network")
    print(f"  Proposed : {PROPOSED}   Backbones: {BACKBONE_NAMES}")
    print(f"  Seeds    : {SEEDS}   Device: {DEVICE}   Epochs: {EPOCHS}   AMP: {AMP_DTYPE}")
    print(f"  Models   : {len(MODELS)}")
    print("=" * 72)
    save_run_config()

    if RUN_AUDIT and not (SAVE_DIR / "dup_test_idx.npy").exists():
        try:
            audit_leakage()
        except Exception as e:
            print(f"  ! audit skipped: {e}")
    try:
        train_groups()
    except Exception as e:
        print(f"  ! train grouping skipped: {e}")

    all_results = []
    for model_name in MODELS:
        for seed in SEEDS:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            all_results.append(run_one_seed(seed, model_name))
            (pd.DataFrame(all_results).drop_duplicates(subset=["model", "seed"])
               .reset_index(drop=True).to_csv(SAVE_DIR / "all_results.csv", index=False))

    df = (pd.read_csv(SAVE_DIR / "all_results.csv")
            .drop_duplicates(subset=["model", "seed"]).reset_index(drop=True))
    df = df[df.model.isin(MODELS)].reset_index(drop=True)   # drop stale rows from older runs
    cols = [c for c in ALL_COLS + ["params_M", "ms_per_img"] if c in df.columns]
    summary = df.groupby("model")[cols].agg(["mean", "std"]).round(4)
    summary.to_csv(SAVE_DIR / "summary_mean_std.csv")
    print("\n=== Mean +/- Std ===")
    print(summary)

    mc_df = run_mcnemar_all_seeds()
    wt_df = run_paired_stats(df)
    build_superiority_table(df, mc_df, wt_df)
    run_pooled_tests(df)
    extra = collect_robust_and_diag()
    save_plots(df, extra)
    print(f"\n{'=' * 72}\n  All outputs saved to: {SAVE_DIR.resolve()}\n{'=' * 72}")
