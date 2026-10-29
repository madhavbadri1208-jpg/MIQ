1. Overview
This document is a professional, results-oriented README for MIQ-Net v2 — a Maximal-Invariant Quotient Network for brain-tumour MRI classification (glioma, meningioma, notumor, pituitary). It summarises the experimental setup, the empirical findings across three lightweight backbones, the perturbation-specific and nuisance-robustness results, the leakage-bound diagnostics, calibration and deduplicated evaluation, computational characteristics, and the statistical analyses (Wilcoxon, bootstrap, McNemar), along with a qualitative Grad-CAM analysis.

The framing is deliberately honest and backbone-specific: MIQ-Net does not uniformly outperform baselines. It shows the strongest, most consistent benefits on MobileNetV3-Large, mixed behaviour on FastViT-T8, and generally underperforms on EdgeNeXt-XXS for clean accuracy. All statistical claims are reported with the appropriate caveats (n = 5 seeds; exact Wilcoxon floor at p ≈ 0.0625; multiple-testing corrections applied).

2. Experimental Setup
2.1 Backbones and models
Three lightweight backbones were tested:

FastViT-T8 — designated primary backbone (best overall accuracy across seeds)

MobileNetV3-Large — lightweight convolutional architecture; strongest MIQ benefit

EdgeNeXt-XXS — lightweight hybrid; relatively underperforming backbone

Five configurations were compared on each backbone:

BASE — plain ERM

AUG — ERM with nuisance augmentation

HOMO — classical homomorphic log-domain filtering + ERM

CONS — raw input + orbit-consistency loss (control for the MIQ training recipe)

MIQ — proposed quotient + orbit-consistency + Lipschitz head

2.2 Seeds and protocol
Five random seeds: 42, 123, 456, 789, 2024

Seed 456 used consistently for all illustrative visualisations (curves, confusion matrices, Grad-CAM) to allow direct visual comparison

All reported means and standard deviations are aggregated across the five seeds

Test set: N = 2,414; de-duplicated test set: N = 759

2.3 Training and validation curves
Training and validation loss/accuracy curves were analysed for FastViT-T8, MobileNetV3-Large, and EdgeNeXt-XXS (seed 456) to assess convergence, optimisation stability, and the train–validation gap. The curves are treated as complementary evidence of optimisation behaviour, not as independent evidence of generalisation or statistical superiority.

3. Conventional Classification Performance
Table 7 (from the results) summarises clean-test performance. Mean ± std across five seeds:

3.1 EdgeNeXt-XXS
Model	Accuracy	Macro F1	Macro AUC	Kappa
BASE	0.8585 ± 0.0187	0.8584 ± 0.0195	0.9744 ± 0.0033	0.8111 ± 0.0242
AUG	0.8210 ± 0.0160	0.8250 ± 0.0147	0.9633 ± 0.0030	0.7617 ± 0.0205
HOMO	0.8180 ± 0.0170	0.8174 ± 0.0172	0.9607 ± 0.0056	0.7579 ± 0.0218
CONS	0.8449 ± 0.0251	0.8472 ± 0.0256	0.9712 ± 0.0048	0.7933 ± 0.0324
MIQ	0.8054 ± 0.0148	0.8089 ± 0.0139	0.9519 ± 0.0065	0.7414 ± 0.0192
Observation: BASE is clearly best on EdgeNeXt-XXS. MIQ underperforms BASE by ~5.3 pp in accuracy and ~1.7 pp in macro F1.

3.2 FastViT-T8
Model	Accuracy	Macro F1	Macro AUC	Kappa
BASE	0.9134 ± 0.0081	0.9135 ± 0.0087	0.9857 ± 0.0025	0.8838 ± 0.0109
AUG	0.8840 ± 0.0169	0.8849 ± 0.0174	0.9791 ± 0.0038	0.8449 ± 0.0226
HOMO	0.8668 ± 0.0059	0.8651 ± 0.0059	0.9733 ± 0.0033	0.8225 ± 0.0078
CONS	0.9055 ± 0.0099	0.9069 ± 0.0096	0.9864 ± 0.0018	0.8736 ± 0.0131
MIQ	0.8930 ± 0.0106	0.8929 ± 0.0104	0.9818 ± 0.0026	0.8572 ± 0.0139
Observation: BASE has the highest accuracy, macro F1 and Kappa. CONS marginally beats BASE on macro AUC. MIQ is second on accuracy, ~2.0 pp below BASE.

3.3 MobileNetV3-Large
Model	Accuracy	Macro F1	Macro AUC	Kappa
BASE	0.8396 ± 0.0135	0.8377 ± 0.0141	0.9609 ± 0.0059	0.7865 ± 0.0178
AUG	0.8111 ± 0.0043	0.8104 ± 0.0043	0.9561 ± 0.0041	0.7495 ± 0.0057
HOMO	0.8418 ± 0.0067	0.8406 ± 0.0060	0.9709 ± 0.0019	0.7894 ± 0.0088
CONS	0.8336 ± 0.0107	0.8328 ± 0.0106	0.9591 ± 0.0053	0.7790 ± 0.0140
MIQ	0.8496 ± 0.0073	0.8514 ± 0.0070	0.9700 ± 0.0031	0.7996 ± 0.0097
Observation: MIQ achieves the highest clean accuracy (84.96%), macro F1 (85.14%), macro precision (85.24%), macro recall (86.29%) and Kappa (0.7996) on MobileNetV3-Large. HOMO marginally leads on macro AUC (97.09% vs 97.00%). The MIQ accuracy advantage over BASE is +1.01 pp.

3.4 Summary of clean performance
Backbone-dependent behaviour. MIQ clean-classification benefit is not uniform.

EdgeNeXt-XXS: BASE > MIQ

FastViT-T8: BASE > MIQ (CONS edges BASE on AUC)

MobileNetV3-Large: MIQ best on accuracy/F1/precision/recall/Kappa

Differences are descriptive and must be interpreted with paired statistical tests (Section 7).

4. Perturbation-Specific Performance
Table 8 reports accuracy under five perturbation families at level 1.0 (plus gamma and noise).

4.1 Key observations
FastViT-T8: MIQ is best on gamma (88.19%) and noise (72.20%). CONS is best on bias_in (86.09%) and radial (89.63%). AUG is best on bias_out (70.76%).

MobileNetV3-Large: MIQ is best on bias_in (78.17%), radial (84.06%), gamma (82.42%), and noise (65.97%). AUG is best on bias_out (65.24%).

EdgeNeXt-XXS: CONS is best on bias_in (80.78%) and radial (82.37%); AUG is best on bias_out (64.66%) and noise (60.57%).

4.2 Interpretive note
The relative ranking is backbone- and perturbation-dependent. MIQ shows a selective, not universal advantage. These metrics are descriptive and must be read alongside the definitions and severity levels of the perturbations, and the evaluation protocol.

5. Nuisance-Robustness Evaluation (MobileNetV3-Large)
Table 9 reports severity-resolved accuracy for BASE and MIQ on MobileNetV3-Large (five-seed mean ± std; Δ in percentage points).

5.1 Selected results
Perturbation	Level	BASE (%)	MIQ (%)	Δ (pp)
Clean	0	83.96 ± 1.35	84.96 ± 0.73	+1.00
Bias-in	1.0	45.12 ± 2.07	78.17 ± 0.84	+33.05
Bias-in	2.0	28.26 ± 3.20	51.91 ± 2.14	+23.65
Bias-out	0.5	46.16 ± 2.69	72.92 ± 2.17	+26.76
Bias-out	2.0	20.32 ± 0.16	23.58 ± 2.45	+3.26
Radial	1.0	69.66 ± 1.16	84.06 ± 0.68	+14.40
Radial	2.0	54.86 ± 1.47	83.10 ± 1.27	+28.24
Gamma	0.5	45.22 ± 2.06	82.42 ± 0.90	+37.20
Gamma	2.0	69.25 ± 2.99	82.15 ± 1.23	+12.89
Noise	0.02	20.20 ± 0.02	81.01 ± 1.24	+60.81
Noise	0.05	20.17 ± 0.00	65.97 ± 4.10	+45.79
5.2 Key findings
MIQ > BASE at every level of every non-clean perturbation on MobileNetV3-Large.

The largest gaps occur under noise (BASE collapses to ~20%), bias-in at mid-to-high severity, and gamma at extrapolated levels.

The bias-out gap narrows at high severity (only +3.26 pp at level 2.0), consistent with the theory: out-of-span bias is not removed exactly.

The gamma accuracy curve is non-monotonic — severity is not equivalent to difficulty.

The large noise gap warrants independent verification of the noise generation, class distribution, and evaluation pipeline before attributing it solely to the representation.

These results are descriptive and limited to MobileNetV3-Large, this dataset, this perturbation implementation, and the tested severity range. Paired statistical testing is required for significance claims.

6. Empirical Evaluation of the Leakage Bound
Table 10 reports the leakage proxy and mean logit change for MIQ-MobileNetV3-Large across five seeds.

6.1 Key observations
bias_in: leakage proxy rises from 0.778 (level 0.5) to 2.874 (level 2.0); mean logit change rises from 0.612 to 1.857.

bias_out: leakage proxy rises from 1.011 to 5.820; mean logit change rises from 1.123 to 2.428 — steeper, as expected for out-of-span bias.

radial: leakage proxy rises from 0.587 to 1.101; logit change from 0.427 to 0.815 — less pronounced.

gamma: non-monotonic; logit change between 0.556 and 0.825.

noise: logit change rises from 1.170 to 1.673; leakage proxy remains near 0.63–0.65.

6.2 Interpretation
Both leakage and logit sensitivity increase with perturbation strength for in-family and out-of-span bias, consistent with the Theorem 3 leakage bound. The relationship is not claimed to be universal across perturbation families or across backbones.

7. Calibration and Deduplicated Evaluation
Table 11 reports ECE, site_excess, and de-duplicated accuracy.

7.1 Calibration (lower ECE is better)
EdgeNeXt-XXS: MIQ lowest (0.1510); BASE worst (0.1887). MIQ better calibrated than BASE.

FastViT-T8: CONS lowest (0.1563); MIQ (0.1657); BASE worst (0.1960). MIQ better than BASE.

MobileNetV3-Large: CONS lowest (0.1534); MIQ (0.1908); BASE (0.1721). MIQ is worse calibrated than BASE here.

Conclusion: MIQ calibration benefit is backbone-dependent and not universal.

7.2 Site-excess (lower is better)
Site-excess is broadly similar across configurations within each backbone. MIQ does not show a consistent site-shortcut advantage.

7.3 De-duplicated accuracy
EdgeNeXt-XXS: BASE 74.81% > MIQ 67.64% (−7.17 pp).

FastViT-T8: BASE 84.14% > CONS 83.93% > MIQ 81.34% (−2.79 pp).

MobileNetV3-Large: MIQ 74.62% > HOMO 72.49% > BASE 71.73% (+2.90 pp).

Conclusion: De-duplicated evaluation does not show a general MIQ advantage; it is backbone-dependent.

8. Computational Characteristics
Table 12 reports head Lipschitz constant (head_L), trainable parameters (params_M), and inference latency (ms_per_img).

8.1 Parameters
Identical across configurations for the same backbone: 1.159 M (EdgeNeXt-XXS), 3.173 M (FastViT-T8), 2.976 M (MobileNetV3-Large).

MIQ adds no trainable parameters beyond the base backbone + head.

8.2 Head Lipschitz constant
MIQ reports head_L ≈ 4.45 (EdgeNeXt-XXS), 4.76 (FastViT-T8), 4.23 (MobileNetV3-Large) — an explicit, finite bound.

8.3 Latency
FastViT-T8: MIQ 0.2262 ms/img (lowest of the five configurations).

MobileNetV3-Large: MIQ 0.2226 ms/img (lowest of the five configurations).

EdgeNeXt-XXS: MIQ 0.5962 ms/img — lower than HOMO and CONS, but higher than BASE and AUG.

8.4 Caveat
Latency shows large run-to-run variability in several configurations (e.g., BASE EdgeNeXt-XXS std ≈ 0.79 ms). A strong efficiency claim requires controlled benchmarking: identical hardware, batch sizes, input dimensions, warm-up, and timing protocol. Equal parameter counts do not imply equal computational complexity.

9. Statistical Analysis
9.1 Wilcoxon signed-rank tests (paired across five seeds)
MobileNetV3-Large was chosen for the primary statistical analysis (most consistent numerical MIQ gain). Table 13 summarises MIQ vs BASE per backbone with Holm correction.

Clean accuracy (MobileNetV3-Large): MIQ 84.96% vs BASE 83.96% (+1.01 pp), not significant after Holm.

De-duplicated accuracy (MobileNetV3-Large): MIQ 74.62% vs BASE 71.73% (+2.90 pp), not significant after Holm.

Gamma robustness (MobileNetV3-Large): MIQ 82.42% vs BASE 45.22% (+37.20 pp), Wilcoxon p = 0.0625, Holm-adjusted p = 0.1875.

Noise robustness (MobileNetV3-Large): MIQ 65.97% vs BASE 20.17% (+45.79 pp), Wilcoxon p = 0.0625, Holm-adjusted p = 0.1875.

FastViT-T8 and EdgeNeXt-XXS: MIQ de-duplicated accuracy is worse than BASE; gamma and noise robustness are descriptively higher but not significant after Holm.

Important caveat: With n = 5 seeds, the exact Wilcoxon signed-rank test cannot produce p < 0.0625. The results are therefore descriptive improvements, not statistically significant ones after Holm correction, and cannot be interpreted as equivalence either.

9.2 Bootstrap analysis (MobileNetV3-Large, per seed)
Table 14 reports per-seed 95% bootstrap CIs for accuracy, macro F1, precision, recall, and AUC for MIQ and BASE (MobileNetV3-Large).

MIQ accuracy CIs: 0.8426–0.8612 across seeds.

BASE accuracy CIs: 0.8210–0.8546 across seeds.

MIQ macro AUC CIs: 0.9674–0.9746; BASE macro AUC CIs: 0.9519–0.9669.

The per-seed CIs overlap in several seeds, and bootstrap CIs reflect within-seed sampling uncertainty, not between-seed variability. Bootstrap analysis alone cannot establish MIQ superiority.

9.3 McNemar tests (MobileNetV3-Large)
Table 15 reports McNemar exact tests on full (N = 2,414) and de-duplicated (N = 759) test sets, across five seeds, with Holm correction.

Full test set (N = 2,414)
Seed	MIQ acc	BASE acc	b	c	Exact p	Holm p	Favoured	Significant
42	85.13	85.46	117	125	0.6528	1.0000	BASE	No
123	86.12	83.97	161	109	0.00186	0.01486	MIQ	Yes
456	84.84	85.00	128	132	0.8525	1.0000	BASE	No
789	84.47	83.26	154	125	0.0935	0.3740	MIQ	No
2024	84.26	82.10	183	131	0.00393	0.02358	MIQ	Yes
MIQ is significantly better on 2/5 full-test seeds after Holm correction.

De-duplicated test set (N = 759)
Seed	MIQ acc	BASE acc	b	c	Exact p	Holm p	Favoured	Significant
42	74.97	74.31	61	56	0.7117	1.0000	MIQ	No
123	74.70	71.41	80	55	0.0385	0.2703	MIQ	No
456	75.49	73.25	71	54	0.1521	0.6085	MIQ	No
789	74.31	69.57	90	54	0.00339	0.03054	MIQ	Yes
2024	73.65	70.09	89	62	0.0340	0.2703	MIQ	No
MIQ is favoured in 5/5 de-duplicated seeds, but only 1/5 is significant after Holm correction.

Conclusion: The evidence is promising but not uniformly significant. MobileNetV3-Large remains the most favourable case.

10. Grad-CAM Analysis (Qualitative)
Grad-CAM panels (Figure 6, seed 456) compare baseline MobileNetV3-Large and MIQ-Net with the same backbone. Each row shows the original MRI, the baseline Grad-CAM, the MIQ quotient representation φ, and the MIQ-Net Grad-CAM.

10.1 Per-class observations
Glioma: baseline shows broad central activation; MIQ shows a more focal activation near the lesion. Different spatial feature distributions are inferred — but one example is insufficient to claim improved localisation or causal attribution.

Meningioma: both models show significant activation near the visible lesion in the upper region — illustrating preserved lesion-associated activation and the potential for similar explanations.

Pituitary: MIQ shows stronger activation in lower and central regions; the difference may reflect different features but cannot be anatomically adjudicated from heatmaps alone. Activation outside the apparent tumour is not proof of wrong reasoning.

No tumour: both models show distributed activation with no focal tumour target, consistent with the absence of a lesion target.

10.2 Overall
The spatial activation distributions differ between baseline and MIQ-Net, with variability across the four classes.

These visualisations do not establish improved explainability, causal invariance, or better tumour localisation. They are illustrative examples, not evidence of consistent behaviour across seeds or the full test set.

11. Key Findings — Consolidated
Conventional classification:

MIQ best on MobileNetV3-Large (+1.01 pp accuracy, +1.37 pp macro F1 over BASE).

MIQ worse than BASE on EdgeNeXt-XXS (−5.31 pp) and FastViT-T8 (−2.04 pp).

Behaviour is backbone-dependent, not universal.

Perturbation robustness:

MIQ shows selective but consistent gains on gamma and noise across backbones.

On MobileNetV3-Large, MIQ > BASE at every severity of every perturbation; largest gaps under noise and gamma.

bias_out gap narrows at high severity, consistent with the out-of-span limitation of the theory.

Leakage bound (Theorem 3):

Empirical leakage proxy and mean logit change both increase with perturbation strength, consistent with the bound.

Relationship is not claimed universal across perturbation families.

Calibration:

MIQ lower ECE than BASE on EdgeNeXt-XXS and FastViT-T8; higher on MobileNetV3-Large.

Backbone-dependent.

Deduplicated accuracy:

MIQ best on MobileNetV3-Large (+2.90 pp); worse on the other two backbones.

No general MIQ advantage.

Computational cost:

No additional trainable parameters from MIQ.

Lowest latency on FastViT-T8 and MobileNetV3-Large; intermediate on EdgeNeXt-XXS.

Efficiency claim requires controlled benchmarking due to run-to-run variability.

Statistics:

McNemar: significant MIQ > BASE on 2/5 full-test seeds and 1/5 de-duplicated seeds (MobileNetV3-Large), Holm-corrected.

Wilcoxon (n = 5): descriptive improvements in gamma and noise robustness (up to +45.79 pp) but not significant after Holm correction — exact Wilcoxon floor at p = 0.0625.

Bootstrap: per-seed 95% CIs reported; CIs overlap in several seeds; within-seed uncertainty only.

Grad-CAM:

Activation patterns differ but do not, by themselves, establish improved explainability or causal invariance.

12. Limitations and Honest Caveats
Backbone-dependent benefit. MIQ-Net does not uniformly improve over baselines; results are specific to each backbone and must not be extrapolated.

Limited statistical power. n = 5 seeds; exact Wilcoxon cannot yield p < 0.0625; multiple-comparison corrections are applied throughout. Non-significance should not be interpreted as equivalence.

Out-of-span bias not removed exactly. bias_out gaps narrow at high severity, consistent with theory; this is a limitation, not a bug.

Noise robustness results require independent verification. The very large MIQ-vs-BASE gap under noise warrants checking of noise generation, class distribution, and evaluation pipeline.

Latency measurements are variable. Strong efficiency claims require controlled benchmarking.

Seed 456 visualisations are illustrative. They are not evidence of population-level behaviour.

Grad-CAM is qualitative. It illustrates prediction-associated activation patterns, not explainability or causal invariance.

13. Conclusion
MIQ-Net v2 provides a provable maximal-invariant quotient front-end that, in the tested protocol, yields:

Selective, backbone-dependent clean-accuracy improvements (strongest on MobileNetV3-Large: +1.01 pp).

Substantial descriptive robustness gains under gamma and noise perturbations (+37.20 pp and +45.79 pp on MobileNetV3-Large), with partial statistical significance under McNemar (2/5 full-test seeds; 1/5 de-duplicated seeds, Holm-corrected).

No additional trainable parameters and competitive inference latency on the primary backbones.

Theoretically consistent leakage and logit-sensitivity behaviour (Theorem 3 validated empirically).

Honest reporting that the benefits are not universal across backbones, perturbations, calibration, or de-duplicated evaluation.

The study's contribution is therefore best framed as a provable, provably bounded, parameter-neutral quotient representation that delivers measurable robustness benefits in specific, well-characterised settings — with all limitations stated explicitly and all inferential claims corrected for multiple comparisons.

