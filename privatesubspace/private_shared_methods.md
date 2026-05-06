## Decomposition of V1 variance into shared and private subspaces

To test whether labeled projection neurons express a disproportionate fraction of V1→PM communication-channel variance, we decomposed V1 population activity into a *shared* subspace (dimensions predictive of PM) and a *private* subspace (dimensions predictive of other V1 neurons but not of PM), and quantified each neuron's alignment with both.

**Subspace definition.** For each session and stimulus orientation, we drew four non-overlapping V1 subpopulations (*N* = 20 each): two unlabeled samples (X1, X2) to define subspaces, a held-out unlabeled sample (X3), and a labeled-neuron sample (X4), plus an unlabeled PM sample (Y). Subspaces were defined from X1 and X2 exclusively, keeping X3 and X4 as unbiased test populations.

The shared subspace was identified by RRR from X1 to Y. The $k_{\text{shared}}$ leading left singular vectors of $\hat{\mathbf{Y}} = \mathbf{X}_1 \hat{\mathbf{B}}_{\text{shared}}$ form an orthonormal basis $\mathbf{Q}_{\text{share}} \in \mathbb{R}^{T \times k_{\text{share}}}$ in trial-time space ($T$ = trials $\times$ time bins).

The private subspace was identified by RRR from X1 to X2. The $k_{\text{private}}$ leading left singular vectors of $\hat{\mathbf{X}}_2$ were orthogonalized against $\mathbf{Q}_{\text{share}}$ (shared component removed, followed by QR re-orthonormalization), yielding $\mathbf{Q}_{\text{priv}} \in \mathbb{R}^{T \times k_{\text{private}}}$ spanning only within-V1 predictive variance absent from the shared channel. Ranks were determined from the cross-validated $R^2$ elbow on full (non-subsampled) populations; sessions were excluded if $k_{\text{private}} \leq k_{\text{shared}}$.

**Per-neuron alignment.** For each held-out neuron $n$ with z-scored activity vector $\mathbf{x}_n \in \mathbb{R}^T$, we computed the fraction of variance aligned with each subspace:

$$\alpha_n = \frac{\| \mathbf{Q}_{\text{share}}^{\top} \mathbf{x}_n \|^2}{\| \mathbf{x}_n \|^2}, \qquad \beta_n = \frac{\| \mathbf{Q}_{\text{priv}}^{\top} \mathbf{x}_n \|^2}{\| \mathbf{x}_n \|^2}$$

Both quantities are bounded in [0, 1] and require no matrix inversion, reducing to sums of squared dot products against orthonormal basis vectors. We summarized relative specialization with a selectivity index:

$$s_n = \frac{\alpha_n}{\alpha_n + \beta_n}$$

Values near 1 indicate dominance of the shared channel; values near 0 indicate dominance of private within-V1 variance. Metrics were averaged across 100 subsampling iterations, stimulus conditions, and neurons within each session to yield one value per session per cell type.

**Statistical testing.** Session-level means of $\alpha$, $\beta$, and $s$ were compared between labeled (X4) and held-out unlabeled (X3) neurons using two-sided Wilcoxon signed-rank tests, with Bonferroni correction across the three metrics.
