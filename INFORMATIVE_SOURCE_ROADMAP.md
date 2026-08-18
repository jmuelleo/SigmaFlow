# Informative Source Distributions for SigmaFlow — Research Roadmap

**Status:** planning document. No experiment below has been run.
**Date:** 2026-08-17
**Gate for everything here:** IS-1 (§7). If IS-1 fails, IS-2…IS-6 are not run.

---

## 1. The current source, component by component

SigmaFlow's source factorises completely — across degrees of freedom and
across fragments. For a ligand with `F` rigid fragments:

```
p_0(x | c)  =  prod_{f=1..F}  N(t_f ; 0, I_3)  ×  Haar(R_f)
```

| Component | Distribution | Conditioned on `c`? | Coupling |
|---|---|---|---|
| Translation `t_f` | `N(0, I_3)` in pocket-centred, scaled coordinates | **implicitly yes** — the origin *is* the pocket centroid | none between fragments |
| Rotation `R_f` | `Haar(SO(3))` (`so3_utils.sample_uniform`) | **no** | none between fragments |

Two asymmetries follow, and they are the whole motivation for this document.

**The translational source is already informative.** Its mean sits at the
pocket centroid, and the data concentrate there too. "Move toward the origin"
is a usable answer at every `t`, including `t = 0`.

**The rotational source is maximally uninformative, by construction.** Haar is
the invariant measure on `SO(3)`; invariance is exactly the property that
makes it carry no orientational information. It has no distinguished mean —
every point is a Karcher mean — so there is no direction in which the model
can hedge.

**Fragments are independent.** At `t = 0` the ligand is not a molecule but `F`
independent rigid bodies scattered about the pocket. Molecular integrity is
recovered only through the learned field.

### 1.1 Why this interacts badly with the geodesic path

With `R_1 = I` (SigmaFlow_Minimal stores the identity for every fragment, so
`R` is a *delta* against the crystal orientation):

```
R_t = R_0 Exp(t Log(R_0^T R_1)) = Exp((1-t) xi_0),   xi_0 = Log(R_0)
u_t = Log(R_0^T R_1) = -xi_0                          (independent of t)
```

The regression target has the same magnitude at every `t` — on average the
Haar angle, ~126.5°. There is no easy regime. At small `t` the state is
uninformative about the target, so the `L^2`-optimal output is a conditional
mean of rotation vectors, which contracts (`||E[xi]|| <= E[||xi||]`). A
shrunken field integrates to almost no rotation and the sample stays near
`R_0 ~ Haar`.

**Measured, 12 h:** median rotation error 138.2° vs Haar 132.3°; enrichment
below 30° is 2.1 % against a Haar rate of ~2 %; predicted/target magnitude
ratio 0.47; cosine at `t < 0.1` ≈ 0.025.

---

## 2. Why source design is a flow-matching-specific degree of freedom

Under diffusion, `p_0` is **not a choice**: it is whatever the forward SDE
converges to (isotropic Gaussian on `R^3`, uniform on `SO(3)`). Changing it
means changing the noising process, its transition kernel, and the score
target.

Under flow matching the only requirements on `p_0` are that it be **cheap to
sample** and that a conditional path connect it to `p_1`. Any such `p_0` is
admissible — including one that depends on the conditioning context `c`.

This is the cleanest example in the project of a modelling freedom that exists
*because* of the substitution, not incidentally alongside it.

### 2.1 Leakage constraints — non-negotiable

`c` may contain **only** information available at inference:

- receptor structure and pocket geometry;
- the ligand molecular graph and its features;
- the unbound/RDKit conformer and quantities derived from it.

`c` may **never** contain:

- the crystallographic bound pose;
- ground-truth fragment centroids or orientations;
- anything derived from `pos_0` after the ground-truth rototranslation.

**This is not hypothetical.** An earlier variant in this project placed
conformer fragments at crystal centres of mass, which supplies part of the
answer through the source. Every source below must ship with a **control in
which the construction degenerates to Haar and must reproduce the baseline
bit-for-bit.** That control is the test, not the intention.

---

## 3. Hand-designed source families

| # | Construction | Information used | Leakage | Expected benefit | Failure mode |
|---|---|---|---|---|---|
| H1 | Pocket centroid (translation) | receptor | none — already in use | already realised | — |
| H2 | Pocket covariance → anisotropic translational Gaussian | receptor | none | small; translation already works | over-concentration if pocket is large |
| H3 | Receptor pocket PCA frame as rotational centre | receptor | none | **the target candidate** | axis sign/permutation ambiguity (§4.2) |
| H4 | Ligand conformer PCA frame aligned to pocket PCA frame | receptor + unbound ligand | none | moderate; couples shape to cavity shape | degenerate for near-spherical fragments |
| H5 | Local surface normal at the nearest pocket wall | receptor | none | plausible for flat/planar fragments | undefined in open pockets |
| H6 | Classical docking pose (Vinardo/smina) as source | receptor + ligand + external tool | none, but expensive | potentially large | needs a preprocessing pass over the whole training set; `gnina` currently unavailable |

**H3 is the primary candidate**, because it is the cheapest construction that
targets the diagnosed mechanism directly. H6 is the strongest but is a
different kind of project — it makes SigmaFlow a refinement model on top of a
classical docker, which is a defensible design but a different thesis.

### 3.1 The prerequisite nobody has tested

All of H3–H5 assume **pocket geometry carries orientational information**. If
the angle between the pocket principal axis and the crystal ligand principal
axis is near-uniform on `SO(3)`, the whole family is worthless. **This is
IS-1 and it gates everything.**

---

## 4. Distributions on SO(3)

| Family | Density | Centre | Concentration | Sampling | Verdict for SigmaFlow |
|---|---|---|---|---|---|
| Haar | uniform | none | — | exact (`sample_uniform`) | current baseline |
| **Wrapped Gaussian** | `R = R_c Exp(xi)`, `xi ~ N(0, sigma^2 I)` | `R_c` | `sigma` | **exact, trivial** | **recommended** |
| IGSO(3) | heat kernel on `SO(3)` | `R_c` | time `tau` | needs series truncation + cached CDF | correct but heavier; SigmaDock already ships the machinery |
| Matrix Fisher | `exp(tr(F^T R))` | via SVD of `F` | singular values of `F` | rejection sampling | most expressive, most implementation risk |
| Mixture | `sum_k pi_k p_k` | several | per component | trivial given components | needed for §4.2 |

**Recommendation: wrapped Gaussian.** Exactly sampleable by two lines
(`R_c @ Exp(sigma * randn(3))`), reduces to Haar in the limit `sigma -> inf`
only approximately — so the Haar control must be an explicit branch, not a
limit. Right-multiplication matches the body-frame convention fixed elsewhere
in the codebase.

Note the concentration is not free to choose blindly: the median of
`||xi||` for `xi ~ N(0, sigma^2 I_3)` is `1.5382 * sigma` (Chi-3 median), so a
target median source-to-data angle of `theta` implies
`sigma ≈ theta / 1.5382`.

### 4.1 Diversity cost

A concentrated source reduces sample diversity. At `N_seeds = 10` this may
*reduce* best-of-K even while improving Top-1. **Both must be reported.** A
source so tight that every seed gives the same pose has destroyed the
generative model's only advantage over a regressor.

### 4.2 Axis ambiguity — the reason mixtures exist

A PCA frame is defined only up to sign flips of each axis and permutation of
degenerate eigenvalues. There are 4 proper-rotation sign assignments
(det = +1) for a fixed axis order. Choosing one arbitrarily injects a
deterministic but meaningless orientation.

Two responses:

- **fix the gauge** by a chemically meaningful convention (e.g. third-moment
  sign fixing along each axis), which is what the EXP-100 machinery already
  does; or
- **model the ambiguity** as a mixture over the 4 valid assignments, which is
  honest but multiplies the source's effective entropy.

---

## 5. Fragment coupling

| Scheme | Form | Cost | Suitable here? |
|---|---|---|---|
| Independent | `prod_f p_0(R_f | c)` | none | current; simplest |
| **Hierarchical** | global ligand orientation `R_g`, then `R_f = R_g Exp(eps_f)` | small | **best fit** — the measured locality gap says fragments should move coherently |
| Graph-correlated | correlation along the fragment bond graph | moderate | plausible but more parameters than the data support |
| Learned joint | full conditional on `SE(3)^F` | large | see §6 |

For the actual complexes here — **median 4 fragments, q90 7, max 11** — a
hierarchical source with one global orientation plus small per-fragment
perturbations is well matched to the problem size. A full joint model over
`SE(3)^11` is over-parameterised for 19k training complexes.

---

## 6. Learned sources

Write `z ~ p(z)`, `x_0 = g_phi(z, c)`.

| Option | Capacity | Risk |
|---|---|---|
| Orientation-proposal head on the existing backbone | low | cheap; shares features with the flow |
| Small equivariant proposal network | medium | a second model to train and validate |
| Mixture density network on `SO(3)` | medium | mode collapse |
| Conditional normalising flow on `SO(3)` | high | substantial implementation risk |

### 6.1 The question that decides whether this is worth doing

> At what point has the source become a second docking model, with SigmaFlow
> demoted to a refinement stage?

This is not rhetorical. Distinguish:

- **Weak learned prior** — the source narrows orientation to a coarse basin
  (say, median error 90° instead of 126.5°), leaving genuine work for the
  flow. Scientifically interesting: it tests whether the flow can exploit a
  better starting point.
- **Strong learned source** — the source already predicts the bound
  orientation to within ~30°. Then the flow is a refiner, the comparison to
  SigmaDock is no longer a comparison of generative engines, and the honest
  description of the system has changed.

**Proposed operational boundary:** if the *source alone* (IS-0 baseline,
§7) achieves better than half the enrichment below 30° of the trained model,
it is a strong source and must be reported as a two-stage architecture, not as
"SigmaFlow with a better prior".

### 6.2 Jointly learned source and flow

`p_phi(x_0 | c)` trained together with `v_theta(x_t, t, c)`.

Mathematically this is **not** a small step, and the CFM identities need
re-examination:

- The marginalisation theorem assumes `p_0` is fixed. With `p_0` depending on
  `phi`, the marginal path `p_t` and the marginal field both depend on `phi`,
  and `grad_phi L_CFM != grad_phi L_FM` in general.
- Gradients must pass through sampling. On `SO(3)` the wrapped-Gaussian
  construction `R = R_c(phi) Exp(sigma(phi) xi)` *is* reparameterisable, which
  is a genuine advantage over families needing rejection sampling.
- **Source collapse** is the dominant failure mode: the objective is minimised
  by making the transport trivial, i.e. `p_0 -> p_1`. Nothing in the CFM loss
  penalises that. Entropy regularisation on `p_0` is required, and its weight
  is a new hyperparameter with no principled default.
- Identifiability: many `(p_0, v)` pairs produce the same `p_1`.

**Verdict: Future Work.** Correct treatment needs its own derivation and a
controlled experiment; there is no room in the MSc timeline and a
half-analysed version would be worse than none.

---

## 7. The experimental ladder

| ID | Hypothesis | Code change | Compute | Primary metric | Go/no-go | Thesis value |
|---|---|---|---|---|---|---|
| **IS-0** | Haar baseline; how much docking does the source alone do? | none | CPU, minutes | median rot. error, <30°/<60°/<90°, translation error | reference point | needed for every later claim |
| **IS-1** | Pocket geometry carries orientational information | none (audit script exists) | **CPU, 2 h** | median angle(pocket axis, ligand axis) vs Haar 132.3° | **if not below Haar by more than its bootstrap CI, STOP** | negative result is publishable and cheap |
| **IS-2** | A concentrated, leakage-free rotational source improves rotation specifically | `conditional_source.py` (exists) + config flag | 1×6 h GPU + bit-identical Haar control | enrichment <30° (primary); RMSD (secondary) | improvement must be rotation-specific | **the central flow-specific experiment** |
| **IS-3** | There is an optimal concentration | config sweep | 3×6 h GPU | enrichment <30°, best-of-10 | U-shape expected | shows the diversity/accuracy trade-off |
| **IS-4** | Modelling axis ambiguity beats fixing the gauge | mixture branch | 1×6 h GPU | enrichment <30°, best-of-10 | only if IS-2 positive | moderate |
| **IS-5** | A weak learned prior beats a hand-designed one | proposal head | 1×12 h GPU + head training | enrichment <30°; **plus source-alone baseline** | only if IS-2 positive | high if it works |
| **IS-6** | Joint source + flow | substantial | large | — | — | Future Work only |

Every experiment reports the **source-alone baseline** alongside the trained
result. Without it, an improvement cannot be attributed to the flow rather
than to the source having done the work.

### 7.1 Measuring informativeness

Report, for the source distribution alone against the crystal target:

- expected/median geodesic distance (Haar reference: mean 126.5°, median
  132.3°);
- enrichment below 30°, 60°, 90° relative to the Haar rates;
- fraction of complexes where the source alone gives RMSD < 2 Å (expected ~0);
- **entropy relative to Haar**, i.e. `log(vol SO(3)) - H[p_0]`, which for a
  wrapped Gaussian is available in closed form to good approximation.

**Do not** attempt mutual-information estimates between `c` and `R_1`. With
~19k training complexes and a 3-dimensional compact target, any MI estimator
would be dominated by its own bias, and a number that cannot be trusted is
worse than no number.

---

## 8. Priority

**DO NOW**
- IS-0 (source-alone baseline) — CPU, needed regardless.
- IS-1 (pocket-geometry audit) — CPU, gates everything, kills the strand
  cheaply if negative.
- IS-2 (concentrated rotational source, H3) — **if and only if IS-1 passes.**

**ONLY IF IS-2 IS POSITIVE**
- IS-3 (concentration sweep).
- IS-4 (mixture over axis assignments).

**FUTURE WORK**
- IS-5 (learned source) — unless IS-2 is strongly positive and time remains.
- IS-6 (joint training) — analysed in §6.2, not attempted.
- H6 (classical docking pose as source) — needs `gnina` and a full
  preprocessing pass.

---

## 9. What this contributes to the thesis

The construction follows from the mathematics rather than from search: the
time-invariance of the geodesic target implies the source is where the
information must come from, and flow matching is what makes the source
editable. That chain — **derivation → prediction → targeted intervention** —
is the strongest available form of the argument, and it holds whether IS-2
succeeds or fails.

If IS-1 fails, the honest report is that receptor geometry does not determine
ligand orientation, which is itself a substantive statement about why the
rotational channel is hard.
