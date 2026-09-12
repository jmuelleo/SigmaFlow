# SigmaFlow

Code for the MSc dissertation *Replacing SE(3) Diffusion with Riemannian Flow
Matching for Fragment-Based Molecular Docking*.

SigmaFlow is a minimally invasive replacement of SigmaDock's score-based
SE(3) diffusion by Riemannian flow matching. The fragment state space, the
EquiformerV2 backbone, the data pipeline and the evaluation protocol are kept
fixed; only the generative mechanism changes.

## The three models

| Directory | Name in the dissertation | Generative mechanism |
|---|---|---|
| *(not in this repository, see below)* | SigmaDock | score-based SE(3) diffusion |
| `SigmaFlow_Minimal/` | **SigmaFlow-NE** | flow matching, inherited Newton–Euler readout |
| `SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/` | **SigmaFlow-TR** | flow matching, separate translation and rotation heads |

The directory names are the ones used during the experiments and are hard-coded
in the SLURM scripts under `arc/`. They are deliberately left unchanged so that
every script in this repository runs exactly as it did for the reported results.

SigmaFlow-NE differs from SigmaDock in the probability path, the learned field
(velocity instead of score) and the source distribution. SigmaFlow-TR differs
from SigmaFlow-NE only in the output block: the shared `l=1` field is replaced
by two blocks on the same trunk, and the Newton–Euler reduction is dropped from
the rotational path.

## Layout

```
SigmaFlow_Minimal/            SigmaFlow-NE: model, configs, training and sampling entry points
SigmaFlow_FM_Specific/
  EXP-110_two_head_vector_field/   SigmaFlow-TR, same structure
arc/                          SLURM scripts: training, sampling, redocking, GNINA scoring
SigmaFlow_Evaluation/         RMSD and PoseBusters evaluation, ranking rules
analysis/
  cluster_selection/          binding-mode clustering, cluster oracle, confidence measures
  dataset_overlap/            training-set overlap and similarity analyses
SigmaFlow_Variants/           analysis, table and figure scripts, and the frozen result set
docs/                         the interactive pages linked from the dissertation
```

## Reproducing the results

**Training.** `arc/train_final_72h.slurm`, submitted through `arc/submit_final.sh`.
The main results of the dissertation come from three 72-hour runs, one per model,
on a single GPU at batch size 32. `arc/final_horizon_*.env` holds the per-model
epoch horizons, calibrated from measured throughput.

Two later families of runs use the training recipe published with SigmaDock
(`pb_check`, random rotational augmentation, protein–ligand interaction
features, longer parameter averaging). They differ in their training data:
family A on the original PDBbind v2020 release, family B on the reprocessed
v2020.R1 release with broken complexes removed. Their configurations are the
`pdbbind2020-*` and `*-kuratiert` experiment files under each model's
`conf/experiments/`.

**Sampling and evaluation.** `arc/eval_snapshots_cpu.slurm` samples a checkpoint
over a grid of integration step counts and seeds; `arc/posebusters_redock.slurm`
computes symmetry-corrected RMSD and the PoseBusters checks;
`arc/score_gnina.slurm` adds the Vinardo scores that the mixed score uses.
`arc/neuer_punkt.sh` chains these three with the completeness checks that the
individual scripts do not perform on their own.

**Tables and figures.** `SigmaFlow_Variants/ergebnisse_72h.py` computes every
reported Top-1 rate as an expectation over random K-subsets, using 10,000 draws
per complex and a fixed seed, and writes `ergebnisse_72h_aggregat.csv`. That
file is the single source of truth: `final_tables.py`, `final_comparison.py` and
`plot_ranking.py` read it rather than recomputing, so tables and figures cannot
drift apart.

## Two conventions worth knowing

**PoseBusters rates are over 307 complexes, not 308.** The complex `7XPO_UPG`
never yields a pose, in any model. Each seed writes 308 SDF files because one
other complex is emitted twice; the duplicate is removed before merging. All
three models see exactly the same 307 complexes, so the comparison is unaffected.

**Astex overlaps the training data.** PoseBusters v2 is a temporal split with no
overlap with PDBbind. The Astex Diverse Set is not: 47 of its 85 complexes appear
in PDBbind v2020, matched on both PDB entry and ligand code. This applies equally
to the published SigmaDock figures, which are trained on the same data, so the
comparison is like for like.

## What is not in this repository

Model checkpoints, generated poses and per-pose evaluation tables are not
committed. They are large and regenerable from the scripts here.
`ergebnisse_72h_je_komplex.csv`, the per-complex table behind the paired
bootstrap tests, is also omitted for size; `ergebnisse_72h.py` writes it
alongside the aggregate file.

The SigmaDock source tree is not included. It is third-party code and not ours
to redistribute. The baseline reported in the dissertation is our own
reproduction, trained from that code under the configuration recorded in `arc/`.

## Environment

Python 3.11, PyTorch with PyTorch Geometric, RDKit, `spyrmsd` for
symmetry-corrected RMSD, `posebusters` for the validity checks, and GNINA for
Vinardo scoring. Per-model dependencies are in each model's `pyproject.toml`.
