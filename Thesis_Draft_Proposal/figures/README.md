# Figures

Empty by design. Every figure in the draft is currently a
`\figplaceholder{...}` in the chapter source, carrying a polished caption and
a full specification of what the final artefact must contain.

Replace a placeholder by dropping the file here and swapping the macro for a
normal `\includegraphics`. The caption text moves across unchanged; the
specification (third argument) is discarded.

Priority order, with what each one costs and replaces:

| # | Figure | Section | Needs new compute? | Replaces |
|---|---|---|---|---|
| 3 | Architecture: inherited vs replaced | §2.3.1 | no | ~400 words |
| 4 | Conditional paths on both factors | §4.4 | no | ~250 words |
| 5 | Rotation-error densities vs Haar | §7.3 | no (12 h arrays exist) | ~300 words |
| 7 | Accuracy vs integration steps | §7.4 | partly (diffusion arm) | ~200 words |
| 1 | Ligand → fragments → product manifold | §2.1 | no | ~250 words |
| 2 | Diffusion vs flow matching | §3.6 | no | ~350 words |
| 6 | Training trajectory, 4 panels | §7.2 | yes | — |
| 8 | Source distributions on SO(3) | §7.6 | yes | — |
| 9 | Confidence: calibration, top-k, vs t | §7.7 | yes | — |

Figures 1–5 and 7 need no new experiments and should be produced first; they
are the slowest artefacts to make and the fastest for a supervisor to react to.
