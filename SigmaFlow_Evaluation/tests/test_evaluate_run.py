"""End-to-End-Test von evaluate_run.py auf synthetischen Daten.

Es geht nicht darum, RDKit zu testen, sondern die Verdrahtung:
Layout-Erkennung, Kristallkopien-Auswahl, Symmetrie-RMSD, TFD-Robustheit,
Fehlerbuchhaltung und die Uebergabe an den Ranking-Evaluator.

Der Test baut ein echtes Molekuel, verschiebt es um einen BEKANNTEN Betrag und
prueft, dass genau dieser Betrag als RMSD herauskommt. Eine Pipeline, die
Koordinaten irgendwo falsch zuordnet, faellt hier durch.

    python -m pytest SigmaFlow_Evaluation/tests/test_evaluate_run.py -q
    python SigmaFlow_Evaluation/tests/test_evaluate_run.py        # ohne pytest
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from SigmaFlow_Evaluation.evaluate_run import (  # noqa: E402
    Failures,
    attach_scores,
    basic_metrics,
    best_copy,
    collect,
    symmetry_rmsd,
)
from SigmaFlow_Evaluation.metrics.ranking import evaluate_ranking  # noqa: E402

RDLogger.DisableLog("rdApp.*")

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def make_mol(smiles: str = "CCOc1ccccc1C(=O)NC", seed: int = 0xC0FFEE) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, ps) == 0, "Einbettung fehlgeschlagen"
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def shifted(mol: Chem.Mol, delta: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    pos = conf.GetPositions() + delta
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, p.tolist())
    return out


def write_sdf(path: Path, mols: list[Chem.Mol]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(path))
    for m in mols:
        w.write(m)
    w.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="evalrun_"))
    try:
        true_dir = tmp / "true"
        root = tmp / "sampling"
        model = "sigmaflow_minimal"

        mol = make_mol()
        n_atoms = mol.GetNumAtoms()

        # --- Komplex A: eine Kopie, Vorhersage um exakt 1.0 A verschoben -----
        shift = np.array([1.0, 0.0, 0.0])
        write_sdf(true_dir / "AAAA_ligands.sdf", [mol])
        write_sdf(root / "results" / "posebusters" / model / "seed_0" / "AAAA__x_seed0.sdf",
                  [shifted(mol, shift)])
        # zweiter Seed, 3.0 A daneben
        write_sdf(root / "results" / "posebusters" / model / "seed_1" / "AAAA__x_seed1.sdf",
                  [shifted(mol, np.array([3.0, 0.0, 0.0]))])

        # --- Komplex B: ZWEI Kristallkopien; die zweite ist die nahe ---------
        far = shifted(mol, np.array([25.0, 0.0, 0.0]))
        write_sdf(true_dir / "BBBB_ligands.sdf", [far, mol])
        write_sdf(root / "results" / "posebusters" / model / "seed_0" / "BBBB__x_seed0.sdf",
                  [shifted(mol, np.array([0.0, 0.5, 0.0]))])
        write_sdf(root / "results" / "posebusters" / model / "seed_1" / "BBBB__x_seed1.sdf",
                  [shifted(mol, np.array([0.0, 0.5, 0.0]))])

        # --- Komplex C: wahre Pose fehlt -> muss als Fehlgrund gezaehlt werden
        write_sdf(root / "results" / "posebusters" / model / "seed_0" / "CCCC__x_seed0.sdf",
                  [mol])

        print("\n1. Einheiten-Checks")
        r = symmetry_rmsd(shifted(mol, shift).GetConformer().GetPositions(), mol)
        check("starre Verschiebung um 1.0 A ergibt RMSD 1.0", abs(r - 1.0) < 1e-6, f"{r:.6f}")
        r0 = symmetry_rmsd(mol.GetConformer().GetPositions(), mol)
        check("identische Pose ergibt RMSD 0", r0 < 1e-9, f"{r0:.2e}")

        m_best, r_best = best_copy(
            shifted(mol, np.array([0.0, 0.5, 0.0])).GetConformer().GetPositions(),
            [far, mol])
        check("naechste Kristallkopie wird gewaehlt, nicht die erste",
              abs(r_best - 0.5) < 1e-6, f"{r_best:.6f} (nicht ~25)")

        print("\n2. Pipeline end-to-end")
        fail = Failures()
        records, tfds = collect(root, true_dir, model, fail)

        check("beide auswertbaren Komplexe gefunden",
              {r.complex_id for r in records} == {"AAAA", "BBBB"},
              str(sorted({r.complex_id for r in records})))
        check("vier Posen gesammelt (2 Komplexe x 2 Seeds)", len(records) == 4,
              str(len(records)))
        check("fehlende wahre Pose wird als Grund gezaehlt, nicht verschluckt",
              any("wahre Pose fehlt" in k for k in fail.counts), str(dict(fail.counts)))

        by = {(r.complex_id, r.pose_id): r.rmsd for r in records}
        check("AAAA/seed0 RMSD == 1.0", abs(by[("AAAA", "seed0")] - 1.0) < 1e-6,
              f"{by[('AAAA', 'seed0')]:.6f}")
        check("AAAA/seed1 RMSD == 3.0", abs(by[("AAAA", "seed1")] - 3.0) < 1e-6,
              f"{by[('AAAA', 'seed1')]:.6f}")
        check("BBBB gegen die NAHE Kopie gewertet", abs(by[("BBBB", "seed0")] - 0.5) < 1e-6,
              f"{by[('BBBB', 'seed0')]:.6f}")

        print("\n3. Metriken")
        basics = basic_metrics(records, tfds)
        check("n_complexes == 2", basics["n_complexes"] == 2)
        check("TFD-Abdeckung wird berichtet (auch wenn 0)",
              "coverage" in basics["tfd"], str(basics["tfd"]))
        check("TFD einer starren Verschiebung ist 0 (falls berechenbar)",
              (basics["tfd"]["median"] is None) or (basics["tfd"]["median"] < 1e-6),
              str(basics["tfd"]["median"]))

        print("\n4. Ranking-Evaluator")
        res = evaluate_ranking(records, k_values=(1, 2))
        check("Ranking laeuft ohne Score durch", res["has_score"] is False)
        check("Oracle@2 >= Oracle@1 (Erfolgsrate monoton in K)",
              res["oracle"][2]["success_2A"] >= res["oracle"][1]["success_2A"],
              f"{res['oracle'][1]['success_2A']:.3f} -> {res['oracle'][2]['success_2A']:.3f}")
        # AAAA: min(1.0, 3.0)=1.0 < 2 -> Erfolg;  BBBB: 0.5 < 2 -> Erfolg
        check("Oracle@2 == 100 % (beide Komplexe haben eine Pose < 2 A)",
              abs(res["oracle"][2]["success_2A"] - 1.0) < 1e-9,
              f"{res['oracle'][2]['success_2A']:.3f}")
        # Oracle@1 mittelt ueber zufaellige Einzelzuege: AAAA hat eine gute und
        # eine schlechte Pose (P=0.5), BBBB zwei gute (P=1.0) -> Erwartung 0.75.
        # Das ist eine Monte-Carlo-Schaetzung ueber n_resamples=20 Teilmengen je
        # Komplex, also 40 Beobachtungen. Die Toleranz MUSS aus dem
        # Stichprobenfehler kommen, nicht geraten sein:
        #     SE = sqrt(0.75*0.25/40) = 0.068  ->  3-Sigma-Band = 0.21
        # Eine engere absolute Toleranz laesst einen korrekten Schaetzer
        # durchfallen. Genau dieser Fehler ist in diesem Projekt schon zweimal
        # passiert (Phase-2- und Phase-3-Audit); hier ist er zum dritten Mal
        # aufgetreten und deshalb jetzt explizit dokumentiert.
        n_obs = res["oracle"][1]["n_observations"]
        se = (0.75 * 0.25 / max(n_obs, 1)) ** 0.5
        check("Oracle@1 ~ 75 % (innerhalb 3 Stichprobenfehlern)",
              abs(res["oracle"][1]["success_2A"] - 0.75) < 3 * se,
              f"{res['oracle'][1]['success_2A']:.3f}  (n={n_obs}, 3*SE={3*se:.3f})")

        print("\n5. Score-Nahtstelle (externer Ranker)")
        import json as _json
        # Perfekter Ranker: hoher Score genau fuer die niedrigen RMSDs.
        good = {"AAAA": {"seed0": 10.0, "seed1": 0.0},
                "BBBB": {"seed0": 10.0, "seed1": 9.0}}
        sp = tmp / "scores_good.json"
        sp.write_text(_json.dumps(good), encoding="utf-8")
        f2 = Failures()
        scored, info = attach_scores(records, sp, f2)
        check("alle Posen bekommen einen Score", info["coverage"] == 1.0, str(info["coverage"]))
        check("Originaldatensaetze bleiben unveraendert (frozen -> Kopie)",
              all(r.score is None for r in records))
        res_g = evaluate_ranking(scored, k_values=(1, 2))
        check("mit Score wird Top-1@K berechnet", res_g["has_score"] is True)
        check("perfekter Ranker erreicht Oracle@2",
              abs(res_g["top1"][2]["success_2A"] - res_g["oracle"][2]["success_2A"]) < 1e-9,
              f"top1={res_g['top1'][2]['success_2A']:.3f} "
              f"oracle={res_g['oracle'][2]['success_2A']:.3f}")

        # Gegenprobe: ein invertierter Ranker MUSS schlechter abschneiden.
        # Ohne diesen Fall wuerde ein Evaluator, der den Score in Wahrheit
        # ignoriert, unbemerkt durchgehen - er saehe beim guten Ranker genauso
        # perfekt aus. Ein Test, den nur die richtige Implementierung besteht,
        # braucht immer auch den Fall, der scheitern MUSS.
        bad = {"AAAA": {"seed0": 0.0, "seed1": 10.0},
               "BBBB": {"seed0": 0.0, "seed1": 1.0}}
        sp2 = tmp / "scores_bad.json"
        sp2.write_text(_json.dumps(bad), encoding="utf-8")
        res_b = evaluate_ranking(attach_scores(records, sp2, Failures())[0], k_values=(1, 2))
        check("invertierter Ranker ist messbar schlechter",
              res_b["top1"][2]["success_2A"] < res_g["top1"][2]["success_2A"],
              f"{res_b['top1'][2]['success_2A']:.3f} < {res_g['top1'][2]['success_2A']:.3f}")

        # Teilabdeckung darf nicht still durchgehen.
        sp3 = tmp / "scores_partial.json"
        sp3.write_text(_json.dumps({"AAAA": {"seed0": 1.0}}), encoding="utf-8")
        f3 = Failures()
        _, info3 = attach_scores(records, sp3, f3)
        check("Teilabdeckung wird als Zahl UND als Fehlgrund ausgewiesen",
              info3["coverage"] < 1.0 and any("kein Score" in k for k in f3.counts),
              f"coverage={info3['coverage']}, Gruende={dict(f3.counts)}")

        print(f"\n(Molekuel: {n_atoms} Schweratome)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


def test_evaluate_run() -> None:          # pytest-Einstiegspunkt
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
