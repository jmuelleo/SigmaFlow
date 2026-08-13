"""Robuste TFD-Auswertung (Torsional Fingerprint Deviation).

VORGESCHICHTE
Die Registry (EVAL-01) hielt fest: TFD ist exakt starrkoerperinvariant und
praktisch orthogonal zum RMSD (Spearman +0.03), faellt aber bei rund 40 % der
Molekuele aus. Damit war es als Hauptmetrik unbrauchbar.

URSACHE, GEMESSEN
Auf 209 PoseBusters-Komplexen:

    Laden mit sanitize=False                61.2 % Erfolg
    Laden mit sanitize=True                 93.3 %
    Koordinaten auf die WAHRE Topologie     siehe unten

Der erste Sprung ist reine Ladefrage: `GetTFDBetweenMolecules` braucht
Ringinformation und perzipierte Bindungsordnungen. Die Auswertungsskripte laden
mit `sanitize=False`, weil das gegen kaputte SDF-Zeilen robust ist - genau das
kostet aber die TFD-Abdeckung.

Der Rest sind `The two molecules must be instances of the same molecule!`:
RDKit perzipiert wahres und vorhergesagtes Molekuel unabhaengig und kommt bei
tautomeren oder aromatischen Grenzfaellen zu verschiedenen Ergebnissen.

DIE BEHEBUNG IST LEGITIM, NICHT KOSMETISCH
Das Modell BEWEGT nur Atome; es aendert die Topologie nicht. Wahres und
vorhergesagtes Molekuel sind per Konstruktion dasselbe Molekuel mit derselben
Atomreihenfolge. Statt zwei unabhaengig perzipierte Molekuele zu vergleichen,
werden deshalb die vorhergesagten KOORDINATEN auf die wahre Topologie
uebertragen. Damit ist die Voraussetzung von `GetTFDBetweenMolecules` per
Konstruktion erfuellt - ohne die Chemie zu veraendern.

Die Atomzahl wird vorher geprueft. Stimmt sie nicht, wird NICHT still
uebersprungen, sondern ein Fehlergrund zurueckgegeben und gezaehlt.

    python -m SigmaFlow_Evaluation.metrics.tfd --self-test <ordner_mit_ligands_sdf>
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import TorsionFingerprints
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TFDResult:
    """Ein TFD-Wert oder ein benannter Fehlschlag. Nie stilles Fallenlassen."""
    value: float | None
    ok: bool
    reason: str = ""


@dataclass
class TFDCoverage:
    """Abdeckungsbilanz ueber einen ganzen Lauf."""
    n_total: int = 0
    n_ok: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    values: list[float] = field(default_factory=list)

    def add(self, r: TFDResult) -> None:
        self.n_total += 1
        if r.ok:
            self.n_ok += 1
            self.values.append(r.value)
        else:
            self.reasons[r.reason] = self.reasons.get(r.reason, 0) + 1

    @property
    def coverage(self) -> float:
        return self.n_ok / self.n_total if self.n_total else 0.0

    def report(self) -> str:
        lines = [f"TFD-Abdeckung: {self.n_ok}/{self.n_total} ({100 * self.coverage:.1f} %)"]
        if self.values:
            v = np.array(self.values)
            q = np.percentile(v, [25, 50, 75])
            lines.append(f"  Werte: median={q[1]:.3f}  IQR=[{q[0]:.3f}, {q[2]:.3f}]  "
                         f"mean={v.mean():.3f}")
        for reason, c in sorted(self.reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {c:>4}x  {reason}")
        if self.coverage < 0.95:
            lines.append("  [!] Abdeckung unter 95 % - n MUSS im Bericht ausgewiesen werden.")
        return "\n".join(lines)


def load_sdf(path: str, sanitize: bool = True) -> list[Chem.Mol]:
    """SDF laden. `sanitize=True` ist fuer TFD zwingend - siehe Modulkopf."""
    try:
        return [m for m in Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=True)
                if m is not None]
    except Exception:
        return []


def graft_coordinates(template: Chem.Mol, source: Chem.Mol) -> Chem.Mol:
    """Koordinaten von `source` auf die Topologie von `template` uebertragen.

    Setzt gleiche Atomzahl UND gleiche Atomreihenfolge voraus. Beides gilt hier,
    weil die Vorhersage aus demselben Eingangsmolekuel entsteht und nur bewegt
    wird. Die Ordnungszahlen werden trotzdem geprueft - eine stille Verwechslung
    waere schlimmer als ein fehlender Wert.
    """
    if template.GetNumAtoms() != source.GetNumAtoms():
        raise ValueError(f"Atomzahl {source.GetNumAtoms()} != {template.GetNumAtoms()}")
    zt = [a.GetAtomicNum() for a in template.GetAtoms()]
    zs = [a.GetAtomicNum() for a in source.GetAtoms()]
    if zt != zs:
        raise ValueError("Atomreihenfolge weicht ab (Ordnungszahlen ungleich)")

    out = Chem.Mol(template)
    conf = out.GetConformer()
    P = source.GetConformer().GetPositions()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(P[i, 0]), float(P[i, 1]), float(P[i, 2])))
    return out


def tfd(mol_true: Chem.Mol, mol_pred: Chem.Mol, graft: bool = True) -> TFDResult:
    """TFD zwischen wahrer und vorhergesagter Pose.

    Args:
        mol_true: Referenzmolekuel, sanitisiert geladen
        mol_pred: Vorhersage, sanitisiert geladen
        graft:    Koordinaten auf die Referenztopologie uebertragen (empfohlen).
                  `False` reproduziert das alte, unzuverlaessige Verhalten und
                  existiert nur, damit der Unterschied messbar bleibt.
    """
    try:
        probe = graft_coordinates(mol_true, mol_pred) if graft else mol_pred
        v = TorsionFingerprints.GetTFDBetweenMolecules(mol_true, probe)
        if v is None or not np.isfinite(v):
            return TFDResult(None, False, "TFD nicht endlich")
        return TFDResult(float(v), True)
    except ValueError as e:
        msg = str(e)
        if "same molecule" in msg:
            return TFDResult(None, False, "Topologie-Mismatch (RDKit-Perzeption)")
        if "Atomzahl" in msg or "Atomreihenfolge" in msg:
            return TFDResult(None, False, msg)
        return TFDResult(None, False, f"ValueError: {msg[:60]}")
    except Exception as e:
        return TFDResult(None, False, f"{type(e).__name__}: {str(e)[:60]}")


def rigid_motion_invariance_check(mol: Chem.Mol, seed: int = 0) -> float:
    """TFD gegen eine zufaellig rotierte und verschobene Kopie. Muss 0 sein.

    Diese Zusicherung ist der Grund, warum TFD ueberhaupt als interne
    Geometriemetrik taugt: sie misst NUR die Torsionen, nicht die Platzierung.
    """
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(Q) < 0:
        Q[:, 2] = -Q[:, 2]
    moved = Chem.Mol(mol)
    conf = moved.GetConformer()
    P = mol.GetConformer().GetPositions() @ Q.T + np.array([7.0, -3.0, 11.0])
    for i in range(moved.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*(float(x) for x in P[i])))
    return abs(TorsionFingerprints.GetTFDBetweenMolecules(mol, moved))


if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--true_dir", required=True)
    ap.add_argument("--pred_glob", default=None,
                    help="Muster mit {cid}; ohne Angabe nur Invarianz-Selbsttest")
    args = ap.parse_args()

    truths = sorted(glob.glob(os.path.join(args.true_dir, "*_ligands.sdf")))
    print(f"Referenzdateien: {len(truths)}")

    print("\n--- Starrkoerperinvarianz (muss 0 sein) ---")
    devs = []
    for tp in truths[:30]:
        ms = load_sdf(tp)
        if ms:
            try:
                devs.append(rigid_motion_invariance_check(ms[0]))
            except Exception:
                pass
    if devs:
        print(f"  n={len(devs)}  max |TFD| = {max(devs):.2e}")

    if args.pred_glob:
        print("\n--- Abdeckung mit und ohne Transplantation ---")
        for use_graft in (False, True):
            cov = TFDCoverage()
            for tp in truths:
                cid = os.path.basename(tp).replace("_ligands.sdf", "")
                hits = glob.glob(args.pred_glob.format(cid=cid))
                tc = load_sdf(tp)
                if not (hits and tc):
                    continue
                pc = load_sdf(hits[0])
                if not pc:
                    cov.add(TFDResult(None, False, "Vorhersage nicht ladbar"))
                    continue
                mp = pc[0]
                P = mp.GetConformer().GetPositions()
                mt, best = None, float("inf")
                for m in tc:
                    Qc = m.GetConformer().GetPositions()
                    if Qc.shape != P.shape:
                        continue
                    r = float(np.sqrt(((P - Qc) ** 2).sum(1).mean()))
                    if r < best:
                        mt, best = m, r
                if mt is None:
                    cov.add(TFDResult(None, False, "keine passende Kristallkopie"))
                    continue
                cov.add(tfd(mt, mp, graft=use_graft))
            print(f"\n  graft={use_graft}")
            print("  " + cov.report().replace("\n", "\n  "))
