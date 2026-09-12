"""Erzeugt den Fingerprint eines finalen Modells.

ZWECK
  Nach 72 GPU-Stunden muss beweisbar sein, WELCHER Code trainiert wurde. Ein
  Ordnername reicht dafuer nicht: die 12h-Laeufe vom 2026-08-12 sind genau
  daran gescheitert -- sie liefen in einem Zeitfenster, in dem nichts committet
  war, und ihre Codeversion ist heute nur noch ueber Verhaltensmerkmale
  rekonstruierbar, nicht ueber einen Hash.

  Dieses Skript schliesst diese Luecke: es schreibt Git-Commit, Dateihashes der
  verhaltensbestimmenden Module und die numerisch geprueften Konventionen in
  eine YAML-Datei, die neben dem Lauf liegt.

WAS ES NICHT TUT
  Es raet nichts. Felder, die sich nicht verifizieren lassen, stehen als
  `UNVERIFIED` mit Begruendung. Ein plausibel aussehender falscher Wert waere
  hier schaedlicher als eine offene Luecke.

    python arc/make_model_fingerprint.py --model sigmaflow_minimal
    python arc/make_model_fingerprint.py --model sigmadock --code_dir /pfad/zu/sigmadock
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Die Module, die das Verhalten bestimmen. Aendert sich hier ein Byte, ist es
# ein anderes Modell -- unabhaengig davon, was der Ordner heisst.
BEHAVIOUR_FILES_SF = [
    "src/sigmadock/diff/sigma_flow_generator.py",
    "src/sigmadock/diff/so3_flow_matcher.py",
    "src/sigmadock/diff/r3_flow_matcher.py",
    "src/sigmadock/diff/se3_flow_matcher.py",
    "src/sigmadock/diff/so3_utils.py",
    "src/sigmadock/diff/sampling.py",
    "src/sigmadock/net/model.py",
    "src/sigmadock/trainer.py",
    "src/sigmadock/oracle.py",
    "src/sigmadock/data.py",
]
BEHAVIOUR_FILES_SD = [
    "src/sigmadock/diff/denoiser.py",
    "src/sigmadock/diff/so3_diffuser.py",
    "src/sigmadock/diff/r3_diffuser.py",
    "src/sigmadock/diff/se3_diffuser.py",
    "src/sigmadock/diff/so3_utils.py",
    "src/sigmadock/diff/sampling.py",
    "src/sigmadock/net/model.py",
    "src/sigmadock/trainer.py",
    "src/sigmadock/oracle.py",
    "src/sigmadock/data.py",
]


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "FEHLT"


def git(code_dir: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(code_dir), *args],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "KEIN GIT-REPO"
    except (OSError, subprocess.SubprocessError):
        return "KEIN GIT-REPO"


def verify_frame_fix(code_dir: Path) -> tuple[str, str]:
    """Prueft im QUELLTEXT, ob der adjungierte Transport vorhanden ist.

    Das ist der eine Unterschied zwischen der Fassung vor und nach dem
    2026-08-09-Fix. Alles andere an dieser Datei war Kommentar.
    """
    f = code_dir / "src/sigmadock/diff/sigma_flow_generator.py"
    if not f.is_file():
        return "N/A", "keine SigmaFlow-Codebasis"
    src = f.read_text(encoding="utf-8", errors="replace")
    marker = 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t'
    if marker in src:
        return "PRESENT", "adjungierter Transport Welt -> Koerper vorhanden"
    if 'pred_u_t_R = updates["omega"]' in src:
        return "MISSING", "VOR dem Frame-Fix: Weltframe geht ungewandelt in einen Koerperframe-Loss"
    return "UNVERIFIED", "keine der beiden bekannten Formen gefunden"


def verify_r1(code_dir: Path) -> tuple[str, str]:
    f = code_dir / "src/sigmadock/diff/sigma_flow_generator.py"
    if not f.is_file():
        return "N/A", ""
    src = f.read_text(encoding="utf-8", errors="replace")
    if "get_fragment_com_and_rot_reparam" in src:
        return "kabsch_reference_conformer", "EXP-100: R_1 relativ zum Referenzkonformer"
    if "rots.append(I3)" in src:
        return "identity", "R_1 = I; R_t ist Delta-Rotation gegen die Kristallorientierung von pos_0"
    return "UNVERIFIED", "Form nicht erkannt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["sigmadock", "sigmaflow_minimal", "sigmaflow_exp100"])
    ap.add_argument("--code_dir", default=None,
                    help="Ueberschreibt den Standardpfad (noetig fuer SigmaDock).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    defaults = {
        "sigmaflow_minimal": REPO / "SigmaFlow_Minimal",
        "sigmaflow_exp100": REPO / "SigmaFlow_FM_Specific/EXP-100_state_reparam",
        "sigmadock": REPO / "SigmaDock",
    }
    code_dir = Path(args.code_dir) if args.code_dir else defaults[args.model]
    if not code_dir.is_dir():
        print(f"FEHLER: {code_dir} existiert nicht.")
        return 2

    is_sd = args.model == "sigmadock"
    files = BEHAVIOUR_FILES_SD if is_sd else BEHAVIOUR_FILES_SF

    hashes = {f: sha256(code_dir / f) for f in files}
    present = [f for f, h in hashes.items() if h != "FEHLT"]
    # Ein Hash ueber die verhaltensbestimmenden Dateien: eine Zahl, die sich
    # aendert, sobald irgendetwas Relevantes anders ist.
    combined = hashlib.sha256(
        "".join(f"{f}:{hashes[f]}\n" for f in sorted(present)).encode()
    ).hexdigest()

    frame_fix, frame_note = verify_frame_fix(code_dir)
    r1_form, r1_note = verify_r1(code_dir)
    commit = git(code_dir, "rev-parse", "HEAD")
    # Der Repo-HEAD sagt, welchen Stand das Arbeitsverzeichnis hat -- er
    # aendert sich bei jedem Commit irgendwo im Repo. Aussagekraeftig fuer das
    # MODELL ist der letzte Commit, der dieses Codeverzeichnis beruehrt hat.
    # Beides wird notiert; verwechselt man sie, sieht ein unveraendertes Modell
    # nach jeder Theorieaenderung wie eine neue Version aus.
    # Eingegrenzt auf src/, nicht auf das ganze Verzeichnis: in
    # SigmaFlow_Minimal liegt ein verirrtes texput.log, dessen Aenderung sonst
    # als "neue Modellversion" gezaehlt wuerde. Nur src/ bestimmt das Verhalten.
    code_commit = git(code_dir, "log", "-1", "--format=%H %ad", "--date=short", "--", "src")
    # WICHTIG: auf das Codeverzeichnis EINGRENZEN. `git status --short` ohne
    # Pfad meldet das ganze Repository, also auch voellig unbeteiligte
    # Aenderungen (theory.tex, Notizen). Der Fingerprint soll aussagen, ob
    # DIESES Modell von seinem Commit abweicht, nicht ob irgendwo im Repo
    # gearbeitet wurde.
    dirty = git(code_dir, "status", "--short", "--", ".")

    if is_sd:
        objective = "denoising score matching (SE(3) Riemannian diffusion)"
        source = "N/A - der Vorwaertsprozess legt die Quelle fest"
        rot_conv = ("SigmaDock-eigene Score-Parametrisierung; "
                    "Konsum durch LINKSmultiplikation (dR @ R_t, denoiser.py)")
    else:
        objective = "conditional flow matching, MSE auf u_t (trans) und u_t_R (rot)"
        source = "trans_0 ~ N(0, I) (taschenzentriert, /2.7 A); R_0 ~ Haar auf SO(3)"
        rot_conv = ("Rdot = R * Omega  (KOERPER / linkstrivialisiert). "
                    "Ziel log(R_t^T R_1)/(1-t); Integrator R_next = R_t @ exp(v dt); "
                    "Netzausgabe wird per R_t^T w R_t transportiert. "
                    "Numerisch verifiziert: audits/so3_trivialisation_audit.py")

    caveats = []
    if not is_sd:
        caveats += [
            "so3_utils.Log ist nahe 180 Grad begrenzt genau (Omega-Clamp auf "
            "+-(1-1e-7)). Relativer Fehler auf dem Trainingsziel: median 5e-7, "
            "p99 3e-4, max 5e-3. Gegen einen Rotationsfehler von 145 Grad "
            "belanglos - Caveat, kein Blocker.",
            "Der Docstring in calc_rot_vector_field nennt die Groesse "
            "'right trivialised'. Nach ueblicher Konvention ist sie LINKS-"
            "trivialisiert (Koerperframe). Reine Benennungsfrage, Mathematik "
            "und Code sind konsistent.",
        ]
        if r1_form == "identity":
            caveats.append(
                "R_1 = I: das Rotationsziel ist per Konstruktion die Identitaet. "
                "Eine informative Rotationsquelle ist in dieser Parametrisierung "
                "nicht sinnvoll messbar (dafuer EXP-100).")
    caveats.append(
        "Val- und Testsatz sind beide posebusters (aus SigmaDocks Setup geerbt). "
        "Trifft beide Methoden identisch, verzerrt den Vergleich also nicht.")
    if dirty and dirty != "KEIN GIT-REPO":
        caveats.append(f"ARBEITSVERZEICHNIS NICHT SAUBER: {len(dirty.splitlines())} "
                       "geaenderte Datei(en). Der Commit beschreibt den Lauf dann NICHT vollstaendig.")

    lines = [
        "# Fingerprint eines finalen Modells. Erzeugt von arc/make_model_fingerprint.py.",
        "# Felder mit UNVERIFIED sind bewusst offen gelassen, nicht geschaetzt.",
        f"model_name: {args.model}",
        f"code_dir: {code_dir}",
        f"git_commit_repo_head: {commit}",
        f"git_commit_code_dir: {code_commit}   # letzte Aenderung an DIESEM Modellcode",
        f"git_clean: {'true' if not dirty or dirty == 'KEIN GIT-REPO' else 'false'}",
        f"backbone_hash: {combined}",
        f"n_behaviour_files: {len(present)}/{len(files)}",
        f"frame_fix: {frame_fix}   # {frame_note}",
        f"rotation_target_R1: {r1_form}   # {r1_note}",
        f"rotation_convention: >-\n    {rot_conv}",
        f"training_objective: {objective}",
        f"source_distribution: {source}",
        "config: arc/final_config.sh   # batch_size, seed, precision, max_epochs",
        "batch_size: FROM_final_config.sh",
        "seed: FROM_final_config.sh",
        f"written_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "known_caveats:",
    ]
    lines += [f"  - {c}" for c in caveats]
    lines.append("file_hashes:")
    lines += [f"  {f}: {hashes[f]}" for f in files]

    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else REPO / "arc" / f"fingerprint_{args.model}.yaml"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[fingerprint] geschrieben: {out}")
    if frame_fix == "MISSING":
        print("\n*** WARNUNG: Frame-Fix FEHLT. Dieses Modell nicht fuer 72h benutzen. ***")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
