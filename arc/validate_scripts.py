"""Statische Pruefung aller ARC-Skripte — alles, was ohne ARC pruefbar ist.

Ein SLURM-Job soll nicht nach zwanzig Minuten an einem Zeilenende, einer
Backtick-Substitution oder einem Tippfehler im Pfad sterben. Diese Datei
prueft genau die Fehlerklassen, die lokal entscheidbar sind.

NICHT pruefbar und deshalb ausgenommen: ob die Partition heisst wie erwartet,
ob die GPU-Klasse noch existiert, ob die Conda-Umgebungen da sind. Das
uebernimmt arc/00_preflight.sh auf ARC.

    python arc/validate_scripts.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ARC = Path(__file__).resolve().parent
REPO = ARC.parent

# Verzeichnisse, die kein Job beschreiben darf.
PROTECTED = ("SigmaDock/", "SigmaFlow_Minimal/")
DESTRUCTIVE = [
    (r"\brm\s+-rf\s+/", "rm -rf auf absolutem Pfad"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|push\s+--force)", "destruktives git"),
    (r"\bscancel\b", "scancel - wuerde fremde Jobs toeten"),
    (r">\s*/dev/sd", "Schreiben auf ein Blockgeraet"),
]
SBATCH_REQUIRED = ["job-name", "time", "output", "error"]


def _find_bash() -> str | None:
    """Eine echte Bash finden, nicht den WSL-Starter aus System32."""
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        "/bin/bash", "/usr/bin/bash",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    which = shutil.which("bash")
    if which and "System32" not in which:
        return which
    return None


BASH_EXE = _find_bash()
problems, warnings = [], []


def bad(f, msg):
    problems.append(f"{f}: {msg}")


def warn(f, msg):
    warnings.append(f"{f}: {msg}")


def check_shell(path: Path) -> None:
    name = path.relative_to(REPO)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    is_slurm = path.suffix == ".slurm"
    is_sourced = path.name.startswith("_")

    # --- Zeilenenden --------------------------------------------------
    # Massgeblich ist der GIT-INDEX, nicht die Arbeitskopie: ARC checkt aus dem
    # Index aus. Eine CRLF-Arbeitskopie unter Windows ist harmlos, solange
    # .gitattributes beim Commit normalisiert - und genau das ist zu pruefen.
    try:
        staged = subprocess.run(["git", "show", f":{name.as_posix()}"],
                                cwd=REPO, capture_output=True)
        if staged.returncode == 0 and b"\r\n" in staged.stdout:
            bad(name, "CRLF IM GIT-INDEX -> auf Linux 'bad interpreter: /bin/bash^M'")
        elif staged.returncode != 0:
            bad(name, "nicht in Git - ARC bekaeme die Datei nicht ueber 'git pull'")
    except Exception:
        warn(name, "Git-Index nicht pruefbar")
    if b"\r\n" in raw:
        warn(name, "CRLF in der Arbeitskopie (Index normalisiert, daher unkritisch)")

    # --- Shebang ------------------------------------------------------
    if is_sourced:
        if lines and lines[0].startswith("#!"):
            warn(name, "wird gesourct, braucht keine Shebang")
    elif not lines or not lines[0].startswith("#!"):
        bad(name, "keine Shebang-Zeile")
    elif "bash" not in lines[0]:
        warn(name, f"ungewoehnliche Shebang: {lines[0]}")

    # --- Syntax -------------------------------------------------------
    # Wichtig: NICHT einfach "bash" aufrufen. Unter Windows liegt auf dem PATH
    # meist C:\Windows\System32\bash.exe (WSL-Starter); ohne installierte
    # Distribution meldet der einen Fehler, der wie ein Syntaxfehler des
    # Skripts aussieht. BASH_EXE sucht deshalb gezielt eine echte Bash.
    if BASH_EXE is None:
        warn(name, "keine nutzbare bash gefunden - Syntaxpruefung uebersprungen")
        return
    posix = str(path).replace("\\", "/")
    r = subprocess.run([BASH_EXE, "-n", posix], capture_output=True, text=True)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout).strip()[:200] or f"Rueckgabewert {r.returncode}"
        bad(name, f"bash -n: {detail}")

    # --- set -euo pipefail --------------------------------------------
    if not is_sourced:
        if "set -euo pipefail" not in text and "set -uo pipefail" not in text:
            bad(name, "kein 'set -euo pipefail' (stille Fehlschlaege moeglich)")
        elif "set -uo pipefail" in text and "set -euo pipefail" not in text:
            # Absicht bei den Preflight-Skripten: alle Probleme sammeln, nicht
            # beim ersten abbrechen. Muss aber im Skript begruendet sein.
            if "bewusst KEIN -e" not in text and "sammeln" not in text.lower():
                warn(name, "'set -uo' ohne -e und ohne Begruendung im Text")

    # --- Backticks ausserhalb Kommentaren und quoted heredocs ---------
    in_quoted_heredoc = False
    heredoc_tag = None
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if heredoc_tag:
            if s == heredoc_tag:
                heredoc_tag, in_quoted_heredoc = None, False
            continue
        m = re.search(r"<<-?'([A-Za-z_][A-Za-z0-9_]*)'", ln)
        if m:
            heredoc_tag, in_quoted_heredoc = m.group(1), True
            continue
        m = re.search(r"<<-?\"?([A-Za-z_][A-Za-z0-9_]*)\"?", ln)
        if m and "<<<" not in ln:
            heredoc_tag = m.group(1)
            continue
        if s.startswith("#"):
            continue
        code = ln.split("#", 1)[0] if not s.startswith("#") else ""
        if "`" in code:
            bad(name, f"Zeile {i}: Backtick in ausfuehrbarem Code -> Kommandosubstitution")

    # --- Platzhalter --------------------------------------------------
    for pat in (r"<SCRIPT>", r"<PATH>", r"/path/to/", r"TODO", r"FIXME", r"XXXX"):
        for i, ln in enumerate(lines, 1):
            if re.search(pat, ln) and not ln.strip().startswith("#"):
                bad(name, f"Zeile {i}: Platzhalter '{pat}' in ausfuehrbarem Code")

    # --- destruktive Kommandos ----------------------------------------
    for pat, why in DESTRUCTIVE:
        for i, ln in enumerate(lines, 1):
            if ln.strip().startswith("#"):
                continue
            if re.search(pat, ln):
                bad(name, f"Zeile {i}: {why}")

    # --- Schreiben in geschuetzte Verzeichnisse ------------------------
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("#"):
            continue
        for p in PROTECTED:
            if re.search(rf"(>|>>|cp .*|mv .*|rm .*)\s*\S*{re.escape(p)}", ln):
                # Lesen ist erlaubt, Schreiben nicht. Heuristik, deshalb Warnung.
                warn(name, f"Zeile {i}: moeglicher Schreibzugriff auf {p}")

    # --- cwd-Abhaengigkeit --------------------------------------------
    if not is_sourced and is_slurm:
        if re.search(r'source\s+"?\$\(dirname\s+"?\$0', text):
            bad(name, "sourct relativ zu $0 - bei sbatch ist $0 eine Kopie in /var/spool")
        if not re.search(r'source\s+"\$\{REPO_ROOT\}/arc/_common\.sh"', text):
            if "_common.sh" in text:
                warn(name, "sourct _common.sh nicht ueber den absoluten REPO_ROOT")

    # --- SBATCH-Direktiven --------------------------------------------
    if is_slurm:
        directives = dict(re.findall(r"^#SBATCH\s+--([a-z-]+)=(\S+)", text, flags=re.M))
        for req in SBATCH_REQUIRED:
            if req not in directives:
                bad(name, f"#SBATCH --{req} fehlt")
        if "output" in directives and not directives["output"].startswith("/"):
            bad(name, "--output ist kein absoluter Pfad (haengt vom Submit-Verzeichnis ab)")
        t = directives.get("time", "")
        part = directives.get("partition", "")
        if t and part:
            h = 0
            m = re.match(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", t)
            if m:
                h = int(m.group(1) or 0) * 24 + int(m.group(2))
            if part == "short" and h > 12:
                bad(name, f"--time={t} passt nicht in 'short' (12h-Limit)")
            if part == "medium" and h > 48:
                bad(name, f"--time={t} passt nicht in 'medium' (48h-Limit)")
        if "gres" in directives and "cpus-per-task" not in directives:
            warn(name, "GPU angefordert, aber keine --cpus-per-task (Dataloader hungert)")

    # --- unaufgeloeste Variablen --------------------------------------
    # Variablen, die benutzt, aber weder gesetzt, exportiert, als Parameter
    # dokumentiert noch von _common.sh geliefert werden.
    common = (ARC / "_common.sh").read_text(errors="replace")
    provided = set(re.findall(r"^export\s+([A-Z_][A-Z0-9_]*)=", common, flags=re.M))
    provided |= set(re.findall(r"^\s*export\s+([A-Z_][A-Z0-9_]*)=", text, flags=re.M))
    provided |= set(re.findall(r"^\s*([A-Z_][A-Z0-9_]*)=", text, flags=re.M))
    provided |= set(re.findall(r"[;\s]([A-Z_][A-Z0-9_]*)=", text))
    provided |= set(re.findall(r'\b([A-Z_][A-Z0-9_]*)="?\$\{?\1', text))
    builtin = {"SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_SUBMIT_DIR", "SLURM_ARRAY_TASK_ID",
               "SLURM_JOB_PARTITION", "SLURM_JOB_GRES", "USER", "HOME", "PATH", "PYTHONPATH",
               "TMPDIR", "CONDA_PREFIX", "BASH_SOURCE", "RUN_DIR", "PWD", "LD_LIBRARY_PATH"}
    used = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)[:}]", text)) | \
           set(re.findall(r"\$([A-Z_][A-Z0-9_]{2,})\b", text))
    # Variablen mit :? oder :- sind explizit als Parameter deklariert
    declared = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):[?-]", text))
    unresolved = used - provided - builtin - declared
    for v in sorted(unresolved):
        warn(name, f"Variable ${v} benutzt, aber nirgends gesetzt oder als Parameter deklariert")


def check_python(path: Path) -> None:
    name = path.relative_to(REPO)
    if path.name == "validate_scripts.py":
        return    # sucht selbst nach diesen Mustern
    src = path.read_text(errors="replace")
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        bad(name, f"SyntaxError Zeile {e.lineno}: {e.msg}")
        return
    for pat in (r"/path/to/", r"<PATH>"):
        if re.search(pat, src):
            bad(name, f"Platzhalter '{pat}'")
    if "argparse" in src and "__main__" not in src:
        warn(name, "argparse ohne __main__-Guard")


def main() -> int:
    shells = sorted(list(ARC.glob("*.sh")) + list(ARC.glob("*.slurm")))
    pys = sorted(ARC.glob("*.py"))

    print("=" * 84)
    print("STATISCHE PRUEFUNG DER ARC-SKRIPTE")
    print("=" * 84)
    print(f"{len(shells)} Shell-/SLURM-Dateien, {len(pys)} Python-Dateien\n")

    for f in shells:
        check_shell(f)
    for f in pys:
        check_python(f)

    if problems:
        print(f"FEHLER ({len(problems)}):")
        for p in problems:
            print(f"  [!!] {p}")
        print()
    if warnings:
        print(f"WARNUNGEN ({len(warnings)}):")
        for w in warnings:
            print(f"  [ ?] {w}")
        print()

    print("NICHT LOKAL PRUEFBAR — 'PENDING ARC ENVIRONMENT CHECK':")
    for item in ("Existenz der Partitionen short/medium",
                 "GPU-Klasse l40s noch vorhanden",
                 "Conda-Umgebungen sigmaflow_env und myenv",
                 "Datensatzpfade unter /data/stat-cadd/shug8458/data",
                 "Modul 'Mamba' ladbar",
                 "Schreibrechte unter /data/stat-cadd/shug8458/arc_runs"):
        print(f"  [--] {item}")
    print("  -> arc/00_preflight.sh prueft all das auf ARC.")

    print()
    print("=" * 84)
    if problems:
        print(f"ERGEBNIS: {len(problems)} FEHLER, {len(warnings)} Warnungen")
        return 1
    print(f"ERGEBNIS: 0 Fehler, {len(warnings)} Warnungen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
