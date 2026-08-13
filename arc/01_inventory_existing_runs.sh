#!/bin/bash -l
#
# SCHRITT 1 — auf dem LOGIN-Knoten ausfuehren, NICHT mit sbatch.
#
# Beantwortet genau eine Frage:
#   "Existiert schon ein WIEDERVERWENDBARER 24h-Lauf, oder muss neu trainiert werden?"
#
# Der Verdacht steht lokal bereits fest (siehe ARC_RUNBOOK_AND_CURRENT_STATE.md
# Abschnitt 3): die 24h-Laeufe vom 2026-08-07, Jobs 8465054 (SigmaFlow) und
# 8465055 (SigmaDock), sind BEIDE ungueltig, weil sie mit `--max_epochs 1000`
# liefen. Dieser Wert speist `max_steps` und damit den LR-Scheduler; bei ~28k
# tatsaechlichen Schritten war die Lernrate praktisch konstant. Die spaetere
# 6h/12h-Reihe benutzt `--max_epochs 3` bzw. `6`. SigmaFlow ist zusaetzlich
# vor-Frame-Fix (Root Cause erst am 2026-08-09 gefunden).
#
# Dieses Skript prueft das auf ARC nach, statt es zu glauben.
#
#   bash arc/01_inventory_existing_runs.sh

set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/_common.sh"

echo "=============================================================="
echo "INVENTUR BESTEHENDER LAEUFE"
echo "=============================================================="

echo
echo "--- 1. Was sagt die SLURM-Buchhaltung zu den 24h-Jobs? ---"
if command -v sacct >/dev/null 2>&1; then
    sacct -j 8465054,8465055 \
          --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,Start,End,NodeList%15 2>/dev/null \
      || echo "  (sacct liefert nichts - die Buchhaltung haelt Jobs oft nur begrenzt vor)"
else
    echo "  sacct nicht verfuegbar"
fi

echo
echo "--- 2. Alle Experimentordner mit Checkpoints ---"
echo "    (sortiert nach Alter; die 24h-Laeufe waeren vom 2026-08-07/08)"
for root in "${ARC_REPO}"/*/experiments/sigmadock \
            "${ARC_USER_ROOT}"/SigmaFlow_Variants_JulianMueller/*/experiments/sigmadock \
            "${ARC_SIGMADOCK}"/experiments/sigmadock; do
    [ -d "$root" ] || continue
    echo
    echo "  ${root}"
    for d in "$root"/*/; do
        [ -d "${d}checkpoints" ] || continue
        n=$(find "${d}checkpoints" -name "*.ckpt" 2>/dev/null | wc -l)
        last="${d}checkpoints/last.ckpt"
        sz="-"
        [ -f "$last" ] && sz="$(du -h "$last" 2>/dev/null | cut -f1)"
        printf "    %-28s ckpts=%-3s last.ckpt=%-6s mtime=%s\n" \
            "$(basename "$d")" "$n" "$sz" \
            "$(date -r "$d" +%Y-%m-%d\ %H:%M 2>/dev/null || echo '?')"
    done
done

echo
echo "--- 3. Entscheidende Pruefung: welches max_epochs lief? ---"
echo "    Ein Lauf ist nur dann eine gueltige 24h-Baseline der aktuellen Reihe,"
echo "    wenn max_epochs zum Budget kalibriert war (24h -> 12, nicht 1000)."
echo
for log in "${ARC_USER_ROOT}"/**/slurm_logs/846505*.out \
           "${ARC_REPO}"/**/slurm_logs/846505*.out; do
    [ -f "$log" ] || continue
    echo "  ${log}"
    grep -m1 -o -- "--max_epochs [0-9]*" "$log" 2>/dev/null | sed 's/^/     /' || echo "     max_epochs nicht im Log"
    grep -m1 "sigmadock loaded from" "$log" 2>/dev/null | sed 's/^/     /'
    grep -m1 "frame fix verified" "$log" 2>/dev/null | sed 's/^/     /' || echo "     KEIN 'frame fix verified' -> vor-Frame-Fix"
done 2>/dev/null
echo "  (Falls keine Logs erscheinen: die .out-Dateien liegen im damaligen"
echo "   Submit-Verzeichnis. Suchbefehl:"
echo "     find ${ARC_USER_ROOT} -name '846505*.out' 2>/dev/null )"

echo
echo "--- 4. Automatische Suche nach den Logs ---"
find "${ARC_USER_ROOT}" -name "846505*.out" -o -name "846505*.err" 2>/dev/null | head -10

echo
echo "=============================================================="
echo "URTEIL"
echo "=============================================================="
cat <<'VERDICT'
Erwartetes Ergebnis, lokal aus STATUS.md hergeleitet:

  SigmaFlow 24h (Job 8465054)  -> NOT VALID 24H BASELINE
      Grund 1: vor dem Frame-Fix (Root Cause 2026-08-09, Lauf 2026-08-07).
      Grund 2: --max_epochs 1000, also unkalibrierter LR-Zeitplan.
      Beides einzeln reicht schon aus.

  SigmaDock 24h (Job 8465055)  -> NOT VALID AS A MATCHED BASELINE
      Der SigmaDock-CODE ist unveraendert und in Ordnung. Aber derselbe
      --max_epochs 1000 LR-Zeitplan macht ihn unvergleichbar mit einer neuen,
      LR-kalibrierten SigmaFlow-Seite. Fairness geht vor Ersparnis.

  => Beide 24h-Laeufe muessen neu gefahren werden.

Falls die Ausgabe oben dem widerspricht - etwa weil ein Log doch
`--max_epochs 12` und `frame fix verified` zeigt - dann NICHT neu starten,
sondern das Runbook korrigieren und den bestehenden Lauf wiederverwenden.
VERDICT
