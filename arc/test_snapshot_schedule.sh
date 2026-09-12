#!/bin/bash
#
# Test des Snapshot-Zeitplans aus train_final_72h.slurm.
#
# WARUM DAS EINEN EIGENEN TEST BRAUCHT
#   Der Zeitplan ist der einzige Teil des 72h-Jobs, dessen Fehler sich erst
#   NACH 72 Stunden zeigt - und dann als fehlende Datei, nicht als Absturz.
#   Eine frueherer Fassung schlief je Eintrag h*3600 Sekunden nacheinander:
#   fuer "6 12 24 48" haette sie bei 6, 18, 42 und 90 Stunden ausgeloest.
#   Der 48h-Snapshot waere nie gefallen, die uebrigen haetten falsche
#   Zeitstempel getragen, und nichts davon haette einen Fehler gemeldet.
#
#   Der Test skaliert Stunden auf Sekunden (Faktor 3600 -> 1) und prueft die
#   tatsaechlichen Ausloesezeitpunkte gegen die Sollwerte.
#
set -euo pipefail

HOURS="${1:-1 2 4 7}"
TOL=1                       # Sekunden Toleranz je Marke
FAILS=0

echo "Zeitplan unter Test: ${HOURS}  (1 Stunde := 1 Sekunde)"
START=$(date +%s)
LOG="$(mktemp)"

# Exakt die Schleifenstruktur aus dem Jobskript, nur mit SCALE=1 statt 3600.
( prev_s=0
  for H in ${HOURS}; do
      target_s=$(awk -v h="$H" 'BEGIN{printf "%d", h*1}')
      delta=$(( target_s - prev_s ))
      [ "$delta" -gt 0 ] && sleep "$delta"
      echo "$H $(( $(date +%s) - START ))" >> "$LOG"
      prev_s="$target_s"
  done )

echo
printf '%-8s %-10s %-10s %s\n' "Marke" "Soll" "Ist" "Urteil"
while read -r h actual; do
    diff=$(( actual - h )); [ "$diff" -lt 0 ] && diff=$(( -diff ))
    if [ "$diff" -le "$TOL" ]; then
        printf '%-8s %-10s %-10s %s\n' "${h}h" "$h" "$actual" "ok"
    else
        printf '%-8s %-10s %-10s %s\n' "${h}h" "$h" "$actual" "FALSCH"
        FAILS=$(( FAILS + 1 ))
    fi
done < "$LOG"

n_marks=$(wc -l < "$LOG")
n_expected=$(echo "$HOURS" | wc -w)
echo
if [ "$n_marks" -ne "$n_expected" ]; then
    echo "FEHLER: ${n_marks} von ${n_expected} Marken ausgeloest."
    FAILS=$(( FAILS + 1 ))
fi

# Gegenprobe: die ALTE, kumulative Fassung MUSS diesen Test nicht bestehen.
# Ohne diesen Nachweis koennte der Test auch eine kaputte Implementierung
# durchwinken.
echo "Gegenprobe - kumulative Fassung (der behobene Fehler):"
START2=$(date +%s)
CUM="$(mktemp)"
( for H in ${HOURS}; do sleep "$H"; echo "$H $(( $(date +%s) - START2 ))" >> "$CUM"; done )
bad=0
while read -r h actual; do
    diff=$(( actual - h )); [ "$diff" -lt 0 ] && diff=$(( -diff ))
    [ "$diff" -gt "$TOL" ] && bad=$(( bad + 1 ))
done < "$CUM"
if [ "$bad" -gt 0 ]; then
    echo "  ok - die alte Fassung weicht bei ${bad} Marke(n) ab, der Test kann also scheitern."
else
    echo "  FEHLER: die alte Fassung besteht den Test ebenfalls. Der Test ist wertlos."
    FAILS=$(( FAILS + 1 ))
fi

rm -f "$LOG" "$CUM"
echo
if [ "$FAILS" -eq 0 ]; then echo "Alle Checks bestanden."; exit 0; fi
echo "${FAILS} CHECK(S) FAILED"; exit 1
