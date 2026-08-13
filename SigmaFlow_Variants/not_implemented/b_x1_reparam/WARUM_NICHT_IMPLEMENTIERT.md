# Warum diese Variante nicht implementiert wurde

Ursprüngliche Idee (Test 3, Variante b): das Netzwerk analog zu Yim et al.
2023 (FrameFlow, SE(3)-Flow-Matching) statt des Vektorfelds direkt die
"saubere" Zielstruktur x̂1 vorhersagen lassen, Loss als ungewichtete MSE
im x̂1-Raum statt im v-Raum.

Beim Durchrechnen vor der Implementierung (siehe `STATUS.md`
PAUSE-PUNKT #14) zeigte sich: für SigmaFlows Parametrisierung gilt exakt

    ||x̂1_pred - x1_true||² = (1-t)² · ||v_pred - v_true||²

d.h. ein UNGEWICHTETER x̂1-Loss ist mathematisch äquivalent zu einem
`(1-t)²`-GEWICHTETEN v-Raum-Loss — der Fehler nahe t=1 (der Problemzone,
da dort die Fragment-Feinplatzierung passiert) würde damit RUNTER-
gewichtet, nicht hoch. Das ist analytisch die falsche Richtung für unsere
Bond-Length-Hypothese, kein Implementierungsdetail, das ein Trainingslauf
erst zeigen müsste.

FrameFlows tatsächlicher Loss benutzt zusätzlich einen kompensierenden,
geclippten `1/(1-t)²`-Faktor, der das wieder aufhebt — dieser Loss ist
dann aber (bis auf den Clip-Bereich t>0.9) selbst wieder äquivalent zu
einem UNGEWICHTETEN v-Raum-Loss, also fast identisch zu dem, was
SigmaFlow schon macht. Eine Kombination aus x̂1-Parametrisierung + Yim et
al.s Gewichtsfaktor hätte also im Wesentlichen Variante (a)
(`../a_time_weighting/`) über einen komplizierteren Umweg reproduziert,
ohne eine wirklich neue Hypothese zu testen.

**Entscheidung (2026-08-05, mit User):** nicht implementiert, kein
3h-ARC-Lauf verschwendet. Falls Variante (a) die Bindungslängen-Lücke zu
SigmaDock/Oracle nicht schließt, ist der nächste Schritt eine WIRKLICH
andere Idee (mehr ODE-Schritte, oder ein gezielter Loss-Term direkt auf
die Fragment-Anker-Atom-Distanz), nicht eine weitere Variante der
Zeitgewichtungsfrage.

Der Quellcode hier ist unverändert die MinimalChange-Basis (keine
Loss-Änderung vorgenommen) — reines Archiv der Design-Diskussion, kein
funktionierender/getesteter Code.
