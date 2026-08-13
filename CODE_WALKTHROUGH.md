# SigmaFlow — Vollständiger Code-Walkthrough (Tracker)

Diese Datei ist der Fortschritts-Tracker für ein separates Vorhaben neben dem
normalen Entwicklungsfortschritt (siehe `STATUS.md`): den **gesamten**
Python-Code in `SigmaFlow_Development/` Zeile für Zeile durchzugehen und zu
verstehen — inklusive aller Python-/PyTorch-Grundlagen, die dabei auftauchen
(siehe `CLAUDE.md` §3a).

Umfang: alle 66 `.py`-Dateien, ca. 20.225 Zeilen insgesamt. Reihenfolge: von
der eigentlichen SigmaFlow-Kernlogik (`diff/`) nach außen zur unveränderten
SigmaDock-Infrastruktur, EquiformerV2-Backbone (`net/`) zuletzt, da am
größten/komplexesten und am wenigsten SigmaFlow-spezifisch.

**Regel:** eine Datei wird erst abgehakt, wenn sie wirklich komplett
Zeile für Zeile besprochen wurde, nicht nur überflogen.

## Aktueller Stand

**Gerade besprochen:** `diff/utils.py`, `diff/r3_flow_matcher.py`,
`diff/so3_utils.py`, `diff/so3_flow_matcher.py` (alle vier komplett) —
SO(3)-Analogie zu R³: Haar-uniforme Quelle, geodätische Interpolation
`R_t=R_0 exp(t log(R_0^T R_1))`, Rechtstrivialisierung (Tangentialvektor via
Rechtsmultiplikation `R @ exp(Ω)`), Äquivalenz `u_t=log_Delta` vs.
`log(R_t^T R_1)/(1-t)` explizit nachgerechnet (Einparameter-Untergruppen-
Kommutativität), dokumentierter SigmaDock-Bugfix (`Omega`-Clamp
`[-0.99,0.99]`→`[-1+1e-7,1-1e-7]`, betraf ~9% der Rotationen nahe 180°,
harmlos für Diffusion, fatal für exakte Geodäten-Interpolation).
Auch `se3_flow_matcher.py` (komplett) — dünne Fassade/Komposition aus
R3+SO3-Matchern, SE(3) als direktes Produkt (kein gekoppelter Term),
Dictionaries als selbstdokumentierende Mehrfach-Rückgabe,
`calc_vector_field` als tatsächlicher Aufrufer der Orakel-Felder (weiterer
Aufrufer: `sigma_flow_generator.py:916`, hängt an `use_true_vector_field`).
**Nächste Datei:** `src/sigmadock/diff/criterion.py` (34 Zeilen) — die
Trainings-Loss-Funktion.

---

## Phase 1 — Kern-Flow-Matching-Logik (`diff/`) — das, was SigmaFlow *ist*

- [x] `src/sigmadock/diff/utils.py` (20)
- [x] `src/sigmadock/diff/r3_flow_matcher.py` (55)
- [x] `src/sigmadock/diff/so3_utils.py` (242)
- [x] `src/sigmadock/diff/so3_flow_matcher.py` (62)
- [x] `src/sigmadock/diff/se3_flow_matcher.py` (65)
- [ ] `src/sigmadock/diff/criterion.py` (34)
- [ ] `src/sigmadock/diff/sigma_flow_generator.py` (1076)
- [ ] `src/sigmadock/diff/sampling.py` (478)

## Phase 2 — Konfiguration & Trainings-/Sampling-Orchestrierung

- [ ] `src/sigmadock/config.py` (890)
- [ ] `src/sigmadock/trainer.py` (375)
- [ ] `src/sigmadock/oracle.py` (327)
- [ ] `src/sigmadock/sampling_setup.py` (209)
- [ ] `scripts/train.py` (369)
- [ ] `scripts/sample.py` (591)
- [ ] `scripts/diagnose_vector_field.py` (129)
- [ ] `scripts/diagnose_step0.py` (187)
- [ ] `scripts/__init__.py` (1)

## Phase 3 — Datenpipeline

- [ ] `src/sigmadock/data.py` (877)
- [ ] `src/sigmadock/datafronts.py` (274)
- [ ] `src/sigmadock/core/data.py` (399)
- [ ] `src/sigmadock/core/loaders.py` (95)
- [ ] `src/sigmadock/core/embeddings.py` (70)
- [ ] `src/sigmadock/core/callbacks.py` (442)
- [ ] `src/sigmadock/core/misc.py` (308)
- [ ] `src/sigmadock/core/__init__.py` (0, leer)

## Phase 4 — Chemie-Utilities

- [ ] `src/sigmadock/chem/processing.py` (1038)
- [ ] `src/sigmadock/chem/fragmentation.py` (1529)
- [ ] `src/sigmadock/chem/parsing.py` (1042)
- [ ] `src/sigmadock/chem/postprocessor.py` (262)
- [ ] `src/sigmadock/chem/ligalign.py` (562)
- [ ] `src/sigmadock/chem/statistics.py` (1109)
- [ ] `src/sigmadock/chem/utils.py` (66)
- [ ] `src/sigmadock/chem/conformer_viz.py` (145)
- [ ] `src/sigmadock/chem/pyviz.py` (783)
- [ ] `src/sigmadock/chem/extract_esm_embeddings.py` (331)
- [ ] `src/sigmadock/chem/__init__.py` (64)

## Phase 5 — Geometrie-Utilities

- [ ] `src/sigmadock/geo/graph_utils.py` (34)
- [ ] `src/sigmadock/geo/misc.py` (54)
- [ ] `src/sigmadock/geo/viz.py` (568)
- [ ] `src/sigmadock/geo/__init__.py` (0, leer)

## Phase 6 — Torch-Utilities & Top-Level-Sonstiges

- [ ] `src/sigmadock/torch_utils/utils.py` (202)
- [ ] `src/sigmadock/torch_utils/dist.py` (74)
- [ ] `src/sigmadock/torch_utils/debug.py` (25)
- [ ] `src/sigmadock/torch_utils/__init__.py` (25)
- [ ] `src/sigmadock/utils.py` (226)
- [ ] `src/sigmadock/__init__.py` (13)

## Phase 7 — EquiformerV2-Backbone (`net/`) — unverändert, laut CLAUDE.md §9 kein Redesign, aber zum Verständnis vollständig durchzugehen

- [ ] `src/sigmadock/net/model.py` (758)
- [ ] `src/sigmadock/net/encoder.py` (369)
- [ ] `src/sigmadock/net/input_block.py` (236)
- [ ] `src/sigmadock/net/edge_rot_mat.py` (86)
- [ ] `src/sigmadock/net/transformer_block.py` (637)
- [ ] `src/sigmadock/net/so2_ops.py` (361)
- [ ] `src/sigmadock/net/so3.py` (644)
- [ ] `src/sigmadock/net/wigner.py` (36)
- [ ] `src/sigmadock/net/gaussian_rbf.py` (42)
- [ ] `src/sigmadock/net/radial_function.py` (30)
- [ ] `src/sigmadock/net/smearing.py` (185)
- [ ] `src/sigmadock/net/timestep_embedder.py` (107)
- [ ] `src/sigmadock/net/layer_norm.py` (405)
- [ ] `src/sigmadock/net/activation.py` (176)
- [ ] `src/sigmadock/net/drop.py` (128)
- [ ] `src/sigmadock/net/module_list.py` (10)
- [ ] `src/sigmadock/net/lr_scheduler.py` (169)
- [ ] `src/sigmadock/net/__init__.py` (44)

## Phase 8 — Notebook-/Dummy-Daten-Skripte

- [ ] `notebooks/dummy_data/build_inference_datafront_csv.py` (31)
- [ ] `notebooks/dummy_data/setup_crossdock_queries.py` (44)

---

## Bereits geklärte Python-/PyTorch-Grundlagen (wird laufend ergänzt)

Damit dieselbe Erklärung nicht in jeder Session neu gegeben werden muss:
Konzepte hier eintragen, sobald sie einmal erklärt wurden, mit Verweis auf
die Datei, in der sie zuerst auftauchten.

- **Type Hints** (`x: torch.Tensor`, `-> torch.Tensor`, Default-Werte `= True`):
  reine Dokumentation für Menschen/Tools (mypy), zur Laufzeit nicht
  erzwungen. Zuerst erklärt in: `diff/utils.py`.
- **Generische Container-Type-Hints** (`list[torch.Tensor]`): sagt, dass die
  Liste nur Tensoren enthalten soll. Zuerst erklärt in: `diff/utils.py`.
- **`torch.ones_like(x)`**: erzeugt einen neuen Tensor aus Einsen mit
  gleicher Shape/Dtype/Device wie `x`. Zuerst erklärt in: `diff/utils.py`.
- **`torch.autograd.grad(...)`**: expliziter Autograd-Aufruf (Alternative zu
  `.backward()`), berechnet Gradienten von `outputs` nach `inputs`, ohne sie
  in `.grad`-Attribute zu schreiben, sondern als Rückgabewert. Wichtige
  Parameter: `grad_outputs` (Vektor-Jacobian-Produkt-Gewicht, hier
  effektiv "Summe aller Elemente"), `retain_graph` (Graph nach dem Aufruf
  nicht freigeben), `create_graph` (Graph des Gradienten selbst wird
  differenzierbar, für höhere Ableitungen), `allow_unused` (kein Fehler,
  wenn `inputs` den Output gar nicht beeinflusst hat, sondern `None`).
  Zuerst erklärt in: `diff/utils.py`.
- **Tuple-Rückgabe + Indexierung `[0]`**: `torch.autograd.grad` gibt ein
  Tuple zurück (ein Gradient pro Element von `inputs`); bei genau einem
  Input holt man ihn mit `[0]` heraus. Zuerst erklärt in: `diff/utils.py`.
- **`None` als Sentinel-Rückgabewert**: PyTorch nutzt `None` statt eines
  Fehlers, um "kein Gradient vorhanden" auszudrücken; muss vom aufrufenden
  Code explizit abgefangen werden. Zuerst erklärt in: `diff/utils.py`.
- **Klassen (`class`, `__init__`, `self`)**: bündeln Daten (Attribute,
  `self.x = ...`) und Methoden zu einem Typ; `self` ist die konkrete Instanz,
  wird von Python automatisch als erstes Argument übergeben. Zuerst erklärt
  in: `diff/r3_flow_matcher.py`.
- **Docstrings (`"""..."""` als erste Anweisung im Funktionskörper)**: echtes
  `__doc__`-Objekt zur Laufzeit, aber ohne Effekt auf die Logik — reine
  Dokumentation. Zuerst erklärt in: `diff/r3_flow_matcher.py`.
- **Broadcasting (`t[:, None]`)**: `None`-Indexierung fügt eine neue
  Achse der Länge 1 ein (`[n]` → `[n,1]`); PyTorch wiederholt
  Größe-1-Dimensionen virtuell, damit `[n,1] * [n,3]` elementweise pro Zeile
  funktioniert. Zuerst erklärt in: `diff/r3_flow_matcher.py`.
- **Mathematischer Rahmen (Gaussian conditional probability path)**: gegen
  Holderrieth & Erives, "An Introduction to Flow Matching and Diffusion
  Models" (`papers/`), Kap. 3.1/3.2 geprüft. SigmaFlows R³-Pfad ist der
  Spezialfall $\alpha_t=t,\beta_t=1-t$ (linearer/"CondOT"-Zeitplan),
  `sigma_min` aktuell unbenutzt (deterministisch, $\beta_1=0$). Zuerst
  erklärt in: `diff/r3_flow_matcher.py`.
- **`torch.einsum`**: Einstein-Summenkonvention für beliebige
  Tensor-Kontraktionen, z.B. `"...ij,...ik->...jk"` = batched $R^\top X$.
  Zuerst erklärt in: `diff/so3_utils.py`.
- **`assert bedingung`**: informeller interner Sanity-Check, wirft
  `AssertionError` ohne eigene Nachricht (Unterschied zu `raise
  ValueError(...)`). Zuerst erklärt in: `diff/so3_utils.py`.
- **Maskierung statt Verzweigung** (`mask * zweig_a + (1-mask) * zweig_b`,
  oder `torch.where(cond, a, b)`): Standardmuster für elementweise
  Fallunterscheidungen über einen Batch, ohne Python-`if`/Kontrollfluss —
  autograd-freundlich, alle Zweige werden für alle Elemente berechnet.
  Zuerst erklärt in: `diff/so3_utils.py`.
- **Unpacking-Operator `*`** (in Listen-Literalen `[*x, 3, 3]` und als
  Funktionsargumente `f(*shape, 3)`): entpackt ein Tupel/eine Liste in
  einzelne Elemente/Argumente. Zuerst erklärt in: `diff/so3_utils.py`.
- **Advanced/Fancy Indexing** (`tensor[range(n), idx_tensor]`): pro
  Batch-Element einen individuell berechneten Index auswählen, statt
  überall denselben Index zu verwenden. Zuerst erklärt in: `diff/so3_utils.py`.
- **Backend-Lücken (MPS)**: nicht jedes PyTorch-Backend (hier: Apples
  Metal-GPU-Backend `"mps"`) unterstützt jede Funktion (`float64`,
  `torch.matrix_exp`) — Workaround: kurzzeitig auf CPU rechnen. Zuerst
  erklärt in: `diff/so3_utils.py`.
- **Numpy vs. Torch**: manche Hilfsfunktionen ohne Gradientenbedarf sind in
  Numpy geschrieben (`np.linspace`, `np.cumsum`, `np.interp`,
  `np.random.randn`), Konvertierung zurück via `torch.tensor(...)`. Zuerst
  erklärt in: `diff/so3_utils.py`.
- **`pass`**: No-Op-Statement, Platzhalter wenn Python syntaktisch einen
  Funktionskörper verlangt, aber nichts zu tun ist. Zuerst erklärt in:
  `diff/so3_flow_matcher.py`.
- **`@`-Operator**: batched Matrixmultiplikation (`__matmul__`, äquivalent
  `torch.matmul`), führende Batch-Achsen werden durchgereicht, die letzten
  zwei Achsen matrixmultipliziert. Zuerst erklärt in:
  `diff/so3_flow_matcher.py`.
- **Dictionaries** (`{"key": wert}`, `dict[str, torch.Tensor]`,
  Zugriff `d["key"]`): Schlüssel-Wert-Abbildung, hier als
  selbstdokumentierende Alternative zu positionsbasierten Tupeln bei
  Funktionen mit vielen Rückgabewerten. Zuerst erklärt in:
  `diff/se3_flow_matcher.py`.
- **Führender Unterstrich `_name`**: Konvention (keine Sprachregel) für
  "internes Implementierungsdetail, nicht Teil der öffentlichen
  Schnittstelle". Zuerst erklärt in: `diff/se3_flow_matcher.py`.
- **Links- vs. Rechtstrivialisierung auf SO(3)**: Tangentialvektor an $R$ als
  $R\exp(\Omega)$ (rechts, $\Omega$ im lokalen/körperfesten System) statt
  $\exp(\Omega)R$ (links, globales System). SigmaFlow nutzt durchgehend
  Rechtstrivialisierung (expliziter Code-Kommentar). Zuerst erklärt in:
  `diff/so3_flow_matcher.py`.
