"""Welche Zellen ausgewertet werden, aus welchem Lauf, bei welcher Epoche.

WARUM SEGMENTE STATT SNAPSHOT-NAMEN
    Die erste Fassung bildete Snapshot-Name -> (Walltime, Epoche) ab. Das
    bricht, sobald ein Arm ueber zwei Jobs laeuft: der Fortsetzungslauf zaehlt
    seine Stunden wieder ab null und vergibt deshalb DIESELBEN Namen. Bei
    SigmaFlow-Minimal existiert `sched255ep_at_006h` zweimal -- einmal bei
    Epoche 20 (Lauf 8648492), einmal bei Epoche 99 (Lauf 8653824). Ein
    Namensschluessel haette die beiden stillschweigend vermengt.

    Deshalb ist die Einheit hier das SEGMENT: ein Arm besteht aus einem oder
    mehreren Laeufen, und jede Zelle wird mit ihrem Lauf zusammen gefuehrt.

EPOCHEN KOMMEN AUS DEM CHECKPOINT, NICHT AUS DER META-DATEI
    `snapshot_meta()` im Trainingsskript liest die Epoche aus dem LAUFENDEN
    stdout-Log zum Kopierzeitpunkt. `last.ckpt` stammt aber vom letzten
    Validierungsschritt und hinkt hinterher -- am 2026-08-28 bei ALLEN 19
    geprueften Snapshots, um 1 bis 11 Epochen, mit wachsendem Abstand.
    Die Werte hier sind deshalb `torch.load(...)["epoch"]`, gegengeprueft am
    `global_step`. Wer sie aus der meta.txt uebernimmt, traegt die Kurve an
    falschen Stellen auf.

WALLTIME IST KUMULATIV
    `walltime_h` zaehlt ueber den Resume hinweg weiter, sonst waeren die
    Punkte eines Arms nicht auf einer Achse. Die Epoche zaehlt ohnehin durch.
    Beides ist gemessen (snap_<tag>.meta.txt), nicht gerechnet -- ausser dem
    Offset, der addiert wird und hier explizit steht.
"""
import pathlib

HIER = pathlib.Path(__file__).resolve().parent
VAR = HIER.parent

# (Segmentname, Redock-Wurzel, per_pose-Muster, Walltime-Offset in Stunden,
#  {Snapshot-Tag: (Walltime im eigenen Lauf, Epoche)})
SEGMENTE = {
    "SigmaFlow-Minimal": [
        # Vorlauf 8648492: abgestuerzt nach 22,607 h bei Epoche 78.
        dict(lauf="8648492",
             redock=HIER / "posebusters_redock_curve",
             per_pose=HIER / "poses" / "{zelle}" / "per_pose.csv",
             offset=0.0,
             pos={"sched255ep_at_006h": (6.003, 19),
                  "sched255ep_at_012h": (12.003, 40),
                  "sched255ep_at_018h": (18.005, 60),
                  "sched255ep_final": (22.607, 77)}),
        # Fortsetzung 8653824: Stunden zaehlen wieder ab null, Epochen weiter.
        dict(lauf="8653824",
             redock=VAR / "learning_curve_8653824" / "posebusters_redock_curve",
             per_pose=(VAR / "learning_curve_8653824" / "learning_curve_cpu" /
                       "{zelle}" / "per_pose.csv"),
             offset=22.607,
             pos={"sched255ep_at_006h": (6.001, 92),
                  "sched255ep_at_012h": (12.001, 114),
                  "sched255ep_at_018h": (18.005, 136),
                  "sched255ep_at_024h": (24.003, 152),
                  "sched255ep_at_030h": (30.003, 168),
                  "sched255ep_at_036h": (36.004, 192),
                  "sched255ep_at_042h": (42.004, 208),
                  # Endpunkt. sched255ep_at_048h ist md5-identisch
                  # (fb8c0ffd) und deshalb NICHT gelistet -- es ist
                  # derselbe Checkpoint unter einem Stundennamen.
                  # Gerechnet wurde bis Epoche 245, gespeichert 232.
                  "sched255ep_emergency": (49.071, 232)}),
    ],
    "SigmaDock": [
        dict(lauf="8648493",
             redock=VAR / "learning_curve_sigmadock" / "posebusters_redock_curve",
             per_pose=(VAR / "learning_curve_sigmadock" / "learning_curve_cpu" /
                       "{zelle}" / "per_pose.csv"),
             offset=0.0,
             pos={"sched239ep_at_006h": (6.001, 19),
                  "sched239ep_at_012h": (12.002, 38),
                  "sched239ep_at_018h": (18.003, 59),
                  "sched239ep_at_024h": (24.005, 79),
                  "sched239ep_at_030h": (30.007, 95),
                  "sched239ep_at_036h": (36.008, 114),
                  "sched239ep_at_042h": (42.008, 132),
                  "sched239ep_at_048h": (48.009, 150),
                  "sched239ep_at_060h": (60.010, 181),
                  # Endpunkt. Der Lauf endete mit TIMEOUT bei 72:00:06;
                  # die USR1-Falle sicherte 15 min vorher. Ein
                  # _final-Snapshot entstand NICHT -- SLURM raeumte den
                  # Job ab, bevor train_final_72h.slurm Zeile 607
                  # erreichte. Bei Abbruch lief Epoche 236 von 239.
                  "sched239ep_emergency": (71.742, 232)}),
        # sched239ep_at_054h und at_066h sind ABSICHTLICH nicht gelistet: sie
        # sind md5-identisch mit at_048h bzw. at_060h (voller md5 am
        # 2026-08-28 ueber alle elf Checkpoints). Der Snapshot-Mechanismus kopierte ein
        # last.ckpt, das sich seit sechs Stunden nicht geaendert hatte, und
        # die meta.txt schrieb trotzdem "epoch = 176", weil sie die Epoche aus
        # dem Trainingslog liest statt aus dem Checkpoint. Der Punkt waere ein
        # verdoppeltes Epoche-157-Modell unter falschem Namen gewesen.
        # Aufgefallen, weil u2 dreimal hintereinander auf 44,87 % stand.
    ],
    "SigmaFlow-Separate": [
        # Vorlauf 8648494: abgestuerzt nach 10,997 h bei Epoche 33.
        dict(lauf="8648494",
             redock=VAR / "learning_curve_8648494" / "posebusters_redock_curve",
             per_pose=(VAR / "learning_curve_8648494" / "learning_curve_cpu" /
                       "{zelle}" / "per_pose.csv"),
             offset=0.0,
             pos={"sched255ep_at_006h": (6.002, 16),
                  "sched255ep_final": (10.997, 30)}),
        # Fortsetzung 8650976: der Snapshot-Mechanismus fiel aus (Checkpoints
        # landeten im Laufverzeichnis des Vorgaengers, find_last_ckpt sucht
        # unter $CODE_DIR/experiments). Dieser eine Punkt ist von Hand
        # gerettet; die Marken bei 6/12/18/24/30 h sind unwiederbringlich weg,
        # also die Epochen ~51 bis ~124.
        dict(lauf="8650976",
             redock=VAR / "learning_curve_8650976" / "posebusters_redock_curve",
             per_pose=(VAR / "learning_curve_8650976" / "learning_curve_cpu" /
                       "{zelle}" / "per_pose.csv"),
             offset=10.997,
             # rescue_end: der Waechter rettete bei Jobende (CUDA OOM) den
             # letzten Checkpoint. Seine meta.txt nennt walltime_h = 50.6,
             # das ist von Hand geschrieben und mischt kumulativ mit
             # Laufzeit. Gemessen (sacct) lief 8650976 39:59:46, und der
             # Snapshot entstand am Jobende -- das passt zur Epochenrate
             # (136 -> 156 in rund fuenf Stunden).
             pos={"sched255ep_manual_20260827_1950": (35.1, 136),
                  "rescue_end_20260828_0204": (39.996, 156)}),
        # Drittes Segment. 8650976 starb an CUDA OOM; 8668713 setzt fort und
        # laeuft in die Walltime. Offset = 11,056 + 39,996 = 50,993 h.
        # Gerechnet wurde bis Epoche 223, gespeichert 222.
        dict(lauf="8668713",
             redock=VAR / "learning_curve_8668713" / "posebusters_redock_curve",
             per_pose=(VAR / "learning_curve_8668713" / "learning_curve_cpu" /
                       "{zelle}" / "per_pose.csv"),
             offset=50.993,
             pos={"sched255ep_at_012h": (12.001, 190),
                  "sched255ep_at_018h": (18.002, 205),
                  "sched255ep_emergency": (20.731, 222)}),
    ],
}


def zellen():
    """Alle auswertbaren Zellen als flache Liste.

    Gibt Dicts mit arm, lauf, zelle, nfe, walltime_h (kumulativ), epoch,
    redock_dir und per_pose. Zellen ohne vorhandene Redock-Tabellen oder ohne
    per_pose.csv werden still uebersprungen -- so laeuft die Auswertung, waehrend
    noch Teile nachkommen.
    """
    raus = []
    for arm, segs in SEGMENTE.items():
        for s in segs:
            if not s["redock"].is_dir():
                continue
            for d in sorted(s["redock"].iterdir()):
                if not d.is_dir():
                    continue
                tag, rest = d.name.split("__nfe", 1)
                if tag not in s["pos"]:
                    continue
                pp = pathlib.Path(str(s["per_pose"]).format(zelle=d.name))
                if not pp.exists() or not list(d.glob("rd_*_seed*.csv")):
                    continue
                h, ep = s["pos"][tag]
                raus.append(dict(arm=arm, lauf=s["lauf"], zelle=d.name,
                                 nfe=int(rest.split("__")[0]),
                                 walltime_h=round(h + s["offset"], 3),
                                 epoch=ep, redock_dir=d, per_pose=pp))
    raus.sort(key=lambda z: (z["arm"], -z["nfe"], z["epoch"]))
    return raus


def arme_vorhanden():
    return sorted({z["arm"] for z in zellen()})


def slug(arm: str) -> str:
    return arm.lower().replace("-", "_")
