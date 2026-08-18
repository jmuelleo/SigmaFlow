"""Erzeugt die .pml-Skripte fuer die Trajektorien- und Statikansicht.

ENTWURFSPRINZIP

  Das Skript soll beim Laden EINE brauchbare Ansicht zeigen, nicht vier
  halbfertige. Die uebrigen Ansichten liegen als benannte Szenen und als
  Python-Funktionen bereit, die man in der PyMOL-Konsole aufruft:

      scrub          Frame-Regler = Zeitregler (Standard nach dem Laden)
      ghost          alle Zustaende gleichzeitig, transparent
      final          nur Endpose gegen Kristallpose
      frag(3)        nur Fragment 3, ueber die Zeit

  PyMOL-Versionen unterscheiden sich in Details; deshalb wird nichts
  benutzt, was neuer als PyMOL 2.0 ist, und alle Objektnamen sind explizit.
"""

from __future__ import annotations

from pathlib import Path

_HEADER = """# {title}
# Erzeugt von visualization/pymol_scripts.py -- nicht von Hand editieren.
#
#   pymol {filename}
#
bg_color white
set ray_opaque_background, 0
set orthoscopic, 1
set stick_radius, 0.13
set cartoon_transparency, 0.55
set sphere_scale, 0.22
"""


def _fragment_selections(n_frag: int, obj: str) -> str:
    lines = [f"# Fragment-Selektionen fuer {obj} "
             "(chain, resi und B-Faktor kodieren alle den Fragmentindex)"]
    for f in range(n_frag):
        lines.append(f"select {obj}_frag{f}, {obj} and chain "
                     f"{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[f] if f < 26 else 'z'}")
    lines.append("deselect")
    return "\n".join(lines)


def write_trajectory_pml(
    path: str | Path,
    *,
    receptor: str | None,
    crystal: str | None,
    trajectory: str,
    final_pose: str | None,
    n_fragments: int,
    complex_id: str,
    model: str,
    other_final: str | None = None,
    other_label: str = "sigmadock",
) -> Path:
    path = Path(path)
    L: list[str] = [_HEADER.format(
        title=f"{complex_id} -- {model} Sampling-Trajektorie",
        filename=path.name)]

    L.append("\n# ---------------------------------------------------- laden")
    if receptor:
        L += [f"load {receptor}, receptor",
              "hide everything, receptor",
              "show cartoon, receptor",
              "color grey80, receptor",
              "set cartoon_side_chain_helper, 1"]
    if crystal:
        L += [f"load {crystal}, crystal",
              "hide everything, crystal",
              "show sticks, crystal",
              "color forest, crystal",
              "set stick_radius, 0.16, crystal"]
    # Mehrzustandsobjekt: EIN load, K Zustaende.
    L += [f"load {trajectory}, traj",
          "hide everything, traj",
          "show sticks, traj",
          "spectrum b, rainbow, traj"]
    if final_pose:
        L += [f"load {final_pose}, final_{model}",
              f"hide everything, final_{model}",
              f"show sticks, final_{model}",
              f"color marine, final_{model}"]
    if other_final:
        L += [f"load {other_final}, final_{other_label}",
              f"hide everything, final_{other_label}",
              f"show sticks, final_{other_label}",
              f"color magenta, final_{other_label}"]

    L.append("\n" + _fragment_selections(n_fragments, "traj"))

    L.append("""
# ------------------------------------------------- Ansichten als Funktionen
python
from pymol import cmd

_TRAJ = "traj"

def scrub():
    \"\"\"Standardansicht: der Frame-Regler wird zum Zeitregler.\"\"\"
    cmd.set("all_states", 0)
    cmd.show("sticks", _TRAJ)
    cmd.set("stick_transparency", 0.0, _TRAJ)
    cmd.disable("final_*")
    cmd.frame(1)
    print("[scrub] Frame-Regler bewegen oder: frame 1 ... frame %d"
          % cmd.count_states(_TRAJ))

def ghost(transparency=0.75):
    \"\"\"Alle Zustaende gleichzeitig -- macht den Weg durch den Raum sichtbar.\"\"\"
    cmd.set("all_states", 1)
    cmd.set("stick_transparency", transparency, _TRAJ)
    print("[ghost] %d Zustaende ueberlagert" % cmd.count_states(_TRAJ))

def final():
    \"\"\"Nur Endpose(n) gegen die Kristallpose.\"\"\"
    cmd.set("all_states", 0)
    cmd.frame(cmd.count_states(_TRAJ))
    cmd.enable("final_*")
    cmd.set("stick_transparency", 0.0, _TRAJ)
    print("[final] letzter Zustand; gruen = Kristall")

def frag(i, transparency=0.6):
    \"\"\"Nur ein starres Fragment ueber die Zeit.\"\"\"
    sel = "%s_frag%d" % (_TRAJ, i)
    if sel not in cmd.get_names("selections"):
        print("[frag] unbekannt: %s" % sel); return
    cmd.hide("everything", _TRAJ)
    cmd.show("sticks", sel)
    cmd.set("all_states", 1)
    cmd.set("stick_transparency", transparency, _TRAJ)
    cmd.orient(sel)
    print("[frag] nur Fragment %d, alle Zustaende" % i)

cmd.extend("scrub", scrub)
cmd.extend("ghost", ghost)
cmd.extend("final", final)
cmd.extend("frag", frag)
python end
""")

    L.append("""
# ------------------------------------------------------------------ Szenen
scrub
scene scrub, store
ghost
scene ghost, store
final
scene final, store
scrub
""")
    L.append("orient traj" if not crystal else "orient crystal")
    L.append(f"""
print "----------------------------------------------------------"
print "{complex_id}  |  {model}"
print "  scrub        Frame-Regler = Zeit (aktiv)"
print "  ghost        alle Zustaende transparent ueberlagert"
print "  final        nur Endpose gegen Kristall"
print "  frag 3       nur Fragment 3 ueber die Zeit  ({n_fragments} Fragmente: 0..{n_fragments - 1})"
print "  scene scrub|ghost|final"
print "----------------------------------------------------------"
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_static_pml(
    path: str | Path,
    *,
    receptor: str | None,
    crystal: str | None,
    poses: dict[str, str],          # Label -> Dateiname
    complex_id: str,
) -> Path:
    """Endposen mehrerer Modelle gegen Kristall und Rezeptor."""
    path = Path(path)
    colours = ["marine", "magenta", "orange", "purple"]
    L = [_HEADER.format(title=f"{complex_id} -- Endposen", filename=path.name)]
    if receptor:
        L += [f"load {receptor}, receptor", "hide everything, receptor",
              "show cartoon, receptor", "color grey80, receptor"]
    if crystal:
        L += [f"load {crystal}, crystal", "hide everything, crystal",
              "show sticks, crystal", "color forest, crystal"]
    for i, (label, fn) in enumerate(sorted(poses.items())):
        L += [f"load {fn}, {label}", f"hide everything, {label}",
              f"show sticks, {label}",
              f"color {colours[i % len(colours)]}, {label}"]
    L += ["", "orient crystal" if crystal else "orient all", ""]
    legend = ["grey = Rezeptor"] + (["gruen = Kristall"] if crystal else [])
    legend += [f"{colours[i % len(colours)]} = {lab}"
               for i, lab in enumerate(sorted(poses))]
    L.append(f'print "{complex_id}:  ' + ";  ".join(legend) + '"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
