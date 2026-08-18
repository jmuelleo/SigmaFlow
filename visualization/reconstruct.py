"""Rekonstruktion der Fragment-Rototranslationen aus Atomkoordinaten.

WARUM NICHT DEN SAMPLER INSTRUMENTIEREN

  R_t und trans_t liegen waehrend der Integration als lebende Variablen vor.
  Sie dort mitzuschreiben waere eine Aenderung an
  `SigmaFlow_Minimal/src/sigmadock/diff/sampling.py` -- und genau dieser Baum
  ist als `sigflow-minimal-baseline-v1` eingefroren und byte-identisch mit
  dem Baum, der die 6h/12h-Ergebnisse erzeugt hat. Jede Zeile dort kostet
  diese Eigenschaft.

  Sie werden hier stattdessen NACHTRAEGLICH aus den Atompositionen
  zurueckgerechnet. Das ist keine Naeherung: die Fragmente sind starr, also
  ist die Rototranslation zwischen Referenzgeometrie und Zustand bei t
  eindeutig bestimmt -- sofern das Fragment sie ueberhaupt bestimmt.

  Nebeneffekt: die Frage "veraendert das Mitschreiben das Sampling?"
  entfaellt strukturell. Es wird waehrend des Samplings nichts mitgeschrieben.

WANN ES NICHT IDENTIFIZIERBAR IST

  Ein Fragment aus einem Atom hat keine Orientierung. Ein Fragment aus zwei
  Atomen (oder mehreren kollinearen) laesst die Drehung um die eigene Achse
  offen -- ein ganzer Freiheitsgrad fehlt. In beiden Faellen liefert diese
  Datei KEINE Rotation, sondern meldet das Fragment als entartet. Eine
  willkuerlich gewaehlte Achse waere hier schlimmer als eine Luecke.
"""

from __future__ import annotations

import numpy as np

# Relatives Verhaeltnis des kleinsten zum groessten Traegheits-Singulaerwert,
# unterhalb dessen die Punktwolke als kollinear gilt.
COLLINEAR_TOL = 1e-6


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimale Rotation R mit R @ P_centred ~ Q_centred.

    P, Q: [n, 3]. Rueckgabe: [3, 3] mit det = +1.

    Reflexionen werden korrigiert: ohne die Determinantenkorrektur liefert
    die SVD bei fast-planaren Punktwolken gelegentlich eine Spiegelung, die
    zwar den Abstand minimiert, aber keine Rotation ist.
    """
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def fragment_is_identifiable(X: np.ndarray) -> tuple[bool, str]:
    """Bestimmt eine Punktwolke eine Rotation eindeutig?"""
    n = X.shape[0]
    if n < 3:
        return False, f"nur {n} Atom(e)"
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    if s[0] <= 0:
        return False, "alle Atome identisch"
    if s[1] / s[0] < COLLINEAR_TOL:
        return False, f"kollinear (s2/s1 = {s[1] / s[0]:.2e})"
    return True, ""


def reconstruct_fragment_states(
    positions: np.ndarray,      # [T, N, 3]
    frag_ids: np.ndarray,       # [N]
    reference: np.ndarray | None = None,   # [N, 3], Default: Zustand bei t=0
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Rototranslation je Fragment und Schritt.

    Returns
    -------
    rotations    [T, F, 3, 3]   Identitaet fuer entartete Fragmente
    translations [T, F, 3]      Fragmentschwerpunkte, immer definiert
    degenerate   {fragment: Grund} fuer alle nicht identifizierbaren

    Die Rotation ist relativ zur Referenzgeometrie definiert:
        X_t^{(f)} ~ R_t^{(f)} (X_ref^{(f)} - c_ref) + c_t
    Mit `reference = None` ist die Referenz der Anfangszustand, sodass
    R_0 = I gilt und die kumulierte Drehung direkt ablesbar ist.
    """
    T, N, _ = positions.shape
    F = int(frag_ids.max()) + 1 if N else 0
    ref = positions[0] if reference is None else reference
    if ref.shape != (N, 3):
        raise ValueError(f"reference muss [{N},3] sein, ist {ref.shape}")

    rotations = np.tile(np.eye(3), (T, F, 1, 1))
    translations = np.zeros((T, F, 3))
    degenerate: dict[int, str] = {}

    for f in range(F):
        idx = np.flatnonzero(frag_ids == f)
        Xref = ref[idx]
        ok, why = fragment_is_identifiable(Xref)
        if not ok:
            degenerate[f] = why
        for t in range(T):
            Xt = positions[t, idx]
            translations[t, f] = Xt.mean(0)
            if ok:
                rotations[t, f] = kabsch(Xref, Xt)
    return rotations, translations, degenerate


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Geodaetischer Drehwinkel in Grad. R: [..., 3, 3].

    Ueber die Chord-Form statt ueber arccos((tr-1)/2): bei kleinen Winkeln
    ist `6 - 2*tr` eine Differenz fast gleich grosser Zahlen und verliert
    katastrophal an Stellen. Diese Falle hat im Projekt bereits einmal einen
    float32-Fehler von 2.3e-2 Grad als "Praezisionsgrenze" erscheinen lassen,
    obwohl er reine Ausloeschung war.
    """
    I = np.eye(3)
    fro = np.linalg.norm(R - I, axis=(-2, -1))
    s = np.clip(fro / (2.0 * np.sqrt(2.0)), 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(s))


def relative_rotation_angle_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Winkel zwischen zwei Rotationen, A und B: [..., 3, 3]."""
    return rotation_angle_deg(np.swapaxes(A, -1, -2) @ B)
