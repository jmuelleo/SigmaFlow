"""EXP-100: Zustandsparametrisierung relativ zu einem inferenzverfuegbaren
Referenzkonformer statt relativ zur Kristallorientierung.

WARUM
In SigmaFlow-Minimal gibt `get_fragment_com_and_rot` unbedingt `R_1 = I` zurueck
(von SigmaDock geerbt; die PCA-Alternative steht dort in einem `if False:`-Block).
Weil die Fragmentkoordinaten aus der gebundenen Kristallpose stammen, bedeutet
die Identitaetsrotation dort "genau die Orientierung, die das Fragment in der
Kristallpose bereits hat". Eine Source-Konzentration um diese Identitaet liesse
sich deshalb nicht als inferenzseitig gewonnene Information interpretieren.

NEUE PARAMETRISIERUNG
Referenzgeometrie ist der leakage-freie ETKDGv3+MMFF-Konformer (erzeugt aus dem
Molekuelgraphen nach `RemoveAllConformers()`, deterministisch geseedet):

    C_F   = X_F^ref - COM(X_F^ref)                     zentriertes Referenzfragment
    p_1F  = COM(Y_F)                                    unveraendert
    R_1F  = argmin_{R in SO(3)} ||R C_F - Y_F^centered||^2
    Y_F  ~= R_1F C_F + p_1F

ZUSTANDSKONVENTION - der Punkt, an dem man sich vertun kann
Ein Zustand ist ein Paar (R, p) und bedeutet die Pose

    pose_F(R, p) = R C_F + p .

Die Zielpose ist damit (R_1F, p_1F), die Quelle (R_0, x_0).

`_apply_transformations` in sigma_flow_generator.py erwartet als Argument `R_1`
NICHT die Zielrotation, sondern die **Referenzrotation der uebergebenen
Geometrie** `pos_0` (es bildet `delta_R = R_t @ R_1^T`). In der alten
Parametrisierung fielen beide zusammen. In der neuen NICHT: die uebergebene
Geometrie ist bereits das kanonische C_F, ihre Referenzrotation ist also die
IDENTITAET, waehrend die Zielrotation R_1F allgemein ist. Alle Aufrufstellen
muessen deshalb `R_ref = I` uebergeben und `R_1F` nur als Flussziel verwenden.

EXP-100 aendert AUSSCHLIESSLICH das. Source-Verteilungen, Architektur, Loss,
Optimizer, Integrator und Sampling-Schedule bleiben identisch zu
SigmaFlow-Minimal.
"""

import torch


def kabsch_rotation(C: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Optimale Rotation R in SO(3) mit R C ~= Y, batched.

    Args:
        C: [..., n, 3] Referenzpunkte, BEREITS zentriert
        Y: [..., n, 3] Zielpunkte, BEREITS zentriert
    Returns:
        R: [..., 3, 3] mit R^T R = I und det R = +1

    Konvention, explizit: die Rotation wird auf C angewandt, `C @ R.transpose(-1,-2)`
    liefert die auf Y ausgerichteten Punkte. Nicht die Inverse - waehrend der
    EXP-100-Analyse trat genau dieser Vorzeichen-/Richtungsfehler kurz auf und
    lieferte ein Residuum in der Groesse des Fragments selbst. Deshalb pruefen
    die Unit-Tests explizit gegen eine BEKANNTE Rotation, nicht nur gegen die
    Identitaet.

    Reflexion wird ausgeschlossen: faellt det(V U^T) negativ aus, wird die
    letzte Singulaerrichtung gespiegelt (Standard-Kabsch-Korrektur).

    NUMERIK: Die SVD laeuft intern IMMER in float64, unabhaengig vom Eingangstyp.
    Gemessen an 2000 Zufallsfragmenten mit C == Y, wo das exakte Ergebnis die
    Identitaet ist, liefert float32 bis zu 6.1e-02 Grad Abweichung, float64
    dagegen 3.0e-06 Grad. Der Grund sind fast entartete Singulaerwerte: U und V
    sind dann einzeln schlecht bestimmt, ihr Produkt zwar gut konditioniert, aber
    float32 verstaerkt den Fehler um mehrere Groessenordnungen. Da die Rotation
    ein TRAININGSZIEL ist, wuerde sich dieser Fehler direkt als Rauschen auf das
    Label legen. Die Kosten sind vernachlaessigbar - es sind 3x3-Matrizen.
    """
    dtype_in = C.dtype
    C64 = C.to(torch.float64)
    Y64 = Y.to(torch.float64)
    H = C64.transpose(-1, -2) @ Y64                     # [...,3,3]
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.linalg.det(V @ U.transpose(-1, -2))     # [...]
    D = torch.eye(3, device=C.device, dtype=torch.float64).expand(*det.shape, 3, 3).clone()
    D[..., 2, 2] = torch.sign(det)
    return (V @ D @ U.transpose(-1, -2)).to(dtype_in)


def kabsch_residual(C: torch.Tensor, Y: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """RMSD nach Ausrichtung, [...]. Diagnostik - misst, wie gut die starre
    SE(3)-Beschreibung das Fragment ueberhaupt erfassen kann."""
    d = C @ R.transpose(-1, -2) - Y
    return torch.sqrt((d ** 2).sum(-1).mean(-1))


@torch.no_grad()
def fragment_targets(
    ref_pos: torch.Tensor,
    true_pos: torch.Tensor,
    frag_index: torch.Tensor,
    num_fragments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Referenzframe und Flussziel je Fragment.

    Args:
        ref_pos:   [N,3] Referenzkonformer (inferenzverfuegbar), gleiche Atomreihenfolge
        true_pos:  [N,3] gebundene Zielpose
        frag_index:[N]   flacher Fragmentindex je Atom, -1 = kein Ligandenfragment
        num_fragments: Anzahl Fragmente
    Returns:
        C_centered: [N,3] zentriertes Referenzfragment je Atom (0 fuer -1-Atome)
        trans_1:    [F,3] Ziel-COMs
        R_1:        [F,3,3] Zielrotationen
        residual:   [F]   Kabsch-Restfehler, Diagnostik

    `torch.no_grad`: das Ziel ist Ground Truth und darf keinen Autograd-Graphen
    aufbauen. Der Gradient laeuft ausschliesslich ueber die Netzvorhersage.
    """
    N = ref_pos.shape[0]
    C_out = torch.zeros_like(ref_pos)
    trans_1 = torch.zeros((num_fragments, 3), device=ref_pos.device, dtype=ref_pos.dtype)
    R_1 = torch.eye(3, device=ref_pos.device, dtype=ref_pos.dtype).repeat(num_fragments, 1, 1)
    resid = torch.zeros(num_fragments, device=ref_pos.device, dtype=ref_pos.dtype)

    for f in range(num_fragments):
        sel = frag_index == f
        n = int(sel.sum())
        if n == 0:
            continue
        C = ref_pos[sel]
        Y = true_pos[sel]
        c_com = C.mean(0)
        y_com = Y.mean(0)
        Cc = C - c_com
        Yc = Y - y_com
        trans_1[f] = y_com
        if n < 3:
            # Unterbestimmt: 1-2 Atome legen keine Rotation fest. Identitaet ist
            # die einzige nicht willkuerliche Wahl; die Translation traegt die
            # gesamte Information. Wird gezaehlt, nicht verschwiegen.
            R = torch.eye(3, device=ref_pos.device, dtype=ref_pos.dtype)
        else:
            R = kabsch_rotation(Cc, Yc)
        R_1[f] = R
        resid[f] = kabsch_residual(Cc, Yc, R)
        C_out[sel] = Cc
    return C_out, trans_1, R_1, resid
