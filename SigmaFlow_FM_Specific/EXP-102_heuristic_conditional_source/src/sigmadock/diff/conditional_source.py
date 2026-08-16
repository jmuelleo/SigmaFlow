"""EXP-102: konditionierte Quellverteilung auf SO(3).

DIE EINE AENDERUNG GEGENUEBER EXP-100
    R_0 ~ Haar          wird zu          R_0 ~ q(R | P, L)

Alles andere -- Architektur, Loss, Optimizer, LR, Integrator, NFE, die
Translationsquelle, die Zustandsparametrisierung aus EXP-100 -- bleibt
unveraendert. Das ist der ganze Sinn: ein positiver Effekt muss der IDEE
zugeschrieben werden koennen, nicht einer zweiten Aenderung.

WARUM DAS UEBERHAUPT FLOW-MATCHING-SPEZIFISCH IST
Bei Diffusion legt der Vorwaertsprozess die Quelle fest: q_0 ist das, wohin
das Rauschen fuehrt, und daran laesst sich nicht drehen, ohne den Prozess zu
wechseln. Flow Matching darf q_0 frei waehlen, solange man daraus ziehen kann.
Das ist einer der wenigen Punkte, an denen FM strukturell mehr erlaubt.

WARUM DIE KONDITIONIERUNG NUR GLOBAL SEIN KANN
Die Quelle wird bei t = 0 gezogen. Zu diesem Zeitpunkt haben die Fragmente noch
KEINE Position -- die wird ja gerade erzeugt. Eine fragmentweise Konditionierung
auf die Umgebung des Fragments ist deshalb unmoeglich. Verfuegbar sind nur
Ligandtopologie, Konformergeometrie und die Tasche als Ganzes. Alle Fragmente
eines Komplexes bekommen daher DIESELBE Zentrumsrotation.

DIE VERTEILUNG
Gewickelte Gaussverteilung auf SO(3) (fuer kleine Streuung praktisch IGSO(3)):

    xi ~ N(0, sigma^2 I_3)  in der Lie-Algebra,      R_0 = R_c @ exp(hat(xi))

Rechtsmultiplikation, konsistent mit der Koerperframe-Konvention des uebrigen
Codes (Rdot = R Omega, euler_step multipliziert rechts).

Parametrisiert wird NICHT ueber sigma, sondern ueber den MEDIANWINKEL, den die
Quelle zum Zentrum haben soll. Der ist direkt mit der Haar-Referenz von
132.3 Grad und mit der Ausgabe von arc/exp101_distance_audit.py vergleichbar;
sigma ist es nicht. Fuer ||xi|| mit xi ~ N(0, sigma^2 I_3) gilt
Median ||xi|| = sigma * 1.5382 (Chi-Verteilung, 3 Freiheitsgrade).

DAS RISIKO, DAS AUSDRUECKLICH FESTGEHALTEN WIRD
Eine konzentrierte Quelle senkt die DIVERSITAET. Die groesste gemessene Luecke
des Projekts ist aber genau dort: ein Zug 1.9 %, best-of-10 29.2 %. Wird die
Quelle zu eng gewaehlt, kann die Einzelziehung besser und Oracle@10 gleichzeitig
SCHLECHTER werden -- ein Nettoverlust, der bei alleiniger Betrachtung der
Einzelziehung wie ein Erfolg aussaehe. Deshalb ist Oracle@K bei diesem
Experiment eine PFLICHTMETRIK, keine Zusatzmetrik.
"""

from __future__ import annotations

import numpy as np
import torch

from sigmadock.diff import so3_utils

# Median der Chi-Verteilung mit 3 Freiheitsgraden.
_CHI3_MEDIAN = 1.5381722544
# Haar auf SO(3): Winkelmedian in Grad. Referenz, gegen die alles antritt.
HAAR_MEDIAN_DEG = 132.3


def sigma_from_median_deg(median_deg: float) -> float:
    """Streuung in der Lie-Algebra aus dem gewuenschten Medianwinkel."""
    if not 0.0 < median_deg < 180.0:
        raise ValueError(f"median_deg muss in (0, 180) liegen, war {median_deg}")
    return float(np.deg2rad(median_deg) / _CHI3_MEDIAN)


def principal_axes(X: torch.Tensor) -> torch.Tensor:
    """Rechtshaendiges Hauptachsensystem, Spalten = Achsen. [n,3] -> [3,3].

    Die Vorzeichen der Eigenvektoren sind nicht eindeutig. Ohne Fixierung waere
    die "Heuristik" in Wahrheit zufaellig zwischen vier gleichwertigen Rahmen --
    ein Fehler, der sich als plausibles Ergebnis tarnt, weil das Ergebnis dann
    einfach nach Haar aussaehe. Fixiert wird ueber das dritte Moment (Schiefe).

    Rechnet in float64: bei fast entarteten Hauptachsen (kugelfoermige Liganden)
    sind die Eigenvektoren einzeln schlecht bestimmt.
    """
    X64 = X.to(torch.float64)
    Xc = X64 - X64.mean(0)
    cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
    _, V = torch.linalg.eigh(cov)
    V = V.flip(-1)                                   # groesste Varianz zuerst
    proj = Xc @ V
    for k in range(3):
        if float((proj[:, k] ** 3).sum()) < 0.0:
            V[:, k] = -V[:, k]
    if float(torch.linalg.det(V)) < 0.0:
        V[:, 2] = -V[:, 2]
    return V.to(X.dtype)


def pocket_alignment_rotation(lig_conf_pos: torch.Tensor,
                              pocket_pos: torch.Tensor) -> torch.Tensor:
    """Zentrumsrotation der Heuristik H1: Konformer-Achsen auf Taschen-Achsen.

    Args:
        lig_conf_pos: [n_lig, 3] Ligandenkonformer IM EIGENEN FRAME.
        pocket_pos:   [n_pkt, 3] Taschenatome.
    Returns:
        [3,3] Rotation Q mit det Q = +1.

    LEAKAGE - der Punkt, an dem dieses Experiment wertlos wuerde:
      `lig_conf_pos` MUSS `ref_conf_pos` sein, der ETKDG-Konformer im eigenen
      Frame. Es darf NICHT `pos_0` sein. EXP-100 baut `pos_0` als zentriertes
      Konformerfragment, VERSCHOBEN AN DEN KRISTALL-COM -- die globale Form
      dieser Punktwolke traegt also die kristallographische Anordnung der
      Fragmente und damit einen Teil der gesuchten Antwort.
      Der Aufrufer ist dafuer verantwortlich; `sample_conditional_init` prueft
      es nicht, weil es Koordinaten nicht ansehen kann.
    """
    if lig_conf_pos.shape[0] < 3 or pocket_pos.shape[0] < 3:
        return torch.eye(3, device=lig_conf_pos.device, dtype=lig_conf_pos.dtype)
    V_lig = principal_axes(lig_conf_pos)
    V_pkt = principal_axes(pocket_pos)
    Q = V_pkt @ V_lig.transpose(-1, -2)
    if float(torch.linalg.det(Q)) < 0.0:
        Q = Q.clone()
        Q[:, 2] = -Q[:, 2]
    return Q


def sample_conditional_init(n: int,
                            R_centre: torch.Tensor | None,
                            median_deg: float,
                            device: str | torch.device,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Zieht n Quellrotationen aus der gewickelten Gaussverteilung um R_centre.

    Args:
        n:          Anzahl Fragmente.
        R_centre:   [3,3] oder [n,3,3]; None bedeutet Haar (exakt EXP-100).
        median_deg: gewuenschter Medianwinkel zum Zentrum, in Grad.
        device:     Zielgeraet.
        generator:  optionaler RNG fuer reproduzierbare Ziehungen.
    Returns:
        [n,3,3] float32.

    Bei R_centre = None wird BIT-IDENTISCH zu EXP-100 gezogen (dieselbe
    so3_utils.sample_uniform). Das ist die Kontrollbedingung: ein Lauf mit
    source_mode=haar muss EXP-100 exakt reproduzieren.
    """
    if R_centre is None:
        return so3_utils.sample_uniform(n).to(device, dtype=torch.float32)

    sigma = sigma_from_median_deg(median_deg)
    xi = torch.randn(n, 3, generator=generator, dtype=torch.float64) * sigma
    perturb = so3_utils.Exp(xi).to(device=device, dtype=torch.float32)
    Rc = R_centre.to(device=device, dtype=torch.float32)
    if Rc.dim() == 2:
        Rc = Rc.unsqueeze(0).expand(n, 3, 3)
    # Rechtsmultiplikation: konsistent mit Rdot = R Omega und euler_step.
    return Rc @ perturb
