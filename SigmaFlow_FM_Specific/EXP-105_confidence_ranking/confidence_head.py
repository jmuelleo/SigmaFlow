"""EXP-105: integrierter Confidence-Head auf dem SigmaFlow-Backbone.

ABGRENZUNG, DIE VORHER SCHARF SEIN MUSS

  EXTERN     C_psi(P, L, x_hat) ist ein EIGENES Netz, separat trainiert.
  INTEGRIERT Der Head teilt den generativen Backbone:
                 C_{theta,psi} = g_psi(h_theta(P, L, x_hat))
             Ob h_theta dabei eingefroren ist oder mittrainiert wird, ist eine
             zweite, unabhaengige Entscheidung.

  Diese Datei implementiert die INTEGRIERTE Variante.

WARUM ZWEISTUFIG UND NICHT GEMEINSAM MIT DEM GENERATOR

  Das Ziel ist P(RMSD < 2 A) FUER EINE GENERIERTE POSE. Waehrend des
  generativen Trainings sieht das Netz aber nie eine generierte Pose, sondern
  x_t -- einen Punkt auf der Interpolation zwischen Quelle und Daten. Bei t = 1
  ist x_t exakt die Kristallpose, das Label also konstant 1. Es gibt zum
  Trainingszeitpunkt schlicht keine Posen VARIIERENDER QUALITAET, an denen ein
  Klassifikator lernen koennte.

  x_t trotzdem als "schlechte Pose" zu deklarieren waere ein erfundenes Label:
  x_t ist keine Vorhersage des Modells, sondern ein Punkt des
  Wahrscheinlichkeitspfads. Der Unterschied ist nicht kosmetisch -- ein bei
  t = 0.3 interpolierter Zustand ist systematisch anders falsch als eine
  tatsaechlich generierte Fehlpose.

  Deshalb: Generator zuerst, dann Posen erzeugen, dann den Head auf diesen
  Posen trainieren. Das ist die minimal-riskante Konstruktion, und sie bleibt
  "integriert" im Sinne des geteilten Backbones.

ANSCHLUSSSTELLE (aus net/model.py verifiziert)

    x.embedding = self.norm(x.embedding)          # [N, (L+1)^2, sphere_channels]
    forces = self.force_block(...)                # danach
    forces = forces.embedding.narrow(1, 1, 3)     # l=1-Anteil

  Der Head greift auf `x.embedding[:, 0, :]` zu -- den l = 0 Kanal, also die
  SO(3)-INVARIANTEN Skalare je Knoten. Das ist die richtige Wahl und nicht nur
  die bequeme: "ist diese Pose gut" darf nicht davon abhaengen, in welchem
  globalen Bezugssystem der Komplex vorliegt. Ein Head auf l >= 1 waere
  equivariant statt invariant und wuerde eine richtungsabhaengige Antwort geben.

  Der Force-Pfad wird nicht angefasst; der Generator bleibt bitgleich.
"""

from __future__ import annotations

import torch
from torch import nn


class InvariantPooling(nn.Module):
    """Maskiertes Pooling ueber Knoten zu einem Vektor je Komplex.

    Warum maskiert: ein Batch enthaelt Komplexe verschiedener Groesse, und die
    Padding-/Dummy-Knoten duerfen nicht mitgemittelt werden. Ohne Maske haetten
    grosse Komplexe systematisch andere Aktivierungen als kleine -- der Head
    lernte dann Molekuelgroesse statt Posenqualitaet.

    `mode="mean"` ist der Default, weil Summe mit der Knotenzahl skaliert und
    damit genau diese Groessenkorrelation einbaut.
    """

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode not in ("mean", "max", "sum"):
            raise ValueError(f"mode unbekannt: {mode}")
        self.mode = mode

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor,
                node_mask: torch.Tensor, n_graphs: int) -> torch.Tensor:
        """h: [N, C], batch_idx: [N], node_mask: [N] bool -> [B, C]."""
        C = h.shape[-1]
        out = h.new_zeros((n_graphs, C))
        m = node_mask.to(h.dtype).unsqueeze(-1)                # [N,1]
        hm = h * m
        if self.mode == "max":
            # Maskierte Knoten auf -inf, damit sie das Maximum nie gewinnen.
            neg = torch.finfo(h.dtype).min
            hmask = torch.where(node_mask.unsqueeze(-1), h, torch.full_like(h, neg))
            out = out.fill_(neg).index_reduce_(0, batch_idx, hmask, "amax",
                                               include_self=True)
            # Komplexe ohne einen einzigen gueltigen Knoten: 0 statt -inf.
            return torch.where(torch.isinf(out) | (out == neg),
                               torch.zeros_like(out), out)
        out.index_add_(0, batch_idx, hm)
        if self.mode == "sum":
            return out
        cnt = h.new_zeros((n_graphs, 1))
        cnt.index_add_(0, batch_idx, m)
        return out / cnt.clamp_min(1.0)


class ConfidenceHead(nn.Module):
    """g_psi: invariante Knotenfeatures -> ein Logit je Komplex.

    Args:
        in_channels: Breite des l = 0 Kanals des Backbones (sphere_channels).
        hidden:      Breite der versteckten Schicht.
        pooling:     "mean" | "sum" | "max".
        dropout:     Regularisierung; der Datensatz ist klein gegenueber dem
                     Backbone, deshalb per Default aktiv.

    Ausgabe ist ein LOGIT, keine Wahrscheinlichkeit. Die Sigmoidfunktion sitzt
    in der Verlustfunktion (BCEWithLogits), weil das numerisch stabiler ist als
    sigmoid gefolgt von log.

    Bewusst klein gehalten (zwei Linearschichten). Der Head soll die Frage
    beantworten, ob die BACKBONE-Features Posenqualitaet tragen. Ein grosser
    Head wuerde diese Frage verwaessern: ein Erfolg waere dann nicht mehr dem
    Backbone zuzuschreiben.
    """

    def __init__(self, in_channels: int, hidden: int = 128,
                 pooling: str = "mean", dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = InvariantPooling(pooling)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Ohne diese Initialisierung startet der Head bei p = 0.5. Da nur rund
        # 5-10 % der generierten Posen unter 2 A liegen, ist das ein schlechter
        # Startpunkt; die ersten Schritte verbrauchen sich damit, den Bias zu
        # senken. Wir setzen ihn direkt auf die Basisrate.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.2)         # sigmoid(-2.2) ~ 0.10

    def forward(self, h_l0: torch.Tensor, batch_idx: torch.Tensor,
                node_mask: torch.Tensor, n_graphs: int) -> torch.Tensor:
        """h_l0: [N, C] invariante Knotenfeatures -> [B] Logits."""
        pooled = self.pool(h_l0, batch_idx, node_mask, n_graphs)   # [B,C]
        return self.mlp(pooled).squeeze(-1)                        # [B]


def extract_invariant_features(embedding: torch.Tensor) -> torch.Tensor:
    """Zieht den l = 0 Kanal aus einer SO3_Embedding-Tensordarstellung.

    Args:
        embedding: [N, (L+1)^2, C]
    Returns:
        [N, C] -- SO(3)-invariant.

    Die l = 0 Komponente steht an Index 0 der mittleren Achse; das ist die
    Konvention der e3nn-artigen Anordnung (l = 0, dann die drei l = 1 usw.),
    dieselbe, die `forces.embedding.narrow(1, 1, 3)` in net/model.py fuer den
    l = 1 Anteil benutzt.
    """
    if embedding.dim() != 3:
        raise ValueError(f"erwartet [N, (L+1)^2, C], bekam {tuple(embedding.shape)}")
    return embedding[:, 0, :]


def confidence_loss(logits: torch.Tensor, rmsd: torch.Tensor,
                    threshold: float = 2.0,
                    pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE gegen das Label RMSD < threshold.

    `pos_weight` gleicht das Klassenungleichgewicht aus. Ohne ihn lernt der
    Head bei ~10 % positiven Beispielen am schnellsten, IMMER "schlecht" zu
    sagen -- das ist zu 90 % richtig und als Ranker vollkommen wertlos.
    Empfohlen: pos_weight = n_negativ / n_positiv, aus dem Datensatz gezaehlt.
    """
    target = (rmsd < threshold).to(logits.dtype)
    return nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight)
