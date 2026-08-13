"""Generischer Ranking-Evaluator — trennt Generatorqualitaet von Rankingqualitaet.

DIE FRAGE, DIE ER BEANTWORTET
Wenn ein Modell K Posen erzeugt und ein Score eine davon auswaehlt, sind zwei
sehr verschiedene Dinge im Spiel:
    Oracle@K  - wie gut waere die BESTE der K Posen?     -> Generator + Diversitaet
    Top-1@K   - wie gut ist die AUSGEWAEHLTE Pose?       -> Ranking
    Regret    - Top-1 minus Oracle                       -> was das Ranking liegen laesst

Diese Trennung ist der Grund, warum dieses Modul modellunabhaengig ist: es
nimmt nur (complex_id, pose_id, score, rmsd) entgegen. Es funktioniert damit
spaeter unveraendert fuer SigmaDocks Energie-Score, PoseBusters-Validitaet,
eine gelernte Confidence, die exakte Likelihood oder eine Mischung daraus.

KONVENTION
`score` ist immer "hoeher = besser". Wer eine Energie hat, uebergibt `-E`.

    from SigmaFlow_Evaluation.metrics.ranking import evaluate_ranking, PoseRecord
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PoseRecord:
    """Eine erzeugte Pose. Das ist die einzige Schnittstelle nach aussen."""
    complex_id: str
    pose_id: str          # Seed, Sample-Index oder beides - nur Eindeutigkeit zaehlt
    rmsd: float
    score: float | None = None   # hoeher = besser; None = kein Ranking verfuegbar
    model_id: str = ""
    checkpoint: str = ""
    nfe: int | None = None
    sampler: str = ""


def _by_complex(records: Iterable[PoseRecord]) -> dict[str, list[PoseRecord]]:
    out: dict[str, list[PoseRecord]] = {}
    for r in records:
        out.setdefault(r.complex_id, []).append(r)
    return out


def oracle_at_k(records: Sequence[PoseRecord], k: int, *,
                n_resamples: int = 20, seed: int = 0) -> dict:
    """Oracle@K, korrekt definiert.

    Fuer jeden Komplex und jede Teilmenge S der Groesse K:
        OracleRMSD  = min_{i in S} rmsd_i
        Success     = 1[ OracleRMSD < Schwelle ]
    Erst minimieren, DANN schwellen. Die umgekehrte Reihenfolge - RMSDs mitteln
    und den Mittelwert schwellen - ist ein realer Fehler, der in einer frueheren
    Fassung genau hier steckte und K=1 von 3.9 % auf 0.0 % verfaelschte.

    Bei K < verfuegbarer Posenzahl wird ueber `n_resamples` zufaellige Teilmengen
    gemittelt; sonst ueberschaetzt eine feste Auswahl systematisch.
    """
    rng = np.random.default_rng(seed)
    groups = _by_complex(records)
    mins: list[float] = []
    per_complex: dict[str, float] = {}

    for cid, poses in groups.items():
        vals = np.array([p.rmsd for p in poses], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if k >= vals.size:
            subset_mins = [float(vals.min())]
        else:
            subset_mins = [float(vals[rng.choice(vals.size, k, replace=False)].min())
                           for _ in range(n_resamples)]
        mins.extend(subset_mins)
        per_complex[cid] = float(np.mean(subset_mins))

    a = np.array(mins) if mins else np.array([np.nan])
    return {
        "k": k, "n_observations": len(mins), "n_complexes": len(per_complex),
        "success_2A": float((a < 2.0).mean()), "success_5A": float((a < 5.0).mean()),
        "median_rmsd": float(np.median(a)), "mean_rmsd": float(a.mean()),
        "per_complex": per_complex,
    }


def top1_at_k(records: Sequence[PoseRecord], k: int, *,
              n_resamples: int = 20, seed: int = 0) -> dict:
    """Top-1@K: die vom Score ausgewaehlte Pose aus K Kandidaten.

    Posen ohne Score werden NICHT still ignoriert - sie werden gezaehlt und
    ausgewiesen, weil sonst ein lueckenhafter Score besser aussieht als er ist.
    """
    rng = np.random.default_rng(seed)
    groups = _by_complex(records)
    chosen: list[float] = []
    regrets: list[float] = []
    n_missing_score = 0

    for cid, poses in groups.items():
        usable = [p for p in poses if p.score is not None and np.isfinite(p.rmsd)]
        n_missing_score += len(poses) - len(usable)
        if not usable:
            continue
        rm = np.array([p.rmsd for p in usable], dtype=float)
        sc = np.array([p.score for p in usable], dtype=float)
        idx_sets = ([np.arange(rm.size)] if k >= rm.size
                    else [rng.choice(rm.size, k, replace=False) for _ in range(n_resamples)])
        for ii in idx_sets:
            pick = ii[int(np.argmax(sc[ii]))]
            chosen.append(float(rm[pick]))
            regrets.append(float(rm[pick] - rm[ii].min()))

    a = np.array(chosen) if chosen else np.array([np.nan])
    g = np.array(regrets) if regrets else np.array([np.nan])
    return {
        "k": k, "n_observations": len(chosen), "n_missing_score": n_missing_score,
        "success_2A": float((a < 2.0).mean()), "success_5A": float((a < 5.0).mean()),
        "median_rmsd": float(np.median(a)), "mean_rmsd": float(a.mean()),
        "median_regret": float(np.median(g)), "mean_regret": float(g.mean()),
    }


def ranking_quality(records: Sequence[PoseRecord]) -> dict:
    """Wie gut korreliert der Score mit der Qualitaet - komplexweise.

    Spearman wird INNERHALB jedes Komplexes berechnet und dann gemittelt. Global
    ueber alle Komplexe zu rechnen wuerde vor allem messen, ob der Score leichte
    von schweren Komplexen unterscheidet - das ist eine andere Frage als die
    eigentliche: kann er innerhalb eines Komplexes die gute Pose finden?
    """
    groups = _by_complex(records)
    rhos, recall1, recall3 = [], [], []
    for poses in groups.values():
        usable = [p for p in poses if p.score is not None and np.isfinite(p.rmsd)]
        if len(usable) < 3:
            continue
        rm = np.array([p.rmsd for p in usable], dtype=float)
        sc = np.array([p.score for p in usable], dtype=float)
        # Spearman ohne scipy: Rangkorrelation von score gegen -rmsd
        rr = np.argsort(np.argsort(-rm)).astype(float)
        rs = np.argsort(np.argsort(sc)).astype(float)
        if rr.std() > 0 and rs.std() > 0:
            rhos.append(float(np.corrcoef(rr, rs)[0, 1]))
        order = np.argsort(-sc)
        best = int(np.argmin(rm))
        recall1.append(float(order[0] == best))
        recall3.append(float(best in order[:3]))
    return {
        "n_complexes": len(rhos),
        "spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "spearman_median": float(np.median(rhos)) if rhos else float("nan"),
        "recall_best_in_top1": float(np.mean(recall1)) if recall1 else float("nan"),
        "recall_best_in_top3": float(np.mean(recall3)) if recall3 else float("nan"),
    }


def calibration(records: Sequence[PoseRecord], threshold: float = 2.0,
                n_bins: int = 10) -> dict:
    """Kalibrierung, falls der Score als Wahrscheinlichkeit gemeint ist.

    AUROC, AUPRC, Brier und ECE - alles ohne sklearn, damit dieses Modul keine
    zusaetzliche Abhaengigkeit einschleppt.
    """
    usable = [p for p in records if p.score is not None and np.isfinite(p.rmsd)]
    if not usable:
        return {"n": 0}
    y = np.array([p.rmsd < threshold for p in usable], dtype=float)
    s = np.array([p.score for p in usable], dtype=float)
    if y.sum() == 0 or y.sum() == y.size:
        return {"n": int(y.size), "note": "nur eine Klasse vorhanden - AUROC undefiniert",
                "positive_rate": float(y.mean())}

    # AUROC ueber die Rangstatistik (Mann-Whitney-U)
    ranks = np.argsort(np.argsort(s)).astype(float) + 1
    n_pos, n_neg = y.sum(), (1 - y).sum()
    auroc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, y.size + 1)
    rec = tp / n_pos
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

    p = np.clip(s, 0, 1) if (0 <= s.min() and s.max() <= 1) else None
    brier = float(np.mean((p - y) ** 2)) if p is not None else float("nan")
    ece = float("nan")
    bins = []
    if p is not None:
        edges = np.linspace(0, 1, n_bins + 1)
        ece, tot = 0.0, 0
        for i in range(n_bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1.0)
            if m.sum() == 0:
                continue
            conf, acc = float(p[m].mean()), float(y[m].mean())
            bins.append({"bin": i, "n": int(m.sum()), "confidence": conf, "accuracy": acc})
            ece += m.sum() * abs(conf - acc)
            tot += int(m.sum())
        ece = ece / tot if tot else float("nan")

    return {"n": int(y.size), "positive_rate": float(y.mean()),
            "auroc": float(auroc), "auprc": auprc, "brier": brier, "ece": ece,
            "reliability_bins": bins}


def evaluate_ranking(records: Sequence[PoseRecord],
                     k_values: Sequence[int] = (1, 5, 10, 20, 40),
                     seed: int = 0) -> dict:
    """Alles auf einmal. Das ist die Funktion, die Auswertungsskripte aufrufen."""
    has_score = any(p.score is not None for p in records)
    out = {
        "n_records": len(records),
        "n_complexes": len(_by_complex(records)),
        "has_score": has_score,
        "oracle": {}, "top1": {},
    }
    n_per = [len(v) for v in _by_complex(records).values()]
    out["poses_per_complex"] = {"min": min(n_per), "max": max(n_per),
                                "median": int(np.median(n_per))} if n_per else {}
    for k in k_values:
        if n_per and k > max(n_per):
            continue
        out["oracle"][k] = oracle_at_k(records, k, seed=seed)
        if has_score:
            out["top1"][k] = top1_at_k(records, k, seed=seed)
    if has_score:
        out["ranking_quality"] = ranking_quality(records)
        out["calibration"] = calibration(records)
    return out


def format_report(res: dict, label: str = "") -> str:
    """Menschenlesbare Fassung. Zahlen unveraendert, keine Interpretation."""
    L = ["=" * 92, f"RANKING-AUSWERTUNG  {label}", "=" * 92,
         f"{res['n_records']} Posen, {res['n_complexes']} Komplexe, "
         f"Posen je Komplex: {res.get('poses_per_complex', {})}",
         f"Score vorhanden: {'ja' if res['has_score'] else 'NEIN (nur Oracle@K berechenbar)'}", ""]
    L.append(f"{'K':>4} {'Oracle<2A':>11} {'Oracle med':>12} "
             + (f"{'Top1<2A':>10} {'Top1 med':>10} {'Regret med':>12}" if res["has_score"] else ""))
    L.append("-" * 92)
    for k in sorted(res["oracle"]):
        o = res["oracle"][k]
        line = f"{k:>4} {100*o['success_2A']:10.1f}% {o['median_rmsd']:11.2f}A"
        if res["has_score"] and k in res["top1"]:
            t = res["top1"][k]
            line += f" {100*t['success_2A']:9.1f}% {t['median_rmsd']:9.2f}A {t['median_regret']:11.2f}A"
        L.append(line)
    if res["has_score"]:
        q = res["ranking_quality"]
        L += ["", f"Spearman (komplexweise): median {q['spearman_median']:.3f}, "
                  f"mean {q['spearman_mean']:.3f}",
              f"Beste Pose auf Rang 1: {100*q['recall_best_in_top1']:.1f} %   "
              f"unter Top 3: {100*q['recall_best_in_top3']:.1f} %"]
        c = res["calibration"]
        if "auroc" in c:
            L.append(f"AUROC {c['auroc']:.3f}  AUPRC {c['auprc']:.3f}  "
                     f"Brier {c['brier']:.3f}  ECE {c['ece']:.3f}  "
                     f"(Positivrate {100*c['positive_rate']:.1f} %)")
    else:
        L += ["", "Ohne Score ist nur die Obergrenze (Oracle@K) berechenbar.",
              "Die Luecke Oracle@K minus Top-1@K ist genau das, was ein",
              "Confidence-Modell spaeter zurueckholen koennte."]
    return "\n".join(L)
