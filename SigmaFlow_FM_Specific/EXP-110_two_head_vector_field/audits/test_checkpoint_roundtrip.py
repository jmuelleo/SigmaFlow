#!/usr/bin/env python3
"""
EXP-110 -- Checkpoint-Round-Trip ueber den ECHTEN Persistenzpfad.

Die Frage: ueberlebt ein Checkpoint, den SLURM mitten im Lauf hinterlaesst,
und laesst er sich als Zwei-Kopf-Modell zurueckladen -- oder faellt irgendwo
etwas still auf die Minimal-Variante zurueck?

Kein torch.save(state_dict). Gespeichert wird mit trainer.save_checkpoint(),
also demselben Aufruf, den ModelCheckpoint benutzt; geladen wird mit
sigmadock.utils.load_from_checkpoint(), also demselben Aufruf, den das
Sampling benutzt.

Aufruf (aus dem Variantenordner):
    PYTHONPATH=src python audits/test_checkpoint_roundtrip.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parent
sys.path.insert(0, str(VARIANT / "src"))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def sec(t: str) -> None:
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


from pytorch_lightning import Trainer  # noqa: E402

from sigmadock.config import RunConfig  # noqa: E402
from sigmadock.trainer import SigmaLightningModule  # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.net.model import EquiformerV2  # noqa: E402
from sigmadock.oracle import HPARAMS  # noqa: E402
from sigmadock.torch_utils.utils import extract_init_kwargs  # noqa: E402
from sigmadock.utils import load_from_checkpoint, load_from_scratch  # noqa: E402

# =========================================================================
sec("1. MODELL WIE IN scripts/train.py AUFBAUEN")
a = RunConfig()

equimodel = EquiformerV2(
    use_esm_embeddings=a.use_esm_embeddings,
    num_layers=a.num_layers,
    num_heads=a.num_heads,
    atom_feature_dims=a.atom_feature_dims,
    average_degrees=a.average_degrees,
    edge_feature_dims=a.edge_feature_dims,
    lmax_list=a.l_max_list,
    mmax_list=a.m_max_list,
    protein_ligand_interactions=a.include_protein_ligand_interactions,
    ligand_ligand_interactions=a.include_fragment_fragment_interactions,
    sphere_channels=a.sphere_channels,
    edge_channels=a.edge_channels,
    attn_hidden_channels=a.attn_hidden_channels,
    attn_alpha_channels=a.attn_alpha_channels,
    attn_value_channels=a.attn_value_channels,
    ffn_hidden_channels=a.ffn_hidden_channels,
    t_emb_dim=a.t_emb_dim,
    t_emb_type=a.t_emb_type,
    t_emb_scale=a.t_emb_scale,
    distance_expansion_dim=a.distance_expansion_dim,
    smearing_type=a.smearing_type,
    radial_cutoff_function=a.radial_type,
    rel_distance=a.rel_distance,
    alpha_drop=a.attention_dropout,
    drop_path_rate=a.edge_dropout,
    zero_init_last=a.zero_init_last,
    share_edge_mlp=False,
)
check("Modell hat trans_block", hasattr(equimodel, "trans_block"))
check("Modell hat rot_block", hasattr(equimodel, "rot_block"))
check("Modell hat KEINEN force_block", not hasattr(equimodel, "force_block"))

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    denoiser = SigmaFlowGenerator(
        equimodel,
        cache_path=tmp / "cache",
        cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max
        if a.include_protein_ligand_interactions
        else -1,
        cutoff_fragment_interactions=HPARAMS.get_edge_spec("inter_fragments").r_max
        if a.include_fragment_fragment_interactions
        else -1,
        cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max,
        **a.__dict__,
    )
    denoiser_cfg = extract_init_kwargs(denoiser, exclude=["model"])
    equiformer_cfg = extract_init_kwargs(equimodel)

    lit = SigmaLightningModule(
        denoiser=denoiser,
        denoiser_config=denoiser_cfg,
        equiformer_config=equiformer_cfg,
        fragment_scaling=a.fragment_scaling,
        trans_score_weight=a.trans_score_weight,
        rot_score_weight=a.rot_score_weight,
        max_steps=13750,
        num_warmup_steps=a.lr_warmup_frac,
        min_lr_start=a.min_lr_start,
        cycle_warmup_frac=a.cycle_warmup_frac,
        num_lr_cycles=a.num_lr_cycles,
        init_lr_start=a.init_lr_start,
        min_lr_end=a.min_lr_end,
        max_lr_start=a.max_lr_start,
        max_lr_end=a.max_lr_end,
        weight_decay=a.weight_decay,
        optimizer_eps=a.optimizer_eps,
        betas=a.betas,
        grad_clip=a.grad_clip,
        compile=False,
    )
    print(f"    Parameter: {sum(p.numel() for p in lit.parameters()):,}")

    # ---------------------------------------------------------------------
    sec("2. MIT trainer.save_checkpoint() SCHREIBEN (der echte Weg)")
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_steps=1,
    )
    trainer.strategy.connect(lit)
    trainer.strategy.setup_environment()
    # configure_optimizers() greift auf self.trainer zu; ausserhalb von
    # trainer.fit() muss die Rueckreferenz von Hand gesetzt werden.
    lit._trainer = trainer
    # Optimizer/Scheduler anlegen, damit ihr Zustand wirklich im Checkpoint
    # landet -- ohne das waere der Test blind fuer genau die Frage.
    trainer.strategy.setup_optimizers(trainer)

    ckpt_path = tmp / "last.ckpt"
    trainer.save_checkpoint(ckpt_path)
    check(
        "Checkpointdatei existiert",
        ckpt_path.exists(),
        f"{ckpt_path.stat().st_size / 1e6:.1f} MB",
    )

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("    Schluessel:", ", ".join(sorted(ck.keys())))

    # ---------------------------------------------------------------------
    sec("3. WAS STEHT DRIN")
    check("state_dict vorhanden", "state_dict" in ck)
    check(
        "optimizer_states vorhanden",
        bool(ck.get("optimizer_states")),
        f"n={len(ck.get('optimizer_states', []))}",
    )
    check(
        "lr_schedulers vorhanden",
        bool(ck.get("lr_schedulers")),
        f"n={len(ck.get('lr_schedulers', []))}",
    )
    check("epoch vorhanden", "epoch" in ck, str(ck.get("epoch")))
    check("global_step vorhanden", "global_step" in ck, str(ck.get("global_step")))
    check("hyper_parameters vorhanden", "hyper_parameters" in ck)
    hp = ck.get("hyper_parameters", {})
    check("equiformer_config im Checkpoint", "equiformer_config" in hp)
    check("denoiser_config im Checkpoint", "denoiser_config" in hp)
    check("max_steps im Checkpoint", hp.get("max_steps") == 13750, str(hp.get("max_steps")))

    keys = list(ck["state_dict"].keys())
    n_tr = sum(1 for k in keys if ".trans_block." in k)
    n_ro = sum(1 for k in keys if ".rot_block." in k)
    n_fo = sum(1 for k in keys if ".force_block." in k)
    check("state_dict enthaelt trans_block-Gewichte", n_tr > 0, f"{n_tr} Tensoren")
    check("state_dict enthaelt rot_block-Gewichte", n_ro > 0, f"{n_ro} Tensoren")
    check("state_dict enthaelt KEINE force_block-Gewichte", n_fo == 0, f"{n_fo}")
    check("beide Koepfe gleich viele Tensoren", n_tr == n_ro, f"{n_tr} vs {n_ro}")

    # ---------------------------------------------------------------------
    sec("4. EMA-ZWEIG NACHSTELLEN (so wie EMAWithRampup ihn schreibt)")
    # EMAWithRampup kopiert pl_module.model -- also den DENOISER -- und legt
    # dessen state_dict unter "ema_state_dict" ab. Das ist eine Praefixebene
    # weniger als ckpt["state_dict"], das am LightningModule haengt. Genau
    # diese Asymmetrie entscheidet, welcher Loader funktioniert.
    import inspect as _i

    from sigmadock.core.callbacks import EMAWithRampup

    src_ema = _i.getsource(EMAWithRampup.on_save_checkpoint)
    check("EMAWithRampup legt ema_state_dict ab", "ema_state_dict" in src_ema)
    ema_sd = lit.model.state_dict()          # == denoiser.state_dict()
    ck["ema_state_dict"] = ema_sd
    torch.save(ck, ckpt_path)

    d_sd = sorted(ck["state_dict"])[0]
    d_ema = sorted(ema_sd)[0]
    print(f"    state_dict     beginnt mit: {d_sd}")
    print(f"    ema_state_dict beginnt mit: {d_ema}")
    check("ema_state_dict hat eine Praefixebene weniger",
          d_sd == "model." + d_ema, f"{d_sd}  vs  {d_ema}")

    # ---------------------------------------------------------------------
    sec("5. ZURUECKLADEN MIT load_from_scratch() -- dem Pfad aus sample.py")
    lit2 = load_from_scratch(ckpt_path, load_ema=True, strict=True)
    m2 = lit2.model.model                     # LightningModule -> Denoiser -> Equiformer
    check("Modell zurueckgeladen", m2 is not None)
    check("zurueckgeladenes Modell hat trans_block", hasattr(m2, "trans_block"))
    check("zurueckgeladenes Modell hat rot_block", hasattr(m2, "rot_block"))
    check("zurueckgeladenes Modell hat KEINEN force_block", not hasattr(m2, "force_block"))
    check("EMA-Modell gesetzt", lit2.ema_model is not None)
    mod_file = str(Path(sys.modules[type(m2).__module__].__file__).resolve()).replace("\\", "/")
    check("Klasse stammt aus DIESER Variante",
          "EXP-110_two_head_vector_field" in mod_file,
          mod_file.split("SigmaFlow_FM_Specific/")[-1])

    check("Optimiererzustand im Checkpoint erhalten", bool(ck.get("optimizer_states")))
    check("Schedulerzustand im Checkpoint erhalten", bool(ck.get("lr_schedulers")))

    # ---------------------------------------------------------------------
    sec("6. SIND DIE GEWICHTE WIRKLICH DIESELBEN")
    for blk in ("trans_block", "rot_block"):
        a_sd = dict(getattr(equimodel, blk).named_parameters())
        b_sd = dict(getattr(m2, blk).named_parameters())
        check(f"{blk}: gleiche Parameternamen", set(a_sd) == set(b_sd), f"{len(a_sd)} Tensoren")
        worst = max((a_sd[k] - b_sd[k]).abs().max().item() for k in a_sd)
        check(f"{blk}: Gewichte bitgleich", worst == 0.0, f"max|delta| = {worst:.3e}")
    # Gegenprobe: waeren die Koepfe zufaellig identisch, bestuende der
    # Vergleich oben auch bei vertauschten Koepfen.
    d = max((p_ - q_).abs().max().item()
            for p_, q_ in zip(m2.trans_block.parameters(), m2.rot_block.parameters()))
    check("Gegenprobe: die Koepfe sind NICHT identisch", d > 1e-6, f"max|trans-rot| = {d:.3e}")

    # ---------------------------------------------------------------------
    sec("7. BEFUND: load_from_checkpoint(load_ema=False) ist kaputt")
    # Kein EXP-110-Problem -- die Funktion streift genau EIN "model." ab,
    # ckpt["state_dict"] traegt aber zwei Ebenen (LightningModule.model =
    # Denoiser, Denoiser.model = Equiformer). Gilt in Minimal genauso.
    # Der benutzte Pfad ist load_from_scratch, deshalb faellt es nicht auf.
    try:
        load_from_checkpoint(ck, load_ema=False)
        broken = False
    except RuntimeError:
        broken = True
    check("BEFUND: load_ema=False scheitert (auch in Minimal, ungenutzt)", broken)
    ok_ema = True
    try:
        r = load_from_checkpoint(ck, load_ema=True)
        ok_ema = hasattr(r.model, "rot_block")
    except Exception as e:
        ok_ema = False
        print("   ", type(e).__name__, str(e)[:80])
    check("load_from_checkpoint(load_ema=True) funktioniert dagegen", ok_ema)

    # ---------------------------------------------------------------------
    sec("8. NEGATIVKONTROLLE: laedt SigmaFlow_Minimal diesen Checkpoint?")
    # Erwartung: NEIN, und zwar laut. Ein stiller Rueckfall auf die
    # Ein-Kopf-Variante waere der gefaehrlichste denkbare Fehler.
    repo = VARIANT.parent.parent
    minimal_src = repo / "SigmaFlow_Minimal" / "src"
    code = chr(10).join([
        "import sys, torch",
        f"sys.path.insert(0, {str(minimal_src)!r})",
        "from sigmadock.utils import load_from_scratch",
        f"load_from_scratch({str(ckpt_path)!r}, load_ema=True, strict=True)",
        "print('GELADEN')",
    ])
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    loaded_ok = "GELADEN" in pr.stdout
    err_lines = [l for l in (pr.stderr or "").splitlines() if l.strip()]
    reason = err_lines[-1][:88] if err_lines else ""
    check("Minimal kann den Zwei-Kopf-Checkpoint NICHT laden",
          not loaded_ok, reason if not loaded_ok else "es hat geladen!")

print()
print("=" * 74)
ok = sum(1 for _, c, _ in CHECKS if c)
print(f"ERGEBNIS: {ok}/{len(CHECKS)} Checks bestanden")
print("=" * 74)
raise SystemExit(0 if ok == len(CHECKS) else 1)
