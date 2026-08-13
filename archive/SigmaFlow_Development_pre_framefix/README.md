# SigmaFlow

SigmaFlow is a minimally invasive replacement of [SigmaDock](https://github.com/alvaroprat97/sigmadock)'s
diffusion-based generative process with Riemannian flow matching for
SE(3)-fragmented molecular docking. It preserves SigmaDock's data pipeline,
model backbone (EquiformerV2), tensor conventions, configuration system, and
inference interface, replacing only the generative process itself: the
diffusion-based SDE forward/reverse process is replaced with a conditional
flow-matching probability path and deterministic ODE sampling.

See `STATUS.md` at the repository root for development progress and design
decisions.

## Install

```bash
bash install.sh cpu train   # or cu126/cu118 for GPU
```

## License

BSD 3-Clause, see `LICENSE`. SigmaFlow is a derivative work of SigmaDock;
the original copyright notice is retained as required.
