"""
equiformer_toy_block.py
=======================

The toy EquiformerV2 block worked through in Chapter 9 of
``Texte/theory.tex``, implemented so that every number printed in that
worked example is machine-generated rather than hand-computed, and so that
the two structural claims can be tested rather than asserted:

  (A) equivariance:      F(Q x, Q h) == Q F(x, h)  for every rotation Q
  (B) gauge-independence: the output does not depend on the arbitrary choice
                          of transverse axis used to build each edge frame

Setting (deliberately minimal; see the chapter for what is exact and what is
a didactic simplification):

  * three atoms: target i, neighbours j and k
  * l_max = 1
  * one scalar channel per node, two l=1 channels per node
  * one attention head
  * l=1 features written in a CARTESIAN basis, in which D^(1)(Q) is the
    rotation matrix Q itself -- exact for l=1, and the reason this example
    can be followed by hand.

Run:  python audits/equiformer_toy_block.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(20260816)
FAIL = []


def report(name, residual, tol, note=""):
    ok = residual <= tol
    if not ok:
        FAIL.append(name)
    extra = f"   ({note})" if note else ""
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<52s} residual = {residual:.3e}  tol = {tol:.1e}{extra}")


def sec(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ---------------------------------------------------------------- geometry
POS = {"i": np.array([0.0, 0.0, 0.0]),
       "j": np.array([1.0, 0.0, 0.0]),
       "k": np.array([0.0, 2.0, 0.0])}

# ---------------------------------------------------- features (Cartesian)
# scalar channel (l=0) and two l=1 channels per node
S = {"i": 1.0, "j": 2.0, "k": -1.0}
V = {"i": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
     "j": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
     "k": np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]])}

# ------------------------------------------------------- radial embedding
MU = np.array([0.5, 1.5])


def radial(r):
    """Didactic simplification: two Gaussian radial basis functions.
    Real EquiformerV2 uses a Bessel basis with a polynomial cutoff."""
    return np.exp(-(r - MU) ** 2)


# --------------------------------------------------------- edge alignment
def edge_frame(d_hat, ref):
    """Orthonormal frame (u, v, n) with n = d_hat.  Q has these as ROWS, so
    Q d_hat = (0,0,1): the edge is aligned onto the third axis, which is the
    slot carrying m = 0 for every degree.  `ref` is the arbitrary reference
    vector used to fix the transverse axes -- claim (B) is that it cancels."""
    n = d_hat / np.linalg.norm(d_hat)
    u = ref - (ref @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    Q = np.stack([u, v, n])
    assert abs(np.linalg.det(Q) - 1) < 1e-12
    return Q


def so2_conv(vec_edge, W0, W1):
    """SO(2)-equivariant linear map on l=1 features already in the edge frame.

    Layout per channel: (m=-1, m=+1, m=0) -> here (transverse_u, transverse_v,
    along-edge).  The m=0 sector may be mixed by an arbitrary real matrix W0;
    the |m|=1 sector may only be mixed by a matrix that commutes with the 2-D
    rotation, i.e. by a real scalar per channel pair (plus, in general, a
    90-degree part which we set to zero -- a didactic simplification).
    The two sectors never mix.  vec_edge: (C, 3)."""
    out = np.empty_like(vec_edge)
    out[:, 2] = W0 @ vec_edge[:, 2]           # m = 0 sector: full mixing
    out[:, 0] = W1 @ vec_edge[:, 0]           # |m| = 1 sector: constrained
    out[:, 1] = W1 @ vec_edge[:, 1]           # same weights on both partners
    return out


W0 = np.array([[2.0, 0.0], [0.0, 0.5]])
W1 = np.array([[1.0, 0.0], [0.0, 0.5]])
A_ATT = np.array([0.9, -0.4, 0.6, 0.3])       # attention read-out vector


def block(pos, scal, vec, ref=np.array([0.0, 0.0, 1.0]), verbose=False):
    """One simplified attention block; returns the updated l=1 feature at i."""
    logits, values = {}, {}
    for nb in ("j", "k"):
        d = pos[nb] - pos["i"]
        r = np.linalg.norm(d)
        d_hat = d / r
        Q = edge_frame(d_hat, ref)

        # rotate the neighbour's l=1 features into the edge frame.
        # For l=1 the Wigner-D matrix in the Cartesian basis IS Q (exact).
        v_edge = vec[nb] @ Q.T                       # (C,3)

        z_edge = so2_conv(v_edge, W0, W1)            # SO(2) convolution

        # invariants: the m=0 (along-edge) components, plus radial + scalar
        inv = np.concatenate([z_edge[:, 2], radial(r) * scal[nb]])
        logits[nb] = float(A_ATT @ np.tanh(inv))
        values[nb] = (Q, z_edge)

        if verbose:
            print(f"  edge {nb}->i : r = {r:.4f}   d_hat = {d_hat}")
            print(f"    Q rows u,v,n =\n{Q}")
            print(f"    neighbour l=1 (global)  =\n{vec[nb]}")
            print(f"    neighbour l=1 (edge)    =\n{v_edge}")
            print(f"    after SO(2) conv        =\n{z_edge}")
            print(f"    m=0 invariants          = {z_edge[:,2]}")
            print(f"    radial(r)*scalar        = {radial(r)*scal[nb]}")
            print(f"    logit s_{nb}i            = {logits[nb]:.4f}")

    ls = np.array([logits["j"], logits["k"]])
    alpha = np.exp(ls - ls.max()); alpha /= alpha.sum()
    if verbose:
        print(f"  softmax over incoming edges: alpha_ij = {alpha[0]:.4f}, "
              f"alpha_ik = {alpha[1]:.4f}")

    out = np.zeros((2, 3))
    for w, nb in zip(alpha, ("j", "k")):
        Q, z_edge = values[nb]
        z_global = z_edge @ Q                        # rotate back: Q^T applied
        out += w * z_global
        if verbose:
            print(f"    message from {nb}, back in global frame =\n{z_global}")
    if verbose:
        print(f"  aggregated message at i =\n{out}")
        print(f"  after residual h + Attn  =\n{vec['i'] + out}")
    return vec["i"] + out                            # residual connection


# ============================================================================
sec("1.  The worked example, step by step (numbers quoted in Chapter 9)")
# ============================================================================
print(f"  positions: i={POS['i']}, j={POS['j']}, k={POS['k']}")
print(f"  scalars:   s_i={S['i']}, s_j={S['j']}, s_k={S['k']}")
print(f"  l=1 at i:\n{V['i']}")
out_ref = block(POS, S, V, verbose=True)


# ============================================================================
sec("2.  Equivariance:  F(Qx, Qh) == Q F(x, h)")
# ============================================================================
worst = 0.0
for _ in range(200):
    M = rng.normal(size=(3, 3))
    Qg, _ = np.linalg.qr(M)
    if np.linalg.det(Qg) < 0:
        Qg[:, 0] *= -1
    pos_r = {n: Qg @ p for n, p in POS.items()}
    vec_r = {n: v @ Qg.T for n, v in V.items()}
    lhs = block(pos_r, S, vec_r)
    rhs = block(POS, S, V) @ Qg.T
    worst = max(worst, np.max(np.abs(lhs - rhs)))
report("block is SO(3)-equivariant", worst, 1e-12,
       "200 random rotations, exact parts only")


# ============================================================================
sec("3.  Gauge-independence: the transverse axis choice cancels")
# ============================================================================
worst = 0.0
base = block(POS, S, V)
for _ in range(200):
    ref = rng.normal(size=3)
    if np.linalg.norm(ref) < 1e-6:
        continue
    worst = max(worst, np.max(np.abs(block(POS, S, V, ref=ref) - base)))
report("output independent of the edge frame's u-axis", worst, 1e-12,
       "200 random reference vectors")

print("\n  Why: the SO(2) convolution treats the two transverse components")
print("  with the SAME weight, and the attention logit reads only m=0.")
print("  Both are invariant under spinning the frame about the edge.")

# a deliberate counter-test: break the SO(2) constraint and watch it fail
def so2_conv_broken(vec_edge, W0, W1):
    out = np.empty_like(vec_edge)
    out[:, 2] = W0 @ vec_edge[:, 2]
    out[:, 0] = W1 @ vec_edge[:, 0]
    out[:, 1] = (W1 * 3.0) @ vec_edge[:, 1]     # different weight: illegal
    return out


_real = so2_conv
globals()["so2_conv"] = so2_conv_broken
broken_spread = max(np.max(np.abs(block(POS, S, V, ref=rng.normal(size=3)) -
                                   block(POS, S, V)))
                    for _ in range(50))
globals()["so2_conv"] = _real
print(f"\n  [INFO] with the |m|=1 constraint deliberately broken, the same")
print(f"         test gives a spread of {broken_spread:.3f} -- i.e. the check")
print(f"         above can fail, and does when the constraint is removed.")
report("counter-test: breaking SO(2)-equivariance IS detected",
       0.0 if broken_spread > 1e-3 else 1.0, 0.5)


# ============================================================================
sec("SUMMARY")
# ============================================================================
if FAIL:
    print(f"  {len(FAIL)} CHECK(S) FAILED: {FAIL}")
    raise SystemExit(1)
print("  All checks passed.")
