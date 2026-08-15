"""Numerische Kontrolle der Grenzuebergaenge im Diffusionskapitel.

Die Behauptungen im Theoriekapitel sollen nicht nur hergeleitet, sondern auch
nachgerechnet sein:

  1. Warum sqrt(dt) und nicht dt: Varianz der akkumulierten Stoerung.
  2. DDPM (diskret) -> VP-SDE (kontinuierlich): konvergieren die Marginale?
  3. SMLD (diskret) -> VE-SDE: stimmt g(t) = sqrt(d sigma^2/dt)?
  4. Euler-Maruyama gegen die analytische Loesung des OU-Prozesses.

    python audits/sde_limit_check.py
"""
import numpy as np

rng = np.random.default_rng(0)


def rule(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ---------------------------------------------------------------- 1
rule("1. WARUM sqrt(dt)?  Varianz der Summe ueber n Schritte, T = 1")
print(f"  {'n':>8} {'Var, Rausch ~ dt':>20} {'~ sqrt(dt)':>14} {'~ 1':>12}")
for n in (10, 100, 1000, 10000):
    dt = 1.0 / n
    e = rng.standard_normal((20000, n))
    print(f"  {n:>8} {(e * dt).sum(1).var():>20.6f} "
          f"{(e * np.sqrt(dt)).sum(1).var():>14.6f} {e.sum(1).var():>12.1f}")
print("  Nur die mittlere Spalte hat einen endlichen, von n unabhaengigen")
print("  Grenzwert (= T = 1). Links kollabiert sie, rechts explodiert sie.")

# ---------------------------------------------------------------- 2
rule("2. DDPM -> VP-SDE:  x_k = sqrt(1-beta_k) x_{k-1} + sqrt(beta_k) eps")
b_min, b_max, x0 = 0.1, 20.0, 1.0

# Beide Marginale sind exakt gaussisch. Deshalb wird hier NICHT simuliert,
# sondern Mittel und Varianz geschlossen berechnet. Monte Carlo wuerde die
# Konvergenzordnung unter Stichprobenrauschen begraben.
#   diskret:      mean = x0 prod sqrt(1-beta_k),  var = 1 - prod (1-beta_k)
#   Grenz-SDE:    mean = x0 exp(-I/2),            var = 1 - exp(-I),
#                 I = int_0^1 beta(t) dt
I = b_min + 0.5 * (b_max - b_min)
m_a, v_a = x0 * np.exp(-I / 2), 1 - np.exp(-I)
print("  Grenz-SDE  dX = -1/2 beta(t) X dt + sqrt(beta(t)) dW  bei t = 1:")
print(f"    Mittel {m_a:.10f}   Varianz {v_a:.10f}")
print("")
print(f"  {'N':>8} {'Mittel (diskret)':>20} {'Fehler':>12} {'Fehler * N':>12}")
for N in (10, 50, 200, 1000, 5000, 20000):
    dt = 1.0 / N
    tk = (np.arange(N) + 1) * dt
    beta_k = (b_min + (b_max - b_min) * tk) * dt
    if beta_k.max() >= 1.0:
        print(f"  {N:>8}   beta_k = {beta_k.max():.2f} >= 1  ->  sqrt(1-beta_k) "
              f"undefiniert")
        continue
    prod = np.prod(1 - beta_k)
    m_d = x0 * np.sqrt(prod)
    err = abs(m_d - m_a)
    print(f"  {N:>8} {m_d:>20.10f} {err:>12.2e} {err*N:>12.4f}")
print("")
print("  Die letzte Spalte ist konstant: der Fehler faellt exakt wie 1/N,")
print("  wie es die Entwicklung sqrt(1-beta) = 1 - beta/2 + O(beta^2) verlangt.")
print("  Die erste Zeile ist kein Rechenfehler, sondern eine echte Schranke:")
print("  die DDPM-Parametrisierung braucht beta_k < 1, das begrenzt dt nach oben.")

# ---------------------------------------------------------------- 3
rule("3. SMLD -> VE-SDE:  hat dX = sqrt(d sigma^2/dt) dW die Marginale"
     " N(x_0, sigma(t)^2)?")
s_min, s_max = 0.01, 50.0
r = s_max / s_min


def sigma(t):
    return s_min * r ** t


# Wieder exakt: die akkumulierte Varianz der driftfreien SDE ist die
# Riemann-Summe sum g(t_k)^2 dt mit g(t)^2 = d sigma^2/dt.
target = sigma(1.0) ** 2 - sigma(0.0) ** 2
print(f"  sigma(t) = sigma_min r^t,  r = {r:.0f}")
print(f"  Ziel: sigma(1)^2 - sigma(0)^2 = {target:.6f}")
print("")
print(f"  {'N':>8} {'Summe g^2 dt':>18} {'rel. Fehler':>14}")
for N in (10, 100, 1000, 10000):
    dt = 1.0 / N
    tk = np.arange(N) * dt
    acc = (2 * sigma(tk) ** 2 * np.log(r) * dt).sum()
    print(f"  {N:>8} {acc:>18.6f} {abs(acc-target)/target:>14.2e}")
print("  Konvergiert gegen die Zielvarianz. Der Drift ist null; die gesamte")
print("  Varianz kommt aus dem Rauschterm.")

# ---------------------------------------------------------------- 4
rule("4. EULER-MARUYAMA gegen die exakte OU-Loesung")
theta, sig, T, X0 = 1.5, 0.8, 2.0, 2.0
m_ex = X0 * np.exp(-theta * T)
v_ex = sig ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * T))
print(f"  dX = -theta X dt + sigma dW,  X_0 = {X0}, theta = {theta}, "
      f"sigma = {sig}")
print(f"  exakt bei T = {T}:  Mittel {m_ex:.10f}   Varianz {v_ex:.10f}")
print("")
print(f"  {'N':>8} {'Mittel':>16} {'Fehler':>11} {'*N':>9} "
      f"{'Varianz':>13} {'Fehler':>11}")
for N in (20, 100, 500, 2500, 12500):
    dt = T / N
    a = 1 - theta * dt
    m_d = X0 * a ** N
    v_d = sig ** 2 * dt * (1 - a ** (2 * N)) / (1 - a ** 2)
    em, ev = abs(m_d - m_ex), abs(v_d - v_ex)
    print(f"  {N:>8} {m_d:>16.10f} {em:>11.2e} {em*N:>9.4f} "
          f"{v_d:>13.10f} {ev:>11.2e}")
print("  Wieder konstante Spalte 'Fehler * N': Konvergenz erster Ordnung.")
print()
