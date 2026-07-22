"""Figures + numerical verification for the online-softmax accumulation study."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from online_softmax_error import run, make_lut, exp2_lut

BLOCK = 64
SEEDS = range(12)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# ---- (1) rescale count K vs number of blocks --------------------------
ax = axes[0]
ns = [1024, 4096, 16384, 65536]
for kind, mk in [("gaussian", "o"), ("drift", "s"), ("descending", "v")]:
    xs, ys = [], []
    for n in ns:
        K = np.mean([run(16, kind, n, BLOCK, s)["K"] for s in SEEDS])
        xs.append(int(np.ceil(n / BLOCK)))
        ys.append(K)
    ax.plot(xs, ys, mk + "-", label=kind)
xs = np.array([16, 64, 256, 1024])
ax.plot(xs, xs - 1, "k:", label="ascending (= nb-1)")
ax.plot(xs, np.log(xs), "r--", lw=2, label=r"theory $\ln(n_{blocks})$")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("number of blocks"); ax.set_ylabel("rescale count K")
ax.set_title("(1) K changes by orders across distributions")
ax.legend(fontsize=8); ax.grid(alpha=.3)

# ---- (2) error vs number of table points ------------------------------
ax = axes[1]
Ns = [4, 8, 16, 32, 64, 128]
for kind, c in [("gaussian", "tab:blue"), ("ascending", "tab:red")]:
    eb, eo = [], []
    for npts in Ns:
        rs = [run(npts, kind, 4096, BLOCK, s) for s in SEEDS]
        eb.append(np.mean([r["err_base"] for r in rs]))
        eo.append(np.mean([r["err_online"] for r in rs]))
    ax.plot(Ns, eb, "o--", color=c, alpha=.5, label=f"{kind}: no accumulation")
    ax.plot(Ns, eo, "o-", color=c, label=f"{kind}: online")
ax.plot(Ns, 3e-2 / np.array(Ns) ** 2.0, "k:", label=r"$\propto 1/N^2$")
ax.axhline(1e-3, color="gray", lw=1)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("table points N"); ax.set_ylabel("relative error")
ax.set_title(r"(2) Error is $1/N^2$; accumulation adds only 1.3-1.7x")
ax.legend(fontsize=8); ax.grid(alpha=.3)

# ---- (3) worst case, growing length -----------------------------------
ax = axes[2]
Ks, eb, eo = [], [], []
for n in [1024, 4096, 16384, 65536, 262144]:
    rs = [run(16, "ascending", n, BLOCK, s) for s in SEEDS[:6]]
    Ks.append(np.mean([r["K"] for r in rs]))
    eb.append(np.mean([r["err_base"] for r in rs]))
    eo.append(np.mean([r["err_online"] for r in rs]))
eb, eo, Ks = np.array(eb), np.array(eo), np.array(Ks)
ax.plot(Ks, eb, "o--", color="gray", label="no accumulation (baseline)")
ax.plot(Ks, eo, "o-", color="tab:red", label="online (N=16)")
ax.plot(Ks, eo - eb, "^-", color="tab:orange", label="accumulation only")
ax.plot(Ks, (eo[0] - eb[0]) * Ks / Ks[0], "k:", label=r"$\propto K$ (linear)")
ax.axhline(1e-3, color="gray", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("rescale count K"); ax.set_ylabel("relative error")
ax.set_title("(3) Accumulation is linear in K, but the coefficient is tiny")
ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig("figures/online_softmax_error.png", dpi=140)
print("saved figures/online_softmax_error.png")

# ---- verification -----------------------------------------------------
print("\n" + "=" * 70)
print("Verify 1: measured K vs theory ln(nb)   [gaussian]")
print("=" * 70)
for n in [1024, 4096, 16384, 65536]:
    nb = int(np.ceil(n / BLOCK))
    K = np.mean([run(16, "gaussian", n, BLOCK, s)["K"] for s in SEEDS])
    print(f"  blocks {nb:>5}   measured K = {K:5.2f}   ln(nb) = {np.log(nb):5.2f}"
          f"   ratio = {K/np.log(nb):.2f}")

print("\n" + "=" * 70)
print("Verify 2: LUT error vs theory h^2/8 * max|f''|")
print("=" * 70)
maxf2 = (np.log(2) ** 2) * 2.0
for npts in [4, 16, 64]:
    lut = make_lut(npts)
    f = np.linspace(0, 1, 200001)[:-1]
    approx = exp2_lut(f, lut, npts)
    err = np.abs(approx - np.exp2(f))
    theo = (1 / npts) ** 2 / 8 * maxf2
    print(f"  N={npts:>4}   measured max err = {err.max():.3e}   theory bound = {theo:.3e}"
          f"   ratio = {err.max()/theo:.2f}")

print("\n" + "=" * 70)
print("Verify 3: sign of interpolation error (2^f convex -> always overshoot)")
print("=" * 70)
lut = make_lut(16)
f = np.linspace(0, 1, 100001)[:-1]
d = exp2_lut(f, lut, 16) - np.exp2(f)
print(f"  fraction positive = {100*np.mean(d >= -1e-15):.2f}%   "
      f"mean error = {d.mean():+.3e}  (systematic bias -> can accumulate)")

print("\n" + "=" * 70)
print("Verify 4: is accumulation linear in K?  [ascending, N=16]")
print("=" * 70)
for i in range(1, len(Ks)):
    print(f"  K: {Ks[i-1]:>7.0f} -> {Ks[i]:>7.0f} ({Ks[i]/Ks[i-1]:.2f}x)   "
          f"accum: {(eo-eb)[i-1]:.2e} -> {(eo-eb)[i]:.2e} "
          f"({(eo-eb)[i]/(eo-eb)[i-1]:.2f}x)")
per = (eo - eb)[-1] / Ks[-1]
print(f"\n  error per rescale        = {per:.2e}")
print(f"  single-shot error (base) = {eb[-1]:.2e}")
print(f"  -> attenuation {per/eb[-1]:.4f}, i.e. {eb[-1]/per:.0f}x below a naive K*eps bound")
