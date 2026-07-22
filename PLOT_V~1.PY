"""Figures + numerical verification for the table-construction comparison."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from table_variants import (UniformLinear, UniformLinearOffset,
                            NonUniformLinear, UniformQuadratic, measure)

Ns = [4, 8, 16, 32]
SCHEMES = [(UniformLinear, "tab:blue", "o"),
           (UniformLinearOffset, "tab:green", "s"),
           (NonUniformLinear, "tab:gray", "v"),
           (UniformQuadratic, "tab:red", "^")]

data = {}
for cls, c, mk in SCHEMES:
    rows = []
    for N in Ns:
        sch = cls(N)
        ed, ewx, ewm = measure(sch)
        rows.append((sch.n_words(), sch.n_mults(), ed, ewm))
    data[cls.name] = (np.array(rows), c, mk)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

ax = axes[0]
for name, (r, c, mk) in data.items():
    ax.plot(r[:, 0], r[:, 3], mk + "-", color=c, label=name)
ax.axhline(1e-3, color="k", lw=1, ls=":")
ax.text(60, 1.15e-3, "~0.1% target for inference", fontsize=8)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("stored words (table size)")
ax.set_ylabel("softmax weight relative error")
ax.set_title("(1) Efficiency frontier\nnon-uniform does nothing / only quadratic helps")
ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[1]
w = 0.35
xs = np.arange(len(Ns))
lin = data["uniform+linear"][0]
off = data["uniform+linear+offset"][0]
ax.bar(xs - w/2, lin[:, 2] / lin[:, 3], w, label="uniform+linear", color="tab:blue")
ax.bar(xs + w/2, off[:, 2] / off[:, 3], w, label="uniform+linear+offset", color="tab:green")
ax.axhline(1.0, color="k", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"N={n}" for n in Ns])
ax.set_ylabel("denominator error / weight error")
ax.set_title("(2) How much error cancels in the division\nabove 1 = cancelling")
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")

plt.tight_layout()
plt.savefig("figures/table_variants.png", dpi=140)
print("saved figures/table_variants.png")

# --- verify: relative curvature of 2^f is constant --------------------
print("\n" + "=" * 70)
print("Verify: why non-uniform spacing does not help")
print("=" * 70)
print("Relative error of linear interp ~ (h^2/8) * f''/f. For f=2^x, f''/f = (ln2)^2 = const.")
f = np.linspace(0, 1, 11)
ratio = ((np.log(2) ** 2) * np.exp2(f)) / np.exp2(f)
print(f"  measured f''/f: min={ratio.min():.6f}  max={ratio.max():.6f}  "
      f"(ln2)^2={np.log(2)**2:.6f}")
print("  -> constant over the whole interval; uniform spacing is already optimal.")

# --- verify: sign of error --------------------------------------------
print("\n" + "=" * 70)
print("Verify: sign of error (why cancellation happens)")
print("=" * 70)
ff = np.linspace(0, 1, 50001)[:-1]
for cls in [UniformLinear, UniformLinearOffset]:
    s = cls(16)
    rel = s.eval_frac(ff) / np.exp2(ff) - 1
    print(f"  {s.name:<22} rel err: mean={rel.mean():+.3e}  "
          f"ripple={rel.max()-rel.min():.3e}  positive={100*np.mean(rel>0):.0f}%")
print("  -> ripple is ~identical; only the mean differs, and the mean divides out.")
