"""
Question (1): can a different table construction hit the same accuracy with a
              smaller table?
Question (2): what does the error look like on the softmax OUTPUT, not just the
              denominator?

Because of range reduction the LUT only ever approximates 2^f, f in [0,1).
So the LUT input is essentially uniform regardless of the score distribution.
-> "non-uniform spacing matched to the input distribution" should NOT help.
   Spacing matched to curvature might. This script checks both.
"""

import numpy as np

LOG2E = 1.4426950408889634


# ===============================================================
# Four table constructions
# ===============================================================
class UniformLinear:
    """(a) Uniform spacing + linear interpolation = baseline"""
    name = "uniform+linear"

    def __init__(self, n):
        self.n = n
        self.tab = np.exp2(np.arange(n + 1) / n)

    def n_words(self):        # stored words
        return self.n + 1

    def n_mults(self):        # multipliers
        return 1

    def cheap_addr(self):     # address = bit slice only?
        return True

    def eval_frac(self, f):
        pos = f * self.n
        i = np.clip(np.floor(pos).astype(np.int64), 0, self.n - 1)
        w = pos - i
        return self.tab[i] + (self.tab[i + 1] - self.tab[i]) * w


class UniformLinearOffset(UniformLinear):
    """(b) Uniform + linear, but table values shifted down so the error is
           balanced instead of one-sided (minimax). 2^f is convex, so linear
           interpolation always overshoots; subtract the mean overshoot.
           Hardware cost is zero (only the stored values change)."""
    name = "uniform+linear+offset"

    def __init__(self, n):
        super().__init__(n)
        h = 1.0 / n
        f = np.arange(n + 1) / n
        # peak mid-interval error ~ h^2/8 * f''; remove half of it from each node
        self.tab = np.exp2(f) - (h * h / 16.0) * (np.log(2) ** 2) * np.exp2(f)


class NonUniformLinear:
    """(c) Curvature-adaptive spacing + linear interpolation.
           Ideal node density h(x) ~ 1/sqrt(|f''(x)|).
           Address is no longer a bit slice (extra hardware cost)."""
    name = "nonuniform+linear"

    def __init__(self, n):
        self.n = n
        # place nodes so density ~ sqrt(|f''|)
        t = np.linspace(0, 1, 20001)
        dens = np.sqrt((np.log(2) ** 2) * np.exp2(t))
        cum = np.concatenate([[0], np.cumsum((dens[:-1] + dens[1:]) / 2)])
        cum /= cum[-1]
        self.nodes = np.interp(np.arange(n + 1) / n, cum, t)
        self.nodes[0], self.nodes[-1] = 0.0, 1.0
        self.tab = np.exp2(self.nodes)

    def n_words(self):
        return 2 * (self.n + 1)   # store node positions in addition to values

    def n_mults(self):
        return 1

    def cheap_addr(self):
        return False

    def eval_frac(self, f):
        i = np.clip(np.searchsorted(self.nodes, f, side="right") - 1, 0, self.n - 1)
        w = (f - self.nodes[i]) / (self.nodes[i + 1] - self.nodes[i])
        return self.tab[i] + (self.tab[i + 1] - self.tab[i]) * w


class UniformQuadratic:
    """(d) Uniform spacing + quadratic interpolation. Error ~ h^3.
           Three coefficients per interval; one extra multiplier (Horner)."""
    name = "uniform+quadratic"

    def __init__(self, n):
        self.n = n
        h = 1.0 / n
        a, b, c = [], [], []
        for i in range(n):
            x0 = i * h
            # Lagrange fit through 3 points, expressed as a quadratic in w in [0,1]
            y0, y1, y2 = np.exp2([x0, x0 + h / 2, x0 + h])
            a.append(y0)
            b.append(-3 * y0 + 4 * y1 - y2)
            c.append(2 * y0 - 4 * y1 + 2 * y2)
        self.a, self.b, self.c = map(np.array, (a, b, c))

    def n_words(self):
        return 3 * self.n

    def n_mults(self):
        return 2

    def cheap_addr(self):
        return True

    def eval_frac(self, f):
        pos = f * self.n
        i = np.clip(np.floor(pos).astype(np.int64), 0, self.n - 1)
        w = pos - i
        return self.a[i] + (self.b[i] + self.c[i] * w) * w


def make_expf(scheme):
    """Range-reduced exp. Integer part is an exponent-field op -> exact."""
    def expf(x):
        y = np.asarray(x, dtype=np.float64) * LOG2E
        yi = np.floor(y)
        return scheme.eval_frac(y - yi) * np.exp2(yi)
    return expf


# ===============================================================
# softmax (measure the output weights, not only the denominator)
# ===============================================================
def online_softmax(scores, block_size, expf):
    m, l, K = -np.inf, 0.0, 0
    for s in range(0, len(scores), block_size):
        blk = scores[s:s + block_size]
        m_new = max(m, blk.max())
        if np.isfinite(m) and m_new > m:
            l = l * float(expf(np.array([m - m_new]))[0])
            K += 1
        l = l + expf(blk - m_new).sum()
        m = m_new
    return l, m, K


def measure(scheme, n=4096, block=64, seeds=range(8), scale=4.0):
    expf = make_expf(scheme)
    e_denom, e_weight_max, e_weight_mean = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        x = rng.standard_normal(n) * scale

        l_ref = np.exp(x - x.max()).sum()
        p_ref = np.exp(x - x.max()) / l_ref

        l_a, m_a, _ = online_softmax(x, block, expf)
        p_a = expf(x - m_a) / l_a

        e_denom.append(abs(l_a - l_ref) / l_ref)
        rel = np.abs(p_a - p_ref) / p_ref
        e_weight_max.append(rel.max())
        e_weight_mean.append(rel.mean())
    return (np.mean(e_denom), np.mean(e_weight_max), np.mean(e_weight_mean))


if __name__ == "__main__":
    print("=" * 92)
    print("(1) Table construction comparison - denominator l error")
    print("=" * 92)
    print(f"{'scheme':<24}{'N':>5}{'words':>8}{'mults':>7}{'cheap-addr':>12}"
          f"{'err(denom)':>14}{'vs base':>9}")
    for N in [4, 8, 16, 32]:
        base = UniformLinear(N)
        eb = measure(base)[0]
        for cls in [UniformLinear, UniformLinearOffset, NonUniformLinear, UniformQuadratic]:
            sch = cls(N)
            e = measure(sch)[0]
            print(f"{sch.name:<24}{N:>5}{sch.n_words():>8}{sch.n_mults():>7}"
                  f"{('yes' if sch.cheap_addr() else 'no'):>12}"
                  f"{e:>14.3e}{eb/e:>8.2f}x")
        print()

    print("=" * 92)
    print("(2) Denominator error vs softmax weight error")
    print("    A systematic (constant) error should cancel in the division")
    print("=" * 92)
    print(f"{'scheme':<24}{'N':>5}{'err(denom)':>14}{'err(weight,max)':>18}"
          f"{'err(weight,mean)':>18}{'cancel':>9}")
    for cls in [UniformLinear, UniformLinearOffset, UniformQuadratic]:
        for N in [4, 8, 16, 32]:
            sch = cls(N)
            ed, ewx, ewm = measure(sch)
            print(f"{sch.name:<24}{N:>5}{ed:>14.3e}{ewx:>18.3e}{ewm:>18.3e}"
                  f"{ed/ewm:>8.1f}x")
        print()
