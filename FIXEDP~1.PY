"""
Fixed-point model: do the two findings survive hardware-style integer arithmetic?

float64 with an exact table revealed
  (i)  self-attenuation (error is linear in K but with a tiny coefficient)
  (ii) bias cancellation (a constant multiplicative bias divides out in softmax)
This checks whether they still hold once we also quantise:
  - the table entries themselves to F bits
  - interpolation, accumulation and rescale to integer shift/multiply

Everything runs in Python int (arbitrary precision) to reproduce bit-level
behaviour. Floating point is used only for the reference value and the final
relative-error computation.
"""

import numpy as np

LOG2E = 1.4426950408889634


class FixedExp:
    """Fixed-point 2^f approximation + range-reduced exp. All integer ops.

    Parameters:
      A   : table address bits (number of points N = 2^A)
      FB  : fractional bits of the reduced argument f (FB >= A)
      F   : fractional bits of the table entries (quantisation coarseness)
      ACC : fractional bits of the exp output and accumulator (ACC <= F)
      offset : if True, bake a minimax offset into the table (for the
               cancellation check)
    """

    def __init__(self, A=4, FB=12, F=12, ACC=12, offset=False):
        self.A, self.FB, self.F, self.ACC = A, FB, F, ACC
        N = 1 << A
        self.N = N
        # quantise table entries to F bits (this is the hardware-specific error source)
        self.tab = []
        for k in range(N + 1):
            v = 2.0 ** (k / N)                       # [1, 2)
            if offset:
                h = 1.0 / N
                v -= (h * h / 16.0) * (np.log(2) ** 2) * (2.0 ** (k / N))
            self.tab.append(int(round(v * (1 << F))))  # Q(1.F) integer
        self.wbits = FB - A                           # interpolation weight bits

    def frac_pow2(self, f_fixed):
        """f_fixed: integer in [0, 2^FB). Returns 2^(f_fixed/2^FB) as a Q(1.F) int."""
        idx = f_fixed >> self.wbits
        w = f_fixed & ((1 << self.wbits) - 1)
        lo = self.tab[idx]
        hi = self.tab[idx + 1]
        # lo + (hi-lo)*w / 2^wbits, all integer
        return lo + (((hi - lo) * w) >> self.wbits)

    def exp_nonpos(self, x):
        """exp(x) for real x <= 0, returned as an ACC-bit fixed-point integer.
        In online softmax the exp argument is always <= 0, so the output is (0, 1]."""
        # y = x * log2e quantised to FB fractional bits (y <= 0)
        y_fixed = int(np.floor(x * LOG2E * (1 << self.FB)))   # negative
        i = y_fixed >> self.FB               # floor(y), negative (arithmetic shift)
        f_fixed = y_fixed - (i << self.FB)   # normalised to [0, 2^FB)
        frac = self.frac_pow2(f_fixed)       # Q(1.F), [2^F, 2^(F+1))
        # real value = frac/2^F * 2^i; to an ACC-bit fixed-point int: frac >> (F - ACC - i)
        sh = self.F - self.ACC - i           # normally i<=0 so sh>=0
        if sh >= 63:
            return 0                         # underflow (negligible term)
        if sh >= 0:
            return frac >> sh
        return frac << (-sh)                 # argument rounding can rarely be positive


def online_softmax_fixed(scores, block, fx):
    """Online softmax holding all state as integers. Returns denominator l (real)."""
    m = None
    l = 0            # integer, ACC fractional bits
    K = 0
    for s in range(0, len(scores), block):
        blk = scores[s:s + block]
        bm = float(blk.max())
        m_new = bm if m is None else max(m, bm)
        if m is not None and m_new > m:
            r = fx.exp_nonpos(m - m_new)          # (0,1], ACC fractional bits
            l = (l * r) >> fx.ACC                 # integer multiply + shift
            K += 1
        for v in blk:
            l += fx.exp_nonpos(float(v) - m_new)
        m = m_new
    return l / (1 << fx.ACC), m, K


def weights_fixed(scores, m, l_real, fx):
    ONE = 1 << fx.ACC
    l_fixed = int(round(l_real * ONE))
    p = np.array([fx.exp_nonpos(float(v) - m) for v in scores], dtype=np.float64)
    return p / l_fixed        # numerator and denominator share the scale -> ratio is real


def run(fx, kind="gaussian", n=4096, block=64, seeds=range(8), scale=4.0):
    ed, ewm, Ks = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        if kind == "gaussian":
            x = rng.standard_normal(n) * scale
        elif kind == "ascending":
            x = np.sort(rng.standard_normal(n) * scale)
        l_ref = np.exp(x - x.max()).sum()
        p_ref = np.exp(x - x.max()) / l_ref

        l_a, m_a, K = online_softmax_fixed(x, block, fx)
        p_a = weights_fixed(x, m_a, l_a, fx)

        ed.append(abs(l_a - l_ref) / l_ref)
        ewm.append((np.abs(p_a - p_ref) / p_ref).mean())
        Ks.append(K)
    return np.mean(ed), np.mean(ewm), np.mean(Ks)


if __name__ == "__main__":
    print("=" * 84)
    print("Check 1: does bias cancellation survive fixed point? (N=16, gaussian, len=4096)")
    print("         compare denominator error and weight error, offset off/on")
    print("=" * 84)
    print(f"{'table bits F':>13}{'offset':>8}"
          f"{'err(denom)':>14}{'err(weight)':>14}{'cancel':>9}")
    for F in [8, 10, 12, 16]:
        for off in [False, True]:
            fx = FixedExp(A=4, FB=max(12, F), F=F, ACC=min(F, 12), offset=off)
            ed, ewm, _ = run(fx)
            print(f"{F:>13}{('on' if off else 'off'):>8}{ed:>14.3e}{ewm:>14.3e}{ed/ewm:>8.1f}x")
        print()

    print("=" * 84)
    print("Check 2: does self-attenuation survive fixed point?")
    print("         worst case ascending, grow length, does error blow up? (N=16, F=12)")
    print("=" * 84)
    print(f"{'length':>8}{'K':>8}{'err(denom)':>14}{'err(weight)':>14}")
    fx = FixedExp(A=4, FB=12, F=12, ACC=12)
    for n in [1024, 4096, 16384, 65536]:
        ed, ewm, K = run(fx, kind="ascending", n=n, seeds=range(6))
        print(f"{n:>8}{K:>8.0f}{ed:>14.3e}{ewm:>14.3e}")

    print("=" * 84)
    print("Check 3: table entry width F vs weight error (N=16)")
    print("=" * 84)
    print(f"{'F (table bits)':>14}{'weight err':>14}")
    for F in [6, 8, 10, 12, 14, 16]:
        fx = FixedExp(A=4, FB=max(12, F), F=F, ACC=min(F, 12))
        _, ewm, _ = run(fx)
        print(f"{F:>14}{ewm:>14.3e}")

    print("\nreference: float64 + exact table weight error (earlier) = 1.06e-04")
