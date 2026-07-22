"""
Error accumulation of a LUT-based exp inside the online-softmax recurrence.

Question:
  How does the rescale count K relate to the final error, and how many
  table points does a hardware exp unit actually need?

Method:
  Using the SAME approximate exp, compare
    (a) a three-pass softmax   -> no error accumulation  = baseline
    (b) online softmax          -> rescaling accumulates  = measured case
  so the difference isolates the accumulation term alone.
"""

import numpy as np

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------
# 1. 2^y approximation via a LUT + linear interpolation.
#    Range reduction: y = yi + yf  ->  2^y = 2^yi * 2^yf
#    2^yi is an exponent-field add and is exact; only 2^yf is approximated.
# ---------------------------------------------------------------
def make_lut(n_points):
    f = np.arange(n_points + 1) / n_points
    return np.exp2(f)


def exp2_lut(y, lut, n_points):
    y = np.asarray(y, dtype=np.float64)
    yi = np.floor(y)
    yf = y - yi                      # [0, 1)
    pos = yf * n_points
    idx = np.floor(pos).astype(np.int64)
    idx = np.clip(idx, 0, n_points - 1)
    w = pos - idx                    # [0, 1)
    frac = lut[idx] + (lut[idx + 1] - lut[idx]) * w
    return frac * np.exp2(yi)


def exp_lut(x, lut, n_points):
    """exp(x) = 2^(x * log2 e); log2e can be folded into the upstream scale."""
    return exp2_lut(x * LOG2E, lut, n_points)


# ---------------------------------------------------------------
# 2. Two ways to compute the softmax denominator l
# ---------------------------------------------------------------
def denom_three_pass(scores, expf):
    """Take max over everything, then sum. No accumulation of error."""
    m = scores.max()
    return expf(scores - m).sum(), m, 0


def denom_online(scores, block_size, expf):
    """Streaming update of the running max and running sum, block by block."""
    m = -np.inf
    l = 0.0
    K = 0                                     # number of rescales that occur
    for s in range(0, len(scores), block_size):
        blk = scores[s:s + block_size]
        bm = blk.max()
        m_new = max(m, bm)
        if np.isfinite(m) and m_new > m:
            l = l * float(expf(np.array([m - m_new]))[0])
            K += 1
        l = l + expf(blk - m_new).sum()
        m = m_new
    return l, m, K


# ---------------------------------------------------------------
# 3. Input distributions
# ---------------------------------------------------------------
def make_scores(kind, n, rng, scale=4.0):
    if kind == "gaussian":
        return rng.standard_normal(n) * scale
    if kind == "ascending":                   # worst case: max updates every block
        return np.sort(rng.standard_normal(n) * scale)
    if kind == "descending":                  # best case: only one update, at the start
        return np.sort(rng.standard_normal(n) * scale)[::-1].copy()
    if kind == "drift":                       # gentle upward trend + noise (realistic-ish)
        return np.linspace(0, scale * 2, n) + rng.standard_normal(n) * scale * 0.3
    raise ValueError(kind)


# ---------------------------------------------------------------
# 4. Measurement
# ---------------------------------------------------------------
def run(n_points, kind, n, block_size, seed):
    rng = np.random.default_rng(seed)
    scores = make_scores(kind, n, rng)

    lut = make_lut(n_points)
    expf = lambda x: exp_lut(x, lut, n_points)

    ref, _, _ = denom_three_pass(scores, np.exp)          # ground truth (float64)
    base, _, _ = denom_three_pass(scores, expf)           # approx, no accumulation
    onl, _, K = denom_online(scores, block_size, expf)    # approx, with accumulation

    return {
        "err_base": abs(base - ref) / ref,
        "err_online": abs(onl - ref) / ref,
        "K": K,
        "n_blocks": int(np.ceil(n / block_size)),
    }


if __name__ == "__main__":
    BLOCK = 64
    SEEDS = range(8)

    print("=" * 74)
    print("A. How the rescale count K varies with length and distribution (block=64)")
    print("=" * 74)
    print(f"{'dist':<12}{'length':>8}{'blocks':>10}{'K(meas)':>10}{'log2(blocks)':>15}")
    for kind in ["gaussian", "descending", "drift", "ascending"]:
        for n in [1024, 4096, 16384, 65536]:
            Ks = [run(16, kind, n, BLOCK, s)["K"] for s in SEEDS]
            nb = int(np.ceil(n / BLOCK))
            print(f"{kind:<12}{n:>8}{nb:>10}{np.mean(Ks):>10.1f}{np.log2(nb):>15.1f}")
        print()

    print("=" * 74)
    print("B. Table points N vs error (length 4096, block=64)")
    print("=" * 74)
    print(f"{'dist':<12}{'N':>5}{'err:no-accum':>16}{'err:online':>16}{'ratio':>8}{'K':>6}")
    for kind in ["gaussian", "ascending"]:
        for npts in [4, 8, 16, 32, 64, 128]:
            rs = [run(npts, kind, 4096, BLOCK, s) for s in SEEDS]
            eb = np.mean([r["err_base"] for r in rs])
            eo = np.mean([r["err_online"] for r in rs])
            K = np.mean([r["K"] for r in rs])
            print(f"{kind:<12}{npts:>5}{eb:>16.3e}{eo:>16.3e}{eo/eb:>8.2f}{K:>6.1f}")
        print()

    print("=" * 74)
    print("C. Worst case (ascending): does error blow up as length grows? (N=16)")
    print("=" * 74)
    print(f"{'length':>8}{'K':>8}{'err:no-accum':>16}{'err:online':>16}")
    for n in [1024, 4096, 16384, 65536]:
        rs = [run(16, "ascending", n, BLOCK, s) for s in SEEDS]
        eb = np.mean([r["err_base"] for r in rs])
        eo = np.mean([r["err_online"] for r in rs])
        K = np.mean([r["K"] for r in rs])
        print(f"{n:>8}{K:>8.1f}{eb:>16.3e}{eo:>16.3e}")
