"""
Fixed-point model v2 - corrected metric.

v1 failure: it measured weight error as "mean relative error over all weights".
Under a gaussian most weights are negligible (~1e-10) and underflow in a 12-bit
fixed-point representation. Averaging "1e-10 was 100% off" lets terms that
contribute nothing to the output dominate the metric.

Correct metrics (two):
  L1 (total variation) = 0.5 * sum|p_a - p_ref|
      Standard distance between softmax distributions, in [0,1]. Negligible
      weights are automatically ignored.
  output error = ||sum p_i v_i - sum p_ref,i v_i|| / ||sum p_ref,i v_i||
      The quantity attention actually computes (weighted sum of value vectors).
"""

import numpy as np
from fixedpoint_model import FixedExp, online_softmax_fixed


def weights_fixed_arr(scores, m, l_real, fx):
    ONE = 1 << fx.ACC
    l_fixed = int(round(l_real * ONE))
    p = np.array([fx.exp_nonpos(float(v) - m) for v in scores], dtype=np.float64)
    return p / l_fixed


def run(fx, kind="gaussian", n=4096, block=64, d_v=64, seeds=range(8), scale=4.0):
    l1, out_err, den_err, Ks = [], [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        if kind == "gaussian":
            x = rng.standard_normal(n) * scale
        else:
            x = np.sort(rng.standard_normal(n) * scale)
        V = rng.standard_normal((n, d_v))            # value vectors

        l_ref = np.exp(x - x.max()).sum()
        p_ref = np.exp(x - x.max()) / l_ref
        o_ref = p_ref @ V

        l_a, m_a, K = online_softmax_fixed(x, block, fx)
        p_a = weights_fixed_arr(x, m_a, l_a, fx)
        o_a = p_a @ V

        l1.append(0.5 * np.abs(p_a - p_ref).sum())
        out_err.append(np.linalg.norm(o_a - o_ref) / np.linalg.norm(o_ref))
        den_err.append(abs(l_a - l_ref) / l_ref)
        Ks.append(K)
    return (np.mean(l1), np.mean(out_err), np.mean(den_err), np.mean(Ks))


if __name__ == "__main__":
    print("=" * 90)
    print("Check 1 (redux): bias cancellation - does the offset change the attention")
    print("           output error? If not, cancellation holds. N=16, gaussian, len=4096")
    print("=" * 90)
    print(f"{'F':>4}{'ACC':>5}{'offset':>8}{'L1':>12}{'out-err':>12}{'denom-err':>12}")
    for F in [10, 12, 16]:
        for off in [False, True]:
            fx = FixedExp(A=4, FB=16, F=F, ACC=min(F, 14), offset=off)
            l1, oe, de, _ = run(fx)
            print(f"{F:>4}{min(F,14):>5}{('on' if off else 'off'):>8}"
                  f"{l1:>12.3e}{oe:>12.3e}{de:>12.3e}")
        print()

    print("=" * 90)
    print("Check 2 (redux): self-attenuation - grow length in worst case, output error?")
    print("           N=16, F=14, ACC=14, ascending")
    print("=" * 90)
    print(f"{'length':>8}{'K':>7}{'L1':>12}{'out-err':>12}{'denom-err':>12}")
    fx = FixedExp(A=4, FB=16, F=14, ACC=14)
    for n in [1024, 4096, 16384, 65536]:
        l1, oe, de, K = run(fx, kind="ascending", n=n, seeds=range(6))
        print(f"{n:>8}{K:>7.0f}{l1:>12.3e}{oe:>12.3e}{de:>12.3e}")

    print("\n" + "=" * 90)
    print("Check 3: bit budget - reduce table F, accumulator ACC, argument FB one at a time")
    print("        (others held generous) N=16, gaussian")
    print("=" * 90)
    base = dict(A=4, FB=16, F=14, ACC=14)
    print("  -- reduce table bits F --")
    for F in [6, 8, 10, 12, 14]:
        fx = FixedExp(**{**base, "F": F, "ACC": min(F, 14)})
        l1, oe, de, _ = run(fx)
        print(f"    F={F:<3} L1={l1:.3e}  out-err={oe:.3e}")
    print("  -- reduce accumulator bits ACC (F=14 fixed) --")
    for ACC in [8, 10, 12, 14, 18]:
        fx = FixedExp(A=4, FB=16, F=14, ACC=ACC)
        l1, oe, de, _ = run(fx)
        print(f"    ACC={ACC:<3} L1={l1:.3e}  out-err={oe:.3e}  denom-err={de:.3e}")
    print("  -- reduce argument bits FB (F=14, ACC=14 fixed) --")
    for FB in [8, 10, 12, 14, 16]:
        fx = FixedExp(A=4, FB=FB, F=14, ACC=14)
        l1, oe, de, _ = run(fx)
        print(f"    FB={FB:<3} L1={l1:.3e}  out-err={oe:.3e}")
