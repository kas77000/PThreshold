"""Regenerate the tables in docs/fat-tails.md. Run from the project root."""
import numpy as np
from perfthreshold import rule

KINDS = {"normal": None, "t8": 8.0, "t5": 5.0, "t4": 4.0, "t3": 3.0, "t2.5": 2.5}


def draw(kind, n, rng):
    df = KINDS[kind]
    return rng.standard_normal(n) if df is None else rng.standard_t(df, n)


def a_cost_of_k4():
    rng = np.random.default_rng(1)
    for kind in KINDS:
        x = draw(kind, 400_000, rng)
        b = rule.bounds(x, k=4.0)
        rate = rule.count_flags(x, b["lo"], b["hi"]) / x.size
        print(f"{kind:6} sd/mad={b['sd']/b['mad_sigma']:.3f} "
              f"outside={100*rate:.4f}% per3000={rate*3000:6.2f} hi={b['hi']:.2f}")


def b_self_reference():
    rng = np.random.default_rng(2)
    for kind in ("normal", "t3"):
        x = draw(kind, 20_000, rng)
        b = rule.bounds(x, k=4.0)
        keep = x[(x > np.percentile(x, 0.25)) & (x < np.percentile(x, 99.75))]
        t = rule.bounds(keep, k=4.0)
        print(f"{kind:6} full sd={b['sd']:.3f} hi={b['hi']:.2f} "
              f"flags={rule.count_flags(x, b['lo'], b['hi']):4d} | "
              f"trimmed sd={t['sd']:.3f} hi={t['hi']:.2f} "
              f"same rows flagged={rule.count_flags(x, t['lo'], t['hi']):4d}")


def c_masking():
    rng = np.random.default_rng(3)
    for kind in ("normal", "t3"):
        x = draw(kind, 3000, rng)
        b = rule.bounds(x, k=4.0)
        base = rule.count_flags(x, b["lo"], b["hi"])
        for extra in (10, 20, 40):
            y = np.append(x, float(extra))
            n = rule.bounds(y, k=4.0)
            print(f"{kind:6} +1 at {extra:3d}: sd {b['sd']:.3f}->{n['sd']:.3f} "
                  f"hi {b['hi']:.2f}->{n['hi']:.2f} "
                  f"flags {base}->{rule.count_flags(y, n['lo'], n['hi'])}")


def d_bound_stability():
    for kind in KINDS:
        rng = np.random.default_rng(4)
        his = np.array([rule.bounds(draw(kind, 3000, rng), k=4.0)["hi"]
                        for _ in range(400)])
        p5, p95 = np.percentile(his, [5, 95])
        print(f"{kind:6} hi median={np.median(his):5.2f} p5={p5:5.2f} "
              f"p95={p95:5.2f} p95/p5={p95/p5:.2f}x")


def e_percentile_error():
    for kind in ("normal", "t3"):
        rng = np.random.default_rng(5)
        truth = np.percentile(draw(kind, 4_000_000, rng), 99.5)
        for n in (250, 1000, 2000, 12000):
            est = np.array([np.percentile(draw(kind, n, rng), 99.5)
                            for _ in range(400)])
            p5, p95 = np.percentile(est, [5, 95])
            print(f"{kind:6} n={n:6d} truth={truth:5.2f} "
                  f"p5..p95={p5:5.2f}..{p95:5.2f} width={100*(p95-p5)/truth:5.1f}%")


def g_mixture():
    rng = np.random.default_rng(10)
    tight, wide = rng.normal(0, 1.0, 51_000), rng.normal(0, 3.0, 9_000)
    pool = np.concatenate([tight, wide])
    pb = rule.bounds(pool, k=4.0)
    for name, x in (("tight", tight), ("wide", wide), ("pooled", pool)):
        b = rule.bounds(x, k=4.0)
        print(f"{name:6} sd={b['sd']:.3f} mad_sigma={b['mad_sigma']:.3f} "
              f"ratio={b['sd']/b['mad_sigma']:.3f} own hi={b['hi']:5.2f} "
              f"own flags/3000={3000*rule.count_flags(x, b['lo'], b['hi'])/x.size:6.2f} "
              f"pooled-band flags/3000="
              f"{3000*rule.count_flags(x, pb['lo'], pb['hi'])/x.size:6.2f}")


def f_seed_variability():
    """Is the count stable across independent years, at fixed tail weight?"""
    for kind in ("t4", "t3", "t2.5"):
        rates = []
        for seed in range(8):
            rng = np.random.default_rng(100 + seed)
            x = draw(kind, 400_000, rng)
            b = rule.bounds(x, k=4.0)
            rates.append(3000 * rule.count_flags(x, b["lo"], b["hi"]) / x.size)
        r = np.array(rates)
        print(f"{kind:6} per3000 over 8 samples: min={r.min():5.2f} "
              f"median={np.median(r):5.2f} max={r.max():5.2f}")


def h_robust_band_instead():
    """What a median +/- 4*mad_sigma band would cost, for comparison only."""
    rng = np.random.default_rng(31)
    for kind in KINDS:
        x = draw(kind, 400_000, rng)
        b = rule.bounds(x, k=4.0)
        classical = 3000 * rule.count_flags(x, b["lo"], b["hi"]) / x.size
        robust = 3000 * rule.count_flags(
            x, b["median"] - 4 * b["mad_sigma"],
            b["median"] + 4 * b["mad_sigma"]) / x.size
        print(f"{kind:6} mean+/-4sd per3000={classical:6.2f}  "
              f"median+/-4*mad_sigma per3000={robust:6.2f}  "
              f"x{robust/max(classical, 1e-9):.1f}")


for name, fn in [("A cost of k=4", a_cost_of_k4),
                 ("B self-reference", b_self_reference),
                 ("C masking", c_masking),
                 ("D bound stability", d_bound_stability),
                 ("E percentile error", e_percentile_error),
                 ("F seed variability", f_seed_variability),
                 ("G mixture", g_mixture),
                 ("H robust band instead", h_robust_band_instead)]:
    print(f"\n=== {name} ===")
    fn()
