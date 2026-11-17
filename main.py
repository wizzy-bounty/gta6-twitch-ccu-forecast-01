"""Forecast GTA VI launch-day Twitch CCU, hour by hour (v5).

Donor-curve log-ratio model: for each forecast hour h = 2..24,
    ln(y_h / p1) = a_h + b_h * ln(p1) + c * s(h) * f
                   + g * u(h) * (vpc1 - med_vpc1)
                   + e * u(h) * f * (ln(p1) - med_ln_p1)
with noise ~ N(0, sigma_h). All coefficients are fit per case from the
comparable launches shipped in the case file itself -- no hard-coded curve.

- f is the early-wave momentum: log of the mean of the last 3 hourly
  points in the 13 hours before the checkpoint, over the hour-1 print.
- vpc1 is viewer concentration: ln(hour-1 viewers / hour-1 channels).
  Its effect is a single global parameter g, active only in the early
  trough (u(h) = 1 for h <= 4, 0.5 for h == 5, 0 after).
- The v5 addition is the f x size interaction e: the momentum effect in
  the trough depends on launch size (ln(p1) centred on the donors'
  median), also on the trough shape u(h). (c, g, e) are fit together by
  alternating least squares (8 rounds).
- sigma_h is NOT the in-sample residual spread: it is the spread of
  internal leave-one-out residuals (refit without each donor, predict
  that donor), times 0.58 and floored at 0.30. LOO residuals are the
  honest dispersion of the median forecast; the 0.58/0.30 calibration
  was chosen on the practice cases with the LOO guardrail held flat.

Local verification (harness-local/eval.py): practice 3-case (p00/p02/p03)
0.1389 (p00 0.1493 / p02 0.0991 / p03 0.1682; 4-case incl. p01: 0.1352),
LOO 19 donors 0.2510, vs v2 0.1574 / 0.2512 and v3 0.1548 / 0.2491.
Margins are thin (practice gate 0.139, LOO gate 0.2512) and most of the
practice gain is sigma-side; LOO is slightly worse than v3's.

Python 3.13 stdlib only, no network, deterministic.
"""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
SIGMA_FLOOR = 0.15
SIGMA_MULT = 0.9
B_MIN = -0.55
B_MAX = 0.05
ALS_ROUNDS = 8
C_INIT = 0.5
G_INIT = 0.0
# v5 interaction switches (0=off, 1=on). INT_FVPC: d*s(h)*f*(vpc1-med_vpc);
# INT_FSIZE: e*s(h)*f*(lnp1-med_lnp1). *_SHAPE: 's' (momentum) or 'u' (trough).
INT_FVPC = 0
INT_FVPC_SHAPE = "s"
INT_FSIZE = 1
INT_FSIZE_SHAPE = "u"
# v5 angle 3: sigma from internal leave-one-out residuals (not in-sample).
LOO_SIGMA = 1
LOO_SIGMA_MULT = 0.58
LOO_SIGMA_FLOOR = 0.30
LOO_ASYM = 0  # 1 = separate lower/upper sigma from LOO residual halves


def _median(vals):
    """Median of a non-empty sequence (mean of the two middles if even)."""
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _pstdev(vals):
    """Population standard deviation of a non-empty sequence."""
    n = len(vals)
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / n)


def _ts_hour(ts):
    """Absolute hour index of an ISO-8601 UTC timestamp, or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(dt.timestamp() // 3600)


def _s(h):
    """Momentum shape: negative in the trough, rising to 1 by hour 13."""
    if h <= 4:
        return -0.55
    if h == 5:
        return 0.0
    if h >= 13:
        return 1.0
    return (h - 5) / 8.0


def _u(h):
    """vpc1 trough shape: full effect hours <= 4, half at hour 5, none after."""
    if h <= 4:
        return 1.0
    if h == 5:
        return 0.5
    return 0.0


def release_day(comp):
    """Release-day curve {hour: viewers} of one comparable.

    Rows are the comparable's hourly rows on the UTC date of day_start;
    hour = UTC hour + 1. Rows without viewers are skipped. A curve
    without hour 1 is not usable, so None is returned for it.
    """
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    curve = {}
    for row in comp.get("hourly") or []:
        frm = row.get("from") or ""
        if frm[:10] != day:
            continue
        v = row.get("viewers")
        if v is None or v <= 0:
            continue
        curve[int(frm[11:13]) + 1] = float(v)
    if 1 not in curve:
        return None
    return curve


def _is_live(curve):
    """Live hour-1: hour-1 viewers at least 3% of the day's peak."""
    return curve[1] >= 0.03 * max(curve.values())


def is_donor(comp, curve, live=True):
    """Donor filters: not GTA VI itself, Tier: comparable, and (when
    live=True) a live hour 1."""
    if "Grand Theft Auto VI" in (comp.get("name") or ""):
        return False
    if "Tier: comparable" not in (comp.get("unlock") or ""):
        return False
    if live and not _is_live(curve):
        return False
    return True


def pick_donors(case):
    """Donor list [(comp, curve)] with the spec's relaxations.

    Live donors (all three filters) when there are at least 8; else drop
    the live filter (need >= 5); else drop the tier filter but keep live
    (need >= 5); else every parsed comparable except GTA VI.
    """
    parsed = []
    for comp in case.get("comparables") or []:
        curve = release_day(comp)
        if curve is not None:
            parsed.append((comp, curve))
    live = [(c, cv) for c, cv in parsed if is_donor(c, cv, live=True)]
    if len(live) >= 8:
        return live
    tier = [(c, cv) for c, cv in parsed if is_donor(c, cv, live=False)]
    if len(tier) >= 5:
        return tier
    liveish = [(c, cv) for c, cv in parsed
               if "Grand Theft Auto VI" not in (c.get("name") or "")
               and _is_live(cv)]
    if len(liveish) >= 5:
        return liveish
    return [(c, cv) for c, cv in parsed
            if "Grand Theft Auto VI" not in (c.get("name") or "")]


def _momentum(rows, day_start, checkpoint, p1):
    """Early-wave momentum f from hourly (hour_index, viewers) rows.

    Window: the 13 hours before the checkpoint, i.e. hour indices in
    [hour(day_start) - 13, hour(checkpoint)). f is the log of the mean
    of the last 3 hourly points in the window over p1. None when the
    window is empty or the mean is not positive.
    """
    start = _ts_hour(day_start)
    end = _ts_hour(checkpoint)
    if start is None or end is None:
        return None
    start -= 13
    pts = sorted((idx, v) for idx, v in rows
                 if idx is not None and v is not None and start <= idx < end)
    if not pts:
        return None
    last = pts[-3:]
    mean = sum(v for _, v in last) / len(last)
    if mean <= 0 or p1 <= 0:
        return None
    return math.log(mean / p1)


def _donor_vpc1(comp, curve):
    """ln(p1 / channels at hour 1) for a donor, or None when the
    hour-1 channels are unknown."""
    day = (comp.get("day_start") or "")[:10]
    first = day + "T00:00:00Z"
    for row in comp.get("hourly") or []:
        if row.get("from") == first:
            ch = row.get("channels")
            if ch:
                return math.log(curve[1] / ch)
            return None
    return None


def _round_half_up_2dp(v):
    """JS Math.round(v * 100) / 100: round half up to 2 decimals."""
    return math.floor(v * 100.0 + 0.5) / 100.0


def _quantile_row(h, lr, sigma, p1, sigma_lo=None):
    qs = []
    prev = None
    for z in Z:
        s = sigma_lo if (sigma_lo is not None and z < 0) else sigma
        q = p1 * math.exp(lr + z * s)
        if q > 1e9:
            q = 1e9
        q = _round_half_up_2dp(q)
        if q < 0.01:
            q = 0.01
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return {"hour": h, "q": qs}


def _theta_terms(h, f, va, x, med_x):
    """(coefficient, shape, feature) triples of the active global terms."""
    terms = [(0, _s(h), f), (1, _u(h), va)]
    if INT_FVPC:
        sh = _s(h) if INT_FVPC_SHAPE == "s" else _u(h)
        terms.append((len(terms), sh, f * va))
    if INT_FSIZE:
        sh = _s(h) if INT_FSIZE_SHAPE == "s" else _u(h)
        terms.append((len(terms), sh, f * (x - med_x)))
    return terms


def _fit_ab(hour_data, theta, med_x=0.0):
    """Per-hour (a_h, b_h) with the current global theta vector.

    Regress the log-ratio net of all global effects on ln(p1); clamp
    the slope to [-0.55, 0.05], then refit the intercept as the mean
    residual after clamping. Hours with fewer than 5 donors are left
    out (they use the median fallback instead).
    """
    ab = {}
    for h, pts in hour_data.items():
        if len(pts) < 5:
            continue
        xs = [x for (x, _y, _f, _va, _di) in pts]
        ys = [y - sum(theta[i] * sh * ft for (i, sh, ft)
                      in _theta_terms(h, f, va, x, med_x))
              for (x, y, f, va, _di) in pts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var > 0:
            b = sum((x - mx) * (yy - my) for x, yy in zip(xs, ys)) / var
        else:
            b = 0.0
        b = min(B_MAX, max(B_MIN, b))
        a = sum(yy - b * x for x, yy in zip(xs, ys)) / len(xs)
        ab[h] = (a, b)
    return ab


def _solve_lin(A, b):
    """Dense Gaussian elimination with partial pivoting (K <= 4 here)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return [0.0] * n
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                fac = M[r][col] / M[col][col]
                for cc in range(col, n + 1):
                    M[r][cc] -= fac * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def _solve_cg(hour_data, ab, med_x=0.0):
    """Joint KxK least squares for the global theta from all residuals.

    Residuals r = y - a_h - b_h * ln(p1); solve for theta in
    r ~= sum_i theta_i * shape_i(h) * feat_i. Singular systems yield
    a zero vector.
    """
    k = 2 + (1 if INT_FVPC else 0) + (1 if INT_FSIZE else 0)
    A = [[0.0] * k for _ in range(k)]
    bb = [0.0] * k
    for h, pts in hour_data.items():
        if h not in ab:
            continue
        a, b = ab[h]
        for (x, y, f, va, _di) in pts:
            r = y - a - b * x
            cols = [sh * ft for (_i, sh, ft)
                    in _theta_terms(h, f, va, x, med_x)]
            for i in range(k):
                bb[i] += cols[i] * r
                for j in range(k):
                    A[i][j] += cols[i] * cols[j]
    return _solve_lin(A, bb)


def _fit_model(hour_data, med_x):
    """ALS fit of (ab, theta) on one hour_data dict."""
    theta = [C_INIT, G_INIT]
    if INT_FVPC:
        theta.append(0.0)
    if INT_FSIZE:
        theta.append(0.0)
    ab = {}
    for _ in range(ALS_ROUNDS):
        ab = _fit_ab(hour_data, theta, med_x)
        theta = _solve_cg(hour_data, ab, med_x)
    return ab, theta


def _predict_lr(h, ab, theta, x, f, va, med_x):
    """Median log-ratio for one hour from a fitted model."""
    a, b = ab[h]
    return a + b * x + sum(theta[i] * sh * ft for (i, sh, ft)
                           in _theta_terms(h, f, va, x, med_x))


def _loo_sigmas(hour_data, feats, med_vpc, med_x, hours):
    """Per-hour sigma from internal leave-one-out residuals.

    For each donor i: refit without i, predict i's own log-ratios, record
    residual (truth - prediction) per hour. sigma_h = pstdev of those
    residuals (fallback: in-sample residual pstdev when < 5 points).
    Asymmetric variant: RMS of the negative half / positive half, each
    shrunk 50% toward the pooled pstdev (few points per hour).
    Returns {h: (sigma_lo, sigma_hi)}.
    """
    resid = {h: [] for h in hours}
    nd = len(feats)
    for di in range(nd):
        sub = {h: [p for p in pts if p[4] != di]
               for h, pts in hour_data.items()}
        ab_i, theta_i = _fit_model(sub, med_x)
        x_i, curve_i, f_i, vpc_i = feats[di]
        xs_i = [ft[0] for ft in feats if ft is not feats[di]]
        lo_i, hi_i = min(xs_i), max(xs_i)
        xw = min(max(x_i, lo_i), hi_i)
        va_i = vpc_i - med_vpc
        for h in hours:
            if h not in ab_i or h not in curve_i:
                continue
            y = math.log(curve_i[h] / curve_i[1])
            lr = _predict_lr(h, ab_i, theta_i, xw, f_i, va_i, med_x)
            resid[h].append(y - lr)
    out = {}
    for h in hours:
        rs = resid[h]
        if len(rs) >= 5:
            sd = _pstdev(rs)
            if LOO_ASYM:
                neg = [r for r in rs if r < 0]
                pos = [r for r in rs if r >= 0]
                rms_neg = math.sqrt(sum(r * r for r in neg) / len(neg)) if neg else sd
                rms_pos = math.sqrt(sum(r * r for r in pos) / len(pos)) if pos else sd
                slo = 0.5 * rms_neg + 0.5 * sd
                shi = 0.5 * rms_pos + 0.5 * sd
                out[h] = (slo, shi)
            else:
                out[h] = (sd, sd)
        else:
            out[h] = (None, None)
    return out


def forecast_case(case, name):
    """Forecast one case: {"hours": [{"hour": h, "q": [...]}]}."""
    known = {r.get("hour"): r for r in case.get("known") or []}
    k1 = known.get(1) or {}
    p1 = float(k1.get("viewers") or 0) or 1.0
    if p1 <= 0:
        p1 = 1.0
    channels1 = k1.get("channels")
    rows = case.get("forecast") or []
    hours = sorted(r["hour"] for r in rows if "hour" in r) or list(range(2, 25))

    donors = pick_donors(case)
    if not donors:
        # No donor at all: house-baseline shape (median = p1).
        return {"hours": [_quantile_row(h, 0.0, 0.15 * math.sqrt(h - 1), p1)
                          for h in hours]}

    # Donor features: ln(p1), release-day curve, momentum, vpc1.
    feats = []
    for comp, curve in donors:
        p1d = curve[1]
        day_start = comp.get("day_start") or ""
        cp = day_start[:10] + "T01:00:00Z"
        drows = [(_ts_hour(r.get("from")), r.get("viewers"))
                 for r in comp.get("hourly") or []]
        f = _momentum(drows, day_start, cp, p1d)
        vpc = _donor_vpc1(comp, curve)
        feats.append([math.log(p1d), curve, f, vpc])
    known_vpcs = [ft[3] for ft in feats if ft[3] is not None]
    med_vpc = _median(known_vpcs) if known_vpcs else 0.0
    known_fs = [ft[2] for ft in feats if ft[2] is not None]
    med_f = _median(known_fs) if known_fs else 0.0
    for ft in feats:
        if ft[2] is None:
            ft[2] = med_f
        if ft[3] is None:
            ft[3] = med_vpc

    # Target features: momentum from history_hourly (or per-hour means
    # of history when that is empty), vpc1 from the known hour-1 row.
    hh = case.get("history_hourly") or []
    if hh:
        trows = [(_ts_hour(r.get("from")), r.get("viewers")) for r in hh]
    else:
        agg = {}
        for r in case.get("history") or []:
            idx = _ts_hour(r.get("at"))
            if idx is not None and r.get("viewers") is not None:
                agg.setdefault(idx, []).append(r["viewers"])
        trows = [(idx, sum(vs) / len(vs)) for idx, vs in sorted(agg.items())]
    f_t = _momentum(trows, case.get("day_start") or "",
                    case.get("checkpoint") or "", p1)
    if f_t is None:
        f_t = med_f
    if channels1:
        vpc_t = math.log(p1 / channels1)
    else:
        vpc_t = med_vpc

    # Winsorise the target's ln(p1) to the donors' ln(p1) range.
    xs = [ft[0] for ft in feats]
    ln_p1 = min(max(math.log(p1), min(xs)), max(xs))

    # Per-hour donor points: (ln p1, log-ratio, f, vpc1 - med_vpc1).
    hour_data = {}
    for h in hours:
        pts = []
        for di, (x, curve, f, vpc) in enumerate(feats):
            if h in curve:
                pts.append((x, math.log(curve[h] / curve[1]), f, vpc - med_vpc, di))
        hour_data[h] = pts

    # Alternating least squares for theta.
    med_x = _median([ft[0] for ft in feats])
    ab, theta = _fit_model(hour_data, med_x)
    c, g = theta[0], theta[1]
    loo_sig = _loo_sigmas(hour_data, feats, med_vpc, med_x, hours) if LOO_SIGMA else None

    out_hours = []
    for h in hours:
        pts = hour_data[h]
        if len(pts) < 5:
            lr = _median([y for (_x, y, _f, _va, _di) in pts]) if pts else 0.0
            sigma = 0.5
            slo = shi = None
        else:
            lr = _predict_lr(h, ab, theta, ln_p1, f_t, vpc_t - med_vpc, med_x)
            resid = [y - _predict_lr(h, ab, theta, x, f, va, med_x)
                     for (x, y, f, va, _di) in pts]
            sigma = max(SIGMA_FLOOR, _pstdev(resid) * SIGMA_MULT)
            slo = shi = None
            if loo_sig is not None:
                lslo, lshi = loo_sig[h]
                if lslo is not None:
                    slo = max(LOO_SIGMA_FLOOR, lslo * LOO_SIGMA_MULT)
                    shi = max(LOO_SIGMA_FLOOR, lshi * LOO_SIGMA_MULT)
                    sigma = shi
        if slo is not None and LOO_ASYM:
            out_hours.append(_quantile_row(h, lr, sigma, p1, sigma_lo=slo))
        else:
            out_hours.append(_quantile_row(h, lr, sigma, p1))
    return {"hours": out_hours}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text())
        out = forecast_case(case, path.stem)
        (out_dir / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
