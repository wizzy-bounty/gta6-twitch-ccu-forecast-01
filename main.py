"""Forecast GTA VI launch-day Twitch CCU, hour by hour.

Donor-curve log-ratio model: for each forecast hour h = 2..24,
    ln(y_h / p1) = a_h + b_h * ln(p1) + c * s(h) * f
                   + g * u(h) * (vpc1 - med_vpc1)
with noise ~ N(0, sigma_h). All coefficients are fit per case from the
comparable launches shipped in the case file itself -- no hard-coded curve.

- f is the early-wave momentum: log of the mean of the last 3 hourly
  points in the 13 hours before the checkpoint, over the hour-1 print.
- vpc1 is viewer concentration: ln(hour-1 viewers / hour-1 channels).
  Its effect is a single global parameter g, active only in the early
  trough (u(h) = 1 for h <= 4, 0.5 for h == 5, 0 after).
- (c, g) are fit together by alternating least squares (8 rounds).

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


def _quantile_row(h, lr, sigma, p1):
    qs = []
    prev = None
    for z in Z:
        q = p1 * math.exp(lr + z * sigma)
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


def _fit_ab(hour_data, c, g):
    """Per-hour (a_h, b_h) with the current (c, g).

    Regress the log-ratio net of the momentum and vpc1 effects on
    ln(p1); clamp the slope to [-0.55, 0.05], then refit the intercept
    as the mean residual after clamping. Hours with fewer than 5
    donors are left out (they use the median fallback instead).
    """
    ab = {}
    for h, pts in hour_data.items():
        if len(pts) < 5:
            continue
        sh, uh = _s(h), _u(h)
        xs = [x for (x, _y, _f, _va) in pts]
        ys = [y - c * sh * f - g * uh * va for (x, y, f, va) in pts]
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


def _solve_cg(hour_data, ab):
    """Joint 2x2 least squares for (c, g) from all residuals.

    Residuals r = y - a_h - b_h * ln(p1); solve for c, g in
    r ~= c * s(h) * f + g * u(h) * (vpc1 - med_vpc1). Singular
    systems yield (0.0, 0.0).
    """
    ss = su = uu = sr = ur = 0.0
    for h, pts in hour_data.items():
        if h not in ab:
            continue
        a, b = ab[h]
        sh, uh = _s(h), _u(h)
        for (x, y, f, va) in pts:
            r = y - a - b * x
            st = sh * f
            ut = uh * va
            ss += st * st
            su += st * ut
            uu += ut * ut
            sr += st * r
            ur += ut * r
    det = ss * uu - su * su
    if abs(det) < 1e-12:
        return 0.0, 0.0
    c = (sr * uu - ur * su) / det
    g = (ss * ur - su * sr) / det
    return c, g


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
        for (x, curve, f, vpc) in feats:
            if h in curve:
                pts.append((x, math.log(curve[h] / curve[1]), f, vpc - med_vpc))
        hour_data[h] = pts

    # Alternating least squares for (c, g): 8 rounds from (0.5, 0.0).
    c, g = C_INIT, G_INIT
    ab = {}
    for _ in range(ALS_ROUNDS):
        ab = _fit_ab(hour_data, c, g)
        c, g = _solve_cg(hour_data, ab)

    out_hours = []
    for h in hours:
        pts = hour_data[h]
        if len(pts) < 5:
            lr = _median([y for (_x, y, _f, _va) in pts]) if pts else 0.0
            sigma = 0.5
        else:
            a, b = ab[h]
            sh, uh = _s(h), _u(h)
            lr = a + b * ln_p1 + c * sh * f_t + g * uh * (vpc_t - med_vpc)
            resid = [y - (a + b * x + c * sh * f + g * uh * va)
                     for (x, y, f, va) in pts]
            sigma = max(SIGMA_FLOOR, _pstdev(resid) * SIGMA_MULT)
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
