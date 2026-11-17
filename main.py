"""GTA VI Twitch CCU forecast v2: size-aware launch curve with early-wave momentum.

Model (computed at run time from the case's own comparables, no hard-coded
curve): for each forecast hour h = 2..24, the log ratio of that hour's
viewers to hour 1's print is

    ln(y_h / p1) = a_h + b_h * ln(p1) + c * s(h) * f + noise,
    noise ~ N(0, sigma_h),

fit by least squares over the donor launches, with the size slope b_h
clamped to [-0.55, 0.05] and the intercept refit after clamping. The size
term is the first edge: across past launches the day's peak-to-hour-1 ratio
falls steeply as hour 1 grows (correlation of ln(p1) with ln(peak/p1) is
about -0.82 over the donors), so a small hour-1 print predicts a steep
all-day ramp and a huge one predicts a nearly flat day.

The second edge is the early-wave momentum f. The game unlocks at rolling
local midnight, so by the checkpoint (the end of hour 1) it has already been
live for ~14 hours in New Zealand / Australia and ~2 hours in central
Europe. f measures how hot that early wave was running into hour 1:

    f = ln(mean of the last 3 hourly points in the 13 hours before the
           checkpoint, divided by p1),

the last three points being the 22:00 and 23:00 UTC hours of the previous
day and hour 1 itself. A launch whose early wave ran hot stays hot: the
fitted effect is negative in the small hours (a hot early wave mean-reverts
into a deeper trough at hours 2-4) and about +1.0 from the early afternoon
on (a hot early wave carries into a higher evening peak). One global scale
c carries that effect through a fixed shape s(h) -- -0.55 at hours 2-4,
0 at hour 5, ramping linearly to 1.0 at hour 13 and flat after -- instead
of 23 free per-hour coefficients, so the momentum term cannot overfit the
nineteen donors. a_h, b_h and the single c are fit by alternating least
squares (8 rounds, started at c = 0.5); sigma_h is the residual standard
deviation of the full fit, times 0.9 and floored at 0.15. ln(p1) is
winsorised to the donors' range so an unprecedented hour-1 print never
extrapolates the line.

Donors are the case's comparables that are real launches of the same kind:
tier "comparable" in their unlock note, with a live hour 1 (hour-1 viewers
at least 3% of that launch's release-day peak, which drops simultaneous-
global launches whose hour 1 was still pre-unlock trickle, early-access-
contaminated and weak launches, and GTA VI's own pre-release category
history). Fallbacks relax the filters if too few donors survive, and with
no donors at all the forecast degrades to the house persistence shape.

Quantiles: q_tau(h) = p1 * exp(a_h + b_h*ln(p1) + c*s(h)*f + z_tau*sigma_h),
rounded half-up to 2 decimals, z = (-1.645, -0.674, 0, 0.674, 1.645).

Practice (ground truth = SullyGnome hourly averages, the same source the
kit was cut from; RDR2's month could not be re-read, so p01 is not scored
here): mean pinball loss 0.157 over Wolverine / God of War Ragnarok /
Spider-Man 2 (v1, size-only: 0.187), against the house baseline's 0.421 on
the same three cases; 90% coverage 0.90 on average. Leave-one-out over the
19 donors: 0.251 against the baseline's 0.403 (v1: 0.261). Deterministic,
Python 3.13 standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Donor filters.
MIN_LIVE_HOUR1 = 0.03   # hour-1 viewers as a share of the launch's day peak
MIN_DONORS = 8           # below this, relax the live-hour-1 filter
MIN_DONORS_HARD = 5      # below this, relax the tier filter too
MIN_PER_HOUR = 5         # donors needed for an hour's own fit
SIGMA_FLOOR = 0.15       # never claim an hour more precisely than this (log)
SIGMA_MULT = 0.9         # residual std is slightly wide; shrink it a touch
B_MIN, B_MAX = -0.55, 0.05  # clamp on the size slope

# Early-wave momentum: the hours before the checkpoint that the feature
# averages, and the shape of its effect over the forecast day. The shape
# is fixed, not fitted: a hot early wave mean-reverts into the small-hours
# trough, is neutral at the US-East unlock hour, then carries one-for-one
# into the evening peak.
WINDOW_HOURS = 13       # window before day_start holding the early wave
FEATURE_POINTS = 3      # last hourly points of the window the feature averages
MOMENTUM_ROUNDS = 8     # alternating least-squares rounds for the global scale
MOMENTUM_START = 0.5    # starting value of the global momentum scale


def momentum_shape(h: int) -> float:
    """s(h): the fixed shape of the early-wave effect over the forecast day."""
    if h <= 4:
        return -0.55
    if h == 5:
        return 0.0
    if h >= 13:
        return 1.0
    return (h - 5) / 8.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def mean(xs):
    return sum(xs) / len(xs)


def pstdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def hour_key(stamp: str) -> int | None:
    """An hourly row's `from` stamp as an absolute hour index, or None.

    Only the date and the hour are used, so a trailing Z or offset cannot
    shift the result.
    """
    if len(stamp) < 13:
        return None
    try:
        d = date(int(stamp[0:4]), int(stamp[5:7]), int(stamp[8:10]))
        return d.toordinal() * 24 + int(stamp[11:13])
    except ValueError:
        return None


def release_day(comp: dict) -> dict | None:
    """{hour 1..24: viewers} for the comparable's release day, or None.

    A launch's release day is the UTC date of its day_start; an hour's point
    is the row whose `from` falls in that hour. Rows with no viewers are
    skipped (SullyGnome samples small categories sparsely).
    """
    day = comp.get("day_start", "")[:10]
    if not day:
        return None
    out = {}
    for row in comp.get("hourly") or []:
        frm = row.get("from", "")
        if frm[:10] != day:
            continue
        viewers = row.get("viewers")
        if not viewers:
            continue
        hour = int(frm[11:13]) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    if 1 not in out:
        return None
    return out


def hourly_points(rows: list | None) -> list:
    """(hour index, viewers) pairs from hourly rows, oldest first."""
    pts = []
    for row in rows or []:
        key = hour_key(row.get("from", ""))
        viewers = row.get("viewers")
        if key is not None and viewers:
            pts.append((key, viewers))
    pts.sort()
    return pts


def window_feature(hourly_rows: list | None, day_start: str,
                   checkpoint: str, p1: float) -> float | None:
    """Early-wave momentum f for one launch, or None if its window is empty.

    The window is the 13 hours before the checkpoint: the rolling-midnight
    unlock wave (New Zealand first) up to and including hour 1. f is the
    log of the mean of the window's last three hourly points over the
    hour-1 print. With fewer than three points the mean of what is there
    is used; with none the feature is undefined and the caller falls back
    to the donors' median.
    """
    start = hour_key(day_start)
    end = hour_key(checkpoint)
    if start is None or end is None:
        return None
    pts = [(k, v) for k, v in hourly_points(hourly_rows)
           if start - WINDOW_HOURS <= k < end]
    if not pts:
        return None
    last = [v for _, v in pts[-FEATURE_POINTS:]]
    return math.log(mean(last) / max(p1, 1))


def is_donor(comp: dict, curve: dict, live: bool) -> bool:
    """Tier + (optionally) live-hour-1 filter. Never the target's own history."""
    name = comp.get("name", "")
    if "Grand Theft Auto VI" in name:
        return False
    unlock = comp.get("unlock", "")
    if "Tier: comparable" not in unlock:
        return False
    if live and curve[1] < MIN_LIVE_HOUR1 * max(curve.values()):
        return False
    return True


def pick_donors(case: dict) -> list:
    """Donor (comparable, curve) pairs, filters relaxed if too few survive."""
    parsed = []
    for comp in case.get("comparables") or []:
        curve = release_day(comp)
        if curve is not None:
            parsed.append((comp, curve))
    live = [(comp, c) for comp, c in parsed if is_donor(comp, c, live=True)]
    if len(live) >= MIN_DONORS:
        return live
    tiered = [(comp, c) for comp, c in parsed if is_donor(comp, c, live=False)]
    if len(tiered) >= MIN_DONORS_HARD:
        return tiered
    any_live = [(comp, c) for comp, c in parsed
                if "Grand Theft Auto VI" not in comp.get("name", "")
                and c[1] >= MIN_LIVE_HOUR1 * max(c.values())]
    if len(any_live) >= MIN_DONORS_HARD:
        return any_live
    return parsed


def donor_momentum(comp: dict, curve: dict) -> float | None:
    """A donor's early-wave momentum, from its own hourly rows."""
    day_start = comp.get("day_start", "")
    if not day_start:
        return None
    start = hour_key(day_start)
    if start is None:
        return None
    # The donor's checkpoint is one hour after its day starts.
    checkpoint = f"{day_start[:10]}T01:00:00Z"
    return window_feature(comp.get("hourly"), day_start, checkpoint, curve[1])


def _fit_size_given_scale(per_hour, scale):
    """Per-hour size regression given the momentum scale.

    Returns {hour: (a, b)}: the intercept and clamped size slope from
    regressing the scale-adjusted log ratios on ln(p1).
    """
    coefs = {}
    for h in range(2, 25):
        pts = per_hour[h]
        if len(pts) < 2:
            coefs[h] = (median([p[2] for p in pts]) if pts else 0.0, 0.0)
            continue
        adj = [(xx, y - scale * momentum_shape(h) * f)
               for xx, f, y in pts]
        mx, my = mean([p[0] for p in adj]), mean([p[1] for p in adj])
        var = sum((p[0] - mx) ** 2 for p in adj) / len(adj)
        b = (sum((p[0] - mx) * (p[1] - my) for p in adj) / len(adj) / var
             if var > 0 else 0.0)
        b = max(B_MIN, min(B_MAX, b))
        a = mean([y - b * xx for xx, y in adj])
        coefs[h] = (a, b)
    return coefs


def _fit_scale(per_hour, coefs):
    """One global momentum scale by least squares given the per-hour fits."""
    num = 0.0
    den = 0.0
    for h in range(2, 25):
        a, b = coefs[h]
        for xx, f, y in per_hour[h]:
            shape = momentum_shape(h)
            num += shape * f * (y - (a + b * xx))
            den += (shape * f) ** 2
    return num / den if den > 0 else 0.0


def fit(donors: list, momenta: list, p1: float, f_target: float) -> dict:
    """Per hour: (log-median-ratio, sigma) from the size + momentum fit.

    Returns {hour: (lr, sigma)} where the median log ratio lr is
    a_h + b_h * ln(p1) + c * s(h) * f, winsorised ln(p1), and sigma the
    residual std times SIGMA_MULT. Hours with too few donors fall back to
    the median log ratio over the donors. Donors without a momentum value
    take the donors' median momentum.
    """
    xs_all = [math.log(c[1]) for c in donors]
    lo, hi = min(xs_all), max(xs_all)
    x = max(lo, min(hi, math.log(max(p1, 1))))
    known = [m for m in momenta if m is not None]
    med_f = median(known) if known else 0.0
    feats = [m if m is not None else med_f for m in momenta]
    f0 = f_target if f_target is not None else med_f
    # Per-hour points: (ln p1, momentum, ln ratio).
    per_hour = {}
    for h in range(2, 25):
        per_hour[h] = [(math.log(c[1]), feats[i], math.log(c[h] / c[1]))
                       for i, c in enumerate(donors) if h in c]
    scale = MOMENTUM_START
    coefs = {}
    for _ in range(MOMENTUM_ROUNDS):
        # Given the scale: per-hour size regression on the adjusted ratios.
        coefs = _fit_size_given_scale(per_hour, scale)
        # Given the per-hour fits: one global momentum scale by least squares.
        scale = _fit_scale(per_hour, coefs)
    out = {}
    for h in range(2, 25):
        pts = per_hour[h]
        if len(pts) < MIN_PER_HOUR:
            lr = median([p[2] for p in pts]) if pts else 0.0
            out[h] = (lr, 0.5)
            continue
        a, b = coefs[h]
        res = [y - (a + b * xx + scale * momentum_shape(h) * f)
               for xx, f, y in pts]
        out[h] = (a + b * x + scale * momentum_shape(h) * f0,
                  max(SIGMA_FLOOR, pstdev(res) * SIGMA_MULT))
    return out


def quantile_row(p1: float, lr: float, sigma: float, z: float) -> float:
    """One raw quantile value, before clamping and rounding."""
    return p1 * math.exp(lr + z * sigma)


def hour_quantiles(p1: float, lr: float, sigma: float) -> list:
    """The five quantiles for one hour, non-decreasing, in (0, 1e9]."""
    row = []
    prev = 0.0
    for z in Z:
        q = quantile_row(p1, lr, sigma, z)
        q = max(q, prev)
        q = min(q, 1e9)
        q = round2(q)
        if q <= 0:
            q = 0.01
        if q < prev:
            q = prev
        row.append(q)
        prev = q
    return row


def target_momentum(case: dict, p1: float) -> float | None:
    """The target launch's early-wave momentum, from its own history.

    Prefers history_hourly; if that is absent, aggregates the minute
    history points into hourly means first.
    """
    hourly = case.get("history_hourly")
    if not hourly:
        buckets: dict = {}
        for pt in case.get("history") or []:
            key = hour_key(pt.get("at", ""))
            viewers = pt.get("viewers")
            if key is not None and viewers:
                buckets.setdefault(key, []).append(viewers)
        hourly = []
        for key, vals in sorted(buckets.items()):
            day = date.fromordinal(key // 24)
            hourly.append({"from": f"{day.isoformat()}T{key % 24:02d}:00:00Z",
                           "viewers": mean(vals)})
    return window_feature(hourly, case.get("day_start", ""),
                          case.get("checkpoint", ""), p1)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    donors = pick_donors(case)
    if not donors:
        # No usable comparable at all: house persistence shape.
        params = {h: (0.0, 0.15 * math.sqrt(h - 1)) for h in range(2, 25)}
    else:
        curves = [curve for _, curve in donors]
        momenta = [donor_momentum(comp, curve) for comp, curve in donors]
        f_target = target_momentum(case, p1)
        params = fit(curves, momenta, p1, f_target)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        lr, sigma = params[h]
        hours.append({"hour": h, "q": hour_quantiles(p1, lr, sigma)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
