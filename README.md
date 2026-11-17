# gta6-twitch-ccu-forecast-01 — GTA VI launch-day Twitch CCU forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/wizzy-bounty/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/wizzy-bounty/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Our entry for make.fun's **gta6-twitch-ccu-forecast-01** — forecast, hour by
hour, how many people watch Grand Theft Auto VI live on Twitch over the 24
UTC hours of its release day (2026-11-19), from everything recorded up to the
end of its first hour, as five quantiles an hour. $250 across the top 20
places; a prize needs a pinball loss strictly below the best house entry.

## Model (v2 — early-wave momentum)

For each forecast hour h = 2..24, the log ratio of that hour's viewers to
hour 1's print is a line in the log of hour 1's print plus one global
early-wave momentum term:

```
ln(y_h / p1) = a_h + b_h * ln(p1) + c * s(h) * f + noise,  noise ~ N(0, sigma_h)
q_tau(h) = p1 * exp(a_h + b_h*ln(p1) + c*s(h)*f + z_tau*sigma_h)
```

fit by alternating least squares at run time over the case's own comparable
launches (nothing is hard-coded — the final case's 42 comparables, practice
launches included, are used the same way). The size term is the first edge:
across past launches the day's peak-to-hour-1 ratio falls steeply as hour 1
grows (correlation of ln(p1) with ln(peak/p1) ≈ −0.82 over the donors), so a
small hour-1 print predicts a steep all-day ramp and a huge one predicts a
nearly flat day. The slope b_h is clamped to [−0.55, 0.05]; the intercept is
refit after clamping; sigma_h is the residual standard deviation of the fit
times 0.9, floored at 0.15. ln(p1) is winsorised to the donors' range, so an
unprecedented hour-1 print never extrapolates the line.

The second edge is the early-wave momentum f. The game unlocks at rolling
local midnight, so by the checkpoint it has already been live ~14 hours in
New Zealand / Australia and ~2 hours in central Europe. f is the log of the
mean of the last three hourly points in the 13 hours before the checkpoint
(22:00, 23:00 UTC the previous day and hour 1 itself) over the hour-1
print: how hot the launch was running into hour 1. One global scale c
carries f through a fixed shape s(h) — −0.55 at hours 2–4 (a hot early wave
mean-reverts into a deeper trough), 0 at hour 5, ramping linearly to 1.0 at
hour 13 and flat after (a hot early wave carries one-for-one into the
evening peak) — instead of 23 free per-hour coefficients, so the momentum
term cannot overfit the nineteen donors.

Donors are the comparables that are real launches of the same kind: tier
"comparable" in their unlock note, with a live hour 1 (hour-1 viewers at
least 3% of that launch's release-day peak). That filter drops
simultaneous-global launches whose hour 1 was still pre-unlock trickle
(their hour-1 ratios explode to 10–90× and wreck the fit),
early-access-contaminated and weak launches, and GTA VI's own pre-release
category history. With too few donors the filters relax in steps; with
none at all the forecast degrades to the house persistence shape.

The shape this recovers is the rolling-midnight unlock wave, now
steepened for launches whose early wave ran hot: a dip to ~0.6× hour 1 in
hours 3–4 (only NZ/CET/UK playing), a step at hour 5–6 (US East unlocks
05:00Z), then a climb into US prime time, peaking ~2–6× hour 1 around
hours 19–21 (18:00–21:00Z ≈ 13:00–16:00 EST) before easing off — the peak
ratio rising sharply for small hour-1 prints (Wolverine's real peak was
9.1× its hour 1; the v2 forecast's late hours sit ~6.5×, still low but far
closer than v1's ~2.5×).

## Practice results (local scoring)

Ground truth for the practice cases was re-read from SullyGnome's public
hourly chart data — the same source the kit was cut from — and checked
against each case file's known hour-1 print (exact match on all three).
Red Dead Redemption 2 (p01) could not be scored locally: its month
failed to load and was not retried, so the mean below is over p00/p02/p03.

| case | p1 | v2 model | v1 model | house baseline | coverage_90 |
|---|---|---|---|---|---|
| p00 Marvel's Wolverine | 23,357 | **0.1673** | 0.2360 | 0.7437 | 0.91 |
| p02 God of War Ragnarök | 269,517 | **0.1192** | 0.1370 | 0.2146 | 1.00 |
| p03 Marvel's Spider-Man 2 | 119,087 | **0.1856** | 0.1877 | 0.3033 | 0.78 |
| **mean** | | **0.1574** | **0.1869** | **0.4205** | **0.90** |

The model beats the house baseline on every scored practice case and beats
v1 on every scored case too (−29% / −13% / −1% loss; −16% on the mean).
Leave-one-out over the 19 donor launches: model 0.251 vs baseline 0.403
(v1: 0.261) — the momentum term wins out-of-sample as well, not just on
the three practice cases. The v1 submission (size-only regression) scored
0.181 on the server's 4-case practice (rank #356/1606); v2's local 0.157
should land clearly ahead of it, though still above the top-10 bar (~0.139).

Honest limits: the model still under-calls the steepest small-launch
ramps (Wolverine's real peak was 9.1× its hour 1; the forecast's late
hours sit ~6.5×, inside the band but low) and the early-hour trough of
rolling launches runs deeper than the median shape (Spider-Man 2's hours
3–4 printed ~0.42× hour 1). A console-only donor subset was tried and
rejected: better on two practice cases, far worse on Wolverine (0.392,
coverage 0.48). Kernel-weighted local-linear fits, hour-smoothing, a
peak-ratio + normalised-shape split, a quadratic size term, pre-release
baseline and diurnal-profile features, and empirical residual quantiles
were all tried and rejected — each either lost on practice or lost
leave-one-out.

## Reference numbers used

- Launch-day Twitch peaks (single-player records): Hogwarts Legacy
  1.28M concurrent viewers (Guinness; early-access peak, release-day
  ~505k), Cyberpunk 2077 1.14M, Elden Ring ~910k, God of War Ragnarök
  ~478k, Black Myth: Wukong 485.3k peak with 5.5M hours watched on
  day 1 (Streams Charts).
- Twitch prime time: 6–10 PM EST is the absolute peak window
  (4–11 PM ET / 6 PM–1 AM CET); evenings carry the most viewers and
  the most competing channels.
- GTA VI unlock waves (rolling local midnight): NZ 11:00Z Nov 18,
  CET 23:00Z, UK 00:00Z, US East 05:00Z, US West 08:00Z Nov 19 —
  so hour 1 (00:00–01:00Z) only captures UK + NZ/CET players, and US
  prime time falls in the last hours of the UTC day.
- GTA VI category probe (kit, 2026-11-17T03:10Z): 1,240 viewers,
  181 channels (GTA V at the same read: 81,611 viewers, 3,514
  channels). GTA VI releases on PS5 / Xbox Series X|S only, no PC
  until 2027, single-player.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13 standard library only, no network. Reads every `*.in` in
the input dir, writes `<id>.out` as
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` —
exactly 23 rows, hours 2..24, each q five finite numbers in (0, 1e9],
non-decreasing, file ~1.5 KB (cap 65,536). Deterministic (pure
arithmetic on the case file); ~0.06 s per case, far under the 8 s cap.

## Fallbacks

- Too few donors after filtering → filters relax in steps
  (comparable tier, then any live hour 1).
- No donors at all → house persistence shape (flat median, spread
  widening like a random walk).
- No hour-1 print, or a malformed case → the run raises, as the
  starter does; a case file that cannot be forecast is a bug, not
  something to paper over with a silent constant.
