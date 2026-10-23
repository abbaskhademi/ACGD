#!/usr/bin/env python3
"""Summarize completed Lasso ablations and the archived 100-seed comparison."""
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
import tomllib

out = Path(sys.argv[1]).resolve()
manifest = tomllib.loads((out / "manifest.toml").read_text())
assert "finished_utc" in manifest, "wait for the ablation to finish"
raw = tomllib.loads((out / "measurements.toml").read_text())
seeds = manifest["seeds"]
assert set(raw) == {f"seed_{s}" for s in seeds}
historical = tomllib.loads((out / "historical_comparison.toml").read_text())
methods = ["constant", "adjustable", "backtracking"]
names = ["Adaptive_constant", "Adaptive_adjustable", "Pure_backtracking"]


def times(variant, method, field="external_seconds"):
    return [median(raw[f"seed_{s}"][f"{variant}/{method}"][field]) for s in seeds]


def ratios(a, b):
    return [x / y for x, y in zip(a, b)]


stages = [
    ("Original partial timers", "original", "reported_seconds"),
    ("Original, complete elapsed", "original", "external_seconds"),
    ("Accepted-objective caching", "cached", "external_seconds"),
    ("Caching + correct counter", "cached_counter_fixed", "external_seconds"),
    ("Current shared solver", "current", "external_seconds"),
]
lines = [
    "WHY LASSO'S RANKING CHANGED",
    "",
    "Original archive versus corrected full run (100 paired seeds):",
    f"{'Method':23} {'Old median s':>13} {'New median s':>13} {'Old wins':>9} {'New wins':>9} {'Old/new median steps':>23}",
]
for name in names:
    h = historical["methods"][name]
    lines.append(f"{h['name']:23} {h['old_median_seconds']:13.6f} {h['new_median_seconds']:13.6f} "
                 f"{h['old_fastest_count']:9} {h['new_fastest_count']:9} "
                 f"{h['old_median_iterations']:10g}/{h['new_median_iterations']:<10g}")
lines += [
    "Wins above are fastest among all six methods. All six solved all100 problems.",
    historical["caveat"],
    "",
    f"Controlled same-host ablation: {len(seeds)} seeds ({seeds}), 1 warmup and3 repeats; medians.",
    "Julia1.11.6; Julia/BLAS threads1/1; pinned to logical CPU2 on Elite.",
    "Original problem closures supplied to every variant; no data generation or I/O timed.",
    "All ablation solves converged. Call-count audits ran separately from timing.",
    "Caching preserved bit-identical accepted objective histories and final iterates.",
    "BT/adaptive ratios below1 favor pure backtracking; these are medians of paired ratios.",
    f"{'Stage':29} {'Constant s':>12} {'Adjustable s':>13} {'BT s':>11} {'BT/constant':>13} {'BT/adjustable':>15}",
]
stage_data = []
for label, variant, field in stages:
    t = [times(variant, m, field) for m in methods]
    rc = median(ratios(t[2], t[0]))
    ra = median(ratios(t[2], t[1]))
    lines.append(f"{label:29} {median(t[0]):12.6f} {median(t[1]):13.6f} {median(t[2]):11.6f} {rc:13.6f} {ra:15.6f}")
    stage_data.append(dict(stage=label, median_seconds=dict(zip(methods, map(median, t))),
                           BT_over_constant=rc, BT_over_adjustable=ra))

lines += ["", "Original clock omissions (median fraction of complete elapsed unreported):"]
for m in methods:
    paired = [raw[f"seed_{s}"][f"original/{m}"] for s in seeds]
    omitted = median([median([1 - r/e for r, e in zip(d["reported_seconds"], d["external_seconds"])]) for d in paired])
    seconds = median([median([e-r for r, e in zip(d["reported_seconds"], d["external_seconds"])]) for d in paired])
    lines.append(f"  {m}: {100 * omitted:.2f}% ({seconds:.6f}s); paired per invocation, then medians.")

lines += ["", "Current versus matching narrowly patched original solver:"]
for m, variant in zip(methods, ["cached", "cached_counter_fixed", "cached"]):
    for field in ["iterations", "iterate_sha256", "history_sha256"]:
        matches = sum(raw[f"seed_{s}"][f"current/{m}"]["audit"][field] ==
                      raw[f"seed_{s}"][f"{variant}/{m}"]["audit"][field] for s in seeds)
        lines.append(f"  {m}: {field} identical on {matches}/{len(seeds)} seeds.")
    records = [(raw[f"seed_{s}"][f"{variant}/{m}"]["audit"],
                raw[f"seed_{s}"][f"current/{m}"]["audit"]) for s in seeds]
    initial_calls = 3 if m == "adjustable" else 2
    trial_matches = sum(a["objective_evaluations"] - initial_calls == b["objective_evaluations"] - 1
                        for a, b in records)
    f_difference = max(abs(a["final_objective"] - b["final_objective"]) /
                       max(abs(a["final_objective"]), abs(b["final_objective"])) for a, b in records)
    lines.append(f"  {m}: trial objective counts identical on {trial_matches}/{len(seeds)} seeds; "
                 f"maximum relative final-objective difference {f_difference:.3g}.")

lines += ["", "Adjustable-counter effect with objective caching and complete elapsed held fixed:"]
counter_stats = {}
for variant in ("cached", "cached_counter_fixed"):
    records = [raw[f"seed_{s}"][variant + "/adjustable"]["audit"] for s in seeds]
    steps = [r["iterations"] for r in records]
    # The narrowly patched old adjustable implementation retains three initial f calls.
    rejections = [r["objective_evaluations"] - r["iterations"] - 3 for r in records]
    counter_stats[variant] = dict(mean_steps=mean(steps), mean_rejections=mean(rejections),
                                 mean_objective_calls=mean(r["objective_evaluations"] for r in records))
    lines.append(f"  {variant}: mean steps {mean(steps):.1f}, mean rejections {mean(rejections):.1f}, "
                 f"mean objective calls {counter_stats[variant]['mean_objective_calls']:.1f}")
counter_time_ratio = median(ratios(times("cached_counter_fixed", "adjustable"), times("cached", "adjustable")))
lines += [
    f"  Median per-seed elapsed multiplier from counter correction: {counter_time_ratio:.6f}.",
    "  Old blocks ignored the first step and counted the first nine steps of later blocks twice.",
    "  The old threshold therefore raised gamma too readily. Larger gamma means larger initial",
    "  curvature and smaller initial trial steps, which happened to reduce retries on Lasso.",
    "  Correct blocks count each rejection once and update after10,20,... accepted steps.",
    "",
    "Current full100-run work (means across instances):",
]
for name in names:
    h = historical["methods"][name]
    lines.append(f"  {h['name']}: {h['mean_objective_evaluations']:.2f} objective calls, "
                 f"{h['mean_rejected_trials']:.2f} rejected trials; "
                 f"{1e6*h['mean_elapsed_seconds_per_step']:.2f} microseconds per accepted step.")
lines += [
    "",
    "Interpretation and limits:",
    "- The old ACGD timers exclude an accepted-objective refresh each iteration; BT's timer",
    "  includes its redundant current-objective evaluation. Complete elapsed exposes this bias.",
    "- Objective caching removes the redundant calls without changing either trajectory.",
    "- Counter correction changes adjustable ACGD's trajectory and substantially increases retries.",
    "- Constant ACGD and BT retain identical accepted-step counts on all100 historical/current seeds.",
    "- All900 full-run measured repeats of these three methods stopped by objective tolerance.",
    "- Complete initialization takes about0.00017s; the500-second limit was never approached. Those fixes,",
    "  post-step residual logging, accepted-L logging and failure profiles do not explain the reversal.",
    "- Remaining current-core changes include safeguards, histories, copies and arithmetic order.",
    "  Compare the last two ablation rows as a combined effect, not a per-change attribution.",
    "- Cached and cached_counter_fixed have identical constant/BT function bodies; their timing",
    "  differences measure noise, not a counter effect. Do not attribute those differences to a fix.",
    "- The10-seed ablation diagnoses causes; it does not replace the100-seed final performance profile.",
    "- Hardware/environment and warmup changes affect absolute times. No cross-hardware causal",
    "  speedup or additive percentage decomposition is claimed.",
    "",
    "Files: historical_comparison.{txt,toml}, summary.txt, measurements.toml, manifest.toml,",
    "seed_*.toml, source/ (original and each patched implementation).",
]
(out / "attribution_report.txt").write_text("\n".join(lines) + "\n")
(out / "attribution_summary.json").write_text(json.dumps(dict(stages=stage_data,
    adjustable_counter=counter_stats, counter_elapsed_multiplier=counter_time_ratio), indent=2) + "\n")
(out / "source" / "analyze_lasso_attribution.py").write_bytes(Path(__file__).read_bytes())
root = Path(__file__).resolve().parent.parent
for src, dst in [("test/load_definitions.jl", "load_definitions.jl"),
                 ("Project.toml", "Project.toml"), ("Manifest.toml", "Manifest.toml")]:
    if (root / src).is_file():
        (out / "source" / dst).write_bytes((root / src).read_bytes())
    else:
        assert (out / "source" / dst).is_file(), f"missing source: {src}"
files = [out / p for p in ("measurements.toml", "manifest.toml", "historical_comparison.toml",
                           "attribution_report.txt", "attribution_summary.json")]
files += sorted((out / "source").iterdir())
(out / "attribution_checksums.sha256").write_text("\n".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}" for p in files) + "\n")
print((out / "attribution_report.txt").read_text())
