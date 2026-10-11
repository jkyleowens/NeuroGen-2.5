"""Reproducible biological mechanism, pathway, and readout experiments.

No labels enter the biological core. A ridge readout probes association spikes;
its supervised learning must not be attributed to local synaptic plasticity.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from neurogen.biology import BiologicalNetwork, BiologyConfig
from neurogen.simulation import sensory_drive, source_hash


def fit_readout(train, labels, test):
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1.)
    x, z = (train - mean) / scale, (test - mean) / scale
    target = 2 * labels - 1
    # Fixed regularization, no test-set selection. Dual form is small here.
    weights = x.T @ np.linalg.solve(x @ x.T + 10 * np.eye(len(x)), target)
    return z @ weights >= 0


def readout_trial(seed, condition, train_trials=40, test_trials=20):
    n = BiologicalNetwork(BiologyConfig(seed=seed, plasticity=condition == "plastic"))
    training_config = asdict(n.config)
    a = n.anatomy
    if condition == "disconnected":
        cut = a.region[a.pre] != a.region[a.post]
        n.active[cut] = False
        n.weight[cut] = 0
    initial = n.weight.copy()
    rng = np.random.default_rng(2026)
    labels = np.concatenate((rng.permutation(np.arange(train_trials) % 2),
                             rng.permutation(np.arange(test_trials) % 2)))
    association = a.select(region=1, excitatory=True)
    features, windows = [], []
    previous_counts = n.spike_count.copy()
    drives = [n.current(column=c) for c in (0, 1)]
    for trial, label in enumerate(labels):
        if trial == train_trials:
            # Held-out evaluation freezes weights/topology, but retains all live
            # state, background drive, adaptation and ongoing spike dynamics.
            n.config = replace(n.config, plasticity=False)
        counts = np.zeros(len(association))
        for tick in range(250):
            spikes = n.step(drives[label] if tick < 50 else None)
            if 50 <= tick < 150:
                counts += spikes[association]
        features.append(counts)
        if (trial + 1) % 4 == 0:
            windows.append(n.report(previous_counts, 1000))
            previous_counts = n.spike_count.copy()
    features = np.array(features)
    predictions = fit_readout(features[:train_trials], labels[:train_trials], features[train_trials:])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.npz"
        n.save(path)
        restored = BiologicalNetwork.load(path)
        for _ in range(30):
            n.step()
            restored.step()
        error = max(float(np.max(np.abs(getattr(n, name).astype(float) - getattr(restored, name).astype(float))))
                    for name in n.ARRAY_STATE)
    return {"seed": seed, "condition": condition, "training_config": training_config,
            "evaluation_config": asdict(n.config),
            "train_trials": train_trials, "test_trials": test_trials,
            "held_out_accuracy": float(np.mean(predictions == labels[train_trials:])),
            "chance_accuracy": .5, "test_labels": labels[train_trials:].tolist(),
            "test_predictions": predictions.astype(int).tolist(),
            "mean_absolute_weight_change_ns": float(np.abs(n.weight - initial).mean()),
            "checkpoint_max_state_error": error, "report": n.report(), "windows": windows}


def timing_curve():
    result = []
    for offset in (-100, -40, -20, -10, -5, 0, 5, 10, 20, 40, 100):
        n = BiologicalNetwork(BiologyConfig(regions=("sensory",), projections=(), columns_per_region=1,
                                             background_hz=0, scaling_tau_ms=1e30, structural=False))
        edge = np.flatnonzero(n.active & n.excitatory_edge)[0]
        n.weight[edge] = 1.5
        pre_time, post_time = max(0, -offset), max(0, offset)
        for tick in range(abs(offset) + 1):
            arrivals, spikes = np.zeros(len(n.weight), bool), np.zeros(n.anatomy.size, bool)
            arrivals[edge] = tick == pre_time
            spikes[n.anatomy.post[edge]] = tick == post_time
            n.learn(arrivals, spikes)
        result.append({"post_minus_arrival_ms": offset, "delta_weight_ns": float(n.weight[edge] - 1.5)})
    return result


def pathway_probe(seed, cut=False):
    # Matched noise and initial state isolate the causal effect of a sensory pulse.
    driven = BiologicalNetwork(BiologyConfig(seed=seed, plasticity=False))
    control = BiologicalNetwork(BiologyConfig(seed=seed, plasticity=False))
    if cut:
        for n in (driven, control):
            mask = n.anatomy.region[n.anatomy.pre] != n.anatomy.region[n.anatomy.post]
            n.active[mask] = False
            n.weight[mask] = 0
    differences = np.zeros(len(driven.config.regions), int)
    first = [None] * len(differences)
    drive = driven.current()
    for tick in range(250):
        x, y = driven.step(drive if tick < 50 else None), control.step()
        for region in range(len(differences)):
            changed = int(np.count_nonzero((x != y) & (driven.anatomy.region == region)))
            differences[region] += changed
            if changed and first[region] is None:
                first[region] = driven.time_ms
    return {"seed": seed, "interregional_cut": cut, "different_spike_events": differences.tolist(),
            "first_different_spike_ms": first, "region_order": list(driven.config.regions)}


def stability_trial(seed, duration_ms):
    n = BiologicalNetwork(BiologyConfig(seed=seed))
    windows, counts = [], n.spike_count.copy()
    for tick in range(duration_ms):
        n.step(sensory_drive(n))
        if (tick + 1) % 10000 == 0:
            windows.append(n.report(counts, 10000))
            counts = n.spike_count.copy()
    return {"seed": seed, "duration_ms": duration_ms, "config": asdict(n.config),
            "report": n.report(), "ten_second_windows": windows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", default="runs/benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "/results.json")
    parser.add_argument("--stability-ms", type=int, default=60000)
    args = parser.parse_args()
    if args.stability_ms < 10000 or args.stability_ms % 10000:
        parser.error("--stability-ms must be a positive multiple of 10000")
    start = time.monotonic()
    results = {"source_sha256": source_hash(), "numpy": np.__version__, "complete": False,
               "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "task": "which sensory column was stimulated; delayed association E spike readout",
               "protocol": "50 ms 250 pA L4 E pulse; association counts 50-150 ms; 250 ms trials; no resets",
               "readout": "ridge alpha=10; standardization from training only; no core task gradients",
               "timing_curve": timing_curve(), "runs": [], "pathway_probes": [], "stability": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for condition in ("plastic", "frozen", "disconnected"):
            row = readout_trial(seed, condition)
            results["runs"].append(row)
            print(json.dumps({"seed": seed, "condition": condition, "accuracy": row["held_out_accuracy"],
                              "mean_hz": row["report"]["mean_hz"], "created": row["report"]["created"],
                              "pruned": row["report"]["pruned"]}), flush=True)
            path.write_text(json.dumps(results, indent=2) + "\n")
        results["pathway_probes"].extend([pathway_probe(seed), pathway_probe(seed, cut=True)])
        stability = stability_trial(seed, args.stability_ms)
        results["stability"].append(stability)
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"seed": seed, "stability_ms": args.stability_ms,
                          "final_window_hz": stability["ten_second_windows"][-1]["mean_hz"]}), flush=True)
    results["complete"] = True
    results["wall_seconds"] = time.monotonic() - start
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
