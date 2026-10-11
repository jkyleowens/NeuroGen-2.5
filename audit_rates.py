"""Independently check reported firing rates against a run's raw spike CSV."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neurogen.visualization import population_spectrum


def audit(directory):
    directory = Path(directory)
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    header = records[0]
    rows = [row for row in records if "mean_hz" in row]
    if not rows:
        raise ValueError("Run contains no completed reporting windows")
    neurons = rows[0]["neurons"]
    dt = header["config"]["dt_ms"]
    end = rows[-1]["time_ms"]
    start = rows[0]["time_ms"] - rows[0]["elapsed_ms"]
    duration = (end - start) / 1000
    recent_ticks = round(min(5000, end - start) / dt)
    recent = np.zeros((recent_ticks, neurons), dtype=bool)
    ends = np.array([row["time_ms"] for row in rows])
    # Offline validation may load all reporting windows, but never all spike rows.
    counts = np.zeros((len(rows), neurons), dtype=np.int64)
    with (directory / "spikes.csv").open() as handle:
        for spike in csv.DictReader(handle):
            when, cell = float(spike["time_ms"]), int(spike["neuron"])
            window = np.searchsorted(ends, when, side="left")
            if not start < when <= end or not 0 <= cell < neurons:
                raise ValueError("Spike outside the recorded run or neuron range")
            counts[window, cell] += 1
            if when > end - recent_ticks * dt:
                index = round((when - (end - recent_ticks * dt)) / dt) - 1
                recent[index, cell] = True
    errors = []
    for row, observed in zip(rows, counts):
        actual = observed.sum() / neurons / (row["elapsed_ms"] / 1000)
        errors.append(abs(actual - row["mean_hz"]))
    rates = counts.sum(axis=0) / duration
    frequency, power = population_spectrum(recent.sum(axis=1), dt, neurons)
    bins = round(5 / dt)
    usable = len(recent) // bins * bins
    binned = recent[-usable:].sum(axis=1).reshape(-1, bins).sum(axis=1) / (neurons * .005) if usable else np.array([])
    cvs, max_isi_hz = [], 0.
    for cell in range(neurons):
        intervals = np.diff(np.flatnonzero(recent[:, cell])) * dt
        if len(intervals):
            max_isi_hz = max(max_isi_hz, 1000 / intervals.min())
        if len(intervals) >= 2:
            cvs.append(float(intervals.std() / intervals.mean()))
    result = {"source_sha256": header["source_sha256"], "stimulus": header["stimulus"],
              "duration_seconds": duration, "neurons": neurons, "total_spikes": int(counts.sum()),
              "mean_hz_from_raw_events": float(rates.mean()),
              "network_spikes_per_second": float(counts.sum() / duration),
              "window_hz_max_verification_error": max(errors), "simulation_tick_hz": 1000 / dt,
              "whole_run_neuron_rate_percentiles_hz": {str(q): float(np.percentile(rates, q)) for q in (0, 25, 50, 75, 95, 100)},
              "recent_window_ms": len(recent) * dt,
              "recent_peak_5ms_population_rate_hz": float(binned.max()) if len(binned) else None,
              "recent_population_spectral_peak_hz": float(frequency[1:][power[1:].argmax()]) if len(frequency) else None,
              "recent_max_inverse_isi_hz": float(max_isi_hz) if max_isi_hz else None,
              "recent_mean_isi_cv": float(np.mean(cvs)) if cvs else None,
              "isi_cv_eligible_neurons": len(cvs),
              "interpretation": "Mean firing, inverse ISI, population oscillation and integration clock are different quantities; spectrum is not EEG."}
    if max(errors) > 1e-10:
        raise ValueError(f"Reported rates disagree with raw spikes: {max(errors)} Hz")
    (directory / "rate_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_directory), indent=2))


if __name__ == "__main__":
    main()
