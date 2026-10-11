"""Headless charts from measured spikes; bounded live buffers and streamed logs."""

from collections import deque
import csv
import html
import os
from pathlib import Path

# Matplotlib must not try writing to a user's read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def population_spectrum(spikes, dt_ms, neurons):
    """One-sided Hann periodogram of 5 ms population rate bins, Hz^2/Hz."""
    bin_ticks = round(5 / dt_ms)
    usable = len(spikes) // bin_ticks * bin_ticks
    if usable < round(1000 / dt_ms):
        return np.array([]), np.array([])
    rate = np.asarray(spikes)[-usable:].reshape(-1, bin_ticks).sum(axis=1) / (neurons * .005)
    rate -= rate.mean()
    window = np.hanning(len(rate))
    power = np.abs(np.fft.rfft(rate * window)) ** 2 / (200 * (window ** 2).sum())
    power[1:-1 if len(rate) % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(len(rate), .005), power


class ActivityDashboard:
    def __init__(self, directory, network):
        self.directory = Path(directory)
        self.plots = self.directory / "plots"
        self.plots.mkdir(parents=True)
        self.dt = network.config.dt_ms
        self.rows = deque(maxlen=2000)
        self.trials = deque(maxlen=2000)
        self.capacity = round(5000 / self.dt)
        self.spikes = np.zeros((self.capacity, network.anatomy.size), dtype=bool)
        self.cursor = self.recorded = 0
        self.last_rates = np.zeros(network.anatomy.size)
        self.handles = []
        self.spike_log = self._csv("spikes.csv", ["time_ms", "neuron"])
        self.trial_log = self._csv("trials.csv", ["time_ms", "trial", "phase", "label", "prediction", "correct",
                                               "loss", "accuracy", "mean_loss", "chance_accuracy"])
        self.rate_fields = ["time_ms", "elapsed_ms", "mean_hz", "excitatory_hz", "inhibitory_hz",
                            "total_spikes_per_second", "median_neuron_hz", "p95_neuron_hz", "max_neuron_hz",
                            "silent_fraction", "active_synapses", "created", "pruned", "simulation_tick_hz",
                            "wall_ticks_per_second", "simulated_seconds_per_wall_second"]
        self.rate_log = self._csv("rates.csv", self.rate_fields)
        self.neuron_log = self._csv("neuron_rates.csv", ["time_ms", "elapsed_ms", "neuron", "rate_hz"])

    def _csv(self, name, fields):
        handle = (self.directory / name).open("w", newline="")
        self.handles.append(handle)
        writer = csv.writer(handle)
        writer.writerow(fields)
        return writer

    def observe(self, network, spikes, trial=None):
        self.spikes[self.cursor] = spikes
        self.cursor = (self.cursor + 1) % self.capacity
        self.recorded += 1
        self.spike_log.writerows((network.time_ms, int(i)) for i in np.flatnonzero(spikes))
        if trial is not None:
            self.trials.append(trial)
            self.trial_log.writerow(list(trial.values()))

    def record(self, report, rates):
        self.rows.append(report)
        self.last_rates = rates.copy()
        self.rate_log.writerow([report[field] for field in self.rate_fields])
        self.neuron_log.writerows((report["time_ms"], report["elapsed_ms"], i, rate)
                                 for i, rate in enumerate(rates))
        for handle in self.handles:
            handle.flush()

    def recent(self):
        if self.recorded < self.capacity:
            return self.spikes[:self.recorded]
        return np.concatenate((self.spikes[self.cursor:], self.spikes[:self.cursor]))

    def _save(self, fig, name):
        fig.tight_layout()
        temporary = self.plots / (name + ".tmp")
        fig.savefig(temporary, format="png", dpi=120)
        os.replace(temporary, self.plots / (name + ".png"))
        plt.close(fig)

    def render(self, network, stimulus):
        if not self.rows:
            return
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        rows, a = list(self.rows), network.anatomy
        seconds = np.array([r["time_ms"] for r in rows]) / 1000
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for key, label in (("mean_hz", "All neurons"), ("excitatory_hz", "Excitatory"), ("inhibitory_hz", "Inhibitory")):
            axes[0, 0].plot(seconds, [r[key] for r in rows], label=label)
        axes[0, 0].set(title="Mean single-neuron firing rates", ylabel="Spikes / neuron / second")
        axes[0, 0].axhline(network.config.target_hz, color="gray", linestyle=":", label="Homeostasis target")
        axes[0, 0].legend()
        for region in network.config.regions:
            axes[0, 1].plot(seconds, [r["region_hz"][region] for r in rows], label=region)
        axes[0, 1].set(title="Regional firing", ylabel="Spikes / neuron / second")
        axes[0, 1].legend()
        axes[1, 0].plot(seconds, [100 * r["silent_fraction"] for r in rows])
        axes[1, 0].set(title="Silent neurons in each reporting window", ylabel="Percent", ylim=(0, 100))
        for key, label in (("active_synapses", "Active contacts"), ("created", "Created (cumulative)"), ("pruned", "Pruned (cumulative)")):
            axes[1, 1].plot(seconds, [r[key] for r in rows], label=label)
        axes[1, 1].set(title="Structural plasticity", ylabel="Contacts")
        axes[1, 1].legend()
        for axis in axes.flat:
            axis.set_xlabel("Simulation time (s)")
            axis.grid(alpha=.2)
        fig.suptitle(f"NeuroGen · {network.time_ms / 1000:.2f} simulated s · {a.size} cells")
        self._save(fig, "activity")
        # Keep an actual chronological series in addition to the latest charts.
        (self.plots / f"activity_{network.tick:012d}.png").write_bytes((self.plots / "activity.png").read_bytes())

        recent = self.recent()
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=(3, 1))
        ts, cells = np.nonzero(recent)
        ts = (network.time_ms - len(recent) * self.dt + (ts + 1) * self.dt) / 1000
        axes[0].scatter(ts, cells, s=1, c=np.where(a.excitatory[cells], "#1674b8", "#cc4b37"), rasterized=True)
        for region, name in enumerate(network.config.regions):
            ids = a.select(region=region)
            axes[0].axhline(ids[0] - .5, color="gray", linewidth=.5)
            axes[0].text(network.time_ms / 1000, ids.mean(), name, ha="right", backgroundcolor="white")
        axes[0].set(title="Actual spike events · recent ≤5 s · blue E / red I", ylabel="Neuron index")
        bin_ticks = round(5 / self.dt)
        usable = len(recent) // bin_ticks * bin_ticks
        if usable:
            values = recent[-usable:].sum(axis=1).reshape(-1, bin_ticks).sum(axis=1) / (a.size * .005)
            time_axis = (network.time_ms - usable * self.dt + (np.arange(len(values)) + 1) * 5) / 1000
            axes[1].plot(time_axis, values)
        axes[1].set(xlabel="Simulation time (s)", ylabel="Hz / neuron", title="Population spike rate in 5 ms bins")
        self._save(fig, "spikes")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for exc, color, label in ((True, "#1674b8", "E"), (False, "#cc4b37", "I")):
            axes[0].hist(self.last_rates[a.excitatory == exc], bins=np.linspace(0, max(1, self.last_rates.max()), 25),
                         alpha=.6, color=color, label=label)
        axes[0].set(title="Per-neuron rate distribution · last reporting window", xlabel="Spikes / second", ylabel="Neurons")
        axes[0].legend()
        matrix = np.array([[rows[-1]["population_hz"].get(f"{r}/L{layer}/{kind}", np.nan)
                            for r in network.config.regions for kind in ("E", "I")] for layer in range(1, 7)])
        im = axes[1].imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
        axes[1].set_xticks(range(matrix.shape[1]), [f"{r}\n{k}" for r in network.config.regions for k in ("E", "I")], rotation=30)
        axes[1].set_yticks(range(6), [f"L{x}" for x in range(1, 7)])
        axes[1].set_title("Layer / region / cell-type firing rates")
        fig.colorbar(im, ax=axes[1], label="Hz / neuron (blank = absent population)")
        self._save(fig, "populations")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for exc, label in ((True, "Excitatory"), (False, "Inhibitory")):
            axes[0].hist(network.weight[network.active & (network.excitatory_edge == exc)], bins=30, alpha=.6, label=label)
        axes[0].set(title="Active synaptic conductance magnitudes", xlabel="nS", ylabel="Contacts")
        axes[0].legend()
        for field, label in (("mean_ge_ns", "Excitatory conductance"), ("mean_gi_ns", "Inhibitory conductance")):
            axes[1].plot(seconds, [r[field] for r in rows], label=label)
        axes[1].set(title="Mean cell conductance at report time", xlabel="Simulation time (s)", ylabel="nS")
        axes[1].legend()
        self._save(fig, "synapses")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        frequency, power = population_spectrum(recent.sum(axis=1), self.dt, a.size)
        if len(frequency):
            axes[0].plot(frequency[1:], power[1:])
        else:
            axes[0].text(.5, .5, "At least 1 s of spikes needed", ha="center", transform=axes[0].transAxes)
        axes[0].set(title="Population-rate spectrum · recent ≤5 s · not EEG", xlabel="Oscillation frequency (Hz)", ylabel="Rate PSD (Hz² / Hz)")
        cvs = []
        for cell in range(a.size):
            intervals = np.diff(np.flatnonzero(recent[:, cell])) * self.dt
            if len(intervals) >= 2:
                cvs.append(intervals.std() / intervals.mean())
        if cvs:
            axes[1].hist(cvs, bins=25)
        axes[1].set(title=f"Inter-spike interval variability · {len(cvs)} eligible cells", xlabel="ISI coefficient of variation (≥3 spikes)", ylabel="Neurons")
        self._save(fig, "frequency")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for phase in ("train", "eval"):
            trials = [t for t in self.trials if t["phase"] == phase]
            if trials:
                tx = [t["time_ms"] / 1000 for t in trials]
                axes[0].plot(tx, [t["accuracy"] for t in trials], label=f"{phase}: cumulative accuracy")
                axes[1].plot(tx, [t["mean_loss"] for t in trials], label=f"{phase}: cumulative CE")
                axes[1].scatter(tx, [t["loss"] for t in trials], s=5, alpha=.2)
        if self.trials:
            axes[0].axhline(self.trials[-1]["chance_accuracy"], color="gray", linestyle=":", label="Chance")
            axes[0].legend()
            axes[1].legend()
        else:
            for axis in axes:
                axis.text(.5, .5, "No completed labeled trials" if stimulus == "cue" else "N/A — no labeled prediction task",
                          ha="center", transform=axis.transAxes)
        axes[0].set(title="Observer cue accuracy · prediction before update", ylabel="Fraction correct", ylim=(0, 1))
        axes[1].set(title="Observer cross-entropy · labels do not train cortex", ylabel="Loss (nats)")
        for axis in axes:
            axis.set_xlabel("Simulation time (s)")
        self._save(fig, "learning")

        latest = rows[-1]
        title = f"NeuroGen · {network.time_ms / 1000:.2f} simulated seconds"
        body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="refresh" content="15"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#eef2f6;color:#16283b}}
img{{width:100%;background:white;border-radius:10px}}section{{margin:24px 0}}p{{line-height:1.6}}code{{background:white;padding:3px}}</style>
<h1>{html.escape(title)}</h1><p>Mean firing: <b>{latest['mean_hz']:.2f} spikes/neuron/s</b> · E {latest['excitatory_hz']:.2f} Hz ·
I {latest['inhibitory_hz']:.2f} Hz · <b>{latest['total_spikes_per_second']:.0f} network spikes/s</b> ·
integration: {1000 / self.dt:.0f} ticks/s of simulated time.</p>
<p>Each point uses the measured simulation window. Raster and spectrum retain the latest 5 seconds;
time-series charts retain up to 2,000 reports/trials. CSV and JSONL preserve the full run.
Accuracy and loss belong to a separate supervised cue observer, not to Hebbian learning in the cortex.
Evaluation trials do not update the observer. The biological core keeps adapting without labels.</p>
<p><a href="metrics.jsonl">Metrics JSONL</a> · <a href="rates.csv">Rates CSV</a> ·
<a href="trials.csv">Trial predictions</a> · <a href="spikes.csv">Spike events</a> ·
<a href="neuron_rates.csv">Per-neuron rates</a> · <a href="state.npz">Checkpoint</a></p>"""
        for name in ("activity", "spikes", "populations", "synapses", "frequency", "learning"):
            body += f'<section><img src="plots/{name}.png?t={network.tick}" alt="{name} charts"></section>'
        body += "</html>"
        temporary = self.directory / "report.html.tmp"
        temporary.write_text(body)
        os.replace(temporary, self.directory / "report.html")

    def close(self):
        for handle in self.handles:
            handle.close()
