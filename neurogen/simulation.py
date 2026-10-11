"""Continuous runner with real measurements, an optional cue probe, and charts."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np

from .biology import BiologicalNetwork, BiologyConfig
from .probe import CueProbe, cue_column


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sensory_drive(network, stimulus="alternating"):
    """50 ms pulses every 250 ms, without resetting neural state."""
    if stimulus == "none" or network.time_ms % 250 >= 50:
        return None
    trial = int(network.time_ms // 250)
    column = (cue_column(network.config.seed, trial, network.config.columns_per_region)
              if stimulus == "cue" else trial % network.config.columns_per_region)
    return network.current(region=network.config.regions[0], column=column)


def add_simulation_parser(commands):
    parser = commands.add_parser("simulate", help="Continuous laminar network with charts and local plasticity")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--duration-ms", type=float, default=15000, help="Additional simulated time")
    duration.add_argument("--forever", action="store_true", help="Run until Ctrl-C, then checkpoint")
    parser.add_argument("--config", help="JSON BiologyConfig for a new network")
    parser.add_argument("--seed", type=int, help="Seed for a new network (default 1)")
    parser.add_argument("--resume", help="Biological .npz checkpoint, including live/probe state")
    parser.add_argument("--output", help="New, empty run directory (default: runs/<UTC timestamp>)")
    parser.add_argument("--report-ms", type=float, default=1000)
    parser.add_argument("--plot-ms", type=float, default=5000, help="Refresh charts and save an activity frame")
    parser.add_argument("--stimulus", choices=("cue", "alternating", "none"),
                        help="Default cue; inherited on resume. Only cue supplies prediction labels")
    parser.add_argument("--frozen", action="store_true", help="Disable all biological synaptic changes")
    parser.add_argument("--realtime", action="store_true", help="Pace against wall time when possible")


def run_simulation(args):
    for name in ("duration_ms", "report_ms", "plot_ms"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.resume:
        if args.config or args.seed is not None:
            raise ValueError("Resume supplies configuration and RNG; omit --config and --seed")
        network = BiologicalNetwork.load(args.resume)
    else:
        settings = json.loads(Path(args.config).read_text()) if args.config else {}
        if args.seed is not None:
            settings["seed"] = args.seed
        network = BiologicalNetwork(BiologyConfig(**settings))
    if args.frozen:
        network.config = replace(network.config, plasticity=False)
    previous_stimulus = network.protocol.get("stimulus", "cue")
    stimulus = args.stimulus or previous_stimulus
    if stimulus not in ("none", "alternating", "cue"):
        raise ValueError("Unknown checkpoint stimulus protocol")
    probe = CueProbe(network, network.protocol.get("probe") if previous_stimulus == "cue" else None) if stimulus == "cue" else None
    dt = network.config.dt_ms
    for value in (args.report_ms, args.duration_ms, args.plot_ms):
        if value < dt or not np.isclose(value / dt, round(value / dt)):
            raise ValueError("Duration/report/plot intervals must be positive integer multiples of dt_ms")
    if args.plot_ms < args.report_ms or not np.isclose(args.plot_ms / args.report_ms, round(args.plot_ms / args.report_ms)):
        raise ValueError("plot-ms must be an integer multiple of report-ms")
    steps, report_ticks = round(args.duration_ms / dt), round(args.report_ms / dt)
    plot_ticks = round(args.plot_ms / dt)
    directory = Path(args.output or ("runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")))
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory must be empty; choose a new directory for each run/resume segment")
    directory.mkdir(parents=True, exist_ok=True)
    from .visualization import ActivityDashboard
    dashboard = ActivityDashboard(directory, network)
    checkpoint = directory / "state.npz"
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True  # complete the current tick and observer update before saving

    def save():
        network.protocol = {"stimulus": stimulus}
        if probe:
            network.protocol["probe"] = probe.state()
        network.save(checkpoint)

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    start = time.monotonic()
    start_tick = previous_tick = network.tick
    previous_counts = network.spike_count.copy()
    last_render_tick = None
    summary = None
    print(f"Output: {directory}\nIntegration: {1000 / dt:.0f} ticks/s of simulated time; "
          f"{network.anatomy.size} neurons; stimulus={stimulus}", flush=True)
    print("Firing Hz means spikes per neuron per simulated second. Probe accuracy/loss measure the separate cue readout.", flush=True)
    try:
        with (directory / "metrics.jsonl").open("w") as log, (directory / "console.log").open("w") as console:
            header = {"event": "start", "config": asdict(network.config), "stimulus": stimulus,
                      "time_ms": network.time_ms, "source_sha256": source_hash(), "numpy": np.__version__,
                      "rate_definition": "spikes / neurons / simulated seconds in reporting window",
                      "probe_definition": "association E counts 50-150 ms after random L4 cue; predict before updating; 40 train/20 eval"}
            log.write(json.dumps(header) + "\n")
            log.flush()

            def report():
                nonlocal previous_tick, previous_counts, summary
                elapsed = (network.tick - previous_tick) * dt
                counts = network.spike_count - previous_counts
                row = network.report(previous_counts, elapsed)
                wall = max(time.monotonic() - start, 1e-9)
                row.update({"elapsed_ms": elapsed, "total_spikes_per_second": float(counts.sum() * 1000 / elapsed),
                            "median_neuron_hz": float(np.median(counts * 1000 / elapsed)),
                            "p95_neuron_hz": float(np.quantile(counts * 1000 / elapsed, .95)),
                            "simulation_tick_hz": 1000 / dt,
                            "wall_ticks_per_second": (network.tick - start_tick) / wall,
                            "simulated_seconds_per_wall_second": (network.tick - start_tick) * dt / (1000 * wall)})
                row.update(probe.summary() if probe else {"probe_train_trials": 0, "probe_eval_trials": 0,
                           "probe_train_accuracy": None, "probe_eval_accuracy": None,
                           "probe_train_loss": None, "probe_eval_loss": None})
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                dashboard.record(row, counts * 1000 / elapsed)
                text = (f"t={network.time_ms / 1000:7.2f}s | firing mean={row['mean_hz']:.2f} Hz/neuron "
                        f"E={row['excitatory_hz']:.2f} I={row['inhibitory_hz']:.2f} "
                        f"p95={row['p95_neuron_hz']:.1f} | network={row['total_spikes_per_second']:.0f} spikes/s "
                        f"| contacts={row['active_synapses']} +{row['created']}/-{row['pruned']}")
                for phase in ("train", "eval"):
                    accuracy, loss = row[f"probe_{phase}_accuracy"], row[f"probe_{phase}_loss"]
                    text += f" | probe {phase}=" + (f"{accuracy:.1%} CE={loss:.3f} n={row[f'probe_{phase}_trials']}" if accuracy is not None else "N/A")
                print(text, flush=True)
                console.write(text + "\n")
                console.flush()
                previous_tick, previous_counts, summary = network.tick, network.spike_count.copy(), row

            while not stop and (args.forever or network.tick - start_tick < steps):
                spikes = network.step(sensory_drive(network, stimulus))
                trial = probe.observe(network, spikes) if probe else None
                dashboard.observe(network, spikes, trial)
                if (network.tick - start_tick) % report_ticks == 0:
                    report()
                    save()
                    if (network.tick - start_tick) % plot_ticks == 0:
                        dashboard.render(network, stimulus)
                        last_render_tick = network.tick
                if args.realtime:
                    remaining = (network.tick - start_tick) * dt / 1000 - (time.monotonic() - start)
                    if remaining > 0:
                        time.sleep(min(remaining, .01))
            if network.tick > previous_tick:
                report()
            save()
            if last_render_tick != network.tick:
                dashboard.render(network, stimulus)
            footer = {"event": "stopped" if stop else "complete", "time_ms": network.time_ms,
                      "wall_seconds": time.monotonic() - start, "checkpoint": str(checkpoint),
                      "report": str(directory / "report.html")}
            log.write(json.dumps(footer) + "\n")
            (directory / "summary.json").write_text(json.dumps({"run": header, "last_window": summary, "completion": footer}, indent=2) + "\n")
            print(json.dumps(footer), flush=True)
    finally:
        dashboard.close()
        signal.signal(signal.SIGINT, previous_handler)
