import csv
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from neurogen import BiologicalNetwork, BiologyConfig
from neurogen.probe import CueProbe
from neurogen.simulation import sensory_drive
from neurogen.visualization import population_spectrum


class ObservabilityTests(unittest.TestCase):
    def test_rate_units_are_per_neuron_not_clock_or_total(self):
        n = BiologicalNetwork()
        n.spike_count[:] = 12
        n.tick = 1000
        self.assertEqual(n.report()["mean_hz"], 12)
        previous = np.full(n.anatomy.size, 10)
        self.assertEqual(n.report(previous, 500)["mean_hz"], 4)
        self.assertEqual(n.spike_count.sum(), 384 * 12)

    def test_spectrum_recovers_known_frequency_and_rejects_short_window(self):
        for dt in (.25, .5, 1):
            time = np.arange(round(5000 / dt)) * dt / 1000
            frequency, power = population_spectrum(5 + np.sin(2 * np.pi * 10 * time), dt, 384)
            self.assertAlmostEqual(frequency[np.argmax(power[1:]) + 1], 10, delta=.21)
        self.assertEqual(len(population_spectrum(np.zeros(99), 1, 384)[0]), 0)

    def test_probe_does_not_change_neurons_and_resumes_mid_trial(self):
        n, control = BiologicalNetwork(), BiologicalNetwork()
        probe = CueProbe(n)
        for _ in range(137):
            spikes = n.step(sensory_drive(n, "cue"))
            control.step(sensory_drive(control, "cue"))
            probe.observe(n, spikes)
        n.protocol = {"stimulus": "cue", "probe": probe.state()}
        with tempfile.TemporaryDirectory() as tmp:
            n.save(Path(tmp) / "state.npz")
            resumed = BiologicalNetwork.load(Path(tmp) / "state.npz")
        resumed_probe = CueProbe(resumed, resumed.protocol["probe"])
        for _ in range(500):
            spikes = n.step(sensory_drive(n, "cue"))
            control.step(sensory_drive(control, "cue"))
            other_spikes = resumed.step(sensory_drive(resumed, "cue"))
            self.assertEqual(probe.observe(n, spikes), resumed_probe.observe(resumed, other_spikes))
        for key in n.ARRAY_STATE:
            np.testing.assert_array_equal(getattr(n, key), getattr(control, key))
            np.testing.assert_array_equal(getattr(n, key), getattr(resumed, key))
        self.assertEqual(probe.state(), resumed_probe.state())

    def test_eval_labels_never_train_readout_and_loss_precedes_update(self):
        n = BiologicalNetwork()
        probe = CueProbe(n)
        n.tick = 1
        probe.observe(n, np.zeros(n.anatomy.size, bool))
        n.tick = 150
        row = probe.observe(n, np.ones(n.anatomy.size, bool))
        self.assertAlmostEqual(row["loss"], np.log(2))
        self.assertEqual(row["phase"], "train")
        self.assertGreater(np.abs(probe.weights).sum(), 0)
        weights, bias = probe.weights.copy(), probe.bias.copy()
        n.tick = 10001
        probe.observe(n, np.zeros(n.anatomy.size, bool))
        n.tick = 10150
        row = probe.observe(n, np.ones(n.anatomy.size, bool))
        self.assertEqual(row["phase"], "eval")
        np.testing.assert_array_equal(probe.weights, weights)
        np.testing.assert_array_equal(probe.bias, bias)

    def test_cli_writes_charts_and_rates_matching_raw_spikes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            command = [sys.executable, "-m", "neurogen", "simulate", "--duration-ms", "200",
                       "--report-ms", "100", "--plot-ms", "200", "--output", str(directory)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            reports = [r for r in rows if "mean_hz" in r]
            with (directory / "spikes.csv").open() as handle:
                spikes = list(csv.DictReader(handle))
            for row in reports:
                count = sum(row["time_ms"] - row["elapsed_ms"] < float(s["time_ms"]) <= row["time_ms"] for s in spikes)
                self.assertAlmostEqual(row["mean_hz"], count / 384 / (row["elapsed_ms"] / 1000))
                self.assertAlmostEqual(row["total_spikes_per_second"], count / (row["elapsed_ms"] / 1000))
            for name in ("activity", "spikes", "populations", "synapses", "frequency", "learning"):
                image = directory / "plots" / f"{name}.png"
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((directory / "report.html").exists())
            self.assertEqual(rows[-1]["event"], "complete")
            from audit_rates import audit
            checked = audit(directory)
            self.assertEqual(checked["window_hz_max_verification_error"], 0)
            self.assertEqual(BiologicalNetwork.load(directory / "state.npz").tick, 200)
            with (directory / "trials.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            # A typo must not mix results into a previous run.
            failed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("empty", failed.stderr)

    def test_ctrl_c_and_cli_resume_preserve_complete_probe_and_neural_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            cmd = [sys.executable, "-m", "neurogen", "simulate", "--forever",
                   "--report-ms", "50", "--plot-ms", "5000", "--output", str(first)]
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 15
                while not (first / "state.npz").exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((first / "state.npz").exists())
                child.send_signal(signal.SIGINT)
                out, err = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 0, out + err)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()
            self.assertEqual(json.loads((first / "summary.json").read_text())["completion"]["event"], "stopped")
            subprocess.run([sys.executable, "-m", "neurogen", "simulate", "--resume", str(first / "state.npz"),
                            "--duration-ms", "200", "--output", str(second)],
                           check=True, capture_output=True, text=True, timeout=30)
            restored = BiologicalNetwork.load(second / "state.npz")
            original = BiologicalNetwork()
            probe = CueProbe(original)
            for _ in range(restored.tick):
                probe.observe(original, original.step(sensory_drive(original, "cue")))
            for key in original.ARRAY_STATE:
                np.testing.assert_array_equal(getattr(original, key), getattr(restored, key))
            self.assertEqual(probe.state(), restored.protocol["probe"])


if __name__ == "__main__":
    unittest.main()
