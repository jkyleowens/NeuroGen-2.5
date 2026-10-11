from pathlib import Path
import tempfile
import unittest

import numpy as np

from neurogen.biology import BiologicalNetwork, BiologyConfig


class BiologyTests(unittest.TestCase):
    def small(self, **kwargs):
        return BiologicalNetwork(BiologyConfig(regions=("sensory",), projections=(),
                                               columns_per_region=1, **kwargs))

    def test_laminar_and_regional_anatomy(self):
        n = BiologicalNetwork()
        a = n.anatomy
        self.assertEqual(a.size, 384)
        self.assertEqual(set(a.layer), set(range(1, 7)))
        self.assertFalse(a.excitatory[a.layer == 1].any())
        self.assertFalse((a.pre == a.post).any())
        self.assertEqual(len(set(zip(a.pre, a.post))), len(a.pre))
        distant = a.region[a.pre] != a.region[a.post]
        self.assertTrue(a.excitatory[a.pre[distant]].all())
        ff = a.pathway == 2
        self.assertTrue(np.isin(a.layer[a.pre[ff]], (2, 3)).all())
        self.assertTrue((a.layer[a.post[ff]] == 4).all())
        fb = a.pathway == 3
        self.assertTrue(np.isin(a.layer[a.pre[fb]], (5, 6)).all())
        self.assertTrue(np.isin(a.layer[a.post[fb]], (1, 2, 6)).all())
        self.assertTrue((n.weight[n.active] > 0).all())
        self.assertTrue((a.delay >= 1).all())

    def test_delayed_transmission_and_dale_sign(self):
        for excitatory in (True, False):
            n = self.small(background_hz=0, plasticity=False)
            a = n.anatomy
            edge = np.flatnonzero(n.excitatory_edge == excitatory)[0]
            a.delay[edge] = 3
            # Resize ring for this controlled single-contact delay experiment.
            n.queue = np.zeros((4, len(a.pre)))
            n.history = np.zeros((4, a.size), bool)
            n.active[:] = False
            n.active[edge] = True
            n.weight[:] = 0
            n.weight[edge] = 2
            n.voltage[:] = -65
            n.voltage[a.pre[edge]] = 0  # induce one emission, no external drive
            self.assertTrue(n.step()[a.pre[edge]])
            for _ in range(2):
                n.step()
                self.assertEqual(n.ge.sum() + n.gi.sum(), 0)
            n.step()
            target = a.post[edge]
            self.assertEqual(n.ge[target] if excitatory else n.gi[target], 2)
            self.assertGreater(n.voltage[target], -65) if excitatory else self.assertLess(n.voltage[target], -65)

    def test_stdp_causal_anticausal_and_simultaneous(self):
        changes = []
        for offset in (10, -10, 0, 100):
            n = self.small(background_hz=0, scaling_tau_ms=1e30, structural=False)
            edge = np.flatnonzero(n.active & n.excitatory_edge)[0]
            n.weight[edge] = 1.5
            pre_time, post_time = (0, offset) if offset >= 0 else (-offset, 0)
            for t in range(abs(offset) + 1):
                arrivals, spikes = np.zeros(len(n.weight), bool), np.zeros(n.anatomy.size, bool)
                arrivals[edge] = t == pre_time
                spikes[n.anatomy.post[edge]] = t == post_time
                n.learn(arrivals, spikes)
            changes.append(n.weight[edge] - 1.5)
        self.assertGreater(changes[0], 0)
        self.assertLess(changes[1], 0)
        self.assertEqual(changes[2], 0)
        self.assertLess(changes[3], changes[0] / 10)

    def test_inhibitory_homeostasis_and_scaling_direction(self):
        n = self.small(background_hz=0, stdp_rate=0, scaling_tau_ms=1000)
        edge = np.flatnonzero(n.active & ~n.excitatory_edge & n.anatomy.excitatory[n.anatomy.post])[0]
        arrivals, spikes = np.zeros(len(n.weight), bool), np.zeros(n.anatomy.size, bool)
        arrivals[edge] = True
        initial = n.weight[edge]
        n.learn(arrivals, spikes)
        self.assertLess(n.weight[edge], initial)
        n.post_trace[n.anatomy.post[edge]] = 1
        initial = n.weight[edge]
        n.learn(arrivals, spikes)
        self.assertGreater(n.weight[edge], initial)
        exc = np.flatnonzero(n.active & n.excitatory_edge)[0]
        for rate, direction in ((0, 1), (20, -1)):
            n.rate[:] = rate
            initial = n.weight[exc]
            n.learn(np.zeros_like(arrivals), np.zeros_like(spikes))
            self.assertGreater((n.weight[exc] - initial) * direction, 0)

    def test_activity_driven_contact_creation_and_removal(self):
        n = self.small(background_hz=0)
        edge = np.flatnonzero(~n.active & n.excitatory_edge)[0]
        self.assertEqual(n.weight[edge], 0)
        n.rewire()
        self.assertFalse(n.active[edge])  # silence does not conjure correlations
        n.coactivity[edge] = .1
        n.rewire()
        self.assertTrue(n.active[edge])
        self.assertEqual(n.created, 1)
        self.assertGreater(n.weight[edge], 0)
        n.coactivity[edge] = 0
        n.weight[edge] = .05
        n.age_ms[edge] = n.config.structural_grace_ms
        n.queue[:, edge] = 1
        n.rewire()
        self.assertFalse(n.active[edge])
        self.assertEqual(n.weight[edge], 0)
        self.assertEqual(n.queue[:, edge].sum(), 0)
        self.assertEqual(n.pruned, 1)
        n.coactivity[:] = 1
        for _ in range(100):
            n.rewire()
        a = n.anatomy
        self.assertLessEqual(np.bincount(a.post[n.active], minlength=a.size).max(), n.config.max_fanin)
        self.assertLessEqual(np.bincount(a.pre[n.active], minlength=a.size).max(), n.config.max_fanout)

    def test_frozen_synapses_and_no_drive_continuity(self):
        n = self.small(plasticity=False)
        initial, active = n.weight.copy(), n.active.copy()
        for _ in range(100):
            n.step()
        self.assertEqual(n.time_ms, 100)
        self.assertGreater(n.spike_count.sum(), 0)
        np.testing.assert_array_equal(n.weight, initial)
        np.testing.assert_array_equal(n.active, active)
        n = self.small(background_hz=0)
        n.voltage[:] = -65
        for _ in range(50):
            self.assertFalse(n.step().any())
        self.assertEqual(n.created, 0)

    def test_live_checkpoint_replays_noise_delays_and_rewiring_exactly(self):
        n = BiologicalNetwork(BiologyConfig(structural_interval_ms=20, structural_grace_ms=50))
        for _ in range(57):
            n.step(n.current())
        self.assertGreater(n.queue.sum(), 0)
        n.protocol = {"stimulus": "none"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npz"
            n.save(path)
            restored = BiologicalNetwork.load(path)
        for _ in range(150):
            np.testing.assert_array_equal(n.step(), restored.step())
        for name in n.ARRAY_STATE:
            np.testing.assert_array_equal(getattr(n, name), getattr(restored, name))
        self.assertEqual((n.tick, n.created, n.pruned, n.protocol),
                         (restored.tick, restored.created, restored.pruned, restored.protocol))
        self.assertEqual(n.rng.bit_generator.state, restored.rng.bit_generator.state)

    def test_invalid_configuration_and_input(self):
        for kwargs in ({"dt_ms": 2}, {"background_hz": -1}, {"target_hz": float("nan")},
                       {"columns_per_region": 0}, {"max_fanout": 1.5},
                       {"regions": ("unknown",)}, {"regions": ("a", "a")},
                       {"seed": -1}, {"plasticity": "false"}, {"structural_interval_ms": 1.5}):
            with self.assertRaises(ValueError):
                BiologyConfig(**kwargs)
        n = self.small()
        for current in (np.zeros(3), np.full(64, float("nan"))):
            with self.assertRaises(ValueError):
                n.step(current)
        self.assertEqual(n.tick, 0)

    def test_region_pathways_carry_a_causal_sensory_effect(self):
        from benchmark_biology import pathway_probe
        intact, cut = pathway_probe(1), pathway_probe(1, cut=True)
        self.assertTrue(all(count > 0 for count in intact["different_spike_events"]))
        self.assertEqual(cut["different_spike_events"][1:], [0, 0])
        first = intact["first_different_spike_ms"]
        self.assertLess(first[0], first[1])
        self.assertLess(first[1], first[2])

    def test_passive_membrane_solution_across_timesteps(self):
        # With no synapses or background, 100 pA drives a 200 pF / 10 nS E
        # membrane toward -55 mV with tau=20 ms. This checks physical units and
        # integration independently of the implementation's spike rule.
        for dt in (.25, .5, 1.):
            n = self.small(background_hz=0, plasticity=False, dt_ms=dt)
            n.active[:] = False
            n.weight[:] = 0
            n.voltage[:] = -65
            current = np.full(n.anatomy.size, 100.)
            for _ in range(round(10 / dt)):
                n.step(current)
            expected = -55 - 10 * np.exp(-10 / 20)
            np.testing.assert_allclose(n.voltage[n.anatomy.excitatory], expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
