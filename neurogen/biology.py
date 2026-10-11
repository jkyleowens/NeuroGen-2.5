"""Continuous conductance-based adaptive LIF network with local learning.

Units: ms, mV, pA, pF, nS, Hz. CPU NumPy; no autograd or task labels.
See BIOLOGICAL_DESIGN.md for equations, sources, and deliberate simplifications.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .anatomy import build_anatomy


@dataclass(frozen=True)
class BiologyConfig:
    regions: tuple = ("sensory", "association", "motor")
    columns_per_region: int = 2
    projections: tuple = (("sensory", "association", "feedforward"),
                          ("association", "motor", "feedforward"),
                          ("motor", "association", "feedback"),
                          ("association", "sensory", "feedback"))
    dt_ms: float = 1.0
    seed: int = 1
    background_hz: float = 800.0
    background_weight_ns: float = .5
    target_hz: float = 5.0
    stdp_rate: float = .005
    inhibitory_rate: float = .002
    scaling_tau_ms: float = 100_000.0
    structural_interval_ms: float = 500.0
    structural_grace_ms: float = 5_000.0
    growth_threshold: float = .001
    max_fanin: int = 48
    max_fanout: int = 80
    plasticity: bool = True
    structural: bool = True

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "projections", tuple(tuple(p) for p in self.projections))
        if not self.regions or len(set(self.regions)) != len(self.regions):
            raise ValueError("Regions must be unique and nonempty")
        if any(not isinstance(r, str) or not r for r in self.regions):
            raise ValueError("Region names must be nonempty strings")
        for name in ("columns_per_region", "max_fanin", "max_fanout"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dt_ms not in (.25, .5, 1.0):
            raise ValueError("dt_ms must be 0.25, 0.5, or 1.0 (delays divide exactly)")
        for name in ("background_hz", "background_weight_ns", "stdp_rate", "inhibitory_rate",
                     "growth_threshold", "structural_grace_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("target_hz", "scaling_tau_ms", "structural_interval_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.structural_interval_ms < self.dt_ms:
            raise ValueError("Structural interval must be at least one tick")
        if not math.isclose(self.structural_interval_ms / self.dt_ms,
                            round(self.structural_interval_ms / self.dt_ms)):
            raise ValueError("Structural interval must be an integer multiple of dt_ms")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.plasticity) is not bool or type(self.structural) is not bool:
            raise ValueError("Plasticity switches must be booleans")
        seen = set()
        for src, dst, kind in self.projections:
            if src not in self.regions or dst not in self.regions or src == dst:
                raise ValueError("Projections must join distinct configured regions")
            if kind not in ("feedforward", "feedback") or (src, dst) in seen:
                raise ValueError("Invalid or duplicate projection")
            seen.add((src, dst))


class BiologicalNetwork:
    ARRAY_STATE = ("weight", "active", "voltage", "adaptation", "ge", "gi", "refractory",
                   "pre_trace", "post_trace", "potential_trace", "coactivity", "usage",
                   "rate", "age_ms", "queue", "history", "spike_count", "last_spikes")

    def __init__(self, config=None):
        self.config = config or BiologyConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)
        self.anatomy = build_anatomy(c, self.rng)
        a = self.anatomy
        self.excitatory_edge = a.excitatory[a.pre]
        self.weight = self.rng.uniform(1.2, 2.0, len(a.pre))
        self.weight[~self.excitatory_edge] = self.rng.uniform(4, 6, (~self.excitatory_edge).sum())
        self.weight[a.pathway == 2] *= 1.5
        self.active = np.zeros(len(a.pre), dtype=bool)
        fanin, fanout = np.zeros(a.size, int), np.zeros(a.size, int)
        for edge in self.rng.permutation(len(a.pre)):
            pre, post = a.pre[edge], a.post[edge]
            if self.rng.random() < .65 and fanin[post] < c.max_fanin and fanout[pre] < c.max_fanout:
                self.active[edge] = True
                fanin[post] += 1
                fanout[pre] += 1
        self.weight[~self.active] = 0
        self.voltage = self.rng.uniform(-68, -60, a.size)
        for name in ("adaptation", "ge", "gi", "refractory", "post_trace", "rate"):
            setattr(self, name, np.zeros(a.size))
        for name in ("pre_trace", "potential_trace", "coactivity", "usage", "age_ms"):
            setattr(self, name, np.zeros(len(a.pre)))
        self.queue = np.zeros((int(a.delay.max()) + 1, len(a.pre)))
        self.history = np.zeros((len(self.queue), a.size), dtype=bool)
        self.spike_count = np.zeros(a.size, dtype=np.int64)
        self.last_spikes = np.zeros(a.size, dtype=bool)
        self.tick = self.created = self.pruned = 0
        self.protocol = {}
        self._edge_indices = np.arange(len(a.pre))

    @property
    def time_ms(self):
        return self.tick * self.config.dt_ms

    def current(self, region="sensory", column=0, layer=4, amplitude_pa=250):
        """Explicit current injection into E cells; no hidden token encoder."""
        if region not in self.config.regions or not 0 <= column < self.config.columns_per_region:
            raise ValueError("Unknown region or column")
        if layer not in range(1, 7) or not math.isfinite(amplitude_pa):
            raise ValueError("Invalid layer or current")
        drive = np.zeros(self.anatomy.size)
        indices = self.anatomy.select(self.config.regions.index(region), column, layer, True)
        drive[indices] = amplitude_pa
        return drive

    def step(self, current_pa=None):
        """Advance all cells one tick. Omitted sensory drive still advances time.

        Delayed amplitudes are captured on emission; pruning cancels in-flight
        events on that contact. Plasticity affects future emissions only.
        """
        c, a = self.config, self.anatomy
        drive = np.zeros(a.size) if current_pa is None else np.asarray(current_pa, dtype=float)
        if drive.shape != (a.size,) or not np.isfinite(drive).all():
            raise ValueError(f"current_pa must be a finite vector of length {a.size}")
        dt = c.dt_ms
        slot = self.tick % len(self.queue)
        amplitudes = self.queue[slot].copy() * self.active
        self.queue[slot] = 0
        arrivals = amplitudes > 0
        potential_arrivals = self.history[(self.tick - a.delay) % len(self.history), a.pre]
        self.ge *= math.exp(-dt / 5)
        self.gi *= math.exp(-dt / 10)
        np.add.at(self.ge, a.post[self.excitatory_edge], amplitudes[self.excitatory_edge])
        np.add.at(self.gi, a.post[~self.excitatory_edge], amplitudes[~self.excitatory_edge])
        self.ge += c.background_weight_ns * self.rng.poisson(c.background_hz * dt / 1000, a.size)
        self.adaptation *= math.exp(-dt / 200)
        # C dV/dt = gL(EL-V) + ge(Ee-V) + gi(Ei-V) - w + I.
        # Exact constant-conductance integration for this tick (operator split).
        capacitance = np.where(a.excitatory, 200., 100.)
        total_g = 10 + self.ge + self.gi
        equilibrium = (-650 - 80 * self.gi - self.adaptation + drive) / total_g
        available = self.refractory <= 0
        self.voltage = equilibrium + (self.voltage - equilibrium) * np.exp(-dt * total_g / capacitance)
        self.voltage[~available] = -65
        spikes = (self.voltage >= -50) & available
        self.voltage[spikes] = -65
        self.refractory = np.maximum(0, self.refractory - dt)
        self.refractory[spikes] = 2
        self.adaptation[spikes & a.excitatory] += 15
        self.rate *= math.exp(-dt / 1000)
        self.rate += (1 - math.exp(-dt / 1000)) * spikes * 1000 / dt
        self.age_ms[self.active] += dt
        self.learn(arrivals, spikes, potential_arrivals)
        if c.plasticity and c.structural and (self.tick + 1) % round(c.structural_interval_ms / dt) == 0:
            self.rewire()
        # Schedule spikes only after all neurons and synapses have advanced.
        self.history[slot] = spikes
        self.queue[(self.tick + a.delay) % len(self.queue), self._edge_indices] = (
            spikes[a.pre] * self.active * self.weight)
        self.last_spikes = spikes
        self.spike_count += spikes
        self.tick += 1
        return spikes.copy()

    def learn(self, arrivals, spikes, potential_arrivals=None):
        """Local pair STDP, inhibitory homeostasis, and contact eligibility.

        Traces are evaluated before new events are inserted: simultaneous events
        alone do not produce an ordered STDP pair. Exposed for controlled probes.
        """
        c, a = self.config, self.anatomy
        dt = c.dt_ms
        decay = math.exp(-dt / 20)
        self.pre_trace *= decay
        self.post_trace *= decay
        self.potential_trace *= decay
        post = spikes[a.post]
        exc = self.excitatory_edge
        if c.plasticity:
            # Multiplicative soft bounds on E weights; LTD slightly dominates.
            delta_e = c.stdp_rate * (self.pre_trace * post * (4 - self.weight)
                                    - 1.05 * self.post_trace[a.post] * arrivals * self.weight)
            alpha = 2 * c.target_hz * .020
            delta_i = c.inhibitory_rate * (arrivals * (self.post_trace[a.post] - alpha)
                                           + self.pre_trace * post)
            # Vogels-style rule applies to inhibitory -> excitatory contacts.
            delta_i *= a.excitatory[a.post]
            self.weight += self.active * np.where(exc, delta_e, delta_i)
            scale = np.exp(np.clip((c.target_hz - self.rate[a.post]) / c.target_hz
                                   * dt / c.scaling_tau_ms, -.001, .001))
            self.weight[exc & self.active] *= scale[exc & self.active]
            np.clip(self.weight, 0.01, np.where(exc, 4., 12.), out=self.weight)
            self.weight[~self.active] = 0
        slow = math.exp(-dt / 1000)
        self.coactivity *= slow
        self.coactivity += (1 - slow) * self.potential_trace * post
        self.usage *= slow
        self.usage += (1 - slow) * arrivals * 1000 / dt
        self.pre_trace += arrivals
        self.potential_trace += arrivals if potential_arrivals is None else potential_arrivals
        self.post_trace += spikes

    def rewire(self):
        """Remove unused/weak contacts; grow correlated E contacts within budgets.

        Candidate axons are fixed by anatomy. Contact growth does not create
        neurons, new regions, or forbidden long-range inhibitory pathways.
        """
        c, a = self.config, self.anatomy
        if not (c.plasticity and c.structural):
            return
        remove = self.active & self.excitatory_edge & (self.age_ms >= c.structural_grace_ms)
        remove &= ((self.weight < .10) | (self.usage < .1)) & (self.coactivity < c.growth_threshold)
        self.active[remove] = False
        self.weight[remove] = self.pre_trace[remove] = self.age_ms[remove] = 0
        self.queue[:, remove] = 0
        self.pruned += int(remove.sum())
        fanin = np.bincount(a.post[self.active], minlength=a.size)
        fanout = np.bincount(a.pre[self.active], minlength=a.size)
        eligible = (~self.active & self.excitatory_edge & ~remove
                    & (self.coactivity >= c.growth_threshold)
                    & (self.rate[a.post] < c.target_hz * 1.5))
        # At most one new incoming contact per neuron and interval.
        added_to = set()
        for edge in np.flatnonzero(eligible)[np.argsort(-self.coactivity[eligible], kind="stable")]:
            pre, post = a.pre[edge], a.post[edge]
            if post in added_to or fanin[post] >= c.max_fanin or fanout[pre] >= c.max_fanout:
                continue
            self.active[edge], self.weight[edge] = True, .3
            self.age_ms[edge] = self.pre_trace[edge] = self.usage[edge] = 0
            self.queue[:, edge] = 0
            fanin[post] += 1
            fanout[pre] += 1
            added_to.add(post)
            self.created += 1

    def report(self, previous_counts=None, elapsed_ms=None):
        counts = self.spike_count if previous_counts is None else self.spike_count - previous_counts
        duration = self.time_ms if elapsed_ms is None else elapsed_ms
        rates = counts * 1000 / max(duration, self.config.dt_ms)
        a = self.anatomy
        populations = {}
        for r, name in enumerate(self.config.regions):
            for layer in range(1, 7):
                for e, label in ((True, "E"), (False, "I")):
                    indices = a.select(region=r, layer=layer, excitatory=e)
                    if len(indices):
                        populations[f"{name}/L{layer}/{label}"] = float(rates[indices].mean())
        return {"time_ms": self.time_ms, "neurons": a.size, "active_synapses": int(self.active.sum()),
                "potential_contacts": len(a.pre), "created": self.created, "pruned": self.pruned,
                "mean_hz": float(rates.mean()), "max_neuron_hz": float(rates.max()),
                "silent_fraction": float((counts == 0).mean()), "population_hz": populations,
                "region_hz": {name: float(rates[a.region == r].mean())
                              for r, name in enumerate(self.config.regions)},
                "excitatory_hz": float(rates[a.excitatory].mean()),
                "inhibitory_hz": float(rates[~a.excitatory].mean()),
                "mean_ge_ns": float(self.ge.mean()), "mean_gi_ns": float(self.gi.mean()),
                "finite": bool(all(np.isfinite(getattr(self, k)).all() for k in self.ARRAY_STATE))}

    def save(self, path):
        """Atomic, pickle-free checkpoint including RNG and all in-flight events."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": 1, "architecture": "laminar_conductance_lif", "config": asdict(self.config),
                "tick": self.tick, "created": self.created, "pruned": self.pruned,
                "rng": self.rng.bit_generator.state, "protocol": self.protocol}
        arrays = {name: getattr(self, name) for name in self.ARRAY_STATE}
        arrays.update({f"anatomy_{name}": value for name, value in vars(self.anatomy).items()})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                np.savez_compressed(handle, metadata=json.dumps(meta), **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("version") != 1 or meta.get("architecture") != "laminar_conductance_lif":
                raise ValueError("Unsupported biological checkpoint")
            network = cls(BiologyConfig(**meta["config"]))
            # Topology is recorded and checked. A changed connectivity generator
            # requires an explicit version migration, never silent replacement.
            for name, original in vars(network.anatomy).items():
                saved = data[f"anatomy_{name}"]
                if saved.shape != original.shape or saved.dtype != original.dtype or not np.array_equal(saved, original):
                    raise ValueError(f"Incompatible topology: {name}")
            for name in cls.ARRAY_STATE:
                saved, original = data[name], getattr(network, name)
                if saved.shape != original.shape or saved.dtype != original.dtype or not np.isfinite(saved).all():
                    raise ValueError(f"Invalid checkpoint state: {name}")
                setattr(network, name, saved.copy())
            for name in ("tick", "created", "pruned"):
                if type(meta[name]) is not int or meta[name] < 0:
                    raise ValueError(f"Invalid checkpoint counter: {name}")
                setattr(network, name, meta[name])
            network.rng.bit_generator.state = meta["rng"]
            network.protocol = meta.get("protocol", {})
            if not isinstance(network.protocol, dict):
                raise ValueError("Invalid checkpoint protocol")
            for name in ("adaptation", "ge", "gi", "refractory", "pre_trace", "post_trace",
                         "potential_trace", "coactivity", "usage", "rate", "age_ms", "queue", "spike_count"):
                if (getattr(network, name) < 0).any():
                    raise ValueError(f"Invalid negative checkpoint state: {name}")
            a, c = network.anatomy, network.config
            if ((network.weight < 0).any() or (network.weight > np.where(network.excitatory_edge, 4, 12)).any()
                    or (network.weight[~network.active] != 0).any()
                    or np.bincount(a.post[network.active], minlength=a.size).max() > c.max_fanin
                    or np.bincount(a.pre[network.active], minlength=a.size).max() > c.max_fanout):
                raise ValueError("Invalid checkpoint synapses")
            return network
