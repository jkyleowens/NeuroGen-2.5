"""A supervised observer of spikes; it never changes the biological network."""

import numpy as np


def cue_column(seed, trial, columns):
    # Stateless protocol RNG: resume and observation cannot perturb neural noise.
    value = np.random.SeedSequence([seed, trial, 9137]).generate_state(1)[0]
    return int(value) % columns


class CueProbe:
    """Predict a random sensory column from later association E spikes.

    Each 60-trial block has 40 online training trials and 20 evaluation trials.
    Predict and log cross-entropy BEFORE updating. Evaluation labels never update
    the readout. The biological core continues its local, label-free plasticity.
    """

    def __init__(self, network, saved=None):
        if len(network.config.regions) < 2 or network.config.columns_per_region < 2:
            raise ValueError("The cue task needs at least two regions and two columns per region")
        self.indices = network.anatomy.select(region=1, excitatory=True)
        self.classes = network.config.columns_per_region
        self.weights = np.zeros((self.classes, len(self.indices)))
        self.bias = np.zeros(self.classes)
        self.counts = np.zeros(len(self.indices))
        self.trial = -1
        self.valid = False
        self.completed = False
        self.stats = {phase: {"n": 0, "correct": 0, "loss_sum": 0.} for phase in ("train", "eval")}
        if saved is not None:
            if saved.get("version") != 1 or saved.get("classes") != self.classes:
                raise ValueError("Incompatible probe checkpoint")
            for name in ("weights", "bias", "counts"):
                value = np.asarray(saved[name], dtype=float)
                if value.shape != getattr(self, name).shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid probe checkpoint: {name}")
                setattr(self, name, value)
            self.trial, self.valid, self.completed = saved["trial"], saved["valid"], saved["completed"]
            self.stats = saved["stats"]

    def state(self):
        return {"version": 1, "classes": self.classes, "weights": self.weights.tolist(),
                "bias": self.bias.tolist(), "counts": self.counts.tolist(), "trial": self.trial,
                "valid": self.valid, "completed": self.completed, "stats": self.stats}

    def observe(self, network, spikes):
        beginning = (network.tick - 1) * network.config.dt_ms
        trial, within = int(beginning // 250), beginning % 250
        if trial != self.trial:
            self.trial, self.valid, self.completed = trial, within == 0, False
            self.counts.fill(0)
        if not self.valid or self.completed:
            return None
        if 50 <= within < 150:
            self.counts += spikes[self.indices]
        if within + network.config.dt_ms != 150:
            return None
        self.completed = True
        features = self.counts / 5.  # fixed scale, no fitted test statistics
        logits = self.weights @ features + self.bias
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        label = cue_column(network.config.seed, trial, self.classes)
        prediction = int(np.argmax(probabilities))
        loss = float(-shifted[label] + np.log(np.exp(shifted).sum()))
        phase = "train" if trial % 60 < 40 else "eval"
        stats = self.stats[phase]
        stats["n"] += 1
        stats["correct"] += int(prediction == label)
        stats["loss_sum"] += loss
        result = {"time_ms": network.time_ms, "trial": trial, "phase": phase, "label": label,
                  "prediction": prediction, "correct": int(prediction == label), "loss": loss,
                  "accuracy": stats["correct"] / stats["n"], "mean_loss": stats["loss_sum"] / stats["n"],
                  "chance_accuracy": 1 / self.classes}
        if phase == "train":
            error = probabilities.copy()
            error[label] -= 1
            self.weights -= .1 * (np.outer(error, features) + .001 * self.weights)
            self.bias -= .1 * error
        return result

    def summary(self):
        return {f"probe_{phase}_{metric}": value for phase, stats in self.stats.items()
                for metric, value in (("trials", stats["n"]),
                                      ("accuracy", stats["correct"] / stats["n"] if stats["n"] else None),
                                      ("loss", stats["loss_sum"] / stats["n"] if stats["n"] else None))}
