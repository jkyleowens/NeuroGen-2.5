"""Downscaled laminar geometry, not a reconstruction of a human connectome."""

from dataclasses import dataclass

import numpy as np


# Layer, excitatory cells, inhibitory cells. These counts are engineering choices.
LAYERS = ((1, 0, 2), (2, 8, 2), (3, 8, 2), (4, 12, 3), (5, 12, 3), (6, 10, 2))


@dataclass
class Anatomy:
    region: np.ndarray
    column: np.ndarray
    layer: np.ndarray
    excitatory: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    delay: np.ndarray
    pathway: np.ndarray  # 0 local, 1 horizontal, 2 feedforward, 3 feedback

    @property
    def size(self):
        return len(self.layer)

    def select(self, region=None, column=None, layer=None, excitatory=None):
        mask = np.ones(self.size, dtype=bool)
        for values, selected in ((self.region, region), (self.column, column),
                                 (self.layer, layer), (self.excitatory, excitatory)):
            if selected is not None:
                mask &= values == selected
        return np.flatnonzero(mask)


def build_anatomy(config, rng):
    cells = [(r, c, layer, e) for r in range(len(config.regions))
             for c in range(config.columns_per_region)
             for layer, ne, ni in LAYERS for e in [True] * ne + [False] * ni]
    region, column, layer, excitatory = map(np.array, zip(*cells))
    excitatory = excitatory.astype(bool)
    edges = []
    # Probabilities of potential contacts; active contacts are a subset. L4 ->
    # superficial -> deep and L6 -> L4 are favored, with recurrent connections.
    favored = {(4, 2), (4, 3), (2, 3), (3, 5), (5, 6), (6, 4)}
    paths = {(src, dst): kind for src, dst, kind in config.projections}
    for pre in range(len(cells)):
        for post in range(len(cells)):
            if pre == post:
                continue
            same_region = region[pre] == region[post]
            distance = abs(int(column[pre]) - int(column[post]))
            if same_region and distance == 0:
                kind = 0
                if not excitatory[pre]:
                    probability = .65 if abs(layer[pre] - layer[post]) <= 1 else .08
                else:
                    probability = .60 if (layer[pre], layer[post]) in favored else .16
                    if layer[pre] == layer[post]:
                        probability = .4
                delay_ms = 1
            elif same_region and distance == 1 and excitatory[pre]:
                kind, probability, delay_ms = 1, .09, 2
            elif not same_region and excitatory[pre] and distance <= 1:
                connection = paths.get((config.regions[region[pre]], config.regions[region[post]]))
                if connection == "feedforward" and layer[pre] in (2, 3) and layer[post] == 4:
                    kind, probability, delay_ms = 2, .55, 4 + distance
                elif connection == "feedback" and layer[pre] in (5, 6) and layer[post] in (1, 2, 6):
                    kind, probability, delay_ms = 3, .15, 6 + distance
                else:
                    continue
            else:
                continue
            if rng.random() < probability:
                edges.append((pre, post, round(delay_ms / config.dt_ms), kind))
    if not edges:
        raise ValueError("Topology has no potential contacts")
    pre, post, delay, pathway = np.array(edges, dtype=np.int64).T
    return Anatomy(region, column, layer, excitatory, pre, post, delay, pathway)
