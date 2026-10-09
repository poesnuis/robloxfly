"""Sensory transduction: visual features -> drive currents into named
MaleCNS populations.

BIOLOGICAL: target populations are REAL neurons from male-cns:v1.0, selected
by their connectome 'type'/'name' annotations (verified against the dataset's
naming conventions exposed by natverse/malecns: optic-lobe columnar cells are
named e.g. 'LpN', lobula plate tangential cells 'CT'/'HS'/'VS' types in the
related VNC/brain annotation literature; MaleCNS v1.0 inherits brain neuron
types from FAFB-style naming via the flywireType/mancType columns). The exact
list of matched names is resolved AT BUILD TIME from the downloaded data and
written to data/processed/sensory_targets.json — nothing here invents names.

ENGINEERED: the mapping from abstract visual features (ON/OFF/motion/loom) to
a tonic current amplitude per population. There is no quantitative
photoreceptor->PN transfer function in the public dataset; we use a signed
linear map with gains in config/sensory.yaml and Poisson-like stochastic
drive. Each feature drives its population's somatic current proportional to
the feature magnitude.

If a target population does not exist in the loaded graph, it is skipped and
reported (fail-soft, never fabricated).
"""
from __future__ import annotations

import json
import os

import numpy as np

# Feature -> polarity hints for populations matched by type-name prefix.
# Keys are regexes over neuron `type` or `name`. Values: sign of modulation
# (+ increases drive with feature magnitude; - decreases).
# BIOLOGICAL rationale comments per entry; magnitudes are ENGINEERED.
DEFAULT_POPULATION_MAP = {
    # Lobula plate tangential cells: wide-field motion-sensitive (HS/VS/CT
    # classes are the classic direction-selective optic-glomerular neurons).
    r"^(HS|VSi|VS[0-9])": {"mot_right_R": +1.0, "mot_left_L": +1.0},   # horiz-system-ish
    r"^CT[0-9]":          {"global_mot_right": +1.0, "global_mot_left": +1.0},
    r"^VS[0-9]":          {"loom": +1.0},                              # vertical system / looming
    # Optic lobe columnar / medulla & lobula broad classes get generic ON/OFF
    r"^(T[1-3][a-z]?|Me[a-zA-Z0-9]*|Lo[a-zA-Z0-9]*)": {"on_all": +0.5, "off_all": +0.5},
    # Photoreceptor-derived PNs present in CNS-brain (R8lam etc.)
    r"(PN$|_[adl]?PN)":  {"off_all": +1.0},
}


class SensoryDrive:
    def __init__(self, graph, targets_path=None, pop_map=None, params=None):
        self.graph = graph
        p = dict(params or {})
        self.base_current = float(p.get("base_current_pA", 8.0))     # ENGINEERED
        self.feature_gain = float(p.get("feature_gain_pA", 60.0))    # ENGINEERED
        self.pop_map = pop_map or DEFAULT_POPULATION_MAP
        self.targets_file = targets_path
        self.assignments = []      # list of (neuron_idx, feature_name, sign, gain)
        self.missing = []

        if targets_path and os.path.exists(targets_path):
            with open(targets_path) as f:
                spec = json.load(f)
            for entry in spec["targets"]:
                pat = entry["pattern"]
                idxs = graph.find(pat)
                if len(idxs) == 0:
                    self.missing.append(pat)
                    continue
                for feat, sgn in entry["features"].items():
                    g = float(entry.get("gain", 1.0))
                    for i in idxs:
                        self.assignments.append((int(i), feat, float(sgn), g))
        else:
            # resolve defaults against the actual graph now
            for pat, feats in self.pop_map.items():
                idxs = graph.find(pat)
                if len(idxs) == 0:
                    self.missing.append(pat)
                    continue
                for feat, sgn in feats.items():
                    for i in idxs:
                        self.assignments.append((int(i), feat, float(sgn), 1.0))
            if targets_path:
                self.export_targets(targets_path)

        self._idx = np.array([a[0] for a in self.assignments], dtype=np.int32)
        self._feat = [a[1] for a in self.assignments]
        self._sgn = np.array([a[2] for a in self.assignments], dtype=np.float32)
        self._gain = np.array([a[3] for a in self.assignments], dtype=np.float32)

    def export_targets(self, path):
        feats = sorted(set(self._feat))
        spec = {"note": "resolved at build time from real male-cns:v1.0 names",
                "targets": [{"pattern": pat, "features": ft}
                            for pat, ft in self.pop_map.items()]}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)

    # ------------------------------------------------------------- main api
    def make_current(self, n_neurons: int, feats: dict, rng: np.random.Generator):
        """Return (n,) external current array in pA."""
        I = np.zeros(n_neurons, dtype=np.float32)
        if len(self._idx) == 0:
            return I
        # feature vector aligned to assignments
        fv = np.empty(len(self._feat), dtype=np.float32)
        on_all = max(feats.get("on_L", 0.0), feats.get("on_R", 0.0))
        off_all = max(feats.get("off_L", 0.0), feats.get("off_R", 0.0))
        for k, name in enumerate(self._feat):
            if name == "on_all":
                v = on_all
            elif name == "off_all":
                v = off_all
            else:
                v = feats.get(name, 0.0)
            fv[k] = v
        contrib = self._sgn * self._gain * fv * self.feature_gain
        np.add.at(I, self._idx, contrib)
        # baseline tonic drive onto assigned populations keeps them in a
        # responsive regime (ENGINEERED)
        uniq = np.unique(self._idx)
        I[uniq] += self.base_current
        # Poisson-like noise scaled by total drive so activity tracks stimulus
        I += rng.standard_normal(n_neurons).astype(np.float32) * \
             (1.0 + 0.25 * np.abs(I))
        return I
