"""Connectome data model for the MaleCNS-based fly.

PROVENANCE SUMMARY (see docs/PROVENANCE.md):
  BIOLOGICAL : neuron identities, directed weighted connections, transmitter
               strings, hemilineage ids, ROIs -- all from Janelia neuPrint
               dataset male-cns:v1.0 (natverse/malecns default).
  ENGINEERED : the derived simulation graph (which subgraph is loaded, index
               remapping, weight->conductance scaling constants).

The raw download (data/raw/*.json) can be huge; the build step
(scripts/build_connectome.py) produces a compact npz under data/processed/.
This module loads either format transparently.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
RAW_DIR = os.path.join(DATA_ROOT, "raw")


@dataclass
class NeuronRecord:
    bodyid: int
    name: str = ""
    type: str = ""
    instance: str = ""
    transmitter: str = ""
    hemilineage: str = ""
    status: str = ""
    pre: int = 0
    posts: int = 0
    rois: list = field(default_factory=list)


@dataclass
class ConnectomeGraph:
    """Directed weighted graph of the simulated sub-network."""
    bodyids: np.ndarray            # (N,) int64 original neuPrint bodyIds
    n: int                         # number of neurons
    row: np.ndarray                # CSR-style: source neuron idx per connection
    col: np.ndarray                # target neuron idx per connection
    weight: np.ndarray             # synaptic site counts (>=1), float32
    sign: np.ndarray               # +1 excitatory / -1 inhibitory / 0 unknown
    names: list                    # display names (may be '')
    types: list                    # 'type' annotation (DNa01 etc.)
    transmitters: list
    hemilineages: list
    bodyid_to_idx: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ props
    @property
    def synapse_count(self) -> int:
        return len(self.row)

    def idx_of(self, bodyid: int) -> int:
        return self.bodyid_to_idx[int(bodyid)]

    def find(self, pattern: str) -> np.ndarray:
        """Indices of neurons whose name or type matches regex `pattern`."""
        import re
        rx = re.compile(pattern)
        return np.array([i for i in range(self.n)
                         if rx.search(self.names[i]) or rx.search(self.types[i])],
                        dtype=np.int64)

    # ------------------------------------------------------------------- io
    def save_npz(self, path: str):
        np.savez_compressed(
            path,
            bodyids=self.bodyids, row=self.row, col=self.col,
            weight=self.weight, sign=self.sign,
            names=np.array(self.names, dtype=object),
            types=np.array(self.types, dtype=object),
            transmitters=np.array(self.transmitters, dtype=object),
            hemilineages=np.array(self.hemilineages, dtype=object),
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load_npz(cls, path: str) -> "ConnectomeGraph":
        z = np.load(path, allow_pickle=True)
        g = cls(
            bodyids=z["bodyids"].astype(np.int64),
            n=int(len(z["bodyids"])),
            row=z["row"].astype(np.int32),
            col=z["col"].astype(np.int32),
            weight=z["weight"].astype(np.float32),
            sign=z["sign"].astype(np.float32),
            names=list(z["names"]),
            types=list(z["types"]),
            transmitters=list(z["transmitters"]),
            hemilineages=list(z["hemilineages"]),
            meta=json.loads(str(z["meta"])) if "meta" in z else {},
        )
        g.bodyid_to_idx = {int(b): i for i, b in enumerate(g.bodyids)}
        return g


# Transmitter polarity table.
# BIOLOGICAL basis: acetylcholine (ACh), glutamate (VGLUT+), ATP (P2X) and
# most neuropeptides act on ionotropic/metabotropic receptors that depolarise
# the Drosophila post-synaptic compartment in CNS circuits; GABA and the
# glycine-like inhibition in insect CNS are hyperpolarising (Rdl Cl- channel).
# ENGINEERED assumption: everything not confidently inhibitory is treated as
# excitatory with full strength; unknown transmitters get sign 0 (weak
# excitation by default, see build script) rather than being dropped.
EXCITATORY_TOKENS = {"ACh", "cholinergic", "glutamate", "Glut", "VGluT",
                     "ATP", "purinergic"}
INHIBITORY_TOKENS = {"GABA", "gabaergic", "glycine"}


def transmitter_sign(tx: str) -> int:
    """Map a neuPrint transmitter string to +1/-1/0. Unknown -> 0."""
    if not tx:
        return 0
    t = tx.strip()
    for tok in INHIBITORY_TOKENS:
        if tok.lower() in t.lower():
            return -1
    for tok in EXCITATORY_TOKENS:
        if tok.lower() == t.lower() or tok.lower() in t.lower():
            return 1
    return 0
