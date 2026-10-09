"""Build the simulation graph from raw neuPrint JSON downloads.

Inputs : data/raw/neurons.json, data/raw/connections.json  (BIOLOGICAL)
Outputs: data/processed/malecns_v1.npz (full graph of all typed neurons that
         appear in the Connection table) plus summary stats json.

Subgraph selection for real-time simulation happens at RUNTIME
(src/runtime/subgraph.py), not here — we keep the full connectome on disk so
reduction is always an explicit, documented, reversible choice.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.connectome.model import (PROCESSED_DIR, RAW_DIR, ConnectomeGraph,
                                  transmitter_sign)


def build(raw_dir=RAW_DIR, out_path=None):
    with open(os.path.join(raw_dir, "neurons.json")) as f:
        neurons = json.load(f)
    with open(os.path.join(raw_dir, "connections.json")) as f:
        conns = json.load(f)

    # Keep neurons that have at least one connection endpoint OR are named
    # visual/PN/DN populations (they may be drive sources with no recorded
    # inputs yet).
    endpoints = set()
    for c in conns:
        endpoints.add(int(c["pre"]))
        endpoints.add(int(c["post"]))

    keep_ids = []
    rec_by_id = {}
    for nr in neurons:
        bid = int(nr["bodyId"])
        rec_by_id[bid] = nr
        if bid in endpoints:
            keep_ids.append(bid)
    keep_ids = sorted(set(keep_ids))
    idx = {b: i for i, b in enumerate(keep_ids)}

    rows, cols, wts, sgn = [], [], [], []
    dropped = 0
    for c in conns:
        p, q = int(c["pre"]), int(c["post"])
        if p not in idx or q not in idx:
            dropped += 1
            continue
        w = float(c.get("weight") or 1)
        rows.append(idx[p])
        cols.append(idx[q])
        wts.append(w)
        tx = (rec_by_id.get(p, {}) or {}).get("transmitter", "") or ""
        sgn.append(transmitter_sign(tx))

    bodyids = np.array(keep_ids, dtype=np.int64)
    g = ConnectomeGraph(
        bodyids=bodyids, n=len(bodyids),
        row=np.array(rows, dtype=np.int32),
        col=np.array(cols, dtype=np.int32),
        weight=np.array(wts, dtype=np.float32),
        sign=np.array(sgn, dtype=np.float32),
        names=[str(rec_by_id.get(b, {}).get("name") or "") for b in keep_ids],
        types=[str(rec_by_id.get(b, {}).get("type") or "") for b in keep_ids],
        transmitters=[str(rec_by_id.get(b, {}).get("transmitter") or "") for b in keep_ids],
        hemilineages=[str(rec_by_id.get(b, {}).get("hemilineage") or "") for b in keep_ids],
        meta={"dataset": "male-cns:v1.0", "server": "https://neuprint.janelia.org",
              "dropped_conns_missing_node": dropped},
    )
    g.bodyid_to_idx = idx

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = out_path or os.path.join(PROCESSED_DIR, "malecns_v1.npz")
    g.save_npz(out_path)

    stats = {
        "neurons": g.n,
        "connections": g.synapse_count,
        "excitatory": int((g.sign > 0).sum()),
        "inhibitory": int((g.sign < 0).sum()),
        "unknown_tx": int((g.sign == 0).sum()),
        "named_neurons": int(sum(1 for nm in g.names if nm)),
        "typed_neurons": int(sum(1 for t in g.types if t)),
        "dn_named": int(sum(1 for nm in g.names if nm.startswith("DN"))),
        "source_dataset": "male-cns:v1.0 @ neuprint.janelia.org",
    }
    with open(os.path.join(PROCESSED_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("wrote", out_path)
    return g


if __name__ == "__main__":
    build()
