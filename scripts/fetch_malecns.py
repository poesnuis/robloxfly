#!/usr/bin/env python3
"""Download the Janelia FlyEM MaleCNS v1.0 connectome from neuPrint.

Data source (verified 2026-10-09):
  server : https://neuprint.janelia.org   (from natverse/malecns R/urls.R line ~90)
  dataset: male-cns:v1.0                  (default in natverse/malecns choose_mcns_dataset)

neuPrint now requires an API token for ALL data access, including public
datasets (the /api/dataset endpoint returns that message explicitly).
Get a free token at https://neuprint.janelia.org/account and export it as
NEUPRINT_TOKEN (or put it in .env / %APPDATA%/malecns_fly/token.txt).
NEVER commit the token.

What this script pulls (all BIOLOGICAL data straight from the connectome DB):
  neurons.json     -> per-neuron metadata: name, type, instance, transmitter,
                      hemilineage, subhemilineage, status, pre/post counts, rois
  connections.json -> full Connection graph: pre bodyId, post bodyId, weight
                      (number of synaptic sites), pre_roi, post_roi

Outputs raw JSON under data/raw/ then let src/connectome/build.py turn them
into the compact simulation matrices under data/processed/.
"""
import argparse
import json
import os
import sys
import time

import requests

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"

GRAPHQL = f"{SERVER}/graphql?dataset={DATASET.replace(':', '%3A')}"


def get_token(cli_token=None):
    if cli_token:
        return cli_token
    tok = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if tok:
        return tok
    # Windows-friendly local file locations (gitignored)
    for cand in [
        os.path.join(os.environ.get("APPDATA", ""), "malecns_fly", "token.txt"),
        os.path.expanduser("~/.config/malecns_fly/token.txt"),
        os.path.join(os.path.dirname(__file__), "..", ".neuprint_token"),
    ]:
        if cand and os.path.isfile(cand):
            with open(cand) as f:
                return f.read().strip()
    return None


def gql(query, token, retries=5):
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(retries):
        try:
            r = requests.post(GRAPHQL, json={"query": query}, headers=headers, timeout=120)
            if r.status_code == 401:
                sys.exit("ERROR: neuPrint rejected the token (401). "
                         "Get a fresh one at https://neuprint.janelia.org/account")
            r.raise_for_status()
            j = r.json()
            if "errors" in j:
                raise RuntimeError(j["errors"])
            return j["data"]
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2 ** attempt
            print(f"  retry after error ({e.__class__.__name__}), waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("too many retries talking to neuPrint")


def fetch_meta(token):
    """All PrimaryNeuron metadata, paginated by bodyId."""
    fields = ("bodyId name type instance labels transmitter hemilineage "
              "subhemilineage status primaryNeuron pre posts inputCount "
              "outputCount skeletonSize roiTagList").split()
    q = ("{ Neuron(where: {primaryNeon: {_eq: true}}, first: 500, "
         "offset: %d, order_by: {bodyId: asc}) { " + " ".join(fields) + " } }")
    out, off = [], 0
    while True:
        data = gql(q % off, token)
        batch = data["Neuron"]
        out.extend(batch)
        print(f"  meta: {len(out)} neurons", flush=True)
        if len(batch) < 500:
            break
        off += 500
    return out


def fetch_connections(token):
    """Full directed connection table via pagination on the Connection type."""
    q = ("{ Connection(first: 1000, offset: %d, order_by: {pre: {bodyId: asc} "
         "post: {bodyId: asc}}) { pre { bodyId } post { bodyId } weight "
         "pre_roi_name post_roi_name confidence } }")
    out, off = [], 0
    while True:
        data = gql(q % off, token)
        batch = data["Connection"]
        for c in batch:
            out.append({
                "pre": c["pre"]["bodyId"],
                "post": c["post"]["bodyId"],
                "weight": c["weight"],
                "pre_roi": c.get("pre_roi_name"),
                "post_roi": c.get("post_roi_name"),
                "confidence": c.get("confidence"),
            })
        print(f"  connections: {len(out)}", flush=True)
        if len(batch) < 1000:
            break
        off += 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="neuPrint token (else env/file)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    args = ap.parse_args()

    tok = get_token(args.token)
    if not tok:
        sys.exit(
            "ERROR: no neuPrint token found.\n"
            "  1. Log in at https://neuprint.janelia.org (free account)\n"
            "  2. Open https://neuprint.janelia.org/account and copy your token\n"
            "  3. set NEUPRINT_TOKEN environment variable, OR save it to\n"
            "     ~/.config/malecns_fly/token.txt (Windows: %%APPDATA%%\\malecns_fly\\token.txt)\n"
            "The token must NOT be pasted into source code or committed.")
    os.makedirs(args.out, exist_ok=True)

    print("[1/3] dataset info ...", flush=True)
    info = gql("{ Dataset { name description neuronCount connectionCount } }", tok)["Dataset"]
    print(json.dumps(info, indent=2))

    print("[2/3] fetching neuron metadata ...", flush=True)
    meta = fetch_meta(tok)
    with open(os.path.join(args.out, "neurons.json"), "w") as f:
        json.dump(meta, f)

    print("[3/3] fetching connections ...", flush=True)
    conns = fetch_connections(tok)
    with open(os.path.join(args.out, "connections.json"), "w") as f:
        json.dump(conns, f)

    print(f"DONE: {len(meta)} neurons, {len(conns)} connections -> {args.out}")
    print("Next: python scripts/build_connectome.py")


if __name__ == "__main__":
    main()
