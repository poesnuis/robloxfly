"""Leaky integrate-and-fire simulation of a MaleCNS subgraph.

ENGINEERED (simulator assumptions, documented in docs/ENGINEERING_ASSUMPTIONS.md):
  * LIF membrane dynamics (MaleCNS is purely structural connectomics; it does
    NOT ship neuron conductances/dynamics — any dynamic model is an overlay).
  * Synaptic current for connection i->j at spike time t0:
        g_ij(t) = G_SCALE * w_ij * sign_i * exp(-(t-t0)/tau_syn)   for t>=t0
    implemented with one exponential conductance state per POST-synaptic
    neuron (sum of incoming channels), which is O(#synapses) memory and
    O(spikes) work per event -> sparse event-driven update.
  * sign 0 (unknown transmitter) treated as weak excitation (0.3x) rather
    than dropped — documented approximation.
BIOLOGICAL:
  * topology, direction, multiplicity (weight) and excitatory/inhibitory
    polarity come from male-cns:v1.0 via neuPrint.

Integration scheme: fixed-step dt with exact exponential decay between steps
(standard first-order synapse integration, e.g. Dayan & Abbott ch.1).
Neurons receive external drive currents I_ext (from sensory layer) plus the
synaptic conductance term  I_syn = g_exc*(E_exc - V) + g_inh*(E_inh - V).
"""
from __future__ import annotations

import numpy as np


class LIFSimulator:
    def __init__(self, row, col, weight, sign, params=None):
        self.row = np.asarray(row, dtype=np.int32)
        self.col = np.asarray(col, dtype=np.int32)
        self.weight = np.asarray(weight, dtype=np.float32)
        self.sign = np.asarray(sign, dtype=np.float32)
        self.n = int(max(self.col.max(initial=0), self.row.max(initial=0)) + 1)
        p = dict(params or {})
        # ---- ENGINEERED default parameters (mV, ms, nS, pA) -------------
        self.dt = float(p.get("dt_ms", 0.5))            # 0.5 ms step
        self.tau_m = float(p.get("tau_ms", 20.0))       # membrane tau ~ typical insect LN
        self.tau_syn_e = float(p.get("tau_syn_e", 5.0))
        self.tau_syn_i = float(p.get("tau_syn_i", 8.0))
        self.V_rest = float(p.get("V_rest", -60.0))
        self.V_thresh = float(p.get("V_thresh", -50.0))
        self.V_reset = float(p.get("V_reset", -65.0))
        self.refractory_ms = float(p.get("refractory_ms", 2.0))
        self.E_exc = float(p.get("E_exc", 0.0))
        self.E_inh = float(p.get("E_inh", -70.0))
        self.g_scale = float(p.get("g_scale", 0.4))     # nS per synaptic site
        self.unknown_gain = float(p.get("unknown_gain", 0.3))
        self.noise_std = float(p.get("noise_std", 0.5)) # pA background noise
        self.I_cap = float(p.get("I_cap", 200.0))       # clamp on ext drive

        # per-neuron effective signed gain multiplier
        sgn = self.sign.copy()
        mult = np.where(sgn > 0, 1.0, np.where(sgn < 0, 1.0, self.unknown_gain))
        self.g_eff = (self.weight * mult).astype(np.float32)
        self.is_inh = (sgn < 0).astype(np.float32)      # route to inhibitory channel

        # sort edges by target for fast segmented updates (optional path)
        order = np.argsort(self.col, kind="stable")
        self._col_sorted = self.col[order]
        self._row_sorted = self.row[order]
        self._g_sorted = self.g_eff[order]
        self._inh_sorted = self.is_inh[order]
        self._w_sorted = self.weight[order]
        bounds = np.searchsorted(self._col_sorted, np.arange(self.n + 1))
        self._targets_start = bounds[:-1]
        self._targets_end = bounds[1:]

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.V = np.full(self.n, self.V_rest, dtype=np.float32)
        self.g_e = np.zeros(self.n, dtype=np.float32)   # excitatory conductance nS
        self.g_i = np.zeros(self.n, dtype=np.float32)   # inhibitory conductance nS
        self.refrac = np.zeros(self.n, dtype=np.float32)
        self.spike_counts = np.zeros(self.n, dtype=np.int64)   # cumulative spikes
        self._spike_times = [[] for _ in range(self.n)]        # per-neuron recent spike times
        self.t = 0.0                                    # sim time in ms
        self.rng = rng
        self.total_spikes = 0

    def spike_window(self, window_ms: float) -> np.ndarray:
        """Number of spikes per neuron in the trailing window (int array)."""
        out = np.zeros(self.n, dtype=np.int64)
        cutoff = self.t - window_ms
        for i, lst in enumerate(self._spike_times):
            if lst:
                k = np.searchsorted(lst, cutoff)  # first index >= cutoff
                del lst[:k]
                out[i] = len(lst)
        return out

    # ------------------------------------------------------------- core step
    def step(self, I_ext=None, spikes_in=None):
        """Advance one dt. I_ext: (n,) pA external current. spikes_in: idx array.

        Returns indices of neurons that spiked this step."""
        dt = self.dt
        # decay conductances exactly (exponential)
        de = np.exp(-dt / self.tau_syn_e).astype(np.float32)
        di = np.exp(-dt / self.tau_syn_i).astype(np.float32)
        self.g_e *= de
        self.g_i *= di

        # deliver incoming spikes (event driven): each spiking pre-synaptic
        # neuron adds g_eff*g_scale to all its post-synaptic targets.
        if spikes_in is not None and len(spikes_in):
            rows = self._row_sorted
            sel = np.isin(rows, spikes_in)  # ENGINEERED: fine for small subgraphs
            # faster CSR-by-source approach used when graph large (see _by_src)
            for src in np.unique(spikes_in):
                start, end = self._src_range(int(src))
                if end > start:
                    tg = self._src_targets[start:end]
                    add = self._src_g[start:end] * self.g_scale
                    inh = self._src_inh[start:end]
                    np.add.at(self.g_e, tg, add * (1.0 - inh))
                    np.add.at(self.g_i, tg, add * inh)

        # membrane update (exponential Euler with conductance-based input)
        g_tot = self.g_e + self.g_i
        I = np.zeros(self.n, dtype=np.float32)
        if I_ext is not None:
            I += np.clip(I_ext, -self.I_cap, self.I_cap)
        if self.noise_std > 0:
            I += (self.rng.standard_normal(self.n) * self.noise_std).astype(np.float32)
        drive = I + self.g_e * (self.E_exc - self.V) + self.g_i * (self.E_inh - self.V)
        alpha = np.exp(-dt * (1.0 + self.tau_m * g_tot / 1000.0) / self.tau_m).astype(np.float32)
        # steady-state target approx; simple linear form keeps stability
        dV = (drive / (1.0 + g_tot / 1000.0)) * (1.0 - alpha) / max(dt, 1e-6) * self.tau_m * (dt / self.tau_m)
        self.V += dV

        # refractory handling
        self.refrac -= dt
        spiked = (self.V >= self.V_thresh) & (self.refrac <= 0)
        idx = np.flatnonzero(spiked)
        if len(idx):
            self.V[idx] = self.V_reset
            self.refrac[idx] = self.refractory_ms
            self.spike_counts[idx] += 1
            for i in idx:
                self._spike_times[int(i)].append(self.t)
        self.total_spikes += len(idx)
        self.t += dt
        return idx

    # ------------------------------------------------------ source-indexed CSR
    def _build_src_csr(self):
        order = np.argsort(self.row, kind="stable")
        self._src_rows = self.row[order]
        self._src_targets = self.col[order]
        self._src_g = self.g_eff[order]
        self._src_inh = self.is_inh[order]
        b = np.searchsorted(self._src_rows, np.arange(self.n + 1))
        self._src_start = b[:-1]
        self._src_end = b[1:]

    def _src_range(self, src: int):
        if not hasattr(self, "_src_start"):
            self._build_src_csr()
        return int(self._src_start[src]), int(self._src_end[src])

    # ------------------------------------------------------------ readout aid
    def firing_rate(self, window_ms=1000.0):
        """Spikes/s per neuron over trailing window (approx via last_spike)."""
        recent = self.last_spike > (self.t - window_ms)
        return recent.astype(np.float32) * (1000.0 / window_ms)
