"""Fly-like visual preprocessing of a captured screen frame.

PIPELINE (all stages ENGINEERED unless noted; see docs/ENGINEERING_ASSUMPTIONS.md):

  RGB frame (H,W,3 uint8)
    -> grayscale luminance, downsample to eye resolution (ENGINEERED:
       ~640x360 full-res -> 64x36 array; approximates coarse spatial
       resolution of the compound eye relative to a wide FOV; the fly's
       horizontal FOV is nearly panoramic, so we map the FULL screen width
       onto both eyes: left half -> left eye, right half -> right eye).
    -> ON/OFF transient channels per pixel (ENGINEERED but biologically
       motivated: photoreceptors R1-R6 signal luminance changes via
       depolarising/hyperpolarising responses; direction-selective T4/T5
       system computes motion in the lobula. We compute simple oriented
       temporal derivatives as a surrogate of elementary motion detectors.)
    -> hemifield aggregate signals used to drive sensory neuron populations:
         L_off, L_on, L_motion_leftward, L_motion_rightward (left eye), same
         for right eye, plus a central "looming" feature = d(area of dark
         blob)/dt near vertical meridian (surrogate of LC looming cells —
         BIOLOGICAL NOTE: Lobula Columnar neurons are known in MaleCNS data
         by 'LC' type names; if present in connectome they get direct drive).

Outputs a dict of scalar/low-dim features at each sampled frame; runtime
converts them into tonic+stochastic spike drive currents for named
connectome populations.
"""
from __future__ import annotations

import numpy as np


class FlyVisualField:
    def __init__(self, eye_cols=32, eye_rows=18, smooth=1.0, gain_on=1.0,
                 gain_off=1.0, gain_motion=1.0):
        self.eye_cols = eye_cols      # per hemi-eye columns
        self.eye_rows = eye_rows
        self.smooth = smooth          # spatial smoothing sigma (px)
        self.gain_on, self.gain_off, self.gain_motion = gain_on, gain_off, gain_motion
        self.prev_lum = None          # previous full-eye luminance (rows, 2*cols)
        self.prev_prev = None
        self.history = []             # ring of recent features for rate calc

    # ----------------------------------------------------------- helpers
    def _to_lum(self, rgb):
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        lum = lum.astype(np.float32) / 255.0
        # downsample by area averaging to eye grid (full width -> both eyes)
        H, W = lum.shape
        tw, th = self.eye_cols * 2, self.eye_rows
        # crop/pad-safe resize via block average
        ys = np.linspace(0, H, th + 1).astype(int)
        xs = np.linspace(0, W, tw + 1).astype(int)
        out = np.zeros((th, tw), dtype=np.float32)
        for i in range(th):
            for j in range(tw):
                blk = lum[ys[i]:max(ys[i + 1], ys[i] + 1),
                          xs[j]:max(xs[j + 1], xs[j] + 1)]
                out[i, j] = blk.mean() if blk.size else 0.0
        return out

    # ------------------------------------------------------------ main
    def process(self, rgb):
        """rgb: uint8 (H,W,3). Returns feature dict."""
        lum = self._to_lum(rgb)
        feats = {}
        rows, tw = lum.shape
        L = lum[:, : self.eye_cols]      # LEFT hemifield of screen -> left eye
        R = lum[:, self.eye_cols:]       # RIGHT -> right eye

        if self.prev_lum is not None:
            dL = lum - self.prev_lum     # OFF negative / ON positive
            # ON/OFF channel energies per hemifield (rectified transients)
            on_L = float(np.clip(dL[:, : self.eye_cols], 0, None).mean()) * self.gain_on
            off_L = float(np.clip(-dL[:, : self.eye_cols], 0, None).mean()) * self.gain_off
            on_R = float(np.clip(dL[:, self.eye_cols:], 0, None).mean()) * self.gain_on
            off_R = float(np.clip(-dL[:, self.eye_cols:], 0, None).mean()) * self.gain_off

            # Elementary-motion-detector surrogate: correlation of temporal
            # derivative with spatial gradient (Barlow-Levick style product)
            dx = np.gradient(lum, axis=1)
            E_right = float((dL * (-dx)).clip(0, None).mean())   # motion to the right
            E_left = float((dL * (dx)).clip(0, None).mean())     # motion to the left
            E_right_L = float((dL[:, : self.eye_cols] * (-dx[:, : self.eye_cols])).clip(0, None).mean())
            E_left_L = float((dL[:, : self.eye_cols] * dx[:, : self.eye_cols]).clip(0, None).mean())
            E_right_R = float((dL[:, self.eye_cols:] * (-dx[:, self.eye_cols:])).clip(0, None).mean())
            E_left_R = float((dL[:, self.eye_cols:] * dx[:, self.eye_cols:]).clip(0, None).mean())

            # Looming surrogate: growth of dark area in central columns
            dark = (lum < 0.35).astype(np.float32)
            c0 = tw // 2 - 4
            center_dark = dark[:, c0:c0 + 8].sum()
            if self.prev_center_dark is not None:
                loom = float(max(center_dark - self.prev_center_dark, 0.0))
            else:
                loom = 0.0
            self.prev_center_dark = center_dark
            feats.update(on_L=on_L, off_L=off_L, on_R=on_R, off_R=off_R,
                         mot_right_L=E_right_L * self.gain_motion,
                         mot_left_L=E_left_L * self.gain_motion,
                         mot_right_R=E_right_R * self.gain_motion,
                         mot_left_R=E_left_R * self.gain_motion,
                         global_mot_right=E_right, global_mot_left=E_left,
                         loom=loom,
                         mean_L=float(L.mean()), mean_R=float(R.mean()),
                         mean_all=float(lum.mean()))
        else:
            self.prev_center_dark = None
            z = 0.0
            feats.update(on_L=z, off_L=z, on_R=z, off_R=z, mot_right_L=z,
                         mot_left_L=z, mot_right_R=z, mot_left_R=z,
                         global_mot_right=z, global_mot_left=z, loom=z,
                         mean_L=float(L.mean()), mean_R=float(R.mean()),
                         mean_all=float(lum.mean()))

        self.prev_lum = lum
        # EMA smoothing of features so downstream drive is continuous
        if not hasattr(self, "_ema"):
            self._ema = {k: v for k, v in feats.items()}
        for k in feats:
            self._ema[k] = 0.7 * self._ema.get(k, feats[k]) + 0.3 * feats[k]
        feats_ema = dict(self._ema)
        feats_ema["raw"] = feats
        feats_ema["lum_grid"] = lum
        return feats_ema

    def reset(self):
        self.prev_lum = None
        self.prev_prev = None
        self.prev_center_dark = None
        self._ema = None
