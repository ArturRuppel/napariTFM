"""Backend-equivalence sweep: PIV disp_device='cpu' (openpiv) vs 'cuda' (torch).

Reuses the sweep harness generators (make_stacks PSF+camera noise, make_scenes
GT-traction->GT-displacement->warped pair) and scorer (scoring.metrics) to
ask: for one parameter set, do the two backends have the same signal/noise
recovery characteristics? Signal axis = peak displacement; noise axis = the SNR
scenarios. Scored at the DISPLACEMENT level (the direct backend output), plus a
direct CPU-vs-GPU field diff. Writes results/device_compare.csv + a PNG.
"""
from __future__ import annotations
import sys, time, csv, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_stacks as MS, make_scenes as SC, scoring as SF, sweep_config as C
from napariTFM.backend.piv_displacement import PIVDisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters

N = C.CROP_SIZE
FOOT = C.FOOTPRINTS_UM[2]           # fixed mid footprint
DISPS = C.PEAK_DISP_PX             # signal ladder
SCEN = [1, 5, 3, 6]                # SNR ladder (spans NA/density/expo); ncc reported
JIT = 1                            # mild registration jitter frame
DS = 4                             # downscale (pipeline convention); common to both devices
EFF_PS = C.PIXEL_SIZE_UM * DS


def block_mean(a, f=DS):
    h = a.shape[0] // f
    return a[:h * f, :h * f].reshape(h, f, h, f).mean((1, 3))


def make_pair(si, dpx):
    nb, NA, expo = MS.SCENARIOS[si]
    xs, ys, d = MS.scene(nb, seed=si)
    f0 = MS.camera(MS.signal(xs, ys, d, NA, expo), seed=5000 * si + 0)
    jr = np.random.default_rng(1000 * si + JIT); jit = MS.JITTERS[JIT]
    xj, yj = xs + jr.normal(0, jit, nb), ys + jr.normal(0, jit, nb)
    fk = MS.camera(MS.signal(xj, yj, d, NA, expo), seed=5000 * si + JIT)
    ut, _ = SF.rasterize_gt(SC.scene_dict("x", "unit", FOOT, 1.0), N)
    unit_peak = float(np.hypot(*SC.greens_displacement(ut, N)).max())
    mag = (dpx * C.PIXEL_SIZE_UM) / unit_peak
    gt_tr, _ = SF.rasterize_gt(SC.scene_dict("x", "x", FOOT, mag, dpx), N)
    u = SC.greens_displacement(gt_tr, N)                      # (2,N,N) µm
    ref = f0.astype(np.float64)
    deformed = SC.warp(fk.astype(np.float64), u, C.PIXEL_SIZE_UM)
    ncc = MS.ncc(f0, fk)
    return ref, deformed, u, ncc


def run_backend(refd, defd, dev):
    p = DisplacementParameters(disp_method="PIV", disp_device=dev)   # DEFAULT knobs
    an = PIVDisplacementAnalyzer(p)
    flow = an.calculate_flow(refd, defd)                     # (h,w,2) px
    return np.stack([flow[..., 0], flow[..., 1]]) * EFF_PS   # (2,h,w) µm


def main():
    rows = []
    for si in SCEN:
        for dpx in DISPS:
            ref, deformed, u, ncc = make_pair(si, dpx)
            refd, defd = block_mean(ref), block_mean(deformed)
            gtd = np.stack([block_mean(u[0]), block_mean(u[1])])     # µm on downscaled grid
            rec = {}
            for dev in ("cpu", "cuda"):
                rec[dev] = run_backend(refd, defd, dev)
            # direct CPU<->GPU agreement (how much the backend choice alone moves it)
            dnum = np.sqrt(((rec["cpu"] - rec["cuda"]) ** 2).sum()) / (np.sqrt((rec["cuda"] ** 2).sum()) or 1.0)
            for dev in ("cpu", "cuda"):
                m = SF.metrics(rec[dev], gtd, do_sabass=False)
                rows.append(dict(scenario=si, ncc=round(ncc, 4), peak_disp_px=dpx, device=dev,
                                 nrmse=round(m["nrmse"], 4), corr=round(m["corr"], 4),
                                 ang_field=round(m["ang_field"], 3), mag_bias=round(m["mag_bias"], 4),
                                 cpu_vs_gpu_nrmse=round(dnum, 4)))
            print(f"s{si}(ncc {ncc:.3f}) u={dpx:>6.3f}px | "
                  f"cpu nrmse {rows[-2]['nrmse']:.3f} corr {rows[-2]['corr']:.3f} | "
                  f"gpu nrmse {rows[-1]['nrmse']:.3f} corr {rows[-1]['corr']:.3f} | "
                  f"cpu<->gpu {dnum:.3f}", flush=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    csvp = os.path.join(os.path.dirname(__file__), "results", "device_compare.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # --- characteristic curves: nrmse & corr vs signal (peak disp), per scenario ---
    fig, axes = plt.subplots(2, len(SCEN), figsize=(4 * len(SCEN), 7), sharex=True)
    for j, si in enumerate(SCEN):
        sr = [r for r in rows if r["scenario"] == si]
        ncc = sr[0]["ncc"]
        for dev, style in (("cpu", "o-"), ("cuda", "s--")):
            dr = [r for r in sr if r["device"] == dev]
            x = [r["peak_disp_px"] for r in dr]
            axes[0, j].plot(x, [r["nrmse"] for r in dr], style, label=f"{dev}")
            axes[1, j].plot(x, [r["corr"] for r in dr], style, label=f"{dev}")
        axes[0, j].set_title(f"scenario {si}  (ncc={ncc:.3f})")
        axes[0, j].set_xscale("log"); axes[1, j].set_xscale("log")
        axes[0, j].set_ylim(0, 1.3); axes[1, j].set_ylim(0, 1.02)
        axes[1, j].set_xlabel("peak displacement (px)  [signal]")
        for i in (0, 1):
            axes[i, j].grid(alpha=.3); axes[i, j].legend(fontsize=8)
    axes[0, 0].set_ylabel("displacement nRMSE  (lower=better)")
    axes[1, 0].set_ylabel("magnitude corr  (higher=better)")
    fig.suptitle("PIV backend equivalence: openpiv (cpu) vs torch (cuda), default knobs "
                 f"(window={DisplacementParameters().piv_window}, passes={DisplacementParameters().piv_passes})")
    fig.tight_layout()
    pngp = os.path.join(os.path.dirname(__file__), "results", "device_compare.png")
    fig.savefig(pngp, dpi=110)
    print("\nwrote", csvp, "and", pngp)


if __name__ == "__main__":
    main()
