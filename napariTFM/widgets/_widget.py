import json
import logging
import math
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime
from typing import Any

# How many experiments' decoded stage-array bundles to keep resident for
# instant re-display. Bundles hold full displacement/force/stress arrays, so
# this trades memory for click latency; small on purpose.
_STAGE_ARRAY_CACHE_SIZE = 4

import napari
from qtpy.QtCore import Qt, QObject, QTimer, QSize, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QSizePolicy, QDoubleSpinBox,
    QHBoxLayout, QFrame, QSpinBox, QComboBox, QFileDialog, QCheckBox,
    QMenu, QToolButton, QApplication, QPushButton
)

from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager

from napariTFM.widgets.displacement_analysis_widget import DisplacementAnalysisWidget
from napariTFM.widgets.fttc_widget import FTTCWidget
from napariTFM.widgets.stress_widget import StressWidget
from napariTFM.widgets._stage_data_status import DataArtifactSpec, compute_stage_status
from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._stage_dependencies import (
    InteractiveStageCoordinator,
    StaleChoice,
)
from napariTFM.widgets._ui_style import title_style, stage_accent, muted_accent, theme_names, active_theme_name, set_active_theme, section_grid, add_section_header, add_section_pair_row, add_section_labeled_full_row, section_label_style, section_subheader_style, caption_style, TIGHT_SPACING
from napariTFM.widgets._icons import stage_action_icon
from napariTFM.widgets._param_controls import dslider, islider, rslider
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider, QLabeledSlider
from napariTFM.widgets._experiments_list import ExperimentsList, PIPELINE_STAGES
from napariTFM.widgets._run_config import (
    build_run_config,
    build_series_config,
    series_records,
)
from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.backend._torch_warmup import warm_up_torch
from napariTFM.backend.ntfm_writer import write_experiment_ntfm
from napariTFM.utilities.batch_output import experiment_ntfm_path, resolve_output_plan

logger = logging.getLogger(__name__)

PROJECT_FORMAT_VERSION = 2


# Per-stage input/output artifact specs, used only to derive each stage's
# in-memory status (compute_stage_status) for the spine node when no experiment
# is selected. No on_view/on_action callbacks: the old per-artifact status-dot
# row was removed as redundant with the colormap-spine rail.
STAGE_DATA_ARTIFACTS = {
    "displacement": [
        DataArtifactSpec("reference", "Reference", "reference", "input"),
        DataArtifactSpec("bead_stack", "Beads", "bead_stack", "input"),
        DataArtifactSpec("cell_stack", "Cells", "cell_stack", "input", required=False),
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "output"),
    ],
    "force": [
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "input"),
        DataArtifactSpec("force_results", "Traction map", "force_results", "output"),
    ],
    "stress": [
        DataArtifactSpec("force_results", "Traction map", "force_results", "input"),
        DataArtifactSpec("mask_stack", "Mask stack", "mask_stack", "input"),
        DataArtifactSpec("stress_results", "Stress map", "stress_results", "output"),
    ],
}


def _derived_pyramid_levels(shape, downscale, min_size):
    """Pyramid depth FFD builds for ``shape`` (h, w) at ``downscale``/``min_size``.

    A torch-free duplicate of ``backend._flow_common.pyramid_num_levels`` (which lives
    in a module that imports torch at load); the panel needs this to show the derived
    level count without dragging torch onto the widget import path. Keep the two in
    sync -- both mirror ``_pyramid``'s size gate."""
    h, w = int(shape[0]), int(shape[1])
    n = 1
    while min(h, w) > downscale * min_size:
        h = math.ceil(h / downscale)
        w = math.ceil(w / downscale)
        n += 1
    return n


# Sentinel marking a sub-group heading inside a section's spec list. A spec of
# the form (GROUP, "Advanced") renders a muted sub-header and starts a fresh
# two-per-row run; it is not a parameter control.
GROUP = object()

# Sentinel marking a displacement-method block. A spec of the form
# (METHOD, "FFD", [sub-specs]) gathers that method's whole knob group -- its GROUP
# sub-headers, controls, and any ADVANCED disclosure -- into a container that is
# shown only while that method is the selected one, so an unselected method's
# parameters are hidden entirely (not merely greyed). It is not a parameter control.
METHOD = object()

# Sentinel marking a collapsible "Advanced" disclosure inside a section. A spec of
# the form (ADVANCED, "Advanced", method) renders a collapsed-by-default toggle and
# gathers the specs that follow (up to the next GROUP/ADVANCED marker, or the section
# end) into a container the toggle shows/hides. The block is owned by ``method``: its
# rows exist/show only while that displacement method is selected AND the disclosure
# is expanded. It is not a parameter control.
ADVANCED = object()


class WorkflowParameterPanel(QWidget):
    """Single visible parameter editor for the workflow shell."""

    # Emitted (with the button's action key) when an in-panel action button is clicked.
    # The shell connects this to the owning stage's controller (see _create_stage_parameter_panels).
    action_requested = Signal(str)

    # Parameters whose value selects a METHOD block to show; each is a dropdown owning the
    # per-method containers built for its section (see _refresh_method_visibility).
    METHOD_DROPDOWNS = ("disp_method", "force_method")

    # Numeric flags the legacy "auto" inference reads: while force_method is stored as "auto",
    # a change to any of these can move the inferred method shown, so visibility must refresh.
    _AUTO_FORCE_FLAGS = ("l1_sparsity", "fwd_mask_strength", "bayesian_l2")

    PARAMETER_SECTIONS = [
        ("General", [
            ("pixel_size", "Pixel Size (um)", "float", 0.001, 100.0, 0.1, 3, None),
            ("frame_interval", "Frame Length (min)", "float", 0.001, 1000.0, 0.1, 3, None),
        ]),
        ("Displacement", [
            # Method + shared device at the top. The method dropdown decides which block
            # below is shown; each method's knobs stay hidden until it is selected.
            ("disp_method", "Method", "choice", None, None, None, None,
             ["PIV", "Lucas-Kanade", "FFD"]),
            ("disp_device", "Device", "choice", None, None, None, None,
             ["auto", "cuda", "cpu"]),
            (METHOD, "PIV", [
                (GROUP, "PIV (cross-correlation)"),
                ("piv_window", "Interrogation Window (px)", "int", 8, 128, 2, 0, None),
                ("piv_overlap", "Window Overlap", "float", 0.0, 0.95, 0.025, 3, None),
                ("piv_passes", "Passes", "int", 1, 12, 1, 0, None),
                ("piv_smooth", "Field Smoothing (σ)", "float", 0.0, 3.0, 0.5, 1, None),
            ]),
            (METHOD, "Lucas-Kanade", [
                (GROUP, "Lucas-Kanade (optical flow)"),
                ("ilk_radius", "Window Radius (px)", "int", 1, 64, 1, 0, None),
                ("ilk_num_warp", "Warp Iterations", "int", 1, 64, 1, 0, None),
            ]),
            (METHOD, "FFD", [
                (GROUP, "FFD (GPU only)"),
                ("ffd_num_iters", "Iterations / Level", "int", 1, 200, 1, 0, None),
                ("ffd_metric", "Image Metric", "choice", None, None, None, None, ["lncc", "mse"]),
                ("ffd_elastic", "Elastic Regularization", "float", 0.0, 10.0, 0.01, 3, None),
                ("ffd_early_stop", "Early-Exit Tolerance", "float", 0.0, 0.01, 0.00001, 6, None),
                (GROUP, "FFD image pyramid"),
                ("ffd_level_spacing", "Control Spacing (px)", "float", 4.0, 64.0, 1.0, 1, None),
                ("ffd_downscale", "Pyramid Downscale", "float", 1.1, 4.0, 0.1, 1, None),
                ("ffd_min_size", "Coarsest Size (px)", "int", 4, 128, 4, 0, None),
                ("ffd_num_levels", "Pyramid Levels", "int_display", None, None, None, None, None),
                (ADVANCED, "Advanced", "FFD"),
                ("ffd_interp", "Warp Interpolation", "choice", None, None, None, None, ["bicubic", "bilinear"]),
            ]),
            (GROUP, "General"),
            ("downscale_factor", "Downscale Factor", "int", 1, 10, 1, 0, None),
            ("disp_downscale_before", "Downsample Before\nMeasurement", "bool", None, None, None, None, None),
            (GROUP, "Mask confinement"),
            ("disp_mask_confine", "Confine to Mask", "bool", None, None, None, None, None),
            ("disp_mask_margin_um", "Mask Margin (um)", "float", 0.0, 200.0, 1.0, 1, None),
            (GROUP, "Visualization"),
            ("disp_vector_stride", "Vector Stride", "int", 1, 100, 1, 0, None),
            ("disp_arrow_scale", "Arrow Scale", "float", 0.1, 50.0, 0.1, 1, None),
            ("d_max", "Max Displacement (um)", "float", 0.1, 200.0, 0.1, 1, None),
        ]),
        ("Force", [
            ("young_modulus", "Young's Modulus (kPa)", "float", 0.1, 1000.0, 0.1, 2, None),
            ("poisson_ratio_substrate", "Poisson Ratio", "float", 0.0, 0.5, 0.01, 2, None),
            ("gel_height", "Gel Height (um)", "float", 0.0, 1000.0, 10.0, 1, None),
            # Method dropdown decides which block below is shown; each method's knobs stay
            # hidden until it is selected (same machinery as the Displacement Method).
            ("force_method", "Method", "choice", None, None, None, None,
             ["Elastic net", "FTTC + GCV", "Bayesian L2"]),
            (METHOD, "Elastic net", [
                (GROUP, "Elastic net (L1 + L2)"),
                ("l1_sparsity", "L1 Sparsity", "float", 0.0, 1.0, 0.01, 2, None),
                ("l2_ridge", "L2 Ridge", "float", 0.0, 32.0, 0.25, 2, None),
                (GROUP, "Mask confiner (soft support)"),
                ("fwd_mask_strength", "Mask Confinement", "float", 0.0, 100.0, 1.0, 0, None),
                ("fwd_mask_reach", "Mask Reach (px)", "float", 0.0, 20.0, 0.5, 1, None),
            ]),
            (METHOD, "FTTC + GCV", [
                (GROUP, "FTTC (Fourier Tikhonov)"),
                ("regularization", "Regularization (10^x)", "float", -21.0, 0.0, 0.5, 1, None),
                ("gcv_lambda", "Auto-pick λ (GCV, this frame)", "button",
                 None, None, None, None, None),
                ("auto_gcv", "Auto-pick λ per frame", "bool", None, None, None, None, None),
            ]),
            (METHOD, "Bayesian L2", [
                (GROUP, "Bayesian L2 (evidence-max λ)"),
                ("bayesian_freeze", "Freeze λ for the movie", "button",
                 None, None, None, None, None),
                ("bayesian_per_frame", "Re-estimate λ per frame", "bool",
                 None, None, None, None, None),
            ]),
            (GROUP, "Visualization"),
            ("force_vector_stride", "Vector Stride", "int", 1, 100, 1, 0, None),
            ("force_arrow_scale", "Arrow Scale", "float", 0.1, 50.0, 0.1, 1, None),
            ("f_max", "Max Force (Pa)", "float", 0.1, 10000.0, 1.0, 1, None),
        ]),
        ("Stress", [
            # BISM (Bayesian, mesh-free): Lambda entered by hand as a base-10
            # exponent, like Force's regularization.
            ("bism_regularization", "Regularization (10^x)", "float", -12.0, 0.0, 0.5, 1, None),
            (GROUP, "Visualization"),
            ("max_stress", "Max Stress (mN/m)", "float", 0.01, 1000.0, 0.1, 2, None),
        ]),
    ]

    # Tooltips keyed by parameter name, applied to both the label and control.
    # The PIV set maps onto napariTFM.backend.piv_displacement (multi-pass FFT
    # cross-correlation, GPU-accelerated when torch + CUDA are available).
    PARAMETER_TOOLTIPS = {
        "l1_sparsity": (
            "Sparse traction inversion (group-L1), the recommended default. Above 0 it "
            "overrides Mask Confinement and plain FTTC. It regularizes by sparsity "
            "instead of smoothing: it thresholds small forces to exactly zero rather "
            "than spreading them, so it recovers the adhesion forces more accurately "
            "and keeps the peak better than FTTC or confinement, with NO mask needed. "
            "The value is a fraction (0..1) of the level that zeros the whole field: "
            "~0.05 is a good start (the sweep's flat basin is 0.02-0.11, and it is "
            "safer to err low), raise it for noisier data (more sparsity), lower it "
            "if real forces are being erased. A loaded mask, if present, acts as a "
            "soft support: Mask Confinement adds an off-mask L2 penalty (ramped over a "
            "collar, no hard edge), so it self-confines but the exterior is discouraged "
            "rather than forbidden. 0 = off."
        ),
        "l2_ridge": (
            "Elastic-net L2 ridge, the second knob of the sparse (L1) solver — it "
            "only does anything when L1 Sparsity > 0, and adds a global ½λ₂‖t‖² "
            "shrinkage on top of the L1 threshold (pure sparsity becomes elastic net). "
            "The value is a fraction of the median per-mode curvature, useful band "
            "~0.1..1. It smooths the traction where L1 alone would leave it spiky, at "
            "the cost of shrinking peak magnitude. In the heuristic sweep it did not "
            "earn its keep — tuned L1 with the ridge off beat every ridge setting on "
            "both compact dipoles and diffuse cells — so it defaults to 0. Raise it "
            "only if a sparse recovery looks too speckled and you would rather trade "
            "peak height for a smoother field. Unrelated to Bayesian L2 and to the "
            "manual Regularization above, which act on the plain-FTTC path instead. 0 = "
            "pure L1."
        ),
        "force_method": (
            "Traction-inversion method. Each shows only its own controls:\n"
            "• Elastic net (L1+L2) — sparse group-L1 with an optional L2 ridge and a soft "
            "mask confiner. The recommended default: thresholds small forces to zero, best "
            "in-cell accuracy and peak recovery, no mask needed.\n"
            "• FTTC + GCV — classic Fourier Tikhonov inversion. Set λ by hand, or let "
            "Generalized Cross-Validation pick it (button = once, checkbox = per frame).\n"
            "• Bayesian L2 — the real-space evidence-max reconstruction (Huang et al. 2019); "
            "λ is chosen from the data automatically, per frame or frozen for a movie."
        ),
        "auto_gcv": (
            "Pick the FTTC λ by Generalized Cross-Validation on every frame automatically, "
            "instead of the manual slider. GCV works on the Fourier operator (the same λ the "
            "slider sets). While on, the manual slider and the one-shot button are disabled. "
            "For a single frozen λ across the movie, use the button instead and leave this off."
        ),
        "gcv_lambda": (
            "Estimate the GCV-optimal λ on the current frame and write it into the "
            "Regularization slider, where it stays editable. One-shot: the same λ is then used "
            "for every frame. For per-frame re-picking, use the checkbox below."
        ),
        "bayesian_freeze": (
            "Estimate the Bayesian evidence-max λ on the current frame and freeze it for the "
            "whole movie (turns off per-frame re-estimation). Freezing keeps the regularization "
            "— and so the traction maps — comparable frame to frame. With a mask loaded the "
            "noise is measured from the cell-free exterior (BL2); without one it is inferred (ABL2)."
        ),
        "bayesian_per_frame": (
            "Re-estimate the Bayesian λ independently on every frame (parameter-free, the "
            "default). Robust, but λ drifts frame to frame, which breaks cross-frame comparison — "
            "use Freeze λ for a time series or a condition comparison."
        ),
        "fwd_mask_strength": (
            "How hard traction is pushed out of the loaded mask's exterior — a "
            "soft support on whichever solver is active. 0 = no confinement (with "
            "L1 Sparsity on, pure sparsity; with it off, plain FTTC). Above 0 = the "
            "off-mask traction is penalized more, log-scaled so every step does "
            "something: both the L1 and the forward solver add an off-mask L2 penalty "
            "to their objective. Same dial, same meaning on both routes; the exterior "
            "is discouraged, never forbidden (no hard edge). "
            "Needs a mask loaded — the same external mask the Stress stage uses."
        ),
        "fwd_mask_reach": (
            "How far past the mask forces are still allowed — an apron (in force-grid "
            "pixels) the free region is grown by before confinement bites. Forces are "
            "free within mask+reach, then pushed out beyond it. Because the apron is a "
            "genuinely zero-penalty region, this is orthogonal to Mask Confinement: "
            "that sets how HARD the exterior is pushed, this sets how FAR OUT the "
            "boundary is — and unlike a soft skirt, reach keeps moving the boundary "
            "even at maximum confinement. The apron only ever reaches outward, so real "
            "forces on the cell rim are never clipped. 0 = confine to the mask itself. "
            "Same apron on both the sparse (L1) and confined routes."
        ),
        "disp_method": (
            "Displacement algorithm. PIV (FFT cross-correlation) is a forgiving "
            "default, quietest off-cell in our benchmark and graceful up to large "
            "motion. Lucas-Kanade (iterative optical flow) tracks it closely at "
            "small motion and is fast and light. FFD (free-form deformation) leads "
            "under large deformation but is GPU-only. Accuracy was a near-tie in the "
            "one regime we tested; they separate on off-cell noise and capture "
            "range. See the 'Choosing a displacement method' doc, and preview a "
            "couple on your own data before committing."
        ),
        "disp_device": (
            "Compute backend, shared by every method. 'auto' uses the GPU when a "
            "CUDA device is present, else the CPU. 'cuda' requires a GPU; 'cpu' "
            "forces CPU. PIV is one torch implementation run on either device, so "
            "CPU and GPU give the same algorithm (differing only by ~1e-4 px "
            "device-level rounding). Lucas-Kanade's GPU port is numerically "
            "identical to its scikit-image CPU path. FFD is GPU-only."
        ),
        "piv_window": (
            "Final PIV interrogation window, in pixels: the primary PIV knob and the "
            "peak-vs-noise trade-off. Cross-correlation runs on windows of this size "
            "on the last (finest) pass. Smaller windows sharpen the near-adhesion "
            "peak but need denser texture and raise off-cell noise; larger windows "
            "are more robust to noise but blur small displacements. ~24 is a "
            "reasonable start."
        ),
        "piv_overlap": (
            "Fractional overlap between neighbouring PIV windows (0-0.95). Higher "
            "overlap samples the field more finely and recovers sharp peaks better, "
            "at more compute and memory (high overlap on large frames can exhaust "
            "GPU memory). A first-order knob, not a default to ignore: tune it with "
            "the window."
        ),
        "piv_passes": (
            "Number of coarse-to-fine PIV passes. Each pass re-warps the moving "
            "image by the running estimate and correlates with a smaller window, so "
            "more passes capture larger displacements and drive convergence at the "
            "cost of runtime; gains taper off after a few. Raise it if large motion "
            "is missed."
        ),
        "piv_smooth": (
            "Per-pass Gaussian smoothing of the displacement field (sigma, in "
            "vector-grid cells), applied after outlier rejection on every pass. It "
            "regularizes noise in the fine passes -- higher = smoother fields but "
            "damped peaks; 0 = raw output (sharper but rougher). ~1.0 is a "
            "reasonable start."
        ),
        "ilk_radius": (
            "Half-window of the local Lucas-Kanade solve, in pixels: the primary iLK "
            "knob. It acts as a noise aperture, not a capture knob (the image "
            "pyramid handles capture). Start near 7-10; larger for noisier images, "
            "which trades peak sharpness for a quieter field."
        ),
        "ilk_num_warp": (
            "Coarse-to-fine warp iterations per pyramid level. More iterations refine "
            "the flow and help it converge, with diminishing returns past ~16. Raise "
            "it if the field looks under-converged; it does not extend capture range."
        ),
        "ffd_level_spacing": (
            "Finest control-grid spacing, in pixels: FFD's primary knob and its "
            "bias-variance dial. Fine (~8) recovers sharp peaks on clean data; coarse "
            "(~16-24) regularizes noise and keeps the off-cell background quiet. Its "
            "best value grows with image noise."
        ),
        "ffd_num_levels": (
            "Read-only: how many pyramid levels FFD will build for the loaded input, "
            "derived from Pyramid Downscale and Coarsest Size — not a knob. Deeper "
            "pyramids capture larger displacements. To change it, adjust Coarsest Size "
            "(smaller = deeper) or Pyramid Downscale. Shows '—' until an input loads."
        ),
        "ffd_metric": (
            "Image-match objective FFD minimises. 'lncc' (local normalised "
            "cross-correlation) weights every window equally, so the sharp "
            "high-motion peak is not drowned by the low-motion bulk: it preserves "
            "peaks better and is the default. 'mse' is simpler but softens the peak."
        ),
        "ffd_elastic": (
            "Weight of the elastic (Navier strain-energy) prior on the deformation — "
            "the physically correct regularizer for a TFM gel, penalising first "
            "derivatives (strain) so it permits concentrated strain at force points "
            "but forbids implausible roughness. 0 = off. Raise it to quieten a noisy "
            "field; too high over-smooths real peaks."
        ),
        "ffd_num_iters": (
            "LBFGS iterations per pyramid level. More iterations refine each level's "
            "control-grid fit with diminishing returns; the default is ample for the "
            "range we tested."
        ),
        "ffd_early_stop": (
            "Per-level LBFGS convergence tolerance: a level stops as soon as its loss "
            "improves by less than this between steps, instead of always running the "
            "full Iterations / Level budget. 0 (default) runs every level to the full "
            "budget, unchanged. Raise it to trade a little accuracy for speed once a "
            "level has essentially converged."
        ),
        "ffd_downscale": (
            "Image-pyramid downscale factor per level. 2.0 halves each axis per level "
            "(the default); smaller factors build a finer pyramid (more levels, gentler "
            "capture steps) at more compute. Rarely needs changing."
        ),
        "ffd_min_size": (
            "Smallest dimension (px) the coarsest pyramid level is allowed to reach: "
            "the pyramid stops adding coarser levels once a level would fall below it. "
            "With Pyramid Downscale this sets the pyramid depth (shown as Pyramid "
            "Levels) and hence FFD's capture range — smaller = deeper = larger motions."
        ),
        "ffd_interp": (
            "Interpolation used to warp the moving image. 'bicubic' (default) preserves "
            "sharp peaks better, matching a cubic-B-spline resample; 'bilinear' is "
            "cheaper and slightly smoother."
        ),
        "disp_downscale_before": (
            "Where the Downscale Factor coarsening happens. Off (default): measure at "
            "full resolution, then average the vector field down to the grid — uses all "
            "bead texture, most accurate. On: average the images first and measure on "
            "1/factor² the pixels — faster, and on real data within ~0.06 px of the "
            "full-res result. Drift registration always runs at full resolution either "
            "way. No effect when Downscale Factor is 1."
        ),
        "disp_mask_confine": (
            "Measure displacement only within the external mask (+ margin), for every "
            "method. Each frame is analysed inside its cell's bounding box plus the "
            "Mask Margin and read as zero outside: this skips the empty, vignette-"
            "corrupted periphery (faster) and removes that far-field artifact from the "
            "saved field structurally, rather than relying on the downstream mask. Needs "
            "an external mask; a no-op without one. Off = full-frame (unchanged)."
        ),
        "disp_mask_margin_um": (
            "How far beyond the cell edge to keep measuring, in microns, when Confine to "
            "Mask is on. It must cover the substrate-displacement halo around the cell — "
            "set it to that halo's decay length. Too small silently clips real "
            "near-cell displacement; too large re-admits the periphery. Only read when "
            "Confine to Mask is on."
        ),
    }

    def __init__(self, parameter_manager: ParameterManager, section_titles: tuple[str, ...] | None = None):
        super().__init__()
        self.parameter_manager = parameter_manager
        self._section_titles = set(section_titles) if section_titles is not None else None
        self.parameter_controls = {}
        # Action buttons (kind == "button") are not bound to a parameter value; they emit
        # action_requested on click. Kept out of parameter_controls so sync/enumeration skip them.
        self._action_buttons = {}
        # (h, w) of the current input frame, pushed in by the shell (set_input_shape),
        # so the read-only FFD Pyramid Levels display can show the real derived depth.
        self._input_shape = None
        # Populated by _setup_ui: one (toggle, container, owner_method) per collapsible
        # Advanced disclosure, for method-scoped show/hide (see _refresh_advanced_visibility).
        self._advanced_blocks = []
        # Populated by _setup_ui: one (container, method_name) per displacement-method
        # block, for show/hide by the selected method (see _refresh_method_visibility).
        self._method_blocks = []
        self._setup_ui()
        self._sync_all_controls()
        self._refresh_confinement_enablement()
        self._refresh_force_lambda_enablement()
        self._apply_method_availability()
        self._refresh_method_visibility()
        self._refresh_advanced_visibility()
        self._refresh_pyramid_levels()
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_manager.parameter_changed.connect(self._refresh_confinement_enablement)
        self.parameter_manager.parameter_changed.connect(self._refresh_force_lambda_enablement)
        self.parameter_manager.parameter_changed.connect(self._refresh_method_visibility)
        self.parameter_manager.parameter_changed.connect(self._refresh_advanced_visibility)
        self.parameter_manager.parameter_changed.connect(self._refresh_pyramid_levels)

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TIGHT_SPACING)

        for title, specs in self.PARAMETER_SECTIONS:
            if self._section_titles is not None and title not in self._section_titles:
                continue
            self._build_section(layout, title, specs)

        self.setLayout(layout)

    def _build_section(self, layout, title, specs):
        """Build one titled parameter section: a header row, then its specs laid out by
        :meth:`_lay_specs` (which peels off METHOD blocks and ADVANCED disclosures)."""
        header = QLabel(title)
        header.setStyleSheet(section_label_style())
        self._lay_specs(layout, specs, header=header)

    def _lay_specs(self, layout, specs, header=None):
        """Lay a run of specs into ``layout`` as two-per-row grids, peeling off the two
        kinds of sub-block into their own show/hide-able widgets:

        * a METHOD marker closes the current grid and emits a per-method container (the
          method's whole group -- sub-headers, controls, its own ADVANCED disclosure --
          shown only while that method is selected), then a fresh grid resumes;
        * an ADVANCED marker likewise emits a collapsible block for the specs up to the
          next GROUP/ADVANCED/METHOD marker.

        Shared by the top-level section and each method container, so a method block may
        itself carry GROUP sub-headers and an ADVANCED disclosure. ``header`` (a section
        title) opens the first grid at row 0 when given."""
        grid = section_grid()
        if header is not None:
            add_section_header(grid, 0, header)
            row = 1
        else:
            row = 0
        pending = None
        i, n = 0, len(specs)
        while i < n:
            spec = specs[i]
            if spec[0] is METHOD:
                row, pending = self._flush_pending(grid, row, pending)
                layout.addLayout(grid)
                self._build_method_block(layout, spec[1], spec[2])
                grid = section_grid()          # fresh grid for the specs after the block
                row, pending = 0, None
                i += 1
                continue
            if spec[0] is ADVANCED:
                row, pending = self._flush_pending(grid, row, pending)
                layout.addLayout(grid)
                j = i + 1
                block = []
                while j < n and specs[j][0] not in (GROUP, ADVANCED, METHOD):
                    block.append(specs[j])
                    j += 1
                self._build_advanced_block(layout, spec[1], spec[2], block)
                grid = section_grid()          # fresh grid for the specs after the block
                row, pending = 0, None
                i = j
                continue
            row, pending = self._place_spec(grid, row, pending, spec)
            i += 1
        row, pending = self._flush_pending(grid, row, pending)
        layout.addLayout(grid)

    def _build_method_block(self, layout, method_name, specs):
        """Emit a container holding one displacement method's whole knob group, shown only
        while that method is selected (so an unselected method's parameters are hidden, not
        merely greyed). Registered in ``_method_blocks`` for _refresh_method_visibility."""
        container = QWidget()
        clayout = QVBoxLayout()
        clayout.setContentsMargins(0, 0, 0, 0)
        clayout.setSpacing(TIGHT_SPACING)
        self._lay_specs(clayout, specs)
        container.setLayout(clayout)
        container.setVisible(False)     # _refresh_method_visibility reveals the active one
        self._method_blocks.append((container, method_name))
        layout.addWidget(container)

    def _build_advanced_block(self, layout, title, owner_method, specs):
        """Emit a collapsed-by-default disclosure toggle + its (own-grid) container.

        The container holds ``specs`` laid out exactly as a normal run of rows, and is
        method-scoped: _refresh_advanced_visibility reveals the toggle only while
        ``owner_method`` is the selected displacement method, and the container only
        when the toggle is also expanded."""
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setAutoRaise(True)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.RightArrow)
        toggle.setStyleSheet(section_subheader_style())

        container = QWidget()
        cgrid = section_grid()
        row, pending = 0, None
        for spec in specs:
            row, pending = self._place_spec(cgrid, row, pending, spec)
        self._flush_pending(cgrid, row, pending)
        container.setLayout(cgrid)
        container.setVisible(False)

        def _on_toggled(checked, tb=toggle):
            tb.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            self._refresh_advanced_visibility()

        toggle.toggled.connect(_on_toggled)
        self._advanced_blocks.append((toggle, container, owner_method))
        layout.addWidget(toggle)
        layout.addWidget(container)

    def _flush_pending(self, grid, row, pending):
        """Emit a lone scalar still waiting for a right-hand partner as a left-only row."""
        if pending is not None:
            add_section_pair_row(grid, row, pending[0], pending[1], left_tooltip=pending[2])
            return row + 1, None
        return row, pending

    def _place_spec(self, grid, row, pending, spec):
        """Place one spec into ``grid``, returning the updated ``(row, pending)``.

        Scalars pair two-per-row; a GROUP sub-header or a range slider takes a full
        row and first flushes any scalar still waiting for a partner. ``pending`` is the
        (label, control, tooltip) of a left-hand scalar awaiting its right-hand mate."""
        if spec[0] is GROUP:
            row, pending = self._flush_pending(grid, row, pending)
            subheader = QLabel(spec[1])
            subheader.setStyleSheet(section_subheader_style())
            add_section_header(grid, row, subheader)
            return row + 1, pending
        if spec[2] == "range":
            row, pending = self._flush_pending(grid, row, pending)
            label, control, tooltip = self._control_for_spec(spec)
            add_section_labeled_full_row(grid, row, label, control, tooltip=tooltip)
            return row + 1, pending
        if spec[2] == "button":
            # A full-row action button (its label is the button text); not a bound parameter.
            row, pending = self._flush_pending(grid, row, pending)
            _, control, _ = self._control_for_spec(spec)
            add_section_header(grid, row, control)
            return row + 1, pending
        if pending is None:
            return row, self._control_for_spec(spec)
        label, control, tooltip = self._control_for_spec(spec)
        add_section_pair_row(grid, row, pending[0], pending[1], label, control,
                             left_tooltip=pending[2], right_tooltip=tooltip)
        return row + 1, None

    def _control_for_spec(self, spec):
        name, label, kind, min_val, max_val, step, decimals, choices = spec
        if kind == "button":
            control = QPushButton(label)
            control.setObjectName(f"workflow_action_{name}")
            control.clicked.connect(lambda _checked=False, n=name: self.action_requested.emit(n))
            self._action_buttons[name] = control
        else:
            control = self._create_control(name, kind, min_val, max_val, step, decimals, choices)
        tooltip = self.PARAMETER_TOOLTIPS.get(name)
        if tooltip:
            control.setToolTip(tooltip)
        return label, control, tooltip

    def _create_control(self, name, kind, min_val, max_val, step, decimals, choices):
        if kind == "range":
            lo_name, hi_name = name
            control = rslider(
                min_val, max_val,
                self.parameter_manager.get_ui_parameter(lo_name),
                self.parameter_manager.get_ui_parameter(hi_name),
                step=step, decimals=decimals,
            )
            control._range_params = (lo_name, hi_name)
            control.valueChanged.connect(
                lambda values, lo=lo_name, hi=hi_name: (
                    self.parameter_manager.set_ui_parameter(lo, values[0]),
                    self.parameter_manager.set_ui_parameter(hi, values[1]),
                )
            )
            control.setObjectName(f"workflow_parameter_{lo_name}")
            # Both names point at this one control so external changes to either
            # bound re-sync the slider (see _sync_parameter).
            self.parameter_controls[lo_name] = control
            self.parameter_controls[hi_name] = control
            return control
        if kind == "int_display":
            # Read-only derived value (e.g. Pyramid Levels): a plain label, not wired to
            # the parameter manager. Its text is set by the owner's refresh hook.
            control = QLabel("—")
            control.setObjectName(f"workflow_parameter_{name}")
            self.parameter_controls[name] = control
            return control
        if kind == "int":
            control = islider(min_val, max_val, self.parameter_manager.get_ui_parameter(name), step=step)
            control.valueChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "float":
            control = dslider(min_val, max_val, self.parameter_manager.get_ui_parameter(name), step=step, decimals=decimals)
            control.valueChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "choice":
            control = QComboBox()
            control.addItems(choices)
            control.currentTextChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "bool":
            control = QCheckBox()
            # bool(state) rather than `state == Qt.Checked`: stateChanged emits a
            # plain int (0/2), and under PyQt6/PySide6 comparing it to the
            # Qt.CheckState enum silently yields False, leaving the box stuck off.
            control.stateChanged.connect(
                lambda state, n=name: self.parameter_manager.set_ui_parameter(n, bool(state))
            )
        else:
            raise ValueError(f"Unsupported parameter control type: {kind}")

        control.setObjectName(f"workflow_parameter_{name}")
        self.parameter_controls[name] = control
        return control

    def set_input_shape(self, shape):
        """Tell the panel the current input frame's ``(h, w)`` (the shell calls this
        when an experiment/input loads) so the read-only FFD Pyramid Levels display
        reflects the real pyramid depth. Panels without that control ignore it."""
        new = None if shape is None else (int(shape[0]), int(shape[1]))
        if new == self._input_shape:
            return
        self._input_shape = new
        self._refresh_pyramid_levels()

    def _refresh_pyramid_levels(self, name=None, value=None):
        """Update the read-only Pyramid Levels label: the depth FFD derives from
        Pyramid Downscale + Coarsest Size for the loaded input frame. A cheap no-op
        when the changed parameter is neither of those two knobs, or the panel has
        no such display (e.g. the Force/Stress panels)."""
        if name is not None and name not in ("ffd_downscale", "ffd_min_size"):
            return
        label = self.parameter_controls.get("ffd_num_levels")
        if label is None:
            return
        if self._input_shape is None:
            label.setText("—")
            return
        downscale = float(self.parameter_manager.get_parameter("ffd_downscale"))
        min_size = int(self.parameter_manager.get_parameter("ffd_min_size"))
        label.setText(str(_derived_pyramid_levels(self._input_shape, downscale, min_size)))

    def _refresh_confinement_enablement(self, name=None, value=None):
        """Grey out the Displacement stage's Mask Margin unless Confine to Mask is on.

        Driven from parameter_changed (a cheap no-op when the changed param is not
        the gate) and once at construction for the initial state.
        """
        if name is not None and name != "disp_mask_confine":
            return
        margin = self.parameter_controls.get("disp_mask_margin_um")
        if margin is not None:
            margin.setEnabled(bool(self.parameter_manager.get_parameter("disp_mask_confine")))

    def _refresh_force_lambda_enablement(self, name=None, value=None):
        """Grey the FTTC manual-λ slider and its one-shot GCV button while per-frame GCV is on
        (``auto_gcv``): per-frame λ selection owns λ, so the manual value is not in play.

        Driven from parameter_changed (no-op unless auto_gcv moved) and once at construction.
        """
        if name is not None and name != "auto_gcv":
            return
        slider = self.parameter_controls.get("regularization")
        button = self._action_buttons.get("gcv_lambda")
        if slider is None and button is None:
            return
        auto = bool(self.parameter_manager.get_parameter("auto_gcv"))
        if slider is not None:
            slider.setEnabled(not auto)
        if button is not None:
            button.setEnabled(not auto)

    def _resolved_method_value(self, dropdown_name):
        """The concrete method a method dropdown is currently on. For ``force_method`` the
        stored ``"auto"`` (legacy/back-compat) is resolved to the inferred concrete method for
        display, without mutating the stored value (so the backend keeps inferring at run time,
        with the real mask, and an explicit user pick still wins)."""
        value = self.parameter_manager.get_parameter(dropdown_name)
        if dropdown_name == "force_method" and value == "auto":
            from napariTFM.backend.fttc import infer_force_method
            return infer_force_method(self.parameter_manager.get_fttc_parameters())
        return value

    def _active_methods(self):
        """The set of concrete method values currently selected across every built method
        dropdown. Method names are unique across dropdowns, so membership picks exactly one
        block per section."""
        return {
            self._resolved_method_value(dd)
            for dd in self.METHOD_DROPDOWNS
            if self.parameter_controls.get(dd) is not None
        }

    def _refresh_method_visibility(self, name=None, value=None):
        """Show only the selected method's parameter block per section; hide the rest, so an
        unselected method's knobs are invisible (not merely greyed).

        Driven from parameter_changed (cheap no-op unless a method dropdown moved, or a flag
        that resolves force_method's "auto" display changed) and once at construction. No-ops
        when no method dropdown is built (the panel can hold a subset of sections).
        """
        if name is not None and name not in self.METHOD_DROPDOWNS and name not in self._AUTO_FORCE_FLAGS:
            return
        if not any(self.parameter_controls.get(dd) is not None for dd in self.METHOD_DROPDOWNS):
            return
        active = self._active_methods()
        for container, method in self._method_blocks:
            container.setVisible(method in active)

    def _refresh_advanced_visibility(self, name=None, value=None):
        """Show each Advanced disclosure only for its owning method, expanded on demand.

        An Advanced block lives inside its method's block, so it is already hidden when
        another method is selected; its toggle is revealed only for the owner method and
        its container only when that toggle is also expanded. Driven from parameter_changed
        (a cheap no-op unless a method dropdown moved), from the toggles themselves (name is
        None), and once at construction. No-ops when no Advanced block was built.
        """
        if name is not None and name not in self.METHOD_DROPDOWNS:
            return
        if not any(self.parameter_controls.get(dd) is not None for dd in self.METHOD_DROPDOWNS):
            return
        active = self._active_methods()
        for toggle, container, owner_method in self._advanced_blocks:
            owner_selected = owner_method in active
            toggle.setVisible(owner_selected)
            container.setVisible(owner_selected and toggle.isChecked())

    def _apply_method_availability(self):
        """Disable options that cannot run on this machine, with an explaining tooltip.

        FFD is GPU-only and the 'cuda' device needs a CUDA GPU + PyTorch, so both are
        greyed in their dropdowns when no GPU is present. The user still sees them (so
        the capability is discoverable) but cannot select an unrunnable configuration.
        """
        from napariTFM.backend.ffd_displacement import ffd_available
        if ffd_available():
            return
        self._disable_combo_item(
            self.parameter_controls.get("disp_method"), "FFD",
            "FFD is GPU-only. Install the GPU extra (pip install napariTFM[gpu]) and "
            "use a CUDA device to enable it.",
        )
        self._disable_combo_item(
            self.parameter_controls.get("disp_device"), "cuda",
            "No CUDA device / PyTorch found; 'cuda' is unavailable. 'auto' uses the "
            "CPU reference here.",
        )

    @staticmethod
    def _disable_combo_item(combo, text, tooltip):
        """Grey out one item of a QComboBox by its text, leaving it visible."""
        if combo is None:
            return
        idx = combo.findText(text, Qt.MatchFixedString)
        if idx < 0:
            return
        item = combo.model().item(idx)
        if item is not None:
            item.setEnabled(False)
            item.setToolTip(tooltip)

    def _sync_all_controls(self):
        for name in self.parameter_controls:
            self._sync_parameter(name, self.parameter_manager.get_ui_parameter(name))

    def _sync_parameter(self, param_name: str, value: Any):
        control = self.parameter_controls.get(param_name)
        if control is None:
            return
        if isinstance(control, QLabel):
            # A read-only derived display (e.g. Pyramid Levels): not editable and not
            # bound to the parameter; its text is maintained by its own refresh hook.
            return

        display_value = self.parameter_manager.get_ui_parameter(param_name)
        if param_name == "force_method" and display_value == "auto":
            # Display the inferred concrete method; storage stays "auto" (signals are blocked
            # here, so selecting the item does not write back). See _resolved_method_value.
            display_value = self._resolved_method_value("force_method")
        control.blockSignals(True)
        try:
            if isinstance(control, QLabeledDoubleRangeSlider):
                lo_name, hi_name = control._range_params
                control.setValue((
                    self.parameter_manager.get_ui_parameter(lo_name),
                    self.parameter_manager.get_ui_parameter(hi_name),
                ))
            elif isinstance(control, QComboBox):
                index = control.findText(str(display_value), Qt.MatchFixedString)
                if index >= 0:
                    control.setCurrentIndex(index)
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(display_value))
            else:
                control.setValue(display_value)
        finally:
            control.blockSignals(False)


class SpinBoxEventFilter(QObject):
    def eventFilter(self, obj, event):
        # Check for all spinnable input widgets
        if (isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox, QLabeledSlider, QLabeledDoubleSlider)) and
                event.type() == event.Wheel):
            if not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class _InteractiveStageAdapter(QObject):
    """Translate an existing stage widget to the dependency coordinator API."""

    completed = Signal(object)
    failed = Signal(object)

    def __init__(self, stage, widget, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.widget = widget
        getattr(widget, f"{stage}_calculated").connect(self.completed.emit)
        widget.controller.analysis_failed.connect(self.failed.emit)

    def preview(self, *, completion=None, **inputs):
        controller = self.widget.controller
        method_name = {
            "displacement": "preview_displacement",
            "force": "preview_force",
            "stress": "preview_current_frame",
        }[self.stage]
        method = getattr(controller, method_name)
        return method(completion=completion, **inputs)

    def run(self):
        self.widget.run_action()

    def cancel(self):
        self.widget.cancel_action()



class napariTFMWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self._applying_state = False
        self.viewer = napari_viewer

        # Create and install event filter
        self.spinbox_filter = SpinBoxEventFilter(self)

        # Find and filter all spinboxes in the application
        def install_filter_on_inputs():
            for widget in self.window().findChildren(
                (QSpinBox, QDoubleSpinBox, QComboBox, QLabeledSlider, QLabeledDoubleSlider)
            ):
                widget.installEventFilter(self.spinbox_filter)
                widget.setFocusPolicy(Qt.StrongFocus)

        # Install filters after a short delay to ensure all widgets are created.
        # Import QTimer locally (not the module-level name) so singleShot uses the
        # real Qt timer even when tests monkeypatch the module-level QTimer with a
        # fake poll-timer stand-in.
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, install_filter_on_inputs)

        # Give the dock a comfortable default/minimum width for the panel body.
        self.setMinimumWidth(420)

        # Create scroll area for widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget for scroll area
        container = QWidget()
        container_layout = QVBoxLayout()
        # Right margin keeps content clear of the scroll area's vertical
        # scrollbar instead of butting right up against it.
        container_layout.setContentsMargins(0, 0, 8, 0)
        container.setLayout(container_layout)

        # Brand row + the Project/Parameters toolbar (the front door), all in
        # one row now: icon-only buttons, right-aligned opposite the title,
        # grouped under inline "Project" / "Parameters" captions (the caption
        # disambiguates the group; load/save share an icon across groups on
        # purpose — direction of the icon's arrow disambiguates load vs save).
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("napariTFM")
        title.setStyleSheet(title_style())
        title_row.addWidget(title)
        title_row.addStretch()

        self.new_project_btn = self._make_toolbar_button("new", "Start a new project")
        self.load_project_btn = self._make_toolbar_button("load", "Load a project")
        self.save_project_btn = self._make_toolbar_button("save", "Save project as…")
        self.load_params_btn = self._make_toolbar_button("load", "Load parameters preset")
        self.save_params_btn = self._make_toolbar_button("save", "Save parameters preset")
        self.reset_params_btn = self._make_toolbar_button("reset", "Reset parameters")

        title_row.addWidget(self._toolbar_caption("Project"))
        for button in (self.new_project_btn, self.load_project_btn, self.save_project_btn):
            title_row.addWidget(button)
        title_row.addWidget(self._toolbar_divider())
        title_row.addWidget(self._toolbar_caption("Parameters"))
        for button in (self.load_params_btn, self.save_params_btn, self.reset_params_btn):
            title_row.addWidget(button)

        self._toolbar_icon_buttons = [
            (self.new_project_btn, "new"),
            (self.load_project_btn, "load"),
            (self.save_project_btn, "save"),
            (self.load_params_btn, "load"),
            (self.save_params_btn, "save"),
            (self.reset_params_btn, "reset"),
        ]

        container_layout.addLayout(title_row)

        # Progressive-disclosure gate (G0/G1/G2). No project is open at launch.
        self._project_open = False
        self._dirty = False

        # Initialize managers
        self.data_manager = DataManager()
        self.parameter_manager = ParameterManager()
        self.visualization_manager = VisualizationManager(self.viewer, self.data_manager)

        self.experiments_list = ExperimentsList(
            status_fn=self._experiment_stage_status,
            parameter_manager=self.parameter_manager,
            data_manager=self.data_manager,
        )
        self.experiments_list.active_changed.connect(
            self._on_active_experiment_changed
        )
        self.experiments_list.experiments_changed.connect(
            self._on_experiments_changed
        )
        self.experiments_list.run_selected_requested.connect(self._run_selected_experiments)
        self.experiments_list.cancel_run_selected_requested.connect(self._cancel_run_selected)
        self.experiments_list.pool_requested.connect(self._on_pool_requested)
        self.experiments_list.stage_load_requested.connect(self._on_row_stage_clicked)
        self._active_batch = None
        container_layout.addWidget(self.experiments_list)

        self._active_experiment: str | None = None
        # Small LRU of decoded stage-array bundles, keyed by the .ntfm's
        # (path, mtime, size), so a circle click that revisits an experiment —
        # or clicks a second stage of the one just loaded — is instant instead
        # of re-decoding the whole OME-TIFF. A stage Run rewrites the .ntfm, so
        # its mtime change invalidates the stale entry automatically.
        self._stage_arrays_cache: "OrderedDict[tuple, object]" = OrderedDict()
        # Monotonic token + worker handle for the off-thread cold decode: a
        # newer circle/row click supersedes an in-flight one so a slow decode
        # can never paint a stage the user has already clicked away from.
        self._stage_load_token = 0
        self._stage_load_worker = None
        # ntfm-backed stages whose arrays are currently decoded into DataManager
        # + the viewer for the active experiment. Loading is display-only and
        self._pipeline_context_label = QLabel("Pipeline")
        self._pipeline_context_label.setStyleSheet(section_label_style())
        container_layout.addWidget(self._pipeline_context_label)

        # One panel-level status line (P2) — replaces the per-stage labels/bars.
        # Stages report run/skip/done into it, prefixed with the reporting stage.
        self.status_label = QLabel("")
        self.status_label.setObjectName("global_status_label")
        self.status_label.setWordWrap(True)
        container_layout.addWidget(self.status_label)

        self._stage_parameter_panels_by_key = self._create_stage_parameter_panels()

        # Parameter preset toolbar (recipe import/export).
        self.save_params_btn.clicked.connect(self._save_params)
        self.load_params_btn.clicked.connect(self._load_params)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        # Project toolbar (the front door): handlers defined below.
        self.new_project_btn.clicked.connect(self._new_project)
        self.load_project_btn.clicked.connect(self._load_project)
        self.save_project_btn.clicked.connect(self._save_project)

        # Initialize all widgets with parameter_manager
        self.displacement_widget = DisplacementAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        self.force_widget = FTTCWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )
        self._wire_force_panel_actions()

        self.stress_widget = StressWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        # Wire Force to the disk: its displacement input may live only on disk (a done
        # experiment selected but not yet viewed). Force's Preview/Run then enable from
        # the on-disk displacement status, and the solver pulls that displacement into
        # memory on demand when it actually runs — matching the "calculations re-read
        # from disk" model rather than requiring the field be resident first.
        self.force_widget.set_displacement_available_check(self._displacement_available)
        self.force_widget.controller.set_displacement_loader(self._ensure_displacement_resident)

        # Same for Stress: its force input may live only on disk (a done experiment
        # selected but not yet viewed). Stress's Preview/Run then enable from the
        # on-disk force status, and the solver pulls that force into memory on
        # demand when it actually runs.
        self.stress_widget.set_force_available_check(self._force_available)
        self.stress_widget.controller.set_force_loader(self._ensure_force_resident)

        self._interactive_stage_adapters = {
            key: _InteractiveStageAdapter(key, stage_widget, self)
            for key, stage_widget in {
                "displacement": self.displacement_widget,
                "force": self.force_widget,
                "stress": self.stress_widget,
            }.items()
        }
        self._interactive_stage_coordinator = InteractiveStageCoordinator(
            stages=self._interactive_stage_adapters,
            artifact_getters={
                "displacement": lambda: self._interactive_artifact("displacement"),
                "force": lambda: self._interactive_artifact("force"),
                "stress": lambda: self._interactive_artifact("stress"),
            },
            parameter_getters={
                stage: (lambda key=stage: self._interactive_parameters(key))
                for stage in ("displacement", "force", "stress")
            },
            prompt=self._prompt_for_stale_stage,
            source_validator=self._validate_interactive_sources,
            progress=self._set_interactive_progress,
            parent=self,
        )

        # Funnel every pipeline stage's progress into the single global status
        # label (P2). Run-selected (the retired batch widget's successor) reports via
        # its own per-folder callback, not a controller progress signal.
        for stage_widget, stage_label in (
            (self.displacement_widget, "Displacement"),
            (self.force_widget, "Force"),
            (self.stress_widget, "Stress"),
        ):
            stage_widget.controller.progress_updated.connect(
                lambda progress, message, label=stage_label: self._relay_stage_status(
                    label, message, progress
                )
            )

        self._stage_sections_by_key = {
            "displacement": StageSection(
                "Displacement",
                self.displacement_widget,
                parameter_panel=self._stage_parameter_panels_by_key.get("displacement"),
                actions={
                    "run": lambda: self._request_interactive_stage("displacement", "run"),
                    "preview": lambda: self._request_interactive_stage("displacement", "preview"),
                    "cancel": lambda: self._cancel_interactive_stage("displacement"),
                },
                action_states=lambda: self._interactive_action_states("displacement"),
                action_states_changed=self.displacement_widget.action_states_changed,
            ),
            "force": StageSection(
                "Force Analysis",
                self.force_widget,
                parameter_panel=self._stage_parameter_panels_by_key.get("force"),
                actions={
                    "run": lambda: self._request_interactive_stage("force", "run"),
                    "preview": lambda: self._request_interactive_stage("force", "preview"),
                    "cancel": lambda: self._cancel_interactive_stage("force"),
                },
                action_states=lambda: self._interactive_action_states("force"),
                action_states_changed=self.force_widget.action_states_changed,
                # The auto-λ actions moved off the header into the Force parameter panel's
                # method blocks (surfaced only where the selected method can use them); see
                # _create_stage_parameter_panels wiring action_requested to the controller.
            ),
            "stress": StageSection(
                "Stress Analysis",
                self.stress_widget,
                parameter_panel=self._stage_parameter_panels_by_key.get("stress"),
                actions={
                    "run": lambda: self._request_interactive_stage("stress", "run"),
                    "preview": lambda: self._request_interactive_stage("stress", "preview"),
                    "cancel": lambda: self._cancel_interactive_stage("stress"),
                },
                action_states=lambda: self._interactive_action_states("stress"),
                action_states_changed=self.stress_widget.action_states_changed,
                optional=True,
                # Stress needs an external mask, so it stays off until the user
                # opts in (D1).
                enabled=False,
            ),
        }
        self._stage_sections = list(self._stage_sections_by_key.values())
        self._apply_stage_accents()
        for key, section in self._stage_sections_by_key.items():
            if section.enable_btn is not None:
                section.enabled_changed.connect(
                    lambda _enabled, k=key: self._on_stage_enabled_changed(k)
                )
            section.spine.clicked.connect(lambda k=key: self._on_stage_node_clicked(k))

        # While a stage's controller has the UI frozen for a long-running op, flip
        # that stage's pill into 'running' so the header's run/cancel button shows
        # the Cancel control (the cancel handler is always wired). On unfreeze,
        # re-read disk truth so the dots settle back to done/ready/off.
        self._freeze_widgets_by_key = {
            "displacement": self.displacement_widget,
            "force": self.force_widget,
            "stress": self.stress_widget,
        }
        for key, stage_widget in self._freeze_widgets_by_key.items():
            stage_widget.controller.ui_frozen.connect(
                lambda frozen, k=key: self._on_stage_freeze(k, frozen)
            )

        container_layout.setSpacing(0)
        for section in self._stage_sections:
            container_layout.addWidget(section)
        container_layout.addStretch()

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self._setup_theme_selector(main_layout)
        self.setLayout(main_layout)

        self.connect_signals()
        self._pending_interactive_request = None
        self._interactive_request_generation = 0
        self._interactive_request_context = None
        self._interactive_retry_scheduled = False
        self.data_manager.add_change_callback(self._on_interactive_data_changed)
        self.refresh_stage_statuses()
        self._update_disclosure()

        # Pay torch's one-time import/CUDA/cuDNN init cost in the background now,
        # so the first displacement run doesn't stall waiting on it.
        warm_up_torch()

    # Title-row icons: larger and thinner-lined than the stage-header icons,
    # and drawn with the document-set's sharp square caps / miter joins.
    _TOOLBAR_ICON_SIZE = 22
    _TOOLBAR_ICON_STROKE = 1.5
    _TOOLBAR_ICON_CAP = "square"
    _TOOLBAR_ICON_JOIN = "miter"

    def _make_toolbar_button(self, icon_name: str, tooltip: str) -> QToolButton:
        """A compact, icon-only, auto-raised button for the title-row toolbar."""
        button = QToolButton()
        button.setIcon(
            stage_action_icon(
                icon_name,
                muted_accent(stage_accent("project")),
                size=self._TOOLBAR_ICON_SIZE,
                stroke_width=self._TOOLBAR_ICON_STROKE,
                linecap=self._TOOLBAR_ICON_CAP,
                linejoin=self._TOOLBAR_ICON_JOIN,
            )
        )
        button.setIconSize(QSize(self._TOOLBAR_ICON_SIZE, self._TOOLBAR_ICON_SIZE))
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    @staticmethod
    def _toolbar_divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        return divider

    @staticmethod
    def _toolbar_caption(text: str) -> QLabel:
        """A small muted group label preceding a cluster of toolbar buttons."""
        caption = QLabel(text)
        caption.setStyleSheet(caption_style())
        return caption

    def _confirm_discard(self) -> bool:
        """True if it's safe to clear the workspace (clean, no project open, or user said yes)."""
        if not self._dirty or not self._project_open:
            return True
        reply = QMessageBox.question(
            self,
            "Discard changes?",
            "This project has unsaved changes. Discard them?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _new_project(self) -> None:
        """Clear to an empty, open workspace (G1): no rows, default knobs."""
        if not self._confirm_discard():
            return
        self._applying_state = True
        try:
            self.parameter_manager.reset_all_parameters()
            self.experiments_list.set_records([])
            for key, section in self._stage_sections_by_key.items():
                if section.enable_btn is not None:
                    section.set_enabled(False)  # stress is off by default (D1)
            self.data_manager.set_output_dir(None)
        finally:
            self._applying_state = False
        self._project_open = True
        self._dirty = False
        self.refresh()
        self._update_disclosure()

    def _save_project(self) -> None:
        """Save the whole project — dataset + recipe — to one YAML (Save-as)."""
        import yaml

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save project", "project.yaml", "YAML Files (*.yaml *.yml)"
            )
            if not file_path:
                return
            if not file_path.lower().endswith((".yml", ".yaml")):
                file_path += ".yaml"
            config = build_series_config(
                self.experiments_list.experiment_records(),
                disabled_stages=self._disabled_stages(),
                processed_root=self.data_manager.output_dir,
            )
            config["format_version"] = PROJECT_FORMAT_VERSION
            config["parameters"] = self.parameter_manager.get_all_parameters()
            with open(file_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            self._dirty = False
            QMessageBox.information(self, "Success", "Project saved!")
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def _load_project(self) -> None:
        """Load a project bundle: dataset + run options + analysis parameters."""
        if not self._confirm_discard():
            return
        import yaml

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load project", "", "YAML Files (*.yaml *.yml)"
            )
            if not file_path:
                return
            with open(file_path) as f:
                config = yaml.safe_load(f) or {}
            self._applying_state = True
            try:
                self._apply_parameters(config.get("parameters", {}) or {})
                run_options = config.get("run_options", {}) or {}
                disabled = set(run_options.get("disabled_stages") or [])
                for key, section in self._stage_sections_by_key.items():
                    if section.enable_btn is not None:
                        section.set_enabled(key not in disabled)
                self.experiments_list.set_records(series_records(config))
            finally:
                self._applying_state = False
            self._project_open = True
            self._dirty = False
            self.refresh()
            self._update_disclosure()
            QMessageBox.information(self, "Success", "Project loaded!")
        except Exception as e:
            logger.error(f"Error loading project: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

    def _setup_theme_selector(self, layout):
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch()

        self.theme_btn = QToolButton()
        self.theme_btn.setText("◐")
        self.theme_btn.setObjectName("theme_selector_button")
        self.theme_btn.setPopupMode(QToolButton.InstantPopup)

        self.theme_menu = QMenu(self.theme_btn)
        self._theme_actions = {}
        for name in theme_names():
            action = self.theme_menu.addAction(name)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, theme_name=name: self._on_theme_selected(theme_name)
            )
            self._theme_actions[name] = action
        self.theme_btn.setMenu(self.theme_menu)
        self._sync_theme_menu_state()

        footer.addWidget(self.theme_btn)
        layout.addLayout(footer)

    def _on_theme_selected(self, name: str):
        set_active_theme(name)
        self._apply_stage_accents()
        self._sync_theme_menu_state()

    def _apply_stage_accents(self):
        """Accent each stage by its pipeline key (not its title slug), then
        blend neighbours into the spine. Titles like "Force Analysis" don't
        slugify to a ramp key, so keying off the dict is what keeps the rail a
        real gradient instead of collapsing to the fallback colour."""
        for key, section in self._stage_sections_by_key.items():
            section.set_accent(stage_accent(key))
        self._apply_spine_neighbours()
        toolbar_accent = muted_accent(stage_accent("project"))
        for button, icon_name in self._toolbar_icon_buttons:
            button.setIcon(
                stage_action_icon(
                    icon_name,
                    toolbar_accent,
                    size=self._TOOLBAR_ICON_SIZE,
                    stroke_width=self._TOOLBAR_ICON_STROKE,
                    linecap=self._TOOLBAR_ICON_CAP,
                    linejoin=self._TOOLBAR_ICON_JOIN,
                )
            )

    def _apply_spine_neighbours(self):
        """Give each stage's spine its neighbours' accents so the rail blends."""
        sections = self._stage_sections
        for i, section in enumerate(sections):
            above = sections[i - 1]._accent if i > 0 else section._accent
            below = sections[i + 1]._accent if i < len(sections) - 1 else section._accent
            section.set_accents(section._accent, above=above, below=below)

    def _sync_theme_menu_state(self):
        current = active_theme_name()
        for name, action in self._theme_actions.items():
            action.setChecked(name == current)
        self.theme_btn.setToolTip(f"Theme: {current}")

    def _create_stage_parameter_panels(self) -> dict[str, WorkflowParameterPanel]:
        """Create inline workflow parameter editors grouped by pipeline stage."""
        stage_sections = {
            "displacement": ("Displacement",),
            "force": ("Force",),
            "stress": ("Stress",),
        }
        return {
            key: WorkflowParameterPanel(self.parameter_manager, section_titles=titles)
            for key, titles in stage_sections.items()
        }

    def _wire_force_panel_actions(self):
        """Route the Force panel's in-block action buttons (GCV pick, Bayesian freeze) to the
        Force controller. Replaces the old header crosshair. Called once force_widget exists.

        Handlers resolve lazily (getattr at click time) so a stubbed force widget in tests, or
        a future button, does not need every handler present at wiring time."""
        panel = self._stage_parameter_panels_by_key.get("force")
        if panel is None:
            return
        handler_names = {
            "gcv_lambda": "gcv_action",
            "bayesian_freeze": "bayesian_action",
        }

        def _dispatch(key, names=handler_names):
            handler = getattr(self.force_widget, names.get(key, ""), None)
            if callable(handler):
                handler()

        panel.action_requested.connect(_dispatch)

    def _request_interactive_stage(self, stage, mode):
        self._interactive_request_generation += 1
        self._pending_interactive_request = None
        generation = self._interactive_request_generation
        self._interactive_request_context = (
            stage, mode, generation, self._active_experiment
        )
        try:
            self._interactive_stage_coordinator.request(stage, mode)
        finally:
            self._interactive_request_context = None

    def _invalidate_pending_interactive_request(self):
        self._interactive_request_generation += 1
        self._pending_interactive_request = None
        self._interactive_request_context = None


    def _cancel_interactive_stage(self, stage):
        self._invalidate_pending_interactive_request()
        coordinator = self._interactive_stage_coordinator
        has_chain = coordinator._target_stage is not None or coordinator._active_stage is not None
        coordinator.cancel(stage)
        if not has_chain:
            self._interactive_stage_adapters[stage].cancel()

    def _interactive_action_states(self, stage):
        """Keep stage actions reachable for a selected experiment."""
        widget = {
            "displacement": self.displacement_widget,
            "force": self.force_widget,
            "stress": self.stress_widget,
        }[stage]
        states = widget.action_states()
        section = getattr(self, "_stage_sections_by_key", {}).get(stage)
        if self._active_experiment and (section is None or section.status != "running"):
            states.update(preview=True, run=True)
        return states

    def _interactive_artifact(self, stage):
        result = getattr(self.data_manager, f"{stage}_results", None)
        if result is not None or not self._active_experiment:
            return result
        loader = {
            "displacement": self._ensure_displacement_resident,
            "force": self._ensure_force_resident,
        }.get(stage)
        if loader is not None and loader():
            return getattr(self.data_manager, f"{stage}_results", None)
        return None

    def _interactive_parameters(self, stage):
        getter_name = {
            "displacement": "get_displacement_parameters",
            "force": "get_fttc_parameters",
            "stress": "get_stress_parameters",
        }[stage]
        getter = getattr(self.parameter_manager, getter_name, None)
        return getter() if callable(getter) else self.parameter_manager.get_all_parameters()

    def _exec_stale_stage_prompt(self, stage):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Upstream data uses different parameters")
        box.setText(
            f"Existing {stage} data was calculated with different parameters. "
            "What would you like to do?"
        )
        recalculate = box.addButton(
            "Recalculate with current parameters", QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton("Use existing data", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(recalculate)
        box.exec()
        clicked = box.clickedButton()
        return clicked.text() if clicked is not None else "Cancel"

    def _prompt_for_stale_stage(self, stage):
        return {
            "Recalculate with current parameters": StaleChoice.RECALCULATE,
            "Use existing data": StaleChoice.REUSE,
            "Cancel": StaleChoice.CANCEL,
        }.get(self._exec_stale_stage_prompt(stage), StaleChoice.CANCEL)

    def _validate_interactive_sources(self, stage):
        raw_input_path = getattr(self.data_manager, "raw_input_path", lambda _slot: None)
        missing = []
        waiting = []
        if stage == "displacement":
            for label, slot, value in (
                ("reference image", "reference", self.data_manager.reference),
                ("bead stack", "beads", self.data_manager.bead_stack),
            ):
                if value is None:
                    (waiting if raw_input_path(slot) is not None else missing).append(label)
        elif stage == "stress" and self.data_manager.mask_stack is None:
            (waiting if raw_input_path("masks") is not None else missing).append("stress mask")
        if missing:
            QMessageBox.warning(
                self, "Missing source data",
                "Cannot calculate this stage without " + " and ".join(missing) + ".",
            )
            self._pending_interactive_request = None
            return False
        if waiting:
            context = self._interactive_request_context
            if context is not None:
                target, mode, generation, experiment = context
                self._pending_interactive_request = {
                    "target": target, "mode": mode,
                    "generation": generation, "experiment": experiment,
                }
            self.status_label.setText(
                "Waiting for " + " and ".join(waiting) + " to finish loading…"
            )
            return False
        return True

    def _on_interactive_data_changed(self):
        self.refresh()
        if self._pending_interactive_request is None or self._interactive_retry_scheduled:
            return
        self._interactive_retry_scheduled = True

        def retry():
            self._interactive_retry_scheduled = False
            pending, self._pending_interactive_request = (
                self._pending_interactive_request, None
            )
            if (
                pending is not None
                and pending["generation"] == self._interactive_request_generation
                and pending["experiment"] == self._active_experiment
            ):
                self._request_interactive_stage(
                    pending["target"], pending["mode"]
                )

        QTimer.singleShot(0, retry)

    def _set_interactive_progress(self, message):
        self.status_label.setText(message)


    def _stage_widgets(self):
        return [
            self.displacement_widget,
            self.force_widget,
            self.stress_widget,
        ]

    def refresh(self):
        """Single reconcile pass: update every stage widget, then statuses."""
        for widget in self._stage_widgets():
            update = getattr(widget, "_update_ui_state", None)
            if callable(update):
                update()
        self._refresh_pyramid_levels_display()
        self.refresh_stage_statuses()

    def _refresh_pyramid_levels_display(self):
        """Feed the displacement panel the loaded input frame's (h, w) so its read-only
        FFD Pyramid Levels shows the depth downscale + coarsest-size will actually build.
        Uses the bead stack (else the reference); (h, w) is the last two axes of either."""
        panel = self._stage_parameter_panels_by_key.get("displacement")
        if panel is None:
            return
        src = self.data_manager.bead_stack
        if src is None:
            src = self.data_manager.reference
        shape = tuple(src.shape[-2:]) if src is not None and getattr(src, "ndim", 0) >= 2 else None
        panel.set_input_shape(shape)

    def _update_disclosure(self) -> None:
        """Reveal only what the current state earns (G0/G1/G2 gated reveal).

        G0 (no project): only the header toolbars + the empty-state hint show.
        G1 (project open, no row selected): the experiments workspace + the
        shared status line. G2 (a row selected): additionally the pipeline
        context label and the four stage pills.
        """
        project_open = self._project_open
        tuning = project_open and self.experiments_list.active() is not None

        self.experiments_list.setVisible(project_open)
        self.status_label.setVisible(project_open)

        self._pipeline_context_label.setVisible(tuning)
        for section in self._stage_sections:
            section.setVisible(tuning)

    def _relay_stage_status(self, stage_label: str, message: str, progress: int = 0) -> None:
        """Render a stage's progress message in the one global status label (P2).

        Interactive runs also echo to the console so live mode prints the same
        progress batch does (worklist §1): batch redirects stdout through its
        TeeLogger and ``print()``s timestamped lines, while live stages report
        only via Qt signals. Mirroring the message to stdout here — in batch's
        ``[timestamp] message`` format — funnels every interactive stage through
        the same console path without disturbing the UI label or the batch
        run-log file. This path never runs under the TeeLogger (run-all reports
        via ``_on_batch_progress``), so there is no double-timestamping.

        ``progress`` (0-100, as emitted by ``progress_updated``) also drives the
        stage's spine node so a running stage's rail fill grows frame by frame
        instead of just sitting on a flat amber dot — a no-op while the stage
        isn't "running" (the spine itself ignores progress then).
        """
        self.status_label.setText(f"{stage_label} — {message}")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {stage_label} — {message}")
        section = self._stage_sections_by_key.get(stage_label.lower())
        if section is not None:
            section.set_progress(progress / 100.0)

    def _run_selected_experiments(self) -> None:
        """Run the selected rows through the pipeline, walking the rail (P4).

        The experiments list is the single run source: its records + the shared
        parameters build the run config, and stress is skipped when its on/off
        glyph is off (D1). A per-folder progress callback drives the live
        mini-rails as the batch advances.

        Only the currently-selected rows are run: the records are filtered down
        to ``ExperimentsList.selected_rows()`` (row order) before building the
        config. The button is disabled when nothing is selected, so this is a
        no-op guard rather than the normal path.

        Every run goes to a background process pool and is polled from a Qt
        timer (see :meth:`_run_selected_experiments_background`) -- the batch
        never runs on the GUI thread and never streams frames into the viewer
        live. ``num_workers`` (read once here from the experiments-list
        spinbox) only sizes that pool; ``1`` means a single background worker,
        not an in-process run. Results are inspected by reading the finished
        ``.ntfm`` from disk, exactly as a manual circle click does.
        """
        selected = set(self.experiments_list.selected_rows())
        records = [
            record
            for record in self.experiments_list.experiment_records()
            if record["path"] in selected
        ]
        if not records:
            return
        num_workers = self.experiments_list.num_workers()
        config = build_run_config(
            records,
            self.parameter_manager.get_all_parameters(),
            disabled_stages=self._disabled_stages(),
            processed_root=self.data_manager.output_dir,
            num_workers=num_workers,
        )
        self._run_selected_experiments_background(config)

    def _run_selected_experiments_background(self, config: dict) -> None:
        """The only run path: submit to a process pool, poll from a timer.

        Workers run headless in separate processes, so nothing streams into
        the viewer live. The per-stage detail rail does not fill frame by
        frame during a run; it catches up via
        ``refresh_stage_statuses`` whenever the followed position completes, or
        at the very end. This is the deliberate trade-off for keeping the GUI
        thread free throughout the run. A single-worker pool
        (``num_workers == 1``) is still a background process, not an in-process
        run.

        The viewer follows the currently-selected row, or the topmost folder
        if nothing is selected yet. Cancellation reuses the existing
        ``_cancel_run_selected`` wiring unchanged -- it calls
        ``self._active_batch.request_cancel()``, and ``self._active_batch`` is
        the same analyzer instance used here.
        """
        analyzer = BatchAnalysis(config, progress_callback=self._on_batch_progress, sink=None)
        self._active_batch = analyzer
        self.experiments_list.set_run_selected_active(True)

        try:
            plan = resolve_output_plan(config['root_folders'], config.get('processed_root'))
            for warning in plan.warnings:
                print(f"WARNING: {warning}")

            # Viewer follows the selected row, else the topmost folder (locked
            # product decision) -- only set it if nothing is already selected,
            # so an existing selection is respected.
            if self._active_experiment is None:
                self._active_experiment = config['root_folders'][0]
                self.experiments_list.follow_streaming(self._active_experiment)

            analyzer.start_parallel(plan, config['num_workers'])
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Run-selected failed")
            QMessageBox.critical(self, "Run selected", f"Batch run failed: {exc}")
            self._active_batch = None
            self.experiments_list.set_run_selected_active(False)
            self.refresh_stage_statuses()
            return

        timer = QTimer(self)
        timer.setInterval(150)

        def _poll():
            events, stage_events, finished = analyzer.poll_parallel_progress()
            for folder, stage, status, fraction in stage_events:
                self._on_batch_stage_progress(folder, stage, status, fraction)
            for folder, status in events:
                # self._active_experiment may have changed since the run
                # started (the user can click a different, already-finished
                # row at any time -- that goes through the existing,
                # unchanged active_changed -> _on_active_experiment_changed
                # path and is not this method's concern). Compare against the
                # CURRENT value, not a value captured once at run start, so
                # "follow" tracks whatever the user is looking at right now.
                if status == "done" and folder == self._active_experiment:
                    # The user is following this run, so a finished folder's
                    # results are decoded into the viewer for them (the live
                    # "follow" feature — distinct from manual selection, which
                    # loads no output until a circle is clicked). Status is
                    # already eager; this just brings the pixels on screen.
                    self._load_stage_results(folder, self._NTFM_STAGES)
                    self.refresh_stage_statuses()
            if finished:
                timer.stop()
                self._active_batch = None
                self.experiments_list.set_run_selected_active(False)
                self.refresh_stage_statuses()

        timer.timeout.connect(_poll)
        self._parallel_run_timer = timer  # keep a reference so it isn't garbage-collected
        timer.start()

    def _on_stage_persisted(self, stage_key: str) -> None:
        """A stage finished interactively: persist it, then reconcile the dots.

        Interactive runs are no longer preview-only — a finished stage writes the
        same ``.ntfm`` a batch run would, at the same canonical path, so the row
        dots (top) and the section dots (below) both read the one on-disk truth.
        """
        self._persist_active_experiment(stage_key)
        self.refresh()

    def _persist_active_experiment(self, stage_key: str) -> None:
        """Write the active experiment's results to disk.

        For displacement / force / stress: gathers results from memory and
        writes them through the shared ``.ntfm`` writer.

        A no-op when nothing is selected or the relevant arrays are absent.
        """
        path = self._active_experiment
        if not path:
            return

        try:
            labels = {}
            for record in self.experiments_list.experiment_records():
                if record.get("path") == path:
                    labels = dict(record.get("columns", {}))
                    break
            ntfm_path = experiment_ntfm_path(path, self.data_manager.output_dir)
            write_experiment_ntfm(
                ntfm_path,
                parameters=self.parameter_manager.get_all_parameters(),
                displacement_result=self.data_manager.displacement_results,
                force_result=self.data_manager.force_results,
                stress_result=self.data_manager.stress_results,
                mask=self.data_manager.mask_stack,
                folder=path,
                input_files=self.experiments_list.input_files_for(path),
                labels=labels,
            )
        except Exception as exc:
            logger.exception("Could not persist results for %s", path)
            self.status_label.setText(f"Save failed: {exc}")

    def _cancel_run_selected(self) -> None:
        """Ask the live batch to stop at the next folder boundary (P4 / item 1).

        The batch runs synchronously on the GUI thread; the per-folder progress
        callback pumps the event loop (``processEvents``), so this click is
        delivered between folders and the cooperative flag halts the next one.
        """
        if self._active_batch is not None:
            self._active_batch.request_cancel()
            self.status_label.setText("Batch — cancelling…")

    def _on_batch_stage_progress(
        self, folder: str, stage: str, status: str, fraction: float | None
    ) -> None:
        """Route one parallel-mode worker's real per-stage progress onto its row.

        The poll timer's per-tick stage events land here and go straight to
        that one folder's one mini-rail dot -- the only live per-stage feedback
        during a run, since the batch computes headless in background
        processes and streams nothing into the detail panel.
        """
        self.experiments_list.set_row_stage_progress(folder, stage, status, fraction)

    def _on_batch_progress(self, folder: str, status: str) -> None:
        """Live per-folder feedback for Run-selected: walk the rail, then refresh (P4).

        'done'/'error' name the one folder that just changed, so read *that*
        folder's real `.ntfm` status directly instead of rescanning every row
        in the list (the bulk `refresh_statuses` path is input-only/cheap and
        would otherwise paint this just-finished folder back to 'not_started').
        """
        from pathlib import Path

        name = Path(folder).name
        if status == "running":
            self.experiments_list.mark_running(folder)
            self.status_label.setText(f"Batch — running {name}")
        elif status == "done":
            self.experiments_list.apply_row_statuses(folder, self._experiment_stage_status(folder))
            self.status_label.setText(f"Batch — finished {name}")
        elif status == "error":
            self.experiments_list.apply_row_statuses(folder, self._experiment_stage_status(folder))
            self.status_label.setText(f"Batch — failed {name}")
        elif status == "cancelled":
            self.experiments_list.refresh_statuses()
            self.status_label.setText("Batch — cancelled")
        # Keep the rail repainting between folders during the in-process run.
        QApplication.processEvents()

    def _on_stage_freeze(self, key: str, frozen: bool) -> None:
        """Surface the Cancel control while a stage runs, then re-read disk truth.

        A frozen controller means a stage (run *or* preview) is in flight; pinning
        the pill to 'running' is what flips the header's run/cancel button to the
        wired Cancel. On unfreeze we refresh from disk so the dots settle.
        """
        section = self._stage_sections_by_key.get(key)
        if section is None:
            return
        if frozen:
            section.set_status("running")
        else:
            self.refresh_stage_statuses()

    def refresh_stage_statuses(self):
        # Each stage's spine node mirrors the same persisted-.ntfm truth as the
        # experiments-list row dots (top), so the two can never disagree for the
        # active experiment — including treating an all-NaN stage (e.g. a failed
        # force) as not-done in both. Status is eager: reading which measures a
        # `.ntfm` carries is a cheap header read (no pixel decode), so every
        # stage's node shows the on-disk truth without waiting for a click. With
        # no experiment selected there is no disk truth, so the in-memory verdict
        # (computed from the data manager) stands.
        disk_status = (
            self._experiment_stage_status(self._active_experiment)
            if self._active_experiment
            else None
        )
        for key, section in self._stage_sections_by_key.items():
            memory_status = compute_stage_status(
                self.data_manager, STAGE_DATA_ARTIFACTS[key]
            )
            status = disk_status.get(key, memory_status) if disk_status else memory_status
            section.set_status(status)
        self.experiments_list.refresh_statuses()
        self._refresh_aggregate_readiness()

    def _aggregate_ntfm_paths(self):
        """Resolve every committed experiment's ``.ntfm`` path (folder → container).

        Returns a list of ``(folder, ntfm_path)`` in table order. The folder name
        is the human label used in readiness/skip messages.
        """
        from napariTFM.utilities.batch_output import experiment_ntfm_path

        out = []
        for record in self.experiments_list.experiment_records():
            folder = record["path"]
            out.append(
                (folder, experiment_ntfm_path(folder, self.data_manager.output_dir))
            )
        return out

    def _refresh_aggregate_readiness(self) -> None:
        """Recompute how many experiments can pool and push it to the section.

        A header-only walk (``aggregate.partition_ready`` reads OME-XML series
        names, no pixel decode), so it is cheap to run on every status refresh.
        """
        from pathlib import Path

        from napariTFM.backend import aggregate

        pairs = self._aggregate_ntfm_paths()
        paths = [ntfm_path for _folder, ntfm_path in pairs]
        ready, skipped = aggregate.partition_ready(paths)
        supported = aggregate.supported_metrics(ready)
        # Map each container path back to its folder basename for the message.
        name_by_path = {ntfm_path: Path(folder).name for folder, ntfm_path in pairs}
        skipped_named = [
            (name_by_path.get(p, Path(p).parent.name), reason) for p, reason in skipped
        ]
        self.experiments_list.set_aggregate_readiness(
            len(ready), len(pairs), skipped_named, supported
        )

    def _on_pool_requested(self) -> None:
        """Pool every ready experiment's ``.ntfm`` into one tidy summary table.

        Runs on a napari ``thread_worker`` (ITASC parity) so the busy bar animates
        and the GUI stays responsive: the aggregator reads + reduces containers
        and writes ``summary.csv`` + ``provenance.json`` + ``schema.json`` into the
        ``TFM_aggregate`` bucket. Only the checked, supported metrics are written.
        """
        from napari.qt.threading import thread_worker

        from napariTFM.backend import aggregate
        from napariTFM.utilities.batch_output import aggregate_output_dir

        pairs = self._aggregate_ntfm_paths()
        if not pairs:
            return
        paths = [ntfm_path for _folder, ntfm_path in pairs]
        out_dir = aggregate_output_dir(
            [folder for folder, _ in pairs], self.data_manager.output_dir
        )
        metrics = self.experiments_list.selected_metrics()

        self.experiments_list.set_pool_active(True)

        @thread_worker
        def _work():
            return aggregate.pool_experiments(paths, out_dir, metrics=metrics or None)

        worker = _work()
        worker.returned.connect(lambda result: self._on_pool_done(result, out_dir))
        worker.errored.connect(self._on_pool_error)
        # Keep a reference so the worker isn't garbage-collected mid-run.
        self._pool_worker = worker
        worker.start()

    def _on_pool_done(self, result, out_dir) -> None:
        """Pool finished: list the written artifacts and note any skipped rows."""
        from pathlib import Path

        self.experiments_list.set_pool_active(False)
        if result.summary_path is None:
            self.experiments_list.set_aggregate_result(
                "No experiments were ready to pool (need force + mask).", status=None
            )
        else:
            artifacts = [
                ("summary.csv", str(result.summary_path)),
                ("schema.json", str(result.schema_path)),
                ("provenance.json", str(result.provenance_path)),
            ]
            text = f"Pooled {result.n_rows} rows into {out_dir}."
            if result.skipped:
                names = ", ".join(Path(p).parent.name for p, _ in result.skipped)
                text += f" Skipped (not ready): {names}."
            self.experiments_list.set_aggregate_result(
                text, artifacts=artifacts, status="done"
            )
        self._refresh_aggregate_readiness()

    def _on_pool_error(self, exc: Exception) -> None:
        """Pool failed: surface the (usually actionable) error."""
        self.experiments_list.set_pool_active(False)
        self.experiments_list.set_aggregate_result(
            f"Aggregate failed: {exc}", status="error"
        )
        self._refresh_aggregate_readiness()
        if isinstance(exc, ValueError):
            # Expected, actionable: colliding experiment ids.
            QMessageBox.warning(self, "Pool experiments", str(exc))
        else:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Pool experiments", f"Pooling failed: {exc}")

    def _on_stage_node_clicked(self, key: str) -> None:
        """Decode one stage's series into the viewer (display-only, on demand).

        Clicking a stage's spine circle is what pulls that stage's arrays out of
        the active experiment's `TFMresults.ome.tif` and streams them to the
        viewer — nothing else is loaded. Purely for display: calculations always
        re-read from disk, so no prerequisite stage needs to be resident. Status
        dots are already eager, so no status change is needed here.
        """
        if self._active_experiment:
            self._load_stage_for_display(self._active_experiment, key)

    def _on_row_stage_clicked(self, path: str, stage: str) -> None:
        """A list row's stage dot was clicked: show that experiment's stage.

        Selects the row if it isn't already active (loading its inputs, same as
        any selection), then decodes just the clicked stage's series into the
        viewer. Display-only, one series — the same action as clicking the
        matching spine circle below, reachable straight from the list.
        """
        if path != self._active_experiment:
            self.experiments_list.set_active(path)
        self._load_stage_for_display(path, stage)

    def _load_stage_for_display(self, path: str, stage: str) -> None:
        """Load one stage's output for viewing, narrating it in the status line.

        A circle click brings a stage's pixels into the viewer. The heavy part —
        decoding the OME-TIFF — runs off the GUI thread so the click never
        freezes; the napari layer mutations then happen back on the GUI thread
        when the decode returns. If the experiment's arrays are already resident
        in the LRU (a re-click, or a second stage of the one just loaded), the
        decode is skipped and display is applied inline, instantly. A newer
        click supersedes an in-flight decode via a monotonic token.
        """
        label = stage.capitalize()
        self._stage_load_token += 1
        token = self._stage_load_token

        cached = self._cached_stage_arrays(path)
        if cached is not None:
            self._finish_stage_display(path, stage, cached, token)
            return

        self.status_label.setText(f"Loading {label}…")
        self.status_label.repaint()
        self._supersede_stage_load_worker()
        worker = self._create_stage_load_worker(path, token)
        self._stage_load_worker = worker
        worker.returned.connect(
            lambda payload, _stage=stage: self._on_stage_arrays_loaded(payload, _stage)
        )
        worker.errored.connect(
            lambda exc, _path=path, _stage=stage, _token=token:
            self._on_stage_load_error(exc, _path, _stage, _token)
        )
        worker.start()

    def _supersede_stage_load_worker(self) -> None:
        """Drop any in-flight cold decode; its stale return is token-guarded anyway."""
        if self._stage_load_worker is not None:
            try:
                self._stage_load_worker.quit()
            except Exception:
                pass
            self._stage_load_worker = None

    def _create_stage_load_worker(self, path: str, token: int):
        """Decode the `.ntfm` off the GUI thread, returning ``(token, path, data)``.

        Uses the cache-free :meth:`_decode_stage_arrays` so it touches no napari
        or Qt state; the GUI-thread slot stores the result in the LRU and paints
        it.
        """
        return self._make_worker(
            lambda: (token, path, self._decode_stage_arrays(path))
        )

    def _make_worker(self, fn):
        """Wrap ``fn`` in a napari ``thread_worker`` and return the worker.

        The single seam for off-thread stage decoding: tests replace this with a
        synchronous runner so a node click resolves deterministically without a
        live Qt event loop.
        """
        from napari.qt.threading import thread_worker

        return thread_worker(fn)()

    def _on_stage_arrays_loaded(self, payload, stage: str) -> None:
        """GUI-thread continuation of a cold decode: cache + paint, if still current."""
        token, path, data = payload
        if token != self._stage_load_token:
            return  # a newer click superseded this decode
        self._stage_load_worker = None
        self._store_stage_arrays(path, data)
        self._finish_stage_display(path, stage, data, token)

    def _on_stage_load_error(self, exc, path: str, stage: str, token: int) -> None:
        logger.exception("Failed to load stage %s for %s: %s", stage, path, exc)
        if token != self._stage_load_token:
            return
        self._stage_load_worker = None
        self.status_label.setText(f"Could not load {stage.capitalize()}")

    def _finish_stage_display(self, path: str, stage: str, data, token: int) -> None:
        """Apply a decoded bundle to the viewer and report the outcome (GUI thread).

        Shared tail of both the instant (cached) and off-thread (cold) paths.
        No-ops if a newer click superseded this one, or if the user switched to
        a different experiment while a cold decode was in flight (guarding
        against a stale decode painting the wrong experiment's data).
        """
        if token != self._stage_load_token:
            return
        if path != self._active_experiment:
            return
        self._apply_stage_data(data, [stage])
        # Loading a stage makes its upstream inputs resident (see _apply_stage_data),
        # which can newly enable a downstream stage's Preview/Run. Refresh action
        # enablement explicitly: the streamed layer adds happen under viewer event
        # blockers, so the widgets' selection-driven refresh may not fire on its own.
        for widget in self._stage_widgets():
            update = getattr(widget, "_update_ui_state", None)
            if callable(update):
                update()
        # Report by what actually landed in the data manager, not merely that a
        # container existed: a container can hold force but no displacement, and
        # a click on the empty stage must say so rather than claim a load.
        label = stage.capitalize()
        shown = getattr(self.data_manager, f"{stage}_results", None) is not None
        self.status_label.setText(
            f"{label} loaded" if shown else f"No {label} output to show yet"
        )

    def _on_stage_enabled_changed(self, key: str) -> None:
        self.refresh_stage_statuses()

    def _disabled_stages(self) -> list[str]:
        return [
            key
            for key, section in self._stage_sections_by_key.items()
            if not section.is_enabled
        ]

    def _experiment_stage_status(self, path: str) -> dict[str, str]:
        """Per-output stage status for an experiment folder (P3).

        Reads the experiment's `.ntfm` for which measures actually carry data
        (`displacement`/`force`/`stress`), so a stage is 'done' only when *its*
        output is present — not merely because a container exists. Each stage's
        immediate predecessor being done makes it 'ready' (the single run-next
        frontier); upstream of that is 'not_started'. Disabled stages read
        'off' (stress is exempt from auto-skip per D1).

        This is eager and runs for every discovered row: reading which measures
        a `.ntfm` carries is a header-only walk of the OME-TIFF (no pixel decode)
        and is cached by `ntfm.populated_measures`, so painting real dots the
        moment folders land in the list is cheap. Decoding a stage's pixels into
        the viewer is the separate, click-driven step (`_load_stage_results`).
        """
        from pathlib import Path

        from napariTFM.utilities import ntfm as _ntfm
        from napariTFM.utilities.batch_output import RESULTS_FILENAME, experiment_output_dir

        folder = Path(path)
        # Resolve the .ntfm exactly where the batch writes it (and where an
        # interactive run persists it): the shared resolve_output_plan bucket.
        # Reading any other path is how the row dots silently went stale.
        out_dir = experiment_output_dir(path, self.data_manager.output_dir)
        ntfm_path = out_dir / RESULTS_FILENAME
        measures = _ntfm.populated_measures(ntfm_path)
        # Inputs live in the experiment folder under their discovery names.
        input_files = self.experiments_list.input_files_for(path) or {}
        beads_name = input_files.get("beads", "beads.tif")
        reference_name = input_files.get("reference", "reference.tif")
        inputs_ready = (folder / beads_name).exists() and (
            folder / reference_name
        ).exists()

        disabled = set(self._disabled_stages())

        def _status(stage: str) -> str:
            if stage in disabled:
                return "off"
            if stage in measures:
                return "done"
            # 'ready' when this stage's immediate input is available. Displacement
            # is the first stage now, so its input is the raw beads/reference.
            ready_when = {
                "displacement": inputs_ready,
                "force": "displacement" in measures,
                "stress": "force" in measures,
            }
            return "ready" if ready_when.get(stage, False) else "not_started"

        return {stage: _status(stage) for stage in PIPELINE_STAGES}

    # ntfm-backed pipeline stages, in dependency order. Every stage's persisted
    # output is a tidy-table measure in the `.ntfm`, so this is also the full
    # pipeline.
    _NTFM_STAGES = ("displacement", "force", "stress")

    def _stage_ntfm_path(self, path: str):
        """Resolve the `.ntfm` for an experiment folder, or ``None`` if absent."""
        from napariTFM.utilities.batch_output import RESULTS_FILENAME, experiment_output_dir

        ntfm_path = (
            experiment_output_dir(path, self.data_manager.output_dir)
            / RESULTS_FILENAME
        )
        return ntfm_path if ntfm_path.exists() else None

    @staticmethod
    def _stage_cache_key(ntfm_path):
        """(path, mtime_ns, size) identity for a `.ntfm` — changes on any rewrite."""
        st = ntfm_path.stat()
        return (str(ntfm_path), st.st_mtime_ns, st.st_size)

    def _cached_stage_arrays(self, path: str):
        """Return the decoded bundle for ``path`` if resident and current, else None.

        A pure cache peek: resolves the `.ntfm` identity and looks it up without
        ever decoding, so the interactive click path can short-circuit to an
        instant re-display. GUI-thread only (the cache is only mutated here and
        in :meth:`_read_stage_arrays`, both on the GUI thread).
        """
        try:
            ntfm_path = self._stage_ntfm_path(path)
            if ntfm_path is None:
                return None
            key = self._stage_cache_key(ntfm_path)
        except Exception:
            return None
        bundle = self._stage_arrays_cache.get(key)
        if bundle is not None:
            self._stage_arrays_cache.move_to_end(key)
        return bundle

    def _store_stage_arrays(self, path: str, bundle) -> None:
        """Insert a freshly decoded bundle into the LRU (GUI thread only)."""
        if bundle is None:
            return
        try:
            ntfm_path = self._stage_ntfm_path(path)
            if ntfm_path is None:
                return
            key = self._stage_cache_key(ntfm_path)
        except Exception:
            return
        self._stage_arrays_cache[key] = bundle
        self._stage_arrays_cache.move_to_end(key)
        while len(self._stage_arrays_cache) > _STAGE_ARRAY_CACHE_SIZE:
            self._stage_arrays_cache.popitem(last=False)

    def _read_stage_arrays(self, path: str):
        """Return the active experiment's decoded `.ntfm` bundle, caching it.

        Cache-first wrapper around :meth:`_decode_stage_arrays`: on a hit the
        whole OME-TIFF decode is skipped, so clicking a second stage of the same
        experiment (or re-clicking one) is instant. GUI-thread only — the
        off-thread decode path calls :meth:`_decode_stage_arrays` directly and
        lets the GUI thread store the result. Returns ``None`` if there's no
        persisted output yet, or it's unreadable.
        """
        cached = self._cached_stage_arrays(path)
        if cached is not None:
            return cached
        bundle = self._decode_stage_arrays(path)
        self._store_stage_arrays(path, bundle)
        return bundle

    def _decode_stage_arrays(self, path: str):
        """Decode an experiment's `.ntfm` into a stage-array bundle (no cache).

        Pure disk read + numpy/metadata assembly with no napari or Qt access, so
        it is safe to run on a background thread. Returns ``None`` if there's no
        persisted output yet, or it's unreadable.
        """
        import types

        from napariTFM.utilities import ntfm as _ntfm

        try:
            ntfm_path = self._stage_ntfm_path(path)
            if ntfm_path is None:
                return None

            # Array-native read: dense stage arrays plus the grid spacing / frame
            # interval recovered from the OME metadata — no tidy table on this
            # display-load path.
            arrays, container_grid_spacing, container_frame_interval, metadata = (
                _ntfm.read_series_ntfm(ntfm_path)
            )

            # Recover physical_scale from stored config (UnifiedParameters asdict).
            config = metadata.get("config", {})

            # Reconstruct each stage's parameter dataclass from the stored config
            # so the viewer's visualize_* methods (which read
            # parameters.downscale_factor, d_max/f_max/max_stress,
            # *_vector_stride, *_arrow_scale) work on a freshly-selected
            # experiment without a re-run. Rebuild UnifiedParameters exactly as
            # config_from_parameters does (filter to valid field names so unknown
            # keys are ignored and missing keys default), then derive per stage.
            # A malformed/empty config falls back to UnifiedParameters defaults.
            from dataclasses import fields as _fields

            from napariTFM.backend.parameter_dataclasses import UnifiedParameters

            try:
                _valid = {f.name for f in _fields(UnifiedParameters)}
                unified = UnifiedParameters(
                    **{k: v for k, v in (config or {}).items() if k in _valid}
                )
            except Exception:
                unified = UnifiedParameters()
            disp_params = unified.to_displacement_parameters()
            force_params = unified.to_fttc_parameters()
            stress_params = unified.to_stress_parameters()

            pixel_size = float(config.get("pixel_size", 0.0))
            downscale_factor = float(config.get("downscale_factor", 0))
            frame_interval = float(config.get("frame_interval", 0.0))

            # Prefer the config's pixel_size * downscale; fall back to the grid
            # spacing / frame interval the container carries in its OME metadata.
            if pixel_size > 0 and downscale_factor > 0:
                grid_spacing = pixel_size * downscale_factor
            else:
                grid_spacing = float(container_grid_spacing)
            if frame_interval <= 0:
                frame_interval = float(container_frame_interval)

            physical_scale = {
                "pixel_size": pixel_size,
                "grid_spacing": grid_spacing,
                "time_interval": frame_interval,
                "grid_spacing_units": "µm",
                "time_interval_units": "min",
            }

            return types.SimpleNamespace(
                arrays=arrays,
                physical_scale=physical_scale,
                disp_params=disp_params,
                force_params=force_params,
                stress_params=stress_params,
            )
        except Exception:
            logger.exception("Failed to load results from .ntfm for %s", path)
            return None

    @staticmethod
    def _array_present(arr) -> bool:
        """True when an array has at least one non-NaN value."""
        import numpy as np

        return arr is not None and not np.all(np.isnan(arr))

    def _set_displacement_result_data(self, data) -> bool:
        """Make the displacement result resident in memory — no viewer stream.

        Force (and Stress upstream of it) needs the displacement field as its input;
        when a downstream stage is decoded for display, that displacement is already
        in ``data``, so keeping it resident is free and is what lets Force's
        Preview/Run enable and re-compute without a separate decode. Returns True
        when a result was set. Display stays lazy: this paints nothing.
        """
        import types

        disp = data.arrays.get("displacement_field")
        if not self._array_present(disp):
            return False
        result = types.SimpleNamespace(
            displacement_field=disp,
            physical_scale=data.physical_scale,
            original_shape=disp.shape[1:3],
            displacement_field_shape=disp.shape[1:3],
            parameters=data.disp_params,
        )
        self.data_manager.set_displacement_results(result, dirty=False)
        return True

    def _apply_displacement_result(self, data) -> None:
        if not self._set_displacement_result_data(data):
            return
        disp = data.arrays.get("displacement_field")
        # Lazy display: only the current frame is built now; the rest render on
        # demand as the user scrubs, so the click doesn't glyph the whole stack.
        self.visualization_manager.display_vector_field(
            'displacement', disp,
            {
                'v_max': data.disp_params.d_max,
                'vector_stride': data.disp_params.disp_vector_stride,
                'arrow_scale': data.disp_params.disp_arrow_scale,
                'downscale_factor': data.disp_params.downscale_factor,
            },
        )

    def _set_force_result_data(self, data) -> bool:
        """Make the force result resident in memory — no viewer stream.

        Stress needs the force field as its input; keeping it resident when Stress is
        decoded (its force input is already in ``data``) lets Stress Preview/Run work
        without a re-decode. Returns True when a result was set. Paints nothing.
        """
        import types

        force = data.arrays.get("force_field")
        if not self._array_present(force):
            return False
        result = types.SimpleNamespace(
            force_field=force,
            physical_scale=data.physical_scale,
            original_shape=force.shape[1:3],
            force_shape=force.shape[1:3],
            parameters=data.force_params,
        )
        self.data_manager.set_force_results(result, dirty=False)
        return True

    def _apply_force_result(self, data) -> None:
        if not self._set_force_result_data(data):
            return
        force = data.arrays.get("force_field")
        # Lazy display: build the current frame now, the rest on scrub.
        self.visualization_manager.display_vector_field(
            'force', force,
            {
                'v_max': data.force_params.f_max,
                'vector_stride': data.force_params.force_vector_stride,
                'arrow_scale': data.force_params.force_arrow_scale,
                'downscale_factor': data.force_params.downscale_factor,
            },
        )

    def _apply_stress_result(self, data) -> None:
        import types

        stress = data.arrays.get("stress_tensor")
        if not self._array_present(stress):
            return
        result = types.SimpleNamespace(
            stress_tensor=stress,
            physical_scale=data.physical_scale,
            original_shape=stress.shape[1:3],
            stress_shape=stress.shape[1:3],
            parameters=data.stress_params,
        )
        self.data_manager.set_stress_results(result, dirty=False)
        # Stress visualization upscales by the force grid's downscale factor.
        # Read it from the parsed file params, not `data_manager.force_results`,
        # so stress displays correctly even when its circle is clicked on its
        # own (force never loaded into memory) — the two share one config.
        stress_downscale = getattr(data.force_params, "downscale_factor", 1)
        self.visualization_manager.begin_stress_stream(
            num_frames=stress.shape[0],
            max_stress=data.stress_params.max_stress,
            downscale_factor=stress_downscale,
        )
        for frame_index in range(stress.shape[0]):
            self.visualization_manager.stream_stress_frame(
                frame_index, stress[frame_index]
            )

    def _load_stage_results(self, path: str, stages) -> list:
        """Decode the requested stages' persisted output into the viewer.

        Returns the list of stage keys actually loaded (used by callers/tests to
        confirm what was decoded).

        Reads the ntfm-backed table once (see `_read_stage_arrays`) but applies
        only the requested stages, so clicking one stage's circle never also
        streams a stage nobody asked to see. Display-only — calculations always
        re-read from disk, so no prerequisite stage is pulled in.
        """
        ntfm_stages = [s for s in self._NTFM_STAGES if s in stages]
        if not ntfm_stages:
            return []
        data = self._read_stage_arrays(path)
        return self._apply_stage_data(data, ntfm_stages)

    def _apply_stage_data(self, data, ntfm_stages) -> list:
        """Stream/resident-load an already-decoded bundle for the given stages.

        The apply half of :meth:`_load_stage_results`, split out so the
        interactive click path can decode off-thread and then call this on the
        GUI thread (napari layer mutations must run there). ``data`` may be
        ``None`` (no persisted output), in which case nothing is loaded.
        """
        if data is None:
            return []
        want = set(ntfm_stages)
        # Keep each requested stage's upstream INPUTS resident in memory
        # (data only — no viewer stream, so display stays lazy). The arrays
        # are already in `data`, so this is free, and it is what lets a
        # downstream stage's Preview/Run enable after only its own circle was
        # clicked: displacement is Force's input, force is Stress's.
        resident = set()
        if "force" in want:
            resident.add("displacement")
        if "stress" in want:
            resident.update({"displacement", "force"})
        # Applied in dependency order (displacement → force → stress) so that
        # set_displacement_results' downstream-invalidation never clears a
        # stage this same call is about to (re)set.
        if "displacement" in want:
            self._apply_displacement_result(data)
        elif "displacement" in resident:
            self._set_displacement_result_data(data)
        if "force" in want:
            self._apply_force_result(data)
        elif "force" in resident:
            self._set_force_result_data(data)
        if "stress" in want:
            self._apply_stress_result(data)
        return list(ntfm_stages)

    def _displacement_available(self) -> bool:
        """Whether displacement is usable as Force's input — resident in memory, or the
        displacement stage is done on disk for the active experiment (so it can be
        pulled in on demand). Lets Force's Preview/Run enable straight from the disk
        status; the `.ntfm` check is header-only (`populated_measures`), no decode.
        """
        if self.data_manager.displacement_results is not None:
            return True
        if not self._active_experiment:
            return False
        from napariTFM.utilities import ntfm as _ntfm
        from napariTFM.utilities.batch_output import RESULTS_FILENAME, experiment_output_dir

        out_dir = experiment_output_dir(self._active_experiment, self.data_manager.output_dir)
        return "displacement" in _ntfm.populated_measures(out_dir / RESULTS_FILENAME)

    def _ensure_displacement_resident(self) -> bool:
        """Pull the active experiment's displacement off disk into memory (data-only,
        no viewer stream) if not already resident, so the Force solver — which reads
        the displacement field from memory — can run. Returns True if resident after.

        Decodes on the calling (GUI) thread, like a stage-circle click; the Preview/
        Run that triggers it is a compute the user just asked for. A no-op once
        resident (e.g. the Force circle was viewed, which already keeps it resident).
        """
        if self.data_manager.displacement_results is not None:
            return True
        if not self._active_experiment:
            return False
        data = self._read_stage_arrays(self._active_experiment)
        if data is None:
            return False
        return self._set_displacement_result_data(data)

    def _force_available(self) -> bool:
        """Whether force is usable as Stress's input — resident in memory, or the
        force stage is done on disk for the active experiment (so it can be pulled
        in on demand). Lets Stress's Preview/Run enable straight from the disk
        status; the `.ntfm` check is header-only (`populated_measures`), no decode.
        """
        if self.data_manager.force_results is not None:
            return True
        if not self._active_experiment:
            return False
        from napariTFM.utilities import ntfm as _ntfm
        from napariTFM.utilities.batch_output import RESULTS_FILENAME, experiment_output_dir

        out_dir = experiment_output_dir(self._active_experiment, self.data_manager.output_dir)
        return "force" in _ntfm.populated_measures(out_dir / RESULTS_FILENAME)

    def _ensure_force_resident(self) -> bool:
        """Pull the active experiment's force off disk into memory (data-only,
        no viewer stream) if not already resident, so the Stress solver — which reads
        the force field from memory — can run. Returns True if resident after.

        Decodes on the calling (GUI) thread, like a stage-circle click; the Preview/
        Run that triggers it is a compute the user just asked for. A no-op once
        resident (e.g. the Stress circle was viewed, which already keeps it resident).
        """
        if self.data_manager.force_results is not None:
            return True
        if not self._active_experiment:
            return False
        data = self._read_stage_arrays(self._active_experiment)
        if data is None:
            return False
        return self._set_force_result_data(data)

    def _on_active_experiment_changed(self, path: str) -> None:
        self._invalidate_pending_interactive_request()
        self._active_experiment = path or None
        # Invalidate any in-flight cold stage decode: bumping the token (and
        # dropping the worker) means a decode started for the previous
        # experiment can't paint into this one when it returns. _finish_stage_display
        # also guards on the active path, so this is belt-and-braces.
        self._stage_load_token += 1
        self._supersede_stage_load_worker()
        # A single selection mutates the data manager several times in a row
        # (clear results, repoint inputs, load the mask). Each mutation would
        # otherwise drive a full ``refresh`` — a whole-list on-disk status walk —
        # so one click walked the list several times over. Coalesce the burst
        # into one notification. The async input load fires its own (separate,
        # later) notifications, which is correct. ``batch_changes`` is defensive
        # here so test stubs without it still work.
        batch = getattr(self.data_manager, "batch_changes", None)
        with (batch() if callable(batch) else nullcontext()):
            # Switching experiments drops the previous one's in-memory results so
            # they can never be persisted into the newly selected experiment's
            # .ntfm. The dots fall back to the new experiment's on-disk truth
            # (which may already read "done" from a prior batch/interactive run).
            self.data_manager.clear_generated_results()
            if self._active_experiment is None:
                self._pipeline_context_label.setText("Pipeline")
                self.data_manager.set_active_inputs(None, {})
                self.visualization_manager.set_display_reference_shape(None)
            else:
                from pathlib import Path

                self._pipeline_context_label.setText(
                    f"Pipeline · tuning ▸ {Path(self._active_experiment).name}"
                )
                # Point the raw-input disk check at the selected experiment so the
                # displacement input dots read green from its discovery files.
                input_files = self.experiments_list.input_files_for(self._active_experiment)
                self.data_manager.set_active_inputs(self._active_experiment, input_files)
                # And actually load those files from disk into memory + the viewer,
                # so Preview and Run (which need the arrays loaded) work on
                # selection. Displacement is the first stage and now owns raw-input
                # loading.
                self.displacement_widget.load_input_files(self._active_experiment, input_files)
                # No output series is decoded on selection: the row/section dots
                # already show the on-disk status eagerly, and a stage's pixels
                # only stream in when its circle is clicked (calculations re-read
                # from disk regardless, so nothing must be resident to run).
                # The bead image's xy size (read cheaply from disk — the bead arrays
                # stream in asynchronously and aren't in memory yet) is the display
                # reference: every analysis-grid field (displacement/force/stress) and
                # the mask are scaled to it, so they overlay the beads at original
                # resolution regardless of the grid or downscale dial they were
                # computed at.
                beads_shape = self.displacement_widget.peek_input_xy_shape(
                    self._active_experiment, input_files, "beads"
                )
                self.visualization_manager.set_display_reference_shape(beads_shape)
                # The mask is an external Stress input; load the discovered
                # masks.tif from disk into memory too, so Stress Run/Preview enable
                # on selection the same way beads/reference do — no manual layer
                # load required.
                mask_name = input_files.get("masks")
                if mask_name:
                    self.stress_widget.load_mask_from_file(
                        Path(self._active_experiment) / mask_name, beads_shape=beads_shape
                    )
        self._update_disclosure()
        # The newly selected experiment's on-disk stage availability may enable or
        # disable a stage's Preview/Run (e.g. Force enables when displacement is on
        # disk). Nothing else re-evaluates enablement on a bare selection, so do it
        # here — cheap header-only `.ntfm` checks.
        for widget in self._stage_widgets():
            update = getattr(widget, "_update_ui_state", None)
            if callable(update):
                update()

    def _on_experiments_changed(self) -> None:
        if not self._applying_state:
            self._dirty = True
        # Keep the pool-readiness count in step with the table (commit/delete):
        # a header-only walk, so cheap to run whenever the row set changes.
        self._refresh_aggregate_readiness()

    def _reset_parameters(self):
        """Reset parameters to default values and notify all widgets."""
        try:
            reply = QMessageBox.question(
                self,
                "Confirm",
                "Are you sure you want to reset all parameters to default values?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Reset parameters
                self.parameter_manager.reset_all_parameters()
                self.refresh()

        except Exception as e:
            logger.error(f"Error resetting parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to reset parameters: {str(e)}")

    def _apply_parameters(self, params) -> None:
        """Apply a name→value dict of analysis knobs onto the shared manager.

        Unknown names are skipped (forward/backward-compatible files) and
        ``registration_mode`` is normalised.
        """
        if not isinstance(params, dict):
            return
        valid = set(self.parameter_manager.get_all_parameters())
        for name, value in params.items():
            if name not in valid:
                continue
            try:
                if name == "registration_mode" and isinstance(value, str):
                    value = value.lower()
                self.parameter_manager.set_parameter(name, value)
            except Exception as exc:
                logger.warning("Skipped parameter %s: %s", name, exc)

    def _save_params(self):
        """Save the analysis knobs as a portable ``tfm_params`` preset (no paths).

        The preset is the reusable "how to analyze" recipe; the dataset it gets
        applied to lives in the separate experiment-series file.
        """
        import yaml

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save parameters preset",
                "tfm_params.yaml",
                "YAML Files (*.yaml *.yml)",
            )
            if not file_path:
                return
            if not file_path.lower().endswith((".yml", ".yaml")):
                file_path += ".yaml"
            preset = {
                "format_version": 1,
                "parameters": self.parameter_manager.get_all_parameters(),
            }
            with open(file_path, "w") as f:
                yaml.safe_dump(preset, f, default_flow_style=False)
            QMessageBox.information(self, "Success", "Parameters preset saved!")
        except Exception as e:
            logger.error(f"Error saving parameters preset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")

    def _load_params(self):
        """Load a ``tfm_params`` preset and apply it to every stage's knobs."""
        import yaml

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load parameters preset",
                "",
                "YAML Files (*.yaml *.yml)",
            )
            if not file_path:
                return
            with open(file_path) as f:
                preset = yaml.safe_load(f) or {}
            # Accept the versioned {"parameters": {...}} shape and, for old
            # files, a bare flat parameter dict.
            params = preset.get("parameters") if isinstance(preset, dict) else None
            if not isinstance(params, dict):
                params = preset if isinstance(preset, dict) else {}
            self._apply_parameters(params)
            self.refresh()
            QMessageBox.information(self, "Success", "Parameters preset loaded!")
        except Exception as e:
            logger.error(f"Error loading parameters preset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def connect_signals(self):
        """Connect signals between components"""
        # A finished stage persists to the active experiment's .ntfm (auto-save),
        # then refreshes so both dot rows reflect the new on-disk truth.
        self.displacement_widget.displacement_calculated.connect(
            lambda *_: self._on_stage_persisted("displacement")
        )
        self.force_widget.force_calculated.connect(
            lambda *_: self._on_stage_persisted("force")
        )
        self.stress_widget.stress_calculated.connect(
            lambda *_: self._on_stage_persisted("stress")
        )

        # Connect parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Propagate parameter edits to the widgets that display them."""
        if not self._applying_state:
            self._dirty = True
        if param_name in ("pixel_size", "frame_interval"):
            # Calibration affects every stage's displayed/derived values.
            for widget in self._stage_widgets():
                update = getattr(widget, "_update_ui_state", None)
                if callable(update):
                    update()
        elif param_name.startswith("force_"):
            self.force_widget._update_ui_state()

