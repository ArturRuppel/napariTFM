"""Tests for the matplotlib batch visualization renderer (worklist §7).

The renderer uses matplotlib's Agg backend, so these run anywhere — no display,
no GL, no xvfb. They cover the shared vector math and that each ``save_*``
product writes an mp4 with the right frame count / distinct frames / component
gating.
"""
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

from napariTFM.backend.batch_visualizations import BatchVisualizationSaver
from napariTFM.utilities.vector_field import build_frame_vectors, upscale_field


# --- shared vector-field math ----------------------------------------------
def test_upscale_field_noop_at_unit_factor():
    field = np.random.rand(4, 5, 2).astype(np.float32)
    assert upscale_field(field, 1) is field


def test_build_frame_vectors_layout_and_colors():
    flow = np.zeros((20, 20, 2), np.float32)
    flow[..., 0] = 1.0
    vectors, colors = build_frame_vectors(flow, flow, stride=10, v_max=1.0)
    assert vectors.shape[1:] == (2, 2)
    assert colors.shape == (vectors.shape[0], 4)
    assert np.allclose(vectors[:, 1, 1], 1.0)   # direction x == U
    assert np.allclose(colors, colors[0])       # uniform magnitude -> uniform colour


# --- fixtures ---------------------------------------------------------------
def _disp_result(t=4, h=32, w=32):
    field = np.zeros((t, h, w, 2), np.float32)
    for i in range(t):
        field[i, ..., 0] = (i + 1) * 0.5  # grows per frame -> distinct frames
    params = SimpleNamespace(downscale_factor=1, d_max=2.0,
                             disp_arrow_scale=1.0, disp_vector_stride=8)
    return SimpleNamespace(displacement_field=field, parameters=params)


def _force_result(t=4, h=32, w=32):
    field = np.zeros((t, h, w, 2), np.float32)
    for i in range(t):
        field[i, ..., 1] = (i + 1) * 100.0
    params = SimpleNamespace(downscale_factor=1, f_max=500.0,
                             force_arrow_scale=1.0, force_vector_stride=8)
    return SimpleNamespace(force_field=field, parameters=params)


def _stress_result(t=4, h=32, w=32):
    tensor = np.zeros((t, h, w, 2, 2), np.float32)
    for i in range(t):
        tensor[i, ..., 0, 0] = (i + 1) * 0.3
        tensor[i, ..., 1, 1] = -(i + 1) * 0.2
    return SimpleNamespace(stress_tensor=tensor, parameters=SimpleNamespace(max_stress=1.0))


def _frames(path):
    return np.array(imageio.mimread(str(path)))


# --- product tests ----------------------------------------------------------
def test_displacement_movie_written_with_distinct_frames(tmp_path):
    saver = BatchVisualizationSaver(str(tmp_path))
    saver.save_displacement_visualization(_disp_result(), fps=5)

    out = tmp_path / "figures" / "displacement_map.mp4"
    assert out.exists() and out.stat().st_size > 0
    frames = _frames(out)
    assert len(frames) == 4
    assert any(not np.array_equal(frames[0], f) for f in frames[1:])


def test_force_movie_written(tmp_path):
    saver = BatchVisualizationSaver(str(tmp_path))
    saver.save_force_visualization(_force_result(), fps=5)
    assert (tmp_path / "figures" / "force_map.mp4").exists()


def test_force_cell_overlay_written(tmp_path):
    saver = BatchVisualizationSaver(str(tmp_path))
    cells = np.random.rand(4, 32, 32).astype(np.float32)
    saver.save_force_cell_overlay(_force_result(), cells, fps=5)
    assert (tmp_path / "figures" / "force_cell_overlay.mp4").exists()


def test_stress_movies_respect_component_flags(tmp_path):
    saver = BatchVisualizationSaver(str(tmp_path))
    saver.save_stress_visualization(_stress_result(), plot_sigma_xx=True,
                                    plot_sigma_yy=False, plot_normal_stress=True, fps=5)
    figs = tmp_path / "figures"
    assert (figs / "sigma_xx.mp4").exists()
    assert (figs / "normal_stress.mp4").exists()
    assert not (figs / "sigma_yy.mp4").exists()  # disabled component skipped


def test_single_frame_inputs_are_handled(tmp_path):
    """A 3D displacement field (single frame) renders a one-frame movie."""
    saver = BatchVisualizationSaver(str(tmp_path))
    res = _disp_result(t=1)
    res = SimpleNamespace(displacement_field=res.displacement_field[0],  # (y, x, 2)
                          parameters=res.parameters)
    saver.save_displacement_visualization(res, fps=5)
    assert (tmp_path / "figures" / "displacement_map.mp4").exists()
