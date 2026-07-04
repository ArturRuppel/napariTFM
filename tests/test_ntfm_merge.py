"""Tests for tidy-table merge: prior-run stage preservation on re-write."""

import numpy as np

from napariTFM.utilities import ntfm
from napariTFM.backend.ntfm_writer import write_experiment_ntfm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _disp_result(arr, scale=None):
    scale = scale or {"grid_spacing": 1.0, "time_interval": 1.0}
    return _FakeResult(displacement_field=arr, physical_scale=scale)


def _force_result(arr, scale=None):
    scale = scale or {"grid_spacing": 1.0, "time_interval": 1.0}
    return _FakeResult(force_field=arr, physical_scale=scale)


# ---------------------------------------------------------------------------
# 1. merge_arrays: fills absent stage from old
# ---------------------------------------------------------------------------

def _arrays(displacement_field=None, force_field=None, stress_tensor=None, mask=None):
    return {
        "displacement_field": displacement_field,
        "force_field": force_field,
        "stress_tensor": stress_tensor,
        "mask": mask,
    }


def test_merge_fills_absent_displacement_from_old():
    """A force-only new dict should inherit displacement from old."""
    rng = np.random.default_rng(42)
    nt, ny, nx = 2, 3, 4
    disp = rng.standard_normal((nt, ny, nx, 2))
    force = rng.standard_normal((nt, ny, nx, 2)) * 10.0

    merged = ntfm.merge_arrays(_arrays(force_field=force), _arrays(displacement_field=disp))

    np.testing.assert_allclose(merged["displacement_field"], disp)
    np.testing.assert_allclose(merged["force_field"], force)  # new force untouched


def test_merge_does_not_resurrect_stale_force_under_fresh_displacement():
    """A fresh displacement in `new` makes any old force stale (B-3): it must NOT
    be resurrected — otherwise disk pairs a new displacement with a force computed
    from the *prior* displacement. This is the direction that was silently wrong."""
    rng = np.random.default_rng(10)
    nt, ny, nx = 2, 3, 3
    disp = rng.standard_normal((nt, ny, nx, 2))
    force = rng.standard_normal((nt, ny, nx, 2)) * 5.0

    merged = ntfm.merge_arrays(_arrays(displacement_field=disp), _arrays(force_field=force))

    np.testing.assert_allclose(merged["displacement_field"], disp)  # fresh displacement wins
    stale_force = merged.get("force_field")
    assert stale_force is None or np.all(np.isnan(stale_force)), (
        "stale force must not be resurrected next to a freshly-written displacement"
    )


def test_merge_does_not_resurrect_stale_stress_under_fresh_force():
    """Symmetric to the above one level down: a fresh force makes an old stress
    stale, so it is not resurrected (B-3)."""
    rng = np.random.default_rng(11)
    nt, ny, nx = 2, 3, 3
    force = rng.standard_normal((nt, ny, nx, 2))
    stress = rng.standard_normal((nt, ny, nx, 4))

    merged = ntfm.merge_arrays(
        _arrays(force_field=force), _arrays(stress_tensor=stress)
    )

    np.testing.assert_allclose(merged["force_field"], force)
    stale_stress = merged.get("stress_tensor")
    assert stale_stress is None or np.all(np.isnan(stale_stress)), (
        "stale stress must not be resurrected next to a freshly-written force"
    )


def test_merge_preserves_upstream_but_drops_stale_downstream_together():
    """Force-only resume: displacement (upstream, absent) is preserved, while an
    old stress (downstream of the fresh force) is dropped — both in one merge."""
    rng = np.random.default_rng(12)
    nt, ny, nx = 2, 3, 3
    disp = rng.standard_normal((nt, ny, nx, 2))
    force_new = rng.standard_normal((nt, ny, nx, 2))
    stress_old = rng.standard_normal((nt, ny, nx, 4))

    merged = ntfm.merge_arrays(
        _arrays(force_field=force_new),  # resume: only force recomputed
        _arrays(displacement_field=disp, stress_tensor=stress_old),
    )

    np.testing.assert_allclose(merged["displacement_field"], disp)  # upstream preserved
    np.testing.assert_allclose(merged["force_field"], force_new)
    stale_stress = merged.get("stress_tensor")
    assert stale_stress is None or np.all(np.isnan(stale_stress)), (
        "stress computed from the prior force must not survive a force re-run"
    )


# ---------------------------------------------------------------------------
# 2. merge_arrays: does NOT overwrite stage present in new
# ---------------------------------------------------------------------------

def test_merge_does_not_overwrite_displacement_present_in_new():
    """When new already has displacement, old's values must not override it."""
    rng = np.random.default_rng(7)
    nt, ny, nx = 2, 3, 3
    disp_new = rng.standard_normal((nt, ny, nx, 2))
    disp_old = rng.standard_normal((nt, ny, nx, 2)) + 1000.0  # clearly different

    merged = ntfm.merge_arrays(
        _arrays(displacement_field=disp_new), _arrays(displacement_field=disp_old)
    )

    np.testing.assert_allclose(merged["displacement_field"], disp_new)
    assert not np.any(merged["displacement_field"] > 100.0)


def test_merge_does_not_overwrite_force_present_in_new():
    """When new has force, old's force must not override it."""
    rng = np.random.default_rng(13)
    nt, ny, nx = 2, 3, 3
    force_new = rng.standard_normal((nt, ny, nx, 2)) * 1.0
    force_old = rng.standard_normal((nt, ny, nx, 2)) * 1000.0

    merged = ntfm.merge_arrays(_arrays(force_field=force_new), _arrays(force_field=force_old))

    np.testing.assert_allclose(merged["force_field"], force_new)


def test_merge_cleared_mask_does_not_resurrect():
    """A re-write with no mask must clear the stored mask, not restore the old
    one — an absent/all-zero mask only ever means 'no mask supplied', never a
    deliberately-empty region."""
    nt, ny, nx = 2, 3, 3
    force = np.ones((nt, ny, nx, 2))
    old_mask = np.ones((nt, ny, nx), dtype=np.int64)  # a real mask was saved before

    merged = ntfm.merge_arrays(
        _arrays(force_field=force),  # new run: force but no mask
        _arrays(force_field=force, mask=old_mask),
    )

    assert merged["mask"] is None  # new mask (absent) wins; old is not resurrected


# ---------------------------------------------------------------------------
# 3. merge_arrays: grid mismatch → returns new unchanged, no crash
# ---------------------------------------------------------------------------

def test_merge_grid_mismatch_different_row_extent_returns_new_unchanged():
    """Different (T, Y, X) → merge skipped, new returned unchanged."""
    nt, ny, nx = 2, 3, 4
    disp = np.ones((nt, ny, nx, 2))
    new = _arrays(displacement_field=disp)
    old = _arrays(displacement_field=np.ones((nt, ny + 1, nx, 2)))  # extra row

    merged = ntfm.merge_arrays(new, old)

    assert merged is new


def test_merge_grid_mismatch_no_crash():
    """Grid mismatch must never raise; it just logs and returns new."""
    new = _arrays(displacement_field=np.ones((1, 3, 3, 2)))
    old = _arrays(displacement_field=np.ones((1, 4, 3, 2)))

    merged = ntfm.merge_arrays(new, old)
    assert merged is not None
    np.testing.assert_allclose(merged["displacement_field"], new["displacement_field"])


# ---------------------------------------------------------------------------
# 4. Container: displacement-only write, then force-only second write preserves
#    displacement (via results_to_ntfm, the core integration path)
# ---------------------------------------------------------------------------

def test_results_to_ntfm_force_second_write_preserves_displacement(tmp_path):
    """Force-only second write must not erase previously-saved displacement."""
    rng = np.random.default_rng(0)
    nt, ny, nx = 2, 3, 3
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp_arr = rng.standard_normal((nt, ny, nx, 2))
    force_arr = rng.standard_normal((nt, ny, nx, 2)) * 10.0

    disp_result = _disp_result(disp_arr, scale)
    force_result = _force_result(force_arr, scale)

    ntfm_path = tmp_path / "exp.ntfm"

    # First write: displacement only
    ntfm.results_to_ntfm(ntfm_path, config={}, displacement_result=disp_result)
    assert ntfm.populated_measures(ntfm_path) == {"displacement"}

    # Second write: force only (merge_existing=True by default)
    ntfm.results_to_ntfm(ntfm_path, config={}, force_result=force_result)

    # Both stages must be reported as populated
    measures = ntfm.populated_measures(ntfm_path)
    assert "displacement" in measures, "displacement was erased by force-only write"
    assert "force" in measures

    # The actual displacement data must be intact
    df, _ = ntfm.read_ntfm(ntfm_path)
    arrays = ntfm.tidy_to_arrays(df)
    np.testing.assert_allclose(arrays["displacement_field"], disp_arr)


def test_results_to_ntfm_displacement_rerun_drops_stale_force_on_disk(tmp_path):
    """B-3 end-to-end: after displacement→force are on disk, re-running
    displacement (force invalidated → absent from the write) must leave the
    container with the new displacement and NO force — never the old force
    computed from the prior displacement."""
    rng = np.random.default_rng(3)
    nt, ny, nx = 2, 3, 3
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp_v1 = rng.standard_normal((nt, ny, nx, 2))
    force_v1 = rng.standard_normal((nt, ny, nx, 2)) * 10.0
    disp_v2 = rng.standard_normal((nt, ny, nx, 2)) + 100.0  # clearly different

    ntfm_path = tmp_path / "exp.ntfm"

    # Establish disp_v1 + force_v1 on disk.
    ntfm.results_to_ntfm(ntfm_path, config={}, displacement_result=_disp_result(disp_v1, scale))
    ntfm.results_to_ntfm(ntfm_path, config={}, force_result=_force_result(force_v1, scale))
    assert ntfm.populated_measures(ntfm_path) == {"displacement", "force"}

    # Re-run displacement only (mirrors set_displacement_results invalidating force).
    ntfm.results_to_ntfm(ntfm_path, config={}, displacement_result=_disp_result(disp_v2, scale))

    measures = ntfm.populated_measures(ntfm_path)
    assert "force" not in measures, "stale force was resurrected next to a fresh displacement"
    assert measures == {"displacement"}

    df, _ = ntfm.read_ntfm(ntfm_path)
    arrays = ntfm.tidy_to_arrays(df)
    np.testing.assert_allclose(arrays["displacement_field"], disp_v2)


# ---------------------------------------------------------------------------
# 5. Container: grid-mismatch on second write → no crash, new data wins
# ---------------------------------------------------------------------------

def test_results_to_ntfm_grid_mismatch_no_crash_new_data_wins(tmp_path):
    """A second write with a different grid must not crash; new data wins."""
    nt = 2
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp_arr = np.ones((nt, 3, 3, 2))
    # Different spatial grid for the second write
    force_arr = np.ones((nt, 4, 3, 2)) * 5.0

    disp_result = _disp_result(disp_arr, scale)
    force_result = _force_result(force_arr, scale)

    ntfm_path = tmp_path / "exp.ntfm"

    # First write
    ntfm.results_to_ntfm(ntfm_path, config={}, displacement_result=disp_result)

    # Second write with a different grid must not raise
    ntfm.results_to_ntfm(ntfm_path, config={}, force_result=force_result)

    # New data must be present in the file
    measures = ntfm.populated_measures(ntfm_path)
    assert "force" in measures


# ---------------------------------------------------------------------------
# 6. results_to_ntfm(..., merge_existing=False) → pure overwrite, old stage erased
# ---------------------------------------------------------------------------

def test_results_to_ntfm_merge_existing_false_erases_old_stage(tmp_path):
    """merge_existing=False must perform a pure overwrite, erasing prior stages."""
    rng = np.random.default_rng(55)
    nt, ny, nx = 2, 3, 3
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp_arr = rng.standard_normal((nt, ny, nx, 2))
    force_arr = rng.standard_normal((nt, ny, nx, 2)) * 10.0

    disp_result = _disp_result(disp_arr, scale)
    force_result = _force_result(force_arr, scale)

    ntfm_path = tmp_path / "exp.ntfm"

    # First write: displacement only
    ntfm.results_to_ntfm(ntfm_path, config={}, displacement_result=disp_result)
    assert "displacement" in ntfm.populated_measures(ntfm_path)

    # Second write: force only WITH merge_existing=False → displacement must be erased
    ntfm.results_to_ntfm(
        ntfm_path, config={}, force_result=force_result, merge_existing=False
    )

    measures = ntfm.populated_measures(ntfm_path)
    assert "displacement" not in measures, (
        "displacement should have been erased by merge_existing=False overwrite"
    )
    assert "force" in measures


# ---------------------------------------------------------------------------
# 7. write_experiment_ntfm end-to-end: force-only second write preserves displacement
# ---------------------------------------------------------------------------

def test_write_experiment_ntfm_preserves_displacement_on_force_only_rewrite(tmp_path):
    """Full stack: write_experiment_ntfm with force only must not erase displacement."""
    rng = np.random.default_rng(99)
    nt, ny, nx = 2, 3, 3
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp_arr = rng.standard_normal((nt, ny, nx, 2))
    force_arr = rng.standard_normal((nt, ny, nx, 2)) * 50.0

    disp_result = _disp_result(disp_arr, scale)
    force_result = _force_result(force_arr, scale)

    ntfm_path = tmp_path / "exp.ntfm"
    params = {"pixel_size": 0.1, "downscale_factor": 4}

    # First write: displacement only
    written = write_experiment_ntfm(
        ntfm_path,
        parameters=params,
        displacement_result=disp_result,
    )
    assert written == ntfm_path
    assert "displacement" in ntfm.populated_measures(ntfm_path)

    # Second write: force only — displacement_result=None, merge must preserve prior data
    write_experiment_ntfm(
        ntfm_path,
        parameters=params,
        force_result=force_result,
    )

    measures = ntfm.populated_measures(ntfm_path)
    assert "displacement" in measures, "displacement erased by force-only second write"
    assert "force" in measures

    # Verify actual array values are preserved
    df, _ = ntfm.read_ntfm(ntfm_path)
    arrays = ntfm.tidy_to_arrays(df)
    np.testing.assert_allclose(
        arrays["displacement_field"], disp_arr, err_msg="displacement_field not preserved"
    )
    np.testing.assert_allclose(
        arrays["force_field"], force_arr, err_msg="force_field not preserved"
    )
