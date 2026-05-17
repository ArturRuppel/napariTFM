import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_targets_current_napari_qt_stack():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    dependencies = pyproject["project"]["dependencies"]
    normalized_dependencies = [dependency.lower() for dependency in dependencies]

    assert "napari[pyqt]>=0.7.0" in normalized_dependencies
    assert not any(
        dependency.startswith(("pyqt5", "pyqt6", "pyside2", "pyside6"))
        for dependency in normalized_dependencies
    )
    assert "console_scripts" not in pyproject["project"].get("entry-points", {})


def test_widget_constructs_with_pyqt6_qtpy_backend():
    if importlib.util.find_spec("PyQt6") is None:
        pytest.skip("PyQt6 is not installed in this environment")

    script = textwrap.dedent(
        """
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ["QT_API"] = "pyqt6"

        import napari
        from qtpy import API_NAME
        from qtpy.QtWidgets import QApplication
        from napariTFM.widgets._widget import napariTFMWidget

        app = QApplication.instance() or QApplication([])
        viewer = napari.Viewer(show=False)
        widget = napariTFMWidget(viewer)
        widget.show()
        app.processEvents()

        assert API_NAME == "PyQt6", API_NAME
        assert len(widget.batch_widget.parameter_spins) == 0
        assert "preprocessing" in widget._stage_sections_by_key

        viewer.close()
        print(f"qtpy={API_NAME}")
        """
    )

    env = os.environ.copy()
    env["QT_API"] = "pyqt6"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "qtpy=PyQt6" in result.stdout
