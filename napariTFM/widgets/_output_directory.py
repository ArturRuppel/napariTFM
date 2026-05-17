from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget

from napariTFM.utilities.data_manager import DataManager


def ensure_output_dir_for_generated_artifacts(parent: QWidget | None, data_manager: DataManager) -> bool:
    """Ensure generated artifacts have a destination before auto-saving."""
    if data_manager.output_dir is not None:
        return True

    reply = QMessageBox.question(
        parent,
        "Output Directory Required",
        "Choose an output directory to save generated pipeline artifacts?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if reply != QMessageBox.Yes:
        return False

    path = QFileDialog.getExistingDirectory(
        parent,
        "Select Pipeline Output Directory",
        str(Path.home()),
        QFileDialog.ShowDirsOnly,
    )
    if not path:
        return False

    data_manager.set_output_dir(path)
    return True
