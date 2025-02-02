import logging
import os
import sys
from pathlib import Path

import napari
from qtpy.QtWidgets import QMessageBox

from napariTFM._widget import napariTFMWidget

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_debug_environment():
    """Setup the debug environment and verify imports"""
    try:
        # Add the parent directory to Python path
        current_dir = Path(__file__).parent.absolute()
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
            logger.debug(f"Added {parent_dir} to Python path")

        return napariTFMWidget


    except Exception as e:
        logger.error(f"Error setting up debug environment: {e}")
        raise


def create_viewer_with_widget():
    """Create the napari viewer and add the widget"""
    try:
        # Create the viewer
        viewer = napari.Viewer()
        logger.debug("Created napari viewer")

        # Get the widget class
        napariTFMWidget = setup_debug_environment()

        # Create and add the widget
        widget = napariTFMWidget(viewer)
        dock_widget = viewer.window.add_dock_widget(
            widget,
            name="napariTFM",
            area='right'
        )
        logger.debug("Added widget to viewer")

        return viewer

    except Exception as e:
        logger.error(f"Error creating viewer with widget: {e}")
        raise


def main():
    """Main entry point with error handling"""
    try:
        logger.debug("Starting napari launcher")
        viewer = create_viewer_with_widget()
        napari.run()

    except Exception as e:
        logger.error(f"Fatal error in napari launcher: {e}")
        # Show error dialog with more detailed information
        error_message = (
            f"Error starting napari: {str(e)}\n\n"
            "This might be caused by:\n"
            "1. Missing required files\n"
            "2. Incorrect file structure\n"
            "3. Missing dependencies\n\n"
            "Check the console for more details."
        )
        QMessageBox.critical(None, "Fatal Error", error_message)
        raise


if __name__ == "__main__":
    main()