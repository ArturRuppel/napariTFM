#!/usr/bin/env python3
"""
Launch napari and automatically load the TFM plugin using napari's plugin system
"""
import sys
import os
from pathlib import Path


def main():
    # Set environment for compatibility
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Import napari main and run it with plugin loading
    from napari.__main__ import main as napari_main
    from napari import Viewer

    # Monkey patch to auto-load our plugin after viewer creation
    original_viewer_init = Viewer.__init__

    def patched_init(self, *args, **kwargs):
        original_viewer_init(self, *args, **kwargs)
        try:
            from napariTFM.widgets._widget import napariTFMWidget

            widget = napariTFMWidget(self)
            self.window.add_dock_widget(widget, name="napariTFM", area="right")
        except Exception as e:
            print(f"Could not load napariTFM plugin: {e}")

    Viewer.__init__ = patched_init

    # Run napari normally
    sys.exit(napari_main())


if __name__ == "__main__":
    main()
