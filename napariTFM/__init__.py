# The widget pulls in napari/Qt/magicgui. Import it lazily so the pure-compute
# backend (napariTFM.backend.*) stays importable in a headless / GUI-free env
# (e.g. the HPC regularization sweep). In a normal install this branch always
# succeeds and behavior is unchanged.
try:
    from napariTFM.widgets._widget import napariTFMWidget

    __all__ = ["napariTFMWidget"]
except ImportError:
    __all__ = []
