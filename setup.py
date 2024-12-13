from setuptools import setup, find_packages

setup(
    name="napari-tfm",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "napari",
        "numpy",
        "qtpy",
        "scikit-image",
        "opencv-python",
        "scipy",
        "pillow",
        "tifffile",
        "pyyaml",
    ],
    entry_points={
        "napari.plugin": [
            "napari-tfm = napari_tfm._widget:TFMWidget",
        ],
    },
    description="A napari plugin for Traction Force Microscopy analysis",
    author="Your Name",
    license="MIT",
    url="https://github.com/yourusername/napari-tfm",
)