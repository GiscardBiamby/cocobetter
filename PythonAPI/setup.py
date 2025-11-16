from pybind11 import get_cmake_dir

# Available at setup time due to pyproject.toml
from setuptools import Extension, setup

# Optional: only needed if you actually enable the pybind11 extension below
from pybind11.setup_helpers import Pybind11Extension, build_ext  # noqa: F401


def get_numpy_includes():
    import numpy as np

    return [np.get_include(), "../common"]


ext_modules = [
    Extension(
        "pycocotools._mask",
        sources=["../common/maskApi.c", "pycocotools/_mask.pyx"],
        include_dirs=get_numpy_includes(),
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-std=c99"],
    ),
    # Pybind11Extension(
    #     "pycocotools._eval",
    #     sources=["../fastcocoeval/cocoeval.cpp"],
    #     include_dirs=get_numpy_includes(),
    # )
]

setup(
    name="pycocotools",
    packages=["pycocotools"],
    package_dir={"pycocotools": "pycocotools"},
    install_requires=[
        "setuptools>=18.0",
        "cython>=0.27.3",
        "matplotlib>=2.1.0",
    ],
    version="2.0.2",
    ext_modules=ext_modules,
)
