# Available at setup time due to pyproject.toml
from setuptools import Extension, find_packages, setup
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
    Pybind11Extension(
        "pycocotools._fasteval",
        sources=["../fastcocoeval/cocoeval.cpp"],
        include_dirs=get_numpy_includes(),
        # extra_compile_args=["-std=c++17"],
        extra_compile_args=["-std=c++17", "-O3", "-march=native"],
    ),
]

setup(
    name="pycocotools",
    packages=find_packages(include=["pycocotools", "pycocotools.*"]),
    package_dir={"pycocotools": "pycocotools"},
    install_requires=[
        "setuptools>=42",
        "cython>=0.27.3",
        "matplotlib>=2.1.0",
    ],
    version="2.0.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
