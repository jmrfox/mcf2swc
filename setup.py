from setuptools import setup, find_packages

# Minimal setup.py to enable editable installs and constrain package discovery
# to only the `mcf2swc` package (avoiding `data/` and `notebooks/`).

setup(
    name="mcf2swc",
    version="0.1.0",
    description=(
        "A lightweight toolkit for converting mesh cross-sections and polyline "
        "guidance into SWC skeletons."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["mcf2swc", "mcf2swc.*"]),
    include_package_data=False,
    python_requires=">=3.12",
    install_requires=[
        "ipykernel>=6.30.1",
        "jupyter>=1.1.1",
        "matplotlib>=3.10.6",
        "navis>=1.10.0",
        "nbformat>=5.10.4",
        "networkx>=3.5",
        "numpy>=2.3.3",
        "pandas>=2.3.2",
        "plotly>=6.3.0",
        "pyvista[jupyter]>=0.46.3",
        "scipy>=1.16.1",
        "shapely>=2.1.1",
        "trame>=3.12.0",
        "trimesh>=4.8.1",
    ],
)
