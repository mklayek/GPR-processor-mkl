# GPR-processor-mkl

**Author:** Mrinal Layek ([layek.mk@gmail.com](mailto:layek.mk@gmail.com))

---

## Description

GPR-processor-mkl is a professional Ground Penetrating Radar (GPR) data processing and visualization application with a dark-themed GUI, built on tkinter and matplotlib. It provides a comprehensive suite of tools for loading, processing, analyzing, and exporting GPR data across multiple industry-standard formats.

---

## Features

- **Multi-format data support** -- Load and process V00, DT, GEOX, and GEC GPR data formats
- **Professional dark theme GUI** -- Over 20 color-coded function categories for intuitive navigation
- **3D Volume Viewer** -- Chair Display (corner cutaway) and Section Display (orthogonal slices) for volumetric data exploration
- **Interactive GPS map integration** -- Powered by tkintermapview with 13 tile server options for geospatial context
- **Signal processing** -- FIR filters, Hilbert transform, envelope extraction, and deconvolution
- **Spectral analysis** -- Background removal and depth conversion utilities
- **Kirchhoff migration** -- Full implementation for subsurface imaging correction
- **Batch processing** -- Folder-based batch processing with segment navigation
- **High-resolution export** -- PNG export for publication-quality figures

---

## Folder Structure

```
GPR-processor-mkl/
|-- assets/
|   |-- logo.ico
|   |-- logo.jpg
|-- docs/
|   |-- README.md
|-- output/
|-- scripts/
|   |-- gprprocessormkl.py
```

---

## Quick Start

```bash
cd GPR-processor-mkl
python scripts/gprprocessormkl.py
```

---

## Requirements

- Python 3.11+
- numpy
- scipy
- matplotlib
- tkinter
- tkintermapview
- h5py
- Pillow

---

## License

Proprietary -- Mrinal Layek. All rights reserved.
