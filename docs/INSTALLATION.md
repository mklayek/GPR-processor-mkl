# Installation Guide -- GPR-processor-mkl

## System Requirements

- Windows 10/11 (64-bit)
- Python 3.11 or later (Anaconda recommended)
- Minimum 8 GB RAM (16 GB recommended for 3D volume rendering)
- GPU optional (CUDA-capable NVIDIA GPU for accelerated smoothing via CuPy)

## Required Libraries

| Library         | Purpose                              | Required |
|-----------------|--------------------------------------|----------|
| numpy           | Numerical array operations           | Yes      |
| scipy           | Signal processing and filtering      | Yes      |
| matplotlib      | Plotting and visualization           | Yes      |
| h5py            | HDF5 file I/O                        | Yes      |
| Pillow (PIL)    | Image handling and export            | Yes      |
| tkintermapview  | Interactive map widget               | Yes      |
| paramiko        | Remote file access via SFTP/SSH      | Optional |
| pyproj          | Coordinate transformations           | Optional |
| contextily      | Basemap tile rendering               | Optional |
| cupy            | GPU-accelerated array operations     | Optional |

### Install core dependencies

```
pip install numpy scipy matplotlib h5py Pillow tkintermapview
```

### Install optional dependencies

```
pip install paramiko pyproj contextily cupy-cuda12x
```

## Running the Application

From the project root directory:

```
cd GPR-processor-mkl
python scripts/gprprocessormkl.py
```

## Anaconda Setup

If you are using an Anaconda distribution, activate your environment first:

```
conda activate base
cd GPR-processor-mkl
python scripts/gprprocessormkl.py
```

## Troubleshooting

- **tkintermapview is missing** -- Install it with `pip install tkintermapview`.
- **Maps do not load** -- Check your internet connection. Tile servers require network access to retrieve map imagery.
- **Blowfish deprecation warning appears** -- This warning is harmless and is already suppressed in the codebase.
- **3D rendering is slow** -- Reduce the segment size or enable GPU acceleration by installing CuPy.
- **Logo not showing** -- Ensure `logo.jpg` is present in the `assets/` folder.
