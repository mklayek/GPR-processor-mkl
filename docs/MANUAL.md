# User Manual — GPR-processor-mkl

## Overview
GPR-processor-mkl is a desktop application for processing and visualizing Ground Penetrating Radar data. The GUI organizes tools into color-coded categories across a 3-row compact toolbar.

## GUI Layout
- **Top toolbar**: 3 rows of categorized buttons (File, Spectral, Declutter, Depth, Colormap, Display, Hilbert, Filters, Geospatial, Analysis, GPS, Control)
- **Main area**: matplotlib plot canvas for 2D profile display
- **Bottom bar**: View controls (colormap selector, GPU toggle, CPU count)
- **Metadata panel**: 6-line panel showing file info, trace count, sample count

## File Loading
- **V00**: Browse for .v00 GPR files
- **DT**: Browse for .dt files
- **GEOX**: Load geospatial coordinate files
- **Folder**: Load entire folder of V00/DT files with segment navigation (Prev/Next buttons)

## Signal Processing
- **FIR Low/Band**: Finite Impulse Response low-pass and band-pass filters
- **BG Rem**: Background removal (mean trace subtraction)
- **Decon**: Deconvolution for improving vertical resolution
- **Hilbert**: Hilbert transform for instantaneous attributes
- **Envelope**: Compute signal envelope
- **Phase**: Instantaneous phase display
- **Kirchhoff**: Kirchhoff migration for dipping reflector correction
- **Depth**: Time-to-depth conversion using velocity model

## Visualization Modes

### 2D Profile View (Main Window)
The default view showing the radargram as a 2D image. Supports colormap changes, zoom, pan via the matplotlib toolbar.

### 3D Volume Viewer (3D Vol Button)
Opens a dark-themed popup window with:
- **GPS map** (left panel): Full survey path in gray, current 40m segment highlighted in red
- **3D view** (right panel): Chair View (3 cutting planes) or Volume View (6-face cube)
- **Controls**: Segment navigation, colormap selector, tile server selector, Z-depth and Y-crossline sliders
- **Bottom panels**: XZ B-scan and XY depth-map (2D reference slices)
- **View modes**: Toggle between Chair and Volume via radio buttons

### Chair Display (Chair Button)
Opens a dedicated corner-cutaway viewer:
- **9-patch geometry**: L-shaped top, internal cut walls, shelf floor, outer faces
- **Smooth interpolated surfaces**: Per-axis cubic interpolation (scipy.ndimage.zoom)
- **Cut controls**: Cut Z (depth), Cut X (distance), Cut Y (channel) entry fields
- **Rotation controls**: Step size, H-left/right, V-up/down, diagonal rotation
- **View-dependent edge lines**: Only visible cube edges drawn, with cut boundary edges
- **GPU acceleration**: Optional CuPy-based smoothing (toggle in main toolbar)
- **Bottom panels**: XZ, XY, YZ reference slices with crosshairs

### 3D Section Display (3D Sec Button)
Opens GPR-GUI-2026 style orthogonal viewer:
- **GPS map** (left panel): Survey path with segment highlighting
- **Three panels**: XY (depth map), XZ (B-scan), YZ (crossline)
- **Click-to-slice**: Click on any panel to move crosshairs and update slices
- **Drag-to-slice**: Drag crosshairs for interactive navigation
- **Crosshairs**: Black outline + yellow fill for visibility on any colormap

## GPS Maps
All map views use tkintermapview popups with:
- 13 tile server options (OSM, CartoDB, Google, ESRI, Stamen)
- Colored track polylines
- Start/end markers
- Tile server dropdown in header bar

## Output and Saving
- **Save buttons** available in each 3D viewer popup
- Figures saved as high-resolution PNG (300 DPI)
- Output directory: `output/` (auto-created)
- Auto-generated filenames: `volume_render.png`, `chair_display.png`, `section_xy.png`

## Keyboard and Mouse Controls
- **Zoom**: Scroll wheel on matplotlib canvas
- **Pan**: Click and drag with middle mouse button
- **Rotate 3D**: Click and drag on 3D view, or use rotation buttons
- **Enter**: Submit cut values in entry fields
