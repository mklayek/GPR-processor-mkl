#v00readerV5.py
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter.simpledialog as simpledialog
import os
import math
import numpy as np
import pyproj  # for coordinate transformations
from pyproj import Transformer
import pandas as pd
from typing import Optional, Literal, Union, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numpy import sinc
import scipy as _scipy  # for pyhht compatibility (scipy.angle may be missing in new versions)

# Newer SciPy releases removed the top-level scipy.angle symbol that pyhht expects.
# Provide a backward-compatible alias so that `from scipy import angle` inside pyhht works.
if not hasattr(_scipy, "angle"):
    import numpy as _np  # local alias to avoid confusion
    _scipy.angle = _np.angle

from pyhht.emd import EMD
from scipy.signal import hilbert
import re
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from sklearn.decomposition import FastICA
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")
import pywt
import cv2
from skimage import feature
import folium
import webbrowser
import tempfile
import h5py

# Output directory for saved figures
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

try:
    import tkintermapview
    _TKMAPVIEW_AVAILABLE = True
except ImportError:
    _TKMAPVIEW_AVAILABLE = False
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # ensure 3D projection registration

try:
    import vtk
    from vtk.util import numpy_support as vtk_numpy_support
    _VTK_AVAILABLE = True
except Exception:
    _VTK_AVAILABLE = False

try:
    from PIL import Image as _PILImage, ImageTk as _PILImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# --- GPU detection ---
_GPU_BACKEND = None   # "cupy", "torch", or None
_GPU_DEVICES = []     # list of dicts: {index, name, memory_mb, cuda_version}
_GPU_INFO = {"cuda_available": False, "backend": None, "device_count": 0}

try:
    import cupy as _cp
    import cupyx.scipy.ndimage  # noqa: F401
    _GPU_BACKEND = "cupy"
    _GPU_INFO["backend"] = "cupy"
    _GPU_INFO["cuda_available"] = True
    _GPU_INFO["device_count"] = _cp.cuda.runtime.getDeviceCount()
    try:
        _GPU_INFO["cuda_version"] = ".".join(str(x) for x in _cp.cuda.runtime.runtimeGetVersion())
    except Exception:
        _GPU_INFO["cuda_version"] = "N/A"
    for _gi in range(_GPU_INFO["device_count"]):
        _props = _cp.cuda.runtime.getDeviceProperties(_gi)
        _gname = _props.get("name", b"GPU")
        if hasattr(_gname, "decode"):
            _gname = _gname.decode()
        _mem_mb = _props.get("totalGlobalMem", 0) // (1024 * 1024)
        _GPU_DEVICES.append({"index": _gi, "name": str(_gname), "memory_mb": _mem_mb})
except Exception:
    pass

if _GPU_BACKEND is None:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _GPU_BACKEND = "torch"
            _GPU_INFO["backend"] = "torch"
            _GPU_INFO["cuda_available"] = True
            _GPU_INFO["device_count"] = _torch.cuda.device_count()
            try:
                _GPU_INFO["cuda_version"] = _torch.version.cuda or "N/A"
            except Exception:
                _GPU_INFO["cuda_version"] = "N/A"
            for _gi in range(_GPU_INFO["device_count"]):
                _mem_mb = _torch.cuda.get_device_properties(_gi).total_mem // (1024 * 1024)
                _GPU_DEVICES.append({"index": _gi, "name": _torch.cuda.get_device_name(_gi),
                                     "memory_mb": _mem_mb})
    except Exception:
        pass


# ---------- Local GEOX → map helpers (independent of plot_geox.py) ----------

GEOX_ORIGIN_LAT = 37.51
GEOX_ORIGIN_LON = 126.995
METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LON = 88_000.0  # approx at Korea latitude
DEFAULT_EPSG_PROJECTED = 32652  # WGS 84 / UTM zone 52N (optional when data are full UTM)

try:
    from pyproj import Transformer as _TestTransformer  # noqa: F401
    _PYPROJ_AVAILABLE = True
except Exception:
    _PYPROJ_AVAILABLE = False


def _looks_like_utm(x: np.ndarray, y: np.ndarray) -> bool:
    """Heuristic: are these values in typical UTM range?"""
    if len(x) == 0:
        return False
    mx, my = float(np.median(x)), float(np.median(y))
    return (100_000 <= mx <= 900_000) and (1_000_000 <= my <= 10_000_000)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth from point 1 to point 2, degrees from North (0–360)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _nmea_to_decimal(coord_str: str, direction: str) -> Optional[float]:
    """Convert NMEA DDMM.MMMM to decimal degrees."""
    try:
        if not coord_str:
            return None
        dot_idx = coord_str.index(".")
        degrees = float(coord_str[: dot_idx - 2])
        minutes = float(coord_str[dot_idx - 2 :])
        decimal = degrees + minutes / 60.0
        if direction in ("S", "W"):
            decimal = -decimal
        return decimal
    except Exception:
        return None


def _read_gps_simple(file_path: str) -> pd.DataFrame:
    """Read .gps NMEA GPGGA/GNGGA as DataFrame(lat,lon)."""
    lats, lons = [], []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or (not line.startswith("$GPGGA") and not line.startswith("$GNGGA")):
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                lat = _nmea_to_decimal(parts[2], parts[3])
                lon = _nmea_to_decimal(parts[4], parts[5])
                if lat is not None and lon is not None and abs(lat) > 1 and abs(lon) > 1:
                    lats.append(lat)
                    lons.append(lon)
    except Exception:
        pass
    return pd.DataFrame({"lat": lats, "lon": lons})


def _first_last_valid_latlon_geox(filepath: str) -> Optional[tuple[tuple[float, float], Optional[tuple[float, float]]]]:
    """First and last valid Lat,Lon from .geox/.gec (cols 4,5) if they exist."""
    points = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("<") or ln.isdigit():
                    continue
                parts = ln.split(",")
                if len(parts) < 6:
                    continue
                try:
                    lat = float(parts[4].strip())
                    lon = float(parts[5].strip())
                except (ValueError, IndexError):
                    continue
                if abs(lat) > 1 and abs(lon) > 1 and -90 <= lat <= 90 and -180 <= lon <= 180:
                    points.append((lat, lon))
    except Exception:
        pass
    if not points:
        return None
    return (points[0], points[-1] if len(points) > 1 else None)


def _local_xy_to_wgs84(
    x: np.ndarray,
    y: np.ndarray,
    epsg: Optional[int] = None,
    origin: Optional[tuple[float, float]] = None,
    bearing_deg: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert local X,Y (metres) to WGS84 lat,lon.
    - If epsg given and pyproj available, or if values look like UTM, use that CRS.
    - Else, treat X,Y as local metres, use origin=(lat0,lon0) or default Korea origin,
      and optionally rotate by bearing so X is along the road.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    use_pyproj = False
    if _PYPROJ_AVAILABLE and epsg is not None:
        use_pyproj = True
        crs = epsg
    elif _PYPROJ_AVAILABLE and _looks_like_utm(x, y):
        use_pyproj = True
        crs = DEFAULT_EPSG_PROJECTED
    if use_pyproj:
        try:
            trans = Transformer.from_crs(f"EPSG:{crs}", "EPSG:4326", always_xy=True)
            lon, lat = trans.transform(x, y)
            return np.asarray(lat), np.asarray(lon)
        except Exception:
            pass
    lat0, lon0 = origin if origin else (GEOX_ORIGIN_LAT, GEOX_ORIGIN_LON)
    if bearing_deg is not None:
        rad = math.radians(bearing_deg)
        c, s = math.cos(rad), math.sin(rad)
        north_m = x * c - y * s
        east_m = x * s + y * c
        lat = lat0 + north_m / METERS_PER_DEG_LAT
        lon = lon0 + east_m / METERS_PER_DEG_LON
    else:
        lat = lat0 + y / METERS_PER_DEG_LAT
        lon = lon0 + x / METERS_PER_DEG_LON
    return lat, lon


def _read_geox_simple(file_path: str) -> pd.DataFrame:
    """Read .geox/.gec into DataFrame(lat,lon) using internal rules."""
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("<") or line.isdigit():
                    continue
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                try:
                    vals = [float(p.strip()) for p in parts[:8]]
                except ValueError:
                    continue
                while len(vals) < 8:
                    vals.append(0.0)
                rows.append(vals[:8])
    except Exception:
        pass
    if not rows:
        return pd.DataFrame(columns=["lat", "lon"])

    df = pd.DataFrame(
        rows, columns=["Marker", "X", "Y", "Z", "Lat", "Lon", "Alt", "Time"]
    )
    has_valid_ll = (
        (df["Lat"].abs() > 1e-6)
        & (df["Lon"].abs() > 1e-6)
        & (df["Lat"].between(-90, 90))
        & (df["Lon"].between(-180, 180))
    )
    if has_valid_ll.any():
        out = df.loc[has_valid_ll, ["Lat", "Lon"]].copy()
        out.columns = ["lat", "lon"]
        return out.reset_index(drop=True)

    # local-only: derive origin & bearing from other geo files in same folder if possible
    dirpath = os.path.dirname(os.path.abspath(file_path))
    origin, bearing = None, None
    ref = _get_reference_origin_and_bearing(dirpath)
    if ref is not None:
        origin = (ref[0], ref[1])
        bearing = ref[2]
    lat, lon = _local_xy_to_wgs84(df["X"].values, df["Y"].values, origin=origin, bearing_deg=bearing)
    return pd.DataFrame({"lat": lat, "lon": lon})


def _get_reference_origin_and_bearing(dirpath: str) -> Optional[tuple[float, float, Optional[float]]]:
    """Use .gps or .gec/.geox in folder to get origin (lat0,lon0) and bearing."""
    if not dirpath or not os.path.isdir(dirpath):
        return None
    for name in sorted(os.listdir(dirpath)):
        if name.lower().endswith(".gps"):
            df = _read_gps_simple(os.path.join(dirpath, name))
            if df is not None and not df.empty:
                lat0 = float(df["lat"].iloc[0])
                lon0 = float(df["lon"].iloc[0])
                bearing = None
                if len(df) >= 2:
                    lat1 = float(df["lat"].iloc[-1])
                    lon1 = float(df["lon"].iloc[-1])
                    bearing = _bearing_deg(lat0, lon0, lat1, lon1)
                return (lat0, lon0, bearing)
    for ext in (".gec", ".geox"):
        for name in sorted(os.listdir(dirpath)):
            if name.lower().endswith(ext):
                fl = _first_last_valid_latlon_geox(os.path.join(dirpath, name))
                if fl is not None:
                    (lat0, lon0), last = fl
                    bearing = None
                    if last is not None:
                        lat1, lon1 = last
                        bearing = _bearing_deg(lat0, lon0, lat1, lon1)
                    return (lat0, lon0, bearing)
    return None


def _load_geo_simple(file_path: str) -> pd.DataFrame:
    """Dispatch loader for .geox/.gec/.gps."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".gps":
        return _read_gps_simple(file_path)
    if ext in (".geox", ".gec"):
        return _read_geox_simple(file_path)
    return _read_geox_simple(file_path)


def _open_map_popup(title, tracks, markers=None, zoom=16, root=None):
    """Open a tkintermapview popup window with survey tracks and markers.

    Args:
        title:   Window title
        tracks:  list of (lats_list, lons_list, color_str, label_str)
        markers: list of (lat, lon, label_str) or None
        zoom:    initial zoom level
        root:    parent tk window (optional)
    """
    if not _TKMAPVIEW_AVAILABLE:
        messagebox.showerror("Missing package",
                             "tkintermapview not installed.\nInstall: pip install tkintermapview")
        return

    # Calculate center from all track coords
    all_lat, all_lon = [], []
    for lats, lons, _c, _n in tracks:
        all_lat.extend(lats)
        all_lon.extend(lons)
    if not all_lat:
        messagebox.showwarning("No data", "No coordinates to display.")
        return
    center_lat = sum(all_lat) / len(all_lat)
    center_lon = sum(all_lon) / len(all_lon)

    # Tile server definitions: (label, url, max_zoom)
    _TILE_SERVERS = [
        ("OpenStreetMap",       "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",                        19),
        ("CartoDB Positron",    "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",               20),
        ("CartoDB Dark Matter", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",                20),
        ("Google Streets",      "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",                     22),
        ("Google Satellite",    "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",                     22),
        ("Google Hybrid",       "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",                     22),
        ("Google Terrain",      "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",                     22),
        ("ESRI World Imagery",  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19),
        ("ESRI Topo",           "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", 19),
        ("ESRI Street Map",     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", 19),
        ("Stamen Terrain",      "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",       18),
        ("Stamen Toner",        "https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png",         18),
        ("Stamen Watercolor",   "https://tiles.stadiamaps.com/tiles/stamen_watercolor/{z}/{x}/{y}.jpg",    18),
    ]

    popup = tk.Toplevel(root)
    popup.title(title)
    popup.geometry("1100x800")
    popup.configure(bg="#1e1e2e")

    # Header bar with title + tile selector
    hdr = tk.Frame(popup, bg="#2b2d3e", padx=8, pady=4)
    hdr.pack(fill=tk.X)
    tk.Label(hdr, text=title, bg="#2b2d3e", fg="#ccd6f6",
             font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
    tk.Label(hdr, text=f"  {len(tracks)} track(s)", bg="#2b2d3e", fg="#8892b0",
             font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)

    # Tile server selector (right side of header)
    tk.Label(hdr, text="  Map:", bg="#2b2d3e", fg="#8892b0",
             font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 2))
    tile_var = tk.StringVar(value="OpenStreetMap")
    tile_combo = ttk.Combobox(hdr, textvariable=tile_var,
                              values=[t[0] for t in _TILE_SERVERS],
                              state="readonly", width=20)
    tile_combo.pack(side=tk.LEFT, padx=2)

    # Map widget
    map_widget = tkintermapview.TkinterMapView(popup, corner_radius=0)
    map_widget.pack(fill=tk.BOTH, expand=True)
    map_widget.set_position(center_lat, center_lon)
    map_widget.set_zoom(zoom)

    # --- Helper: draw all paths and markers on the map ---
    _PATH_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                    "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60",
                    "#8e44ad", "#16a085", "#d35400", "#2c3e50", "#f1c40f"]

    def _draw_overlays():
        for idx, (lats, lons, color, label) in enumerate(tracks):
            if len(lats) < 2:
                continue
            path_coords = list(zip(lats, lons))
            draw_color = color if color and color.startswith("#") else _PATH_COLORS[idx % len(_PATH_COLORS)]
            map_widget.set_path(path_coords, color=draw_color, width=3)
            map_widget.set_marker(lats[0], lons[0], text=f"{label} [S]")
        if markers:
            for mlat, mlon, mlabel in markers:
                try:
                    map_widget.set_marker(float(mlat), float(mlon), text=str(mlabel))
                except (TypeError, ValueError):
                    continue

    _draw_overlays()

    # --- Tile server switch callback ---
    def _on_tile_change(_event=None):
        name = tile_var.get()
        for lbl, url, mz in _TILE_SERVERS:
            if lbl == name:
                map_widget.set_tile_server(url, max_zoom=mz)
                break

    tile_combo.bind("<<ComboboxSelected>>", _on_tile_change)

    # Status bar
    sbar = tk.Frame(popup, bg="#2b2d3e", padx=8, pady=3)
    sbar.pack(fill=tk.X, side=tk.BOTTOM)
    tk.Label(sbar, text=f"Center: {center_lat:.6f}, {center_lon:.6f}  |  Points: {len(all_lat)}",
             bg="#2b2d3e", fg="#8892b0", font=("Consolas", 8)).pack(side=tk.LEFT)

    popup.transient(root)
    popup.grab_set()


# =============================================================================
#                           GPR2DLoader Class
# =============================================================================
class GPR2DLoader:
    """
    GPR data loader class
    Supports .v00, .d00, .dt
    Auto-loads HDR and GEOX/GEC when available
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================
    def __init__(self):

        # ---------------- Core data ----------------
        self.data: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.xyz: Optional[pd.DataFrame] = None

        # ---------------- File info ----------------
        self.base_path: Optional[str] = None
        self.line_name: Optional[str] = None
        self.data_type: Optional[str] = None
        self.file_path: Optional[str] = None
        self.file_size: Optional[int] = None
        self.header_size: Optional[int] = None

        # ---------------- Data geometry ----------------
        self.samples_per_trace: Optional[int] = None
        self.num_traces: Optional[int] = None
        self.shape: Optional[tuple] = None
        self.dtype: Optional[np.dtype] = None

        # ---------------- HDR parameters ----------------
        self.hdr_info: Optional[Dict] = None
        self._sample_interval_s: Optional[float] = None
        self._velocity: Optional[float] = None
        self._x_cell_m: Optional[float] = None
        self._x_offset_m: float = 0.0   # start distance (m) for this segment (GRED_HD/IDS: X_OFFSET in HDR)
        self._num_traces_hdr: Optional[int] = None
        self._samples_per_trace_hdr: Optional[int] = None

        # ---------------- GEO / GPS ----------------
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None
        self.coord_type: Optional[str] = None
        self.y_coordinate: Optional[float] = None

        # ---------------- Processing history ----------------
        self.process_history: list[str] = []

    # =========================================================
    # PROCESS TRACKING
    # =========================================================
    def add_process(self, tag: str):
        if tag not in self.process_history:
            self.process_history.append(tag)

    # =========================================================
    # DATA LOADING
    # =========================================================
    def load_data(
        self,
        base_path: str,
        line_name: str,
        data_type: Literal["v00", "d00", "dt"] = "v00",
        samples_per_trace: Optional[int] = None,
        num_traces: Optional[int] = None,
        verbose: bool = True
    ) -> np.ndarray:

        self.base_path = base_path
        self.line_name = line_name
        self.data_type = data_type.lower()

        self._load_hdr_00_auto(base_path, line_name)

        filename = f"{line_name}.{data_type.upper()}"
        self.file_path = os.path.join(base_path, filename)

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(self.file_path)

        self.file_size = os.path.getsize(self.file_path)

        if verbose:
            print(f"Loading {self.file_path}")

        if self.data_type == "v00":
            self.data = self._load_v00(self.file_path, samples_per_trace, num_traces)
        elif self.data_type in ("d00", "dt"):
            self.data = self._load_d00(self.file_path, samples_per_trace, num_traces)
        else:
            raise ValueError("Unsupported data type")

        self._load_geox_auto(base_path, line_name)

        if self.hdr_info:
            self.get_hdr_parameters()
            self.depth = self.get_depth_axis()

        self.process_history = ["RAW"]
        return self.data

    # =========================================================
    # V00 / D00 / DT LOADERS
    # =========================================================
    def _load_v00(self, file_path, samples_per_trace, num_traces):

        dtype = np.int16
        self.dtype = dtype
        dtype_size = np.dtype(dtype).itemsize

        if samples_per_trace is None:
            samples_per_trace = 512

        if num_traces is None:
            est_header = 1024
            est_data = self.file_size - est_header
            num_traces = est_data // (samples_per_trace * dtype_size)

        expected_data_size = samples_per_trace * num_traces * dtype_size
        self.header_size = self.file_size - expected_data_size

        with open(file_path, "rb") as f:
            f.seek(self.header_size)
            raw = np.fromfile(f, dtype=dtype)

        expected = samples_per_trace * num_traces
        if raw.size != expected:
            raw = np.pad(raw[:expected], (0, max(0, expected - raw.size)))

        self.data = raw.reshape(num_traces, samples_per_trace).T
        self.samples_per_trace, self.num_traces = self.data.shape
        self.shape = self.data.shape
        return self.data

    def _load_d00(self, file_path, *_):

        dtype = np.int16
        self.dtype = dtype

        header_size = 5140
        rh_nsamp = 514

        vec = np.fromfile(file_path, dtype=dtype)
        data = vec[header_size // 2:]

        num_traces = data.size // rh_nsamp
        reshaped = data[:num_traces * rh_nsamp].reshape(num_traces, rh_nsamp)

        self.data = reshaped[:, 2:].T
        self.samples_per_trace, self.num_traces = self.data.shape
        self.shape = self.data.shape
        self.header_size = header_size
        return self.data
    def _load_dt(self, file_path: str, samples_per_trace: Optional[int] = None, 
                 num_traces: Optional[int] = None) -> np.ndarray:
        return self._load_d00(file_path, samples_per_trace, num_traces)
    # =========================================================
    # HDR HANDLING
    # =========================================================
    def _load_hdr_00_auto(self, base_path, line_name):
        for ext in ("hdr_00", "hdr_dt", "HDR_00", "HDR_DT"):
            path = os.path.join(base_path, f"{line_name}.{ext}")
            if os.path.exists(path):
                self.hdr_info = self._load_hdr_00(path)
                return
        self.hdr_info = None

    def _load_hdr_00(self, hdr_path):
        res = {}
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f]

        i = 0
        while i < len(lines):
            if lines[i].startswith("<") and lines[i].endswith(">"):
                tag = lines[i][1:-1]
                i += 1
                vals = [self._num(v) for v in lines[i].split()]
                res[tag] = vals if len(vals) > 1 else vals[0]
            i += 1

        self._sample_interval_s = res.get("Y_TIME_CELL")
        self._velocity = res.get("PROP_VEL")
        self._x_cell_m = res.get("X_CELL")
        xoff = res.get("X_OFFSET")
        if xoff is not None:
            try:
                self._x_offset_m = float(xoff) if isinstance(xoff, (int, float)) else float(xoff[0])
            except (TypeError, ValueError, IndexError):
                self._x_offset_m = 0.0
        else:
            self._x_offset_m = 0.0

        camp = res.get("CAMP", [None, None])
        if isinstance(camp, list) and len(camp) >= 2:
            self._num_traces_hdr = int(camp[0])
            self._samples_per_trace_hdr = int(camp[1])

        return res

    @staticmethod
    def _num(x):
        try:
            return int(x) if "." not in x and "E" not in x.upper() else float(x)
        except ValueError:
            return x
    def get_hdr_parameters(self):
        """
        Extract commonly used parameters from HDR file
        """
        if self.hdr_info is None:
            return None

        self._sample_interval_s = self.hdr_info.get("Y_TIME_CELL", self._sample_interval_s)
        self._velocity = self.hdr_info.get("PROP_VEL", self._velocity)
        self._x_cell_m = self.hdr_info.get("X_CELL", self._x_cell_m)
        xoff = self.hdr_info.get("X_OFFSET")
        if xoff is not None:
            try:
                self._x_offset_m = float(xoff) if isinstance(xoff, (int, float)) else float(xoff[0])
            except (TypeError, ValueError, IndexError):
                self._x_offset_m = 0.0
        camp = self.hdr_info.get("CAMP")
        if isinstance(camp, (list, tuple)) and len(camp) >= 2:
            self._num_traces_hdr = int(camp[0])
            self._samples_per_trace_hdr = int(camp[1])

        return {
            "dt": self._sample_interval_s,
            "velocity": self._velocity,
            "dx": self._x_cell_m,
            "ntraces": self._num_traces_hdr,
            "nsamples": self._samples_per_trace_hdr
        }
    def get_metadata(self):
        """
        Return metadata dictionary for GUI, plotting, export
        """
        meta = {
            "line_name": self.line_name,
            "data_type": self.data_type,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "header_size": self.header_size,
            "shape": self.shape,
            "samples_per_trace": self.samples_per_trace,
            "num_traces": self.num_traces,
            "dtype": str(self.dtype),
            "process_history": self.process_history.copy(),
        }

        # HDR-derived parameters
        meta.update({
            "dt": self._sample_interval_s,
            "velocity": self._velocity,
            "dx": self._x_cell_m,
            "hdr_ntraces": self._num_traces_hdr,
            "hdr_nsamples": self._samples_per_trace_hdr,
        })

        # GEOX / GPS
        if self.xyz is not None:
            meta["has_geox"] = True
            meta["y_coordinate"] = self.y_coordinate
        else:
            meta["has_geox"] = False

        return meta

    # =========================================================
    # GEOX / GPS
    # =========================================================
    def _load_geox_auto(self, base_path, line_name):
        path = os.path.join(base_path, f"{line_name}.geox")
        if os.path.exists(path):
            self.xyz = self._load_geox(path)
            if len(self.xyz):
                self.y_coordinate = float(self.xyz["Y"].iloc[0])

    def _load_geox(self, file_path):
        """Load GEOX (GRED_HD/IDS format). Skips header lines and trace-count line; expects rows with Marker,X,Y,Z,Lat,Lon,Alt[,Time]."""
        rows = []
        with open(file_path, "r", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("<"):
                    continue
                parts = ln.split(",")
                if len(parts) < 7:
                    continue
                try:
                    vals = list(map(float, parts[:7]))
                    while len(vals) < 8:
                        vals.append(0.0)
                    rows.append(vals[:8])
                except (ValueError, TypeError):
                    continue
        if not rows:
            return pd.DataFrame(columns=["Marker","X","Y","Z","Lat","Lon","Alt","Time"])
        return pd.DataFrame(rows, columns=["Marker","X","Y","Z","Lat","Lon","Alt","Time"])

    def get_statistics(self):
        if self.data is None:
            return None
        stats = {
            "line_name": self.line_name,
            "shape": self.data.shape,
            "dtype": str(self.dtype),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "median": float(np.median(self.data)),
            "rms": float(np.sqrt(np.mean(self.data.astype(float) ** 2))),
            "samples_per_trace": self.samples_per_trace,
            "num_traces": self.num_traces,
        }
        if self.depth is not None:
            stats["depth_range_m"] = f"0 – {self.depth[-1]:.4f}"
        if self._sample_interval_s is not None:
            stats["sample_interval_s"] = self._sample_interval_s
        if self._velocity is not None:
            stats["velocity_m_s"] = self._velocity
        if self._x_cell_m is not None:
            stats["x_cell_m"] = self._x_cell_m
        return stats

    # =========================================================
    # DEPTH AXIS
    # =========================================================
    def get_depth_axis(self):
        if self._sample_interval_s is None or self._velocity is None:
            return None
        t = np.arange(self.data.shape[0]) * self._sample_interval_s
        self.depth = t * self._velocity / 2
        return self.depth

    # =========================================================
    # PROCESSING METHODS (TRACKED)
    # =========================================================
    
    # =========================================================
    # Background Removal
    # =========================================================
    def background_removal(self, br_type="full", filter_length=200, start_scan=None, end_scan=None):
        if self.data is None:
            return
        d = self.data.astype(float)
        if br_type == "full":
            d -= np.mean(d, axis=1, keepdims=True)
        elif br_type == "adaptive":
            for i in range(d.shape[1]):
                i1, i2 = max(0, i-filter_length//2), min(d.shape[1], i+filter_length//2)
                d[:, i] -= np.mean(d[:, i1:i2], axis=1)
        elif br_type == "scan_range":
            bg = np.mean(d[:, start_scan:end_scan], axis=1, keepdims=True)
            d[:, start_scan:end_scan] -= bg
        self.data = d
        self.add_process("BR")

    # =========================================================
    # Gain
    # =========================================================
    def range_gain(self, gain_type="automatic", n_points=6, overall_gain_db=3.0, horiz_tc=15):
        if self.data is None:
            return
        d = self.data.astype(float)
        z = np.linspace(0, 1, d.shape[0])
        if gain_type == "linear":
            g = np.interp(z, np.linspace(0,1,n_points), np.linspace(1,overall_gain_db,n_points))
            d *= g[:,None]
        elif gain_type == "exponential":
            g = np.interp(z, np.linspace(0,1,n_points), np.logspace(0,np.log10(overall_gain_db),n_points))
            d *= g[:,None]
        self.data = d
        self.add_process(f"RG-{gain_type}")
    # =========================================================
    # Peak extraction
    # =========================================================        
    def extract_peaks(
            self,
            peak_type="all",
            max_peaks=3,
            samples_per_point=3,
            start_sample=0,
            end_sample=None
        ):
            """
            Peak extraction similar to commercial GPR software.
            """

            if self.data is None:
                return None

            data = self.data.copy()
            nrows, ncols = data.shape

            if end_sample is None or end_sample > nrows:
                end_sample = nrows

            output = np.zeros_like(data)

            from scipy.signal import find_peaks

            half_w = samples_per_point // 2

            for col in range(ncols):
                trace = data[start_sample:end_sample, col]

                if peak_type == "positive":
                    peaks, _ = find_peaks(trace)
                elif peak_type == "negative":
                    peaks, _ = find_peaks(-trace)
                else:
                    p1, _ = find_peaks(trace)
                    p2, _ = find_peaks(-trace)
                    peaks = np.unique(np.concatenate([p1, p2]))

                if peaks.size == 0:
                    continue

                # Sort by amplitude
                amps = np.abs(trace[peaks])
                idx = np.argsort(amps)[::-1][:max_peaks]
                peaks = peaks[idx]

                for p in peaks:
                    r0 = start_sample + p
                    r1 = max(0, r0 - half_w)
                    r2 = min(nrows, r0 + half_w + 1)
                    output[r1:r2, col] = data[r1:r2, col]

            return output    
    # =========================================================
    # Denoise
    # =========================================================
    def pca_gradient_wavelet_denoise(
            self,
            bscan,
            pca_keep_ratio=0.45,
            wavelet="db7",
            wavelet_level=7
        ):
            import numpy as np
            import cv2
            import pywt
            from sklearn.decomposition import PCA

            # -------------------------
            # 1. PCA reconstruction
            # -------------------------
            X = bscan.copy()
            X -= np.mean(X, axis=0)

            n_comp = int(pca_keep_ratio * min(X.shape))
            pca = PCA(n_components=n_comp, svd_solver="full")
            Xp = pca.fit_transform(X.T)
            Xrec = pca.inverse_transform(Xp).T

            # -------------------------
            # 2. Gradient magnitude
            # -------------------------
            gx = cv2.Sobel(Xrec, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(Xrec, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(gx**2 + gy**2)

            # -------------------------
            # 3. Otsu threshold
            # -------------------------
            g_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12)
            g_uint8 = (g_norm * 255).astype(np.uint8)

            _, mask = cv2.threshold(
                g_uint8, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            thresholded = Xrec * (mask > 0)

            # -------------------------
            # 4. Wavelet denoising (2D)
            # -------------------------
            coeffs = pywt.wavedec2(thresholded, wavelet, level=wavelet_level)

            cH, cV, cD = coeffs[-1]
            sigma = np.median(np.abs(cD)) / 0.6745
            T = sigma * np.sqrt(2 * np.log(thresholded.size))

            new_coeffs = [coeffs[0]]
            for d in coeffs[1:]:
                new_coeffs.append(tuple(
                    pywt.threshold(x, T, mode="soft") for x in d
                ))

            out = pywt.waverec2(new_coeffs, wavelet)
            return out[:bscan.shape[0], :bscan.shape[1]]
            
    def extract_interpretable_reflectors_hough(
        self,
        canny_low=60,
        canny_high=160,
        min_length_ratio=0.15,     # fraction of section width
        max_dip_deg=25,            # reflector dip limit
        amp_percentile=70,         # amplitude threshold
        continuity_tol=6,          # px gap tolerance
        cluster_dist=12            # px for merging similar lines
    ):
        """
        Human-like reflector extraction:
        - keeps only laterally continuous reflectors
        - suppresses hyperbola flanks and clutter
        """

        if self.data is None:
            return []

        nrows, ncols = self.data.shape
        min_line_length = int(min_length_ratio * ncols)

        # ---------------------------------
        # 1. Normalize + gradient emphasis
        # ---------------------------------
        img = self.data.astype(float)
        img -= img.min()
        img /= (img.max() + 1e-12)

        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        grad = (grad / grad.max() * 255).astype(np.uint8)

        # ---------------------------------
        # 2. Edge detection
        # ---------------------------------
        edges = cv2.Canny(grad, canny_low, canny_high)

        # ---------------------------------
        # 3. Probabilistic Hough
        # ---------------------------------
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=120,
            minLineLength=min_line_length,
            maxLineGap=continuity_tol
        )

        if lines is None:
            return []

        # ---------------------------------
        # 4. Geophysical filtering
        # ---------------------------------
        candidates = []

        amp_thresh = np.percentile(np.abs(self.data), amp_percentile)

        for ln in lines:
            x1, y1, x2, y2 = ln[0]

            length = np.hypot(x2 - x1, y2 - y1)
            dip = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # --- dip constraint (remove hyperbola flanks) ---
            if abs(dip) > max_dip_deg:
                continue

            # --- amplitude coherence along line ---
            num = int(length)
            xs = np.linspace(x1, x2, num).astype(int)
            ys = np.linspace(y1, y2, num).astype(int)

            valid = (
                (xs >= 0) & (xs < ncols) &
                (ys >= 0) & (ys < nrows)
            )

            xs = xs[valid]
            ys = ys[valid]

            if xs.size < 0.5 * length:
                continue

            mean_amp = np.mean(np.abs(self.data[ys, xs]))

            if mean_amp < amp_thresh:
                continue

            candidates.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "length": length,
                "dip": dip,
                "mean_amp": mean_amp
            })

        if not candidates:
            return []

        # ---------------------------------
        # 5. Merge overlapping / redundant lines
        # ---------------------------------
        merged = []

        for c in sorted(candidates, key=lambda x: -x["length"]):
            keep = True
            for m in merged:
                dy = abs((c["y1"] + c["y2"]) / 2 - (m["y1"] + m["y2"]) / 2)
                if dy < cluster_dist and abs(c["dip"] - m["dip"]) < 5:
                    keep = False
                    break
            if keep:
                merged.append(c)

        # ---------------------------------
        # 6. Output
        # ---------------------------------
        reflectors = []
        for m in merged:
            reflectors.append({
                "length": m["length"],
                "dip_angle_deg": m["dip"],
                "mean_amplitude": m["mean_amp"],
                "endpoints": (m["x1"], m["y1"], m["x2"], m["y2"])
            })

        return reflectors
    
    def plot_hough_reflectors(self, lines_info):
        if self.data is None or not lines_info:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))

        vmin, vmax = np.percentile(self.data, [2, 98])
        ax.imshow(
            self.data,
            cmap="gray",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )

        for ln in lines_info:
            x1, y1, x2, y2 = ln["endpoints"]
            ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)

        ax.set_title("Long Continuous Reflectors (Hough)")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample / Depth")
        # ax.invert_yaxis()

        plt.tight_layout()
        return fig
    def pick_layers_semi_auto(
        self,
        amp_percentile=75,
        max_vertical_jump=4,
        min_trace_coverage=0.4
    ):
        """
        Semi-automatic GPR layer picking (horizon tracking)
        """

        if self.data is None:
            return []

        data = self.data.copy()

        # ---------------------------------
        # 1. Envelope (Hilbert)
        # ---------------------------------
        from scipy.signal import hilbert
        env = np.abs(hilbert(data, axis=0))

        # ---------------------------------
        # 2. Threshold strong reflectors
        # ---------------------------------
        thresh = np.percentile(env, amp_percentile)
        mask = env > thresh

        nrows, ncols = env.shape

        visited = np.zeros_like(mask, dtype=bool)
        layers = []

        # ---------------------------------
        # 3. Track laterally (region growing)
        # ---------------------------------
        for col in range(ncols):
            for row in np.where(mask[:, col])[0]:

                if visited[row, col]:
                    continue

                layer = [(row, col)]
                visited[row, col] = True
                cur_row = row

                for c in range(col + 1, ncols):
                    search = np.arange(
                        max(0, cur_row - max_vertical_jump),
                        min(nrows, cur_row + max_vertical_jump + 1)
                    )

                    candidates = search[mask[search, c]]

                    if len(candidates) == 0:
                        break

                    cur_row = candidates[np.argmax(env[candidates, c])]
                    visited[cur_row, c] = True
                    layer.append((cur_row, c))

                if len(layer) / ncols >= min_trace_coverage:
                    layers.append(layer)

        return layers
    # =========================================================
    # Predictive decon
    # =========================================================                    
    def predictive_deconvolution(
        self,
        operator_length=32,
        prediction_lag=8,
        prewhitening=0.1,
        overall_gain=1.0,
        start_sample=0,
        end_sample=None,
        progress_callback=None  # ← NEW: for progress bar
    ):
        """
        RADAN-style Predictive Deconvolution
        Removes antenna ringing and multiples while preserving reflections.
        
        The prediction-error filter (PEF) predicts horizontally-correlated
        noise (ringing, multiples) and subtracts it, leaving the 
        unpredictable target reflections.
        """
        if self.data is None:
            return

        data = self.data.astype(float)
        n_samples, n_traces = data.shape

        if end_sample is None or end_sample > n_samples:
            end_sample = n_samples

        out = data.copy()
        
        # Store original RMS for amplitude preservation
        orig_rms = np.sqrt(np.mean(data[start_sample:end_sample, :]**2)) + 1e-12

        for itr in range(n_traces):
            trace = data[start_sample:end_sample, itr].copy()
            trace_len = len(trace)
            
            # Skip if trace is too short
            if trace_len < operator_length + prediction_lag + 10:
                continue
            
            # Store original trace amplitude for scaling
            trace_std = np.std(trace) + 1e-12

            # Autocorrelation (unbiased estimate)
            autocorr = np.correlate(trace, trace, mode='full')
            autocorr = autocorr[len(trace) - 1:]  # Take positive lags only
            
            # Normalize by number of overlapping samples (unbiased)
            norm_factors = np.arange(trace_len, 0, -1)
            autocorr = autocorr / norm_factors
            
            # Add prewhitening to diagonal (stabilizes inversion)
            autocorr[0] *= (1.0 + prewhitening)

            # Build Toeplitz matrix for Wiener-Levinson
            n = min(operator_length, trace_len // 4)  # Limit operator length
            lag = min(prediction_lag, n - 1)
            
            # Autocorrelation matrix R (Toeplitz structure)
            R = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    idx = abs(i - j)
                    if idx < len(autocorr):
                        R[i, j] = autocorr[idx]
            
            # Cross-correlation vector (prediction at lag samples ahead)
            g = np.zeros(n)
            for i in range(n):
                idx = lag + i
                if idx < len(autocorr):
                    g[i] = autocorr[idx]

            # Solve Wiener-Hopf equation for prediction filter
            try:
                # Add regularization to prevent singular matrix
                R += np.eye(n) * (prewhitening * autocorr[0])
                f = np.linalg.solve(R, g)
            except np.linalg.LinAlgError:
                # If solve fails, use zero filter (no deconvolution)
                f = np.zeros(n)

            # Build prediction-error filter (PEF): spike minus prediction filter
            # PEF = [1, 0, 0, ..., 0, -f[0], -f[1], ..., -f[n-1]]
            pef = np.zeros(lag + n)
            pef[0] = 1.0  # Unit spike at time 0
            pef[lag:lag + n] = -f  # Negative prediction filter at lag
            
            # Apply PEF via convolution
            filtered = np.convolve(trace, pef, mode='full')[:trace_len]
            
            # Normalize output to preserve amplitude characteristics
            filtered_std = np.std(filtered) + 1e-12
            filtered = filtered * (trace_std / filtered_std)
            
            # Apply user gain
            filtered *= overall_gain

            out[start_sample:end_sample, itr] = filtered

            # Report progress
            if progress_callback is not None:
                progress_callback((itr + 1) / n_traces)

        self.data = out
        self.add_process("DECON")     
    def clear_data(self):
        for attr in ['data','xyz','hdr_info','depth','base_path','line_name',
                     'data_type','file_path','file_size','header_size',
                     'samples_per_trace','num_traces','shape','dtype',
                     '_sample_interval_s','_velocity','_x_cell_m',
                     '_num_traces_hdr','_samples_per_trace_hdr']:
            setattr(self, attr, None)
        self._x_offset_m = 0.0
                    
    def depth_crop(self, max_depth_m):
        """
        Crop GPR section to a maximum depth (meters)

        Requires:
        - Depth axis must exist (HDR loaded)
        """

        if self.data is None or self.depth is None:
            raise ValueError("Depth axis not available. Load HDR first.")

        max_depth_m = float(max_depth_m)

        idx = np.where(self.depth <= max_depth_m)[0]

        if len(idx) == 0:
            raise ValueError("Depth crop exceeds data range")

        last = idx[-1] + 1

        self.data = self.data[:last, :]
        self.depth = self.depth[:last]

        self.add_process(f"ZC-{max_depth_m:.2f}m")
    def load_gps_nmea(self, filename):
        latitudes = []
        longitudes = []

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                if not line.startswith("$GPGGA"):
                    continue

                parts = line.split(",")

                if len(parts) < 6:
                    continue

                if not parts[2] or not parts[4]:
                    continue

                try:
                    # Latitude
                    lat_raw = float(parts[2])
                    lat_deg = int(lat_raw // 100)
                    lat_min = lat_raw - lat_deg * 100
                    lat = lat_deg + lat_min / 60.0
                    if parts[3] == "S":
                        lat = -lat

                    # Longitude
                    lon_raw = float(parts[4])
                    lon_deg = int(lon_raw // 100)
                    lon_min = lon_raw - lon_deg * 100
                    lon = lon_deg + lon_min / 60.0
                    if parts[5] == "W":
                        lon = -lon

                    latitudes.append(lat)
                    longitudes.append(lon)

                except ValueError:
                    continue

        if len(latitudes) == 0:
            raise ValueError("No valid $GPGGA GPS coordinates found.")

        self.lat = np.asarray(latitudes)
        self.lon = np.asarray(longitudes)
        self.coord_type = "gps_nmea"       


        
 


# ===================================================================================================================================================================================
#                             GUI Application -MKL
# ====================================================================================================================================================================================

class V00ReaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPR .V00 File Viewer | Mrinal: layek.mk@gmail.com")
        self.root.geometry("1200x900")

        # --- Window icon (logo) — title bar + taskbar + Alt+Tab ---
        self._icon_photos = []
        _logo_dir = os.path.dirname(os.path.abspath(__file__))
        _logo_path = os.path.join(_logo_dir, "logo.jpg")
        if not os.path.isfile(_logo_path):
            _logo_path = os.path.join(_logo_dir, "..", "assets", "logo.jpg")
        if _PIL_AVAILABLE and os.path.isfile(_logo_path):
            try:
                _img = _PILImage.open(_logo_path).convert("RGBA")
                self._icon_photos = [_PILImageTk.PhotoImage(_img.resize((sz, sz), _PILImage.LANCZOS))
                                     for sz in (256, 128, 64, 48, 32, 16)]
                self.root.wm_iconphoto(True, *self._icon_photos)
            except Exception as _icon_err:
                print(f"[WARN] wm_iconphoto failed: {_icon_err}")
            # Also set .ico for Windows taskbar (iconbitmap gives sharper taskbar icon)
            try:
                _ico_path = os.path.join(_logo_dir, "logo.ico")
                if not os.path.isfile(_ico_path):
                    _ico_path = os.path.join(_logo_dir, "..", "assets", "logo.ico")
                if not os.path.isfile(_ico_path):
                    _img.resize((256, 256), _PILImage.LANCZOS).save(
                        _ico_path, format="ICO",
                        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
                self.root.iconbitmap(default=_ico_path)
            except Exception as _ico_err:
                print(f"[WARN] iconbitmap failed: {_ico_err}")

        self.loader = GPR2DLoader()
        
        # Main frames – professional dark theme
        self.root.configure(bg="#1e1e2e")
        self.root.state("zoomed")  # start maximized so all buttons fit

        # --- ttk styling ---
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Toolbar.TFrame", background="#2b2d3e")
        style.configure("Dark.TFrame", background="#1e1e2e")
        style.configure("Meta.TFrame", background="#1e1e2e")
        style.configure("Dark.TCombobox", fieldbackground="#3a3d52", foreground="white",
                        background="#3a3d52", arrowcolor="white")
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", "#3a3d52")],
                  foreground=[("readonly", "white")])

        # ---- PACK ORDER MATTERS: top toolbar → plot (expand) → view bar → metadata (fixed) ----
        self.input_frame = tk.Frame(root, bg="#2b2d3e", padx=6, pady=2)
        self.input_frame.pack(fill=tk.X, side=tk.TOP)

        self.plot_frame = tk.Frame(root, bg="#1e1e2e")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        self.view_control_frame = tk.Frame(root, bg="#1a1b2e", padx=10, pady=4)
        self.view_control_frame.pack(fill=tk.X, side=tk.TOP)

        self.meta_frame = tk.Frame(root, bg="#1e1e2e", padx=8, pady=2)
        self.meta_frame.pack(fill=tk.X, side=tk.BOTTOM)  # fixed height, no expand

        # 2D view state
        self.X_WINDOW_M = 40.0
        self.view_x_start_m = 0.0
        self._total_length_m = None
        self.folder_segment_files = []
        self.current_segment_index = -1
        self._folder_x0_offset = 0.0
        self.view_max_depth = None

        # --- View control bar ---
        _vcl = {"bg": "#1a1b2e", "fg": "#8892b0", "font": ("Segoe UI", 9)}
        _vcb = {"bg": "#3a3d52", "fg": "white", "activebackground": "#4a4d62", "activeforeground": "white",
                "relief": "flat", "padx": 6, "pady": 2, "bd": 0, "font": ("Segoe UI", 9), "cursor": "hand2"}
        _vce = {"bg": "#2a2d42", "fg": "#ccd6f6", "insertbackground": "#ccd6f6",
                "font": ("Consolas", 9), "relief": "flat", "bd": 2, "highlightthickness": 1, "highlightcolor": "#4a4d62"}
        tk.Label(self.view_control_frame, text="X(m):", **_vcl).pack(side=tk.LEFT, padx=(0, 2))
        self.x_pos_var = tk.StringVar(value="0.0")
        tk.Button(self.view_control_frame, text="\u25c0", command=self._view_prev_40m, **_vcb).pack(side=tk.LEFT, padx=1)
        self._x_entry = tk.Entry(self.view_control_frame, textvariable=self.x_pos_var, width=8, **_vce)
        self._x_entry.pack(side=tk.LEFT, padx=1)
        self._x_entry.bind("<Return>", lambda e: self._apply_x_from_entry())
        tk.Button(self.view_control_frame, text="\u25b6", command=self._view_next_40m, **_vcb).pack(side=tk.LEFT, padx=1)
        tk.Label(self.view_control_frame, text="  Dist(m):", **_vcl).pack(side=tk.LEFT, padx=(8, 2))
        self.distance_var = tk.StringVar(value="40")
        self._distance_entry = tk.Entry(self.view_control_frame, textvariable=self.distance_var, width=6, **_vce)
        self._distance_entry.pack(side=tk.LEFT, padx=1)
        self._distance_entry.bind("<Return>", lambda e: self._apply_distance_and_redraw())
        tk.Label(self.view_control_frame, text="  Seg:", **_vcl).pack(side=tk.LEFT, padx=(8, 2))
        self.segment_var = tk.StringVar(value="1")
        tk.Button(self.view_control_frame, text="\u25c0", command=self._view_prev_segment, **_vcb).pack(side=tk.LEFT, padx=1)
        self._segment_entry = tk.Entry(self.view_control_frame, textvariable=self.segment_var, width=5, **_vce)
        self._segment_entry.pack(side=tk.LEFT, padx=1)
        self._segment_entry.bind("<Return>", lambda e: self._apply_segment_from_entry())
        self._segment_entry.bind("<FocusOut>", lambda e: self._apply_segment_from_entry())
        tk.Button(self.view_control_frame, text="\u25b6", command=self._view_next_segment, **_vcb).pack(side=tk.LEFT, padx=1)
        tk.Label(self.view_control_frame, text="  (Seg = file in folder)", **_vcl).pack(side=tk.LEFT, padx=6)

        # --- GPU config (right side of view bar) ---
        self._use_gpu_var = tk.BooleanVar(value=bool(_GPU_BACKEND))
        self._gpu_device_var = tk.IntVar(value=0)
        if _GPU_BACKEND and _GPU_DEVICES:
            _gd = _GPU_DEVICES[0]
            _gpu_short = f"{_gd['name'][:22]} ({_gd['memory_mb']}MB)"
            _cuda_lbl = f"CUDA: Yes ({_GPU_INFO.get('cuda_version', '?')})"
            tk.Checkbutton(self.view_control_frame, text="GPU", variable=self._use_gpu_var,
                           bg="#1a1b2e", fg="#8892b0", selectcolor="#2a2d42",
                           activebackground="#1a1b2e", activeforeground="#8892b0",
                           font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=(0, 2))
            if len(_GPU_DEVICES) > 1:
                _gpu_labels = [f"{d['index']}: {d['name'][:18]}" for d in _GPU_DEVICES]
                _gpu_combo = ttk.Combobox(self.view_control_frame, textvariable=self._gpu_device_var,
                                          values=list(range(len(_GPU_DEVICES))), state="readonly", width=3)
                _gpu_combo.pack(side=tk.RIGHT, padx=(0, 2))
            tk.Label(self.view_control_frame, text=_gpu_short,
                     **_vcl).pack(side=tk.RIGHT, padx=(2, 1))
            tk.Label(self.view_control_frame, text=_cuda_lbl,
                     bg="#1a1b2e", fg="#66bb6a", font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=(4, 1))
        elif _GPU_BACKEND:
            tk.Label(self.view_control_frame, text=f"GPU: {_GPU_BACKEND} (no details)",
                     **_vcl).pack(side=tk.RIGHT, padx=(4, 2))
        else:
            tk.Label(self.view_control_frame, text="GPU: N/A  CUDA: No",
                     bg="#1a1b2e", fg="#ef5350", font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=(4, 2))

        # --- CPU config (right side of view bar) ---
        import multiprocessing as _mp
        _avail_cpus = max(1, _mp.cpu_count())
        self._ncpus_var = tk.IntVar(value=min(20, max(2, _avail_cpus)))
        tk.Label(self.view_control_frame, text=f"  CPUs: {_avail_cpus} avail,  use:",
                 **_vcl).pack(side=tk.RIGHT, padx=(8, 2))
        _cpu_entry = tk.Entry(self.view_control_frame, textvariable=self._ncpus_var,
                              width=3, **_vce)
        _cpu_entry.pack(side=tk.RIGHT, padx=(0, 4))

        def _validate_cpus(*_a):
            try:
                v = self._ncpus_var.get()
                if v < 2:
                    self._ncpus_var.set(2)
                elif v > _avail_cpus:
                    self._ncpus_var.set(_avail_cpus)
            except (tk.TclError, ValueError):
                self._ncpus_var.set(min(20, max(2, _avail_cpus)))
        _cpu_entry.bind("<FocusOut>", _validate_cpus)
        _cpu_entry.bind("<Return>", _validate_cpus)

        # --- Metadata (fixed 6 lines) ---
        tk.Label(self.meta_frame, text="Metadata:", bg="#1e1e2e", fg="#8892b0",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.meta_text = scrolledtext.ScrolledText(self.meta_frame, height=6, width=100,
                                                    font=("Consolas", 9), bg="#1a1b2e", fg="#ccd6f6",
                                                    insertbackground="#ccd6f6", relief="flat", bd=2)
        self.meta_text.pack(fill=tk.X, expand=False, pady=(2, 4))

        # --- Compact button factory (small font, tight padding) ---
        _F = ("Segoe UI", 8)
        _FB = ("Segoe UI", 8, "bold")
        _GP = {"padx": 1, "pady": 1}  # grid padding

        CATS = {
            "file":      {"bg": "#2d4a3e", "fg": "#a3d9b1", "hover": "#3a6050"},
            "file_act":  {"bg": "#2e7d32", "fg": "#ffffff", "hover": "#388e3c"},
            "spectral":  {"bg": "#2d3a4a", "fg": "#90caf9", "hover": "#3a4a5e"},
            "declutter": {"bg": "#4a2d2d", "fg": "#ef9a9a", "hover": "#5e3a3a"},
            "decon":     {"bg": "#6d1b1b", "fg": "#ffcdd2", "hover": "#8b2525"},
            "depth":     {"bg": "#1a3a4a", "fg": "#80deea", "hover": "#2a4a5e"},
            "gps":       {"bg": "#3a1a4a", "fg": "#ce93d8", "hover": "#4e2a62"},
            "cmap":      {"bg": "#4a1a1a", "fg": "#ef5350", "hover": "#6d2525"},
            "zoom":      {"bg": "#1a2a4a", "fg": "#90caf9", "hover": "#2a3a5e"},
            "save":      {"bg": "#1565c0", "fg": "#ffffff", "hover": "#1976d2"},
            "view":      {"bg": "#2d2d4a", "fg": "#b39ddb", "hover": "#3a3a5e"},
            "hilbert":   {"bg": "#1a1a3a", "fg": "#9fa8da", "hover": "#2a2a4e"},
            "filter":    {"bg": "#2d3a3a", "fg": "#80cbc4", "hover": "#3a4a4a"},
            "geo":       {"bg": "#1b3a2d", "fg": "#a5d6a7", "hover": "#2a4e3e"},
            "geo_act":   {"bg": "#283593", "fg": "#ffffff", "hover": "#3949ab"},
            "analysis":  {"bg": "#2d2d3a", "fg": "#b0bec5", "hover": "#3a3a4e"},
            "warn":      {"bg": "#e65100", "fg": "#ffffff", "hover": "#f57c00"},
            "danger":    {"bg": "#b71c1c", "fg": "#ffffff", "hover": "#d32f2f"},
            "accent":    {"bg": "#4a3a1a", "fg": "#ffe082", "hover": "#5e4e2a"},
            "default":   {"bg": "#2d2d3a", "fg": "#ccd6f6", "hover": "#3a3a4e"},
        }

        def _mkbtn(parent, text, cmd, cat="default", row=0, col=0, colspan=1, bold=False):
            c = CATS.get(cat, CATS["default"])
            btn = tk.Button(parent, text=text, command=cmd,
                            bg=c["bg"], fg=c["fg"],
                            activebackground=c["hover"], activeforeground=c["fg"],
                            relief="flat", bd=0, padx=6, pady=2,
                            font=_FB if bold else _F, cursor="hand2",
                            highlightthickness=0)
            btn.grid(row=row, column=col, columnspan=colspan, **_GP)
            btn.bind("<Enter>", lambda e, b=btn, h=c["hover"]: b.configure(bg=h))
            btn.bind("<Leave>", lambda e, b=btn, bg=c["bg"]: b.configure(bg=bg))
            return btn

        def _sep(parent, row, col):
            tk.Frame(parent, width=1, height=18, bg="#3a3d52").grid(row=row, column=col, padx=3, pady=1)

        def _lbl(parent, text, row, col, colspan=1):
            tk.Label(parent, text=text, bg="#2b2d3e", fg="#555e78",
                     font=("Segoe UI", 7), anchor="w").grid(row=row, column=col, columnspan=colspan, sticky="w", padx=2)

        # ====== ROW 0: FILE | SPECTRAL ======
        r = 0
        _lbl(self.input_frame, "FILE:", r, 0)
        self.file_var = tk.StringVar()
        fe = tk.Entry(self.input_frame, textvariable=self.file_var, width=30, state="readonly",
                      bg="#2a2d42", fg="#ccd6f6", readonlybackground="#2a2d42",
                      font=("Consolas", 8), relief="flat", bd=2, highlightthickness=0)
        fe.grid(row=r, column=1, columnspan=2, padx=2, pady=1, sticky="ew")
        _mkbtn(self.input_frame, "V00",          self.browse_file,            "file",     r, 3)
        _mkbtn(self.input_frame, "DT",           self.browse_dt_file,         "file",     r, 4)
        _mkbtn(self.input_frame, "Folder",       self.browse_folder,          "file",     r, 5)
        _mkbtn(self.input_frame, "Load",         self.load_selected_file,     "file_act", r, 6, bold=True)
        _sep(self.input_frame, r, 7)
        _mkbtn(self.input_frame, "FFT",          self.fft_dialog,            "spectral", r, 8)
        _mkbtn(self.input_frame, "HHT",          self.hht_dialog,            "spectral", r, 9)
        _mkbtn(self.input_frame, "HHT-TF",       self.hht_tf_dialog,         "spectral", r, 10)
        _mkbtn(self.input_frame, "ICA",          self.ica_denoise_dialog,     "spectral", r, 11)
        _mkbtn(self.input_frame, "Peaks",        self.peaks_extraction_popup, "spectral", r, 12)
        _sep(self.input_frame, r, 13)
        _mkbtn(self.input_frame, "MS",           self.ms_declutter_dialog,    "declutter", r, 14)
        _mkbtn(self.input_frame, "SVD",          self.svd_declutter_dialog,   "declutter", r, 15)
        _mkbtn(self.input_frame, "RNMF",         self.rnmf_declutter_dialog,  "declutter", r, 16)
        _mkbtn(self.input_frame, "DECON",        self.deconvolution_dialog,   "decon",     r, 17, bold=True)
        _sep(self.input_frame, r, 18)
        _mkbtn(self.input_frame, "D.Crop",       self.depth_crop_dialog,      "depth",     r, 19)
        _mkbtn(self.input_frame, "D.View",       self.depth_view_dialog,      "depth",     r, 20)

        # ====== ROW 1: CMAP | DISPLAY | HILBERT | FILTERS ======
        r = 1
        _lbl(self.input_frame, "CMAP:", r, 0)
        self.cmap_var = tk.StringVar(value="gray")
        cmap_combo = ttk.Combobox(self.input_frame, textvariable=self.cmap_var,
                                  values=["gray", "seismic", "wiggle", "RdBu_r", "coolwarm", "viridis", "magma", "plasma", "hot"],
                                  state="readonly", width=8, style="Dark.TCombobox")
        cmap_combo.grid(row=r, column=1, padx=1, pady=1, sticky="w")
        _mkbtn(self.input_frame, "Apply",        self.change_colormap,                 "cmap",    r, 2, bold=True)
        _mkbtn(self.input_frame, "Zoom+",        lambda: self.zoom(0.8),               "zoom",    r, 3)
        _mkbtn(self.input_frame, "Zoom-",        lambda: self.zoom(1.25),              "zoom",    r, 4)
        _mkbtn(self.input_frame, "Save",         self.save_figure,                     "save",    r, 5, bold=True)
        _mkbtn(self.input_frame, "A-scan",       self.toggle_ascan,                    "view",    r, 6)
        self.crosshair_enabled = tk.BooleanVar(value=False)
        self.crosshair_btn = _mkbtn(self.input_frame, "Cross", self.toggle_crosshair,  "view",   r, 7)
        _sep(self.input_frame, r, 8)
        _mkbtn(self.input_frame, "Envelope",     self.apply_envelope,                  "hilbert", r, 9)
        _mkbtn(self.input_frame, "Phase",        self.apply_instantaneous_phase,       "hilbert", r, 10)
        _mkbtn(self.input_frame, "Freq",         self.apply_instantaneous_frequency,   "hilbert", r, 11)
        _mkbtn(self.input_frame, "ShowICA",      self.show_ica_components,             "hilbert", r, 12)
        _sep(self.input_frame, r, 13)
        _mkbtn(self.input_frame, "FIR-Lo",       self.apply_fir_low,                   "filter",  r, 14)
        _mkbtn(self.input_frame, "FIR-BP",       self.fir_bandpass_dialog,             "filter",  r, 15)
        _mkbtn(self.input_frame, "BG Rem",       self.background_removal_dialog,       "filter",  r, 16)
        _mkbtn(self.input_frame, "RGain",        self.range_gain_dialog,               "filter",  r, 17)
        _sep(self.input_frame, r, 18)
        _mkbtn(self.input_frame, "T-Zero",       self.time_zero_correction_dialog,     "filter",  r, 19)
        _mkbtn(self.input_frame, "Kirchhoff",    self.kirchhoff_migration_dialog,      "filter",  r, 20)

        # ====== ROW 2: GEO/MAP | ANALYSIS | CONTROL ======
        r = 2
        _lbl(self.input_frame, "GEO:", r, 0)
        _mkbtn(self.input_frame, "LoadGEOX",     self.load_geo_file,          "geo",      r, 1)
        _mkbtn(self.input_frame, "SurveyMap",    self.plot_survey_map,        "geo_act",  r, 2, bold=True)
        _mkbtn(self.input_frame, "GPS",          self.plot_gps_nmea,          "geo",      r, 3)
        _mkbtn(self.input_frame, "GEOXpath",     self.plot_geox_path,         "geo",      r, 4)
        _mkbtn(self.input_frame, "GEOX/GPS",     self.plot_geox_on_gps,       "geo",      r, 5)
        _mkbtn(self.input_frame, "GEOXmap",      self.plot_geox_map,          "geo",      r, 6)
        _mkbtn(self.input_frame, "GEOXfolder",   self.plot_geox_folder_map,   "geo",      r, 7)
        _sep(self.input_frame, r, 8)
        _mkbtn(self.input_frame, "Stats",        self.show_stats,             "analysis", r, 9)
        _mkbtn(self.input_frame, "XYZ",          self.plot_geox,              "analysis", r, 10)
        _mkbtn(self.input_frame, "Hough",        self.run_hough_reflectors,   "analysis", r, 11)
        _mkbtn(self.input_frame, "Layers",       self.layer_picker_popup,     "analysis", r, 12)
        _mkbtn(self.input_frame, "AmpMap",       self.show_amplitude_map,     "analysis", r, 13)
        _sep(self.input_frame, r, 14)
        _mkbtn(self.input_frame, "GPSsearch",    self.gps_location_search_dialog, "gps", r, 15, bold=True)
        _mkbtn(self.input_frame, "AllProfiles",  self.plot_all_profiles_dialog,   "gps", r, 16)
        _sep(self.input_frame, r, 17)
        _mkbtn(self.input_frame, "Reset",        self.reset_data,             "warn",     r, 18, bold=True)
        _mkbtn(self.input_frame, "Clear",        self.clear_all,              "danger",   r, 19, bold=True)
        _mkbtn(self.input_frame, "3D Vol",       self.open_3d_volume_viewer,    "accent",   r, 20)
        _mkbtn(self.input_frame, "Chair",        self.open_chair_volume_viewer,  "accent",   r, 21)
        _mkbtn(self.input_frame, "3D Sec",       self.open_3d_section_viewer,    "accent",   r, 22)

        # --- Plot area (gets all remaining space) ---
        self.figure = plt.Figure(figsize=(12, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # Cross-hair state for main GPR figure
        self.crosshair_hline = None
        self.crosshair_vline = None
        self.crosshair_text = None
        # ---------------- A-sCAN WINDOW ----------------
        self.ascan_win = tk.Toplevel(self.root)
        self.ascan_win.title("A-scan")
        self.ascan_win.geometry("350x600")
        self.ascan_win.withdraw()  # hidden until data loaded

        self.ascan_fig = plt.Figure(figsize=(3, 5), dpi=100)
        self.ascan_ax = self.ascan_fig.add_subplot(111)

        self.ascan_canvas = FigureCanvasTkAgg(self.ascan_fig, master=self.ascan_win)
        self.ascan_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.ascan_enabled = False
        self.ascan_cid = None

        # For colormap update
        self.current_image = None
        # 3D volume (HDF5) state
        self._vol3d = None  # dict with data, x, y, z, lat, lon, metadata
        self._current_3d_default_path = ""
    
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select .V00 File",
            filetypes=[("V00 files", "*.V00 *.v00"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)
    def browse_dt_file(self):
        path = filedialog.askopenfilename(
            title="Select .DT File",
            filetypes=[("DT files", "*.DT *.dt"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)

    def browse_folder(self):
        """Select extension (.DT or .v00), then folder; load all matching files sorted by name. Segment = file index."""
        choice = tk.StringVar(value="v00")
        win = tk.Toplevel(self.root)
        win.title("Browse folder – extension")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        tk.Label(win, text="Load files with extension:", font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(12, 6))
        f = tk.Frame(win)
        f.pack(anchor="w", padx=12, pady=4)
        tk.Radiobutton(f, text=".DT only", variable=choice, value="dt").pack(side=tk.LEFT, padx=(0, 12))
        tk.Radiobutton(f, text=".v00 only", variable=choice, value="v00").pack(side=tk.LEFT, padx=(0, 12))
        tk.Radiobutton(f, text="Both (.DT and .v00)", variable=choice, value="both").pack(side=tk.LEFT)
        ok = [False]

        def on_ok():
            ok[0] = True
            win.destroy()

        def on_cancel():
            win.destroy()

        tk.Button(win, text="OK", command=on_ok, width=8).pack(side=tk.LEFT, padx=12, pady=12)
        tk.Button(win, text="Cancel", command=on_cancel, width=8).pack(side=tk.LEFT, pady=12)
        win.wait_window(win)
        if not ok[0]:
            return

        ext_choice = choice.get()
        if ext_choice == "dt":
            exts = (".dt", ".DT")
        elif ext_choice == "v00":
            exts = (".v00", ".V00")
        else:
            exts = (".v00", ".V00", ".dt", ".DT")

        folder = filedialog.askdirectory(title="Select folder with V00/DT files")
        if not folder or not os.path.isdir(folder):
            return
        paths = []
        for f in os.listdir(folder):
            base, ext = os.path.splitext(f)
            if ext in exts:
                paths.append((os.path.join(folder, f), base, ext.lower()))
        if not paths:
            messagebox.showinfo("No files", f"No files with extension(s) {ext_choice} found in the selected folder.")
            return
        paths.sort(key=lambda x: x[1])
        self.folder_segment_files = [(os.path.dirname(p[0]), p[1], "v00" if p[2] == ".v00" else "dt", p[2]) for p in paths]
        # Load the first segment automatically and enter folder/segment mode
        self.current_segment_index = 0
        self._load_file_at_segment_index(0)
        self._sync_segment_display(len(self.folder_segment_files))
        messagebox.showinfo("Folder loaded",
                            f"Found {len(self.folder_segment_files)} file(s). Use Segment controls to navigate.")

    # ------------------------------------------------------------------
    # 3D VOLUME (HDF5) — FULL VIEWER MODULE
    # ------------------------------------------------------------------
    def _load_h5_volume_full(self, path: str) -> dict:
        """Load 3D volume with coordinates, GPS, and metadata from HDF5."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            # Try structured format first (GPR-GUI-2026 style)
            if "data" in f and "coordinates" in f:
                data = np.array(f["data"])
                x = np.array(f["coordinates/x"])
                y = np.array(f["coordinates/y"])
                z = np.array(f["coordinates/z"])
                lat = np.array(f["gps_coordinates/latitude"]) if "gps_coordinates/latitude" in f else None
                lon = np.array(f["gps_coordinates/longitude"]) if "gps_coordinates/longitude" in f else None
                meta = dict(f["metadata"].attrs) if "metadata" in f else {}
            else:
                # Fallback: find first 3D dataset
                ds, _ = self._find_first_3d_dataset_h5(f)
                if ds is None:
                    raise ValueError("No 3D dataset found in HDF5 file.")
                data = np.array(ds)
                nz, nx, ny = data.shape
                x = np.arange(nx, dtype=float)
                y = np.arange(ny, dtype=float)
                z = np.arange(nz, dtype=float)
                lat, lon, meta = None, None, {}
        if data.ndim != 3:
            raise ValueError("Dataset is not 3D.")
        vmin = float(np.percentile(data, 2))
        vmax = float(np.percentile(data, 98))
        if vmin == vmax:
            vmin, vmax = float(data.min()), float(data.max())
        return {"data": data, "x": x, "y": y, "z": z, "lat": lat, "lon": lon,
                "metadata": meta, "vmin": vmin, "vmax": vmax, "path": path}

    @staticmethod
    def _find_first_3d_dataset_h5(h5_obj):
        """Recursively find first 3D dataset in HDF5."""
        def _walk(group, prefix=""):
            for key in group.keys():
                obj = group[key]
                p = f"{prefix}/{key}" if prefix else key
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    return obj, p
                elif isinstance(obj, h5py.Group):
                    out = _walk(obj, p)
                    if out[0] is not None:
                        return out
            return None, None
        return _walk(h5_obj)

    def _select_and_load_volume(self):
        """Show file dialog, load HDF5 volume, return volume dict or None."""
        default_path = self._current_3d_default_path
        volume_path = None
        if default_path and os.path.isfile(default_path):
            if messagebox.askyesno("3D Volume", f"Open default volume?\n\n{default_path}\n\nPress No to browse."):
                volume_path = default_path
        if not volume_path:
            volume_path = filedialog.askopenfilename(
                title="Select 3D HDF5 Volume",
                initialdir=os.path.dirname(default_path) if default_path and os.path.isdir(os.path.dirname(default_path)) else None,
                filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")])
        if not volume_path:
            return None
        try:
            vd = self._load_h5_volume_full(volume_path)
        except Exception as e:
            messagebox.showerror("3D Volume Error", f"Failed to load:\n{e}")
            return None
        self._vol3d = vd
        self._current_3d_default_path = volume_path
        return vd

    def open_3d_volume_viewer(self):
        """Open the redesigned 3D volume viewer with GPS map, chair/volume views, and 2D slices."""
        vd = self._select_and_load_volume()
        if vd is None:
            return

        vol = vd["data"]  # (nz, nx, ny)
        x_m, y_m, z_m = vd["x"], vd["y"], vd["z"]
        lat_arr, lon_arr = vd["lat"], vd["lon"]
        nz, nx, ny = vol.shape
        vmin, vmax = vd["vmin"], vd["vmax"]
        volume_path = vd["path"]

        # --- Segment calculation (40m blocks) ---
        SEG_WIDTH_M = 40.0
        x_min_m, x_max_m = float(x_m[0]), float(x_m[-1])
        total_len = x_max_m - x_min_m
        n_segs = max(1, int(np.ceil(total_len / SEG_WIDTH_M)))

        def _seg_indices(seg_idx):
            s_m = x_min_m + seg_idx * SEG_WIDTH_M
            e_m = min(s_m + SEG_WIDTH_M, x_max_m)
            i0 = int(np.searchsorted(x_m, s_m))
            i1 = int(np.searchsorted(x_m, e_m, side="right"))
            i1 = max(i0 + 1, min(i1, nx))
            return i0, i1

        # --- Theme colors ---
        BD = "#1e1e2e"; BP = "#2b2d3e"; BH = "#2d2d42"
        FT = "#ccd6f6"; FD = "#8892b0"; AC = "#007acc"
        _btn_kw = {"bg": "#3a3d52", "fg": "white", "activebackground": "#4a4d62",
                   "activeforeground": "white", "relief": "flat", "bd": 0,
                   "padx": 8, "pady": 3, "font": ("Segoe UI", 9), "cursor": "hand2"}
        _lbl_kw = {"bg": BP, "fg": FD, "font": ("Segoe UI", 9)}
        _scale_kw = {"bg": BP, "fg": FT, "troughcolor": "#3a3d52", "highlightthickness": 0,
                     "font": ("Consolas", 8), "orient": tk.HORIZONTAL, "length": 200}

        # === POPUP WINDOW ===
        win = tk.Toplevel(self.root)
        win.title(f"3D Volume — {os.path.basename(volume_path)}")
        win.geometry("1400x920")
        win.configure(bg=BD)
        try:
            win.state("zoomed")
        except Exception:
            pass

        # === HEADER BAR ===
        hdr = tk.Frame(win, bg=BH, padx=8, pady=4)
        hdr.pack(fill=tk.X, side=tk.TOP)

        seg_var = tk.IntVar(value=0)
        seg_label = tk.Label(hdr, text=f"Segment 1 / {n_segs}", bg=BH, fg=FT,
                             font=("Segoe UI", 10, "bold"))
        seg_label.pack(side=tk.LEFT, padx=(0, 10))

        def _nav_seg(delta):
            s = max(0, min(n_segs - 1, seg_var.get() + delta))
            seg_var.set(s)
            _on_segment_change()

        tk.Button(hdr, text="\u25c0 Prev", command=lambda: _nav_seg(-1), **_btn_kw).pack(side=tk.LEFT, padx=2)
        tk.Button(hdr, text="Next \u25b6", command=lambda: _nav_seg(1), **_btn_kw).pack(side=tk.LEFT, padx=2)

        # View mode
        view_mode = tk.StringVar(value="chair")
        tk.Label(hdr, text="  View:", **{"bg": BH, "fg": FD, "font": ("Segoe UI", 9)}).pack(side=tk.LEFT, padx=(16, 2))
        for txt, val in [("Chair", "chair"), ("Volume", "volume")]:
            tk.Radiobutton(hdr, text=txt, variable=view_mode, value=val,
                           bg=BH, fg=FT, selectcolor="#3a3d52", activebackground=BH,
                           font=("Segoe UI", 9), command=lambda: _update_all()).pack(side=tk.LEFT, padx=2)

        # Colormap
        tk.Label(hdr, text="  Cmap:", **{"bg": BH, "fg": FD, "font": ("Segoe UI", 9)}).pack(side=tk.LEFT, padx=(12, 2))
        cmap_var = tk.StringVar(value="seismic")
        cmap_combo = ttk.Combobox(hdr, textvariable=cmap_var, state="readonly", width=10,
                                  values=["gray", "seismic", "RdBu_r", "coolwarm", "viridis", "magma", "plasma", "hot"])
        cmap_combo.pack(side=tk.LEFT, padx=2)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: _update_all())

        # Tile server for map
        if _TKMAPVIEW_AVAILABLE:
            tk.Label(hdr, text="  Map:", **{"bg": BH, "fg": FD, "font": ("Segoe UI", 9)}).pack(side=tk.LEFT, padx=(12, 2))
            tile_var = tk.StringVar(value="OpenStreetMap")
            _TS = [("OpenStreetMap", "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", 19),
                   ("CartoDB Positron", "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", 20),
                   ("CartoDB Dark", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", 20),
                   ("Google Satellite", "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 22),
                   ("Google Hybrid", "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", 22),
                   ("ESRI Imagery", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19)]
            tile_combo = ttk.Combobox(hdr, textvariable=tile_var, state="readonly", width=16,
                                     values=[t[0] for t in _TS])
            tile_combo.pack(side=tk.LEFT, padx=2)

        # Save button
        def _save_view():
            fname = filedialog.asksaveasfilename(parent=win, title="Save View", defaultextension=".png",
                                                 initialdir=_OUTPUT_DIR,
                                                 filetypes=[("PNG", "*.png"), ("All", "*.*")])
            if fname:
                fig3d.savefig(fname, dpi=300, bbox_inches="tight", facecolor=BD)
        tk.Button(hdr, text="Save", command=_save_view, **_btn_kw).pack(side=tk.RIGHT, padx=4)

        # === MAIN AREA: GPS map (left) + 3D view (right) ===
        main_pw = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=BD, sashwidth=4, sashrelief="flat")
        main_pw.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # -- GPS MAP --
        map_frame = tk.Frame(main_pw, bg=BD)
        main_pw.add(map_frame, width=320, minsize=200)
        map_widget = None
        _map_seg_path = [None]  # mutable ref for segment highlight

        if _TKMAPVIEW_AVAILABLE and lat_arr is not None and lon_arr is not None:
            map_widget = tkintermapview.TkinterMapView(map_frame, corner_radius=0)
            map_widget.pack(fill=tk.BOTH, expand=True)
            # Draw full survey path (dim)
            valid = np.isfinite(lat_arr) & np.isfinite(lon_arr) & (np.abs(lat_arr) > 1) & (np.abs(lon_arr) > 1)
            if np.sum(valid) > 1:
                full_path = list(zip(lat_arr[valid].tolist(), lon_arr[valid].tolist()))
                map_widget.set_path(full_path, color="#555555", width=2)
            map_widget.set_position(float(np.nanmean(lat_arr[valid])), float(np.nanmean(lon_arr[valid])))
            map_widget.set_zoom(16)

            def _on_tile_change(_e=None):
                name = tile_var.get()
                for lbl, url, mz in _TS:
                    if lbl == name:
                        map_widget.set_tile_server(url, max_zoom=mz)
                        break
            tile_combo.bind("<<ComboboxSelected>>", _on_tile_change)
        else:
            tk.Label(map_frame, text="No GPS data", bg=BD, fg=FD, font=("Segoe UI", 11)).pack(expand=True)

        # -- 3D VIEW --
        view_frame = tk.Frame(main_pw, bg=BD)
        main_pw.add(view_frame, minsize=500)

        fig3d = plt.Figure(figsize=(8, 5.5), dpi=100, facecolor=BD)
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.set_facecolor(BD)
        canvas3d = FigureCanvasTkAgg(fig3d, master=view_frame)
        canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === CONTROL BAR ===
        ctrl = tk.Frame(win, bg=BP, padx=8, pady=4)
        ctrl.pack(fill=tk.X, side=tk.TOP)

        # Slice sliders
        zi_var = tk.IntVar(value=nz // 4)
        yi_var = tk.IntVar(value=ny // 2)
        xi_var = tk.IntVar(value=0)
        opacity_var = tk.DoubleVar(value=0.85)

        c = 0
        for lbl_txt, var, mx in [("Depth(Z):", zi_var, nz - 1), ("Channel(Y):", yi_var, ny - 1)]:
            tk.Label(ctrl, text=lbl_txt, **_lbl_kw).grid(row=0, column=c, padx=(0, 2)); c += 1
            tk.Scale(ctrl, variable=var, from_=0, to=max(mx, 0), command=lambda v: _update_all(), **_scale_kw).grid(row=0, column=c, padx=2); c += 1

        tk.Label(ctrl, text="Opacity:", **_lbl_kw).grid(row=0, column=c, padx=(8, 2)); c += 1
        tk.Scale(ctrl, variable=opacity_var, from_=0.3, to=1.0, resolution=0.05,
                 command=lambda v: _update_all(), **_scale_kw).grid(row=0, column=c, padx=2); c += 1

        # Chair-specific controls
        show_top = tk.BooleanVar(value=True)
        show_inline = tk.BooleanVar(value=True)
        show_cross = tk.BooleanVar(value=True)
        _chk_kw = {"bg": BP, "fg": FT, "selectcolor": "#3a3d52", "activebackground": BP,
                   "font": ("Segoe UI", 8), "command": lambda: _update_all()}
        tk.Checkbutton(ctrl, text="Top", variable=show_top, **_chk_kw).grid(row=0, column=c, padx=4); c += 1
        tk.Checkbutton(ctrl, text="Inline", variable=show_inline, **_chk_kw).grid(row=0, column=c, padx=4); c += 1
        tk.Checkbutton(ctrl, text="Xline", variable=show_cross, **_chk_kw).grid(row=0, column=c, padx=4); c += 1

        # === BOTTOM: 2D SLICES ===
        bot_pw = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=BD, sashwidth=4, sashrelief="flat")
        bot_pw.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=False)

        fig_xz = plt.Figure(figsize=(6, 2.8), dpi=100, facecolor=BD)
        ax_xz = fig_xz.add_subplot(111)
        ax_xz.set_facecolor(BD)
        canvas_xz = FigureCanvasTkAgg(fig_xz, master=bot_pw)
        f_xz = canvas_xz.get_tk_widget()
        f_xz.configure(height=250)
        bot_pw.add(f_xz, minsize=300)

        fig_xy = plt.Figure(figsize=(4, 2.8), dpi=100, facecolor=BD)
        ax_xy = fig_xy.add_subplot(111)
        ax_xy.set_facecolor(BD)
        canvas_xy = FigureCanvasTkAgg(fig_xy, master=bot_pw)
        f_xy = canvas_xy.get_tk_widget()
        f_xy.configure(height=250)
        bot_pw.add(f_xy, minsize=200)

        # === RENDERING HELPERS ===
        def _mappable():
            cn = cmap_var.get() or "seismic"
            try:
                cmp = plt.get_cmap(cn)
            except Exception:
                cmp = plt.get_cmap("seismic")
            return cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmp)

        def _style_ax3d(ax):
            ax.set_facecolor(BD)
            ax.tick_params(colors=FD, labelsize=7)
            ax.xaxis.label.set_color(FT)
            ax.yaxis.label.set_color(FT)
            ax.zaxis.label.set_color(FT)
            try:
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor(FD)
                ax.yaxis.pane.set_edgecolor(FD)
                ax.zaxis.pane.set_edgecolor(FD)
            except Exception:
                pass

        def _style_ax2d(ax, title=""):
            ax.set_facecolor("#1a1b2e")
            ax.tick_params(colors=FD, labelsize=7)
            ax.set_title(title, color=FT, fontsize=9)
            ax.xaxis.label.set_color(FT)
            ax.yaxis.label.set_color(FT)
            for sp in ax.spines.values():
                sp.set_color("#3a3d52")

        def _wireframe(ax, x0, x1, y0, y1, z0, z1):
            """Draw bounding wireframe on 3D axes."""
            c = "#888888"
            for zv in [z0, z1]:
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        [zv, zv, zv, zv, zv], color=c, linewidth=0.6, alpha=0.7)
            for xv, yv in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                ax.plot([xv, xv], [yv, yv], [z0, z1], color=c, linewidth=0.6, alpha=0.7)

        # === MAIN UPDATE FUNCTION ===
        def _update_all():
            try:
                _update_all_impl()
            except Exception as _err:
                import traceback; traceback.print_exc()
                try:
                    from tkinter import messagebox
                    messagebox.showerror("3D Vol Render Error", str(_err), parent=win)
                except Exception:
                    pass

        def _update_all_impl():
            si = seg_var.get()
            i0, i1 = _seg_indices(si)
            seg_label.config(text=f"Segment {si + 1} / {n_segs}  ({x_m[i0]:.0f}–{x_m[min(i1 - 1, nx - 1)]:.0f} m)")

            zi = max(0, min(nz - 1, zi_var.get()))
            yi = max(0, min(ny - 1, yi_var.get()))
            seg_nx = i1 - i0
            xi_local = seg_nx // 2  # crossline in middle of segment
            alpha = opacity_var.get()
            mp = _mappable()

            seg_x = x_m[i0:i1]
            seg_z = z_m
            seg_y = y_m

            # --- Update GPS map highlight ---
            if map_widget is not None and lat_arr is not None:
                # Remove old highlight
                if _map_seg_path[0] is not None:
                    try:
                        _map_seg_path[0].delete()
                    except Exception:
                        pass
                seg_lat = lat_arr[i0:i1]
                seg_lon = lon_arr[i0:i1]
                v = np.isfinite(seg_lat) & np.isfinite(seg_lon) & (np.abs(seg_lat) > 1)
                if np.sum(v) > 1:
                    sp = list(zip(seg_lat[v].tolist(), seg_lon[v].tolist()))
                    _map_seg_path[0] = map_widget.set_path(sp, color="#e74c3c", width=4)
                    map_widget.set_position(float(np.mean(seg_lat[v])), float(np.mean(seg_lon[v])))

            # --- 3D VIEW ---
            ax3d.clear()
            _style_ax3d(ax3d)
            mode = view_mode.get()
            seg_data = vol[:, i0:i1, :]  # (nz, seg_nx, ny)

            zs = max(1, nz // 60)
            xs = max(1, seg_nx // 80)
            ys = max(1, ny // 30)

            x_ax = seg_x[::xs]
            y_ax = seg_y[::ys]
            z_ax = seg_z[::zs]

            x0v, x1v = float(seg_x[0]), float(seg_x[-1])
            y0v, y1v = float(seg_y[0]), float(seg_y[-1])
            z0v, z1v = float(seg_z[0]), float(seg_z[-1])

            if mode == "chair":
                # --- CHAIR VIEW: 3 orthogonal cutting planes ---
                # Top face (depth slice at zi)
                if show_top.get():
                    Yg, Xg = np.meshgrid(y_ax, x_ax, indexing="ij")
                    Zg = np.full_like(Xg, seg_z[zi])
                    sl = seg_data[zi, ::xs, ::ys].T
                    ax3d.plot_surface(Xg, Yg, Zg, facecolors=mp.to_rgba(sl),
                                     rstride=1, cstride=1, linewidth=0, antialiased=False,
                                     shade=False, alpha=alpha)

                # Inline face (Y fixed at yi)
                if show_inline.get():
                    Zg2, Xg2 = np.meshgrid(z_ax, x_ax, indexing="ij")
                    Yg2 = np.full_like(Zg2, seg_y[yi])
                    sl2 = seg_data[::zs, ::xs, yi]
                    ax3d.plot_surface(Xg2, Yg2, Zg2, facecolors=mp.to_rgba(sl2),
                                     rstride=1, cstride=1, linewidth=0, antialiased=False,
                                     shade=False, alpha=alpha)

                # Crossline face (X fixed at xi_local)
                if show_cross.get():
                    xi_abs = min(seg_nx - 1, xi_local)
                    Zg3, Yg3 = np.meshgrid(z_ax, y_ax, indexing="ij")
                    Xg3 = np.full_like(Zg3, seg_x[xi_abs])
                    sl3 = seg_data[::zs, xi_abs, ::ys]
                    ax3d.plot_surface(Xg3, Yg3, Zg3, facecolors=mp.to_rgba(sl3),
                                     rstride=1, cstride=1, linewidth=0, antialiased=False,
                                     shade=False, alpha=alpha)

            elif mode == "volume":
                # --- VOLUME VIEW: 6-face solid cube ---
                # TOP (Z=0 surface)
                Yg, Xg = np.meshgrid(y_ax, x_ax, indexing="ij")
                ax3d.plot_surface(Xg, Yg, np.full_like(Xg, z0v),
                                  facecolors=mp.to_rgba(seg_data[0, ::xs, ::ys].T),
                                  rstride=1, cstride=1, linewidth=0, antialiased=False,
                                  shade=False, alpha=alpha)
                # BOTTOM (Z=max depth)
                ax3d.plot_surface(Xg, Yg, np.full_like(Xg, z1v),
                                  facecolors=mp.to_rgba(seg_data[-1, ::xs, ::ys].T),
                                  rstride=1, cstride=1, linewidth=0, antialiased=False,
                                  shade=False, alpha=alpha)
                # FRONT (Y=0 first channel)
                Zg4, Xg4 = np.meshgrid(z_ax, x_ax, indexing="ij")
                ax3d.plot_surface(Xg4, np.full_like(Xg4, y0v), Zg4,
                                  facecolors=mp.to_rgba(seg_data[::zs, ::xs, 0]),
                                  rstride=1, cstride=1, linewidth=0, antialiased=False,
                                  shade=False, alpha=alpha)
                # BACK (Y=max last channel)
                ax3d.plot_surface(Xg4, np.full_like(Xg4, y1v), Zg4,
                                  facecolors=mp.to_rgba(seg_data[::zs, ::xs, -1]),
                                  rstride=1, cstride=1, linewidth=0, antialiased=False,
                                  shade=False, alpha=alpha)
                # LEFT (X=seg_start)
                Zg5, Yg5 = np.meshgrid(z_ax, y_ax, indexing="ij")
                ax3d.plot_surface(np.full_like(Zg5, x0v), Yg5, Zg5,
                                  facecolors=mp.to_rgba(seg_data[::zs, 0, ::ys]),
                                  rstride=1, cstride=1, linewidth=0, antialiased=False,
                                  shade=False, alpha=alpha)
                # RIGHT (X=seg_end)
                ax3d.plot_surface(np.full_like(Zg5, x1v), Yg5, Zg5,
                                  facecolors=mp.to_rgba(seg_data[::zs, -1, ::ys]),
                                  rstride=1, cstride=1, linewidth=0, antialiased=False,
                                  shade=False, alpha=alpha)

            # Wireframe + labels (both modes)
            _wireframe(ax3d, x0v, x1v, y0v, y1v, z0v, z1v)
            ax3d.set_xlim(x0v, x1v)
            ax3d.set_ylim(y0v, y1v)
            ax3d.set_zlim(z1v, z0v)  # depth inverted
            ax3d.set_xlabel("Distance (m)", fontsize=8)
            ax3d.set_ylabel("Width (m)", fontsize=8)
            ax3d.set_zlabel("Depth (m)", fontsize=8)
            ax3d.view_init(elev=25, azim=-45)
            canvas3d.draw_idle()

            # --- 2D XZ SLICE (B-scan at channel yi) ---
            ax_xz.clear()
            _style_ax2d(ax_xz, f"B-scan (Ch {yi}, Y={seg_y[yi]:.2f}m)")
            xz_data = seg_data[:, :, yi]
            ax_xz.imshow(xz_data, aspect="auto", cmap=mp.cmap, norm=mp.norm,
                         extent=[float(seg_x[0]), float(seg_x[-1]), float(seg_z[-1]), float(seg_z[0])],
                         origin="upper")
            ax_xz.set_xlabel("Distance (m)", fontsize=8)
            ax_xz.set_ylabel("Depth (m)", fontsize=8)
            fig_xz.tight_layout(pad=0.5)
            canvas_xz.draw_idle()

            # --- 2D XY SLICE (depth map at zi) ---
            ax_xy.clear()
            _style_ax2d(ax_xy, f"Depth map (Z={seg_z[zi]:.2f}m)")
            xy_data = seg_data[zi, :, :].T
            ax_xy.imshow(xy_data, aspect="auto", cmap=mp.cmap, norm=mp.norm,
                         extent=[float(seg_x[0]), float(seg_x[-1]), float(seg_y[-1]), float(seg_y[0])],
                         origin="upper")
            ax_xy.set_xlabel("Distance (m)", fontsize=8)
            ax_xy.set_ylabel("Width (m)", fontsize=8)
            fig_xy.tight_layout(pad=0.5)
            canvas_xy.draw_idle()

        def _on_segment_change():
            _update_all()

        # Cleanup on close
        def _on_close_vol():
            plt.close(fig3d); plt.close(fig_xz); plt.close(fig_xy)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close_vol)

        # Initial draw
        _update_all()

    # ------------------------------------------------------------------
    # CHAIR DISPLAY — 3D chair-cut volume with smooth interpolated surfaces
    # ------------------------------------------------------------------
    def open_chair_volume_viewer(self):
        """Open chair-cut 3D volume viewer: corner cutaway with smooth surfaces, GPS map, section slices."""
        from scipy.ndimage import zoom as ndizoom
        from concurrent.futures import ThreadPoolExecutor

        vd = self._select_and_load_volume()
        if vd is None:
            return

        vol = vd["data"]  # (nz, nx, ny)
        x_m, y_m, z_m = vd["x"], vd["y"], vd["z"]
        lat_arr, lon_arr = vd["lat"], vd["lon"]
        nz, nx, ny = vol.shape
        vmin, vmax = vd["vmin"], vd["vmax"]
        volume_path = vd["path"]

        # Segment calculation (40m blocks)
        SEG_WIDTH_M = 40.0
        x_min_m, x_max_m = float(x_m[0]), float(x_m[-1])
        total_len = x_max_m - x_min_m
        n_segs = max(1, int(np.ceil(total_len / SEG_WIDTH_M)))

        def _seg_indices(seg_idx):
            s_m = x_min_m + seg_idx * SEG_WIDTH_M
            e_m = min(s_m + SEG_WIDTH_M, x_max_m)
            i0 = int(np.searchsorted(x_m, s_m))
            i1 = int(np.searchsorted(x_m, e_m, side="right"))
            i1 = max(i0 + 1, min(i1, nx))
            return i0, i1

        # Theme
        BD = "#1e1e2e"; BP = "#2b2d3e"; BH = "#2d2d42"
        FT = "#ccd6f6"; FD = "#8892b0"
        _btn_kw = {"bg": "#3a3d52", "fg": "white", "activebackground": "#4a4d62",
                   "activeforeground": "white", "relief": "flat", "bd": 0,
                   "padx": 8, "pady": 3, "font": ("Segoe UI", 9), "cursor": "hand2"}
        _lbl_kw = {"bg": BP, "fg": FD, "font": ("Segoe UI", 9)}

        # === POPUP ===
        win = tk.Toplevel(self.root)
        win.title(f"Chair Display \u2014 {os.path.basename(volume_path)}")
        win.geometry("1400x920")
        win.configure(bg=BD)
        try:
            win.state("zoomed")
        except Exception:
            pass

        # === HEADER ===
        hdr = tk.Frame(win, bg=BH, padx=8, pady=4)
        hdr.pack(fill=tk.X, side=tk.TOP)

        seg_var = tk.IntVar(value=0)
        seg_label = tk.Label(hdr, text=f"Segment 1 / {n_segs}", bg=BH, fg=FT,
                             font=("Segoe UI", 10, "bold"))
        seg_label.pack(side=tk.LEFT, padx=(0, 10))

        _pending_ch = [None]

        def _schedule(v=None):
            if _pending_ch[0] is not None:
                win.after_cancel(_pending_ch[0])
            _pending_ch[0] = win.after(50, _update_all)

        def _nav_seg(delta):
            s = max(0, min(n_segs - 1, seg_var.get() + delta))
            seg_var.set(s)
            _schedule()

        tk.Button(hdr, text="\u25c0 Prev", command=lambda: _nav_seg(-1), **_btn_kw).pack(side=tk.LEFT, padx=2)
        tk.Button(hdr, text="Next \u25b6", command=lambda: _nav_seg(1), **_btn_kw).pack(side=tk.LEFT, padx=2)

        # Colormap
        tk.Label(hdr, text="  Cmap:", bg=BH, fg=FD, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
        cmap_var = tk.StringVar(value="seismic")
        cmap_combo = ttk.Combobox(hdr, textvariable=cmap_var, state="readonly", width=10,
                                  values=["gray", "seismic", "RdBu_r", "coolwarm", "viridis", "magma", "plasma", "hot"])
        cmap_combo.pack(side=tk.LEFT, padx=2)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: _schedule())

        # Tile server
        tile_var = tk.StringVar(value="OpenStreetMap")
        if _TKMAPVIEW_AVAILABLE:
            tk.Label(hdr, text="  Map:", bg=BH, fg=FD, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
            _TS = [("OpenStreetMap", "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", 19),
                   ("CartoDB Positron", "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", 20),
                   ("CartoDB Dark", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", 20),
                   ("Google Satellite", "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 22),
                   ("Google Hybrid", "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", 22),
                   ("ESRI Imagery", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19)]
            tile_combo = ttk.Combobox(hdr, textvariable=tile_var, state="readonly", width=16,
                                     values=[t[0] for t in _TS])
            tile_combo.pack(side=tk.LEFT, padx=2)

        # Save
        def _save_view():
            fname = filedialog.asksaveasfilename(parent=win, title="Save View", defaultextension=".png",
                                                 initialdir=_OUTPUT_DIR,
                                                 filetypes=[("PNG", "*.png"), ("All", "*.*")])
            if fname:
                fig3d.savefig(fname, dpi=300, bbox_inches="tight", facecolor=BD)
        tk.Button(hdr, text="Save", command=_save_view, **_btn_kw).pack(side=tk.RIGHT, padx=4)

        # === MAIN: GPS map (left) + 3D (right) ===
        main_pw = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=BD, sashwidth=4, sashrelief="flat")
        main_pw.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # GPS MAP
        map_frame = tk.Frame(main_pw, bg=BD)
        main_pw.add(map_frame, width=320, minsize=200)
        map_widget = None
        _map_seg_path = [None]

        if _TKMAPVIEW_AVAILABLE and lat_arr is not None and lon_arr is not None:
            map_widget = tkintermapview.TkinterMapView(map_frame, corner_radius=0)
            map_widget.pack(fill=tk.BOTH, expand=True)
            valid = np.isfinite(lat_arr) & np.isfinite(lon_arr) & (np.abs(lat_arr) > 1) & (np.abs(lon_arr) > 1)
            if np.sum(valid) > 1:
                full_path = list(zip(lat_arr[valid].tolist(), lon_arr[valid].tolist()))
                map_widget.set_path(full_path, color="#555555", width=2)
            map_widget.set_position(float(np.nanmean(lat_arr[valid])), float(np.nanmean(lon_arr[valid])))
            map_widget.set_zoom(16)

            def _on_tile_change(_e=None):
                name = tile_var.get()
                for lbl, url, mz in _TS:
                    if lbl == name:
                        map_widget.set_tile_server(url, max_zoom=mz)
                        break
            tile_combo.bind("<<ComboboxSelected>>", _on_tile_change)
        else:
            tk.Label(map_frame, text="No GPS data", bg=BD, fg=FD, font=("Segoe UI", 11)).pack(expand=True)

        # 3D chair canvas
        view_frame = tk.Frame(main_pw, bg=BD)
        main_pw.add(view_frame, minsize=500)
        fig3d = plt.Figure(figsize=(8, 5.5), dpi=100, facecolor=BD)
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.set_facecolor(BD)
        canvas3d = FigureCanvasTkAgg(fig3d, master=view_frame)
        canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === CONTROL BAR ===
        ctrl = tk.Frame(win, bg=BP, padx=8, pady=4)
        ctrl.pack(fill=tk.X, side=tk.TOP)

        # Defaults: ~33% depth, ~50% distance, ~66% width → interior is visible
        _def_zi = max(1, nz // 3)
        _def_yi = max(1, ny * 2 // 3)
        zi_var = tk.IntVar(value=_def_zi)
        xi_var = tk.IntVar(value=1)      # will be set per segment in _update_all
        yi_var = tk.IntVar(value=_def_yi)

        _entry_kw = {"bg": "#2a2d42", "fg": "#ccd6f6", "insertbackground": "#ccd6f6",
                     "font": ("Consolas", 9), "relief": "flat", "bd": 2,
                     "highlightthickness": 1, "highlightcolor": "#4a4d62", "width": 6}

        c = 0
        for lbl_txt, var, rng_txt in [("Cut Z:", zi_var, f"1\u2013{nz-2}"),
                                       ("Cut X:", xi_var, "(auto)"),
                                       ("Cut Y:", yi_var, f"1\u2013{ny-2}")]:
            tk.Label(ctrl, text=lbl_txt, **_lbl_kw).grid(row=0, column=c, padx=(0, 2)); c += 1
            e = tk.Entry(ctrl, textvariable=var, **_entry_kw)
            e.grid(row=0, column=c, padx=2); c += 1
            e.bind("<Return>", lambda ev: _schedule())
            tk.Label(ctrl, text=rng_txt, bg=BP, fg="#555e78",
                     font=("Segoe UI", 7)).grid(row=0, column=c, padx=(0, 6)); c += 1

        tk.Button(ctrl, text="Render", command=lambda: _schedule(),
                  bg="#2e7d32", fg="white", activebackground="#388e3c",
                  activeforeground="white", relief="flat", bd=0,
                  padx=10, pady=3, font=("Segoe UI", 9, "bold"),
                  cursor="hand2").grid(row=0, column=c, padx=(8, 4)); c += 1

        # Progress bar
        _prog = ttk.Progressbar(ctrl, orient=tk.HORIZONTAL, length=120, mode="indeterminate")
        _prog.grid(row=0, column=c, padx=(4, 2)); c += 1
        _status_lbl = tk.Label(ctrl, text="Ready", bg=BP, fg=FD, font=("Segoe UI", 8))
        _status_lbl.grid(row=0, column=c, padx=4); c += 1

        # --- Rotation controls (row 1 — separate row to avoid overflow) ---
        elev_var = tk.IntVar(value=30)
        azim_var = tk.IntVar(value=135)
        _rot_step = tk.IntVar(value=15)

        _rot_btn_kw = dict(_btn_kw)
        _rot_btn_kw["font"] = ("Segoe UI", 8)

        def _rotate_only():
            """Lightweight rotation — only changes view angle, no re-smoothing."""
            ax3d.view_init(elev=elev_var.get(), azim=azim_var.get())
            canvas3d.draw_idle()

        def _rotate(d_elev, d_azim):
            try:
                step = int(_rot_step.get())
            except (tk.TclError, ValueError):
                step = 15
            elev_var.set(elev_var.get() + d_elev * step)
            azim_var.set(azim_var.get() + d_azim * step)
            _rotate_only()

        r1 = 0
        tk.Label(ctrl, text="Rot step:", **_lbl_kw).grid(row=1, column=r1, padx=(0, 2)); r1 += 1
        tk.Entry(ctrl, textvariable=_rot_step, **{**_entry_kw, "width": 3}).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Label(ctrl, text="\u00b0", bg=BP, fg="#555e78", font=("Segoe UI", 7)).grid(row=1, column=r1, padx=(0, 6)); r1 += 1
        tk.Button(ctrl, text="H\u2190", command=lambda: _rotate(0, -1), **_rot_btn_kw).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Button(ctrl, text="H\u2192", command=lambda: _rotate(0, 1), **_rot_btn_kw).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Button(ctrl, text="V\u2191", command=lambda: _rotate(1, 0), **_rot_btn_kw).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Button(ctrl, text="V\u2193", command=lambda: _rotate(-1, 0), **_rot_btn_kw).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Button(ctrl, text="Ang", command=lambda: _rotate(1, 1), **_rot_btn_kw).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Label(ctrl, text="  Elev:", **_lbl_kw).grid(row=1, column=r1, padx=(8, 2)); r1 += 1
        tk.Entry(ctrl, textvariable=elev_var, **{**_entry_kw, "width": 4}).grid(row=1, column=r1, padx=1); r1 += 1
        tk.Label(ctrl, text="  Azim:", **_lbl_kw).grid(row=1, column=r1, padx=(4, 2)); r1 += 1
        tk.Entry(ctrl, textvariable=azim_var, **{**_entry_kw, "width": 4}).grid(row=1, column=r1, padx=1); r1 += 1

        # === BOTTOM: 2D reference slices (XZ, XY, YZ) ===
        bot_pw = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=BD, sashwidth=4, sashrelief="flat")
        bot_pw.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=False)

        fig_bxz = plt.Figure(figsize=(5, 2.4), dpi=100, facecolor=BD)
        ax_bxz = fig_bxz.add_subplot(111); ax_bxz.set_facecolor(BD)
        canvas_bxz = FigureCanvasTkAgg(fig_bxz, master=bot_pw)
        w_bxz = canvas_bxz.get_tk_widget(); w_bxz.configure(height=210)
        bot_pw.add(w_bxz, minsize=220)

        fig_bxy = plt.Figure(figsize=(3.5, 2.4), dpi=100, facecolor=BD)
        ax_bxy = fig_bxy.add_subplot(111); ax_bxy.set_facecolor(BD)
        canvas_bxy = FigureCanvasTkAgg(fig_bxy, master=bot_pw)
        w_bxy = canvas_bxy.get_tk_widget(); w_bxy.configure(height=210)
        bot_pw.add(w_bxy, minsize=180)

        fig_byz = plt.Figure(figsize=(3, 2.4), dpi=100, facecolor=BD)
        ax_byz = fig_byz.add_subplot(111); ax_byz.set_facecolor(BD)
        canvas_byz = FigureCanvasTkAgg(fig_byz, master=bot_pw)
        w_byz = canvas_byz.get_tk_widget(); w_byz.configure(height=210)
        bot_pw.add(w_byz, minsize=140)

        # === HELPERS ===
        def _mappable():
            cn = cmap_var.get() or "seismic"
            try:
                cmp = plt.get_cmap(cn)
            except Exception:
                cmp = plt.get_cmap("seismic")
            return cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmp)

        def _style3d(ax):
            ax.set_facecolor(BD)

        def _style2d(ax, title=""):
            ax.set_facecolor("#1a1b2e")
            ax.tick_params(colors=FD, labelsize=7)
            ax.set_title(title, color=FT, fontsize=9)
            for a in (ax.xaxis, ax.yaxis):
                a.label.set_color(FT)
            for sp in ax.spines.values():
                sp.set_color("#3a3d52")

        _use_gpu = (hasattr(self, "_use_gpu_var") and _GPU_BACKEND == "cupy")

        def _smooth(d2d):
            """Upsample 2D array with per-axis cubic interpolation (GPU or CPU)."""
            if d2d.shape[0] < 2 or d2d.shape[1] < 2:
                return d2d.astype(float)
            target = 250
            f0 = max(1.0, target / d2d.shape[0])
            f1 = max(1.0, target / d2d.shape[1])
            max_px = 800000
            out_px = d2d.shape[0] * f0 * d2d.shape[1] * f1
            if out_px > max_px:
                s = np.sqrt(max_px / out_px)
                f0 *= s; f1 *= s
            if f0 < 1.05 and f1 < 1.05:
                return d2d.astype(float)
            # GPU path (CuPy)
            if _use_gpu and self._use_gpu_var.get():
                try:
                    import cupy as _cp
                    import cupyx.scipy.ndimage as _cpndimage
                    d_gpu = _cp.asarray(d2d.astype(float))
                    result = _cpndimage.zoom(d_gpu, (f0, f1), order=3)
                    return _cp.asnumpy(result)
                except Exception:
                    pass  # fallback to CPU
            return ndizoom(d2d.astype(float), (f0, f1), order=3)

        # === MAIN UPDATE ===
        _prev_key = [None]  # cache key: (si, zi, xi, yi, cmap) → avoid redundant smoothing

        def _update_all():
            _pending_ch[0] = None
            _prog.start(10)
            _status_lbl.config(text="Rendering...")
            win.update_idletasks()
            try:
                _update_all_core()
            except Exception as _err:
                import traceback; traceback.print_exc()
                _status_lbl.config(text="Error!")
                try:
                    messagebox.showerror("Chair Render Error", str(_err), parent=win)
                except Exception:
                    pass
            finally:
                _prog.stop()
                win.update_idletasks()

        def _update_all_core():
            si = seg_var.get()
            i0, i1 = _seg_indices(si)
            seg_nx = i1 - i0
            seg_label.config(text=f"Segment {si + 1} / {n_segs}  "
                             f"({x_m[i0]:.0f}\u2013{x_m[min(i1 - 1, nx - 1)]:.0f} m)")

            # Clamp & read cut values
            try:
                zi = max(1, min(nz - 2, int(zi_var.get())))
            except (tk.TclError, ValueError):
                zi = max(1, nz // 3)
            try:
                xi = max(1, min(seg_nx - 2, int(xi_var.get())))
            except (tk.TclError, ValueError):
                xi = max(1, seg_nx // 2)
            try:
                yi = max(1, min(ny - 2, int(yi_var.get())))
            except (tk.TclError, ValueError):
                yi = max(1, ny * 2 // 3)
            # Set xi default when segment changes and xi=1
            if xi <= 1:
                xi = max(1, seg_nx // 2)
                xi_var.set(xi)
            alpha = 1.0
            mp = _mappable()

            seg_data = vol[:, i0:i1, :]  # (nz, seg_nx, ny)
            seg_x = x_m[i0:i1]

            x0v, x1v = float(seg_x[0]), float(seg_x[-1])
            y0v, y1v = float(y_m[0]), float(y_m[-1])
            z0v, z1v = float(z_m[0]), float(z_m[-1])
            xcv = float(seg_x[xi])   # X cut position
            ycv = float(y_m[yi])     # Y cut position
            zcv = float(z_m[zi])     # Z cut position

            # GPS map highlight
            if map_widget is not None and lat_arr is not None:
                if _map_seg_path[0] is not None:
                    try:
                        _map_seg_path[0].delete()
                    except Exception:
                        pass
                seg_lat, seg_lon = lat_arr[i0:i1], lon_arr[i0:i1]
                v = np.isfinite(seg_lat) & np.isfinite(seg_lon) & (np.abs(seg_lat) > 1)
                if np.sum(v) > 1:
                    sp = list(zip(seg_lat[v].tolist(), seg_lon[v].tolist()))
                    _map_seg_path[0] = map_widget.set_path(sp, color="#e74c3c", width=4)
                    map_widget.set_position(float(np.mean(seg_lat[v])), float(np.mean(seg_lon[v])))

            # ---- 3D CHAIR RENDERING (parallel smoothing) ----
            ax3d.clear()
            _style3d(ax3d)

            # Build face list: (kind, data2d, coord1, coord2, fixed_val)
            faces = []
            # -- HORIZONTAL SURFACES --
            if yi > 0:
                faces.append(("h", seg_data[0, :, :yi].T, seg_x, y_m[:yi], z0v))
            if xi < seg_nx - 1 and yi < ny:
                faces.append(("h", seg_data[0, xi:, yi:].T, seg_x[xi:], y_m[yi:], z0v))
            if xi > 0 and (ny - yi) > 0:
                faces.append(("h", seg_data[zi, :xi, yi:].T, seg_x[:xi], y_m[yi:], zcv))
            faces.append(("h", seg_data[-1, :, :].T, seg_x, y_m, z1v))
            # -- INTERNAL CUT WALLS --
            if xi > 0 and (ny - yi) > 0 and zi > 0:
                faces.append(("yz", seg_data[:zi, xi, yi:], y_m[yi:], z_m[:zi], xcv))
            if yi > 0 and xi > 0 and zi > 0:
                faces.append(("xz", seg_data[:zi, :xi, yi], seg_x[:xi], z_m[:zi], ycv))
            # -- OUTER FACES --
            if xi > 0 and (nz - zi) > 1:
                faces.append(("xz", seg_data[zi:, :xi, -1], seg_x[:xi], z_m[zi:], y1v))
            if (seg_nx - xi) > 1:
                faces.append(("xz", seg_data[:, xi:, -1], seg_x[xi:], z_m, y1v))
            faces.append(("xz", seg_data[:, :, 0], seg_x, z_m, y0v))
            if yi > 0:
                faces.append(("yz", seg_data[:, 0, :yi], y_m[:yi], z_m, x0v))
            if (ny - yi) > 0 and (nz - zi) > 1:
                faces.append(("yz", seg_data[zi:, 0, yi:], y_m[yi:], z_m[zi:], x0v))
            faces.append(("yz", seg_data[:, -1, :], y_m, z_m, x1v))

            # Smoothing: GPU (sequential) or CPU (parallel)
            ncpus = max(2, self._ncpus_var.get()) if hasattr(self, "_ncpus_var") else 4
            gpu_active = _use_gpu and self._use_gpu_var.get()
            _device_lbl = "GPU" if gpu_active else f"{ncpus} CPUs"
            _status_lbl.config(text=f"Smoothing {len(faces)} faces ({_device_lbl})...")
            win.update_idletasks()

            smoothed = [None] * len(faces)
            if gpu_active:
                # CUDA contexts are per-thread — run sequentially on GPU
                for i in range(len(faces)):
                    smoothed[i] = _smooth(faces[i][1])
            else:
                def _smooth_face(idx):
                    return idx, _smooth(faces[idx][1])
                with ThreadPoolExecutor(max_workers=max(1, min(ncpus, len(faces)))) as pool:
                    for idx, sm in pool.map(lambda i: _smooth_face(i), range(len(faces))):
                        smoothed[idx] = sm

            # Paint all faces (sequential — matplotlib not thread-safe)
            _status_lbl.config(text="Painting surfaces...")
            win.update_idletasks()
            for fi, (kind, _raw, c1, c2, fv) in enumerate(faces):
                sm = smoothed[fi]
                if kind == "h":
                    nsy, nsx = sm.shape
                    xs = np.linspace(float(c1[0]), float(c1[-1]), nsx)
                    ys = np.linspace(float(c2[0]), float(c2[-1]), nsy)
                    Xg, Yg = np.meshgrid(xs, ys)
                    Zg = np.full_like(Xg, fv)
                    ax3d.plot_surface(Xg, Yg, Zg, facecolors=mp.to_rgba(sm),
                                      rstride=1, cstride=1, linewidth=0,
                                      antialiased=True, shade=False, alpha=alpha)
                elif kind == "xz":
                    nsz, nsx = sm.shape
                    xs = np.linspace(float(c1[0]), float(c1[-1]), nsx)
                    zs = np.linspace(float(c2[0]), float(c2[-1]), nsz)
                    Xg, Zg = np.meshgrid(xs, zs)
                    Yg = np.full_like(Xg, fv)
                    ax3d.plot_surface(Xg, Yg, Zg, facecolors=mp.to_rgba(sm),
                                      rstride=1, cstride=1, linewidth=0,
                                      antialiased=True, shade=False, alpha=alpha)
                elif kind == "yz":
                    nsz, nsy = sm.shape
                    ys = np.linspace(float(c1[0]), float(c1[-1]), nsy)
                    zs = np.linspace(float(c2[0]), float(c2[-1]), nsz)
                    Yg, Zg = np.meshgrid(ys, zs)
                    Xg = np.full_like(Yg, fv)
                    ax3d.plot_surface(Xg, Yg, Zg, facecolors=mp.to_rgba(sm),
                                      rstride=1, cstride=1, linewidth=0,
                                      antialiased=True, shade=False, alpha=alpha)

            # Cut colour for 2D crosshairs
            cc_col = "#ffaa00"

            # Axis limits + labels (no grid, no panes — clean solid look)
            ax3d.set_xlim(x0v, x1v); ax3d.set_ylim(y0v, y1v); ax3d.set_zlim(z1v, z0v)
            ax3d.set_xlabel("Distance (m)", fontsize=8, color=FT, labelpad=6)
            ax3d.set_ylabel("Width (m)", fontsize=8, color=FT, labelpad=6)
            ax3d.set_zlabel("Depth (m)", fontsize=8, color=FT, labelpad=6)
            ax3d.tick_params(colors=FD, labelsize=7)
            ax3d.grid(False)
            try:
                for a in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
                    a.pane.fill = False
                    a.pane.set_edgecolor("none")
                    a.line.set_color(FD)
                    a.line.set_linewidth(0.3)
            except Exception:
                pass

            ax3d.view_init(elev=elev_var.get(), azim=azim_var.get())

            # ---- CHAIR-CUT EDGES (view-dependent, includes cut boundary) ----
            _ec = "black"; _elw = 0.8; _ea = 0.8

            # Camera direction from elev/azim (z negated for inverted z-axis: zlim z1v→z0v)
            _elev_r = math.radians(elev_var.get())
            _azim_r = math.radians(azim_var.get())
            _cam_dx = math.cos(_elev_r) * math.cos(_azim_r)
            _cam_dy = math.cos(_elev_r) * math.sin(_azim_r)
            _cam_dz = -math.sin(_elev_r)  # Negated: z-axis inverted in display

            # 14 vertices of the chair-cut solid
            _cv = [
                (x0v, y0v, z0v),  # 0  top-front-left
                (x1v, y0v, z0v),  # 1  top-front-right
                (x1v, y1v, z0v),  # 2  top-back-right
                (xcv, y1v, z0v),  # 3  top-back at cut X
                (xcv, ycv, z0v),  # 4  top inner corner
                (x0v, ycv, z0v),  # 5  top-left at cut Y
                (x0v, y0v, z1v),  # 6  bottom-front-left
                (x1v, y0v, z1v),  # 7  bottom-front-right
                (x1v, y1v, z1v),  # 8  bottom-back-right
                (x0v, y1v, z1v),  # 9  bottom-back-left
                (x0v, ycv, zcv),  # 10 shelf-left at cut Y
                (xcv, ycv, zcv),  # 11 shelf inner corner (3D cut vertex)
                (xcv, y1v, zcv),  # 12 shelf-back at cut X
                (x0v, y1v, zcv),  # 13 shelf-back-left
            ]

            # 9 faces: (outward_normal, edge_list)
            _chair_faces = [
                ((0, 0, -1),  [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]),       # Top L-shape
                ((0, 0,  1),  [(6,7),(7,8),(8,9),(9,6)]),                   # Bottom
                ((0, -1, 0),  [(0,1),(1,7),(7,6),(6,0)]),                   # Front
                ((1,  0, 0),  [(1,2),(2,8),(8,7),(7,1)]),                   # Right
                ((0,  1, 0),  [(2,3),(3,12),(12,13),(13,9),(9,8),(8,2)]),   # Back (L)
                ((-1, 0, 0),  [(0,5),(5,10),(10,13),(13,9),(9,6),(6,0)]),   # Left (L)
                ((-1, 0, 0),  [(4,3),(3,12),(12,11),(11,4)]),               # Cut wall x=xcv
                ((0,  1, 0),  [(5,4),(4,11),(11,10),(10,5)]),               # Cut wall y=ycv
                ((0, 0, -1),  [(10,11),(11,12),(12,13),(13,10)]),           # Cut shelf z=zcv
            ]

            # Draw edges belonging to at least one camera-facing face
            _vis_edges = set()
            for (_fnx, _fny, _fnz), fedges in _chair_faces:
                if _fnx * _cam_dx + _fny * _cam_dy + _fnz * _cam_dz > 0:
                    for i, j in fedges:
                        _vis_edges.add((min(i, j), max(i, j)))

            for i, j in _vis_edges:
                xa, ya, za = _cv[i]
                xb, yb, zb = _cv[j]
                ax3d.plot([xa, xb], [ya, yb], [za, zb], color=_ec, lw=_elw, alpha=_ea, zorder=50)

            # Colorbar
            if _cbar_ref[0] is not None:
                try:
                    _cbar_ref[0].remove()
                except Exception:
                    pass
            _cbar_ref[0] = fig3d.colorbar(mp, ax=ax3d, shrink=0.55, pad=0.01,
                                           label="Amplitude", aspect=30)
            _cbar_ref[0].ax.yaxis.set_tick_params(color=FT, labelcolor=FT, labelsize=7)
            _cbar_ref[0].set_label("Amplitude", color=FT, fontsize=8)

            _status_lbl.config(text="Drawing...")
            win.update_idletasks()
            canvas3d.draw_idle()

            # ---- 2D REFERENCE SLICES ----
            # XZ B-scan at Y-cut
            ax_bxz.clear(); _style2d(ax_bxz, f"XZ Slice (Y={ycv:.2f}m)")
            ax_bxz.imshow(seg_data[:, :, yi], aspect="auto", cmap=mp.cmap, norm=mp.norm,
                          extent=[x0v, x1v, z1v, z0v], origin="upper")
            ax_bxz.axvline(xcv, color=cc_col, lw=0.8, ls="--", alpha=0.7)
            ax_bxz.axhline(zcv, color=cc_col, lw=0.8, ls="--", alpha=0.7)
            ax_bxz.set_xlabel("Distance (m)", fontsize=8)
            ax_bxz.set_ylabel("Depth (m)", fontsize=8)
            fig_bxz.tight_layout(pad=0.5); canvas_bxz.draw_idle()

            # XY depth map at Z-cut
            ax_bxy.clear(); _style2d(ax_bxy, f"XY Slice (Z={zcv:.2f}m)")
            ax_bxy.imshow(seg_data[zi, :, :].T, aspect="auto", cmap=mp.cmap, norm=mp.norm,
                          extent=[x0v, x1v, y1v, y0v], origin="upper")
            ax_bxy.axvline(xcv, color=cc_col, lw=0.8, ls="--", alpha=0.7)
            ax_bxy.axhline(ycv, color=cc_col, lw=0.8, ls="--", alpha=0.7)
            ax_bxy.set_xlabel("Distance (m)", fontsize=8)
            ax_bxy.set_ylabel("Width (m)", fontsize=8)
            fig_bxy.tight_layout(pad=0.5); canvas_bxy.draw_idle()

            # YZ crossline at X-cut
            ax_byz.clear(); _style2d(ax_byz, f"YZ Slice (X={xcv:.1f}m)")
            ax_byz.imshow(seg_data[:, xi, :], aspect="auto", cmap=mp.cmap, norm=mp.norm,
                          extent=[y0v, y1v, z1v, z0v], origin="upper")
            ax_byz.axhline(zcv, color=cc_col, lw=0.8, ls="--", alpha=0.7)
            ax_byz.axvline(ycv, color=cc_col, lw=0.8, ls="--", alpha=0.7)
            ax_byz.set_xlabel("Width (m)", fontsize=8)
            ax_byz.set_ylabel("Depth (m)", fontsize=8)
            fig_byz.tight_layout(pad=0.5); canvas_byz.draw_idle()

            _status_lbl.config(text=f"Done ({len(faces)} faces)")

        # Cleanup
        def _on_close():
            plt.close(fig3d); plt.close(fig_bxz); plt.close(fig_bxy); plt.close(fig_byz)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        # Colorbar (persistent)
        _cbar_ref = [None]

        # Auto-render on open: show window first, then render after 100ms
        win.after(100, _update_all)

    # ------------------------------------------------------------------
    # 3D SECTION DISPLAY — GPR-GUI-2026 style orthogonal slice viewer
    # ------------------------------------------------------------------
    def open_3d_section_viewer(self):
        """Open GPR-GUI-2026 style 3D section display with GPS map, XY/XZ/YZ panels, and click-to-slice crosshairs."""
        vd = self._select_and_load_volume()
        if vd is None:
            return

        vol = vd["data"]  # (nz, nx, ny)
        x_m, y_m, z_m = vd["x"], vd["y"], vd["z"]
        lat_arr, lon_arr = vd["lat"], vd["lon"]
        nz, nx, ny = vol.shape
        vmin, vmax = vd["vmin"], vd["vmax"]
        volume_path = vd["path"]

        # Segment calculation (40m blocks)
        SEG_WIDTH_M = 40.0
        x_min_m, x_max_m = float(x_m[0]), float(x_m[-1])
        total_len = x_max_m - x_min_m
        n_segs = max(1, int(np.ceil(total_len / SEG_WIDTH_M)))

        def _seg_indices(seg_idx):
            s_m = x_min_m + seg_idx * SEG_WIDTH_M
            e_m = min(s_m + SEG_WIDTH_M, x_max_m)
            i0 = int(np.searchsorted(x_m, s_m))
            i1 = int(np.searchsorted(x_m, e_m, side="right"))
            i1 = max(i0 + 1, min(i1, nx))
            return i0, i1

        # Theme
        BG = "#1e1e1e"; BP = "#2b2d3e"; BH = "#2d2d42"
        FT = "#ccd6f6"; FD = "#8892b0"
        _btn_kw = {"bg": "#3a3d52", "fg": "white", "activebackground": "#4a4d62",
                   "activeforeground": "white", "relief": "flat", "bd": 0,
                   "padx": 8, "pady": 3, "font": ("Segoe UI", 9), "cursor": "hand2"}
        _lbl_kw = {"bg": BP, "fg": FD, "font": ("Segoe UI", 9)}
        _scale_kw = {"bg": BP, "fg": FT, "troughcolor": "#3a3d52", "highlightthickness": 0,
                     "font": ("Consolas", 8), "orient": tk.HORIZONTAL, "length": 220}

        # === POPUP ===
        win = tk.Toplevel(self.root)
        win.title(f"3D Section Display \u2014 {os.path.basename(volume_path)}")
        win.geometry("1400x920")
        win.configure(bg=BG)
        try:
            win.state("zoomed")
        except Exception:
            pass

        # === HEADER ===
        hdr = tk.Frame(win, bg=BH, padx=8, pady=4)
        hdr.pack(fill=tk.X, side=tk.TOP)

        seg_var = tk.IntVar(value=0)
        seg_label = tk.Label(hdr, text=f"Segment 1 / {n_segs}", bg=BH, fg=FT,
                             font=("Segoe UI", 10, "bold"))
        seg_label.pack(side=tk.LEFT, padx=(0, 10))

        def _nav_seg(delta):
            s = max(0, min(n_segs - 1, seg_var.get() + delta))
            seg_var.set(s)
            _full_redraw()

        tk.Button(hdr, text="\u25c0 Prev", command=lambda: _nav_seg(-1), **_btn_kw).pack(side=tk.LEFT, padx=2)
        tk.Button(hdr, text="Next \u25b6", command=lambda: _nav_seg(1), **_btn_kw).pack(side=tk.LEFT, padx=2)

        # Colormap
        tk.Label(hdr, text="  Cmap:", bg=BH, fg=FD, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
        cmap_var = tk.StringVar(value="gray")
        cmap_combo = ttk.Combobox(hdr, textvariable=cmap_var, state="readonly", width=10,
                                  values=["gray", "seismic", "RdBu_r", "coolwarm", "viridis", "magma", "plasma", "hot", "bone"])
        cmap_combo.pack(side=tk.LEFT, padx=2)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: _full_redraw())

        # Tile server for GPS map
        tile_var = tk.StringVar(value="OpenStreetMap")
        if _TKMAPVIEW_AVAILABLE:
            tk.Label(hdr, text="  Map:", bg=BH, fg=FD, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
            _TS = [("OpenStreetMap", "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", 19),
                   ("CartoDB Positron", "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", 20),
                   ("CartoDB Dark", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", 20),
                   ("Google Satellite", "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 22),
                   ("Google Hybrid", "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", 22),
                   ("ESRI Imagery", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19)]
            tile_combo = ttk.Combobox(hdr, textvariable=tile_var, state="readonly", width=16,
                                     values=[t[0] for t in _TS])
            tile_combo.pack(side=tk.LEFT, padx=2)

        # Save
        def _save_view():
            fname = filedialog.asksaveasfilename(parent=win, title="Save Section View", defaultextension=".png",
                                                 initialdir=_OUTPUT_DIR,
                                                 filetypes=[("PNG", "*.png"), ("All", "*.*")])
            if fname:
                fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor=BG)
        tk.Button(hdr, text="Save", command=_save_view, **_btn_kw).pack(side=tk.RIGHT, padx=4)

        # === MAIN AREA: GPS map (left) + section panels (right) ===
        main_pw = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=BG, sashwidth=4, sashrelief="flat")
        main_pw.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # -- GPS MAP --
        map_frame = tk.Frame(main_pw, bg=BG)
        main_pw.add(map_frame, width=300, minsize=180)
        map_widget = None
        _map_seg_path = [None]

        if _TKMAPVIEW_AVAILABLE and lat_arr is not None and lon_arr is not None:
            map_widget = tkintermapview.TkinterMapView(map_frame, corner_radius=0)
            map_widget.pack(fill=tk.BOTH, expand=True)
            valid = np.isfinite(lat_arr) & np.isfinite(lon_arr) & (np.abs(lat_arr) > 1) & (np.abs(lon_arr) > 1)
            if np.sum(valid) > 1:
                full_path = list(zip(lat_arr[valid].tolist(), lon_arr[valid].tolist()))
                map_widget.set_path(full_path, color="#555555", width=2)
            map_widget.set_position(float(np.nanmean(lat_arr[valid])), float(np.nanmean(lon_arr[valid])))
            map_widget.set_zoom(16)

            def _on_tile_change(_e=None):
                name = tile_var.get()
                for lbl, url, mz in _TS:
                    if lbl == name:
                        map_widget.set_tile_server(url, max_zoom=mz)
                        break
            tile_combo.bind("<<ComboboxSelected>>", _on_tile_change)
        else:
            tk.Label(map_frame, text="No GPS data", bg=BG, fg=FD, font=("Segoe UI", 11)).pack(expand=True)

        # -- SECTION PANELS --
        fig_frame = tk.Frame(main_pw, bg=BG)
        main_pw.add(fig_frame, minsize=500)

        fig = plt.Figure(figsize=(14, 9), dpi=100, facecolor=BG)
        gs = GridSpec(2, 2, figure=fig,
                      width_ratios=[7, 1],
                      height_ratios=[1, 3],
                      hspace=0.10, wspace=0.08)

        ax_xy = fig.add_subplot(gs[0, 0])   # XY slice (Distance x Channel)
        ax_ph = fig.add_subplot(gs[0, 1])   # placeholder (top-right)
        ax_xz = fig.add_subplot(gs[1, 0])   # XZ slice (Distance x Depth)
        ax_yz = fig.add_subplot(gs[1, 1])   # YZ slice (Channel x Depth)

        ax_ph.set_facecolor(BG)
        ax_ph.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === CONTROL BAR ===
        ctrl = tk.Frame(win, bg=BP, padx=8, pady=4)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        zi_var = tk.IntVar(value=nz // 4)
        xi_var = tk.IntVar(value=0)
        yi_var = tk.IntVar(value=ny // 2)

        _pending = [None]

        def _schedule_update(v=None):
            if _pending[0] is not None:
                win.after_cancel(_pending[0])
            _pending[0] = win.after(30, _full_redraw)

        cc = 0
        for lbl_txt, var, mx in [("Depth(Z):", zi_var, nz - 1),
                                  ("Distance(X):", xi_var, nx - 1),
                                  ("Channel(Y):", yi_var, ny - 1)]:
            tk.Label(ctrl, text=lbl_txt, **_lbl_kw).grid(row=0, column=cc, padx=(0, 2)); cc += 1
            tk.Scale(ctrl, variable=var, from_=0, to=max(mx, 0),
                     command=_schedule_update, **_scale_kw).grid(row=0, column=cc, padx=2); cc += 1

        _seg_x_ref = [x_m[:1]]

        # === STYLING ===
        def _style_sec_ax(ax):
            ax.set_facecolor("#1e1e1e")
            for sp in ax.spines.values():
                sp.set_color("#333333")
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

        def _add_label(ax, text):
            ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                              edgecolor="none", alpha=0.85))

        def _mappable():
            cn = cmap_var.get() or "gray"
            try:
                cmp = plt.get_cmap(cn)
            except Exception:
                cmp = plt.get_cmap("gray")
            return cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmp)

        GAP_FRAC = 0.03

        def _draw_gapped_cross(ax, h_val, v_val, xlim, ylim):
            """Draw crosshairs with gap at intersection. Black outline + yellow fill for contrast."""
            x0, x1 = min(xlim), max(xlim)
            y0, y1 = min(ylim), max(ylim)
            x_gap = (x1 - x0) * GAP_FRAC
            y_gap = (y1 - y0) * GAP_FRAC
            segs = [
                ([x0, v_val - x_gap], [h_val, h_val]),      # h-left
                ([v_val + x_gap, x1], [h_val, h_val]),       # h-right
                ([v_val, v_val], [y0, h_val - y_gap]),       # v-top
                ([v_val, v_val], [h_val + y_gap, y1]),       # v-bottom
            ]
            for sx, sy in segs:
                ax.plot(sx, sy, color="black", alpha=0.9, linewidth=2.5, zorder=19)
                ax.plot(sx, sy, color="#ffff00", alpha=0.95, linewidth=1.2, zorder=20)

        # === MAIN REDRAW ===
        def _full_redraw():
            _pending[0] = None
            si = seg_var.get()
            i0, i1 = _seg_indices(si)
            seg_nx = i1 - i0
            seg_label.config(text=f"Segment {si + 1} / {n_segs}  "
                             f"({x_m[i0]:.0f}\u2013{x_m[min(i1 - 1, nx - 1)]:.0f} m)")

            zi = max(0, min(nz - 1, zi_var.get()))
            yi = max(0, min(ny - 1, yi_var.get()))
            xi_global = xi_var.get()
            xi_local = max(0, min(seg_nx - 1, xi_global - i0))

            seg_data = vol[:, i0:i1, :]
            seg_x = x_m[i0:i1]
            _seg_x_ref[0] = seg_x

            mp = _mappable()
            sx0, sx1 = float(seg_x[0]), float(seg_x[-1])
            y0, y1 = float(y_m[0]), float(y_m[-1])
            z0, z1 = float(z_m[0]), float(z_m[-1])

            x_cross = float(seg_x[xi_local])
            y_cross = float(y_m[yi])
            z_cross = float(z_m[zi])

            # GPS map highlight
            if map_widget is not None and lat_arr is not None:
                if _map_seg_path[0] is not None:
                    try:
                        _map_seg_path[0].delete()
                    except Exception:
                        pass
                seg_lat, seg_lon = lat_arr[i0:i1], lon_arr[i0:i1]
                v = np.isfinite(seg_lat) & np.isfinite(seg_lon) & (np.abs(seg_lat) > 1)
                if np.sum(v) > 1:
                    sp = list(zip(seg_lat[v].tolist(), seg_lon[v].tolist()))
                    _map_seg_path[0] = map_widget.set_path(sp, color="#e74c3c", width=4)
                    map_widget.set_position(float(np.mean(seg_lat[v])), float(np.mean(seg_lon[v])))

            # XY Slice
            ax_xy.clear()
            _style_sec_ax(ax_xy)
            ax_xy.imshow(seg_data[zi, :, :].T, aspect="auto", cmap=mp.cmap, norm=mp.norm,
                         extent=[sx0, sx1, y1, y0], origin="upper")
            ax_xy.set_xlabel("Distance (m)", fontsize=8)
            ax_xy.set_ylabel("Channel (m)", fontsize=8)
            _add_label(ax_xy, "XY Slice")
            _draw_gapped_cross(ax_xy, y_cross, x_cross, (sx0, sx1), (y0, y1))

            # XZ Slice
            ax_xz.clear()
            _style_sec_ax(ax_xz)
            ax_xz.imshow(seg_data[:, :, yi], aspect="auto", cmap=mp.cmap, norm=mp.norm,
                         extent=[sx0, sx1, z1, z0], origin="upper")
            ax_xz.set_xlabel("Distance (m)", fontsize=8)
            ax_xz.set_ylabel("Depth (m)", fontsize=8)
            _add_label(ax_xz, "XZ Slice")
            _draw_gapped_cross(ax_xz, z_cross, x_cross, (sx0, sx1), (z0, z1))

            # YZ Slice
            ax_yz.clear()
            _style_sec_ax(ax_yz)
            ax_yz.imshow(seg_data[:, xi_local, :], aspect="auto", cmap=mp.cmap, norm=mp.norm,
                         extent=[y0, y1, z1, z0], origin="upper")
            ax_yz.set_xlabel("Channel (m)", fontsize=8)
            ax_yz.set_ylabel("Depth (m)", fontsize=8)
            _add_label(ax_yz, "YZ Slice")
            _draw_gapped_cross(ax_yz, z_cross, y_cross, (y0, y1), (z0, z1))

            # Info panel
            ax_ph.clear()
            ax_ph.set_facecolor(BG)
            ax_ph.axis("off")
            ax_ph.text(0.5, 0.5,
                       f"Z={z_cross:.2f}m\nX={x_cross:.1f}m\nY={y_cross:.2f}m",
                       transform=ax_ph.transAxes, ha="center", va="center",
                       fontsize=9, color=FT, fontfamily="monospace",
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="#2b2d3e",
                                 edgecolor="#444", alpha=0.9))

            fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.07)
            canvas.draw_idle()

        # === CLICK-TO-SLICE ===
        def _on_click(event):
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            si = seg_var.get()
            i0, i1 = _seg_indices(si)
            seg_x = _seg_x_ref[0]

            if event.inaxes is ax_xy:
                xi_local = int(np.clip(np.searchsorted(seg_x, event.xdata), 0, len(seg_x) - 1))
                xi_var.set(i0 + xi_local)
                yi_var.set(int(np.clip(np.searchsorted(y_m, event.ydata), 0, ny - 1)))
            elif event.inaxes is ax_xz:
                xi_local = int(np.clip(np.searchsorted(seg_x, event.xdata), 0, len(seg_x) - 1))
                xi_var.set(i0 + xi_local)
                zi_var.set(int(np.clip(np.searchsorted(z_m, event.ydata), 0, nz - 1)))
            elif event.inaxes is ax_yz:
                yi_var.set(int(np.clip(np.searchsorted(y_m, event.xdata), 0, ny - 1)))
                zi_var.set(int(np.clip(np.searchsorted(z_m, event.ydata), 0, nz - 1)))
            else:
                return
            _schedule_update()  # sync all panels to the new position

        _dragging = [False]

        def _on_press(event):
            if event.inaxes in (ax_xy, ax_xz, ax_yz):
                _dragging[0] = True
                _on_click(event)

        def _on_release(event):
            _dragging[0] = False

        def _on_motion(event):
            if _dragging[0] and event.inaxes is not None:
                _on_click(event)

        fig.canvas.mpl_connect("button_press_event", _on_press)
        fig.canvas.mpl_connect("button_release_event", _on_release)
        fig.canvas.mpl_connect("motion_notify_event", _on_motion)

        # Cleanup
        def _on_close():
            plt.close(fig)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        _full_redraw()

    def _load_file_at_segment_index(self, index: int):
        """Load the file at given 0-based index from folder_segment_files."""
        if not self.folder_segment_files or index < 0 or index >= len(self.folder_segment_files):
            return
        entry = self.folder_segment_files[index]
        base, name, data_type = entry[0], entry[1], entry[2]
        # Use the original extension from disk (entry[3]) to reconstruct path correctly
        orig_ext = entry[3] if len(entry) > 3 else (".V00" if data_type == "v00" else ".DT")
        path = os.path.join(base, name + orig_ext)
        self.file_var.set(path)
        self.loader.load_data(base, name, data_type)
        self.original_data = self.loader.data.copy()
        self._hilbert_display_mode = None
        self.view_x_start_m = 0.0
        self._total_length_m = None
        # For the very first segment in folder mode, compute X offset so that it starts at 0 m.
        # We only set this once (when loading the first segment) so all segments share the same base.
        if index == 0:
            if self.loader.xyz is not None and len(self.loader.xyz) > 0:
                self._folder_x0_offset = float(self.loader.xyz["X"].iloc[0])
            else:
                self._folder_x0_offset = 0.0
        self.show_metadata()
        self._sync_x_ch_vars()  # Update X position entry with absolute distance
        self.plot_gpr()

    def _sync_segment_display(self, total: int):
        """Update segment entry and label when in folder mode (total = number of files)."""
        if total <= 0:
            self.segment_var.set("1")
            return
        self.segment_var.set(str(self.current_segment_index + 1))

    def _get_window_m(self) -> float:
        """Current X window length (m) from bottom bar; default 40."""
        try:
            v = float(self.distance_var.get().strip())
            if v > 0:
                return v
        except (ValueError, TypeError):
            pass
        return 40.0

    def _apply_x_from_entry(self):
        """Apply X (m) start from entry and redraw. User enters absolute distance."""
        try:
            user_abs_x = float(self.x_pos_var.get().strip())
            # Convert absolute X to relative offset
            x0 = self._get_x0()
            # view_x_start_m is relative offset from x0
            self.view_x_start_m = max(0.0, user_abs_x - x0)
            self._sync_x_ch_vars()
            self.plot_gpr()
        except (ValueError, TypeError):
            pass

    def _apply_distance_and_redraw(self):
        """Apply distance (window length) and redraw."""
        self._sync_x_ch_vars()
        self.plot_gpr()

    def _view_prev_segment(self):
        """Previous segment = previous file in folder."""
        if not self.folder_segment_files:
            return
        if self.current_segment_index <= 0:
            return
        self.current_segment_index -= 1
        self._load_file_at_segment_index(self.current_segment_index)
        self._sync_segment_display(len(self.folder_segment_files))
        self._sync_x_ch_vars()

    def _view_next_segment(self):
        """Next segment = next file in folder."""
        if not self.folder_segment_files:
            return
        if self.current_segment_index >= len(self.folder_segment_files) - 1:
            return
        self.current_segment_index += 1
        self._load_file_at_segment_index(self.current_segment_index)
        self._sync_segment_display(len(self.folder_segment_files))
        self._sync_x_ch_vars()

    def _apply_segment_from_entry(self):
        """Go to segment number typed in entry (1-based)."""
        if not self.folder_segment_files:
            return
        try:
            seg = int(self.segment_var.get().strip())
            n = len(self.folder_segment_files)
            seg = max(1, min(n, seg))
            self.current_segment_index = seg - 1
            self._load_file_at_segment_index(self.current_segment_index)
            self._sync_segment_display(n)
            self._sync_x_ch_vars()
        except (ValueError, TypeError):
            pass

    def load_selected_file(self):
        path = self.file_var.get()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid file first.")
            return

        try:
            base = os.path.dirname(path)
            name, ext = os.path.splitext(os.path.basename(path))
            ext = ext.lower()

            if ext == ".v00":
                data_type = "v00"
            elif ext == ".dt":
                data_type = "dt"
            else:
                messagebox.showerror(
                    "Unsupported",
                    "Only .V00 and .DT files are supported."
                )
                return
            # Single-file load: clear folder mode only if user loaded via Browse V00/DT (not folder)
            if self.current_segment_index < 0:
                self.folder_segment_files = []
                self._folder_x0_offset = 0.0
                self.segment_var.set("1")

            self.loader.load_data(base, name, data_type)
            self._hilbert_display_mode = None
            # Save original (unfiltered) data safely
            self.original_data = self.loader.data.copy()
            self.view_x_start_m = 0.0
            self._total_length_m = None
            self._sync_x_ch_vars()
            self.show_metadata()
            self.plot_gpr()
                         

            # self.loader.load_data(base, name, data_type)

            # self.show_metadata()
            # self.plot_gpr()

        except Exception as e:
            messagebox.showerror("Load Failed", str(e))

    def fir_filter(self, data, dt, f1=None, f2=None, numtaps=101, ftype="bandpass"):
        """
        FIR filtering using windowed-sinc method.
        dt: sample interval (seconds)
        f1, f2: cutoff frequencies (Hz)
        """
        fs = 1.0 / dt
        nyq = fs / 2.0

        if ftype == "lowpass":
            fc = f2 / nyq
            h = sinc(2 * fc * (np.arange(numtaps) - (numtaps - 1) / 2))
        elif ftype == "highpass":
            fc = f1 / nyq
            h = sinc(np.arange(numtaps) - (numtaps - 1) / 2) - \
                sinc(2 * fc * (np.arange(numtaps) - (numtaps - 1) / 2))
        else:  # bandpass
            fc1 = f1 / nyq
            fc2 = f2 / nyq
            h = (
                sinc(2 * fc2 * (np.arange(numtaps) - (numtaps - 1) / 2)) -
                sinc(2 * fc1 * (np.arange(numtaps) - (numtaps - 1) / 2))
            )

        window = np.hamming(numtaps)
        h *= window
        h /= np.sum(h)

        # Filter trace by trace
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = np.convolve(data[:, i], h, mode="same")

        return filtered
        
    def apply_fir_low(self):
        if self.loader.data is None or self.loader._sample_interval_s is None:
            messagebox.showerror("Error", "Load data with valid HDR first.")
            return

        self.loader.data = self.fir_filter(
            self.loader.data,
            dt=self.loader._sample_interval_s,
            f2=200e6,     # example cutoff (200 MHz)
            numtaps=121,
            ftype="lowpass"
        )
        self.loader.add_process("FIR-low")
        self.plot_gpr()


    def apply_fir_band(self):
        if self.loader.data is None or self.loader._sample_interval_s is None:
            messagebox.showerror("Error", "Load data with valid HDR first.")
            return

        self.loader.data = self.fir_filter(
            self.loader.data,
            dt=self.loader._sample_interval_s,
            f1=50e6,      # example band
            f2=250e6,
            numtaps=151,
            ftype="bandpass"
        )
        self.loader.add_process("FIR-BP")
        self.plot_gpr()

    def fir_bandpass_dialog(self):
        if self.loader.data is None or self.loader._sample_interval_s is None:
            messagebox.showerror("Error", "Load data with valid HDR first.")
            return

        win = tk.Toplevel(self.root)
        win.title("FIR Bandpass Filter")
        win.geometry("330x250")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()   # modal dialog

        tk.Label(win, text="Low cut frequency (MHz)").pack(pady=(10, 0))
        f1_mhz = tk.DoubleVar(value=50.0)
        tk.Entry(win, textvariable=f1_mhz, width=15).pack()

        tk.Label(win, text="High cut frequency (MHz)").pack(pady=(10, 0))
        f2_mhz = tk.DoubleVar(value=250.0)
        tk.Entry(win, textvariable=f2_mhz, width=15).pack()

        tk.Label(win, text="Number of taps").pack(pady=(10, 0))
        ntap_var = tk.IntVar(value=151)
        tk.Entry(win, textvariable=ntap_var, width=15).pack()

        # ---------------- Apply logic ----------------
        def apply_and_close():
            try:
                f1 = f1_mhz.get() * 1e6   # MHz → Hz
                f2 = f2_mhz.get() * 1e6
                ntaps = ntap_var.get()

                if f1 <= 0 or f2 <= 0 or f2 <= f1:
                    raise ValueError("Require: 0 < f_low < f_high (MHz)")

                self.loader.data = self.fir_filter(
                    self.loader.data,
                    dt=self.loader._sample_interval_s,
                    f1=f1,
                    f2=f2,
                    numtaps=ntaps,
                    ftype="bandpass"
                )

                win.destroy()
                self.plot_gpr()

            except Exception as e:
                messagebox.showerror("Invalid input", str(e))

        # ---------------- Buttons ----------------
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame, text="OK",
            width=10, bg="#4CAF50", fg="white",
            command=apply_and_close
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            btn_frame, text="Cancel",
            width=10,
            command=win.destroy
        ).grid(row=0, column=1, padx=5)


    def update_ascan(self, trace, y, idx):
        self.ascan_ax.cla()

        self.ascan_ax.plot(trace, y, color="black", linewidth=1.0)
        self.ascan_ax.axvline(0, color="gray", linewidth=0.5)

        self.ascan_ax.set_ylim(y[-1], y[0])  # depth down
        self.ascan_ax.set_xlabel("Amplitude")
        self.ascan_ax.set_ylabel("Depth (m)")
        self.ascan_ax.set_title(f"A-scan | Trace {idx}")

        self.ascan_ax.grid(True, alpha=0.3)

        self.ascan_fig.tight_layout()
        self.ascan_canvas.draw_idle()

        if not self.ascan_win.winfo_viewable():
            self.ascan_win.deiconify()
    def toggle_ascan(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        if not self.ascan_enabled:
            # Enable A-scan
            self.ascan_cid = self.canvas.mpl_connect(
                "motion_notify_event", self.on_mouse_move
            )
            self.ascan_enabled = True
            self.ascan_win.deiconify()
        else:
            # Disable A-scan
            if self.ascan_cid is not None:
                self.canvas.mpl_disconnect(self.ascan_cid)
                self.ascan_cid = None

            self.ascan_enabled = False
            self.ascan_win.withdraw()
    def on_mouse_move(self, event):
        """
        Mouse move handler over main GPR figure.
        - Optionally updates a cross-hair showing distance/depth (when enabled).
        - When A-scan is enabled, also updates the A-scan window.
        """
        if (
            self.loader.data is None
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            return

        # Only react to the main GPR figure, not other matplotlib figures
        if event.inaxes.figure is not self.figure:
            return

        ax = event.inaxes

        x_val = float(event.xdata)
        y_val = float(event.ydata)

        # --- Cross-hair update (only when enabled) ---
        if self.crosshair_enabled.get():
            if self.crosshair_hline is None:
                # First-time creation of cross-hair artists
                self.crosshair_hline = ax.axhline(
                    y=y_val, color="yellow", linewidth=0.6, alpha=0.8
                )
                self.crosshair_vline = ax.axvline(
                    x=x_val, color="yellow", linewidth=0.6, alpha=0.8
                )
                self.crosshair_text = ax.text(
                    0.02,
                    0.98,
                    "",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    color="yellow",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
                )
            else:
                self.crosshair_hline.set_ydata(y_val)
                self.crosshair_vline.set_xdata(x_val)

            # Label with profile-distance and depth
            if self.crosshair_text is not None:
                self.crosshair_text.set_text(
                    f"X: {x_val:.1f} m\nDepth: {y_val:.2f} m"
                )

            self.canvas.draw_idle()

        # --- Optional A-scan update (only when enabled) ---
        if not self.ascan_enabled:
            return

        data = self.loader.data
        n_traces = data.shape[1]

        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            xvals = self.loader.xyz["X"].values
        else:
            xvals = np.arange(n_traces)

        idx = np.argmin(np.abs(xvals - x_val))
        idx = np.clip(idx, 0, n_traces - 1)

        trace = data[:, idx]

        y = (
            self.loader.depth
            if self.loader.depth is not None
            else np.arange(len(trace))
        )

        self.update_ascan(trace, y, idx)
        self.last_trace_index = idx

    def toggle_crosshair(self):
        """Toggle cross-hair visibility on the main GPR figure."""
        enabled = not self.crosshair_enabled.get()
        self.crosshair_enabled.set(enabled)

        # Update button style to reflect state
        if enabled:
            self.crosshair_btn.configure(bg="#FFC107", fg="black")
        else:
            self.crosshair_btn.configure(bg="#EEEEEE", fg="black")

        # When turning off, remove existing cross-hair artists
        if not enabled:
            ax = None
            if self.crosshair_hline is not None:
                ax = self.crosshair_hline.axes
                self.crosshair_hline.remove()
            if self.crosshair_vline is not None:
                ax = self.crosshair_vline.axes
                self.crosshair_vline.remove()
            if self.crosshair_text is not None:
                ax = self.crosshair_text.axes
                self.crosshair_text.remove()

            self.crosshair_hline = None
            self.crosshair_vline = None
            self.crosshair_text = None

            if ax is not None:
                self.canvas.draw_idle()

    def hilbert_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Hilbert Transform")
        win.resizable(False, False)

        tk.Label(win, text="Hilbert Output:", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=10, sticky="e")

        mode_var = tk.StringVar(value="envelope")
        modes = [("Envelope", "envelope"), ("Phase", "phase"), ("Frequency", "frequency")]

        for i, (txt, val) in enumerate(modes):
            tk.Radiobutton(win, text=txt, variable=mode_var, value=val).grid(row=i, column=1, padx=5, pady=5, sticky="w")

        def apply_hilbert():
            self.apply_hilbert(mode_var.get())
            win.destroy()

        tk.Button(win, text="OK", command=apply_hilbert, bg="#4CAF50", fg="white", width=10).grid(row=4, column=0, columnspan=2, pady=12)

    def apply_envelope(self):
        """Instantaneous amplitude (envelope), normalized to [0, 1]. Norm. Envelope E [-]."""
        self.apply_hilbert("envelope")

    def apply_instantaneous_phase(self):
        """Instantaneous phase in degrees. Highlights reflector continuity."""
        self.apply_hilbert("phase")

    def apply_instantaneous_frequency(self):
        """Instantaneous frequency in MHz. Sensitive to attenuation/dispersion."""
        self.apply_hilbert("frequency")

    def apply_hilbert(self, mode="envelope"):
        """
        Compute Hilbert-based attribute and replace displayed data.
        - envelope: E(t) = |z(t)|, normalized to [0, 1] (Norm. Envelope E [-])
        - phase: φ(t) in degrees (Instantaneous phase φ [deg])
        - frequency: f(t) = (1/2π) dφ/dt in MHz (Instantaneous frequency f [MHz])
        """
        data = self.loader.data
        if data is None:
            messagebox.showinfo("Info", "Load data first.")
            return
        analytic = hilbert(data.astype(np.float64), axis=0)
        phase_unwrap = np.unwrap(np.angle(analytic), axis=0)

        if mode == "envelope":
            env = np.abs(analytic)
            peak = np.percentile(env, 99)
            if peak <= 0:
                peak = 1.0
            self.loader.data = (env / peak).astype(np.float64)
            self._hilbert_display_mode = "Envelope (Norm. E [-])"
        elif mode == "phase":
            # Instantaneous phase in degrees [0, 360] or unwrapped for display
            phase_deg = np.degrees(phase_unwrap)
            self.loader.data = phase_deg.astype(np.float64)
            self._hilbert_display_mode = "Instantaneous phase φ [deg]"
        elif mode == "frequency":
            dt_s = getattr(self.loader, "_sample_interval_s", None)
            if dt_s is None or dt_s <= 0:
                dt_s = 1.0
            # Central difference to keep shape: f_Hz = (1/2π) * dφ/dt
            dphase = np.zeros_like(phase_unwrap)
            dphase[1:-1] = (phase_unwrap[2:] - phase_unwrap[:-2]) / (2.0 * dt_s)
            dphase[0] = (phase_unwrap[1] - phase_unwrap[0]) / dt_s
            dphase[-1] = (phase_unwrap[-1] - phase_unwrap[-2]) / dt_s
            f_Hz = dphase / (2.0 * np.pi)
            f_MHz = f_Hz / 1e6
            # Clip unrealistic values for display (e.g. 50–300 MHz typical for GPR)
            f_MHz = np.clip(f_MHz, 0, 500)
            self.loader.data = f_MHz.astype(np.float64)
            self._hilbert_display_mode = "Instantaneous frequency f [MHz]"
        else:
            self._hilbert_display_mode = None
            return
        if not hasattr(self, "_hilbert_display_mode"):
            self._hilbert_display_mode = None
        self.plot_gpr()

    def fft_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Fourier Transform (FFT)")
        win.resizable(False, False)

        tk.Label(win, text="FFT Mode:", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=8, sticky="e")

        mode_var = tk.StringVar(value="ascan")
        tk.Radiobutton(win, text="Current A-scan (cursor)", variable=mode_var, value="ascan").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(win, text="Trace Average", variable=mode_var, value="average").grid(row=1, column=1, sticky="w")

        tk.Label(win, text="Spectrum:", font=("Helvetica", 11)).grid(row=2, column=0, padx=10, pady=8, sticky="e")

        spec_var = tk.StringVar(value="amplitude")
        tk.Radiobutton(win, text="Amplitude", variable=spec_var, value="amplitude").grid(row=2, column=1, sticky="w")
        tk.Radiobutton(win, text="Power", variable=spec_var, value="power").grid(row=3, column=1, sticky="w")

        def apply_fft():
            self.compute_fft(mode_var.get(), spec_var.get())
            win.destroy()

        tk.Button(win, text="OK", command=apply_fft, bg="#4CAF50", fg="white", width=10).grid(row=4, column=0, columnspan=2, pady=12)
    def compute_fft(self, mode="ascan", spectrum="amplitude"):
        data = self.loader.data

        if mode == "ascan":
            if not hasattr(self, "last_trace_index"):
                messagebox.showinfo("Info", "Move mouse over section to select A-scan.")
                return
            trace = data[:, self.last_trace_index]
        else:
            trace = np.mean(data, axis=1)

        n = len(trace)
        dt = self.loader.dt if hasattr(self.loader, "dt") and self.loader.dt is not None else 1.0

        freq = np.fft.rfftfreq(n, d=dt)
        # freq = np.fft.rfftfreq(n, d=dt) * 1e-6

        fftv = np.fft.rfft(trace)

        if spectrum == "power":
            spec = np.abs(fftv) ** 2
            ylabel = "Power"
        else:
            spec = np.abs(fftv)
            ylabel = "Amplitude"

        self.plot_fft(freq, spec, ylabel)
        
    def plot_fft(self, freq, spec, ylabel):
        win = tk.Toplevel(self.root)
        win.title("FFT Spectrum")

        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(freq, spec, color="black", linewidth=1.2)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title("Frequency Spectrum")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw_idle()
    def hht_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Hilbert-Huang Transform (HHT)")
        win.resizable(False, False)

        tk.Label(win, text="Trace index:", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=8, sticky="e")
        trace_var = tk.IntVar(value=0)
        tk.Entry(win, textvariable=trace_var, width=10).grid(row=0, column=1, pady=8)

        tk.Label(win, text="IMF number:", font=("Helvetica", 11)).grid(row=1, column=0, padx=10, pady=8, sticky="e")
        imf_var = tk.IntVar(value=1)
        tk.Entry(win, textvariable=imf_var, width=10).grid(row=1, column=1, pady=8)

        def apply_hht():
            self.apply_hht(trace_var.get(), imf_var.get())
            win.destroy()

        tk.Button(
            win,
            text="OK",
            command=apply_hht,
            bg="#4CAF50",
            fg="white",
            width=10
        ).grid(row=2, column=0, columnspan=2, pady=12)
      
    def apply_hht(self, trace_idx=0, imf_idx=1):
        data = self.loader.data

        if trace_idx < 0 or trace_idx >= data.shape[1]:
            messagebox.showerror("Error", "Invalid trace index.")
            return

        trace = data[:, trace_idx]

        emd = EMD(trace)
        imfs = emd.decompose()

        if imf_idx < 1 or imf_idx > imfs.shape[0]:
            messagebox.showerror("Error", "Invalid IMF number.")
            return

        imf = imfs[imf_idx - 1]

        analytic = hilbert(imf)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))

        # Time axis
        y = self.loader.depth if self.loader.depth is not None else np.arange(len(imf))

        self.figure.clf()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        ax1.plot(amplitude, y, color="black")
        ax1.set_title(f"HHT Envelope | Trace {trace_idx} | IMF {imf_idx}")
        ax1.set_ylabel("Depth (m)")
        # ax1.invert_yaxis()

        ax2.plot(phase, y, color="red")
        ax2.set_title("Instantaneous Phase")
        ax2.set_xlabel("Amplitude / Phase")
        ax2.set_ylabel("Depth (m)")
        # ax2.invert_yaxis()

        self.figure.tight_layout()
        self.canvas.draw_idle()
    def hht_tf_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("HHT Time–Frequency (H(t,f))")
        win.resizable(False, False)

        tk.Label(win, text="Trace index:", font=("Helvetica", 11)).grid(
            row=0, column=0, padx=10, pady=8, sticky="e"
        )
        trace_var = tk.IntVar(value=0)
        tk.Entry(win, textvariable=trace_var, width=10).grid(row=0, column=1, pady=8)

        def apply():
            self.plot_hht_tf(trace_var.get())
            win.destroy()

        tk.Button(
            win,
            text="OK",
            command=apply,
            bg="#4CAF50",
            fg="white",
            width=10
        ).grid(row=1, column=0, columnspan=2, pady=12)
    def plot_hht_tf(self, trace_idx=0):
        data = self.loader.data

        if trace_idx < 0 or trace_idx >= data.shape[1]:
            messagebox.showerror("Error", "Invalid trace index.")
            return

        trace = data[:, trace_idx]

        # --- EMD decomposition ---
        emd = EMD(trace)
        imfs = emd.decompose()

        # --- Sampling interval ---
        dt = self.loader._sample_interval_s
        if dt is None:
            dt = 1.0  # fallback (relative units)

        n = len(trace)

        # Depth or time axis
        if self.loader.depth is not None:
            y = self.loader.depth
            ylabel = "Depth (m)"
        else:
            y = np.arange(n) * dt
            ylabel = "Time (s)"

        # --- Build HHT spectrum ---
        freq_bins = np.linspace(0, 1.0 / (2 * dt), 300)   # Hz
        freq_bins_mhz = freq_bins * 1e-6                  # MHz

        H = np.zeros((n, len(freq_bins)))

        for imf in imfs:
            analytic = hilbert(imf)
            amp = np.abs(analytic)

            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) / (2 * np.pi * dt)
            inst_freq = np.concatenate([[inst_freq[0]], inst_freq])  # align length

            for i in range(n):
                f = inst_freq[i]
                if 0 < f < freq_bins[-1]:
                    k = np.searchsorted(freq_bins, f)
                    H[i, k] += amp[i]

        # --- Plot ---
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        pcm = ax.pcolormesh(
            freq_bins_mhz,
            y,
            H,
            shading="auto",
            cmap="jet"
        )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"HHT Time–Frequency | Trace {trace_idx}")
        # ax.invert_yaxis()

        self.figure.colorbar(pcm, ax=ax, label="Amplitude")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        
    def plot_survey_map(self):
        if (
            self.loader is None or
            not hasattr(self.loader, "lat") or
            not hasattr(self.loader, "lon") or
            self.loader.lat is None or
            self.loader.lon is None
        ):
            messagebox.showerror(
                "Error",
                "No geographic coordinates found.\n"
                "Load GEO / GEOX data first."
            )
            return

        lat = self.loader.lat
        lon = self.loader.lon

        if len(lat) < 2:
            messagebox.showerror("Error", "Not enough points to plot survey line.")
            return

        tracks = [(lat.tolist(), lon.tolist(), "#f1c40f", self.loader.line_name or "Survey")]
        _open_map_popup("GPR Survey Map", tracks, zoom=18, root=self.root)
        
    def projected_to_latlon(self, x, y, epsg):
        """
        Convert projected coordinates to latitude longitude.
        epsg: int (e.g., 32645, 5179, 4326)
        """
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}",
            "EPSG:4326",
            always_xy=True
        )

        lon, lat = transformer.transform(x, y)
        return np.asarray(lat), np.asarray(lon)
    def load_geo(self, filename):
        data = np.loadtxt(filename)

        self.lat = data[:, 1]
        self.lon = data[:, 2]

        self.coord_type = "geographic"
    def load_geox(self, filename, epsg):
        data = np.loadtxt(filename)

        x = data[:, 1]
        y = data[:, 2]

        self.lat, self.lon = self.projected_to_latlon(x, y, epsg)
        self.coord_type = "projected"


    def ask_epsg(self):
        win = tk.Toplevel(self.root)
        win.title("Select EPSG Code")
        win.resizable(False, False)

        tk.Label(win, text="EPSG Code:").grid(row=0, column=0, padx=10, pady=5)

        epsg_var = tk.StringVar(value="32645")

        tk.Entry(win, textvariable=epsg_var, width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        result = {"epsg": None}

        def ok():
            result["epsg"] = int(epsg_var.get())
            win.destroy()

        tk.Button(
            win,
            text="OK",
            command=ok,
            bg="#4CAF50",
            fg="white"
        ).grid(row=1, column=0, columnspan=2, pady=10)

        win.grab_set()
        win.wait_window()

        return result["epsg"]


    def load_geo_file(self):
        path = filedialog.askopenfilename(
            title="Select GEOX or GEC file",
            filetypes=[
                ("GEOX files", "*.geox *.GEOX"),
                ("GEC files", "*.gec *.GEC"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".gec":
                epsg = self.ask_epsg()
                if epsg is None:
                    return
                self.loader.load_geox(path, epsg)

            elif ext == ".geox":
                messagebox.showwarning(
                    "GEOX ignored",
                    "GEOX coordinates are often local.\n"
                    "Use GEC (GPS) for correct map location."
                )
                epsg = self.ask_epsg()
                if epsg is None:
                    return
                self.loader.load_geox(path, epsg)


            else:
                messagebox.showerror("Error", "Unsupported geo file format.")
                return

            messagebox.showinfo("Success", "Geographic data loaded.")
            self.plot_survey_map()

        except Exception as e:
            messagebox.showerror("Geo load failed", str(e))
    def plot_gps_nmea(self):
        path = filedialog.askopenfilename(
            title="Select GPS NMEA file",
            filetypes=[
                ("GPS files", "*.gps *.GPS *.txt"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.loader.load_gps_nmea(path)
            self._plot_gps_contextily()

        except Exception as e:
            messagebox.showerror("GPS error", str(e))
    def _plot_gps_contextily(self):
        lat = self.loader.lat
        lon = self.loader.lon

        if lat is None or lon is None or len(lat) == 0:
            raise ValueError("No GPS data available to plot.")

        # ----------------------------------
        # Create GeoDataFrame (WGS84)
        # ----------------------------------
        geometry = [Point(xy) for xy in zip(lon, lat)]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

        # Convert to Web Mercator for basemap
        gdf_3857 = gdf.to_crs(epsg=3857)

        # ----------------------------------
        # Plot
        # ----------------------------------
        fig, ax = plt.subplots(figsize=(14, 9))

        gdf_3857.plot(
            ax=ax,
            linewidth=1.0,
            alpha=0.9,
            color="blue",
            label="Survey Path"
        )

        # Start / End points
        gdf_3857.iloc[[0]].plot(ax=ax, color="green", markersize=30, label="Start")
        gdf_3857.iloc[[-1]].plot(ax=ax, color="red", markersize=30, label="End")

        # ----------------------------------
        # Compute extent with margin
        # ----------------------------------
        minx, miny, maxx, maxy = gdf_3857.total_bounds
        dx = maxx - minx
        dy = maxy - miny
        margin = 0.3

        ax.set_xlim(minx - dx * margin, maxx + dx * margin)
        ax.set_ylim(miny - dy * margin, maxy + dy * margin)

        # ----------------------------------
        # Basemap (ESRI)
        # ----------------------------------
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            attribution=False
        )

        ax.set_aspect("equal")
        ax.set_title("GPR Survey Path (GPS)")
        ax.legend()
        ax.axis("off")

        # ----------------------------------
        # Enable scroll zoom
        # ----------------------------------
        self._enable_scroll_zoom(fig, ax)

        plt.show()
    def _enable_scroll_zoom(self, fig, ax):
        base_scale = 1.2

        def zoom(event):
            if event.inaxes != ax:
                return

            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if xdata is None or ydata is None:
                return

            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([
                xdata - new_width * (1 - relx),
                xdata + new_width * relx
            ])
            ax.set_ylim([
                ydata - new_height * (1 - rely),
                ydata + new_height * rely
            ])

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", zoom)

    def plot_geox_path(self):
        if self.loader.xyz is None or len(self.loader.xyz) == 0:
            messagebox.showerror("Error", "No GEOX data loaded.\nLoad a GEOX/GEC file first.")
            return

        geo_x = self.loader.xyz["X"].values
        geo_y = self.loader.xyz["Y"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(geo_x, geo_y, "-o", color="blue", markersize=3, linewidth=1, label="GEOX Path")
        ax.scatter(geo_x[0], geo_y[0], color="green", s=50, zorder=5, label="Start")
        ax.scatter(geo_x[-1], geo_y[-1], color="red", s=50, zorder=5, label="End")

        ax.set_aspect("equal")
        ax.set_xlabel("X (meters, relative)")
        ax.set_ylabel("Y (meters, relative)")
        ax.set_title("GEOX Survey Path (relative coordinates)")
        ax.legend()
        plt.tight_layout()
        plt.show()
    def plot_geox_on_gps(self):
        if self.loader.xyz is None or len(self.loader.xyz) == 0:
            messagebox.showerror("Error", "No GEOX data loaded.\nLoad a GEOX/GEC file first.")
            return
        if not hasattr(self.loader, "lat") or self.loader.lat is None or len(self.loader.lat) == 0:
            messagebox.showerror("Error", "No GPS data loaded.\nLoad a GPS file first.")
            return
        try:
            df = self.loader.xyz
            # Try to use valid Lat/Lon from GEOX if available
            has_valid = (
                (df["Lat"].abs() > 1e-6) & (df["Lon"].abs() > 1e-6)
                & (df["Lat"] >= -90) & (df["Lat"] <= 90)
                & (df["Lon"] >= -180) & (df["Lon"] <= 180)
            )
            if has_valid.any():
                lat_geo = df.loc[has_valid, "Lat"].values
                lon_geo = df.loc[has_valid, "Lon"].values
            else:
                # Convert local X,Y to WGS84 using helper
                origin, bearing = None, None
                if getattr(self.loader, "base_path", None):
                    ref = _get_reference_origin_and_bearing(self.loader.base_path)
                    if ref:
                        origin = (ref[0], ref[1])
                        bearing = ref[2]
                lat_geo, lon_geo = _local_xy_to_wgs84(
                    df["X"].values, df["Y"].values, origin=origin, bearing_deg=bearing)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(lon_geo, lat_geo, "-o", color="blue", markersize=3, linewidth=1, label="GEOX Path")
            ax.scatter(lon_geo[0], lat_geo[0], color="cyan", s=60, zorder=5, label="GEOX Start")
            ax.scatter(lon_geo[-1], lat_geo[-1], color="orange", s=60, zorder=5, label="GEOX End")
            ax.scatter(self.loader.lon[0], self.loader.lat[0], color="green", s=80, zorder=5, marker="^", label="GPS Start")
            ax.scatter(self.loader.lon[-1], self.loader.lat[-1], color="red", s=80, zorder=5, marker="v", label="GPS End")

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("GEOX path overlay on GPS")
            ax.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Cannot plot GEOX on GPS: {e}")

    def plot_geox_map(self):
        """Open current GEOX track (or chosen .geox file) on CartoDB Positron map in browser."""
        # Optional marker point dialog
        markers: list[tuple[float, float]] = []
        lat_s = simpledialog.askstring(
            "Marker latitude (optional)",
            "Marker latitude in decimal degrees (leave blank for none):",
            parent=self.root,
        )
        lon_s = None
        if lat_s:
            lon_s = simpledialog.askstring(
                "Marker longitude (optional)",
                "Marker longitude in decimal degrees:",
                parent=self.root,
            )
        if lat_s and lon_s:
            try:
                markers.append((float(lat_s.strip()), float(lon_s.strip())))
            except ValueError:
                messagebox.showwarning("Marker", "Marker lat/lon invalid – ignoring marker.")
        # Use loaded GEOX if available
        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            df = self.loader.xyz
            has_valid = (
                (df["Lat"].abs() > 1e-6) & (df["Lon"].abs() > 1e-6)
                & (df["Lat"] >= -90) & (df["Lat"] <= 90)
                & (df["Lon"] >= -180) & (df["Lon"] <= 180)
            )
            if has_valid.any():
                out = df.loc[has_valid][["Lat", "Lon"]].copy()
                out.columns = ["lat", "lon"]
            else:
                origin, bearing = None, None
                if getattr(self.loader, "base_path", None):
                    ref = _get_reference_origin_and_bearing(self.loader.base_path)
                    if ref:
                        origin = (ref[0], ref[1])
                        bearing = ref[2]
                lat, lon = _local_xy_to_wgs84(df["X"].values, df["Y"].values, origin=origin, bearing_deg=bearing)
                out = pd.DataFrame({"lat": lat, "lon": lon})
            try:
                lbl = self.loader.line_name or "GEOX"
                tracks = [(out["lat"].tolist(), out["lon"].tolist(), "#3498db", lbl)]
                mk = [(m[0], m[1], f"Marker {i+1}") for i, m in enumerate(markers)] if markers else None
                _open_map_popup(f"GEOX Map — {lbl}", tracks, markers=mk, zoom=18, root=self.root)
            except Exception as e:
                messagebox.showerror("Error", str(e))
            return
        # No GEOX loaded: open file dialog
        path = filedialog.askopenfilename(
            title="Select .geox file",
            filetypes=[("GEOX files", "*.geox *.GEOX")],
        )
        if not path:
            return
        try:
            df = _load_geo_simple(path)
            if df.empty:
                messagebox.showwarning("No coordinates", "No valid coordinates in file.")
                return
            lbl = os.path.basename(path)
            tracks = [(df["lat"].tolist(), df["lon"].tolist(), "#3498db", lbl)]
            mk = [(m[0], m[1], f"Marker {i+1}") for i, m in enumerate(markers)] if markers else None
            _open_map_popup(f"GEOX Map — {lbl}", tracks, markers=mk, zoom=18, root=self.root)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_geox_folder_map(self):
        """Plot all .geox files in a folder on CartoDB Positron map with labels and optional marker."""
        markers: list[tuple[float, float]] = []
        lat_s = simpledialog.askstring(
            "Marker latitude (optional)",
            "Marker latitude in decimal degrees (leave blank for none):",
            parent=self.root,
        )
        lon_s = None
        if lat_s:
            lon_s = simpledialog.askstring(
                "Marker longitude (optional)",
                "Marker longitude in decimal degrees:",
                parent=self.root,
            )
        if lat_s and lon_s:
            try:
                markers.append((float(lat_s.strip()), float(lon_s.strip())))
            except ValueError:
                messagebox.showwarning("Marker", "Marker lat/lon invalid – ignoring marker.")
        folder = filedialog.askdirectory(title="Select folder with .geox files")
        if not folder:
            return
        try:
            if not os.path.isdir(folder):
                raise ValueError(f"Not a folder: {folder}")
            track_list = []
            _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                        "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60"]
            idx = 0
            for name in sorted(os.listdir(folder)):
                ext = os.path.splitext(name)[1].lower()
                if ext != ".geox":
                    continue
                path = os.path.join(folder, name)
                try:
                    df = _load_geo_simple(path)
                    if df is not None and not df.empty:
                        track_list.append((df["lat"].tolist(), df["lon"].tolist(),
                                           _colors[idx % len(_colors)], name))
                        idx += 1
                except Exception:
                    continue
            if not track_list:
                raise ValueError("No valid .geox tracks found in folder.")
            mk = [(m[0], m[1], f"Marker {i+1}") for i, m in enumerate(markers)] if markers else None
            _open_map_popup(f"GEOX Folder — {os.path.basename(folder)}", track_list,
                            markers=mk, zoom=18, root=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"GEOX folder map error: {e}")

    def preprocess_for_ica(self, data, agc_window=None):
        """
        VERY conservative preprocessing for ICA.
        No trace-wise AGC. No Z-score.
        Only weak global normalization.
        """

        proc = data.astype(float).copy()

        # --- remove DC bias per trace (safe) ---
        proc -= np.mean(proc, axis=0, keepdims=True)

        # --- global RMS normalization (very weak, reversible) ---
        rms = np.sqrt(np.mean(proc**2))
        if rms > 0:
            proc /= rms

        return proc, rms


    def ica_multifractal_denoise(
            self,
            block_size=8,
            reject_ratio=0.08,
            agc_window=None,
            progress_var=None,
            progress_win=None
    ):
        raw = self.loader.data
        if raw is None:
            return

        # --- conservative preprocessing ---
        data, rms = self.preprocess_for_ica(raw)

        n_samples, n_traces = data.shape
        output = np.zeros_like(data)

        from sklearn.decomposition import FastICA

        for i in range(0, n_traces, block_size):
            j = min(i + block_size, n_traces)
            block = data[:, i:j].T

            if block.shape[0] < 2:
                output[:, i:j] = block.T
                continue

            ica = FastICA(
                whiten="unit-variance",
                max_iter=2000,
                tol=1e-3,
                random_state=0
            )

            try:
                S = ica.fit_transform(block)
                A = ica.mixing_
            except Exception:
                output[:, i:j] = block.T
                continue

            if A is None:
                output[:, i:j] = block.T
                continue

            # --- multifractal discrimination ---
            widths = np.array([
                self.multifractal_spectrum_width(S[:, k])
                for k in range(S.shape[1])
            ])

            # reject only extreme outliers
            threshold = np.percentile(widths, 100 * (1 - reject_ratio))
            reject_idx = np.where(widths > threshold)[0]

            S[:, reject_idx] = 0.0
            recon = S @ A.T

            output[:, i:j] = recon.T

            # --- progress ---
            if progress_var is not None:
                progress_var.set(100.0 * j / n_traces)
                if progress_win is not None:
                    progress_win.update_idletasks()

        # --- restore physical scale ---
        output *= rms

        self.loader.data = output

        if progress_var is not None:
            progress_var.set(100.0)
            if progress_win is not None:
                progress_win.update_idletasks()
        self.loader.data = self.loader.pca_gradient_wavelet_denoise(
            self.loader.data
        )
        self.plot_gpr()

    @staticmethod
    def multifractal_spectrum_width(signal):
        signal = np.asarray(signal)
        signal = signal[np.isfinite(signal)]

        if signal.size < 256:
            return 0.0

        scales = [4, 8, 16, 32]
        q = [-1, 1]

        eps = 1e-12
        slopes = []

        for qq in q:
            vals = []
            used = []
            for s in scales:
                if s >= signal.size:
                    continue
                diff = np.abs(signal[s:] - signal[:-s]) + eps
                vals.append(np.mean(diff ** qq))
                used.append(s)

            if len(vals) > 1:
                slopes.append(
                    np.polyfit(np.log(used), np.log(vals), 1)[0]
                )

        if len(slopes) < 2:
            return 0.0

        return abs(slopes[1] - slopes[0])

    def ica_denoise_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("ICA Multifractal Denoising (Conservative)")
        win.resizable(False, False)

        # ---------------- variables ----------------
        block_var = tk.IntVar(value=8)     # SAFE default
        perc_var = tk.IntVar(value=8)      # SAFE default (%)
        progress_var = tk.DoubleVar(value=0.0)

        row = 0

        # ---------------- block size ----------------
        tk.Label(win, text="Block size (traces)").grid(
            row=row, column=0, padx=10, pady=6, sticky="w"
        )
        tk.Spinbox(
            win, from_=4, to=20, increment=1,
            textvariable=block_var, width=8
        ).grid(row=row, column=1, padx=5, pady=6)
        row += 1

        # ---------------- reject ratio ----------------
        tk.Label(win, text="Reject ratio (%)").grid(
            row=row, column=0, padx=10, pady=6, sticky="w"
        )
        tk.Spinbox(
            win, from_=2, to=20, increment=1,
            textvariable=perc_var, width=8
        ).grid(row=row, column=1, padx=5, pady=6)
        row += 1

        # ---------------- progress bar ----------------
        ttk.Label(win, text="Progress").grid(
            row=row, column=0, padx=10, pady=(10, 2), sticky="w"
        )
        ttk.Progressbar(
            win,
            variable=progress_var,
            maximum=100,
            length=180,
            mode="determinate"
        ).grid(row=row, column=1, padx=10, pady=(10, 2))
        row += 1

        # ---------------- buttons ----------------
        tk.Button(
            win,
            text="Apply",
            command=lambda: self._apply_ica_denoise(
                block_var.get(),
                perc_var.get(),
                progress_var,
                win
            )
        ).grid(row=row, column=0, padx=10, pady=12)

        tk.Button(
            win,
            text="Cancel",
            command=win.destroy
        ).grid(row=row, column=1, padx=10, pady=12)

        win.grab_set()
    def _apply_ica_denoise(self, block, perc, progress_var, win):
        self.ica_multifractal_denoise(
            block_size=block,
            reject_ratio=perc / 100.0,
            progress_var=progress_var,
            progress_win=win
        )
        
    def run_hough_reflectors(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        try:
            # --- Extract long reflectors ---
            lines = self.loader.extract_interpretable_reflectors_hough()


            if not lines:
                messagebox.showinfo(
                    "Result",
                    "No continuous reflectors detected."
                )
                return

            # --- Plot ---
            fig = self.loader.plot_hough_reflectors(lines)

            # Display in main canvas
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(self.loader.data, [2, 98])
            ax.imshow(
                self.loader.data,
                cmap=self.cmap_var.get(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            for ln in lines:
                x1, y1, x2, y2 = ln["endpoints"]
                ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)

            ax.set_title("Long Continuous Reflectors (Hough Transform)")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            ax.invert_yaxis()

            self.figure.tight_layout()
            self.canvas.draw_idle()

            # --- Optional: print summary ---
            print(f"Hough reflectors detected: {len(lines)}")

        except Exception as e:
            messagebox.showerror("Hough Error", str(e))
        
    # self.meta_label.config(text=self.meta_text)

    # def preprocess_for_ica(self, data, agc_window=40):
        # """AGC + normalization (trace wise)"""

        # proc = data.copy()

        # # --- AGC ---
        # for i in range(proc.shape[1]):
            # trace = proc[:, i]
            # env = np.abs(hilbert(trace))
            # smooth = np.convolve(env, np.ones(agc_window)/agc_window, mode="same")
            # smooth[smooth == 0] = 1.0
            # proc[:, i] = trace / smooth

        # # --- Z-score normalization ---
        # mean = np.mean(proc, axis=0)
        # std = np.std(proc, axis=0)
        # std[std == 0] = 1.0
        # proc = (proc - mean) / std

        # return proc, mean, std
    # @staticmethod
    # def multifractal_spectrum_width(signal):
        # """
        # Robust multifractal spectrum width estimator
        # Noise-dominated ICA components have larger width
        # """
        # import numpy as np

        # signal = np.asarray(signal)
        # signal = signal[np.isfinite(signal)]

        # if signal.size < 128:
            # return 0.0

        # scales = [2, 4, 8, 16, 32, 64]
        # q = [-2, -1, 1, 2]

        # eps = 1e-12
        # Fq = []

        # for qq in q:
            # vals = []
            # used_scales = []
            # for s in scales:
                # if s >= signal.size:
                    # continue
                # diff = np.abs(signal[s:] - signal[:-s]) + eps
                # vals.append(np.mean(diff ** qq))
                # used_scales.append(s)

            # if len(vals) > 1:
                # Fq.append(
                    # np.polyfit(np.log(used_scales),
                               # np.log(vals), 1)[0]
                # )

        # if len(Fq) < 2:
            # return 0.0

        # return float(max(Fq) - min(Fq))

        

    # def ica_multifractal_denoise(self, block_size=10, reject_ratio=0.12,  agc_window=50, progress_var=None, progress_win=None):
        # raw = self.loader.data
        # if raw is None:
            # return

        # # --- preprocess ---
        # data, mean, std = self.preprocess_for_ica(raw)

        # n_samples, n_traces = data.shape
        # output = np.zeros_like(data)

        # from sklearn.decomposition import FastICA

        # for i in range(0, n_traces, block_size):
            # j = min(i + block_size, n_traces)
            # block = data[:, i:j].T

            # ica = FastICA(
                # whiten="unit-variance",
                # max_iter=3000,
                # tol=1e-3,
                # random_state=0
            # )

            # try:
                # S = ica.fit_transform(block)
                # A = ica.mixing_
            # except Exception:
                # output[:, i:j] = block.T
                # continue

            # if A is None:
                # output[:, i:j] = block.T
                # continue

            # # --- multifractal discrimination ---
            # widths = np.array([
                # # multifractal_spectrum_width(S[:, k])
                # self.multifractal_spectrum_width(S[:, k])

                # for k in range(S.shape[1])
            # ])

            # n_reject = int(reject_ratio * len(widths))
            # reject_idx = np.argsort(widths)[-n_reject:]

            # S[:, reject_idx] = 0.0
            # recon = np.dot(S, A.T)

            # output[:, i:j] = recon.T

        # # --- undo normalization ---
        # output = output * std + mean
        # if progress_var is not None:
            # progress = 100.0 * min(i + block_size, n_traces) / n_traces
            # progress_var.set(progress)
            # if progress_win is not None:
                # progress_win.update_idletasks()

        # self.loader.data = output
        # if progress_var is not None:
            # progress_var.set(100.0)
            # if progress_win is not None:
                # progress_win.update_idletasks()
        
        # self.plot_gpr()
        
    # def _apply_ica_denoise(self, block, agc, perc, progress_var, win):
        # self.ica_multifractal_denoise(
            # block_size=block,
            # agc_window=agc,
            # reject_ratio=perc / 100.0,
            # progress_var=progress_var,
            # progress_win=win
        # )


    # def ica_denoise_dialog(self):
        # win = tk.Toplevel(self.root)
        # win.title("ICA Multifractal Denoising")
        # win.resizable(False, False)

        # # --- variables ---
        # block_var = tk.IntVar(value=10)
        # agc_var = tk.IntVar(value=50)
        # perc_var = tk.IntVar(value=12)
        # progress_var = tk.DoubleVar(value=0.0)


        # # --- layout ---
        # row = 0

        # tk.Label(win, text="Block size (traces)").grid(
            # row=row, column=0, padx=10, pady=6, sticky="w"
        # )
        # tk.Spinbox(
            # win, from_=4, to=40, increment=1,
            # textvariable=block_var, width=8
        # ).grid(row=row, column=1, padx=5, pady=6)
        # row += 1

        # tk.Label(win, text="AGC window (samples)").grid(
            # row=row, column=0, padx=10, pady=6, sticky="w"
        # )
        # tk.Spinbox(
            # win, from_=10, to=200, increment=5,
            # textvariable=agc_var, width=8
        # ).grid(row=row, column=1, padx=5, pady=6)
        # row += 1

        # tk.Label(win, text="Reject ratio (%)").grid(
            # row=row, column=0, padx=10, pady=6, sticky="w"
        # )
        # tk.Spinbox(
            # win, from_=5, to=50, increment=1,
            # textvariable=perc_var, width=8
        # ).grid(row=row, column=1, padx=5, pady=6)
        # row += 1
        # # --- progress bar ---
        # ttk.Label(win, text="Progress").grid(
            # row=row, column=0, padx=10, pady=(10, 2), sticky="w"
        # )

        # progress = ttk.Progressbar(
            # win,
            # variable=progress_var,
            # maximum=100,
            # length=180,
            # mode="determinate"
        # )
        # progress.grid(row=row, column=1, padx=10, pady=(10, 2))
        # row += 1

        # # --- buttons ---
        # # --- buttons --- 
        # tk.Button( win, text="Apply", command=lambda: self._apply_ica_denoise( block_var.get(), agc_var.get(), perc_var.get(), progress_var, win ) ) .grid(row=row, column=0, padx=10, pady=12)
        

        # tk.Button(
            # win, text="Cancel",
            # command=win.destroy
        # ).grid(row=row, column=1, padx=10, pady=12)

        # win.grab_set()


 
    def show_ica_components(self, trace_start=0, block_size=20):
        data, _ = self.preprocess_for_ica(self.loader.data)

        block = data[:, trace_start:trace_start + block_size].T

        from sklearn.decomposition import FastICA
        ica = FastICA(whiten="unit-variance", random_state=0)
        S = ica.fit_transform(block)

        fig, axes = plt.subplots(
            nrows=min(8, S.shape[1]),
            figsize=(6, 10),
            sharex=True
        )

        for i, ax in enumerate(axes):
            ax.plot(S[:, i], color="black")
            ax.set_title(f"ICA Component {i}")

        fig.tight_layout()
        plt.show()

    def run_layer_picking(self, amp_p, max_jump, min_cov):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        # ---- Progress bar popup ----
        prog = tk.Toplevel(self.root)
        prog.title("Picking Layers")
        prog.geometry("300x90")
        prog.transient(self.root)
        prog.grab_set()

        tk.Label(prog, text="Picking layers...").pack(pady=5)
        pb = ttk.Progressbar(prog, mode="indeterminate", length=250)
        pb.pack(pady=10)
        pb.start(10)

        try:
            self.root.update_idletasks()

            layers = self.loader.pick_layers_semi_auto(
                amp_percentile=amp_p,
                max_vertical_jump=max_jump,
                min_trace_coverage=min_cov
            )

            pb.stop()
            prog.destroy()

            if not layers:
                messagebox.showinfo("Result", "No layers detected.")
                return

            # ---- Plot ----
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(self.loader.data, [2, 98])
            ax.imshow(
                self.loader.data,
                cmap=self.cmap_var.get(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            for lyr in layers:
                xs = [p[1] for p in lyr]
                ys = [p[0] for p in lyr]
                ax.plot(xs, ys, 'r', linewidth=1.5)

            ax.set_title("Semi Automatic Layer Picking")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            ax.invert_yaxis()

            self.figure.tight_layout()
            self.canvas.draw_idle()

            print(f"Layers picked: {len(layers)}")

        except Exception as e:
            pb.stop()
            prog.destroy()
            messagebox.showerror("Layer Picking Error", str(e))

    
    
    def layer_picker_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Layer Picking Parameters")
        popup.geometry("300x220")
        popup.transient(self.root)
        popup.grab_set()

        tk.Label(popup, text="Amplitude Percentile").pack(pady=4)
        amp_var = tk.DoubleVar(value=55)
        tk.Entry(popup, textvariable=amp_var).pack()

        tk.Label(popup, text="Max Vertical Jump (samples)").pack(pady=4)
        jump_var = tk.IntVar(value=4)
        tk.Entry(popup, textvariable=jump_var).pack()

        tk.Label(popup, text="Min Trace Coverage (0–1)").pack(pady=4)
        cov_var = tk.DoubleVar(value=0.4)
        tk.Entry(popup, textvariable=cov_var).pack()

        def run():
            popup.destroy()
            self.run_layer_picking(
                amp_var.get(),
                jump_var.get(),
                cov_var.get()
            )

        tk.Button(popup, text="Run", bg="#C8E6C9", command=run).pack(pady=12)

    def show_amplitude_map(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        try:
            # ---- Compute envelope ----
            from scipy.signal import hilbert
            env = np.abs(hilbert(self.loader.data, axis=0))

            # ---- Plot on existing canvas ----
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(env, [5, 95])
            im = ax.imshow(
                env,
                cmap="hot",
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title("Instantaneous Amplitude (Envelope Map)")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            # ax.invert_yaxis()

            self.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            self.figure.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Amplitude Map Error", str(e))
    def peaks_extraction_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Peaks Extraction")
        popup.geometry("320x360")
        popup.transient(self.root)
        popup.grab_set()

        # --- Peak type ---
        tk.Label(popup, text="Select Peaks").pack(pady=4)
        peak_type = tk.StringVar(value="all")
        ttk.Combobox(
            popup,
            textvariable=peak_type,
            values=["all", "positive", "negative"],
            state="readonly"
        ).pack()

        # --- Max peaks ---
        tk.Label(popup, text="Max # of Points (per trace)").pack(pady=4)
        max_peaks = tk.IntVar(value=3)
        tk.Entry(popup, textvariable=max_peaks).pack()

        # --- Vertical width ---
        tk.Label(popup, text="Samples / Point").pack(pady=4)
        samp_width = tk.IntVar(value=3)
        tk.Entry(popup, textvariable=samp_width).pack()

        # --- Start / End samples ---
        tk.Label(popup, text="Start Sample").pack(pady=4)
        start_samp = tk.IntVar(value=0)
        tk.Entry(popup, textvariable=start_samp).pack()

        tk.Label(popup, text="End Sample").pack(pady=4)
        end_samp = tk.IntVar(value=self.loader.data.shape[0] if self.loader.data is not None else 0)
        tk.Entry(popup, textvariable=end_samp).pack()

        def run():
            popup.destroy()
            self.run_peaks_extraction(
                peak_type.get(),
                max_peaks.get(),
                samp_width.get(),
                start_samp.get(),
                end_samp.get()
            )

        tk.Button(popup, text="Apply", bg="#C8E6C9", command=run).pack(pady=12)
            
    def run_peaks_extraction(
        self,
        peak_type,
        max_peaks,
        samp_width,
        start_samp,
        end_samp
    ):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        try:
            peak_data = self.loader.extract_peaks(
                peak_type=peak_type,
                max_peaks=max_peaks,
                samples_per_point=samp_width,
                start_sample=start_samp,
                end_sample=end_samp
            )

            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(peak_data[peak_data != 0], [5, 95]) \
                         if np.any(peak_data) else (None, None)

            ax.imshow(
                peak_data,
                cmap=self.cmap_var.get(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(f'Peaks Extraction - {self.loader.line_name}: {self.loader.file_path}')
            # ax.set_title("Peaks Extraction")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            # ax.invert_yaxis()

            self.figure.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Peaks Extraction Error", str(e))
    def background_removal_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Background Removal (Horizontal FIR)")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # ------------------ BR Type ------------------
        tk.Label(win, text="BR Type").grid(row=0, column=0, padx=10, pady=6, sticky="e")

        br_type = tk.StringVar(value="full")
        ttk.Combobox(
            win,
            textvariable=br_type,
            values=["full", "scan_range", "adaptive"],
            state="readonly",
            width=15
        ).grid(row=0, column=1, padx=5, pady=6)

        # ------------------ Filter Length ------------------
        tk.Label(win, text="Filter Length (scans)").grid(
            row=1, column=0, padx=10, pady=6, sticky="e"
        )
        flen = tk.IntVar(value=200)
        tk.Entry(win, textvariable=flen, width=10).grid(row=1, column=1)

        # ------------------ Scan Range ------------------
        tk.Label(win, text="Start Scan").grid(row=2, column=0, padx=10, pady=6, sticky="e")
        start_var = tk.IntVar(value=0)
        tk.Entry(win, textvariable=start_var, width=10).grid(row=2, column=1)

        tk.Label(win, text="End Scan").grid(row=3, column=0, padx=10, pady=6, sticky="e")
        end_var = tk.IntVar(value=self.loader.data.shape[1])
        tk.Entry(win, textvariable=end_var, width=10).grid(row=3, column=1)

        # ------------------ APPLY ------------------
        def apply():
            try:
                self.loader.background_removal(
                    br_type=br_type.get(),
                    filter_length=flen.get(),
                    start_scan=start_var.get(),
                    end_scan=end_var.get()
                )
                win.destroy()
                self.plot_gpr()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(
            win, text="Apply",
            bg="#4CAF50", fg="white",
            width=12,
            command=apply
        ).grid(row=4, column=0, columnspan=2, pady=12)
    def range_gain_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Range Gain")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # ---------------- Gain Type ----------------
        tk.Label(win, text="Gain Type").grid(row=0, column=0, padx=10, pady=6, sticky="e")
        gain_type = tk.StringVar(value="automatic")
        ttk.Combobox(
            win,
            textvariable=gain_type,
            values=["automatic", "linear", "exponential", "smart"],
            state="readonly",
            width=15
        ).grid(row=0, column=1)

        # ---------------- Number of Points ----------------
        tk.Label(win, text="# of Points").grid(row=1, column=0, padx=10, pady=6, sticky="e")
        npts = tk.IntVar(value=6)
        tk.Entry(win, textvariable=npts, width=10).grid(row=1, column=1)

        # ---------------- Overall Gain ----------------
        tk.Label(win, text="Overall Gain (dB)").grid(row=2, column=0, padx=10, pady=6, sticky="e")
        ogain = tk.DoubleVar(value=3.0)
        tk.Entry(win, textvariable=ogain, width=10).grid(row=2, column=1)

        # ---------------- Horizontal TC ----------------
        tk.Label(win, text="Horiz TC (scans)").grid(row=3, column=0, padx=10, pady=6, sticky="e")
        htc = tk.IntVar(value=15)
        tk.Entry(win, textvariable=htc, width=10).grid(row=3, column=1)

        # ---------------- Apply ----------------
        def apply():
            try:
                self.loader.range_gain(
                    gain_type=gain_type.get(),
                    n_points=npts.get(),
                    overall_gain_db=ogain.get(),
                    horiz_tc=htc.get()
                )
                win.destroy()
                self.plot_gpr()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(
            win,
            text="Apply",
            bg="#2196F3",
            fg="white",
            width=12,
            command=apply
        ).grid(row=4, column=0, columnspan=2, pady=12)
    
    # ────────────────────────────────────────────────────────────────
    #  Deconvolution with parameter dialog + progress bar
    # ────────────────────────────────────────────────────────────────

    def deconvolution_dialog(self):
        """
        Opens a dialog where user can set deconvolution parameters
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Predictive Deconvolution Settings")
        dialog.geometry("420x380")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # ─── Variables ───────────────────────────────────────────────
        op_len_var   = tk.IntVar(value=32)
        lag_var      = tk.IntVar(value=4)
        prewhite_var = tk.DoubleVar(value=0.1)
        gain_var     = tk.DoubleVar(value=1.0)
        start_var    = tk.IntVar(value=0)
        end_var      = tk.IntVar(value=self.loader.data.shape[0])

        progress_var = tk.DoubleVar(value=0.0)

        row = 0

        # ─── Labels & Entries ──────────────────────────────────────
        tk.Label(dialog, text="Operator length (samples):").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=op_len_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Prediction lag (samples):").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=lag_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Prewhitening (%):").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=prewhite_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Overall gain factor:").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=gain_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Start sample:").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=start_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="End sample:").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=end_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        # Progress bar
        tk.Label(dialog, text="Progress:").grid(row=row, column=0, padx=12, pady=(16,4), sticky="e")
        progress = ttk.Progressbar(dialog, orient="horizontal", length=260,
                                   mode="determinate", variable=progress_var)
        progress.grid(row=row, column=1, padx=8, pady=(16,4), sticky="w")
        row += 1

        # ─── Buttons ───────────────────────────────────────────────
        btn_frame = tk.Frame(dialog)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=20)

        def run_decon():
            try:
                op_len   = op_len_var.get()
                lag      = lag_var.get()
                prewhite = prewhite_var.get()
                gain     = gain_var.get()
                start    = start_var.get()
                end      = end_var.get()

                if op_len < 4 or lag < 1 or lag >= op_len:
                    messagebox.showwarning("Invalid parameters", "Check operator length and lag values.")
                    return

                if start < 0 or end <= start or end > self.loader.data.shape[0]:
                    messagebox.showwarning("Invalid gate", "Check start/end sample values.")
                    return

                # Close dialog and run processing
                dialog.destroy()

                self._run_predictive_decon_with_progress(
                    operator_length=op_len,
                    prediction_lag=lag,
                    prewhitening=prewhite,
                    overall_gain=gain,
                    start_sample=start,
                    end_sample=end,
                    progress_var=progress_var
                )

            # except Exception as exc:
            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showerror("Deconvolution failed", str(err)))

        tk.Button(btn_frame, text="Apply Deconvolution",
                  command=run_decon,
                  bg="#4CAF50", fg="white", width=18, font=("Helvetica", 10, "bold")).pack(side="left", padx=10)

        tk.Button(btn_frame, text="Cancel",
                  command=dialog.destroy,
                  width=12).pack(side="left", padx=10)

    def _run_predictive_decon_with_progress(self,
                                           operator_length,
                                           prediction_lag,
                                           prewhitening,
                                           overall_gain,
                                           start_sample,
                                           end_sample,
                                           progress_var):
        """
        Runs deconvolution in the background thread + updates progress
        """
        def task():
            try:
                self.loader.predictive_deconvolution(
                    operator_length=operator_length,
                    prediction_lag=prediction_lag,
                    prewhitening=prewhitening,
                    overall_gain=overall_gain,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    progress_callback=lambda p: progress_var.set(p * 100)
                )
                # Refresh display on main thread
                self.root.after(0, self.plot_gpr)
                self.root.after(0, lambda: messagebox.showinfo("Done", "Predictive deconvolution completed."))

            except Exception as exc:
                self.root.after(0, lambda err=exc: messagebox.showerror("Deconvolution failed", str(err)))

            finally:
                self.root.after(0, lambda: progress_var.set(0))

        # Run in background thread so GUI stays responsive
        import threading
        thread = threading.Thread(target=task, daemon=True)
        thread.start() 
    def depth_crop_dialog(self):
        if self.loader.data is None or self.loader.depth is None:
            tk.messagebox.showerror(
                "Depth Crop",
                "Depth information not available.\nLoad HDR data first."
            )
            return

        win = tk.Toplevel(self.root)
        win.title("Depth Crop")
        win.resizable(False, False)

        tk.Label(
            win,
            text="Maximum Depth (meters)",
            font=("Helvetica", 11)
        ).grid(row=0, column=0, padx=10, pady=8)

        depth_var = tk.DoubleVar(value=float(self.loader.depth[-1]))

        tk.Entry(
            win,
            textvariable=depth_var,
            width=10
        ).grid(row=0, column=1, padx=10, pady=8)

        info = f"Available depth: 0 – {self.loader.depth[-1]:.2f} m"
        tk.Label(win, text=info, fg="gray").grid(
            row=1, column=0, columnspan=2, pady=4
        )

        def apply_crop():
            try:
                self.loader.depth_crop(depth_var.get())
                self.canvas.draw_idle()
                win.destroy()
            except Exception as e:
                tk.messagebox.showerror("Depth Crop Error", str(e))

        btn_frame = tk.Frame(win)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Button(
            btn_frame,
            text="Apply",
            width=10,
            bg="#4CAF50",
            fg="white",
            command=apply_crop
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            command=win.destroy
        ).pack(side="right", padx=5)
    def depth_view_dialog(self):
        if self.loader.depth is None:
            tk.messagebox.showerror(
                "Depth View",
                "Depth information not available.\nLoad HDR data first."
            )
            return

        win = tk.Toplevel(self.root)
        win.title("Depth View")
        win.resizable(False, False)

        max_depth = float(self.loader.depth[-1])

        tk.Label(
            win,
            text="Display Depth (meters)",
            font=("Helvetica", 11)
        ).grid(row=0, column=0, padx=10, pady=8)

        depth_var = tk.DoubleVar(
            value=max_depth if self.view_max_depth is None else self.view_max_depth
        )

        entry = tk.Entry(win, textvariable=depth_var, width=8)
        entry.grid(row=0, column=1, padx=10, pady=8)
        entry.focus()

        tk.Label(
            win,
            text=f"Available depth: 0 – {max_depth:.2f} m",
            fg="gray"
        ).grid(row=1, column=0, columnspan=2)

        def apply_view():
            try:
                val = float(depth_var.get())
                if val <= 0 or val > max_depth:
                    raise ValueError

                self.view_max_depth = val
                self.plot_gpr()   
                win.destroy()

            except Exception:
                tk.messagebox.showerror(
                    "Invalid Depth",
                    f"Enter a value between 0 and {max_depth:.2f}"
                )

        def reset_view():
            self.view_max_depth = None
            self.canvas.draw_idle()
            win.destroy()

        btns = tk.Frame(win)
        btns.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Button(
            btns, text="Apply",
            bg="#4CAF50", fg="white",
            width=10, command=apply_view
        ).pack(side="left", padx=5)

        tk.Button(
            btns, text="Reset",
            width=10, command=reset_view
        ).pack(side="right", padx=5)

    # ------------------------------------------------------------------
    # Time Zero Correction (from V3, adapted)
    # ------------------------------------------------------------------
    def time_zero_correction_dialog(self):
        """
        Time Zero Correction (RADAN7-style): popup A-scan to set first positive peak
        of the direct wave to time zero. User clicks on the A-scan where the first
        arrival is; profile is shifted so that sample becomes time zero.
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        data = self.loader.data.astype(float)
        n_samples, n_traces = data.shape
        mean_trace = np.mean(data, axis=1)
        y_axis = (
            self.loader.depth
            if self.loader.depth is not None
            else np.arange(n_samples, dtype=float)
        )
        if y_axis is None or len(y_axis) != n_samples:
            y_axis = np.arange(n_samples, dtype=float)

        win = tk.Toplevel(self.root)
        win.title("Time Zero Correction (RADAN7-style)")
        win.geometry("420x520")
        win.transient(self.root)

        tk.Label(
            win,
            text="Click on the A-scan at the first positive peak of the direct wave (time zero).\nThen click Apply to shift the profile.",
            font=("Helvetica", 10),
            justify="left",
            wraplength=380,
        ).pack(pady=8, padx=10, anchor="w")

        fig = plt.Figure(figsize=(4, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(mean_trace, y_axis, color="black", linewidth=1.0)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_ylim(y_axis[-1], y_axis[0])
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Depth (m)" if self.loader.depth is not None else "Sample")
        ax.set_title("A-scan (mean trace) – click to set time zero")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        canvas.draw()

        shift_sample = [0]
        time_zero_line = [None]

        def click_to_sample_index(event):
            """Map click y-position to sample index (0 = top, n_samples-1 = bottom)."""
            if event.inaxes is None or event.ydata is None:
                return None
            y_lim = event.inaxes.get_ylim()
            y_top, y_bottom = y_lim[1], y_lim[0]
            if abs(y_top - y_bottom) < 1e-12:
                return 0
            frac = (y_top - event.ydata) / (y_top - y_bottom)
            frac = max(0.0, min(1.0, frac))
            idx = int(round(frac * (n_samples - 1)))
            return np.clip(idx, 0, n_samples - 1)

        def on_click(event):
            idx = click_to_sample_index(event)
            if idx is None:
                return
            shift_sample[0] = idx
            if time_zero_line[0] is not None:
                time_zero_line[0].remove()
            y_val = y_axis[idx]
            time_zero_line[0] = ax.axhline(
                y=y_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Time zero",
            )
            ax.set_title(f"Time zero at sample {idx} (click to change)")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_click)

        def apply_time_zero():
            shift = shift_sample[0]
            if shift <= 0:
                win.destroy()
                return
            self._apply_time_zero_correction(shift)
            win.destroy()

        def auto_first_peak():
            from scipy.signal import find_peaks

            trace = mean_trace
            peaks, _ = find_peaks(trace)
            if len(peaks) > 0:
                idx = int(peaks[0])
                shift_sample[0] = idx
                if time_zero_line[0] is not None:
                    time_zero_line[0].remove()
                y_val = y_axis[idx]
                time_zero_line[0] = ax.axhline(
                    y=y_val,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label="Time zero",
                )
                ax.set_title(f"Time zero at sample {idx} (Auto first peak)")
                fig.canvas.draw_idle()
            else:
                messagebox.showinfo("Info", "No positive peak found; click manually.")

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Auto (first peak)",
            command=auto_first_peak,
            bg="#E3F2FD",
            width=14,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            btn_frame,
            text="Apply",
            command=apply_time_zero,
            bg="#4CAF50",
            fg="white",
            width=10,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            btn_frame,
            text="Cancel",
            command=win.destroy,
            width=10,
        ).pack(side=tk.LEFT, padx=4)

    def _apply_time_zero_correction(self, shift_samples: int):
        """Shift all traces so that sample index shift_samples becomes time zero (roll up, zero-fill)."""
        if self.loader.data is None or shift_samples <= 0:
            return
        data = self.loader.data.astype(float)
        shifted = np.roll(data, -shift_samples, axis=0)
        shifted[-shift_samples:, :] = 0
        self.loader.data = shifted
        self.loader.add_process("TimeZero")
        self.plot_gpr()

    # ------------------------------------------------------------------
    # Kirchhoff Migration (from V3, adapted)
    # ------------------------------------------------------------------
    def kirchhoff_migration_dialog(self):
        """RADAN7-style: B-scan with interactive hyperbola overlay. Drag apex to fit reflection, then Apply to run Kirchhoff migration."""
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return
        data = self.loader.data.astype(float)
        n_samples, n_traces = data.shape
        dt_s = getattr(self.loader, "_sample_interval_s", None) or (1.0 / 1e9)
        dt_ns = dt_s * 1e9
        dx_m = 1.0
        if getattr(self.loader, "_x_cell_m", None) is not None and self.loader._x_cell_m > 0:
            dx_m = float(self.loader._x_cell_m)
        c_light = 0.299792458  # m/ns
        v_default = c_light / np.sqrt(9.0)

        win = tk.Toplevel(self.root)
        win.title("Kirchhoff Migration – Fit hyperbola (RADAN7)")
        win.geometry("820x620")
        win.transient(self.root)

        # State: apex (trace, sample), velocity; drag mode
        trace_apex = [max(0, n_traces // 2)]
        sample_apex = [max(0, n_samples // 4)]
        vel_m_per_ns = [v_default]
        dragging = [False]

        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        extent = [0, n_traces, n_samples, 0]
        vmin, vmax = np.percentile(data, [2, 98])
        ax.imshow(data, aspect="auto", extent=extent, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample (depth)")
        ax.set_title("Drag red hyperbola apex (white square) to fit reflection; set velocity for curvature")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        hyperbola_line = [None]
        apex_handle = [None]

        def hyperbola_curve(tr_apex, sm_apex, v):
            """Sample index at each trace for hyperbola with apex (sm_apex, tr_apex) and velocity v (m/ns)."""
            z_m = v * (sm_apex * dt_ns) / 2.0
            traces = np.arange(n_traces, dtype=float)
            x_off = (traces - tr_apex) * dx_m
            t_ns = (2.0 / v) * np.sqrt(z_m**2 + x_off**2)
            samples = t_ns / dt_ns
            samples = np.clip(samples, 0, n_samples - 1)
            return traces, samples

        def draw_hyperbola():
            tr_apex, sm_apex = trace_apex[0], sample_apex[0]
            v = vel_m_per_ns[0]
            if v <= 0:
                return
            xx, yy = hyperbola_curve(tr_apex, sm_apex, v)
            if hyperbola_line[0] is not None:
                hyperbola_line[0].remove()
            hyperbola_line[0] = ax.plot(xx, yy, "r-", linewidth=2.5, alpha=0.85, label="Fit hyperbola")[0]
            if apex_handle[0] is not None:
                apex_handle[0].remove()
            apex_handle[0] = ax.plot(
                tr_apex,
                sm_apex,
                "s",
                color="white",
                markersize=10,
                markeredgecolor="red",
                markeredgewidth=2,
                picker=5,
                zorder=5,
            )[0]
            fig.canvas.draw_idle()

        draw_hyperbola()

        def on_press(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            tx, sy = event.xdata, event.ydata
            tr_a, sm_a = trace_apex[0], sample_apex[0]
            if abs(tx - tr_a) < 20 and abs(sy - sm_a) < 20:
                dragging[0] = True

        def on_motion(event):
            if not dragging[0] or event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            trace_apex[0] = int(round(np.clip(event.xdata, 0, n_traces - 1)))
            sample_apex[0] = int(round(np.clip(event.ydata, 0, n_samples - 1)))
            draw_hyperbola()

        def on_release(event):
            dragging[0] = False

        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("motion_notify_event", on_motion)
        canvas.mpl_connect("button_release_event", on_release)

        params = tk.Frame(win)
        params.pack(fill=tk.X, padx=10, pady=6)
        tk.Label(params, text="Method: Kirchhoff", font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, padx=8, pady=4, sticky="w"
        )
        tk.Label(params, text="Velocity (m/ns):").grid(row=1, column=0, padx=8, pady=2, sticky="e")
        vel_var = tk.DoubleVar(value=round(v_default, 4))
        tk.Entry(params, textvariable=vel_var, width=10).grid(row=1, column=1, padx=4, pady=2)
        tk.Label(params, text="Or Dielectric (εr):").grid(row=2, column=0, padx=8, pady=2, sticky="e")
        eps_var = tk.DoubleVar(value=9.0)
        tk.Entry(params, textvariable=eps_var, width=10).grid(row=2, column=1, padx=4, pady=2)

        def sync_velocity():
            v = vel_var.get()
            if v <= 0 and eps_var.get() > 0:
                v = c_light / np.sqrt(eps_var.get())
                vel_var.set(round(v, 4))
            if v > 0:
                vel_m_per_ns[0] = v
                draw_hyperbola()

        tk.Button(params, text="Update curve", command=sync_velocity, bg="#E3F2FD").grid(
            row=1, column=2, padx=8, pady=2
        )

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=8)

        def apply_mig():
            v = vel_var.get()
            if v <= 0 and eps_var.get() > 0:
                v = c_light / np.sqrt(eps_var.get())
            if v <= 0:
                messagebox.showerror("Error", "Velocity must be positive.")
                return
            self._apply_kirchhoff_migration(v)
            win.destroy()

        tk.Button(btn_frame, text="Apply", command=apply_mig, bg="#4CAF50", fg="white", width=10).pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(btn_frame, text="Cancel", command=win.destroy, width=10).pack(side=tk.LEFT, padx=6)

    def _apply_kirchhoff_migration(self, velocity_m_per_ns: float):
        """
        Kirchhoff summation migration: for each output (depth, position), sum the
        display data along the diffraction hyperbola that a point scatterer at that
        (z, x) would produce.
        """
        if self.loader.data is None or velocity_m_per_ns <= 0:
            return
        data = self.loader.data.astype(float)
        n_samples, n_traces = data.shape
        dt_s = getattr(self.loader, "_sample_interval_s", None) or (1.0 / 1e9)
        dt_ns = dt_s * 1e9
        dx_m = 1.0
        if getattr(self.loader, "_x_cell_m", None) is not None and self.loader._x_cell_m > 0:
            dx_m = float(self.loader._x_cell_m)
        v = velocity_m_per_ns
        out = np.zeros_like(data)
        for iz in range(n_samples):
            # Depth (m) corresponding to output sample iz: z = v * t_one_way = v * (iz*dt_ns)/2
            z_m = v * (iz * dt_ns) / 2.0
            for ix in range(n_traces):
                # Sum input along the hyperbola for scatterer at (z_m, trace ix).
                weighted_sum = 0.0
                weight_sum = 0.0
                for jx in range(n_traces):
                    x_offset_m = (jx - ix) * dx_m
                    t_ns = (2.0 / v) * np.sqrt(z_m**2 + x_offset_m**2)
                    it = int(round(t_ns / dt_ns))
                    if 0 <= it < n_samples:
                        # Amplitude correction for spherical divergence (1/sqrt(t))
                        w = 1.0 / np.sqrt(t_ns + dt_ns * 0.5)
                        weighted_sum += w * data[it, jx]
                        weight_sum += w
                if weight_sum > 0:
                    out[iz, ix] = weighted_sum / weight_sum
        self.loader.data = out
        self.loader.add_process("KirchhoffMig")
        self.plot_gpr()

    # ----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------- 
    def show_metadata(self):
        meta = self.loader.get_metadata() if hasattr(self.loader, "get_metadata") else None
        hdr = self.loader.get_hdr_parameters() if hasattr(self.loader, "get_hdr_parameters") else None
        meta = meta if meta is not None else {}
        hdr = hdr if hdr is not None else {}

        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, "=== FILE & DATA INFO ===\n")
        for k, v in meta.items():
            self.meta_text.insert(tk.END, f"{k:22}: {v}\n")

        self.meta_text.insert(tk.END, "\n=== HDR PARAMETERS ===\n")
        for k, v in hdr.items():
            self.meta_text.insert(tk.END, f"{k:22}: {v}\n")

    # def plot_gpr(self):
        # if self.cmap_var.get() == "wiggle":
            # self.plot_wiggle()
            # return

        # self.figure.clf()
        # ax = self.figure.add_subplot(111)

        # data = self.loader.data
        # if data is None:
            # return

        # x_axis = (
            # self.loader.xyz['X'].values
            # if self.loader.xyz is not None and len(self.loader.xyz) > 0
            # else np.arange(data.shape[1])
        # )

        # y_axis = (
            # self.loader.depth
            # if self.loader.depth is not None
            # else np.arange(data.shape[0])
        # )

        # extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]

        # # ────────────────────────────────────────────────
        # # Improved contrast handling
        # # ────────────────────────────────────────────────
        # abs_data = np.abs(data)
        # mad = np.median(abs_data)                      # robust scale
        # if mad < 1e-6:                                 # almost zero data
            # vmin, vmax = -1, 1
        # else:
            # clip = 5.0 * mad                           # aggressive but GPR-friendly
            # vmin = -clip
            # vmax = clip

        # # Alternative: still use percentiles but with floor
        # # p2, p98 = np.percentile(data, [2, 98])
        # # vmin = max(p2, -3 * mad)
        # # vmax = min(p98,  3 * mad)

        # self.current_image = ax.imshow(
            # data,
            # cmap=self.cmap_var.get(),
            # aspect='auto',
            # extent=extent,
            # vmin=vmin,
            # vmax=vmax,
            # interpolation='nearest'   # optional: sharper look
        # )

        # ax.set_xlabel("Distance (m)")
        # ax.set_ylabel("Depth (m)")
        # ax.set_title(f'GPR Section - {self.loader.line_name} {self.loader.file_path}')

        # # ax.invert_yaxis()           # ← make sure this is here!
        # self.figure.tight_layout()
        # self.canvas.draw_idle()            
    def _get_x_all_and_length(self):
        """Return (x_all in m for each trace, total_length_m).

        IMPORTANT: For distance we now strictly follow the GPR-GUI-2026 style:
        - If GEOX / xyz is available, use its X column directly (absolute distance from GRED/IDS).
        - If GEOX is missing, fall back ONLY to a simple synthetic axis based on X_CELL (or trace index).
        - We DO NOT force additional offsets from HDR (no manual X_OFFSET accumulation here).
        """
        data = self.loader.data
        if data is None:
            return None, None
        n_traces = data.shape[1]
        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            x_all = np.asarray(self.loader.xyz["X"].values, dtype=float)
            if len(x_all) != n_traces:
                x_all = np.linspace(float(x_all[0]), float(x_all[-1]), n_traces)
        else:
            dx = getattr(self.loader, "_x_cell_m", None)
            if dx is not None:
                x_all = np.arange(n_traces, dtype=float) * float(dx)
            else:
                x_all = np.arange(n_traces, dtype=float)

        # In folder/segment mode we want the first segment to start at 0 m
        # and subsequent segments to be contiguous. We rebase by the first
        # segment's first X value (stored in _folder_x0_offset).
        if self.folder_segment_files and self._folder_x0_offset is not None:
            x_all = x_all - float(self._folder_x0_offset)

        total_m = float(x_all[-1] - x_all[0]) if n_traces > 1 else 0.0
        self._total_length_m = total_m
        return x_all, total_m

    def _num_x_window_segments(self):
        """Number of X-window segments (by distance) for current line."""
        if self._total_length_m is None:
            self._get_x_all_and_length()
        if self._total_length_m is None or self._total_length_m <= 0:
            return 1
        w = self._get_window_m()
        return max(1, int(np.ceil(self._total_length_m / w)))

    def _view_prev_40m(self):
        if self.loader.data is None:
            return
        w = self._get_window_m()
        self.view_x_start_m = max(0.0, self.view_x_start_m - w)
        self._sync_x_ch_vars()
        self.plot_gpr()

    def _view_next_40m(self):
        if self.loader.data is None:
            return
        x_all, total_m = self._get_x_all_and_length()
        if total_m is None:
            return
        w = self._get_window_m()
        max_start = max(0.0, total_m - w)
        self.view_x_start_m = min(max_start, self.view_x_start_m + w)
        self._sync_x_ch_vars()
        self.plot_gpr()

    def _sync_x_ch_vars(self):
        """Sync the X position entry to show absolute distance (not relative offset)."""
        # Get the actual starting X value from GEOX (or 0 if not available)
        x0 = self._get_x0()
        # Display absolute X position = relative offset + x0
        abs_x = self.view_x_start_m + x0
        self.x_pos_var.set(f"{abs_x:.1f}")
        if self.folder_segment_files:
            self._sync_segment_display(len(self.folder_segment_files))

    def _get_x0(self):
        """Get the starting X value for the current profile (for X entry box).

        - In folder/segment mode, we use the profile-rebased axis where the
          very first segment starts at 0 m.
        - In single-file mode, we show the raw GEOX / XYZ distance.
        """
        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            x0_raw = float(self.loader.xyz["X"].iloc[0])
            if self.folder_segment_files and self._folder_x0_offset is not None:
                return x0_raw - float(self._folder_x0_offset)
            return x0_raw
        return 0.0

    def plot_gpr(self):
        if self.cmap_var.get() == "wiggle":
            self.plot_wiggle()
            return

        self.figure.clf()
        ax = self.figure.add_subplot(111)

        # Reset crosshair artists after a redraw; they will be recreated on next hover
        self.crosshair_hline = None
        self.crosshair_vline = None
        self.crosshair_text = None

        data = self.loader.data
        if data is None:
            return

        x_all, total_m = self._get_x_all_and_length()
        n_traces = data.shape[1]
        if x_all is None or n_traces == 0:
            return

        # 40 m window: select trace range
        x0 = float(x_all[0])
        w_m = self._get_window_m()
        use_window = total_m is not None and total_m > w_m and self.view_x_start_m is not None
        if use_window:
            start_m = self.view_x_start_m + x0
            end_m = start_m + w_m
            mask = (x_all >= start_m) & (x_all < end_m)
            if not np.any(mask):
                mask = (x_all >= start_m) & (x_all <= end_m)
            t_idx = np.where(mask)[0]
            if len(t_idx) == 0:
                t_idx = np.array([0, min(1, n_traces - 1)])
            t0, t1 = int(t_idx[0]), int(t_idx[-1]) + 1
            data = data[:, t0:t1]
            x_axis = x_all[t0:t1]
        else:
            x_axis = x_all

        y_axis = (
            self.loader.depth
            if self.loader.depth is not None
            else np.arange(data.shape[0])
        )

        x0_plot = float(x_axis[0])
        x1_plot = float(x_axis[-1])
        y0_plot = float(y_axis[-1])
        y1_plot = float(y_axis[0])
        if x1_plot <= x0_plot:
            x1_plot = x0_plot + 1.0
        if y0_plot == y1_plot:
            y0_plot = y1_plot + 1.0
        extent = [x0_plot, x1_plot, y0_plot, y1_plot]
        vmin, vmax = np.percentile(data, [2, 98])

        self.current_image = ax.imshow(
            data,
            cmap=self.cmap_var.get(),
            aspect='auto',
            extent=extent,
            vmin=vmin,
            vmax=vmax
        )
        # ---- Depth view (non-destructive) ----
        if self.view_max_depth is not None and self.loader.depth is not None:
            ax.set_ylim(self.view_max_depth, 0)

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        title = self.loader.file_path if self.loader.file_path else f"GPR Section - {self.loader.line_name}"
        if getattr(self.loader, "process_history", None) and len(self.loader.process_history) > 0:
            title += "  [Processing: " + ", ".join(self.loader.process_history) + "]"
        if getattr(self, "_hilbert_display_mode", None):
            title += "  — " + self._hilbert_display_mode
        if use_window:
            title += f"  [X: {float(x_axis[0]):.1f}–{float(x_axis[-1]):.1f} m]"
        ax.set_title(title)

        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.canvas.flush_events()

    # def reset_data(self):
        # if hasattr(self, "original_data") and self.original_data is not None:
            # self.loader.data = self.original_data.copy()
            # self.plot_gpr()
        # else:
            # messagebox.showinfo("Info", "No original data to reset.")
    def reset_data(self):
        if hasattr(self, "original_data") and self.original_data is not None:
            self.loader.data = self.original_data.copy()
            self._hilbert_display_mode = None
            # ---- RESET VIEW STATE ----
            self.view_max_depth = None
            self.view_x_start_m = 0.0
            self._sync_x_ch_vars()
            self.plot_gpr()
        else:
            messagebox.showinfo("Info", "No original data to reset.")
        
    def save_figure(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Nothing to save yet.")
            return

        filetypes = [
            ("PNG image", "*.png"),
            ("JPEG image", "*.jpg"),
            ("PDF document", "*.pdf"),
            ("SVG vector", "*.svg"),
            ("TIFF image", "*.tiff"),
            ("All files", "*.*")
        ]

        proc_tag = "-".join(self.loader.process_history) if self.loader.process_history else "RAW"
        default_name = f"{self.loader.line_name}_{proc_tag}_{self.cmap_var.get()}"

        # default_name = f"{self.loader.line_name}_{self.cmap_var.get()}"

        filepath = filedialog.asksaveasfilename(
            title="Save GPR Figure",
            defaultextension=".png",
            initialfile=default_name,
            initialdir=_OUTPUT_DIR,
            filetypes=filetypes
        )

        if not filepath:
            return

        try:
            self.figure.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight"
            )
            messagebox.showinfo("Saved", f"Figure saved successfully:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def zoom(self, scale_factor):
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_center = np.mean(xlim)
        y_center = np.mean(ylim)

        x_half = (xlim[1] - xlim[0]) * scale_factor / 2
        y_half = (ylim[1] - ylim[0]) * scale_factor / 2

        ax.set_xlim(x_center - x_half, x_center + x_half)
        ax.set_ylim(y_center - y_half, y_center + y_half)

        self.canvas.draw_idle()
    
 
    def change_colormap(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "No GPR data loaded yet.")
            return

        self.plot_gpr()
        

    def plot_wiggle(self):
        """
        Opens a dialog for wiggle plot parameters, then renders the wiggle display.
        RADAN7-style variable area wiggle plot with professional options.
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        n_samples, n_traces = self.loader.data.shape

        # Create parameter dialog
        win = tk.Toplevel(self.root)
        win.title("Wiggle Plot Settings")
        win.geometry("420x520")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="Wiggle Plot Settings", font=("Helvetica", 12, "bold")).pack(pady=10)

        frame = tk.Frame(win)
        frame.pack(padx=20, pady=10, fill="x")

        row = 0

        # Trace skip (decimation)
        tk.Label(frame, text="Trace skip (1=all, 2=every 2nd):").grid(row=row, column=0, sticky="e", pady=4)
        skip_var = tk.IntVar(value=max(1, n_traces // 200))  # Auto-adjust for large datasets
        tk.Spinbox(frame, from_=1, to=100, textvariable=skip_var, width=8).grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Amplitude scale
        tk.Label(frame, text="Amplitude scale:").grid(row=row, column=0, sticky="e", pady=4)
        amp_var = tk.DoubleVar(value=1.0)
        tk.Entry(frame, textvariable=amp_var, width=10).grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Clip factor (percentage)
        tk.Label(frame, text="Clip (% of max):").grid(row=row, column=0, sticky="e", pady=4)
        clip_var = tk.DoubleVar(value=100.0)
        tk.Entry(frame, textvariable=clip_var, width=10).grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Fill mode
        tk.Label(frame, text="Fill mode:").grid(row=row, column=0, sticky="e", pady=4)
        fill_var = tk.StringVar(value="positive")
        fill_menu = ttk.Combobox(frame, textvariable=fill_var, width=12, state="readonly")
        fill_menu['values'] = ("positive", "negative", "both", "none")
        fill_menu.grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Positive fill color
        tk.Label(frame, text="Positive fill color:").grid(row=row, column=0, sticky="e", pady=4)
        pos_color_var = tk.StringVar(value="black")
        pos_menu = ttk.Combobox(frame, textvariable=pos_color_var, width=12, state="readonly")
        pos_menu['values'] = ("black", "blue", "red", "gray", "darkblue")
        pos_menu.grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Negative fill color
        tk.Label(frame, text="Negative fill color:").grid(row=row, column=0, sticky="e", pady=4)
        neg_color_var = tk.StringVar(value="white")
        neg_menu = ttk.Combobox(frame, textvariable=neg_color_var, width=12, state="readonly")
        neg_menu['values'] = ("white", "red", "blue", "gray", "lightgray")
        neg_menu.grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Line width
        tk.Label(frame, text="Line width:").grid(row=row, column=0, sticky="e", pady=4)
        lw_var = tk.DoubleVar(value=0.5)
        tk.Entry(frame, textvariable=lw_var, width=10).grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Show wiggle lines
        tk.Label(frame, text="Show wiggle lines:").grid(row=row, column=0, sticky="e", pady=4)
        show_lines_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, variable=show_lines_var).grid(row=row, column=1, sticky="w", pady=4, padx=8)
        row += 1

        # Fill alpha
        tk.Label(frame, text="Fill opacity (0-1):").grid(row=row, column=0, sticky="e", pady=4)
        alpha_var = tk.DoubleVar(value=0.85)
        tk.Entry(frame, textvariable=alpha_var, width=10).grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Background color
        tk.Label(frame, text="Background:").grid(row=row, column=0, sticky="e", pady=4)
        bg_var = tk.StringVar(value="white")
        bg_menu = ttk.Combobox(frame, textvariable=bg_var, width=12, state="readonly")
        bg_menu['values'] = ("white", "lightgray", "beige", "black")
        bg_menu.grid(row=row, column=1, pady=4, padx=8)
        row += 1

        # Info label
        tk.Label(win, text=f"Total traces: {n_traces} | Samples: {n_samples}",
                 fg="gray", font=("Helvetica", 9)).pack(pady=5)

        def apply_wiggle():
            self._render_wiggle(
                trace_skip=skip_var.get(),
                amplitude_scale=amp_var.get(),
                clip_percent=clip_var.get(),
                fill_mode=fill_var.get(),
                pos_color=pos_color_var.get(),
                neg_color=neg_color_var.get(),
                line_width=lw_var.get(),
                show_lines=show_lines_var.get(),
                fill_alpha=alpha_var.get(),
                bg_color=bg_var.get()
            )
            win.destroy()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="Plot", bg="#4CAF50", fg="white", width=12,
                  command=apply_wiggle).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Cancel", width=10,
                  command=win.destroy).pack(side="left", padx=10)

    def _render_wiggle(self, trace_skip=1, amplitude_scale=1.0, clip_percent=100.0,
                       fill_mode="positive", pos_color="black", neg_color="white",
                       line_width=0.5, show_lines=True, fill_alpha=0.85, bg_color="white"):
        """
        Renders RADAN7-style variable area wiggle plot.
        
        Parameters:
        -----------
        trace_skip : int
            Plot every Nth trace (1=all traces, 2=every 2nd, etc.)
        amplitude_scale : float
            Scale factor for trace amplitude (1.0 = auto-fit)
        clip_percent : float
            Clip amplitudes at this percentage of max (100 = no clip)
        fill_mode : str
            'positive' - fill positive peaks (standard VA)
            'negative' - fill negative peaks
            'both' - fill both with different colors
            'none' - wiggle lines only, no fill
        pos_color : str
            Fill color for positive amplitudes
        neg_color : str
            Fill color for negative amplitudes
        line_width : float
            Wiggle trace line width
        show_lines : bool
            Whether to show wiggle trace lines
        fill_alpha : float
            Opacity of fill (0-1)
        bg_color : str
            Background color
        """
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(bg_color)

        data = self.loader.data
        if data is None:
            return

        n_samples, n_traces = data.shape

        # X coordinates (distance): follow GPR-GUI-2026 style.
        # - If GEOX / xyz exists, use its X column directly (absolute distance).
        # - Otherwise, use a simple synthetic axis based on X_CELL (or trace index).
        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            x_all = np.asarray(self.loader.xyz['X'].values, dtype=float)
            if len(x_all) != n_traces:
                x_all = np.linspace(float(x_all[0]), float(x_all[-1]), n_traces)
        else:
            dx = getattr(self.loader, "_x_cell_m", None)
            if dx is not None:
                x_all = np.arange(n_traces, dtype=float) * float(dx)
            else:
                x_all = np.arange(n_traces, dtype=float)

        # In folder/segment mode, rebase X so profile starts at 0 m
        if self.folder_segment_files and self._folder_x0_offset is not None:
            x_all = x_all - float(self._folder_x0_offset)

        total_m = float(x_all[-1] - x_all[0]) if n_traces > 1 else 0.0
        w_m = self._get_window_m()
        use_window = total_m > w_m and self.view_x_start_m is not None
        if use_window:
            x0 = float(x_all[0])
            start_m = self.view_x_start_m + x0
            end_m = start_m + w_m
            mask = (x_all >= start_m) & (x_all < end_m)
            if not np.any(mask):
                mask = (x_all >= start_m) & (x_all <= end_m)
            t_idx = np.where(mask)[0]
            if len(t_idx) == 0:
                t_idx = np.array([0, min(1, n_traces - 1)])
            t0, t1 = int(t_idx[0]), int(t_idx[-1]) + 1
            data = data[:, t0:t1]
            x_all = x_all[t0:t1]
            n_traces = data.shape[1]

        # Y coordinates (depth/time)
        if self.loader.depth is not None:
            y = self.loader.depth
            ylabel = "Depth (m)"
        else:
            y = np.arange(n_samples)
            ylabel = "Sample"

        # Select traces to plot (decimation)
        trace_indices = np.arange(0, n_traces, trace_skip)
        x = x_all[trace_indices]
        n_plot = len(trace_indices)

        # Calculate trace spacing for scaling
        if n_plot > 1:
            trace_spacing = np.median(np.diff(x))
        else:
            trace_spacing = 1.0

        # Normalize data
        data_subset = data[:, trace_indices].astype(float)
        
        # Global normalization for consistent amplitude display
        max_amp = np.percentile(np.abs(data_subset), 99)  # Use 99th percentile to avoid outliers
        if max_amp == 0:
            max_amp = 1.0

        # Apply clipping
        clip_level = (clip_percent / 100.0) * max_amp
        data_clipped = np.clip(data_subset, -clip_level, clip_level)

        # Scale factor: traces should span about 0.8 * trace_spacing
        scale = 0.8 * trace_spacing * amplitude_scale / clip_level

        # Plot each trace
        for i, trace_idx in enumerate(range(n_plot)):
            trace = data_clipped[:, trace_idx]
            x_pos = x[trace_idx]
            
            # Compute wiggle positions
            wiggle_x = x_pos + trace * scale

            # Draw fills first (behind lines)
            if fill_mode in ("positive", "both"):
                # Fill positive amplitudes
                ax.fill_betweenx(
                    y, x_pos, wiggle_x,
                    where=(trace > 0),
                    facecolor=pos_color,
                    alpha=fill_alpha,
                    linewidth=0
                )

            if fill_mode in ("negative", "both"):
                # Fill negative amplitudes
                ax.fill_betweenx(
                    y, x_pos, wiggle_x,
                    where=(trace < 0),
                    facecolor=neg_color,
                    alpha=fill_alpha,
                    linewidth=0
                )

            # Draw wiggle trace line
            if show_lines:
                line_color = "black" if bg_color != "black" else "white"
                ax.plot(wiggle_x, y, color=line_color, linewidth=line_width)

        # Configure axes
        ax.invert_yaxis()
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(ylabel)
        
        # Set proper axis limits
        x_margin = trace_spacing * 2
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.max(), y.min())  # Inverted

        # Title with info
        title = f"GPR Wiggle Plot - {self.loader.line_name}"
        if trace_skip > 1:
            title += f" (every {trace_skip} traces)"
        ax.set_title(title)

        # Grid styling
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', color='gray')
        ax.grid(False, axis='x')

        # Tight layout
        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.current_image = None

            
    def plot_geox(self):
        if self.loader.xyz is None or len(self.loader.xyz) == 0:
            messagebox.showinfo("Info", "No GEOX data available.")
            return

        self.figure.clf()
        axes = self.figure.subplots(2, 2)
        axes = axes.flat

        xyz = self.loader.xyz

        axes[0].plot(xyz['X'], xyz['Y'])
        axes[0].set_title("Survey Path (X-Y)")

        axes[1].plot(xyz['Marker'], xyz['Z'])
        axes[1].set_title("Elevation")

        axes[2].plot(xyz['Marker'], xyz['Lat'])
        axes[2].set_title("Latitude")

        axes[3].plot(xyz['Marker'], xyz['Lon'])
        axes[3].set_title("Longitude")

        self.figure.tight_layout()
        self.canvas.draw_idle()

 
    
    def show_stats(self):
        stats = self.loader.get_statistics()
        if not stats:
            messagebox.showinfo("Info", "No data loaded yet.")
            return
        
        win = tk.Toplevel(self.root)
        win.title("Data Statistics")
        win.geometry("600x700")
        
        text = scrolledtext.ScrolledText(win, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for k, v in stats.items():
            text.insert(tk.END, f"{k:24}: {v}\n")
            
    def on_scroll(self, event):
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.canvas.draw_idle()
        
    def clear_all(self):
        self.loader.clear_data()
        self.meta_text.delete("1.0", tk.END)

        self.view_x_start_m = 0.0
        self._total_length_m = None
        self.folder_segment_files = []
        self.current_segment_index = -1
        self.x_pos_var.set("0.0")
        self.segment_var.set("1")
        self._hilbert_display_mode = None
        self.figure.clf()
        self.current_image = None
        self.canvas.draw_idle()

        if self.ascan_cid is not None:
            self.canvas.mpl_disconnect(self.ascan_cid)
            self.ascan_cid = None

        self.ascan_enabled = False
        self.ascan_win.withdraw()

    # =========================================================================
    #                   DECLUTTERING METHODS (MS, SVD, RNMF)
    #       Based on: Ge et al. 2024, IEEE TGRS - Wavelet-GAN paper
    # =========================================================================

    # -------------------------------------------------------------------------
    # MEAN SUBTRACTION (MS) - Traditional GPR clutter removal
    # -------------------------------------------------------------------------
    def ms_declutter_dialog(self):
        """
        Mean Subtraction dialog for GPR clutter removal.
        Removes horizontal clutter (direct wave, ringing noise).
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Mean Subtraction (MS) Declutter")
        win.geometry("350x180")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text="Mean Subtraction (MS)",
            font=("Helvetica", 12, "bold")
        ).pack(pady=10)

        tk.Label(
            win,
            text="Subtracts the mean trace from all traces\nto remove horizontal clutter.",
            justify="center"
        ).pack(pady=5)

        def apply_ms():
            self._apply_mean_subtraction()
            win.destroy()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=15)

        tk.Button(
            btn_frame,
            text="Apply",
            bg="#4CAF50",
            fg="white",
            width=10,
            command=apply_ms
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            command=win.destroy
        ).pack(side="right", padx=10)

    def _apply_mean_subtraction(self):
        """
        Mean Subtraction (MS) for GPR clutter removal.
        Subtracts the mean trace from all traces to remove horizontal clutter
        (direct wave, ringing noise with strict horizontal distribution).
        Reference: Ge et al. 2024, Section I - Traditional methods
        """
        if self.loader.data is None:
            return

        # Work on a copy to preserve original data
        data = self.loader.data.astype(float).copy()

        # Compute mean trace across all traces (axis=1 for columns/traces)
        mean_trace = np.mean(data, axis=1, keepdims=True)

        # Subtract mean from all traces
        data = data - mean_trace

        self.loader.data = data
        self.loader.add_process("MS")
        self.plot_gpr()

    # -------------------------------------------------------------------------
    # SINGULAR VALUE DECOMPOSITION (SVD) - Subspace projection method
    # -------------------------------------------------------------------------
    def svd_declutter_dialog(self):
        """
        SVD-based declutter dialog with parameter selection.
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("SVD Declutter")
        win.geometry("380x280")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text="Singular Value Decomposition (SVD)",
            font=("Helvetica", 12, "bold")
        ).pack(pady=10)

        tk.Label(
            win,
            text="Removes first N singular value components\nwhich represent horizontal clutter.",
            justify="center"
        ).pack(pady=5)

        # Number of components to remove
        tk.Label(win, text="Components to remove:").pack(pady=(10, 2))
        n_remove_var = tk.IntVar(value=1)
        tk.Spinbox(
            win,
            from_=1,
            to=20,
            increment=1,
            textvariable=n_remove_var,
            width=8
        ).pack()

        tk.Label(
            win,
            text="(Start with 1-3 for direct wave removal)",
            fg="gray",
            font=("Helvetica", 9)
        ).pack(pady=2)

        def apply_svd():
            self._apply_svd_declutter(n_remove_var.get())
            win.destroy()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame,
            text="Apply",
            bg="#4CAF50",
            fg="white",
            width=10,
            command=apply_svd
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            command=win.destroy
        ).pack(side="right", padx=10)

    def _apply_svd_declutter(self, n_components_remove=1):
        """
        SVD-based clutter removal for GPR data.
        Removes the first N singular value components which typically
        represent horizontal clutter (direct wave, ringing noise).
        
        Based on the assumption that clutter signal is stronger than target
        reflection and exhibits low-rank horizontal distribution.
        Reference: Ge et al. 2024, Section I - Subspace projection methods
        """
        if self.loader.data is None:
            return

        # Work on a copy to preserve original data
        data = self.loader.data.astype(float).copy()

        # SVD decomposition: data = U @ diag(s) @ Vt
        U, s, Vt = np.linalg.svd(data, full_matrices=False)

        # Zero out first n components (these capture horizontal clutter)
        s_filtered = s.copy()
        s_filtered[:n_components_remove] = 0

        # Reconstruct the decluttered data
        data = U @ np.diag(s_filtered) @ Vt

        self.loader.data = data
        self.loader.add_process(f"SVD-{n_components_remove}")
        self.plot_gpr()

    # -------------------------------------------------------------------------
    # ROBUST NON-NEGATIVE MATRIX FACTORIZATION (RNMF)
    # -------------------------------------------------------------------------
    def rnmf_declutter_dialog(self):
        """
        RNMF-based declutter dialog with parameter selection.
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("RNMF Declutter")
        win.geometry("400x380")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text="Robust Non-negative Matrix Factorization",
            font=("Helvetica", 12, "bold")
        ).pack(pady=10)

        tk.Label(
            win,
            text="Separates low-rank clutter from sparse target responses.",
            justify="center"
        ).pack(pady=5)

        # Number of components (rank)
        tk.Label(win, text="Number of components (rank):").pack(pady=(10, 2))
        n_comp_var = tk.IntVar(value=5)
        tk.Spinbox(
            win,
            from_=2,
            to=30,
            increment=1,
            textvariable=n_comp_var,
            width=8
        ).pack()

        # Max iterations
        tk.Label(win, text="Max iterations:").pack(pady=(10, 2))
        max_iter_var = tk.IntVar(value=200)
        tk.Spinbox(
            win,
            from_=50,
            to=1000,
            increment=50,
            textvariable=max_iter_var,
            width=8
        ).pack()

        # Sparsity weight (alpha)
        tk.Label(win, text="Sparsity weight (alpha):").pack(pady=(10, 2))
        alpha_var = tk.DoubleVar(value=0.1)
        tk.Entry(win, textvariable=alpha_var, width=10).pack()

        tk.Label(
            win,
            text="Higher alpha = sparser target, more clutter removed",
            fg="gray",
            font=("Helvetica", 9)
        ).pack(pady=2)

        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            win,
            variable=progress_var,
            maximum=100,
            length=250,
            mode="determinate"
        ).pack(pady=10)

        def apply_rnmf():
            self._apply_rnmf_declutter(
                n_components=n_comp_var.get(),
                max_iter=max_iter_var.get(),
                alpha=alpha_var.get(),
                progress_var=progress_var,
                win=win
            )

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=15)

        tk.Button(
            btn_frame,
            text="Apply",
            bg="#4CAF50",
            fg="white",
            width=10,
            command=apply_rnmf
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            command=win.destroy
        ).pack(side="right", padx=10)

    def _apply_rnmf_declutter(self, n_components=5, max_iter=200, alpha=0.1,
                               progress_var=None, win=None):
        """
        Robust Non-negative Matrix Factorization (RNMF) for GPR clutter removal.
        
        Utilizes the low rank of GPR horizontal clutter signal and the sparsity
        of target response to separate clutter components from target signals.
        
        The method decomposes: X ≈ W @ H + S
        Where W @ H is the low-rank clutter and S is the sparse target response.
        
        Reference: Ge et al. 2024, Section I - Low-rank matrix decomposition methods
        """
        if self.loader.data is None:
            return

        from sklearn.decomposition import NMF

        # Work on a copy to preserve original data
        data = self.loader.data.astype(float).copy()

        # NMF requires non-negative data, so shift if necessary
        data_min = data.min()
        if data_min < 0:
            data_shifted = data - data_min
        else:
            data_shifted = data.copy()
            data_min = 0

        # Apply NMF decomposition
        # Using 'nndsvd' initialization for better convergence
        model = NMF(
            n_components=n_components,
            init='nndsvd',
            max_iter=max_iter,
            alpha_W=alpha,
            alpha_H=alpha,
            l1_ratio=0.5,  # Mix of L1 (sparsity) and L2 regularization
            random_state=42
        )

        try:
            if progress_var is not None:
                progress_var.set(20)
                if win is not None:
                    win.update_idletasks()

            W = model.fit_transform(data_shifted)

            if progress_var is not None:
                progress_var.set(60)
                if win is not None:
                    win.update_idletasks()

            H = model.components_

            # Reconstruct low-rank approximation (this represents the clutter)
            clutter = W @ H

            if progress_var is not None:
                progress_var.set(80)
                if win is not None:
                    win.update_idletasks()

            # Target signal = Original - Clutter (low-rank)
            target = data_shifted - clutter

            # Shift back to original range
            if data_min < 0:
                target = target + data_min

            self.loader.data = target
            self.loader.add_process(f"RNMF-{n_components}")

            if progress_var is not None:
                progress_var.set(100)
                if win is not None:
                    win.update_idletasks()
                    win.after(200, win.destroy)

            self.plot_gpr()

        except Exception as e:
            messagebox.showerror("RNMF Error", str(e))
            if win is not None:
                win.destroy()

    # =========================================================================
    #                   GPS LOCATION SEARCH FUNCTION
    #     Search GPR profiles by Lat/Long coordinates in a folder
    # =========================================================================

    def gps_location_search_dialog(self):
        """
        Opens a dialog to search GPR profiles by GPS coordinates.
        Searches GEOX/GEC files in a folder to find profiles containing
        or near the specified Lat/Long location.
        """
        win = tk.Toplevel(self.root)
        win.title("GPS Location Search")
        win.geometry("650x600")
        win.resizable(True, True)
        win.transient(self.root)

        # Title
        tk.Label(
            win,
            text="Search GPR Profiles by GPS Location",
            font=("Helvetica", 14, "bold")
        ).pack(pady=15)

        # Frame for inputs
        input_frame = tk.Frame(win)
        input_frame.pack(padx=20, pady=10, fill="x")

        # Folder selection
        tk.Label(input_frame, text="Data Folder:", font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=5
        )
        folder_var = tk.StringVar(value="")
        folder_entry = tk.Entry(input_frame, textvariable=folder_var, width=50)
        folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        def browse_folder():
            folder = filedialog.askdirectory(title="Select GPR Data Folder")
            if folder:
                folder_var.set(folder)

        tk.Button(input_frame, text="Browse...", command=browse_folder).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Latitude input
        tk.Label(input_frame, text="Latitude (WGS84):", font=("Helvetica", 10)).grid(
            row=1, column=0, sticky="w", pady=5
        )
        lat_var = tk.DoubleVar(value=0.0)
        tk.Entry(input_frame, textvariable=lat_var, width=20).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )

        # Longitude input
        tk.Label(input_frame, text="Longitude (WGS84):", font=("Helvetica", 10)).grid(
            row=2, column=0, sticky="w", pady=5
        )
        lon_var = tk.DoubleVar(value=0.0)
        tk.Entry(input_frame, textvariable=lon_var, width=20).grid(
            row=2, column=1, sticky="w", padx=5, pady=5
        )

        # Search radius
        tk.Label(input_frame, text="Search Radius (m):", font=("Helvetica", 10)).grid(
            row=3, column=0, sticky="w", pady=5
        )
        radius_var = tk.DoubleVar(value=50.0)
        tk.Entry(input_frame, textvariable=radius_var, width=20).grid(
            row=3, column=1, sticky="w", padx=5, pady=5
        )

        # Include subfolders
        subfolder_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            input_frame,
            text="Include subfolders",
            variable=subfolder_var
        ).grid(row=4, column=1, sticky="w", pady=5)

        input_frame.columnconfigure(1, weight=1)

        # Results area
        tk.Label(win, text="Search Results:", font=("Helvetica", 10, "bold")).pack(
            anchor="w", padx=20, pady=(15, 5)
        )

        results_frame = tk.Frame(win)
        results_frame.pack(padx=20, pady=5, fill="both", expand=True)

        results_text = scrolledtext.ScrolledText(
            results_frame,
            font=("Consolas", 10),
            width=70,
            height=15
        )
        results_text.pack(fill="both", expand=True)

        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress = ttk.Progressbar(
            win,
            variable=progress_var,
            maximum=100,
            length=400,
            mode="determinate"
        )
        progress.pack(pady=10)

        # Status label
        status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(win, textvariable=status_var, fg="gray")
        status_label.pack(pady=5)

        def run_search():
            folder = folder_var.get().strip()
            if not folder or not os.path.isdir(folder):
                messagebox.showwarning("Warning", "Please select a valid folder.")
                return

            lat = lat_var.get()
            lon = lon_var.get()
            radius = radius_var.get()
            include_subfolders = subfolder_var.get()

            if lat == 0.0 and lon == 0.0:
                messagebox.showwarning("Warning", "Please enter valid Lat/Long coordinates.")
                return

            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, f"TARGET: Lat {lat:.6f}, Lon {lon:.6f}\n")
            results_text.insert(tk.END, f"RADIUS: {radius:.1f} m\n")
            results_text.insert(tk.END, f"FOLDER: {folder}\n")
            results_text.insert(tk.END, "-" * 60 + "\n\n")
            win.update_idletasks()

            # Run search
            matches = self._search_profiles_by_gps(
                folder=folder,
                target_lat=lat,
                target_lon=lon,
                radius_m=radius,
                include_subfolders=include_subfolders,
                progress_callback=lambda p, s: self._update_search_progress(p, s, progress_var, status_var, win),
                results_callback=lambda msg: self._append_search_result(msg, results_text, win)
            )

            # Final summary
            results_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            if matches:
                results_text.insert(tk.END, f">>> FOUND {len(matches)} GPR FILE(S) <<<\n\n")
                # Sort by distance
                matches.sort(key=lambda x: x['distance'])
                for m in matches:
                    subfolder = m.get('subfolder', '')
                    if subfolder:
                        results_text.insert(tk.END, f"  [{subfolder}] {m['filename']}\n")
                    else:
                        results_text.insert(tk.END, f"  {m['filename']}\n")
                    results_text.insert(tk.END, f"    Full Path: {m['file']}\n")
                    results_text.insert(tk.END, f"    Distance: {m['distance']:.1f} m\n")
                    results_text.insert(tk.END, f"    GPS Coord: {m['lat']:.6f}, {m['lon']:.6f}\n\n")
            else:
                results_text.insert(tk.END, ">>> NOT FOUND <<<\n")
                results_text.insert(tk.END, "No GPR profiles found within the search radius.\n")
                results_text.insert(tk.END, "Try increasing the search radius.\n")

            results_text.see(tk.END)
            status_var.set("Search complete - Change folder and click Search again for new search")
            progress_var.set(100)

        def clear_results():
            """Clear results and reset for new search."""
            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, "Results cleared. Ready for new search.\n")
            progress_var.set(0)
            status_var.set("Ready")

        def change_folder():
            """Browse for a new folder."""
            new_folder = filedialog.askdirectory(title="Select New GPR Data Folder")
            if new_folder:
                folder_var.set(new_folder)
                results_text.delete("1.0", tk.END)
                results_text.insert(tk.END, f"Folder changed to:\n{new_folder}\n\n")
                results_text.insert(tk.END, "Click 'Search' to search in new folder.\n")
                progress_var.set(0)
                status_var.set("New folder selected - Click Search")

        # Buttons - Top row (main actions)
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Search",
            bg="#4CAF50",
            fg="white",
            width=12,
            font=("Helvetica", 10, "bold"),
            command=run_search
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="New Folder",
            bg="#2196F3",
            fg="white",
            width=12,
            command=change_folder
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="Clear",
            bg="#FF9800",
            fg="white",
            width=10,
            command=clear_results
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="Close",
            width=10,
            command=win.destroy
        ).pack(side="left", padx=8)

    def _update_search_progress(self, percent, status, progress_var, status_var, win):
        """Update progress bar and status during search."""
        progress_var.set(percent)
        status_var.set(status)
        win.update_idletasks()

    def _append_search_result(self, msg, results_text, win):
        """Append message to results text widget."""
        results_text.insert(tk.END, msg)
        results_text.see(tk.END)
        win.update_idletasks()

    def _search_profiles_by_gps(self, folder, target_lat, target_lon, radius_m=50.0,
                                 include_subfolders=True, progress_callback=None,
                                 results_callback=None):
        """
        FAST GPS coordinate search for GPR profiles.
        Scans ALL subfolders for GPS files, then reports matching .v00/.d00/.dt files.
        Handles duplicate filenames in different subfolders.
        """
        matches = []
        
        # All supported GPS file extensions (prioritized)
        geo_extensions = {'.gec', '.GEC', '.geox', '.GEOX', '.geo', '.GEO', 
                          '.gps', '.GPS', '.gpx', '.GPX', '.csv', '.CSV'}
        
        # GPR data extensions to look for
        gpr_extensions = ['.v00', '.V00', '.d00', '.D00', '.dt', '.DT']
        
        if progress_callback:
            progress_callback(2, "Scanning folders...")
        
        # Build list of ALL GPS files (allowing duplicates in different folders)
        geo_files = []  # List of {geo_path, folder, line_name}
        folder_count = 0
        
        def scan_folder(path, depth=0):
            """Fast recursive folder scan using scandir"""
            nonlocal folder_count
            try:
                folder_count += 1
                with os.scandir(path) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                ext = os.path.splitext(entry.name)[1]
                                if ext in geo_extensions:
                                    line_name = os.path.splitext(entry.name)[0]
                                    geo_files.append({
                                        'geo_path': entry.path,
                                        'folder': path,
                                        'line_name': line_name,
                                        'ext': ext
                                    })
                            elif entry.is_dir(follow_symlinks=False) and include_subfolders:
                                scan_folder(entry.path, depth + 1)
                        except (PermissionError, OSError):
                            continue
            except (PermissionError, OSError):
                pass
        
        if results_callback:
            results_callback("Scanning folders for GPS files...\n")
        
        scan_folder(folder)
        
        if results_callback:
            results_callback(f"Scanned {folder_count} folder(s)\n")
        
        if not geo_files:
            if results_callback:
                results_callback("\nNo GPS files found (.gec, .geox, .geo, .gps, .gpx, .csv)\n")
            return matches
        
        if results_callback:
            results_callback(f"Found {len(geo_files)} GPS file(s). Checking coordinates...\n")
            results_callback("-" * 50 + "\n\n")
        
        if progress_callback:
            progress_callback(10, f"Checking {len(geo_files)} GPS files...")
        
        # Process each GPS file
        total = len(geo_files)
        files_with_gpr = 0
        files_checked = 0
        
        for idx, info in enumerate(geo_files):
            files_checked += 1
            
            # Update progress every 5 files or every 2%
            if progress_callback and (idx % max(1, total // 50) == 0):
                percent = 10 + (idx / total) * 85
                progress_callback(percent, f"Checking {idx+1}/{total}...")
            
            geo_path = info['geo_path']
            base_folder = info['folder']
            line_name = info['line_name']
            
            # First check if GPR file exists (fast check)
            gpr_file = None
            for ext in gpr_extensions:
                candidate = os.path.join(base_folder, line_name + ext)
                if os.path.exists(candidate):
                    gpr_file = candidate
                    break
            
            # Skip if no GPR file
            if gpr_file is None:
                continue
            
            files_with_gpr += 1
            
            # Load and check GPS coordinates
            try:
                lats, lons = self._fast_load_gps_coords(geo_path)
                
                if lats is None or len(lats) == 0:
                    continue
                
                # Vectorized distance calculation
                min_dist, closest_idx = self._fast_min_distance(
                    target_lat, target_lon, lats, lons
                )
                
                gpr_filename = os.path.basename(gpr_file)
                
                # Get relative subfolder path for display
                rel_folder = os.path.relpath(base_folder, folder)
                if rel_folder == '.':
                    rel_folder = ''
                
                if min_dist <= radius_m:
                    matches.append({
                        'file': gpr_file,
                        'filename': gpr_filename,
                        'geo_file': geo_path,
                        'distance': min_dist,
                        'lat': lats[closest_idx],
                        'lon': lons[closest_idx],
                        'line_name': line_name,
                        'subfolder': rel_folder
                    })
                    if results_callback:
                        if rel_folder:
                            results_callback(f"  ** MATCH: {rel_folder}/{gpr_filename}\n")
                        else:
                            results_callback(f"  ** MATCH: {gpr_filename}\n")
                        results_callback(f"     Distance: {min_dist:.1f} m\n")
                        results_callback(f"     Coords: {lats[closest_idx]:.6f}, {lons[closest_idx]:.6f}\n\n")
                        
            except Exception as e:
                continue
        
        if progress_callback:
            progress_callback(100, "Search complete")
        
        if results_callback:
            results_callback(f"\nChecked {files_checked} GPS files, {files_with_gpr} had matching GPR data\n")
        
        return matches

    def _fast_load_gps_coords(self, file_path):
        """
        Fast GPS coordinate loader.
        Returns (lats, lons) numpy arrays.
        
        Supported formats:
        - .gps: NMEA $GPGGA format (DDMM.MMMM)
        - .csv: Decimal degrees at columns 1,2
        - .geox/.gec: Lat at col 4, Lon at col 5 (if not zero)
        """
        ext = os.path.splitext(file_path)[1].lower()
        lats = []
        lons = []
        
        def nmea_to_decimal(coord_str, direction):
            """Convert NMEA DDMM.MMMM to decimal degrees"""
            try:
                if not coord_str:
                    return None
                dot_idx = coord_str.index('.')
                degrees = float(coord_str[:dot_idx-2])
                minutes = float(coord_str[dot_idx-2:])
                decimal = degrees + minutes / 60.0
                if direction in ('S', 'W'):
                    decimal = -decimal
                return decimal
            except:
                return None
        
        try:
            with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
                content = f.read()
            
            lines = content.split('\n')
            
            if ext == '.gps':
                # NMEA $GPGGA format: $GPGGA,time,lat,N/S,lon,E/W,...
                for ln in lines:
                    ln = ln.strip()
                    if ln.startswith('$GPGGA') or ln.startswith('$GNGGA'):
                        parts = ln.split(',')
                        if len(parts) >= 6:
                            lat = nmea_to_decimal(parts[2], parts[3])
                            lon = nmea_to_decimal(parts[4], parts[5])
                            if lat and lon and abs(lat) > 1 and abs(lon) > 1:
                                lats.append(lat)
                                lons.append(lon)
            
            elif ext == '.csv':
                # CSV format: scan, lat, lon, alt, ...
                # Lat at index 1, Lon at index 2
                for ln in lines:
                    ln = ln.strip()
                    if not ln or ln.startswith('#'):
                        continue
                    
                    parts = ln.split(',')
                    if len(parts) >= 3:
                        try:
                            lat = float(parts[1].strip())
                            lon = float(parts[2].strip())
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                if abs(lat) > 1.0 and abs(lon) > 1.0:
                                    lats.append(lat)
                                    lons.append(lon)
                        except (ValueError, IndexError):
                            continue
            
            elif ext in ('.gec', '.geox'):
                # GEOX format: Marker, X, Y, Z, Lat, Lon, Alt, Time
                for ln in lines:
                    ln = ln.strip()
                    if not ln or ln.startswith('<') or ln.isdigit():
                        continue
                    
                    parts = ln.split(',')
                    if len(parts) >= 6:
                        try:
                            lat = float(parts[4].strip())
                            lon = float(parts[5].strip())
                            # Skip zero coordinates (no GPS data)
                            if abs(lat) > 1.0 and abs(lon) > 1.0:
                                if -90 <= lat <= 90 and -180 <= lon <= 180:
                                    lats.append(lat)
                                    lons.append(lon)
                        except (ValueError, IndexError):
                            continue
            
            elif ext == '.geo':
                # Try multiple formats
                for ln in lines:
                    ln = ln.strip()
                    if not ln or ln.startswith('<') or ln.startswith('#'):
                        continue
                    
                    parts = ln.replace('\t', ',').replace(';', ',').split(',')
                    for lat_idx, lon_idx in [(1, 2), (4, 5), (0, 1)]:
                        if len(parts) > max(lat_idx, lon_idx):
                            try:
                                lat = float(parts[lat_idx].strip())
                                lon = float(parts[lon_idx].strip())
                                if abs(lat) > 1.0 and abs(lon) > 1.0:
                                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                                        lats.append(lat)
                                        lons.append(lon)
                                        break
                            except:
                                continue
                                
            elif ext == '.gpx':
                # GPX XML format - simple regex extraction
                import re
                lat_pattern = re.compile(r'lat="([+-]?\d+\.?\d*)"')
                lon_pattern = re.compile(r'lon="([+-]?\d+\.?\d*)"')
                lat_matches = lat_pattern.findall(content)
                lon_matches = lon_pattern.findall(content)
                for lat_s, lon_s in zip(lat_matches, lon_matches):
                    try:
                        lat = float(lat_s)
                        lon = float(lon_s)
                        if abs(lat) > 1.0 and abs(lon) > 1.0:
                            lats.append(lat)
                            lons.append(lon)
                    except ValueError:
                        continue
                        
            elif ext == '.csv':
                # CSV - try to find lat/lon columns
                header_found = False
                lat_col = lon_col = -1
                for ln in lines:
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split(',')
                    
                    # Try to identify header
                    if not header_found:
                        lower_parts = [p.lower().strip() for p in parts]
                        for i, p in enumerate(lower_parts):
                            if 'lat' in p:
                                lat_col = i
                            if 'lon' in p or 'lng' in p:
                                lon_col = i
                        if lat_col >= 0 and lon_col >= 0:
                            header_found = True
                            continue
                        # No header - assume format like GEOX
                        if len(parts) >= 6:
                            lat_col, lon_col = 4, 5
                            header_found = True
                    
                    if header_found and lat_col >= 0 and lon_col >= 0:
                        if len(parts) > max(lat_col, lon_col):
                            try:
                                lat = float(parts[lat_col])
                                lon = float(parts[lon_col])
                                if abs(lat) > 1.0 and abs(lon) > 1.0:
                                    lats.append(lat)
                                    lons.append(lon)
                            except ValueError:
                                continue
                                
        except Exception:
            return None, None
        
        if lats:
            return np.array(lats), np.array(lons)
        return None, None

    def _fast_min_distance(self, lat1, lon1, lats, lons):
        """
        Fast vectorized minimum distance calculation using Haversine.
        Returns (min_distance_meters, closest_index)
        """
        R = 6371000  # Earth radius in meters
        
        # Convert to radians (vectorized)
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        
        # Haversine formula (vectorized)
        dlat = lats_rad - lat1_rad
        dlon = lons_rad - lon1_rad
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c
        
        min_idx = np.argmin(distances)
        return distances[min_idx], min_idx

    def _load_geo_file_for_search(self, file_path):
        """Legacy function - kept for compatibility."""
        lats, lons = self._fast_load_gps_coords(file_path)
        if lats is not None:
            return pd.DataFrame({'Lat': lats, 'Lon': lons})
        return None

    # =========================================================================
    #           PLOT ALL PROFILES FROM MULTIPLE FOLDERS
    #   Plot all GeoX/GEC profiles on a single map with Google Earth imagery
    # =========================================================================

    def plot_all_profiles_dialog(self):
        """
        Dialog to plot all GPR profiles from multiple folders on a single map.
        Supports up to 4 folders with different colors.
        """
        win = tk.Toplevel(self.root)
        win.title("Plot All Profiles in Folders")
        win.geometry("700x650")
        win.resizable(True, True)
        win.transient(self.root)

        # Title
        tk.Label(
            win,
            text="Plot All GPR Profiles on Map",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)

        # Folder selection frame
        folders_frame = tk.LabelFrame(win, text="Data Folders (up to 4)", padx=10, pady=10)
        folders_frame.pack(padx=20, pady=10, fill="x")

        # Colors for each folder
        folder_colors = ["yellow", "cyan", "magenta", "lime"]
        color_options = ["yellow", "cyan", "magenta", "lime", "red", "blue", "orange", "white", "green"]
        
        folder_vars = []
        color_vars = []

        for i in range(4):
            row_frame = tk.Frame(folders_frame)
            row_frame.pack(fill="x", pady=3)

            label_text = "Main Folder:" if i == 0 else f"Folder {i+1} (optional):"
            tk.Label(row_frame, text=label_text, width=18, anchor="e").pack(side="left")

            folder_var = tk.StringVar(value="")
            folder_vars.append(folder_var)
            
            entry = tk.Entry(row_frame, textvariable=folder_var, width=45)
            entry.pack(side="left", padx=5)

            def make_browse(var=folder_var):
                def browse():
                    path = filedialog.askdirectory(title="Select GPR Data Folder")
                    if path:
                        var.set(path)
                return browse

            tk.Button(row_frame, text="Browse", command=make_browse()).pack(side="left", padx=2)

            # Color selector
            color_var = tk.StringVar(value=folder_colors[i])
            color_vars.append(color_var)
            
            tk.Label(row_frame, text="Color:").pack(side="left", padx=(10, 2))
            color_menu = ttk.Combobox(row_frame, textvariable=color_var, width=8, state="readonly")
            color_menu['values'] = color_options
            color_menu.pack(side="left")

        # Marker point frame
        marker_frame = tk.LabelFrame(win, text="Optional: Add Marker Point", padx=10, pady=10)
        marker_frame.pack(padx=20, pady=10, fill="x")

        marker_row = tk.Frame(marker_frame)
        marker_row.pack(fill="x")

        tk.Label(marker_row, text="Latitude:").pack(side="left", padx=5)
        marker_lat_var = tk.StringVar(value="")
        tk.Entry(marker_row, textvariable=marker_lat_var, width=15).pack(side="left", padx=5)

        tk.Label(marker_row, text="Longitude:").pack(side="left", padx=5)
        marker_lon_var = tk.StringVar(value="")
        tk.Entry(marker_row, textvariable=marker_lon_var, width=15).pack(side="left", padx=5)

        tk.Label(marker_row, text="Label:").pack(side="left", padx=5)
        marker_label_var = tk.StringVar(value="Target")
        tk.Entry(marker_row, textvariable=marker_label_var, width=12).pack(side="left", padx=5)

        # Options frame
        options_frame = tk.LabelFrame(win, text="Options", padx=10, pady=10)
        options_frame.pack(padx=20, pady=10, fill="x")

        opt_row = tk.Frame(options_frame)
        opt_row.pack(fill="x")

        include_subfolders_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_row, text="Include subfolders", variable=include_subfolders_var).pack(side="left", padx=10)

        show_labels_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_row, text="Show profile labels", variable=show_labels_var).pack(side="left", padx=10)

        line_weight_var = tk.IntVar(value=3)
        tk.Label(opt_row, text="Line weight:").pack(side="left", padx=(20, 5))
        tk.Spinbox(opt_row, from_=1, to=10, textvariable=line_weight_var, width=5).pack(side="left")

        # Status
        status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(win, textvariable=status_var, fg="gray")
        status_label.pack(pady=5)

        # Progress
        progress_var = tk.DoubleVar(value=0.0)
        progress = ttk.Progressbar(win, variable=progress_var, maximum=100, length=400)
        progress.pack(pady=5)

        # Results text
        results_frame = tk.Frame(win)
        results_frame.pack(padx=20, pady=5, fill="both", expand=True)

        results_text = scrolledtext.ScrolledText(results_frame, font=("Consolas", 9), height=8)
        results_text.pack(fill="both", expand=True)

        def plot_profiles():
            # Collect folders
            folders = []
            colors = []
            for i, (fvar, cvar) in enumerate(zip(folder_vars, color_vars)):
                path = fvar.get().strip()
                if path and os.path.isdir(path):
                    folders.append(path)
                    colors.append(cvar.get())

            if not folders:
                messagebox.showwarning("Warning", "Please select at least one folder.")
                return

            # Get marker point
            marker_point = None
            try:
                lat_str = marker_lat_var.get().strip()
                lon_str = marker_lon_var.get().strip()
                if lat_str and lon_str:
                    marker_point = {
                        'lat': float(lat_str),
                        'lon': float(lon_str),
                        'label': marker_label_var.get() or "Target"
                    }
            except ValueError:
                pass

            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, f"Scanning {len(folders)} folder(s)...\n")
            results_text.insert(tk.END, "=" * 50 + "\n\n")
            win.update_idletasks()

            # Plot
            self._plot_all_profiles_on_map(
                folders=folders,
                colors=colors,
                marker_point=marker_point,
                include_subfolders=include_subfolders_var.get(),
                show_labels=show_labels_var.get(),
                line_weight=line_weight_var.get(),
                status_var=status_var,
                progress_var=progress_var,
                results_text=results_text,
                win=win
            )
            
            status_var.set("Done - Change folders and click Plot Map for new search")

        def clear_folders():
            """Clear all folder selections."""
            for fvar in folder_vars:
                fvar.set("")
            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, "Folders cleared. Select new folders and click Plot Map.\n")
            progress_var.set(0)
            status_var.set("Ready - Select folders")

        def clear_results():
            """Clear results only, keep folders."""
            results_text.delete("1.0", tk.END)
            results_text.insert(tk.END, "Results cleared. Click Plot Map to scan again.\n")
            progress_var.set(0)
            status_var.set("Ready")

        # Buttons row 1 - main actions
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame,
            text="Plot Map",
            bg="#4CAF50",
            fg="white",
            width=12,
            font=("Helvetica", 10, "bold"),
            command=plot_profiles
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="Clear Folders",
            bg="#2196F3",
            fg="white",
            width=12,
            command=clear_folders
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="Clear Results",
            bg="#FF9800",
            fg="white",
            width=12,
            command=clear_results
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="Close",
            width=10,
            command=win.destroy
        ).pack(side="left", padx=8)

    def _plot_all_profiles_on_map(self, folders, colors, marker_point=None,
                                   include_subfolders=True, show_labels=False,
                                   line_weight=3, status_var=None, progress_var=None,
                                   results_text=None, win=None):
        """
        Plot all GPR profiles from multiple folders on a single map.
        Uses matplotlib + contextily for popup window display .
        """
        all_profiles = []
        all_lats = []
        all_lons = []

        # Scan each folder
        for folder_idx, (folder, color) in enumerate(zip(folders, colors)):
            if status_var:
                status_var.set(f"Scanning folder {folder_idx + 1}/{len(folders)}...")
            if win:
                win.update_idletasks()

            folder_name = os.path.basename(folder)
            profile_count = 0
            subfolder_count = 0

            if results_text:
                results_text.insert(tk.END, f"[Folder {folder_idx + 1}] {folder_name} ({color})\n")
                results_text.insert(tk.END, f"  Path: {folder}\n")
                results_text.see(tk.END)
                if win:
                    win.update_idletasks()

            files_checked = 0
            geo_files_found = 0
            
            # Priority: .gps and .csv have real GPS; .geox often has 0,0
            # So we collect by base name and prefer .gps/.csv over .geox
            geo_exts = ('.gec', '.geox', '.geo', '.gps', '.gpx', '.csv')
            priority_order = ('.gps', '.csv', '.gpx', '.geox', '.gec', '.geo')  # prefer these first
            
            def scan_folder(path, depth=0):
                nonlocal profile_count, subfolder_count, files_checked, geo_files_found
                try:
                    with os.scandir(path) as entries:
                        all_entries = list(entries)
                    
                    geo_exts = ('.gec', '.geox', '.geo', '.gps', '.gpx', '.csv')
                    priority_order = ('.gps', '.csv', '.gpx', '.geox', '.gec', '.geo')
                    by_basename = {}
                    for entry in all_entries:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                files_checked += 1
                                ext = os.path.splitext(entry.name)[1].lower()
                                if ext in geo_exts:
                                    geo_files_found += 1
                                    base = os.path.splitext(entry.name)[0]
                                    if base not in by_basename:
                                        by_basename[base] = []
                                    by_basename[base].append((entry.path, ext))
                        except (PermissionError, OSError):
                            continue
                    
                    for base, file_list in by_basename.items():
                        file_list_sorted = sorted(file_list, key=lambda x: (priority_order.index(x[1]) if x[1] in priority_order else 999))
                        lats, lons = None, None
                        used_path = None
                        for entry_path, ext in file_list_sorted:
                            lats, lons = self._fast_load_gps_coords(entry_path)
                            if lats is not None and len(lats) > 1:
                                used_path = entry_path
                                break
                        
                        if lats is not None and len(lats) > 1:
                            lats_arr = np.array(lats)
                            lons_arr = np.array(lons)
                            lon_range = np.max(lons_arr) - np.min(lons_arr)
                            lat_range = np.max(lats_arr) - np.min(lats_arr)
                            if lon_range < 0.0001 and lat_range > 0.001:
                                lats, lons = lons_arr.tolist(), lats_arr.tolist()
                                lats_arr, lons_arr = np.array(lats), np.array(lons)
                            if np.max(lons_arr) - np.min(lons_arr) < 0.0001:
                                if results_text:
                                    rel_path = os.path.relpath(path, folder)
                                    display_path = base if rel_path == '.' else f"{rel_path}/{base}"
                                    results_text.insert(tk.END, f"    - {display_path} (constant lon, skipped)\n")
                                    results_text.see(tk.END)
                                    win.update()
                                continue
                            
                            rel_path = os.path.relpath(path, folder)
                            display_path = base if rel_path == '.' else f"{rel_path}/{base}"
                            all_profiles.append({
                                'name': base,
                                'lats': lats_arr.tolist(),
                                'lons': lons_arr.tolist(),
                                'color': color,
                                'folder': folder_name
                            })
                            all_lats.extend(lats_arr.tolist())
                            all_lons.extend(lons_arr.tolist())
                            profile_count += 1
                            if results_text:
                                src = os.path.basename(used_path or '') if used_path else ''
                                results_text.insert(tk.END, f"    + {display_path} ({len(lats)} pts) [{src}]\n")
                                results_text.see(tk.END)
                                win.update()
                        else:
                            if results_text and file_list:
                                rel_path = os.path.relpath(path, folder)
                                display_path = base if rel_path == '.' else f"{rel_path}/{base}"
                                results_text.insert(tk.END, f"    - {display_path} (no valid coords)\n")
                                results_text.see(tk.END)
                                win.update()
                    
                    for entry in all_entries:
                        try:
                            if entry.is_dir(follow_symlinks=False) and include_subfolders:
                                subfolder_count += 1
                                if results_text:
                                    rel_sub = os.path.relpath(entry.path, folder)
                                    results_text.insert(tk.END, f"  >> Scanning: {rel_sub}/\n")
                                    results_text.see(tk.END)
                                    win.update()
                                scan_folder(entry.path, depth + 1)
                        except (PermissionError, OSError):
                            continue
                except (PermissionError, OSError):
                    pass

            scan_folder(folder)

            if results_text:
                results_text.insert(tk.END, f"  >> Found {profile_count} profiles with valid GPS\n\n")
                results_text.see(tk.END)
                if win:
                    win.update()

            if progress_var:
                progress_var.set((folder_idx + 1) / len(folders) * 50)

        if not all_profiles:
            if results_text:
                results_text.insert(tk.END, "\nNo GPS profiles found.\n")
            if status_var:
                status_var.set("No profiles found")
            return

        if results_text:
            results_text.insert(tk.END, f"\n{'='*50}\n")
            results_text.insert(tk.END, f"Total: {len(all_profiles)} profiles\n")
            
            # Show coordinate sample for debugging
            if all_profiles:
                sample = all_profiles[0]
                sample_lats = sample['lats'][:3] if len(sample['lats']) >= 3 else sample['lats']
                sample_lons = sample['lons'][:3] if len(sample['lons']) >= 3 else sample['lons']
                results_text.insert(tk.END, f"\nSample coords from '{sample['name']}':\n")
                for i, (lat, lon) in enumerate(zip(sample_lats, sample_lons)):
                    results_text.insert(tk.END, f"  Point {i+1}: Lat={lat:.6f}, Lon={lon:.6f}\n")
                
                # Calculate bounds
                all_lat_vals = [lat for p in all_profiles for lat in p['lats']]
                all_lon_vals = [lon for p in all_profiles for lon in p['lons']]
                results_text.insert(tk.END, f"\nCoord ranges:\n")
                results_text.insert(tk.END, f"  Lat: {min(all_lat_vals):.4f} to {max(all_lat_vals):.4f}\n")
                results_text.insert(tk.END, f"  Lon: {min(all_lon_vals):.4f} to {max(all_lon_vals):.4f}\n")
            
            results_text.insert(tk.END, "\nCreating map popup window...\n")
            results_text.see(tk.END)
            if win:
                win.update()

        if status_var:
            status_var.set("Creating map...")

        try:
            if results_text:
                results_text.insert(tk.END, "Creating map popup...\n")
                results_text.see(tk.END)
                if win:
                    win.update()

            # Collect valid coordinates for map bounds
            valid_profiles = []
            all_valid_lats = []
            all_valid_lons = []

            for profile in all_profiles:
                lats = np.array(profile['lats'])
                lons = np.array(profile['lons'])

                # Filter valid WGS84 coordinates
                valid_mask = (
                    np.isfinite(lats) & np.isfinite(lons) &
                    (np.abs(lats) > 1) & (np.abs(lons) > 1) &
                    (np.abs(lats) <= 90) & (np.abs(lons) <= 180)
                )

                if np.sum(valid_mask) >= 2:
                    filtered_lats = lats[valid_mask].tolist()
                    filtered_lons = lons[valid_mask].tolist()
                    valid_profiles.append({
                        'name': profile['name'],
                        'lats': filtered_lats,
                        'lons': filtered_lons,
                        'color': profile['color'],
                        'folder': profile['folder']
                    })
                    all_valid_lats.extend(filtered_lats)
                    all_valid_lons.extend(filtered_lons)

            if not valid_profiles:
                if results_text:
                    results_text.insert(tk.END, "\nERROR: No valid coordinates found!\n")
                    results_text.see(tk.END)
                messagebox.showerror("Error", "No valid GPS coordinates found.")
                return

            if results_text:
                results_text.insert(tk.END, f"Found {len(valid_profiles)} profiles with valid coordinates.\n")
                results_text.see(tk.END)
                if win:
                    win.update()

            if progress_var:
                progress_var.set(70)

            # Build tracks list for _open_map_popup
            tracks = []
            for profile in valid_profiles:
                tracks.append((
                    profile['lats'],
                    profile['lons'],
                    profile['color'],
                    f"{profile['name']} ({profile['folder']})"
                ))

            # Build markers
            mk = None
            if marker_point:
                mk = [(marker_point['lat'], marker_point['lon'], marker_point['label'])]

            _open_map_popup(
                f"All GPR Profiles — {len(valid_profiles)} profiles",
                tracks, markers=mk, zoom=15, root=self.root
            )

            if progress_var:
                progress_var.set(100)
            if status_var:
                status_var.set("Map displayed!")

            if results_text:
                results_text.insert(tk.END, "\n*** MAP OPENED ***\n")
                results_text.see(tk.END)

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            if results_text:
                results_text.insert(tk.END, f"\n!!! ERROR !!!\n{error_msg}\n")
                results_text.see(tk.END)
            if status_var:
                status_var.set("Error creating map")
            messagebox.showerror("Map Error", str(e))

  
if __name__ == "__main__":
    # Windows taskbar: set AppUserModelID so the app gets its own icon
    # (without this, Python apps share python.exe's taskbar icon)
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mkl.mklgpr.v5")
    except Exception:
        pass

    root = tk.Tk()
    app = V00ReaderGUI(root)
    root.mainloop()