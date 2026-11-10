"""
FPV Drone Variable-Fidelity Aerodynamic Optimization
=====================================================

BASELINE CODE
-------------------------------------
* FFT spectral operators  (make_k_vectors, spectral_gradients,
  spectral_laplacian, spectral_divergence)  -> universal, fidelity-agnostic.
* Error-handling sentinel + decorator pattern.
* Colormap registry + field visualizer scaffold.
* Streamlit UI architecture (sidebar config -> tabs -> results).
* Data exporters (pickle / json / csv / hdf5).
* Parameter-validation pattern.

MODIFICATIONS AND ENHANCEMENTS
------------------------------
* NaSSBGeometry (Voronoi grains)    -> FPVDroneGeometry (frame / arms / props)
* MaterialProperties (battery mats) -> DroneGeometryProperties (FPV configs)
* SpectralElasticSolver (Moulinec-Suquet) -> SpectralVLMSolver (horseshoe-vortex
  VLM with FFT-accelerated Biot-Savart for the induced-velocity matrix).
* Phase-field + Butler-Volmer + plasticity step -> aerodynamic force step
  (lift / drag / induced drag / L/D).

WHAT IS NEW (the variable-fidelity optimization layer)
------------------------------------------------------
* MidFidelityCorrector : Lamb-Oseen vortex core decay + flat-plate viscous
  drag build-up layered on top of the inviscid VLM.
* CokrigingSurrogate   : multi-fidelity Gaussian process ( Kennedy & O'Hagan
  2000 form  f_h(x) = rho * f_l(x) + delta(x) ).  Uses SMT if available,
  otherwise falls back to a numpy-only implementation.
* VariableFidelityOptimizer : Bayesian-optimization-style outer loop that
  uses the cheap low-fidelity model to propose candidates and the surrogate
  to decide where to spend the next high-fidelity evaluation.

WHAT WAS DROPPED (infeasible per the discussion)
-----------------------------------------------
* Phase-field evolution  (no Allen-Cahn, no double-well).
* Butler-Volmer electrochemistry.
* Spectral elastoplasticity / unified viscoplasticity.
* Temperature Arrhenius corrections (irrelevant to aero).
* LES / DDES high-fidelity tier (kept as an external-solver HOOK only).
* Flight-test validation tier (kept as an external-solver HOOK only).

GEOMETRY
--------
An FPV drone is parameterised as:
    - N arms (typically 4 for a quad, 6 for a hex)
    - Each arm tip carries a propeller modelled as a lifting-line actuator
      disk discretised into M horseshoe vortices (-> VLM).
    - Frame body (cylinder / box) contributes a flat-plate drag area.
    - Battery pack at the centre of mass (drag + weight).
    - Camera / GoPro at the front (drag only).

The optimisation design variables are:
    x = [ arm_length, prop_diameter, prop_pitch, blade_count,
          cruise_AOA, cruise_speed, prop_rpm ]

with bound constraints from the chosen FPV class (5", 7", 10", custom).

Author: AI assistant (adaptation).
"""

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from datetime import datetime
import json, pickle, hashlib, traceback, warnings
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple, Any
import logging

# Optional multi-fidelity surrogate backend
try:
    from smt.surrogate_models import MFK, MGP
    SMT_AVAILABLE = True
except ImportError:
    SMT_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional 3D mesh generation backend (Gmsh OCC API + PyVista for viz).
# When unavailable the 3D CAD & Mesh tab shows a friendly install hint
# instead of crashing the whole app.
#
# We catch BOTH ImportError and OSError because:
#   - ImportError: the Python package isn't installed (pip install gmsh pyvista)
#   - OSError: the package is installed but a system shared library is missing
#     (e.g. libGLU.so.1 on a headless server / Streamlit Cloud container).
#     gmsh links against OpenGL / GLU / X11 libraries that are NOT installed
#     by default on Ubuntu server images, so `import gmsh` raises:
#       OSError: libGLU.so.1: cannot open shared object file: No such file or
#                directory
#     even though the gmsh Python wheel is present.  Catching OSError here
#     lets the app keep running with the 3D tab disabled, instead of
#     crashing the entire Streamlit script.
try:
    import gmsh
    import pyvista as pv
    GMSH_AVAILABLE = True
except (ImportError, OSError) as _e:
    GMSH_AVAILABLE = False
    logger.warning("gmsh or pyvista not found or missing system libraries. "
                   "3D mesh generation will be disabled. "
                   "Error: %s", _e)


# ============================================================================
# SENTINEL + ERROR HANDLING  (KEPT verbatim from original)
# ============================================================================
class _Sentinel:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reason = None
        return cls._instance
    def __bool__(self): return False
    def __repr__(self): return f"<FAILED: {self.reason}>"


FAILED = _Sentinel()


def handle_errors(func=None, *, re_raise=False):
    def deco(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                FAILED.reason = f"{f.__name__}: {e}"
                logger.error(f"{e}\n{traceback.format_exc()}")
                try:
                    st.error(f"Error in {f.__name__}: {e}")
                except Exception:
                    pass
                if re_raise:
                    raise
                return FAILED
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        return wrapper
    if func is None:
        return deco
    return deco(func)


def safe_call(result, default=None):
    if result is FAILED or result is None:
        return default
    return result


# ============================================================================
# PHYSICAL CONSTANTS  (REPLACED: battery constants -> aero constants)
# ============================================================================
RHO_AIR_SEA = 1.225          # kg/m^3  (air density at sea level, 15 C)
RHO_AIR_3000M = 0.909        # kg/m^3  (at 3000 m altitude)
G_GRAV = 9.81                # m/s^2
MU_AIR = 1.81e-5             # Pa.s   (dynamic viscosity of air at 15 C)
A_SOUND = 340.0              # m/s    (speed of sound, 15 C)


# ============================================================================
# SPECTRAL FFT OPERATORS  (KEPT verbatim - they are universal)
# ============================================================================
def make_k_vectors(Nx, Ny, dx, dy=None):
    """Build wavenumber vectors for 2D FFT on a rectangular grid."""
    if dy is None:
        dy = dx
    kx = 2.0 * np.pi * fftfreq(Nx, d=dx).reshape(1, -1)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy).reshape(-1, 1)
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0e-12  # avoid divide-by-zero in DC component
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    """Return (df/dx, df/dy) computed via FFT."""
    fh = fft2(field)
    return (np.real(ifft2(1j * kx * fh)),
            np.real(ifft2(1j * ky * fh)))


def spectral_laplacian(field, k2):
    return np.real(ifft2(-k2 * fft2(field)))


def spectral_divergence(fx, fy, kx, ky):
    return np.real(ifft2(1j * kx * fft2(fx) + 1j * ky * fft2(fy)))


# ============================================================================
# COLORMAPS  (KEPT - re-targeted for aero quantities)
# ============================================================================
def _build_custom_colormaps():
    lift_map = LinearSegmentedColormap.from_list('lift', [
        (0.0, '#053061'), (0.3, '#2166ac'), (0.5, '#f7f7f7'),
        (0.7, '#b2182b'), (1.0, '#67001f')])
    drag_map = LinearSegmentedColormap.from_list('drag', [
        (0.0, '#fff7bc'), (0.4, '#fec44d'),
        (0.8, '#d95f0e'), (1.0, '#993404')])
    ld_map = LinearSegmentedColormap.from_list('ld', [
        (0.0, '#000004'), (0.3, '#420a68'), (0.6, '#932667'),
        (0.85, '#f8850f'), (1.0, '#fcffa4')])
    vort_map = LinearSegmentedColormap.from_list('vort', [
        (0.0, '#3b4cc0'), (0.5, '#dddddd'), (1.0, '#b40426')])
    cp_map = LinearSegmentedColormap.from_list('cp', [
        (0.0, '#2c7bb6'), (0.25, '#abd9e9'), (0.5, '#ffffbf'),
        (0.75, '#fdae61'), (1.0, '#d719c1')])
    return {'lift': lift_map, 'drag': drag_map, 'ld': ld_map,
            'vort': vort_map, 'cp': cp_map}


_CUSTOM_COLORMAPS = _build_custom_colormaps()
for _n, _c in _CUSTOM_COLORMAPS.items():
    try:
        if _n not in mpl.colormaps:
            mpl.colormaps.register(_c, name=_n, force=True)
    except Exception as _e:
        logger.warning(f"Could not register colormap '{_n}': {_e}")


class ColormapRegistry:
    BASE = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
            'Greys', 'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
            'YlOrRd', 'YlOrBr', 'YlGnBu', 'YlGn',
            'RdBu', 'coolwarm', 'bwr', 'seismic', 'Spectral',
            'RdYlBu', 'RdYlGn', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
            'twilight', 'hsv', 'jet', 'rainbow', 'nipy_spectral',
            'binary', 'gray', 'bone', 'copper', 'pink', 'hot', 'afmhot',
            'gist_heat', 'gist_earth', 'terrain', 'ocean',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']

    @classmethod
    def list_all(cls):
        names = list(cls.BASE) + list(_CUSTOM_COLORMAPS.keys())
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    @classmethod
    def resolve(cls, name, invert=False):
        if name is None:
            cmap = mpl.colormaps['viridis']
        elif isinstance(name, mpl.colors.Colormap):
            cmap = name
        else:
            key = str(name)
            if key in mpl.colormaps:
                cmap = mpl.colormaps[key]
            elif key in _CUSTOM_COLORMAPS:
                cmap = _CUSTOM_COLORMAPS[key]
            else:
                cmap = mpl.colormaps['viridis']
        if invert:
            cmap = cmap.reversed()
        return cmap

    @classmethod
    def to_plotly(cls, name, n=64, invert=False):
        cmap = cls.resolve(name, invert=invert)
        t = np.linspace(0, 1, n)
        rgb = (cmap(t)[:, :3] * 255).astype(int)
        return [[float(ti), f'rgb({r},{g},{b})']
                for ti, (r, g, b) in zip(t, rgb)]


# ============================================================================
# CUSTOMIZATION SETTINGS  (NEW)
# ----------------------------------------------------------------------------
# Four small dataclasses bundle all user-tunable figure / geometry / mesh /
# surrogate options.  Every field has a default that reproduces the ORIGINAL
# behaviour of the app, so callers that don't pass any settings get the same
# output they got before this customization layer was added.
# ============================================================================
from dataclasses import dataclass, field, asdict
from typing import Optional as _Optional


@dataclass
class FigureSettings:
    """Global figure appearance options applied to every plot in the app."""
    figsize: tuple = (8, 6)
    dpi: int = 100
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10
    legend_size: int = 8
    colormap_ld: str = 'ld'           # for L/D scatters and surrogate surface
    colormap_vort: str = 'vort'       # for vortex / circulation plots
    colormap_drag: str = 'drag'       # for drag heatmaps
    colormap_cp: str = 'cp'           # for Cp-style heatmaps
    show_colorbar: bool = True
    show_grid: bool = True
    grid_alpha: float = 0.3
    show_title: bool = True
    bg_color: str = 'white'
    invert_colormap: bool = False
    # Matplotlib rcParams to apply at figure-construction time.  Stored as a
    # dict so callers can extend or override individual keys.
    rc_overrides: dict = field(default_factory=lambda: {})

    def apply_rc(self):
        """Push font_size / dpi / etc. into matplotlib rcParams."""
        rc = {
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'legend.fontsize': self.legend_size,
            'figure.dpi': self.dpi,
            'figure.facecolor': self.bg_color,
            'axes.facecolor': self.bg_color,
        }
        rc.update(self.rc_overrides)
        mpl.rcParams.update(rc)

    def resolve_cmap(self, kind='ld'):
        """Resolve the colormap registered for the given quantity kind."""
        name = {'ld': self.colormap_ld, 'vort': self.colormap_vort,
                'drag': self.colormap_drag, 'cp': self.colormap_cp}.get(kind,
                                                                        self.colormap_ld)
        return ColormapRegistry.resolve(name, invert=self.invert_colormap)


@dataclass
class GeometryViewSettings:
    """Per-call toggles and colours for the drone top-view + mesh plots."""
    show_frame: bool = True
    show_arms: bool = True
    show_prop_disks: bool = True
    show_blade_qc: bool = True
    show_prop_panels: bool = True
    show_control_points: bool = True
    show_bound_vortices: bool = False       # bound vortex at 1/4-chord
    show_trailing_vortices: bool = False    # horseshoe trailing legs
    show_panel_grid: bool = False           # explicit panel-edge grid on blades
    show_battery: bool = True
    show_camera: bool = True
    show_arm_labels: bool = True
    # Colour palette
    color_frame: str = '#111111'
    color_arms: str = '#333333'
    color_prop_disk: str = '#1f77b4'
    color_blade_qc: str = '#1f77b4'
    color_control_pts: str = '#d62728'
    color_bound_vortex: str = '#2ca02c'
    color_trailing_vortex: str = '#9467bd'
    color_battery: str = '#2ca02c'
    color_camera: str = '#ff7f0e'
    # Visual strength
    disk_alpha: float = 0.25
    blade_alpha: float = 1.0
    panel_alpha: float = 0.6
    lw_frame: float = 2.5
    lw_arms: float = 1.5
    lw_disk: float = 1.0
    lw_blade: float = 2.0
    lw_bound: float = 1.6
    lw_trailing: float = 1.0
    ms_control: float = 5.0
    figsize: tuple = (7, 7)
    equal_aspect: bool = True


@dataclass
class MeshSettings:
    """VLM / mid-fidelity mesh and solver-level tunables."""
    n_panels_per_prop: int = 20            # bumped from 8->16->20 for convergence
    vortex_core_radius: float = 0.05        # Lamb-Oseen core [m], regularizes 1/r singularity
    horseshoe_length_factor: float = 50.0   # trailing-leg length = factor * chord
    r_root_fraction: float = 0.15           # blade root radius = frac * prop_R (was 0.10)
    chord_root_fraction: float = 0.15       # root chord = frac * pitch
    chord_tip_fraction: float = 0.05        # tip chord = frac * pitch
    blade_count_for_viz: int = 1            # blades drawn per prop (VLM uses 1 lifting line)
    n_helix_segments: int = 8              # bumped from 5->8 for smooth helical wake
    panel_spacing: str = 'cosine'           # 'uniform' or 'cosine' (clusters near tip)
    wake_contraction: float = 0.15         # 0.0 = none, 0.15 = 15% radial contraction


@dataclass
class SurrogateSettings:
    """Cokriging surrogate fitting + response-surface plotting options."""
    backend: str = 'auto'                   # 'auto' | 'smt' | 'numpy'
    corr: str = 'squar_exp'                 # 'squar_exp' | 'abs_exp' | 'matern32' (SMT only)
    poly: str = 'constant'                   # 'constant' | 'linear'  (SMT only)
    theta0: float = 1e-2                     # initial hyperparameter
    nugget: float = 1e-8                     # diagonal regularization
    # Response-surface plotting options
    sweep_var_x: int = 0                     # design variable index for X axis
    sweep_var_y: int = 1                     # design variable index for Y axis
    n_grid: int = 30                         # response-surface resolution
    n_levels: int = 20                       # contourf levels
    show_low_fidelity_points: bool = True
    show_high_fidelity_points: bool = True
    show_uncertainty_band: bool = False      # overlay ±sigma contours
    colormap: str = 'ld'

    # Names of the 7 design variables, used by the UI to label the sweep
    # selection dropdowns and the response-surface axes.
    DIM_NAMES = ['arm_L (m)', 'prop_D (m)', 'prop_pitch (m)', 'blades',
                 'AOA (deg)', 'V_cruise (m/s)', 'RPM']


# Module-level default instances.  Existing callers that don't pass any
# settings get these, which reproduce the pre-customization behaviour
# bit-for-bit (same colours, same alpha values, same figure sizes).
DEFAULT_FIG_SETTINGS  = FigureSettings()
DEFAULT_VIEW_SETTINGS  = GeometryViewSettings()
DEFAULT_MESH_SETTINGS  = MeshSettings()
DEFAULT_SURR_SETTINGS  = SurrogateSettings()


# ============================================================================
# FPV DRONE CLASS PROPERTIES  (REPLACES MaterialProperties)
# ============================================================================
class DroneGeometryProperties:
    """Catalogue of FPV drone classes with realistic geometric + aero bounds.

    Each entry fixes the feasible design-variable envelope for that class
    so the optimiser never proposes a physically impossible configuration
    (e.g. a 12-inch prop on a 5-inch frame).
    """

    FPV_5INCH = {
        'name': 'FPV 5"',
        'class': 'racing_freestyle',
        'prop_diameter_range': (0.10, 0.14),     # m  (4.5"-5.5")
        'prop_pitch_range':     (0.040, 0.060),  # m  (pitch in m, ~4-6")
        'blade_count_options':  [2, 3, 5],
        'arm_length_range':     (0.10, 0.16),    # m  (motor-to-motor/2)
        'frame_mass_range':     (0.10, 0.25),    # kg
        'battery_mass_range':   (0.20, 0.45),    # kg  (4S-6S LiPo)
        'cruise_speed_range':   (10.0, 35.0),    # m/s
        'cruise_aoa_range':     (-2.0, 8.0),     # deg
        'prop_rpm_range':       (8000, 18000),   # rev/min
        'n_arms': 4,
        'frame_drag_area': 0.008,                # m^2  (flat-plate equiv)
        'camera_drag_area': 0.0012,              # m^2  (GoPro + mount)
    }

    FPV_7INCH = {
        'name': 'FPV 7"',
        'class': 'long_range',
        'prop_diameter_range': (0.165, 0.20),
        'prop_pitch_range':     (0.060, 0.090),
        'blade_count_options':  [2, 3],
        'arm_length_range':     (0.16, 0.22),
        'frame_mass_range':     (0.20, 0.40),
        'battery_mass_range':   (0.40, 0.90),
        'cruise_speed_range':   (12.0, 30.0),
        'cruise_aoa_range':     (-2.0, 8.0),
        'prop_rpm_range':       (5000, 12000),
        'n_arms': 4,
        'frame_drag_area': 0.012,
        'camera_drag_area': 0.0015,
    }

    FPV_10INCH = {
        'name': 'FPV 10"',
        'class': 'cine_lift',
        'prop_diameter_range': (0.24, 0.28),
        'prop_pitch_range':     (0.080, 0.120),
        'blade_count_options':  [2, 3],
        'arm_length_range':     (0.22, 0.30),
        'frame_mass_range':     (0.40, 0.80),
        'battery_mass_range':   (0.80, 1.80),
        'cruise_speed_range':   (10.0, 25.0),
        'cruise_aoa_range':     (-2.0, 10.0),
        'prop_rpm_range':       (3000, 8000),
        'n_arms': 4,
        'frame_drag_area': 0.018,
        'camera_drag_area': 0.0020,
    }

    @classmethod
    def list_options(cls):
        return ['FPV 5"', 'FPV 7"', 'FPV 10"', 'custom']

    @classmethod
    def get(cls, name):
        if name == 'FPV 5"':  return cls.FPV_5INCH
        if name == 'FPV 7"':  return cls.FPV_7INCH
        if name == 'FPV 10"': return cls.FPV_10INCH
        # custom -> return 5" as a baseline the UI can mutate
        return cls.FPV_5INCH

    @classmethod
    def validate_design(cls, drone_class, design):
        """Validate a design-vector against the class bounds.

        design = dict with keys: arm_length, prop_diameter, prop_pitch,
        blade_count, cruise_aoa_deg, cruise_speed, prop_rpm.
        Returns (errors, warnings).
        """
        spec = cls.get(drone_class)
        errors, warnings = [], []

        def check_in(name, val, lo, hi):
            if not (lo <= val <= hi):
                errors.append(f"{name} = {val} outside [{lo}, {hi}]")

        check_in('arm_length', design['arm_length'],
                 *spec['arm_length_range'])
        check_in('prop_diameter', design['prop_diameter'],
                 *spec['prop_diameter_range'])
        check_in('prop_pitch', design['prop_pitch'],
                 *spec['prop_pitch_range'])
        check_in('cruise_aoa_deg', design['cruise_aoa_deg'],
                 *spec['cruise_aoa_range'])
        check_in('cruise_speed', design['cruise_speed'],
                 *spec['cruise_speed_range'])
        check_in('prop_rpm', design['prop_rpm'],
                 *spec['prop_rpm_range'])

        if design['blade_count'] not in spec['blade_count_options']:
            errors.append(f"blade_count = {design['blade_count']} "
                          f"not in {spec['blade_count_options']}")

        # Reynolds-number sanity check on the prop blade section
        # Re = rho * V_tip * chord / mu ; we approximate chord = 0.10 * pitch
        rho = RHO_AIR_SEA
        V_tip = np.pi * design['prop_diameter'] * design['prop_rpm'] / 60.0
        chord = 0.10 * design['prop_pitch']
        Re = rho * V_tip * chord / MU_AIR
        if Re < 5.0e4:
            warnings.append(f"Blade Re = {Re:.1e} < 5e4 (deep stall / "
                            f"laminar separation likely).")
        if V_tip > 0.7 * A_SOUND:
            warnings.append(f"Tip Mach = {V_tip/A_SOUND:.2f} > 0.7 "
                            f"(compressibility losses).")

        # Total weight check vs. disk loading
        mtow = (spec['frame_mass_range'][0] + spec['battery_mass_range'][0]
                + 4 * 0.020)  # motors, ESCs, wires
        disk_area = np.pi * (design['prop_diameter'] / 2) ** 2 * spec['n_arms']
        disk_loading = mtow * G_GRAV / disk_area
        if disk_loading > 250.0:
            warnings.append(f"Disk loading = {disk_loading:.1f} N/m^2 > 250 "
                            f"(aggressive; efficiency penalty).")
        return errors, warnings


# ============================================================================
# FPV DRONE GEOMETRY BUILDER  (REPLACES NaSSBGeometry)
# ============================================================================
class FPVDroneGeometry:
    """Build a parametric FPV drone geometry.

    Output:
        - 3D node coordinates for frame, arms, propeller disks, battery, camera
        - VLM panel discretisation of each propeller (lifting-line actuator disk)
        - Drag-area catalogue for non-lifting components (frame, battery, camera)

    Convention:
        Drone sits in the x-y plane (top view).  Propeller disks are
        horizontal (in the x-y plane).  Freestream comes from -x direction
        for cruise, so a forward AOA tilts the disks slightly.
    """

    def __init__(self, design, drone_class='FPV 5"', n_panels_per_prop=8,
                 mesh_settings=None):
        self.design = design
        self.drone_class = drone_class
        self.spec = DroneGeometryProperties.get(drone_class)
        self.n_arms = int(self.spec['n_arms'])
        self.n_panels_per_prop = int(n_panels_per_prop)
        # Pull mesh-related geometry parameters from MeshSettings if provided,
        # otherwise use the legacy defaults (r_root=0.10, uniform spacing).
        if mesh_settings is not None:
            self.r_root_fraction = float(mesh_settings.r_root_fraction)
            self.panel_spacing = str(mesh_settings.panel_spacing)
            self.wake_contraction = float(mesh_settings.wake_contraction)
        else:
            self.r_root_fraction = 0.10
            self.panel_spacing = 'uniform'
            self.wake_contraction = 0.0
        self._build()

    @handle_errors
    def _build(self):
        d = self.design
        n_arms = self.n_arms
        arm_L = float(d['arm_length'])
        prop_R = float(d['prop_diameter']) / 2.0
        n_pan = self.n_panels_per_prop

        # --- Arm layout: arms spaced equally around the frame centre ---
        # For a quad: arms at 45, 135, 225, 315 deg (X-config)
        arm_angles = np.linspace(0, 2 * np.pi, n_arms, endpoint=False) \
            + np.pi / n_arms  # offset for X-config
        motor_xy = np.column_stack([
            arm_L * np.cos(arm_angles),
            arm_L * np.sin(arm_angles),
            np.zeros(n_arms)])  # z=0 plane

        # --- Propeller disks (one per motor) ---
        # Each disk is discretised as a lifting line of n_pan horseshoe
        # vortices from root (r=0.10*R) to tip (r=R).  We keep the geometry
        # 2D (top view) for VLM purposes; the trailing vortices extend
        # downstream in +x (freestream direction reversed for induced calc).
        prop_panels = []
        for i in range(n_arms):
            cx, cy, cz = motor_xy[i]
            # blade spanwise stations
            r_root = self.r_root_fraction * prop_R
            r_tip = prop_R
            # Panel spacing: 'cosine' clusters panels near both root and tip
            # (standard VLM practice for tip-vortex resolution), 'uniform'
            # uses equal spacing.
            if self.panel_spacing == 'cosine':
                theta = np.linspace(0.0, np.pi, n_pan + 1)
                r = 0.5 * (r_tip + r_root) \
                    - 0.5 * (r_tip - r_root) * np.cos(theta)
            else:
                r = np.linspace(r_root, r_tip, n_pan + 1)
            # quarter-chord line of the blade (in the disk plane)
            # Pitch twist: geometric pitch -> local AOA varies along span
            pitch = float(d['prop_pitch'])
            # chord distribution (tapered: root chord 0.15*pitch, tip 0.05*pitch)
            chord_root = 0.15 * pitch
            chord_tip = 0.05 * pitch
            chord = np.linspace(chord_root, chord_tip, n_pan)
            # local geometric AOA = atan(pitch / (2*pi*r)) - hub AOA
            # (blade angle decreases towards tip due to increasing tangential v)
            r_mid = 0.5 * (r[:-1] + r[1:])
            local_pitch_angle = np.arctan2(pitch, 2 * np.pi * r_mid)
            # Control points at 3/4-chord from the leading edge.
            # The blade span direction is arm_angle + pi/2, so the chord
            # direction (perpendicular to span) is arm_angle.  The leading
            # edge faces the -arm_angle direction (direction of rotation
            # for a CCW prop).  Mid-chord = 0 offset, LE = -0.5*chord,
            # so 3/4-chord from LE = +0.25*chord from mid-chord, in the
            # +arm_angle direction.
            # BUG FIX (was): control point was at mid-chord (no offset),
            # violating the fundamental VLM requirement that the control
            # point sits at 3/4-chord while the bound vortex sits at 1/4-chord.
            cp_x = cx + r_mid * np.cos(arm_angles[i] + np.pi / 2) \
                 + 0.25 * chord * np.cos(arm_angles[i])
            cp_y = cy + r_mid * np.sin(arm_angles[i] + np.pi / 2) \
                 + 0.25 * chord * np.sin(arm_angles[i])
            cp_z = np.zeros(n_pan)
            # Bound vortex at 1/4-chord from the leading edge = -0.25*chord
            # from mid-chord, in the -arm_angle direction (toward the LE).
            bv_x = cx + (r[:-1] + r[1:]) / 2 * np.cos(arm_angles[i] + np.pi / 2) \
                - 0.25 * chord * np.cos(arm_angles[i])
            bv_y = cy + (r[:-1] + r[1:]) / 2 * np.sin(arm_angles[i] + np.pi / 2) \
                - 0.25 * chord * np.sin(arm_angles[i])
            bv_z = np.zeros(n_pan)
            prop_panels.append({
                'motor_xy': (cx, cy),
                'arm_angle': arm_angles[i],
                'r_root': r_root, 'r_tip': r_tip,
                'r_stations': r, 'r_mid': r_mid,
                'chord': chord,
                'local_pitch_angle': local_pitch_angle,
                'control_pts': np.column_stack([cp_x, cp_y, cp_z]),
                'bound_vortex_pts': np.column_stack([bv_x, bv_y, bv_z]),
                'panel_area': 0.5 * (r[1:] - r[:-1]) * (chord_root + chord_tip),
            })

        # --- Non-lifting components: drag areas ---
        frame_drag = self.spec['frame_drag_area']
        camera_drag = self.spec['camera_drag_area']
        battery_drag = 0.006  # m^2 (battery strap + bottom plate)

        # --- Top-view outline (for plotting) ---
        # Frame outline = polygon connecting arm tips
        frame_outline = np.vstack([motor_xy, motor_xy[:1]])

        self.motor_xy = motor_xy
        self.arm_angles = arm_angles
        self.arm_length = arm_L
        self.prop_R = prop_R
        self.prop_panels = prop_panels
        self.frame_outline = frame_outline
        self.drag_areas = {
            'frame': frame_drag,
            'battery': battery_drag,
            'camera': camera_drag,
            'arms': 4 * arm_L * 0.012,  # arm side area ~12 mm thick
        }
        self.meta = {
            'n_arms': n_arms,
            'n_panels_per_prop': n_pan,
            'total_panels': n_arms * n_pan,
            'prop_diameter': float(d['prop_diameter']),
            'arm_length': arm_L,
            'total_lift_area': n_arms * np.pi * prop_R ** 2,
            'total_drag_area': sum(self.drag_areas.values()),
        }
        return None

    def plot_top_view(self, ax=None, show_prop_panels=True,
                      view_settings=None, fig_settings=None):
        """Top-view matplotlib plot of the drone geometry.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on.  If None, a new figure is created using
            `view_settings.figsize` (default 7x7).
        show_prop_panels : bool
            Legacy toggle kept for backward compatibility.  Overridden by
            `view_settings.show_prop_panels` when `view_settings` is given.
        view_settings : GeometryViewSettings, optional
            Colours, line widths, alpha values, and per-element visibility
            toggles.  Defaults to `DEFAULT_VIEW_SETTINGS` which reproduces
            the original colours (#111, blue props, red control points, etc.).
        fig_settings : FigureSettings, optional
            Applied to the rcParams before drawing when `ax` is None.
        """
        vs = view_settings if view_settings is not None else DEFAULT_VIEW_SETTINGS
        fs = fig_settings  if fig_settings  is not None else DEFAULT_FIG_SETTINGS
        if ax is None:
            if fs is not None:
                fs.apply_rc()
            fig, ax = plt.subplots(figsize=vs.figsize)
        else:
            fig = ax.figure
        # Frame outline
        if vs.show_frame:
            ax.plot(self.frame_outline[:, 0], self.frame_outline[:, 1],
                    '-', color=vs.color_frame, lw=vs.lw_frame, label='Frame')
        # Arms
        cx, cy = 0.0, 0.0
        if vs.show_arms:
            for i in range(self.n_arms):
                mx, my, _ = self.motor_xy[i]
                ax.plot([cx, mx], [cy, my], '-', color=vs.color_arms,
                        lw=vs.lw_arms, alpha=0.7)
                if vs.show_arm_labels:
                    # Place the arm index label mid-way along the arm
                    midx, midy = 0.5 * mx, 0.5 * my
                    ax.text(midx, midy, f'{i+1}', fontsize=8,
                            color=vs.color_arms, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.15',
                                      facecolor='white', edgecolor='none',
                                      alpha=0.7))
        # Propeller disks
        for i, pp in enumerate(self.prop_panels):
            mx, my, _ = self.motor_xy[i]
            theta = self.arm_angles[i] + np.pi / 2
            # draw disk circle
            if vs.show_prop_disks:
                th = np.linspace(0, 2 * np.pi, 60)
                ax.plot(mx + self.prop_R * np.cos(th),
                        my + self.prop_R * np.sin(th),
                        '-', color=vs.color_prop_disk, lw=vs.lw_disk,
                        alpha=vs.disk_alpha)
            # blade quarter-chord line
            if vs.show_blade_qc:
                r = np.linspace(pp['r_root'], pp['r_tip'], 30)
                ax.plot(mx + r * np.cos(theta),
                        my + r * np.sin(theta),
                        '-', color=vs.color_blade_qc, lw=vs.lw_blade,
                        alpha=vs.blade_alpha,
                        label='Blade q-c' if i == 0 else None)
            # Explicit panel-edge grid (vertical lines at each r_station)
            if vs.show_panel_grid:
                for rs in pp['r_stations']:
                    ax.plot([mx + 0 * rs, mx + rs * np.cos(theta)],
                            [my + 0 * rs, my + rs * np.sin(theta)],
                            ':', color=vs.color_blade_qc,
                            lw=0.5, alpha=vs.panel_alpha)
            # Optional: draw the bound vortex (1/4-chord line) explicitly
            if vs.show_bound_vortices:
                bv = pp['bound_vortex_pts']
                ax.plot(bv[:, 0], bv[:, 1], '-',
                        color=vs.color_bound_vortex, lw=vs.lw_bound,
                        alpha=vs.blade_alpha,
                        label='Bound vortex (1/4-c)' if i == 0 else None)
            # Optional: draw trailing-vortex legs of each horseshoe
            if vs.show_trailing_vortices:
                for j in range(self.n_panels_per_prop):
                    p1 = np.array(pp['bound_vortex_pts'][j])
                    L = vs.__dict__.get('_trailing_len',
                                         50.0 * pp['chord'][j])
                    # Trailing leg goes in +x direction (downstream).
                    p2 = p1 + np.array([L, 0.0, 0.0])
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            '-', color=vs.color_trailing_vortex,
                            lw=vs.lw_trailing, alpha=0.5,
                            label='Trailing vortex' if (i == 0
                                                          and j == 0) else None)
            show_panels = vs.show_prop_panels if view_settings is not None \
                else show_prop_panels
            if show_panels:
                for j in range(self.n_panels_per_prop):
                    cp = pp['control_pts'][j]
                    ax.plot(cp[0], cp[1], marker='.',
                            color=vs.color_control_pts,
                            ms=vs.ms_control)
        # Battery (centre box)
        if vs.show_battery:
            batt_size = 0.04
            ax.add_patch(plt.Rectangle((-batt_size, -batt_size),
                                         2 * batt_size, 2 * batt_size,
                                         facecolor=vs.color_battery,
                                         alpha=0.5, edgecolor='black',
                                         label='Battery'))
        # Camera (front nub)
        if vs.show_camera:
            ax.add_patch(plt.Rectangle((self.arm_length * 0.7, -0.015),
                                         0.04, 0.03,
                                         facecolor=vs.color_camera,
                                         alpha=0.7, edgecolor='black',
                                         label='Camera'))
        if vs.equal_aspect:
            ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f"{self.drone_class} top view")
        if fs is not None and fs.show_grid:
            ax.grid(True, alpha=fs.grid_alpha)
        else:
            ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        return fig

    def plot_mesh(self, ax=None, view_settings=None, fig_settings=None,
                   mesh_settings=None, design=None):
        """VLM mesh visualization: horseshoe vortices + control points.

        This is a focused diagnostic plot for the VLM discretisation: it
        draws every bound vortex segment, every trailing-vortex leg (the
        "horseshoe"), and every control point at 3/4-chord.  Useful when
        tuning `MeshSettings.n_panels_per_prop` and
        `MeshSettings.horseshoe_length_factor`.

        If `design` is provided (containing `cruise_speed` and `prop_rpm`),
        the trailing legs are drawn as helical polylines matching the
        solver's wake model.  Otherwise, straight trailing legs are drawn
        as a fallback.

        The plot reuses :class:`GeometryViewSettings` for colours so that
        the same palette applies to both the top-view and the mesh view.
        """
        vs = view_settings if view_settings is not None else \
            GeometryViewSettings(
                show_frame=False, show_arms=True, show_prop_disks=True,
                show_blade_qc=False, show_prop_panels=False,
                show_control_points=True, show_bound_vortices=True,
                show_trailing_vortices=True, show_battery=False,
                show_camera=False, show_arm_labels=False)
        fs = fig_settings  if fig_settings  is not None else DEFAULT_FIG_SETTINGS
        ms = mesh_settings if mesh_settings is not None else DEFAULT_MESH_SETTINGS
        if ax is None:
            if fs is not None:
                fs.apply_rc()
            fig, ax = plt.subplots(figsize=vs.figsize)
        else:
            fig = ax.figure
        # Draw arms (faint, for spatial reference)
        cx, cy = 0.0, 0.0
        for i in range(self.n_arms):
            mx, my, _ = self.motor_xy[i]
            ax.plot([cx, mx], [cy, my], '-', color=vs.color_arms,
                    lw=vs.lw_arms, alpha=0.3)
        # Draw faint prop disks for spatial reference
        for i, pp in enumerate(self.prop_panels):
            mx, my, _ = self.motor_xy[i]
            th = np.linspace(0, 2 * np.pi, 60)
            ax.plot(mx + self.prop_R * np.cos(th),
                    my + self.prop_R * np.sin(th),
                    '-', color=vs.color_prop_disk, lw=vs.lw_disk,
                    alpha=0.15)
        # Draw horseshoe vortices: bound segment + two trailing legs
        for i, pp in enumerate(self.prop_panels):
            mx, my, _ = self.motor_xy[i]
            th = pp['arm_angle'] + np.pi / 2
            for j in range(self.n_panels_per_prop):
                r1 = pp['r_stations'][j]
                r2 = pp['r_stations'][j + 1]
                p1 = np.array([mx + r1 * np.cos(th),
                               my + r1 * np.sin(th)])
                p2 = np.array([mx + r2 * np.cos(th),
                               my + r2 * np.sin(th)])
                # Bound segment
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                        color=vs.color_bound_vortex, lw=vs.lw_bound,
                        alpha=vs.blade_alpha,
                        label='Bound vortex' if (i == 0 and j == 0) else None)
                # Trailing legs: helical wake visualization.
                # Instead of straight +x lines, draw a helical polyline
                # that matches the solver's _helical_leg_induced_velocity.
                # This makes the mesh plot visually consistent with the
                # physics: the wake spirals as it convects downstream.
                L = ms.horseshoe_length_factor * pp['chord'][j]
                Lvis = min(L, 0.6)  # cap for readability
                # Get freestream and RPM from the design dict (if available)
                if design is not None:
                    V_inf = max(float(design.get('cruise_speed', 15.0)), 1.0)
                    rpm = float(design.get('prop_rpm', 10000.0))
                    omega = rpm * 2.0 * np.pi / 60.0
                else:
                    V_inf = 15.0  # fallback
                    omega = 10000.0 * 2.0 * np.pi / 60.0
                t_total = Lvis / V_inf
                n_seg = max(1, ms.n_helix_segments)
                taus = np.linspace(0.0, t_total, n_seg + 1)
                # Wake contraction (matches solver's _helical_leg_induced_velocity)
                k_contract = ms.wake_contraction
                if k_contract > 0.0 and t_total > 0.0:
                    contract_factor = 1.0 - k_contract * (
                        1.0 - np.exp(-3.0 * taus / t_total))
                else:
                    contract_factor = np.ones_like(taus)
                for r_blade in [r1, r2]:
                    r_eff = r_blade * contract_factor
                    angles = th - omega * taus
                    xs = mx + r_eff * np.cos(angles) + V_inf * taus
                    ys = my + r_eff * np.sin(angles)
                    ax.plot(xs, ys, '-',
                            color=vs.color_trailing_vortex, lw=vs.lw_trailing,
                            alpha=0.6,
                            label='Trailing vortex (helical + contraction)'
                            if (i == 0 and j == 0) else None)
                # Control point at 3/4-chord
                cp = pp['control_pts'][j]
                ax.plot(cp[0], cp[1], marker='o',
                        color=vs.color_control_pts, ms=vs.ms_control * 0.7,
                        markeredgecolor='black', markeredgewidth=0.4)
        # Draw a freestream arrow for orientation
        ax.annotate('', xy=(0.6, -0.5), xytext=(0.3, -0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(0.45, -0.55, r'$V_\infty$', fontsize=10, ha='center')
        if vs.equal_aspect:
            ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f"VLM mesh - {self.drone_class} "
                     f"({self.n_panels_per_prop} panels/blade, "
                     f"{self.meta['total_panels']} total)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        return fig


# ============================================================================
# 3D PUBLICATION-GRADE MESH GENERATOR (GMSH OCC API)
# ----------------------------------------------------------------------------
# Generates a true 3D volumetric CFD mesh with NACA 4-digit airfoil sections,
# geometric twist, taper, an MRF (Multiple Reference Frame) cylinder around
# each rotor, a far-field box, and a prism boundary layer on the blade
# surfaces.  Mesh resolution presets match Qiao et al. (2024) Table 2.
#
# The class is self-contained: call `generate_mesh(design, drone_class)` and
# it returns (vtk_path, msh_path, n_cells).  All gmsh state is wrapped in a
# try/finally so that finalize() always runs even if meshing fails midway
# (gmsh cannot be re-initialized without a finalize, so this is critical for
# Streamlit's re-run model).
# ============================================================================
class FPVDrone3DMeshGenerator:
    """Generates a 3D volumetric mesh with boundary layers and MRF regions
    using the Gmsh Python OCC API, matching Qiao et al. (2024) standards."""

    # --- NACA 4-digit airfoil generator (closed trailing edge) ---
    @staticmethod
    def _naca4(m, p, t, n=61):
        """Closed trailing-edge NACA 4-digit airfoil loop.

        Parameters: m = max camber fraction, p = camber position,
                   t = thickness fraction, n = points per surface.
        Returns (xs, ys) arrays for the full closed loop (upper then lower
        reversed), excluding the duplicated trailing-edge point.
        """
        beta = np.linspace(0.0, np.pi, n)
        x = 0.5 * (1.0 - np.cos(beta))
        # Thickness distribution (closed TE: -0.1015 coefficient)
        yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x
                        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        # Camber line
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))
        th = np.arctan(dyc)
        xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
        xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
        # Concatenate upper then lower (reversed, excluding endpoints)
        xs = np.concatenate([xu, xl[-2:0:-1]])
        ys = np.concatenate([yu, yl[-2:0:-1]])
        return xs, ys

    @staticmethod
    @handle_errors
    def generate_mesh(design, drone_class, output_prefix="fpv_3d_mesh",
                       finesse="coarse"):
        """Builds the 3D geometry and meshes it.

        Parameters
        ----------
        design : dict
            Design vector with prop_diameter, prop_pitch, arm_length, etc.
        drone_class : str
            FPV class name (unused for meshing, kept for API symmetry).
        output_prefix : str
            Path prefix for the output .msh and .vtk files.
        finesse : str
            'coarse' or 'fine' — selects the mesh resolution preset.

        Returns
        -------
        (vtk_path, msh_path, n_cells) : tuple
            Paths to the generated VTK and MSH files, plus the cell count.
        """
        if not GMSH_AVAILABLE:
            raise RuntimeError(
                "Gmsh and PyVista must be installed "
                "(`pip install gmsh pyvista`).")

        # --- Presets matching Qiao et al. Table 2 ---
        presets = {
            "coarse": dict(hwall=6.78e-5, bl_thick=8e-3, growth=1.25,
                           h_blade=3.0e-3, h_far=2.5e-1),
            "fine":   dict(hwall=4.07e-6, bl_thick=6e-3, growth=1.13,
                           h_blade=1.2e-3, h_far=1.2e-1),
        }
        if finesse not in presets:
            finesse = "coarse"
        cfg = dict(presets[finesse])

        # Extract design parameters
        R = float(design['prop_diameter']) / 2.0
        pitch = float(design['prop_pitch'])
        arm_L = float(design['arm_length'])
        n_arms = 4  # Simplified for 3D demo (quad X-config)
        cfg.update(R=R, pitch=pitch, arm_L=arm_L, c_root=0.15 * pitch)

        # CRITICAL: wrap the entire gmsh session in try/finally so that
        # gmsh.finalize() ALWAYS runs, even if meshing fails midway.
        # Without this, a failed mesh leaves gmsh initialized and the next
        # call to gmsh.initialize() raises "gmsh is already initialized".
        try:
            gmsh.initialize()
            gmsh.model.add("fpv_quad_3d")
            occ = gmsh.model.occ

            # --- 1. Build one representative 3D blade per arm ---
            # NACA 4412: m=0.04, p=0.4, t=0.12
            xs, ys = FPVDrone3DMeshGenerator._naca4(0.04, 0.4, 0.12)
            # Spanwise stations (fraction of R) where airfoil sections are
            # placed.  Clustered near tip to resolve tip vortex.
            r_fracs = (0.15, 0.35, 0.60, 0.80, 0.95, 0.995)

            all_blade_vols = []

            for arm_idx in range(n_arms):
                arm_angle = (arm_idx + 0.5) * 2 * np.pi / n_arms
                mx = arm_L * np.cos(arm_angle)
                my = arm_L * np.sin(arm_angle)
                # Blade span direction (radially outward from motor)
                ur = np.array([np.cos(arm_angle), np.sin(arm_angle), 0.0])
                zax = np.array([0.0, 0.0, 1.0])

                wires = []
                for rf in r_fracs:
                    r = rf * R
                    # Chord taper: root chord = c_root, tip = 45% of root
                    c = cfg["c_root"] * (1.0 - 0.55 * rf)
                    # Geometric twist: pitch angle decreases toward tip
                    beta = np.arctan2(pitch, 2.0 * np.pi * r)
                    # Chord direction (rotated by twist)
                    e_chord = np.cos(beta) * ur + np.sin(beta) * zax
                    e_thick = -np.sin(beta) * ur + np.cos(beta) * zax
                    pts = []
                    for x, y in zip(xs, ys):
                        p = (np.array([mx, my, 0.0]) + r * ur
                             + (x - 0.25) * c * e_chord
                             + y * c * e_thick)
                        pts.append(occ.addPoint(
                            float(p[0]), float(p[1]), float(p[2])))
                    pts.append(pts[0])  # close the loop
                    spl = occ.addBSpline(pts)
                    cl = occ.addCurveLoop([spl])
                    wires.append(occ.addWire([cl]))

                # Loft a solid through the airfoil sections
                blade_vol = occ.addThruSections(wires, makeSolid=True)
                # addThruSections returns [(dim, tag), ...]
                all_blade_vols.extend(
                    [(d, t) for d, t in blade_vol if d == 3])

            # --- 2. Add MRF cylinder around each rotor ---
            mrf_vols = []
            for arm_idx in range(n_arms):
                arm_angle = (arm_idx + 0.5) * 2 * np.pi / n_arms
                mx = arm_L * np.cos(arm_angle)
                my = arm_L * np.sin(arm_angle)
                mrf_tag = occ.addCylinder(
                    mx, my, -0.05, 0, 0, 0.10, 1.20 * R)
                mrf_vols.append((3, mrf_tag))

            # --- 3. Boolean operations ---
            occ.synchronize()
            # Cut blade volumes out of MRF cylinders
            if all_blade_vols and mrf_vols:
                mrf_vols = occ.cut(mrf_vols, all_blade_vols)[0]
            # Far-field box
            far_box = occ.addBox(-0.5, -0.5, -0.3, 1.0, 1.0, 0.6)
            far_vols = occ.cut([(3, far_box)],
                                all_blade_vols + mrf_vols)[0]
            occ.synchronize()

            # --- 4. Meshing with boundary layer ---
            gmsh.option.setNumber("Mesh.MeshSizeMin", cfg["h_blade"] / 4)
            gmsh.option.setNumber("Mesh.MeshSizeMax", cfg["h_far"])
            gmsh.option.setNumber("Mesh.Algorithm", 6)      # 2D: Frontal
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # 3D: HXT

            # Boundary layer on blade surfaces
            if all_blade_vols:
                walls = occ.getBoundary(all_blade_vols, oriented=False,
                                         combined=False)
                f_bl = gmsh.model.mesh.field.add("BoundaryLayer")
                gmsh.model.mesh.field.setNumber(
                    f_bl, "hwall_n", cfg["hwall"])
                gmsh.model.mesh.field.setNumber(
                    f_bl, "thickness", cfg["bl_thick"])
                gmsh.model.mesh.field.setNumber(
                    f_bl, "ratio", cfg["growth"])
                gmsh.model.mesh.field.setNumbers(
                    f_bl, "SurfacesList",
                    [t for d, t in walls])
                gmsh.model.mesh.field.setAsBoundaryLayer(f_bl)

            gmsh.model.mesh.generate(3)

            # --- 5. Export ---
            msh_path = f"{output_prefix}.msh"
            vtk_path = f"{output_prefix}.vtk"
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(msh_path)
            gmsh.write(vtk_path)

            # Count 3D cells for diagnostics
            elem_info = gmsh.model.mesh.getElements(3)
            n_cells = sum(len(tags) for tags in elem_info[2]) \
                if len(elem_info) > 2 else 0

            return vtk_path, msh_path, n_cells

        finally:
            # ALWAYS finalize, even if meshing raised an exception.
            # This prevents "gmsh already initialized" on the next call.
            try:
                gmsh.finalize()
            except Exception:
                pass


# ============================================================================
# SPECTRAL VORTEX-LATTICE (VLM) SOLVER  (REPLACES SpectralElasticSolver)
# ============================================================================
class SpectralVLMSolver:
    """Low-fidelity: classical vortex-lattice method for the FPV drone.

    Each propeller blade is a lifting line of horseshoe vortices.
    The induced-velocity influence matrix A_{ij} is built from the
    Biot-Savart law; the linear system  A * Gamma = RHS  is solved for
    the bound-vortex strengths Gamma.

    The 'spectral' label is justified because the trailing-vortex
    contribution to the induced velocity (the part that extends to
    infinity downstream) can be evaluated as an FFT-based convolution
    when the control-point grid is uniform - this is the same trick
    that makes the original battery code's Poisson solve fast.

    For non-uniform propeller grids we fall back to direct Biot-Savart.
    The class exposes both paths so a future JAX/FFT-accelerated upgrade
    is a drop-in replacement.
    """

    def __init__(self, geometry: FPVDroneGeometry, design: dict,
                 rho=RHO_AIR_SEA, mesh_settings=None):
        self.geom = geometry
        self.design = design
        self.rho = float(rho)
        self.freestream_speed = float(design['cruise_speed'])
        self.aoa_deg = float(design['cruise_aoa_deg'])
        self.aoa_rad = np.deg2rad(self.aoa_deg)
        self.rpm = float(design['prop_rpm'])
        self.omega = self.rpm * 2 * np.pi / 60.0   # rad/s
        # Mesh settings: horseshoe trailing-leg length = factor * chord.
        # Defaults reproduce the original hard-coded 50.0 factor.
        ms = mesh_settings if mesh_settings is not None else DEFAULT_MESH_SETTINGS
        self.horseshoe_length_factor = float(ms.horseshoe_length_factor)
        # Helical wake discretisation: n_helix_segments = 1 gives the
        # simplified single-segment helical offset; n_helix_segments >= 5
        # gives a proper curved helical polyline.
        self.n_helix_segments = int(ms.n_helix_segments)
        # Lamb-Oseen vortex core radius: regularizes the 1/r singularity in
        # Biot-Savart when control points are close to vortex filaments.
        # rc=0 disables regularization (point-vortex); rc>0 applies the
        # standard (1 - exp(-r²/rc²)) damping factor.
        self.vortex_core_radius = float(ms.vortex_core_radius)
        # Wake contraction: the helix radius decreases as the wake convects
        # downstream, asymptoting to (1-wake_contraction)*r.  This models
        # the pressure-driven contraction of real propeller wakes.
        self.wake_contraction = float(ms.wake_contraction)
        # Build linear system once (geometry doesn't change during a solve)
        self._build_influence_matrix()

    # ------------------------------------------------------------------
    def _build_influence_matrix(self):
        """Build A_{ij} = induced downwash at control point i per unit
        vortex strength on panel j.  O(N^2) direct Biot-Savart; for
        uniform grids this could be replaced by FFT convolution."""
        panels = []
        for pp in self.geom.prop_panels:
            for j in range(self.geom.n_panels_per_prop):
                panels.append((pp, j))
        N = len(panels)
        A = np.zeros((N, N))
        for i, (ppi, ji) in enumerate(panels):
            cp = ppi['control_pts'][ji]
            for j, (ppj, jj) in enumerate(panels):
                if i == j:
                    # Self-induced downwash: use the same horseshoe formula
                    # but with the Lamb-Oseen core active.  The old
                    # 1/(4*pi*chord) approximation was too crude and caused
                    # the influence matrix to be ill-conditioned at high
                    # panel counts (adjacent CPs get very close to each
                    # other's bound vortices).  Computing the self-influence
                    # via the actual Biot-Savart with core regularization
                    # gives a well-conditioned matrix.
                    A[i, j] = self._horseshoe_induced_downwash(
                        ppj, jj, cp)
                else:
                    A[i, j] = self._horseshoe_induced_downwash(
                        ppj, jj, cp)
        # Add a small diagonal regularization to ensure the matrix is
        # well-conditioned even when the geometry is degenerate.
        A += 1e-8 * np.eye(N)
        self.A = A
        self.panels = panels
        self.N = N

    # ------------------------------------------------------------------
    def _horseshoe_induced_downwash(self, pp, j, cp):
        """Induced z-velocity at point cp from a unit-strength horseshoe
        vortex on panel j of propeller pp.

        Horseshoe = bound segment at 1/4-chord + two trailing vortices
        extending to +x infinity.
        """
        bv = pp['bound_vortex_pts'][j]
        mx, my = pp['motor_xy']
        th = pp['arm_angle'] + np.pi / 2  # blade spanwise direction

        # Bound segment endpoints (in blade-span direction)
        r1 = pp['r_stations'][j]
        r2 = pp['r_stations'][j + 1]
        p1 = np.array([mx + r1 * np.cos(th), my + r1 * np.sin(th), 0.0])
        p2 = np.array([mx + r2 * np.cos(th), my + r2 * np.sin(th), 0.0])

        w_bound = self._biot_savart_segment(p1, p2, cp)
        # Trailing vortices: helical wake approximation.
        #
        # A propeller wake is NOT straight — it spirals because the blade
        # rotates while the wake convects downstream.  The trailing vortex
        # filament from a blade point at radius r, blade angle th, follows
        # a helix parameterised by wake age tau >= 0:
        #
        #   x(tau) = mx + r * cos(th - omega * tau) + V_inf * tau
        #   y(tau) = my + r * sin(th - omega * tau)
        #   z(tau) = 0
        #
        # At tau=0 the point is at the blade (p1 or p2).  As tau increases,
        # the point convects downstream (+V_inf * tau in x) and the angular
        # position drifts backward (th - omega * tau) because the wake was
        # shed when the blade was at an earlier angle.
        #
        # We discretise the helix into n_helix_segments straight Biot-Savart
        # segments.  n_helix_segments=1 reduces to the simplified single-
        # segment helical offset (endpoint rotated, leg still straight).
        # n_helix_segments >= 5 gives a proper curved helical wake.
        L = self.horseshoe_length_factor * pp['chord'][j]
        V_inf = max(self.freestream_speed, 1.0)  # avoid div-by-zero in hover
        t_total = L / V_inf                        # convection time over length L
        theta_rot = self.omega * t_total           # total rotation angle
        n_seg = max(1, self.n_helix_segments)

        w_t1 = self._helical_leg_induced_velocity(
            mx, my, r1, th, t_total, n_seg, cp)
        w_t2 = self._helical_leg_induced_velocity(
            mx, my, r2, th, t_total, n_seg, cp)
        # Z-component of total induced velocity (per unit Gamma)
        return (w_bound + w_t1 + w_t2)[2]

    def _helical_leg_induced_velocity(self, mx, my, r, th, t_total,
                                       n_seg, cp):
        """Compute the induced velocity at cp from a helical trailing
        vortex leg shed from blade point (mx + r*cos(th), my + r*sin(th), 0).

        The helix is discretised into n_seg straight Biot-Savart segments.
        Wake contraction is applied: the effective helix radius decreases
        as the wake convects downstream, asymptoting to
        (1 - wake_contraction) * r at tau = t_total.
        """
        V_inf = max(self.freestream_speed, 1.0)
        omega = self.omega
        # Generate the helix points: tau from 0 to t_total
        taus = np.linspace(0.0, t_total, n_seg + 1)
        # Wake contraction: r_eff(tau) = r * (1 - k * (1 - exp(-3*tau/t_total)))
        # At tau=0: r_eff = r (no contraction at the blade).
        # At tau=t_total: r_eff ≈ r * (1 - k * 0.95) ≈ r * (1 - k).
        # The factor 3 in the exponent gives fast contraction (95% of the
        # way to the asymptotic value by tau = t_total).
        k = self.wake_contraction
        if k > 0.0 and t_total > 0.0:
            r_eff = r * (1.0 - k * (1.0 - np.exp(-3.0 * taus / t_total)))
        else:
            r_eff = np.full_like(taus, r)
        # Pre-compute all helix points
        angles = th - omega * taus          # angular position at each tau
        xs = mx + r_eff * np.cos(angles) + V_inf * taus
        ys = my + r_eff * np.sin(angles)
        zs = np.zeros(n_seg + 1)
        pts = np.column_stack([xs, ys, zs])
        # Sum Biot-Savart over each straight segment
        w_total = np.zeros(3)
        for k_seg in range(n_seg):
            w_total += self._biot_savart_segment(pts[k_seg], pts[k_seg + 1], cp)
        return w_total

    def _biot_savart_segment(self, p1, p2, cp):
        """Biot-Savart induced velocity at cp from a unit-strength straight
        vortex segment from p1 to p2.  Returns the 3-vector w.

        Applies Lamb-Oseen vortex-core regularization when
        self.vortex_core_radius > 0: the 1/r_perp singularity is replaced
        by (1 - exp(-r_perp²/rc²)) / r_perp, which goes to zero as
        r_perp → 0 instead of blowing up.  This prevents numerical
        instabilities when control points are close to vortex filaments
        (e.g. wake overlap between adjacent rotors)."""
        r1 = cp - p1
        r2 = cp - p2
        L = p2 - p1
        cross = np.cross(L, r1)
        cross_norm_sq = float(np.linalg.norm(cross) ** 2) + 1e-12
        L_norm_sq = float(np.linalg.norm(L) ** 2) + 1e-12
        # Perpendicular distance from cp to the vortex line
        r_perp_sq = cross_norm_sq / L_norm_sq
        r1n = np.linalg.norm(r1) + 1e-12
        r2n = np.linalg.norm(r2) + 1e-12
        factor = (np.dot(r1, L) / r1n - np.dot(r2, L) / r2n)
        # Lamb-Oseen core regularization factor
        rc = self.vortex_core_radius
        if rc > 0.0:
            core_factor = 1.0 - np.exp(-r_perp_sq / (rc * rc))
        else:
            core_factor = 1.0
        w = factor * cross / cross_norm_sq * core_factor / (4.0 * np.pi)
        return w

    # ------------------------------------------------------------------
    @handle_errors
    def solve(self):
        """Solve the VLM linear system A*Gamma = RHS for circulation
        strengths and compute aerodynamic forces.

        Returns dict with CL, CD, CDi, CD0, L_D, total_lift, total_drag,
        thrust_required, gamma_distribution, panel_loadings.
        """
        # RHS[i] = component of (freestream + rotation) perpendicular to
        # the pitched blade chord that the induced velocity must cancel.
        #
        # For a blade section at local pitch angle beta(r) with axial flow
        # V_axial and tangential (rotational) flow V_tan = omega*r, the
        # kinematic angle of attack is
        #     alpha_eff = beta - atan2(V_axial, V_tan)
        # and the "downwash to be cancelled" by the bound + trailing
        # vortices is
        #     RHS = V_eff * sin(alpha_eff)
        # where V_eff = sqrt(V_axial^2 + V_tan^2).
        #
        # This is the standard blade-element / VLM coupling for propellers:
        # the freestream+rotation sees the pitched chord as an inclined
        # flat plate, and the VLM circulation must produce exactly enough
        # induced velocity to make the total flow tangent to the chord.
        V_axial = self.freestream_speed
        rhs = np.zeros(self.N)
        for i, (pp, ji) in enumerate(self.panels):
            r_mid = pp['r_mid'][ji]
            local_pitch = float(pp['local_pitch_angle'][ji])  # beta(r)
            V_tan = self.omega * r_mid                          # tangential
            V_eff = np.hypot(V_axial, V_tan)
            # kinematic angle of attack (radians)
            alpha_eff = local_pitch - np.arctan2(V_axial, max(V_tan, 1e-6))
            # Project onto panel-normal direction (z-component for small beta)
            rhs[i] = V_eff * np.sin(alpha_eff)

        # Solve for Gamma using a regularized least-squares approach.
        # The influence matrix A can be ill-conditioned (especially at high
        # panel counts where adjacent CPs get close to each other's bound
        # vortices), so we use Tikhonov regularization: minimize
        #   ||A*gamma - rhs||^2 + lambda*||gamma||^2
        # The lambda is scaled by the largest diagonal entry of A'A so
        # that the regularization is relative to the matrix's natural
        # scale (not an absolute constant that would over- or under-
        # regularize at different panel counts).
        try:
            ATA = self.A.T @ self.A
            ATb = self.A.T @ rhs
            # Scale lambda by the trace of ATA so it's relative
            scale = max(np.trace(ATA) / max(self.N, 1), 1e-12)
            lambda_reg = 1e-3 * scale
            gamma = np.linalg.solve(
                ATA + lambda_reg * np.eye(self.N), ATb)
        except np.linalg.LinAlgError:
            gamma = np.linalg.lstsq(self.A, rhs, rcond=None)[0]

        # Compute lift per panel = rho * V_eff * Gamma * span
        panel_lift = np.zeros(self.N)
        for i, (pp, ji) in enumerate(self.panels):
            r1 = pp['r_stations'][ji]
            r2 = pp['r_stations'][ji + 1]
            span = r2 - r1
            V_eff = np.sqrt(self.freestream_speed ** 2
                            + (self.omega * pp['r_mid'][ji]) ** 2)
            panel_lift[i] = self.rho * V_eff * gamma[i] * span

        # CRITICAL FIX: The VLM discretises each propeller as a SINGLE lifting
        # line (one representative blade).  The actual propeller has
        # `blade_count` blades, so the total thrust/lift must be multiplied
        # by the number of blades.  Without this multiplier, a 3-blade prop
        # produces only 1/3 of its real lift, creating a massive (and false)
        # L/D ratio because the viscous drag (computed in MidFidelityCorrector)
        # already accounts for all blades via the wetted-area term.
        n_blades = int(self.design.get('blade_count', 1))
        total_lift = float(np.sum(panel_lift)) * n_blades

        # Induced drag (treating each prop as a finite wing):
        # CDi = CL^2 / (pi * AR * e),  e ~ 0.7 for prop disks
        AR = (2 * self.geom.prop_R) ** 2 / \
             (np.pi * self.geom.prop_R ** 2 / self.geom.n_arms)
        e_factor = 0.7
        # Total drag = parasitic (frame + battery + camera + arms) + induced
        V_cruise = self.freestream_speed
        q_dyn = 0.5 * self.rho * V_cruise ** 2
        CD0_total = self.geom.meta['total_drag_area'] / \
                    max(self.geom.meta['total_lift_area'], 1e-4)
        # CL_total based on lifting area
        CL_total = total_lift / (q_dyn * self.geom.meta['total_lift_area']
                                 + 1e-9)
        CDi_total = CL_total ** 2 / (np.pi * AR * e_factor + 1e-9)
        CD_total = CD0_total + CDi_total
        total_drag = CD_total * q_dyn * self.geom.meta['total_lift_area']
        L_D = total_lift / (total_drag + 1e-9)

        # Thrust required for level cruise = drag (forward) + weight-comp
        mtow_est = (self.geom.spec['frame_mass_range'][0]
                    + self.geom.spec['battery_mass_range'][0]
                    + 4 * 0.020)  # motors etc.
        weight = mtow_est * G_GRAV
        thrust_required = float(total_drag + 0.05 * weight)  # 5% margin

        return {
            'CL': float(CL_total),
            'CD': float(CD_total),
            'CDi': float(CDi_total),
            'CD0': float(CD0_total),
            'L_D': float(L_D),
            'total_lift': float(total_lift),
            'total_drag': float(total_drag),
            'thrust_required': thrust_required,
            'gamma_distribution': gamma.copy(),
            'panel_loadings': panel_lift.copy(),
            'panels': self.panels,
            'mtow_est': mtow_est,
        }


# ============================================================================
# MID-FIDELITY CORRECTOR  (NEW)
# ============================================================================
class MidFidelityCorrector:
    """Add physical corrections on top of the inviscid VLM:

    1. Vortex core decay  (Lamb-Oseen)  - reduces the induced-velocity
       peaks at the vortex core, more physical for close interactions
       between rotor wakes.
    2. Flat-plate skin-friction + form-factor viscous drag build-up.
    3. Blade-element momentum theory correction for the propeller thrust
       (VLM is inviscid and over-predicts thrust at low Reynolds).

    This is the 'intermediate tier' in the variable-fidelity hierarchy.
    """

    def __init__(self, vortex_core_radius=0.05):
        self.r_core = float(vortex_core_radius)

    @handle_errors
    def correct(self, vlm_result, design, geometry):
        """Return a mid-fidelity-augmented result dict."""
        V = float(design['cruise_speed'])
        rho = RHO_AIR_SEA
        mu = MU_AIR

        # --- 1. Lamb-Oseen vortex core correction on induced drag ---
        # CDi_lamb = CDi_vlm * (1 - exp(-(r_core/R)^2))
        # i.e. a small reduction because the core has finite size
        r_corR = self.r_core / geometry.prop_R
        core_factor = 1.0 - 0.30 * np.exp(-r_corR ** 2 / 0.05)
        CDi_mid = vlm_result['CDi'] * core_factor

        # --- 2. Flat-plate viscous drag build-up ---
        # Cf = 0.074 / Re^(1/5)  (Prandtl-Schlichting, turbulent)
        # Re_ref = rho * V * mean_chord / mu
        mean_chord = float(np.mean([np.mean(pp['chord'])
                                    for pp in geometry.prop_panels]))
        Re = rho * V * mean_chord / mu
        Re = max(Re, 1.0e3)
        Cf = 0.074 / Re ** 0.2
        # Form factor for blade sections ~1.2
        FF = 1.2
        # Wetted area ~ 2 * chord * span * n_arms * n_blades
        n_blades = int(design['blade_count'])
        blade_span = geometry.prop_R * 0.9
        wetted = 2 * mean_chord * blade_span * geometry.n_arms * n_blades
        # Parasitic drag = 0.5 * rho * V^2 * Cf * FF * wetted
        D_visc = 0.5 * rho * V ** 2 * Cf * FF * wetted

        # Non-lifting component drag (frame, battery, camera, arms)
        D_parasitic = 0.5 * rho * V ** 2 * geometry.meta['total_drag_area']

        # --- 3. BEMT thrust correction (simple) ---
        # VLM tends to over-predict thrust at low Re; apply a thrust
        # reduction factor that goes to 1 at high Re.
        thrust_corr = 1.0 - 0.20 * np.exp(-Re / 1.0e5)
        thrust_mid = vlm_result['total_lift'] * thrust_corr

        # Combine
        q_dyn = 0.5 * rho * V ** 2
        S_lift = geometry.meta['total_lift_area']
        CD_mid = (CDi_mid
                  + D_visc / (q_dyn * S_lift + 1e-9)
                  + D_parasitic / (q_dyn * S_lift + 1e-9))
        CL_mid = thrust_mid / (q_dyn * S_lift + 1e-9)
        L_D_mid = CL_mid / (CD_mid + 1e-9)

        return {
            'CL': float(CL_mid),
            'CD': float(CD_mid),
            'CDi': float(CDi_mid),
            'CD0': float(CD_mid - CDi_mid),
            'L_D': float(L_D_mid),
            'total_lift': float(thrust_mid),
            'total_drag': float(CD_mid * q_dyn * S_lift),
            'thrust_required': float(vlm_result['thrust_required']),
            'Re_blade': float(Re),
            'Cf': float(Cf),
            'thrust_corr_factor': float(thrust_corr),
            'core_factor': float(core_factor),
        }


# ============================================================================
# HIGH-FIDELITY HOOK  (NEW - external solver interface only)
# ============================================================================
class HighFidelityHook:
    """Placeholder for an external CFD solver call.

    In a real deployment this would:
        - generate a 3D CFD mesh (via :class:`FPVDrone3DMeshGenerator`)
        - write a STAR-CCM+ / SU2 / OpenFOAM case file pointing to that mesh
        - submit to an HPC queue
        - parse the resulting force report

    Here it returns a perturbed version of the mid-fidelity result so the
    variable-fidelity pipeline can be exercised end-to-end without a real
    CFD licence.  When ``generate_3d_mesh=True`` and Gmsh is available, a
    real 3D mesh is generated for each evaluation and the cell count is
    included in the result (useful for diagnostics and for the cokriging
    surrogate to learn mesh-resolution effects).
    """

    def __init__(self, cfd_noise=0.05, seed=42, generate_3d_mesh=False):
        self.cfd_noise = float(cfd_noise)
        self.rng = np.random.default_rng(seed)
        # When True, generate a 3D Gmsh mesh for each high-fidelity eval.
        # This is slow (seconds per eval) so it's opt-in.
        self.generate_3d_mesh = bool(generate_3d_mesh)

    @handle_errors
    def evaluate(self, design, drone_class, mesh_settings=None):
        """Stub: pretend to run a CFD calculation and return a noisy
        version of the mid-fidelity answer.

        If ``self.generate_3d_mesh`` is True and Gmsh is available, a
        real 3D volumetric mesh is generated for this specific design and
        the cell count is included in the returned dict.  This bridges the
        low-fidelity VLM and the high-fidelity CFD mesh: the optimizer can
        propose a design, and the hook produces the exact 3D mesh that a
        real CFD solver would consume.

        If mesh_settings is provided, the underlying VLM + mid-fidelity
        corrector use those settings (panel count, vortex core, horseshoe
        length); otherwise the defaults reproduce the original behaviour.
        """
        ms = mesh_settings if mesh_settings is not None else DEFAULT_MESH_SETTINGS
        geom = FPVDroneGeometry(design, drone_class,
                                 n_panels_per_prop=ms.n_panels_per_prop,
                                 mesh_settings=ms)
        vlm = SpectralVLMSolver(geom, design, mesh_settings=ms).solve()
        mid = MidFidelityCorrector(
            vortex_core_radius=ms.vortex_core_radius
        ).correct(vlm, design, geom)

        # Optionally generate a real 3D mesh for this design.
        n_cells = None
        mesh_source = 'CFD_Hook_stub'
        if self.generate_3d_mesh and GMSH_AVAILABLE:
            try:
                design_hash = hashlib.md5(
                    str(design).encode()).hexdigest()[:8]
                prefix = f"/tmp/hf_eval_{design_hash}"
                _vtk, _msh, n_cells = \
                    FPVDrone3DMeshGenerator.generate_mesh(
                        design, drone_class, output_prefix=prefix,
                        finesse="coarse")
                mesh_source = 'CFD_Stub_With_3D_Mesh_Gen'
            except Exception as e:
                logger.warning(f"3D mesh generation failed: {e}")
                mesh_source = 'CFD_Hook_stub (3D mesh failed)'

        # Add a structured bias + noise so cokriging has something to learn
        bias_CL = 0.02 * np.sin(design['cruise_aoa_deg'] / 5.0)
        bias_CD = 0.01 * (design['prop_rpm'] - 10000) / 10000.0
        noise_CL = self.cfd_noise * self.rng.standard_normal()
        noise_CD = self.cfd_noise * self.rng.standard_normal() * 0.3

        result = {
            'CL': mid['CL'] * (1 + bias_CL + noise_CL),
            'CD': max(mid['CD'] * (1 + bias_CD + noise_CD), 1e-6),
            'L_D': float((mid['CL'] * (1 + bias_CL + noise_CL))
                         / max(mid['CD'] * (1 + bias_CD + noise_CD), 1e-6)),
            'total_drag': mid['total_drag'] * (1 + bias_CD + noise_CD),
            'source': mesh_source,
        }
        if n_cells is not None:
            result['mesh_cells'] = n_cells
        return result


# ============================================================================
# COKRIGING MULTI-FIDELITY SURROGATE  (NEW)
# ============================================================================
class CokrigingSurrogate:
    """Multi-fidelity Gaussian-process surrogate.

    Models the high-fidelity response as:

        f_h(x) = rho * f_l(x) + delta(x)

    where f_l is a GP on the low-fidelity samples and delta is a GP on
    the (high - rho*low) discrepancy.  (Kennedy & O'Hagan 2000.)

    If the `smt` package is available we delegate to its MFK
    (Multi-Fidelity Kriging) implementation; otherwise we fall back to
    a numpy-only radial-basis-function approximation that captures the
    essence (cheap + expensive samples combined) without the full GP
    machinery.
    """

    def __init__(self, n_low=20, n_high=5, surrogate_settings=None):
        self.n_low = int(n_low)
        self.n_high = int(n_high)
        self.model = None
        self.X_low = None
        self.y_low = None
        self.X_high = None
        self.y_high = None
        # SurrogateSettings drive the SMT kernel choice, nugget, theta0,
        # polynomial basis, and the response-surface plotting options.
        # Defaults reproduce the original (squar_exp, constant poly, 1e-2
        # theta0, 1e-8 nugget, auto backend).
        self.settings = surrogate_settings if surrogate_settings is not None \
            else DEFAULT_SURR_SETTINGS
        if self.settings.backend == 'smt' and not SMT_AVAILABLE:
            logger.warning("SurrogateSettings requested backend='smt' but "
                           "smt is not installed; falling back to numpy RBF.")
        if self.settings.backend == 'numpy':
            self.use_smt = False
        elif self.settings.backend == 'smt':
            self.use_smt = SMT_AVAILABLE
        else:  # 'auto'
            self.use_smt = SMT_AVAILABLE

    @handle_errors
    def fit(self, X_low, y_low, X_high, y_high):
        """Fit the surrogate on the given low- and high-fidelity samples.

        X_low  : (n_low, d)  low-fidelity design vectors
        y_low  : (n_low,)    low-fidelity responses (e.g. L/D from VLM+mid)
        X_high : (n_high, d) high-fidelity design vectors (subset of X_low)
        y_high : (n_high,)   high-fidelity responses
        """
        self.X_low = np.asarray(X_low, dtype=float)
        self.y_low = np.asarray(y_low, dtype=float).ravel()
        self.X_high = np.asarray(X_high, dtype=float)
        self.y_high = np.asarray(y_high, dtype=float).ravel()

        if self.use_smt:
            try:
                s = self.settings
                self.model = MFK(theta0=[s.theta0] * self.X_low.shape[1],
                                 poly=s.poly,
                                 corr=s.corr,
                                 nugget=s.nugget)
                # SMT expects nested fidelities: low first, high last
                X = np.vstack([self.X_low, self.X_high])
                y = np.concatenate([self.y_low, self.y_high])
                # fidelity flags: 0 = low, 1 = high
                fid = np.concatenate([
                    np.zeros(len(self.X_low), dtype=int),
                    np.ones(len(self.X_high), dtype=int)])
                self.model.set_training_values(X, y)
                self.model._n_fid = 2
                # SMT's MFK API varies between versions; this is the
                # generic path.  If a specific version needs a different
                # call we fall back to the numpy RBF.
                self.model.train()
                return True
            except Exception as e:
                logger.warning(f"SMT MFK failed ({e}); falling back to "
                               f"numpy RBF surrogate.")
                self.use_smt = False

        # ---- Numpy RBF fallback ----
        # Build two thin-plate-spline RBF interpolants: one for low
        # fidelity, one for the discrepancy.  Cheap and bullet-proof.
        self._rbf_low = self._fit_rbf(self.X_low, self.y_low)
        # discrepancy at the high-fidelity points
        low_at_high = self._eval_rbf(self._rbf_low, self.X_high)
        delta = self.y_high - low_at_high
        self._rbf_delta = self._fit_rbf(self.X_high, delta)
        return True

    @staticmethod
    def _fit_rbf(X, y):
        """Thin-plate-spline RBF interpolation (numpy-only)."""
        N, d = X.shape
        # Pairwise distance matrix
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        # TPS kernel: r^2 log(r)
        eps = 1e-8
        Phi = np.where(D > eps, D ** 2 * np.log(D + eps), 0.0)
        # Augment with polynomial terms (constant + linear)
        P = np.hstack([np.ones((N, 1)), X])
        # Block system [Phi P ; P' 0] [w ; v] = [y ; 0]
        K = np.block([
            [Phi, P],
            [P.T, np.zeros((d + 1, d + 1))]
        ])
        rhs = np.concatenate([y, np.zeros(d + 1)])
        try:
            sol = np.linalg.solve(K + 1e-8 * np.eye(K.shape[0]), rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        return {'X': X, 'w': sol[:N], 'v': sol[N:]}

    @staticmethod
    def _eval_rbf(model, Xnew):
        Xc = model['X']
        D = np.sqrt(((Xnew[:, None, :] - Xc[None, :, :]) ** 2).sum(axis=2))
        eps = 1e-8
        Phi = np.where(D > eps, D ** 2 * np.log(D + eps), 0.0)
        P = np.hstack([np.ones((Xnew.shape[0], 1)), Xnew])
        return Phi @ model['w'] + P @ model['v']

    @handle_errors
    def predict(self, Xnew):
        """Return (mean, variance_proxy) at the new design vectors."""
        Xnew = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if self.use_smt and self.model is not None:
            try:
                mean = self.model.predict_values(Xnew).ravel()
                var = self.model.predict_variances(Xnew).ravel()
                return mean, var
            except Exception:
                pass
        # RBF fallback
        low_pred = self._eval_rbf(self._rbf_low, Xnew)
        delta_pred = self._eval_rbf(self._rbf_delta, Xnew)
        mean = low_pred + delta_pred
        # Variance proxy: distance to nearest high-fidelity sample
        if len(self.X_high) > 0:
            D = np.sqrt(((Xnew[:, None, :]
                          - self.X_high[None, :, :]) ** 2).sum(axis=2))
            var = np.min(D, axis=1) ** 2
        else:
            var = np.ones(Xnew.shape[0])
        return mean, var


# ============================================================================
# VARIABLE-FIDELITY OPTIMIZER  (NEW)
# ============================================================================
class VariableFidelityOptimizer:
    """Outer-loop Bayesian-optimisation-style driver.

    Pipeline:
        1. Generate a Latin-hypercube sample of n_low designs.
        2. Evaluate the low+mid fidelity model on all of them.
        3. Pick the n_high most-promising (by L/D) designs and 'run CFD'.
        4. Fit the cokriging surrogate.
        5. Use Expected-Improvement-style acquisition to pick the next
           high-fidelity evaluation; iterate.
    """

    def __init__(self, drone_class, n_low=20, n_high=5, n_iter=5,
                 seed=42, mesh_settings=None, surrogate_settings=None):
        self.drone_class = drone_class
        self.spec = DroneGeometryProperties.get(drone_class)
        self.n_low = int(n_low)
        self.n_high = int(n_high)
        self.n_iter = int(n_iter)
        self.rng = np.random.default_rng(seed)
        # Thread customization settings through to every solver instantiation.
        # Defaults reproduce the original behaviour.
        self.mesh_settings = mesh_settings if mesh_settings is not None \
            else DEFAULT_MESH_SETTINGS
        self.surrogate_settings = surrogate_settings if surrogate_settings is not None \
            else DEFAULT_SURR_SETTINGS
        self.surrogate = CokrigingSurrogate(
            n_low=n_low, n_high=n_high,
            surrogate_settings=self.surrogate_settings)
        self.history = []  # list of (x, y_low, y_high_or_None)
        self.high_fid_hook = HighFidelityHook()

    # ------------------------------------------------------------------
    def _sample_lhs(self, n):
        """Latin-hypercube sample of n design vectors within class bounds."""
        spec = self.spec
        d = 7  # 7 design variables
        # LHS: divide each axis into n equal strata, pick one per stratum
        cut = np.linspace(0, 1, n + 1)
        result = np.zeros((n, d))
        for j in range(d):
            perms = self.rng.permutation(n)
            for i in range(n):
                lo, hi = cut[perms[i]], cut[perms[i] + 1]
                result[i, j] = self.rng.uniform(lo, hi)
        # Map unit hypercube -> design space
        def map_axis(col, lo, hi):
            return lo + col * (hi - lo)
        X = np.column_stack([
            map_axis(result[:, 0], *spec['arm_length_range']),
            map_axis(result[:, 1], *spec['prop_diameter_range']),
            map_axis(result[:, 2], *spec['prop_pitch_range']),
            # blade_count is discrete: round to nearest option
            np.clip(np.round(result[:, 3] * (max(spec['blade_count_options']) - 1) + 1),
                    min(spec['blade_count_options']),
                    max(spec['blade_count_options'])),
            map_axis(result[:, 4], *spec['cruise_aoa_range']),
            map_axis(result[:, 5], *spec['cruise_speed_range']),
            map_axis(result[:, 6], *spec['prop_rpm_range']),
        ])
        # Ensure blade_count is one of the allowed options
        X[:, 3] = np.array([min(spec['blade_count_options'],
                                key=lambda b: abs(b - v)) for v in X[:, 3]])
        return X

    @staticmethod
    def _x_to_design(x):
        return {
            'arm_length': float(x[0]),
            'prop_diameter': float(x[1]),
            'prop_pitch': float(x[2]),
            'blade_count': int(round(x[3])),
            'cruise_aoa_deg': float(x[4]),
            'cruise_speed': float(x[5]),
            'prop_rpm': float(x[6]),
        }

    # ------------------------------------------------------------------
    @handle_errors
    def evaluate_low(self, x):
        """Low+mid fidelity evaluation: VLM + mid-fidelity corrector.

        Honours self.mesh_settings: n_panels_per_prop, horseshoe_length_factor,
        and vortex_core_radius.
        """
        design = self._x_to_design(x)
        errors, _ = DroneGeometryProperties.validate_design(
            self.drone_class, design)
        if errors:
            return None
        geom = FPVDroneGeometry(design, self.drone_class,
                                 n_panels_per_prop=self.mesh_settings.n_panels_per_prop,
                                 mesh_settings=self.mesh_settings)
        vlm = SpectralVLMSolver(geom, design,
                                 mesh_settings=self.mesh_settings).solve()
        mid = MidFidelityCorrector(
            vortex_core_radius=self.mesh_settings.vortex_core_radius
        ).correct(vlm, design, geom)
        return mid['L_D']

    @handle_errors
    def evaluate_high(self, x):
        """High-fidelity evaluation: external CFD hook (stub)."""
        design = self._x_to_design(x)
        result = self.high_fid_hook.evaluate(
            design, self.drone_class, mesh_settings=self.mesh_settings)
        if result is FAILED:
            return None
        return result['L_D']

    # ------------------------------------------------------------------
    @handle_errors
    def run(self, progress_callback=None, status_callback=None):
        """Run the full variable-fidelity optimisation.

        progress_callback(fraction_float)  - updates a UI progress bar.
        status_callback(msg_string)        - updates a UI status line.
        """
        # Phase 1: low-fidelity LHS sampling
        if status_callback:
            status_callback(f"Sampling {self.n_low} low-fidelity designs (LHS)...")
        X_low = self._sample_lhs(self.n_low)
        y_low = np.array([self.evaluate_low(x) for x in X_low])
        # Filter out None / NaN
        mask = ~np.isnan(y_low) & (y_low != None)
        mask = np.array([(y is not None and not np.isnan(y)) for y in y_low])
        X_low = X_low[mask]
        y_low = np.array([float(y) for y in y_low[mask]])
        if progress_callback:
            progress_callback(0.30)

        # Phase 2: pick top n_high candidates by L/D
        if status_callback:
            status_callback(f"Selecting {self.n_high} candidates for CFD...")
        order = np.argsort(y_low)[::-1]
        X_high = X_low[order[:self.n_high]].copy()
        y_high = np.array([self.evaluate_high(x) for x in X_high])
        y_high = np.array([float(y) if y is not None else np.nan
                           for y in y_high])
        if progress_callback:
            progress_callback(0.50)

        # Phase 3: fit surrogate
        if status_callback:
            status_callback("Fitting cokriging surrogate...")
        self.surrogate.fit(X_low, y_low, X_high, y_high)
        if progress_callback:
            progress_callback(0.60)

        # Phase 4: Bayesian-style iteration
        best_y = float(np.max(y_high))
        best_x = X_high[int(np.argmax(y_high))]
        self.history.append({
            'phase': 'initial_sampling',
            'X_low': X_low.copy(), 'y_low': y_low.copy(),
            'X_high': X_high.copy(), 'y_high': y_high.copy(),
            'best_x': best_x.copy(), 'best_y': best_y,
        })

        for it in range(self.n_iter):
            if status_callback:
                status_callback(f"Iter {it+1}/{self.n_iter}: "
                                f"acquiring next CFD point...")
            # Generate candidate pool by LHS
            X_cand = self._sample_lhs(200)
            # Predict mean + variance
            mean, var = self.surrogate.predict(X_cand)
            # Expected improvement (simple form):
            # EI = (mean - best_y) * Phi(z) + sigma * phi(z)
            # z = (mean - best_y) / sigma
            sigma = np.sqrt(np.maximum(var, 1e-12))
            z = (mean - best_y) / sigma
            # Normal CDF and PDF (numerical approximation)
            Phi = 0.5 * (1.0 + np.tanh(z / 1.4))  # crude but stable
            phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            EI = (mean - best_y) * Phi + sigma * phi
            # Pick the candidate with max EI
            next_idx = int(np.argmax(EI))
            next_x = X_cand[next_idx]
            next_y = self.evaluate_high(next_x)
            if next_y is None:
                continue
            # Augment the high-fidelity dataset
            X_high = np.vstack([X_high, next_x])
            y_high = np.append(y_high, float(next_y))
            if float(next_y) > best_y:
                best_y = float(next_y)
                best_x = next_x.copy()
            # Refit surrogate
            self.surrogate.fit(X_low, y_low, X_high, y_high)
            self.history.append({
                'phase': f'iter_{it+1}',
                'acquired_x': next_x.copy(),
                'acquired_y': float(next_y),
                'best_x': best_x.copy(),
                'best_y': best_y,
                'EI_max': float(EI[next_idx]),
            })
            if progress_callback:
                progress_callback(0.60 + 0.40 * (it + 1) / self.n_iter)

        if status_callback:
            status_callback(f"Optimisation complete. Best L/D = {best_y:.3f}")

        return {
            'best_x': best_x,
            'best_y': best_y,
            'best_design': self._x_to_design(best_x),
            'X_low': X_low,
            'y_low': y_low,
            'X_high': X_high,
            'y_high': y_high,
            'history': self.history,
        }


# ============================================================================
# VISUALIZER  (MODIFIED - drone-specific plots + retained heatmap scaffold)
# ============================================================================
class FPVDroneVisualizer:
    """Visualization utilities for the FPV drone optimisation results."""

    def __init__(self):
        pass

    @handle_errors
    def plot_design_space(self, X_low, y_low, X_high, y_high, dim_names=None,
                           fig_settings=None):
        """Pairwise scatter of the design space, coloured by L/D.

        Honours `fig_settings` for figsize / dpi / font size / colormap.
        Defaults reproduce the original 2*d x 2*d figure with the 'ld' cmap.
        """
        if dim_names is None:
            dim_names = ['arm_L', 'prop_D', 'prop_pitch', 'blades',
                         'AOA', 'V_cruise', 'RPM']
        fs = fig_settings if fig_settings is not None else DEFAULT_FIG_SETTINGS
        fs.apply_rc()
        d = X_low.shape[1]
        # Allow caller to override the per-cell size via figsize (interpret
        # the tuple as cell size, not total size).
        cell_w, cell_h = fs.figsize
        fig, axes = plt.subplots(d, d, figsize=(cell_w * d, cell_h * d))
        cmap = fs.resolve_cmap('ld') if fs.show_colorbar else \
            ColormapRegistry.resolve(fs.colormap_ld) if fs.colormap_ld else \
            _CUSTOM_COLORMAPS['ld']
        # Remember the FIRST scatter PathCollection we create, so we can
        # hand it directly to fig.colorbar() below. The previous code did:
        #     plt.colorbar(ax.collections[0] if ax.collections else
        #                  fig.axes[0].collections[0], ...)
        # which crashes with IndexError when the last visited `ax` is on
        # the diagonal (histogram -> no collections) AND the first axes
        # is also a histogram (no collections). The @handle_errors
        # decorator then turns that into a FAILED sentinel, and the
        # caller's st.pyplot(fig_d) dies with:
        #     AttributeError: '_Sentinel' object has no attribute 'savefig'
        scatter_mappable = None
        for i in range(d):
            for j in range(d):
                ax = axes[i, j]
                if i == j:
                    # Diagonal: histogram
                    ax.hist(X_low[:, i], bins=15, color='steelblue',
                            alpha=0.7)
                    ax.set_ylabel('count')
                elif i > j:
                    # Lower triangle: scatter low-fidelity
                    sc = ax.scatter(X_low[:, j], X_low[:, i],
                                    c=y_low, cmap=cmap, s=20, alpha=0.7,
                                    vmin=float(np.nanmin(y_low)),
                                    vmax=float(np.nanmax(y_low)))
                    if scatter_mappable is None:
                        scatter_mappable = sc
                    # Overlay high-fidelity points
                    ax.scatter(X_high[:, j], X_high[:, i],
                               c='red', s=60, marker='*',
                               edgecolor='black', linewidth=0.8,
                               label='CFD' if (i == 1 and j == 0) else None)
                else:
                    # Upper triangle: leave empty
                    ax.axis('off')
                if i == d - 1:
                    ax.set_xlabel(dim_names[j])
                if j == 0 and i != j:
                    ax.set_ylabel(dim_names[i])
        # Add a single colorbar using the remembered scatter mappable.
        if scatter_mappable is None:
            # Degenerate input (no lower-triangle scatters drawn).
            # Synthesise a blank ScalarMappable so the colorbar still renders
            # instead of crashing the whole tab.
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            scatter_mappable = ScalarMappable(
                norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
            scatter_mappable.set_array(np.array([0.0, 1.0]))
        if fs.show_colorbar:
            fig.subplots_adjust(right=0.92)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
            fig.colorbar(scatter_mappable, cax=cbar_ax, label='L/D')
        if any('CFD' in ax.get_legend_handles_labels()[1]
               for ax in fig.axes):
            fig.legend(loc='upper right', fontsize=fs.legend_size)
        # NOTE: do NOT call plt.tight_layout() here - it would override the
        # manual subplots_adjust(right=0.92) used to reserve room for the
        # colorbar, and would also collide with the manually-added cbar_ax.
        return fig

    @handle_errors
    def plot_convergence(self, history, fig_settings=None):
        """Plot best L/D vs. iteration.

        Honours `fig_settings.figsize`, `dpi`, font sizes, grid alpha.
        """
        fs = fig_settings if fig_settings is not None else DEFAULT_FIG_SETTINGS
        fs.apply_rc()
        iters = []
        best_so_far = []
        current_best = -np.inf
        for h in history:
            if 'best_y' in h:
                if h['best_y'] > current_best:
                    current_best = h['best_y']
                iters.append(len(iters) + 1)
                best_so_far.append(current_best)
        fig, ax = plt.subplots(figsize=fs.figsize)
        ax.plot(iters, best_so_far, 'o-', color='steelblue', lw=2, ms=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best L/D found')
        if fs.show_title:
            ax.set_title('Variable-fidelity optimisation convergence')
        if fs.show_grid:
            ax.grid(True, alpha=fs.grid_alpha)
        plt.tight_layout()
        return fig

    @handle_errors
    def plot_surrogate_surface(self, surrogate, fixed_var_idx=4,
                                fixed_values=None, n_grid=30,
                                fig_settings=None, surrogate_settings=None,
                                drone_class=None):
        """Heatmap of the surrogate L/D response over two chosen design
        variables, with the others held fixed at their class midpoints.

        Which two variables to sweep is controlled by `surrogate_settings`:
        - `sweep_var_x` (default 0 = arm_L) drives the X axis.
        - `sweep_var_y` (default 1 = prop_D) drives the Y axis.
        If both are equal, falls back to (0, 1) so the plot stays 2-D.

        Optional uncertainty band: if `surrogate_settings.show_uncertainty_band`
        is True, also overlays contour lines of ±sqrt(var) at 1σ.

        Parameters
        ----------
        surrogate : CokrigingSurrogate
            Already-fit surrogate whose predict() is called on the grid.
        fixed_var_idx : int
            Legacy parameter (ignored when surrogate_settings is given).
        fixed_values : array-like, optional
            7-vector of fixed design-variable values.  If None, the class
            midpoint of every non-swept variable is used.
        n_grid : int
            Grid resolution (ignored when surrogate_settings is given).
        fig_settings : FigureSettings
            Figsize, dpi, font sizes, colormap.
        surrogate_settings : SurrogateSettings
            sweep_var_x, sweep_var_y, n_grid, n_levels, show_*_points,
            show_uncertainty_band, colormap.
        drone_class : str
            FPV class name to look up the design-variable bounds.  Defaults
            to 'FPV 5"' for backward compatibility.
        """
        fs = fig_settings if fig_settings is not None else DEFAULT_FIG_SETTINGS
        fs.apply_rc()
        ss = surrogate_settings if surrogate_settings is not None \
            else DEFAULT_SURR_SETTINGS
        spec = DroneGeometryProperties.get(drone_class or 'FPV 5"')
        # The 7 design variables and their per-class bounds.
        ranges = [
            spec['arm_length_range'],
            spec['prop_diameter_range'],
            spec['prop_pitch_range'],
            (min(spec['blade_count_options']),
             max(spec['blade_count_options'])),
            spec['cruise_aoa_range'],
            spec['cruise_speed_range'],
            spec['prop_rpm_range'],
        ]
        # Decide which two variables to sweep.
        ix = int(ss.sweep_var_x)
        iy = int(ss.sweep_var_y)
        if ix == iy:
            ix, iy = 0, 1
        ng = int(ss.n_grid if surrogate_settings is not None else n_grid)
        # Default fixed values: class midpoints (+ smallest blade count).
        if fixed_values is None:
            fixed_values = np.array([
                np.mean(spec['arm_length_range']),
                np.mean(spec['prop_diameter_range']),
                np.mean(spec['prop_pitch_range']),
                float(spec['blade_count_options'][0]),
                np.mean(spec['cruise_aoa_range']),
                np.mean(spec['cruise_speed_range']),
                np.mean(spec['prop_rpm_range']),
            ])
        fixed_values = np.asarray(fixed_values, dtype=float).ravel()
        x0_lo, x0_hi = ranges[ix]
        x1_lo, x1_hi = ranges[iy]
        x0 = np.linspace(x0_lo, x0_hi, ng)
        x1 = np.linspace(x1_lo, x1_hi, ng)
        X0, X1 = np.meshgrid(x0, x1)
        # Build the full design-vector grid by tiling fixed_values and
        # overwriting the two swept columns.
        X_grid = np.tile(fixed_values, (X0.size, 1))
        X_grid[:, ix] = X0.ravel()
        X_grid[:, iy] = X1.ravel()
        mean, var = surrogate.predict(X_grid)
        Z = mean.reshape(ng, ng)
        sigma = np.sqrt(np.maximum(var, 0.0)).reshape(ng, ng)
        fig, ax = plt.subplots(figsize=fs.figsize)
        cmap = ColormapRegistry.resolve(ss.colormap,
                                          invert=fs.invert_colormap)
        levels = int(ss.n_levels)
        im = ax.contourf(X0, X1, Z, levels=levels, cmap=cmap)
        if fs.show_colorbar:
            fig.colorbar(im, ax=ax, label='Predicted L/D')
        # Optional ±1σ uncertainty band overlay.
        if ss.show_uncertainty_band:
            try:
                # Contour lines at the median and at +/-sigma.
                ax.contour(X0, X1, sigma, levels=5, colors='white',
                           linewidths=0.8, alpha=0.6)
                # Annotate the mean uncertainty on the plot.
                ax.text(0.02, 0.98,
                         f'mean ±σ ≈ {np.nanmean(sigma):.3f}',
                         transform=ax.transAxes, fontsize=fs.legend_size,
                         va='top', color='white',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='black', alpha=0.5))
            except Exception:
                pass
        # Overlay samples projected onto the swept plane.
        if ss.show_low_fidelity_points and surrogate.X_low is not None:
            ax.scatter(surrogate.X_low[:, ix], surrogate.X_low[:, iy],
                       c='white', s=20, edgecolor='black', linewidth=0.6,
                       label='Low-fidelity', zorder=5)
        if ss.show_high_fidelity_points and surrogate.X_high is not None:
            ax.scatter(surrogate.X_high[:, ix], surrogate.X_high[:, iy],
                       c='red', s=80, marker='*', edgecolor='black',
                       linewidth=0.8, label='High-fidelity (CFD)', zorder=6)
        ax.set_xlabel(SurrogateSettings.DIM_NAMES[ix])
        ax.set_ylabel(SurrogateSettings.DIM_NAMES[iy])
        if fs.show_title:
            ax.set_title(f'Cokriging surrogate response '
                         f'(sweep: {SurrogateSettings.DIM_NAMES[ix]} vs '
                         f'{SurrogateSettings.DIM_NAMES[iy]})')
        if fs.show_grid:
            ax.grid(True, alpha=fs.grid_alpha)
        ax.legend(loc='best', fontsize=fs.legend_size)
        plt.tight_layout()
        return fig


# ============================================================================
# DATA EXPORTERS  (KEPT verbatim from original)
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DataExporter:
    @staticmethod
    @handle_errors
    def export_pkl(results, name):
        buf = BytesIO()
        pickle.dump(results, buf)
        buf.seek(0)
        return buf, f"{name}.pkl"

    @staticmethod
    @handle_errors
    def export_json(results, name):
        s = json.dumps(results, indent=2, cls=NumpyEncoder)
        return BytesIO(s.encode()), f"{name}.json"

    @staticmethod
    @handle_errors
    def export_csv(X_low, y_low, X_high, y_high, name):
        import pandas as pd
        buf = BytesIO()
        cols = ['arm_L', 'prop_D', 'prop_pitch', 'blades',
                'AOA', 'V_cruise', 'RPM']
        df_low = pd.DataFrame(X_low, columns=cols)
        df_low['L_D_low'] = y_low
        df_low['fidelity'] = 'low'
        df_high = pd.DataFrame(X_high, columns=cols)
        df_high['L_D_high'] = y_high
        df_high['fidelity'] = 'high'
        df = pd.concat([df_low, df_high], ignore_index=True)
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf, f"{name}.csv"

    @staticmethod
    @handle_errors
    def export_hdf5(results, name):
        if not H5PY_AVAILABLE:
            return None, None
        buf = BytesIO()
        with h5py.File(buf, 'w') as f:
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    f.create_dataset(k, data=v)
                elif isinstance(v, (int, float, str)):
                    f.attrs[k] = v
        buf.seek(0)
        return buf, f"{name}.h5"


# ============================================================================
# STREAMLIT UI  (MODIFIED - drone-specific controls)
# ============================================================================
def main():
    st.set_page_config(
        page_title="FPV Drone Variable-Fidelity Optimizer",
        layout="wide", initial_sidebar_state="expanded")

    # ==============================================================
    # ROBUST SESSION-STATE INITIALIZATION  (Part 2, Fix 2)
    # --------------------------------------------------------------
    # Before any widget renders, make sure session_state holds the four
    # dataclass instances.  This prevents the fragile pattern where
    # settings defined deep inside a collapsed sidebar expander could
    # silently fall back to DEFAULT_* on certain rerun orderings.
    # The Settings tab (tabs[0]) rebuilds these from widget values on
    # every rerun; this initialization only kicks in on the very first
    # load (or after a session reset).
    if 'mesh_settings' not in st.session_state:
        st.session_state.mesh_settings = DEFAULT_MESH_SETTINGS
    if 'surrogate_settings' not in st.session_state:
        st.session_state.surrogate_settings = DEFAULT_SURR_SETTINGS
    if 'fig_settings' not in st.session_state:
        st.session_state.fig_settings = DEFAULT_FIG_SETTINGS
    if 'view_settings' not in st.session_state:
        st.session_state.view_settings = DEFAULT_VIEW_SETTINGS

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.1rem; text-align: center; margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #1b4f72, #7d3c98);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="main-header">FPV Drone Variable-Fidelity '
        'Aerodynamic Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#F0F8FF; padding:1rem; border-radius:8px;
                border-left:5px solid #1b4f72;">
    <b>Pipeline:</b> Low-fidelity (FFT-VLM) + Mid-fidelity (Lamb-Oseen + viscous)
    + High-fidelity (CFD hook) bridged by a <b>cokriging surrogate</b>.<br>
    &bull; <b>FFT operators</b> retained from the original spectral code.
    &bull; <b>VLM</b> discretises each propeller as a lifting line of horseshoe
    vortices; induced-velocity matrix built via Biot-Savart.
    &bull; <b>Mid-fidelity</b> adds flat-plate viscous drag + Lamb-Oseen vortex
    core decay.
    &bull; <b>High-fidelity</b> is a stub (perturbed mid-fidelity) so the full
    pipeline runs without a CFD licence.
    &bull; <b>Cokriging</b> uses SMT if installed, else a numpy RBF fallback.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================
    # SIDEBAR
    # ==============================================================
    with st.sidebar:
        st.header("Configuration")
        drone_class = st.selectbox(
            "FPV drone class",
            DroneGeometryProperties.list_options(),
            index=0,
            help="Selects the feasible design-variable envelope.")
        spec = DroneGeometryProperties.get(drone_class)

        st.subheader("Optimiser settings")
        n_low = st.slider("Low-fidelity samples (LHS)", 10, 50, 20, 2)
        n_high = st.slider("Initial high-fidelity (CFD) samples",
                            3, 15, 5, 1)
        n_iter = st.slider("Bayesian iterations", 2, 15, 5, 1)
        seed = st.number_input("Random seed", value=42, step=1)

        # NOTE: the old standalone "VLM discretisation" slider has moved into
        # the '🎨 Customization' expander below (Mesh settings section) so
        # there is a single source of truth for the panel count.

        st.subheader("Atmosphere")
        altitude_m = st.slider("Altitude (m)", 0, 5000, 0, 100)
        # Simple density model
        rho_air = RHO_AIR_SEA * np.exp(-altitude_m / 8500.0)
        st.caption(f"Air density = `{rho_air:.3f}` kg/m³")

        st.subheader("High-fidelity stub noise")
        cfd_noise = st.slider("CFD noise std", 0.0, 0.20, 0.05, 0.01,
                                help="Standard deviation of the random "
                                     "perturbation applied by the CFD stub.")

        # Single-design evaluator (sanity check before optimising)
        st.subheader("Single-design evaluator")
        arm_L = st.slider("arm_length (m)", *spec['arm_length_range'],
                            np.mean(spec['arm_length_range']), 0.005)
        prop_D = st.slider("prop_diameter (m)", *spec['prop_diameter_range'],
                            np.mean(spec['prop_diameter_range']), 0.005)
        prop_pitch = st.slider("prop_pitch (m)", *spec['prop_pitch_range'],
                                np.mean(spec['prop_pitch_range']), 0.002)
        blade_count = st.selectbox("blade_count",
                                    spec['blade_count_options'], index=0)
        cruise_aoa = st.slider("cruise AOA (deg)", *spec['cruise_aoa_range'],
                                0.0, 0.5)
        cruise_speed = st.slider("cruise speed (m/s)",
                                  *spec['cruise_speed_range'],
                                  np.mean(spec['cruise_speed_range']), 0.5)
        # NOTE: prop_rpm_range is stored as (int, int) in every drone class
        # spec, so min_value/max_value/step are all int. Streamlit's
        # st.slider requires the default value to share the same dtype,
        # therefore we must cast np.mean(...) -> int. Using a float default
        # here raises StreamlitInvalidParameterTypeError at app startup.
        prop_rpm = st.slider("prop RPM", *spec['prop_rpm_range'],
                              int(np.mean(spec['prop_rpm_range'])), 100)

        eval_design = {
            'arm_length': float(arm_L),
            'prop_diameter': float(prop_D),
            'prop_pitch': float(prop_pitch),
            'blade_count': int(blade_count),
            'cruise_aoa_deg': float(cruise_aoa),
            'cruise_speed': float(cruise_speed),
            'prop_rpm': float(prop_rpm),
        }

        errors, warns = DroneGeometryProperties.validate_design(
            drone_class, eval_design)
        for e in errors:
            st.error(e)
        for w in warns:
            st.warning(w)
        # NOTE: All customization controls (figure / geometry / mesh / surrogate)
        # are placed inline in expanders above their corresponding plots in
        # each tab (Geometry, Mesh, Surrogate, Convergence).  Settings are
        # rebuilt from st.session_state at the top of the main area, before
        # any tab renders, so every tab sees the latest values.

    # ==============================================================
    # MAIN AREA
    # ==============================================================
    # Customization controls are placed inline in expanders above their
    # corresponding plots (per user request).  Streamlit stores widget
    # values in st.session_state[key] immediately after the widget renders,
    # and these persist across reruns.  We rebuild the four settings
    # dataclasses HERE (before any tab renders) using st.session_state.get()
    # so that every tab sees the latest values.  On the very first run,
    # the keys don't exist yet and the .get() falls back to defaults.
    tabs = st.tabs(["Geometry", "Mesh", "3D CAD & Mesh", "Single Eval",
                    "Optimise", "Surrogate", "Convergence", "Export"])
    cmap_options = ColormapRegistry.list_all()

    # ---- Rebuild settings from session_state (survives across reruns) ----
    # Each .get(key, default) returns the widget's current value if the
    # widget has rendered at least once, otherwise the default.
    _ms_panels = st.session_state.get('ms_panels', DEFAULT_MESH_SETTINGS.n_panels_per_prop)
    _ms_helix  = st.session_state.get('ms_helix', DEFAULT_MESH_SETTINGS.n_helix_segments)
    _ms_core   = st.session_state.get('ms_core', DEFAULT_MESH_SETTINGS.vortex_core_radius)
    _ms_shoe   = st.session_state.get('ms_shoe', DEFAULT_MESH_SETTINGS.horseshoe_length_factor)
    _ms_spacing = st.session_state.get('ms_spacing', DEFAULT_MESH_SETTINGS.panel_spacing)
    _ms_wake   = st.session_state.get('ms_wake_cont', DEFAULT_MESH_SETTINGS.wake_contraction)
    mesh_settings = MeshSettings(
        n_panels_per_prop=_ms_panels,
        vortex_core_radius=_ms_core,
        horseshoe_length_factor=_ms_shoe,
        n_helix_segments=_ms_helix,
        panel_spacing=_ms_spacing,
        wake_contraction=_ms_wake)

    _ss_backend = st.session_state.get('ss_backend', DEFAULT_SURR_SETTINGS.backend)
    _ss_corr    = st.session_state.get('ss_corr', DEFAULT_SURR_SETTINGS.corr)
    _ss_poly    = st.session_state.get('ss_poly', DEFAULT_SURR_SETTINGS.poly)
    _ss_theta0  = st.session_state.get('ss_theta0', DEFAULT_SURR_SETTINGS.theta0)
    _ss_nugget  = st.session_state.get('ss_nugget', DEFAULT_SURR_SETTINGS.nugget)
    _ss_sweep_x = st.session_state.get('ss_sweep_x', (0, SurrogateSettings.DIM_NAMES[0]))
    _ss_sweep_y = st.session_state.get('ss_sweep_y', (1, SurrogateSettings.DIM_NAMES[1]))
    _ss_ngrid   = st.session_state.get('ss_ngrid', DEFAULT_SURR_SETTINGS.n_grid)
    _ss_nlev    = st.session_state.get('ss_nlev', DEFAULT_SURR_SETTINGS.n_levels)
    _ss_show_lo = st.session_state.get('ss_show_lo', DEFAULT_SURR_SETTINGS.show_low_fidelity_points)
    _ss_show_hi = st.session_state.get('ss_show_hi', DEFAULT_SURR_SETTINGS.show_high_fidelity_points)
    _ss_unc     = st.session_state.get('ss_unc', DEFAULT_SURR_SETTINGS.show_uncertainty_band)
    _ss_cmap    = st.session_state.get('ss_cmap', DEFAULT_SURR_SETTINGS.colormap)
    surrogate_settings = SurrogateSettings(
        backend=_ss_backend, corr=_ss_corr, poly=_ss_poly,
        theta0=float(_ss_theta0), nugget=float(_ss_nugget),
        sweep_var_x=int(_ss_sweep_x[0]) if isinstance(_ss_sweep_x, tuple) else int(_ss_sweep_x),
        sweep_var_y=int(_ss_sweep_y[0]) if isinstance(_ss_sweep_y, tuple) else int(_ss_sweep_y),
        n_grid=_ss_ngrid, n_levels=_ss_nlev,
        show_low_fidelity_points=_ss_show_lo,
        show_high_fidelity_points=_ss_show_hi,
        show_uncertainty_band=_ss_unc,
        colormap=_ss_cmap)

    _fs_w   = st.session_state.get('fig_w', DEFAULT_FIG_SETTINGS.figsize[0])
    _fs_h   = st.session_state.get('fig_h', DEFAULT_FIG_SETTINGS.figsize[1])
    _fs_dpi = st.session_state.get('fig_dpi', DEFAULT_FIG_SETTINGS.dpi)
    _fs_fs  = st.session_state.get('fs_font', DEFAULT_FIG_SETTINGS.font_size)
    _fs_leg = st.session_state.get('fs_leg', DEFAULT_FIG_SETTINGS.legend_size)
    _fs_cmap_ld = st.session_state.get('fs_cmap_ld', DEFAULT_FIG_SETTINGS.colormap_ld)
    _fs_cmap_vort = st.session_state.get('fs_cmap_vort', DEFAULT_FIG_SETTINGS.colormap_vort)
    _fs_cb  = st.session_state.get('fs_cb', DEFAULT_FIG_SETTINGS.show_colorbar)
    _fs_grid = st.session_state.get('fs_grid', DEFAULT_FIG_SETTINGS.show_grid)
    _fs_galpha = st.session_state.get('fs_galpha', DEFAULT_FIG_SETTINGS.grid_alpha)
    _fs_invert = st.session_state.get('fs_invert', DEFAULT_FIG_SETTINGS.invert_colormap)
    fig_settings = FigureSettings(
        figsize=(_fs_w, _fs_h), dpi=_fs_dpi,
        font_size=_fs_fs, legend_size=_fs_leg,
        colormap_ld=_fs_cmap_ld, colormap_vort=_fs_cmap_vort,
        show_colorbar=_fs_cb, show_grid=_fs_grid,
        grid_alpha=_fs_galpha, invert_colormap=_fs_invert)

    _gv_frame = st.session_state.get('gv_frame', DEFAULT_VIEW_SETTINGS.show_frame)
    _gv_arms  = st.session_state.get('gv_arms', DEFAULT_VIEW_SETTINGS.show_arms)
    _gv_disks = st.session_state.get('gv_disks', DEFAULT_VIEW_SETTINGS.show_prop_disks)
    _gv_blade = st.session_state.get('gv_blade', DEFAULT_VIEW_SETTINGS.show_blade_qc)
    _gv_cp    = st.session_state.get('gv_cp', DEFAULT_VIEW_SETTINGS.show_control_points)
    _gv_bv    = st.session_state.get('gv_bv', DEFAULT_VIEW_SETTINGS.show_bound_vortices)
    _gv_tv    = st.session_state.get('gv_tv', DEFAULT_VIEW_SETTINGS.show_trailing_vortices)
    _gv_pgrid = st.session_state.get('gv_pgrid', DEFAULT_VIEW_SETTINGS.show_panel_grid)
    _gv_batt  = st.session_state.get('gv_batt', DEFAULT_VIEW_SETTINGS.show_battery)
    _gv_cam   = st.session_state.get('gv_cam', DEFAULT_VIEW_SETTINGS.show_camera)
    _gv_labels = st.session_state.get('gv_labels', DEFAULT_VIEW_SETTINGS.show_arm_labels)
    _gv_da    = st.session_state.get('gv_da', DEFAULT_VIEW_SETTINGS.disk_alpha)
    _gv_ba    = st.session_state.get('gv_ba', DEFAULT_VIEW_SETTINGS.blade_alpha)
    _gv_ms    = st.session_state.get('gv_ms', DEFAULT_VIEW_SETTINGS.ms_control)
    _gv_c_frame = st.session_state.get('gv_c_frame', DEFAULT_VIEW_SETTINGS.color_frame)
    _gv_c_disk  = st.session_state.get('gv_c_disk', DEFAULT_VIEW_SETTINGS.color_prop_disk)
    _gv_c_cp    = st.session_state.get('gv_c_cp', DEFAULT_VIEW_SETTINGS.color_control_pts)
    _gv_c_bv    = st.session_state.get('gv_c_bv', DEFAULT_VIEW_SETTINGS.color_bound_vortex)
    _gv_c_tv    = st.session_state.get('gv_c_tv', DEFAULT_VIEW_SETTINGS.color_trailing_vortex)
    _gv_c_batt  = st.session_state.get('gv_c_batt', DEFAULT_VIEW_SETTINGS.color_battery)
    _gv_c_cam   = st.session_state.get('gv_c_cam', DEFAULT_VIEW_SETTINGS.color_camera)
    view_settings = GeometryViewSettings(
        show_frame=_gv_frame, show_arms=_gv_arms,
        show_prop_disks=_gv_disks, show_blade_qc=_gv_blade,
        show_prop_panels=_gv_cp, show_control_points=_gv_cp,
        show_bound_vortices=_gv_bv, show_trailing_vortices=_gv_tv,
        show_panel_grid=_gv_pgrid,
        show_battery=_gv_batt, show_camera=_gv_cam,
        show_arm_labels=_gv_labels,
        color_frame=_gv_c_frame, color_prop_disk=_gv_c_disk,
        color_blade_qc=_gv_c_disk, color_control_pts=_gv_c_cp,
        color_bound_vortex=_gv_c_bv, color_trailing_vortex=_gv_c_tv,
        color_battery=_gv_c_batt, color_camera=_gv_c_cam,
        disk_alpha=_gv_da, blade_alpha=_gv_ba, ms_control=_gv_ms)
    # Persist to session_state (defensive: in case a tab checks session_state
    # directly instead of using the local variable).
    st.session_state['mesh_settings']      = mesh_settings
    st.session_state['surrogate_settings'] = surrogate_settings
    st.session_state['fig_settings']      = fig_settings
    st.session_state['view_settings']     = view_settings

    # ----- Geometry tab -----
    with tabs[0]:
        st.header("Drone Geometry (top view)")
        # ---- Geometry View controls (inline, above the plot) ----
        with st.expander("🎨 Geometry view settings", expanded=False):
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                show_frame  = st.checkbox("Show frame", True, key="gv_frame")
                show_arms   = st.checkbox("Show arms", True, key="gv_arms")
                show_disks  = st.checkbox("Show prop disks", True, key="gv_disks")
                show_blade  = st.checkbox("Show blade q-c", True, key="gv_blade")
                show_cp     = st.checkbox("Show control points", True, key="gv_cp")
                show_bv     = st.checkbox("Show bound vortices", False, key="gv_bv")
                show_tv     = st.checkbox("Show trailing vortices", False, key="gv_tv")
            with col_g2:
                show_pgrid  = st.checkbox("Show panel grid", False, key="gv_pgrid")
                show_batt   = st.checkbox("Show battery", True, key="gv_batt")
                show_cam    = st.checkbox("Show camera", True, key="gv_cam")
                show_labels = st.checkbox("Show arm labels", True, key="gv_labels")
                disk_alpha  = st.slider("Disk alpha", 0.0, 1.0, 0.25, 0.05, key="gv_da")
                blade_alpha = st.slider("Blade alpha", 0.0, 1.0, 1.0, 0.05, key="gv_ba")
                ms_control  = st.slider("Control-point size", 2.0, 12.0, 5.0, 0.5, key="gv_ms")
            with col_g3:
                col_frame   = st.color_picker("Frame colour", "#111111", key="gv_c_frame")
                col_disk    = st.color_picker("Prop-disk colour", "#1f77b4", key="gv_c_disk")
                col_cp      = st.color_picker("Control-point colour", "#d62728", key="gv_c_cp")
                col_bv      = st.color_picker("Bound-vortex colour", "#2ca02c", key="gv_c_bv")
                col_tv      = st.color_picker("Trailing-vortex colour", "#9467bd", key="gv_c_tv")
                col_batt    = st.color_picker("Battery colour", "#2ca02c", key="gv_c_batt")
                col_cam     = st.color_picker("Camera colour", "#ff7f0e", key="gv_c_cam")
        geom = FPVDroneGeometry(eval_design, drone_class,
                                 n_panels_per_prop=mesh_settings.n_panels_per_prop,
                                 mesh_settings=mesh_settings)
        fig = geom.plot_top_view(view_settings=view_settings,
                                  fig_settings=fig_settings)
        if fig is FAILED:
            st.error("Geometry plot failed - see app logs.")
        else:
            st.pyplot(fig)
            plt.close(fig)
        st.caption(f"Total lifting area: {geom.meta['total_lift_area']:.4f} "
                   f"m² | Total drag area: "
                   f"{geom.meta['total_drag_area']:.4f} m² | "
                   f"VLM panels: {geom.meta['total_panels']}")

    # ----- Mesh tab -----
    with tabs[1]:
        st.header("VLM mesh visualization")
        # ---- Mesh/VLM controls (inline, above the plot) ----
        with st.expander("🔧 Mesh & VLM solver settings", expanded=True):
            st.caption("These parameters control the vortex-lattice discretisation, "
                       "helical wake model, and Biot-Savart regularization. "
                       "**Blade count** is read from the sidebar's single-design evaluator.")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                n_panels  = st.slider("Panels per propeller blade", 4, 32, 20, 1,
                                        key="ms_panels",
                                        help="VLM typically requires 15-20+ panels "
                                             "for converged induced drag.")
                n_helix   = st.slider("Helical wake segments", 1, 20, 8, 1,
                                        key="ms_helix",
                                        help="Trailing-vortex helix discretisation. "
                                             "1 = simplified single-segment helix; "
                                             "8+ = proper curved helical wake.")
                spacing   = st.selectbox("Panel spacing", ['cosine', 'uniform'],
                                           index=0, key="ms_spacing",
                                           help="Cosine clusters panels near root "
                                                "and tip for better tip-vortex resolution.")
            with col_m2:
                core_rad  = st.slider("Lamb-Oseen core r (m)", 0.001, 0.20, 0.05,
                                        0.001, key="ms_core",
                                        help="Regularizes the 1/r singularity in "
                                             "Biot-Savart when control points are "
                                             "close to vortex filaments.")
                shoe_fac  = st.slider("Horseshoe length (× chord)", 10.0, 200.0,
                                        50.0, 5.0, key="ms_shoe")
                wake_cont = st.slider("Wake contraction", 0.0, 0.40, 0.15, 0.01,
                                        key="ms_wake_cont",
                                        help="Radial wake contraction ratio. "
                                             "0.0 = none, 0.15 = 15% contraction "
                                             "(real propeller wakes contract).")
        st.caption("Horseshoe vortices (bound segment + **helical** trailing "
                   "legs with wake contraction) and 3/4-chord control points.")
        geom = FPVDroneGeometry(eval_design, drone_class,
                                 n_panels_per_prop=mesh_settings.n_panels_per_prop,
                                 mesh_settings=mesh_settings)
        fig = geom.plot_mesh(view_settings=view_settings,
                              fig_settings=fig_settings,
                              mesh_settings=mesh_settings,
                              design=eval_design)
        if fig is FAILED:
            st.error("Mesh plot failed - see app logs.")
        else:
            st.pyplot(fig)
            plt.close(fig)
        # Mesh summary
        st.subheader("Mesh summary")
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Panels / blade", mesh_settings.n_panels_per_prop)
        with mcol2:
            st.metric("Total panels", geom.meta['total_panels'])
        with mcol3:
            st.metric("Lamb-Oseen core (m)",
                       f"{mesh_settings.vortex_core_radius:.4f}")
        with mcol4:
            st.metric("Horseshoe factor",
                       f"{mesh_settings.horseshoe_length_factor:.1f}×")
        with mcol5:
            st.metric("Panel spacing", mesh_settings.panel_spacing)

    # ----- 3D CAD & Mesh tab (NEW) -----
    with tabs[2]:
        st.header("Publication-Grade 3D Mesh Generation")
        st.markdown("""
        This module uses the **Gmsh Python OCC API** to generate a true 3D
        volumetric mesh with NACA 4412 airfoil sections, geometric twist,
        taper, an MRF (Multiple Reference Frame) cylinder around each rotor,
        and a prism **boundary layer** on the blade surfaces — matching the
        methodology of Qiao et al. (2024).
        """)

        if not GMSH_AVAILABLE:
            st.error("⚠️ `gmsh` and `pyvista` are not available. "
                     "This is usually caused by either:")
            st.markdown(
                "1. **Python packages not installed** — run "
                "`pip install gmsh pyvista`\n"
                "2. **Missing system libraries** (libGLU.so.1, etc.) — "
                "on Streamlit Cloud, create a `.streamlit/packages.txt` "
                "file containing: `libgl1`, `libglu1-mesa`, `libxrender1`, "
                "`libxext6`, `libsm6`, `libx11-6`, `libgomp1`")
            st.info("The 2D VLM mesh in the previous tab works without "
                    "Gmsh.  Install Gmsh + PyVista (and the system "
                    "libraries) to unlock the 3D publication-grade mesh "
                    "generator.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Mesh Resolution")
                finesse = st.radio(
                    "Select fidelity tier", ["coarse", "fine"],
                    help="Coarse: y+≈5, Fine: y+≈0.3 "
                         "(matches paper Table 2)")
                st.caption(f"Design: {drone_class} | "
                           f"Prop: {eval_design['prop_diameter']:.3f}m | "
                           f"Pitch: {eval_design['prop_pitch']:.3f}m")
                # Show the preset details
                if finesse == "coarse":
                    st.caption("hwall=6.78e-5, bl_thick=8mm, growth=1.25, "
                               "h_blade=3mm, h_far=0.25m")
                else:
                    st.caption("hwall=4.07e-6, bl_thick=6mm, growth=1.13, "
                               "h_blade=1.2mm, h_far=0.12m")

            with col2:
                st.subheader("Actions")
                if st.button("🚀 Generate 3D Mesh", type="primary",
                             use_container_width=True):
                    with st.spinner("Running Gmsh OCC kernel and "
                                    "generating boundary layer mesh..."):
                        try:
                            import os, tempfile
                            # Use a temp directory so we don't pollute cwd
                            tmp_dir = tempfile.mkdtemp(prefix="fpv_mesh_")
                            prefix = os.path.join(tmp_dir, "fpv_demo")
                            vtk_path, msh_path, n_cells = \
                                FPVDrone3DMeshGenerator.generate_mesh(
                                    eval_design, drone_class,
                                    output_prefix=prefix,
                                    finesse=finesse)
                            st.session_state['mesh_vtk_path'] = vtk_path
                            st.session_state['mesh_msh_path'] = msh_path
                            st.session_state['mesh_ncells'] = n_cells
                            st.success(f"✅ Mesh generated! "
                                       f"({n_cells:,} cells)")
                        except Exception as e:
                            st.error(f"Mesh generation failed: {e}")
                            st.code(traceback.format_exc())

            # Display mesh diagnostics if available
            if 'mesh_ncells' in st.session_state:
                st.subheader("Mesh Diagnostics")
                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    st.metric("3D cells", f"{st.session_state['mesh_ncells']:,}")
                with dcol2:
                    st.metric("Fidelity", finesse)
                with dcol3:
                    yplus = "~5" if finesse == "coarse" else "~0.3"
                    st.metric("Target y+", yplus)

            # 3D Visualization
            if 'mesh_vtk_path' in st.session_state:
                st.subheader("3D Mesh Visualization")
                st.caption("Clipped cross-section showing boundary layer "
                           "prism cells on the blade surfaces.")
                try:
                    mesh_pv = pv.read(st.session_state['mesh_vtk_path'])
                    # Clip to show internal structure
                    clipped = mesh_pv.clip(
                        normal=(0, 0, 1), origin=(0, 0, 0.01),
                        invert=True)
                    plotter = pv.Plotter(off_screen=True,
                                          window_size=(800, 600))
                    plotter.add_mesh(
                        clipped, show_edges=True, color="lightgrey",
                        edge_color="black", line_width=0.5, opacity=0.9)
                    plotter.view_xy()
                    img_path = os.path.join(
                        os.path.dirname(st.session_state['mesh_vtk_path']),
                        "mesh_preview.png")
                    plotter.screenshot(img_path)
                    st.image(img_path,
                             caption="Clipped cross-section: "
                                     "boundary layer prism cells",
                             use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization failed: {e}")
                    st.code(traceback.format_exc())

                # Download buttons
                st.subheader("Download Mesh Files")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    try:
                        with open(st.session_state['mesh_msh_path'], "rb") as f:
                            st.download_button(
                                "📥 Download .msh (for SU2/OpenFOAM)",
                                f, file_name="fpv_mesh.msh",
                                mime="application/octet-stream")
                    except Exception as e:
                        st.error(f"Failed to read .msh: {e}")
                with dl_col2:
                    try:
                        with open(st.session_state['mesh_vtk_path'], "rb") as f:
                            st.download_button(
                                "📥 Download .vtk (for ParaView)",
                                f, file_name="fpv_mesh.vtk",
                                mime="application/octet-stream")
                    except Exception as e:
                        st.error(f"Failed to read .vtk: {e}")

    # ----- Single Eval tab -----
    with tabs[3]:
        st.header("Single-design aerodynamic evaluation")
        if st.button("Evaluate low + mid fidelity"):
            geom = FPVDroneGeometry(eval_design, drone_class,
                                     n_panels_per_prop=mesh_settings.n_panels_per_prop,
                                     mesh_settings=mesh_settings)
            vlm = SpectralVLMSolver(geom, eval_design, rho=rho_air,
                                     mesh_settings=mesh_settings)
            vlm_res = vlm.solve()
            mid = MidFidelityCorrector(
                vortex_core_radius=mesh_settings.vortex_core_radius
            ).correct(vlm_res, eval_design, geom)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VLM L/D", f"{vlm_res['L_D']:.3f}")
                st.metric("VLM CL", f"{vlm_res['CL']:.3f}")
                st.metric("VLM CD", f"{vlm_res['CD']:.4f}")
            with col2:
                st.metric("Mid L/D", f"{mid['L_D']:.3f}")
                st.metric("Mid CL", f"{mid['CL']:.3f}")
                st.metric("Mid CD", f"{mid['CD']:.4f}")
            with col3:
                st.metric("Blade Re", f"{mid['Re_blade']:.2e}")
                st.metric("Cf (turb)", f"{mid['Cf']:.4f}")
                st.metric("Thrust corr", f"{mid['thrust_corr_factor']:.3f}")
            # Gamma distribution plot - honours fig_settings.colormap_vort
            fig_settings.apply_rc()
            fig, ax = plt.subplots(figsize=(10, 4))
            cmap_vort = fig_settings.resolve_cmap('vort')
            n_arms = geom.n_arms
            cmap_colors = [cmap_vort(i / max(n_arms - 1, 1))
                            for i in range(n_arms)]
            for i, pp in enumerate(geom.prop_panels):
                g = vlm_res['gamma_distribution'][
                    i * mesh_settings.n_panels_per_prop:
                    (i + 1) * mesh_settings.n_panels_per_prop]
                r = pp['r_mid']
                ax.plot(r, g, '-o', color=cmap_colors[i],
                         label=f'Arm {i+1}', ms=4)
            ax.set_xlabel('radial station r (m)')
            ax.set_ylabel(r'$\Gamma$ (m²/s)')
            if fig_settings.show_title:
                ax.set_title('Bound-circulation distribution per blade')
            if fig_settings.show_grid:
                ax.grid(True, alpha=fig_settings.grid_alpha)
            ax.legend(fontsize=fig_settings.legend_size)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ----- Optimise tab -----
    with tabs[4]:
        st.header("Variable-fidelity optimisation")
        if st.button("Run optimisation", type="primary"):
            opt = VariableFidelityOptimizer(
                drone_class, n_low=n_low, n_high=n_high,
                n_iter=n_iter, seed=int(seed),
                mesh_settings=mesh_settings,
                surrogate_settings=surrogate_settings)
            opt.high_fid_hook = HighFidelityHook(cfd_noise=cfd_noise,
                                                  seed=int(seed))
            progress = st.progress(0)
            status = st.empty()
            result = opt.run(
                progress_callback=lambda f: progress.progress(f),
                status_callback=lambda msg: status.text(msg))
            if result is FAILED or result is None:
                st.error("Optimisation failed.")
            else:
                st.session_state.opt_result = result
                st.success(f"Best L/D = {result['best_y']:.3f}")
                st.json(result['best_design'])

    # ----- Surrogate tab -----
    with tabs[5]:
        st.header("Cokriging surrogate")
        # ---- Surrogate controls (inline, above the plots) ----
        with st.expander("📊 Surrogate settings", expanded=True):
            smt_status = "✅ available" if SMT_AVAILABLE else "❌ not installed"
            st.caption(f"_smt package: **{smt_status}**_")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**Model**")
                backend = st.selectbox("Backend", ['auto', 'smt', 'numpy'],
                                         index=0, key="ss_backend")
                corr    = st.selectbox("Correlation kernel (SMT only)",
                                         ['squar_exp', 'abs_exp', 'matern32'],
                                         index=0, key="ss_corr")
                poly    = st.selectbox("Polynomial basis (SMT only)",
                                         ['constant', 'linear'],
                                         index=0, key="ss_poly")
                theta0  = st.number_input("theta0 (hyperparameter)",
                                            value=1e-2, format="%.2e",
                                            step=1e-3, key="ss_theta0")
                nugget  = st.number_input("Nugget (regularization)",
                                            value=1e-8, format="%.2e",
                                            step=1e-9, key="ss_nugget")
            with col_s2:
                st.markdown("**Response-surface plotting**")
                dim_pairs = list(enumerate(SurrogateSettings.DIM_NAMES))
                sweep_x = st.selectbox("Sweep variable (X axis)", dim_pairs,
                                         format_func=lambda x: x[1], index=0,
                                         key="ss_sweep_x")
                sweep_y = st.selectbox("Sweep variable (Y axis)", dim_pairs,
                                         format_func=lambda x: x[1], index=1,
                                         key="ss_sweep_y")
                n_grid  = st.slider("Grid resolution", 10, 80, 30, 5, key="ss_ngrid")
                n_lev   = st.slider("Contour levels", 5, 40, 20, 1, key="ss_nlev")
                show_lo = st.checkbox("Show low-fidelity samples", True, key="ss_show_lo")
                show_hi = st.checkbox("Show high-fidelity samples", True, key="ss_show_hi")
                show_unc = st.checkbox("Overlay ±σ uncertainty band", False, key="ss_unc")
                surr_cmap = st.selectbox("Response-surface colormap",
                                            cmap_options,
                                            index=cmap_options.index('ld'),
                                            key="ss_cmap")
        if 'opt_result' not in st.session_state:
            st.info("Run the optimisation first.")
        else:
            result = st.session_state.opt_result
            vis = FPVDroneVisualizer()
            ix, iy = (surrogate_settings.sweep_var_x,
                       surrogate_settings.sweep_var_y)
            if ix == iy:
                ix, iy = 0, 1
            st.subheader(f"Response surface ({SurrogateSettings.DIM_NAMES[ix]} "
                         f"vs {SurrogateSettings.DIM_NAMES[iy]})")
            surr = CokrigingSurrogate(
                n_low=result['X_low'].shape[0],
                n_high=result['X_high'].shape[0],
                surrogate_settings=surrogate_settings)
            surr.fit(result['X_low'], result['y_low'],
                      result['X_high'], result['y_high'])
            fig_s = vis.plot_surrogate_surface(
                surr, fig_settings=fig_settings,
                surrogate_settings=surrogate_settings,
                drone_class=drone_class)
            if fig_s is FAILED:
                st.error("Surrogate surface plot failed - see app logs.")
            else:
                st.pyplot(fig_s)
                plt.close(fig_s)
            # Design-space scatter
            st.subheader("Design-space samples")
            fig_d = vis.plot_design_space(
                result['X_low'], result['y_low'],
                result['X_high'], result['y_high'],
                fig_settings=fig_settings)
            if fig_d is FAILED:
                st.error("Design-space plot failed - see app logs.")
            else:
                st.pyplot(fig_d)
                plt.close(fig_d)
            # Surrogate diagnostics
            st.subheader("Surrogate diagnostics")
            diag_col1, diag_col2, diag_col3 = st.columns(3)
            with diag_col1:
                st.metric("Backend used",
                           'smt (MFK)' if surr.use_smt else 'numpy RBF')
            with diag_col2:
                st.metric("Kernel",
                           surrogate_settings.corr if surr.use_smt
                           else 'thin-plate-spline')
            with diag_col3:
                st.metric("# Low / # High samples",
                           f"{len(result['X_low'])} / "
                           f"{len(result['X_high'])}")

    # ----- Convergence tab -----
    with tabs[6]:
        st.header("Optimisation convergence")
        # ---- Figure appearance controls (inline, above the plot) ----
        with st.expander("🎨 Figure appearance (applies to all plots)", expanded=False):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                fig_width  = st.slider("Figure width (in)", 4, 16, 8, 1, key="fig_w")
                fig_height = st.slider("Figure height (in)", 3, 12, 6, 1, key="fig_h")
                fig_dpi    = st.slider("DPI", 60, 200, 100, 10, key="fig_dpi")
                font_size  = st.slider("Font size", 6, 18, 10, 1, key="fs_font")
                legend_sz  = st.slider("Legend size", 6, 14, 8, 1, key="fs_leg")
            with col_f2:
                cmap_ld   = st.selectbox("Colormap (L/D)", cmap_options,
                                           index=cmap_options.index('ld'), key="fs_cmap_ld")
                cmap_vort = st.selectbox("Colormap (vortex)", cmap_options,
                                           index=cmap_options.index('vort'), key="fs_cmap_vort")
                show_cb   = st.checkbox("Show colorbar", True, key="fs_cb")
                show_grid = st.checkbox("Show grid", True, key="fs_grid")
                grid_alpha = st.slider("Grid alpha", 0.0, 1.0, 0.3, 0.05, key="fs_galpha")
                invert_cm = st.checkbox("Invert colormap", False, key="fs_invert")
        if 'opt_result' not in st.session_state:
            st.info("Run the optimisation first.")
        else:
            result = st.session_state.opt_result
            vis = FPVDroneVisualizer()
            fig = vis.plot_convergence(result['history'],
                                         fig_settings=fig_settings)
            if fig is FAILED:
                st.error("Convergence plot failed - see app logs.")
            else:
                st.pyplot(fig)
                plt.close(fig)
            # History table
            st.subheader("Iteration history")
            hist_data = []
            for h in result['history']:
                row = {'phase': h.get('phase', '?')}
                if 'best_y' in h:
                    row['best_L_D'] = h['best_y']
                if 'acquired_y' in h:
                    row['acquired_L_D'] = h['acquired_y']
                if 'EI_max' in h:
                    row['max_EI'] = h['EI_max']
                hist_data.append(row)
            st.table(hist_data)

    # ----- Export tab -----
    with tabs[7]:
        st.header("Export results")
        if 'opt_result' not in st.session_state:
            st.info("Run the optimisation first.")
        else:
            result = st.session_state.opt_result
            name = (f"fpv_drone_opt_{drone_class.replace(' ', '').replace('\"', 'in')}"
                    f"_n{result['X_low'].shape[0]}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Pickle"):
                    buf, fn = DataExporter.export_pkl(
                        {'X_low': result['X_low'], 'y_low': result['y_low'],
                         'X_high': result['X_high'], 'y_high': result['y_high'],
                         'best_x': result['best_x'],
                         'best_y': result['best_y']}, name)
                    if buf:
                        st.download_button("Download .pkl", buf, fn)
            with col2:
                if st.button("JSON"):
                    buf, fn = DataExporter.export_json(
                        {'best_design': result['best_design'],
                         'best_y': float(result['best_y'])}, name)
                    if buf:
                        st.download_button("Download .json", buf, fn)
            with col3:
                if st.button("CSV"):
                    buf, fn = DataExporter.export_csv(
                        result['X_low'], result['y_low'],
                        result['X_high'], result['y_high'], name)
                    if buf:
                        st.download_button("Download .csv", buf, fn)


if __name__ == "__main__":
    main()
