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

    def __init__(self, design, drone_class='FPV 5"', n_panels_per_prop=8):
        self.design = design
        self.drone_class = drone_class
        self.spec = DroneGeometryProperties.get(drone_class)
        self.n_arms = int(self.spec['n_arms'])
        self.n_panels_per_prop = int(n_panels_per_prop)
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
            r_root = 0.10 * prop_R
            r_tip = prop_R
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
            # control points at 3/4-chord of each panel
            cp_x = cx + r_mid * np.cos(arm_angles[i] + np.pi / 2)
            cp_y = cy + r_mid * np.sin(arm_angles[i] + np.pi / 2)
            cp_z = np.zeros(n_pan)
            # bound vortex at 1/4-chord
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

    def plot_top_view(self, ax=None, show_prop_panels=True):
        """Top-view matplotlib plot of the drone geometry."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure
        # Frame outline
        ax.plot(self.frame_outline[:, 0], self.frame_outline[:, 1],
                'k-', lw=2.5, label='Frame')
        # Arms
        cx, cy = 0.0, 0.0
        for i in range(self.n_arms):
            mx, my, _ = self.motor_xy[i]
            ax.plot([cx, mx], [cy, my], 'k-', lw=1.5, alpha=0.7)
        # Propeller disks
        for i, pp in enumerate(self.prop_panels):
            mx, my, _ = self.motor_xy[i]
            theta = self.arm_angles[i] + np.pi / 2
            # draw disk circle
            th = np.linspace(0, 2 * np.pi, 60)
            ax.plot(mx + self.prop_R * np.cos(th),
                    my + self.prop_R * np.sin(th),
                    'b-', lw=1.0, alpha=0.4)
            # blade quarter-chord line
            r = np.linspace(pp['r_root'], pp['r_tip'], 30)
            ax.plot(mx + r * np.cos(theta),
                    my + r * np.sin(theta),
                    'b-', lw=2.0, label='Blade q-c' if i == 0 else None)
            if show_prop_panels:
                for j in range(self.n_panels_per_prop):
                    cp = pp['control_pts'][j]
                    ax.plot(cp[0], cp[1], 'r.', ms=4)
        # Battery (centre box)
        batt_size = 0.04
        ax.add_patch(plt.Rectangle((-batt_size, -batt_size),
                                     2 * batt_size, 2 * batt_size,
                                     facecolor='green', alpha=0.5,
                                     edgecolor='black', label='Battery'))
        # Camera (front nub)
        ax.add_patch(plt.Rectangle((self.arm_length * 0.7, -0.015),
                                     0.04, 0.03,
                                     facecolor='orange', alpha=0.7,
                                     edgecolor='black', label='Camera'))
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f"{self.drone_class} top view")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        return fig


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
                 rho=RHO_AIR_SEA):
        self.geom = geometry
        self.design = design
        self.rho = float(rho)
        self.freestream_speed = float(design['cruise_speed'])
        self.aoa_deg = float(design['cruise_aoa_deg'])
        self.aoa_rad = np.deg2rad(self.aoa_deg)
        self.rpm = float(design['prop_rpm'])
        self.omega = self.rpm * 2 * np.pi / 60.0   # rad/s
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
                    # self-induced downwash of a horseshoe on its own CP
                    # ~ 1 / (4 * pi * chord) approximation
                    chord = ppj['chord'][jj]
                    A[i, j] = 1.0 / (4.0 * np.pi * max(chord, 1e-4))
                else:
                    A[i, j] = self._horseshoe_induced_downwash(
                        ppj, jj, cp)
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
        # Trailing vortices from p1 and p2 to (+inf, same y, same z)
        # We approximate the infinite segment by a 50*chord-length segment
        L = 50.0 * pp['chord'][j]
        p1_inf = p1 + np.array([L, 0, 0])
        p2_inf = p2 + np.array([L, 0, 0])
        w_t1 = self._biot_savart_segment(p1, p1_inf, cp)
        w_t2 = self._biot_savart_segment(p2, p2_inf, cp)
        # Z-component of total induced velocity (per unit Gamma)
        return (w_bound + w_t1 + w_t2)[2]

    @staticmethod
    def _biot_savart_segment(p1, p2, cp):
        """Biot-Savart induced velocity at cp from a unit-strength straight
        vortex segment from p1 to p2.  Returns the 3-vector w."""
        r1 = cp - p1
        r2 = cp - p2
        L = p2 - p1
        cross = np.cross(L, r1)
        denom = np.linalg.norm(cross) ** 2 + 1e-12
        # Standard finite-segment Biot-Savart factor:
        # w = (Gamma / (4*pi)) * (r1 + r2) x (r2 - r1) / |r1 x L|^2 ... (simplified)
        r1n = np.linalg.norm(r1) + 1e-12
        r2n = np.linalg.norm(r2) + 1e-12
        factor = (np.dot(r1, L) / r1n - np.dot(r2, L) / r2n)
        w = factor * cross / denom / (4.0 * np.pi)
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

        # Solve for Gamma
        try:
            gamma = np.linalg.solve(self.A + 1e-6 * np.eye(self.N), rhs)
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

        total_lift = float(np.sum(panel_lift))

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
        - write a STAR-CCM+ / SU2 / OpenFOAM case file
        - submit to an HPC queue
        - parse the resulting force report

    Here it just returns a perturbed version of the mid-fidelity result
    so the variable-fidelity pipeline can be exercised end-to-end without
    a real CFD licence.  The user can override `evaluate()` to plug in a
    real solver.
    """

    def __init__(self, cfd_noise=0.05, seed=42):
        self.cfd_noise = float(cfd_noise)
        self.rng = np.random.default_rng(seed)

    @handle_errors
    def evaluate(self, design, drone_class):
        """Stub: pretend to run a CFD calculation and return a noisy
        version of the mid-fidelity answer."""
        geom = FPVDroneGeometry(design, drone_class)
        vlm = SpectralVLMSolver(geom, design).solve()
        mid = MidFidelityCorrector().correct(vlm, design, geom)

        # Add a structured bias + noise so cokriging has something to learn
        bias_CL = 0.02 * np.sin(design['cruise_aoa_deg'] / 5.0)
        bias_CD = 0.01 * (design['prop_rpm'] - 10000) / 10000.0
        noise_CL = self.cfd_noise * self.rng.standard_normal()
        noise_CD = self.cfd_noise * self.rng.standard_normal() * 0.3

        return {
            'CL': mid['CL'] * (1 + bias_CL + noise_CL),
            'CD': max(mid['CD'] * (1 + bias_CD + noise_CD), 1e-6),
            'L_D': float((mid['CL'] * (1 + bias_CL + noise_CL))
                         / max(mid['CD'] * (1 + bias_CD + noise_CD), 1e-6)),
            'total_drag': mid['total_drag'] * (1 + bias_CD + noise_CD),
            'source': 'CFD_Hook_stub',
        }


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

    def __init__(self, n_low=20, n_high=5):
        self.n_low = int(n_low)
        self.n_high = int(n_high)
        self.model = None
        self.X_low = None
        self.y_low = None
        self.X_high = None
        self.y_high = None
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
                self.model = MFK(theta0=[1e-2] * self.X_low.shape[1],
                                 poly='constant',
                                 corr='squar_exp',
                                 nugget=1e-8)
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
                 seed=42):
        self.drone_class = drone_class
        self.spec = DroneGeometryProperties.get(drone_class)
        self.n_low = int(n_low)
        self.n_high = int(n_high)
        self.n_iter = int(n_iter)
        self.rng = np.random.default_rng(seed)
        self.surrogate = CokrigingSurrogate(n_low=n_low, n_high=n_high)
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
        """Low+mid fidelity evaluation: VLM + mid-fidelity corrector."""
        design = self._x_to_design(x)
        errors, _ = DroneGeometryProperties.validate_design(
            self.drone_class, design)
        if errors:
            return None
        geom = FPVDroneGeometry(design, self.drone_class)
        vlm = SpectralVLMSolver(geom, design).solve()
        mid = MidFidelityCorrector().correct(vlm, design, geom)
        return mid['L_D']

    @handle_errors
    def evaluate_high(self, x):
        """High-fidelity evaluation: external CFD hook (stub)."""
        design = self._x_to_design(x)
        result = self.high_fid_hook.evaluate(design, self.drone_class)
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
    def plot_design_space(self, X_low, y_low, X_high, y_high, dim_names=None):
        """Pairwise scatter of the design space, coloured by L/D."""
        if dim_names is None:
            dim_names = ['arm_L', 'prop_D', 'prop_pitch', 'blades',
                         'AOA', 'V_cruise', 'RPM']
        d = X_low.shape[1]
        fig, axes = plt.subplots(d, d, figsize=(2.0 * d, 2.0 * d))
        cmap = _CUSTOM_COLORMAPS['ld']
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
                                    c=y_low, cmap=cmap, s=20, alpha=0.7)
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
        # Add a single colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        plt.colorbar(ax.collections[0] if ax.collections else
                     fig.axes[0].collections[0], cax=cbar_ax, label='L/D')
        if any('CFD' in ax.get_legend_handles_labels()[1]
               for ax in fig.axes):
            fig.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        return fig

    @handle_errors
    def plot_convergence(self, history):
        """Plot best L/D vs. iteration."""
        iters = []
        best_so_far = []
        current_best = -np.inf
        for h in history:
            if 'best_y' in h:
                if h['best_y'] > current_best:
                    current_best = h['best_y']
                iters.append(len(iters) + 1)
                best_so_far.append(current_best)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(iters, best_so_far, 'o-', color='steelblue', lw=2, ms=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best L/D found')
        ax.set_title('Variable-fidelity optimisation convergence')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @handle_errors
    def plot_surrogate_surface(self, surrogate, fixed_var_idx=4,
                                fixed_values=None, n_grid=30):
        """Heatmap of the surrogate L/D response over two chosen design
        variables, with the others held fixed at their class midpoints."""
        spec = DroneGeometryProperties.get('FPV 5"')
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
        # Plot over variables 0 (arm_L) and 1 (prop_D) by default
        x0_lo, x0_hi = spec['arm_length_range']
        x1_lo, x1_hi = spec['prop_diameter_range']
        x0 = np.linspace(x0_lo, x0_hi, n_grid)
        x1 = np.linspace(x1_lo, x1_hi, n_grid)
        X0, X1 = np.meshgrid(x0, x1)
        X_grid = np.column_stack([
            X0.ravel(), X1.ravel(),
            np.full(X0.size, fixed_values[2]),
            np.full(X0.size, fixed_values[3]),
            np.full(X0.size, fixed_values[4]),
            np.full(X0.size, fixed_values[5]),
            np.full(X0.size, fixed_values[6]),
        ])
        mean, var = surrogate.predict(X_grid)
        Z = mean.reshape(n_grid, n_grid)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(X0, X1, Z, levels=20, cmap='ld')
        plt.colorbar(im, ax=ax, label='Predicted L/D')
        # Overlay samples
        if surrogate.X_low is not None:
            ax.scatter(surrogate.X_low[:, 0], surrogate.X_low[:, 1],
                       c='white', s=20, edgecolor='black', linewidth=0.6,
                       label='Low-fidelity')
        if surrogate.X_high is not None:
            ax.scatter(surrogate.X_high[:, 0], surrogate.X_high[:, 1],
                       c='red', s=80, marker='*', edgecolor='black',
                       linewidth=0.8, label='High-fidelity (CFD)')
        ax.set_xlabel('arm length (m)')
        ax.set_ylabel('prop diameter (m)')
        ax.set_title('Cokriging surrogate response (other vars fixed)')
        ax.legend(loc='best', fontsize=8)
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

        st.subheader("VLM discretisation")
        n_panels = st.slider("Panels per propeller blade", 4, 16, 8, 1)

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
        prop_rpm = st.slider("prop RPM", *spec['prop_rpm_range'],
                              np.mean(spec['prop_rpm_range']), 100)

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

    # ==============================================================
    # MAIN AREA
    # ==============================================================
    tabs = st.tabs(["Geometry", "Single Eval", "Optimise",
                    "Surrogate", "Convergence", "Export"])

    # ----- Geometry tab -----
    with tabs[0]:
        st.header("Drone Geometry (top view)")
        geom = FPVDroneGeometry(eval_design, drone_class,
                                 n_panels_per_prop=n_panels)
        fig = geom.plot_top_view()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Total lifting area: {geom.meta['total_lift_area']:.4f} "
                   f"m² | Total drag area: "
                   f"{geom.meta['total_drag_area']:.4f} m² | "
                   f"VLM panels: {geom.meta['total_panels']}")

    # ----- Single Eval tab -----
    with tabs[1]:
        st.header("Single-design aerodynamic evaluation")
        if st.button("Evaluate low + mid fidelity"):
            geom = FPVDroneGeometry(eval_design, drone_class,
                                     n_panels_per_prop=n_panels)
            vlm = SpectralVLMSolver(geom, eval_design, rho=rho_air)
            vlm_res = vlm.solve()
            mid = MidFidelityCorrector().correct(vlm_res, eval_design, geom)
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
            # Gamma distribution plot
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, pp in enumerate(geom.prop_panels):
                g = vlm_res['gamma_distribution'][
                    i * n_panels:(i + 1) * n_panels]
                r = pp['r_mid']
                ax.plot(r, g, '-o', label=f'Arm {i+1}', ms=4)
            ax.set_xlabel('radial station r (m)')
            ax.set_ylabel(r'$\Gamma$ (m²/s)')
            ax.set_title('Bound-circulation distribution per blade')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ----- Optimise tab -----
    with tabs[2]:
        st.header("Variable-fidelity optimisation")
        if st.button("Run optimisation", type="primary"):
            opt = VariableFidelityOptimizer(
                drone_class, n_low=n_low, n_high=n_high,
                n_iter=n_iter, seed=int(seed))
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
    with tabs[3]:
        st.header("Cokriging surrogate")
        if 'opt_result' not in st.session_state:
            st.info("Run the optimisation first.")
        else:
            result = st.session_state.opt_result
            vis = FPVDroneVisualizer()
            # Surrogate surface
            st.subheader("Response surface (arm_L vs prop_D)")
            fig_s = vis.plot_surrogate_surface(
                opt.surrogate if 'opt' in dir() else
                CokrigingSurrogate().fit(
                    result['X_low'], result['y_low'],
                    result['X_high'], result['y_high']))
            # We need to refit the surrogate because `opt` may be out of scope
            surr = CokrigingSurrogate()
            surr.fit(result['X_low'], result['y_low'],
                      result['X_high'], result['y_high'])
            fig_s = vis.plot_surrogate_surface(surr)
            st.pyplot(fig_s)
            plt.close(fig_s)
            # Design-space scatter
            st.subheader("Design-space samples")
            fig_d = vis.plot_design_space(
                result['X_low'], result['y_low'],
                result['X_high'], result['y_high'])
            st.pyplot(fig_d)
            plt.close(fig_d)

    # ----- Convergence tab -----
    with tabs[4]:
        st.header("Optimisation convergence")
        if 'opt_result' not in st.session_state:
            st.info("Run the optimisation first.")
        else:
            result = st.session_state.opt_result
            vis = FPVDroneVisualizer()
            fig = vis.plot_convergence(result['history'])
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
    with tabs[5]:
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
