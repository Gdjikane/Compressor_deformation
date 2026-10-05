import gc
import time

import numpy as np
import pandas as pd
import scipy.constants as sc

from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import gaussian_filter

from .Compressor import PulseCompressor, Modified_grating_eq_numba
from . import Postproc_stc as stc
from scipy import fft as spfft

C = sc.c

# In-memory caches. Numba itself can also reuse its on-disk cache
# when the @njit functions were declared with cache=True.
_NUMBA_WARMED_UP = False
_FLAT_OPL_CACHE = {}

# ============================================================
# GPU / CPU backend
# ============================================================

def get_cupy(use_gpu=True):
    """
    Return CuPy if CUDA is actually usable, otherwise return None.

    This catches:
      - CuPy not installed
      - no NVIDIA GPU
      - CUDA driver/runtime problems
      - CuPy installed for the wrong CUDA version
    """

    if not use_gpu:
        return None

    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device found")

        # Force CUDA context creation now rather than failing later
        _ = cp.empty(1, dtype=cp.float32)

        return cp

    except Exception as exc:
        print(
            f"CuPy/CUDA unavailable "
            f"({type(exc).__name__}: {exc}). "
            "Using CPU instead."
        )
        return None


# ============================================================
# Reference geometry
# ============================================================

def compute_roof_reference(
    lam_ref=810e-9,
    d=1.0 / 1.8e6,
    L=0.58,
    alpha_deg=56.0,
    gamma_deg=0.0,
    x1_ref=0.15,
    y1_ref=0.05,
):
    """
    Compute the nominal reference ray used to define the roof-mirror geometry.

    Flat reference geometry:
        G1 : z = 0
        G2 : z = L

    Returns
    -------
    pos_ref : ndarray, shape (3,)
        Nominal reference intersection point on G2.
    s2_ref : ndarray, shape (3,)
        Nominal propagation direction immediately after G2.
    """

    om = 2.0 * np.pi * C / lam_ref
    alpha = np.deg2rad(alpha_deg)
    gamma = np.deg2rad(gamma_deg)

    s_in = np.array(
        [
            np.sin(alpha) * np.cos(gamma),
            np.sin(gamma),
            -np.cos(alpha) * np.cos(gamma),
        ],
        dtype=np.float64,
    )

    beta1, theta1 = Modified_grating_eq_numba(
        alpha,
        gamma,
        om,
        0.0,   # dz_x
        0.0,   # dz_y
        -1.0,  # m
        0.0,   # dux
        0.0,   # duy
        d,
    )

    s1 = np.array(
        [
            np.sin(beta1) * np.cos(theta1),
            np.sin(theta1),
            np.cos(beta1) * np.cos(theta1),
        ],
        dtype=np.float64,
    )
    s1 /= np.linalg.norm(s1)

    t12 = L / s1[2]

    pos_ref = np.array(
        [
            x1_ref + t12 * s1[0],
            y1_ref + t12 * s1[1],
            L,
        ],
        dtype=np.float64,
    )

    gamma2 = np.arcsin(np.clip(s1[1], -1.0, 1.0))
    alpha2 = np.arcsin(
        np.clip(
            s1[0] / np.cos(gamma2),
            -1.0,
            1.0,
        )
    )

    beta2, theta2 = Modified_grating_eq_numba(
        alpha2,
        gamma2,
        om,
        0.0,
        0.0,
        +1.0,
        0.0,
        0.0,
        d,
    )

    s2_ref = np.array(
        [
            np.sin(beta2) * np.cos(theta2),
            np.sin(theta2),
            -np.cos(beta2) * np.cos(theta2),
        ],
        dtype=np.float64,
    )
    s2_ref /= np.linalg.norm(s2_ref)

    return pos_ref, s2_ref


# ============================================================
# COMSOL maps
# ============================================================

def load_comsol_grating_maps(
    vert_G1_file,
    vert_G2_file,
    hor_G1_file,
    hor_G2_file,
    pos_ref,
    slice_idx=35,
    n_grid=256,
    sigma=0.0,
    zero_boundary_slopes=True,
    xlim_G1=(0.0, 0.30),
    ylim_G1=(0.0, 0.24),
    G2_width=0.35,
    G2_height=0.30,
    y0_G2=0.05,
    y0_from_lower_edge_G2=0.08,
):
    """
    Load COMSOL vertical/horizontal displacement maps and prepare the
    arrays required by PulseCompressor.

    The G2 x-domain is centered on pos_ref[0].

    The G2 y-domain is defined from the nominal beam impact:
        y_min_G2 = y0_G2 - y0_from_lower_edge_G2
        y_max_G2 = y_min_G2 + G2_height

    With the defaults this gives [-0.03, 0.27] m.
    """

    i0 = n_grid * slice_idx
    i1 = i0 + n_grid

    Z1 = pd.read_csv(vert_G1_file, delimiter=",").iloc[i0:i1].to_numpy()
    Z2 = pd.read_csv(vert_G2_file, delimiter=",").iloc[i0:i1].to_numpy()
    U1 = pd.read_csv(hor_G1_file, delimiter=",").iloc[i0:i1].to_numpy()
    U2 = pd.read_csv(hor_G2_file, delimiter=",").iloc[i0:i1].to_numpy()

    if U1.shape != Z1.shape:
        raise ValueError(
            f"G1 horizontal/vertical maps have different shapes: "
            f"{U1.shape} vs {Z1.shape}"
        )

    if U2.shape != Z2.shape:
        raise ValueError(
            f"G2 horizontal/vertical maps have different shapes: "
            f"{U2.shape} vs {Z2.shape}"
        )

    ny1, nx1 = Z1.shape
    ny2, nx2 = Z2.shape

    x_min1, x_max1 = xlim_G1
    y_min1, y_max1 = ylim_G1

    x_min2 = pos_ref[0] - G2_width / 2.0
    x_max2 = pos_ref[0] + G2_width / 2.0

    y_min2 = y0_G2 - y0_from_lower_edge_G2
    y_max2 = y_min2 + G2_height

    xarr1 = np.linspace(x_min1, x_max1, nx1)
    yarr1 = np.linspace(y_min1, y_max1, ny1)
    xarr2 = np.linspace(x_min2, x_max2, nx2)
    yarr2 = np.linspace(y_min2, y_max2, ny2)

    dx1 = xarr1[1] - xarr1[0]
    dy1 = yarr1[1] - yarr1[0]
    dx2 = xarr2[1] - xarr2[0]
    dy2 = yarr2[1] - yarr2[0]

    if sigma > 0:
        z1_map = gaussian_filter(Z1, sigma=(sigma, sigma), mode="nearest")
        z2_map = gaussian_filter(Z2, sigma=(sigma, sigma), mode="nearest")
    else:
        z1_map = Z1.copy()
        z2_map = Z2.copy()

    # np.gradient: axis 0 -> y, axis 1 -> x
    dzy1_map, dzx1_map = np.gradient(z1_map, dy1, dx1, edge_order=2)
    dzy2_map, dzx2_map = np.gradient(z2_map, dy2, dx2, edge_order=2)

    if zero_boundary_slopes:
        for derivative in (dzx1_map, dzy1_map, dzx2_map, dzy2_map):
            derivative[0, :] = 0.0
            derivative[-1, :] = 0.0
            derivative[:, 0] = 0.0
            derivative[:, -1] = 0.0

    grad1_mag = np.sqrt(dzx1_map**2 + dzy1_map**2)
    grad2_mag = np.sqrt(dzx2_map**2 + dzy2_map**2)

    duy1_map, dux1_map = np.gradient(U1, dy1, dx1, edge_order=2)
    duy2_map, dux2_map = np.gradient(U2, dy2, dx2, edge_order=2)

    return {
        "Z1": Z1,
        "Z2": Z2,
        "U1": U1,
        "U2": U2,
        "z1_map": z1_map,
        "z2_map": z2_map,
        "dzx1_map": dzx1_map,
        "dzy1_map": dzy1_map,
        "dzx2_map": dzx2_map,
        "dzy2_map": dzy2_map,
        "dux1_map": dux1_map,
        "duy1_map": duy1_map,
        "dux2_map": dux2_map,
        "duy2_map": duy2_map,
        "grad1_mag": grad1_mag,
        "grad2_mag": grad2_mag,
        "xarr1": xarr1,
        "yarr1": yarr1,
        "xarr2": xarr2,
        "yarr2": yarr2,
        "dx1": dx1,
        "dy1": dy1,
        "dx2": dx2,
        "dy2": dy2,
        "extent1": (x_min1, x_max1, y_min1, y_max1),
        "extent2": (x_min2, x_max2, y_min2, y_max2),
        "xlim_G1": (x_min1, x_max1),
        "ylim_G1": (y_min1, y_max1),
        "xlim_G2": (x_min2, x_max2),
        "ylim_G2": (y_min2, y_max2),
        "y0_G2": y0_G2,
    }


# ============================================================
# OPL calculation
# ============================================================

def _array_cache_key(arr):
    arr = np.asarray(arr)
    return (
        arr.shape,
        arr.dtype.str,
        hash(arr.tobytes()),
    )


def warmup_compressor_numba(
    compressor,
    maps,
    lams,
    s2_ref,
    pos_ref,
    mask,
    d,
    alpha_deg,
    s_in_user,
    use_horizontal_displacement=True,
):
    """
    Trigger the Numba signatures with one wavelength only.

    Runs once per Python session. Numba's own cache can still make this
    warm-up very fast in later sessions.
    """

    global _NUMBA_WARMED_UP

    if _NUMBA_WARMED_UP:
        return

    print("First compressor call: warming up Numba with one wavelength...")

    lams_warmup = np.asarray(lams[:1])

    if use_horizontal_displacement:
        dux1_map = maps["dux1_map"]
        duy1_map = maps["duy1_map"]
        dux2_map = maps["dux2_map"]
        duy2_map = maps["duy2_map"]
    else:
        dux1_map = None
        duy1_map = None
        dux2_map = None
        duy2_map = None

    # Deformed signature (array dux/duy).
    compressor.compute_opl(
        maps["xarr1"], maps["yarr1"], lams_warmup,
        maps["z1_map"], maps["z2_map"],
        maps["dzx1_map"], maps["dzy1_map"], maps["xarr1"],
        maps["dzx2_map"], maps["dzy2_map"], maps["xarr2"],
        maps["yarr1"], maps["yarr2"],
        s2_ref, pos_ref, mask,
        dux1_map=dux1_map,
        duy1_map=duy1_map,
        dux2_map=dux2_map,
        duy2_map=duy2_map,
        d=d,
        alpha_deg=alpha_deg,
        s_in_user=s_in_user,
    )

    # Flat signature (None dux/duy).
    zero_z1 = np.zeros_like(maps["z1_map"])
    zero_z2 = np.zeros_like(maps["z2_map"])
    zero_dzx1 = np.zeros_like(maps["dzx1_map"])
    zero_dzy1 = np.zeros_like(maps["dzy1_map"])
    zero_dzx2 = np.zeros_like(maps["dzx2_map"])
    zero_dzy2 = np.zeros_like(maps["dzy2_map"])

    compressor.compute_opl(
        maps["xarr1"], maps["yarr1"], lams_warmup,
        zero_z1, zero_z2,
        zero_dzx1, zero_dzy1, maps["xarr1"],
        zero_dzx2, zero_dzy2, maps["xarr2"],
        maps["yarr1"], maps["yarr2"],
        s2_ref, pos_ref, mask,
        dux1_map=None,
        duy1_map=None,
        dux2_map=None,
        duy2_map=None,
        d=d,
        alpha_deg=alpha_deg,
        s_in_user=s_in_user,
    )

    _NUMBA_WARMED_UP = True
    print("Numba warm-up complete.")


def compute_compressor_opl(
    maps,
    pos_ref,
    s2_ref,
    lam_min=780e-9,
    lam_max=836e-9,
    n_lam=120,
    d=1.0 / 1.8e6,
    L=0.58,
    alpha_deg=56.0,
    gamma_deg=0.0,
    x0=0.15,
    y0=0.05,
    wy=0.028,
    wx=None,
    coeff=1.0,
    order=2,
    mask_threshold=5e-2,
    use_horizontal_displacement=True,
    warmup_numba=True,
    use_flat_cache=True,
):
    """
    Compute both the deformed compressor and nominal-flat OPL.

    The flat calculation is cached in memory and reused if the grid,
    wavelength axis, mask and reference geometry are unchanged.
    """

    lams = np.linspace(lam_min, lam_max, n_lam)

    xarr1 = maps["xarr1"]
    yarr1 = maps["yarr1"]

    X, Y = np.meshgrid(xarr1, yarr1, indexing="ij")

    alpha = np.deg2rad(alpha_deg)
    gamma = np.deg2rad(gamma_deg)

    if wx is None:
        wx = wy / np.cos(alpha)

    beam_amp = np.exp(
        -coeff
        * (
            ((X - x0) / wx) ** 2
            + ((Y - y0) / wy) ** 2
        ) ** order
    )

    mask = beam_amp > mask_threshold

    s_in_user = np.array(
        [
            np.sin(alpha) * np.cos(gamma),
            np.sin(gamma),
            -np.cos(alpha) * np.cos(gamma),
        ],
        dtype=np.float64,
    )

    compressor = PulseCompressor(
        d=d,
        L=L,
        alpha_deg=alpha_deg,
    )

    if warmup_numba:
        warmup_compressor_numba(
            compressor=compressor,
            maps=maps,
            lams=lams,
            s2_ref=s2_ref,
            pos_ref=pos_ref,
            mask=mask,
            d=d,
            alpha_deg=alpha_deg,
            s_in_user=s_in_user,
            use_horizontal_displacement=use_horizontal_displacement,
        )

    if use_horizontal_displacement:
        dux1_map = maps["dux1_map"]
        duy1_map = maps["duy1_map"]
        dux2_map = maps["dux2_map"]
        duy2_map = maps["duy2_map"]
    else:
        dux1_map = None
        duy1_map = None
        dux2_map = None
        duy2_map = None

    start_stc = time.time()

    OPL_stc_raw = compressor.compute_opl(
        maps["xarr1"], maps["yarr1"], lams,
        maps["z1_map"], maps["z2_map"],
        maps["dzx1_map"], maps["dzy1_map"], maps["xarr1"],
        maps["dzx2_map"], maps["dzy2_map"], maps["xarr2"],
        maps["yarr1"], maps["yarr2"],
        s2_ref, pos_ref, mask,
        dux1_map=dux1_map,
        duy1_map=duy1_map,
        dux2_map=dux2_map,
        duy2_map=duy2_map,
        d=d,
        alpha_deg=alpha_deg,
        s_in_user=s_in_user,
    )

    elapsed_stc = time.time() - start_stc

    OPLstc = np.asarray(OPL_stc_raw[0])
    r_primestc = np.asarray(OPL_stc_raw[1])
    debug_stc = np.asarray(OPL_stc_raw[2])
    phi_g_stc = np.asarray(OPL_stc_raw[3])

    print(f"Deformed compressor OPL: {elapsed_stc:.2f} s")

    flat_key = (
        _array_cache_key(maps["xarr1"]),
        _array_cache_key(maps["yarr1"]),
        _array_cache_key(maps["xarr2"]),
        _array_cache_key(maps["yarr2"]),
        _array_cache_key(lams),
        _array_cache_key(mask),
        float(d),
        float(L),
        float(alpha_deg),
        float(gamma_deg),
        tuple(np.asarray(pos_ref, dtype=float)),
        tuple(np.asarray(s2_ref, dtype=float)),
    )

    if use_flat_cache and flat_key in _FLAT_OPL_CACHE:
        flat_result = _FLAT_OPL_CACHE[flat_key]
        print("Flat compressor OPL: using cached result")

    else:
        zero_z1 = np.zeros_like(maps["z1_map"])
        zero_z2 = np.zeros_like(maps["z2_map"])
        zero_dzx1 = np.zeros_like(maps["dzx1_map"])
        zero_dzy1 = np.zeros_like(maps["dzy1_map"])
        zero_dzx2 = np.zeros_like(maps["dzx2_map"])
        zero_dzy2 = np.zeros_like(maps["dzy2_map"])

        start_flat = time.time()

        OPL_flat_raw = compressor.compute_opl(
            maps["xarr1"], maps["yarr1"], lams,
            zero_z1, zero_z2,
            zero_dzx1, zero_dzy1, maps["xarr1"],
            zero_dzx2, zero_dzy2, maps["xarr2"],
            maps["yarr1"], maps["yarr2"],
            s2_ref, pos_ref, mask,
            dux1_map=None,
            duy1_map=None,
            dux2_map=None,
            duy2_map=None,
            d=d,
            alpha_deg=alpha_deg,
            s_in_user=s_in_user,
        )

        elapsed_flat = time.time() - start_flat

        flat_result = {
            "OPL0": np.asarray(OPL_flat_raw[0]),
            "r_prime0": np.asarray(OPL_flat_raw[1]),
            "debug_flat": np.asarray(OPL_flat_raw[2]),
            "phi_g_0": np.asarray(OPL_flat_raw[3]),
            "elapsed_flat": elapsed_flat,
        }

        if use_flat_cache:
            _FLAT_OPL_CACHE[flat_key] = flat_result

        print(f"Flat compressor OPL: {elapsed_flat:.2f} s")

    return {
        "lams": lams,
        "X": X,
        "Y": Y,
        "beam_amp": beam_amp,
        "mask": mask,
        "wx": wx,
        "wy": wy,
        "x0": x0,
        "y0": y0,
        "s_in_user": s_in_user,
        "OPLstc": OPLstc,
        "r_primestc": r_primestc,
        "debug_stc": debug_stc,
        "phi_g_stc": phi_g_stc,
        "OPL0": flat_result["OPL0"],
        "r_prime0": flat_result["r_prime0"],
        "debug_flat": flat_result["debug_flat"],
        "phi_g_0": flat_result["phi_g_0"],
        "debug_stc_flat": flat_result["debug_flat"],
        "elapsed_stc": elapsed_stc,
        "elapsed_flat": flat_result["elapsed_flat"],
        "compressor": compressor,
    }


# ============================================================
# Near-field beam on the common interpolation grid
# ============================================================

def prepare_output_beam(
    maps,
    alpha_deg=56.0,
    wx0=0.028,
    wy0=0.028,
    x0=0.15,
    y0=0.05,
    order=2,
    interp_size=0.35,
    interp_points=650,
    beam_size_func=None,
):
    """
    Construct the nominal beam on G1, interpolate onto the square output
    grid and recenter it.

    If beam_size_func is omitted, laserbeamsize.beam_size is used.
    """

    if beam_size_func is None:
        try:
            import laserbeamsize as lbs
        except ImportError as exc:
            raise ImportError(
                "laserbeamsize is required, or pass beam_size_func explicitly."
            ) from exc
        beam_size_func = lbs.beam_size

    xarr1 = maps["xarr1"]
    yarr1 = maps["yarr1"]

    alpha = np.deg2rad(alpha_deg)

    X, Y = np.meshgrid(
        xarr1,
        yarr1,
        indexing="ij",
    )

    beam_amp_G1 = stc.gauss2D(
        X,
        Y,
        wx=wx0 / np.cos(alpha),
        wy=wy0,
        x0=x0,
        y0=y0,
        order=order,
    )

    # Exact orientation used in the original notebook before the spline.
    beam_amp = beam_amp_G1.T

    spline = RectBivariateSpline(
        yarr1,
        xarr1 * np.cos(alpha),
        beam_amp,
    )

    x_new = np.linspace(0.0, interp_size, interp_points)
    y_new = np.linspace(0.0, interp_size, interp_points)

    beam_amp2 = spline(y_new, x_new)

    beam_size_result = beam_size_func(beam_amp2)
    n_x0, n_y0, wx, wy = beam_size_result[:4]

    Eout = stc.recenter(
        beam_amp2,
        n_x0,
        n_y0,
    )

    return {
        "beam_amp_G1": beam_amp_G1,
        "beam_amp": beam_amp,
        "beam_amp2": beam_amp2,
        "Eout": Eout,
        "x_new": x_new,
        "y_new": y_new,
        "n_x0": n_x0,
        "n_y0": n_y0,
        "wx": wx,
        "wy": wy,
        "beam_size_result": beam_size_result,
    }


# ============================================================
# Phase interpolation / recentering
# ============================================================

def prepare_phase_cube(
    phi_omega,
    maps,
    beam,
    alpha_deg=56.0,
    mask_threshold=3e-3,
):
    """
    Transpose the OPL-derived phase cube, interpolate every spectral slice
    onto the same square grid as Eout, and recenter every slice.
    """

    alpha = np.deg2rad(alpha_deg)

    xarr1 = maps["xarr1"]
    yarr1 = maps["yarr1"]

    x_new = beam["x_new"]
    y_new = beam["y_new"]
    n_x0 = beam["n_x0"]
    n_y0 = beam["n_y0"]

    phi_omegatrans = np.transpose(phi_omega, (1, 0, 2))
    n_omega = phi_omegatrans.shape[2]

    phi_omega_interp = np.zeros(
        (len(y_new), len(x_new), n_omega),
        dtype=phi_omegatrans.dtype,
    )

    for i in range(n_omega):
        spline = RectBivariateSpline(
            yarr1,
            xarr1 * np.cos(alpha),
            phi_omegatrans[:, :, i],
        )
        phi_omega_interp[:, :, i] = spline(y_new, x_new)

    phi_omega_center = np.empty_like(phi_omega_interp)

    for w in range(n_omega):
        phi_omega_center[:, :, w] = stc.recenter(
            phi_omega_interp[:, :, w],
            n_x0,
            n_y0,
        )

    mask = beam["Eout"] > mask_threshold

    return {
        "phi_omegatrans": phi_omegatrans,
        "phi_omega_interp": phi_omega_interp,
        "phi_omega_center": phi_omega_center,
        "mask": mask,
        "x_new": x_new,
        "y_new": y_new,
    }


# ============================================================
# Spectrum
# ============================================================

def prepare_measured_spectrum(
    spectrum_file,
    lams,
    row_start=1254,
    row_stop=1427,
):
    """
    Reproduce the measured-spectrum processing used in the notebook.

    Important: this intentionally preserves the legacy construction:
    only the selected spectrum samples are used, reversed, then placed
    on a uniformly spaced omega axis spanning the simulation bandwidth.
    """

    a = pd.read_csv(
        spectrum_file,
        sep=r"\s+",
        header=None,
        comment="#",
    )
    a.columns = ["wavelength", "spectrum"]

    selected = a["spectrum"].iloc[row_start:row_stop].to_numpy(dtype=float)

    if selected.size == 0:
        raise ValueError("Selected spectrum slice is empty.")

    selected /= np.max(selected)
    spectrum = selected[::-1]

    lams = np.asarray(lams)

    omega_max = 2.0 * np.pi * C / np.min(lams)
    omega_min = 2.0 * np.pi * C / np.max(lams)

    omega_uniform = np.linspace(
        omega_min,
        omega_max,
        len(lams),
    )

    omega_spectrum = np.linspace(
        omega_min,
        omega_max,
        len(spectrum),
    )

    interp_spectrum = interp1d(
        omega_spectrum,
        spectrum,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    spectrum_uniform = interp_spectrum(omega_uniform)

    return {
        "raw_table": a,
        "spectrum": spectrum,
        "omega_spectrum": omega_spectrum,
        "omega_uniform": omega_uniform,
        "spectrum_uniform": spectrum_uniform,
    }


# ============================================================
# Global defocus removal
# ============================================================

def remove_global_defocus(
    phi_omega_center,
    x,
    y,
    k,
    idx0=60,
    N_zern=15,
    fit_mask=None,
):
    """
    Fit global defocus at one reference wavelength using Zernikes,
    then remove the corresponding physical quadratic phase from all
    wavelengths with k/k0 scaling.

    This is the same numerical procedure as in the notebook, moved here
    without changing its physical choice.
    """

    import zern.zern_core as zern
    from skimage.restoration import unwrap_phase

    phase0 = phi_omega_center[:, :, idx0].copy()

    if fit_mask is None:
        fit_mask = np.ones_like(phase0, dtype=bool)

    fit_mask = fit_mask.astype(bool)

    iy, ix = np.where(fit_mask)

    if iy.size == 0:
        raise ValueError("fit_mask contains no valid pixels.")

    iy0, iy1 = iy.min(), iy.max() + 1
    ix0, ix1 = ix.min(), ix.max() + 1

    phase_crop = phase0[iy0:iy1, ix0:ix1]
    mask_crop = fit_mask[iy0:iy1, ix0:ix1]

    x_crop = np.asarray(x)[ix0:ix1]
    y_crop = np.asarray(y)[iy0:iy1]

    phase_masked = np.ma.array(
        phase_crop,
        mask=~mask_crop,
    )

    phase_unwrapped = unwrap_phase(phase_masked)
    phase_unwrapped = np.asarray(phase_unwrapped)

    Ny_c, Nx_c = phase_crop.shape

    xn = np.linspace(-1.0, 1.0, Nx_c)
    yn = np.linspace(-1.0, 1.0, Ny_c)

    XXn, YYn = np.meshgrid(xn, yn)

    rho = np.sqrt(XXn**2 + YYn**2)
    theta = np.arctan2(YYn, XXn)

    circular_mask = rho <= 1.0
    aperture_mask = circular_mask & mask_crop

    rho_flat = rho[aperture_mask]
    theta_flat = theta[aperture_mask]

    z = zern.Zernike(mask=aperture_mask)

    z.create_model_matrix(
        rho_flat,
        theta_flat,
        n_zernike=N_zern,
        mode="Jacobi",
        normalize_noll=False,
    )

    H = z.model_matrix_flat
    flat_phase = phase_unwrapped[aperture_mask]

    fit_coef = np.linalg.lstsq(
        H,
        flat_phase,
        rcond=None,
    )[0]

    analytical_defocus = 2.0 * rho_flat**2 - 1.0
    analytical_defocus -= np.mean(analytical_defocus)

    correlations = np.zeros(H.shape[1])

    for j in range(H.shape[1]):
        mode = H[:, j].copy()
        mode -= np.mean(mode)

        denom = (
            np.linalg.norm(mode)
            * np.linalg.norm(analytical_defocus)
        )

        if denom > 0:
            correlations[j] = np.abs(
                np.dot(mode, analytical_defocus) / denom
            )

    defocus_idx = np.argmax(correlations)

    defocus_coef = np.zeros_like(fit_coef)
    defocus_coef[defocus_idx] = fit_coef[defocus_idx]

    defocus_flat = H @ defocus_coef

    defocus_crop = np.zeros_like(phase_crop, dtype=float)
    defocus_crop[aperture_mask] = defocus_flat

    Xc, Yc = np.meshgrid(x_crop, y_crop)

    xc = 0.5 * (x_crop[0] + x_crop[-1])
    yc = 0.5 * (y_crop[0] + y_crop[-1])

    Xrel = Xc - xc
    Yrel = Yc - yc

    valid = aperture_mask

    A = np.column_stack(
        [
            Xrel[valid] ** 2,
            Yrel[valid] ** 2,
            np.ones(np.sum(valid)),
        ]
    )

    coef_quad = np.linalg.lstsq(
        A,
        defocus_crop[valid],
        rcond=None,
    )[0]

    ax = coef_quad[0]
    ay = coef_quad[1]

    Rx = k[idx0] / (2.0 * ax)
    Ry = k[idx0] / (2.0 * ay)

    X, Y = np.meshgrid(x, y)

    quad_phase = (
        ax * (X - xc) ** 2
        + ay * (Y - yc) ** 2
    )

    phi_corr = phi_omega_center.copy()

    for iw in range(len(k)):
        phi_corr[:, :, iw] -= (
            k[iw] / k[idx0]
        ) * quad_phase

    print("Zernike defocus mode index:", defocus_idx)
    print("Defocus coefficient:", fit_coef[defocus_idx], "rad")
    print("Removed curvature:")
    print("Rx =", Rx, "m")
    print("Ry =", Ry, "m")
    print("Zernike matrix condition number:", np.linalg.cond(H))

    return {
        "phi_corr": phi_corr,
        "Rx": Rx,
        "Ry": Ry,
        "quad_phase": quad_phase,
        "fit_coef": fit_coef,
        "defocus_idx": defocus_idx,
        "ax": ax,
        "ay": ay,
        "xc": xc,
        "yc": yc,
    }


# ============================================================
# GPU utilities / far-field FFT
# ============================================================

def clear_cupy_memory():
    """Release CuPy cached GPU memory if CuPy is available."""
    try:
        import cupy as cp
    except ImportError:
        return

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def compute_far_field(
    Eout,
    phi_omega_center,
    spectrum_uniform,
    pad_factor=2,
    lambda_focus=800e-9,
    focal_length=1.468,
    x_min=0.0,
    x_max=0.35,
    use_gpu=True,
):
    """
    Build E(x,y,omega), pad spatially and compute the 2D FFT.

    Uses CuPy/CUDA when available.
    Falls back automatically to SciPy/CPU otherwise.
    """

    cp = get_cupy(use_gpu)

    Eout = np.asarray(Eout)
    phi_omega_center = np.asarray(phi_omega_center)
    spectrum_uniform = np.asarray(spectrum_uniform)

    if phi_omega_center.shape[:2] != Eout.shape:
        raise ValueError(
            "Eout and phi_omega_center spatial shapes do not match: "
            f"{Eout.shape} vs {phi_omega_center.shape[:2]}"
        )

    if phi_omega_center.shape[2] != spectrum_uniform.size:
        raise ValueError(
            "Phase cube and spectrum have different spectral lengths: "
            f"{phi_omega_center.shape[2]} vs {spectrum_uniform.size}"
        )

    # ========================================================
    # Build spectral field
    # ========================================================

    E_field = (
        Eout[:, :, None]
        * np.exp(1j * phi_omega_center)
        * spectrum_uniform[None, None, :]
    ).astype(np.complex64, copy=False)

    # ========================================================
    # Spatial padding
    # ========================================================

    padded_E = stc.pad_array_3d(
        E_field,
        pad_factor,
    )

    padded_E = np.ascontiguousarray(
        padded_E,
        dtype=np.complex64,
    )

    # ========================================================
    # GPU path
    # ========================================================

    if cp is not None:

        print("Far-field FFT: using GPU (CuPy)")

        padded_E_gpu = cp.empty(
            padded_E.shape,
            dtype=cp.complex64,
        )

        # Direct synchronous copy avoids CuPy pinned staging
        cp.cuda.runtime.memcpy(
            padded_E_gpu.data.ptr,
            padded_E.ctypes.data,
            padded_E.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
        )

        shifted_gpu = cp.fft.ifftshift(
            padded_E_gpu,
            axes=(0, 1),
        )

        E_ff_gpu = cp.fft.fftshift(
            cp.fft.fft2(
                shifted_gpu,
                axes=(0, 1),
                norm="ortho",
            ),
            axes=(0, 1),
        )

        E_ff_om = cp.asnumpy(
            E_ff_gpu
        )

        del E_ff_gpu
        del shifted_gpu
        del padded_E_gpu

        clear_cupy_memory()

        backend = "GPU (CuPy)"

    # ========================================================
    # CPU fallback
    # ========================================================

    else:

        print("Far-field FFT: using CPU (SciPy)")

        shifted = np.fft.ifftshift(
            padded_E,
            axes=(0, 1),
        )

        E_ff_om = np.fft.fftshift(
            spfft.fft2(
                shifted,
                axes=(0, 1),
                norm="ortho",
                workers=-1,
            ),
            axes=(0, 1),
        )

        # Keep memory consumption comparable to CuPy complex64
        E_ff_om = np.asarray(
            E_ff_om,
            dtype=np.complex64,
        )

        backend = "CPU (SciPy)"

    # ========================================================
    # Intensity
    # ========================================================

    I_ff = np.sum(
        np.abs(E_ff_om)**2,
        axis=2,
    )

    # ========================================================
    # Focal-plane pixel size
    # ========================================================

    Nx_input = Eout.shape[1]

    dx = (
        x_max - x_min
    ) / (Nx_input - 1)

    N_fft = (
        pad_factor
        * Nx_input
    )

    dx_focus = (
        lambda_focus
        * focal_length
        / (N_fft * dx)
    )

    print(f"Backend                    : {backend}")
    print(f"Real-space pixel size      : {dx:.3e} m")
    print(f"Focus-plane pixel size     : {dx_focus * 1e6:.3f} µm")

    return {
        "E_field": E_field,
        "E_ff_om": E_ff_om,
        "I_ff": I_ff,

        "dx": dx,
        "dx_focus": dx_focus,

        "pad_factor": pad_factor,
        "lambda_focus": lambda_focus,
        "focal_length": focal_length,

        "backend": backend,
    }
# ============================================================
# Far-field diagnostic preparation
# ============================================================


def centered_coordinate_axis(n_pixels, pixel_size, scale=1.0):
    """
    Centered pixel coordinate axis with the exact FFT-plane pixel spacing.
    """
    return (
        np.arange(n_pixels)
        - (n_pixels - 1) / 2.0
    ) * pixel_size * scale


# ============================================================
# Centered 2D crop
# ============================================================

def center_crop_2d(array, crop_size):
    """
    Return a centered square crop of size crop_size x crop_size.
    """

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            "center_crop_2d expects a 2D array."
        )

    ny, nx = array.shape

    if crop_size > min(ny, nx):
        raise ValueError(
            f"crop_size={crop_size} exceeds "
            f"input shape {array.shape}"
        )

    y0 = (ny - crop_size) // 2
    x0 = (nx - crop_size) // 2

    return array[
        y0:y0 + crop_size,
        x0:x0 + crop_size,
    ]


# ============================================================
# Fixed-wavelength far-field diagnostics
# ============================================================

def prepare_fixed_wavelength_slices(
    E_ff_om,
    omegas,
    dx_focus,
    wavelengths_nm=np.array(
        [785, 793, 803, 812, 824, 832]
    ),
    crop_size=162,
):
    """
    Prepare far-field phase/intensity maps at selected wavelengths.

    IMPORTANT
    ---------
    E_ff_om is assumed to already have the same spectral ordering
    as `omegas`.

    The wavelength selection reproduces the original notebook:

        lambda_target
            -> omega_target
            -> nearest index in omegas
            -> same index in E_ff_om
    """

    wavelengths_nm = np.asarray(
        wavelengths_nm,
        dtype=float,
    )

    omegas = np.asarray(
        omegas
    )

    # --------------------------------------------------------
    # Requested wavelengths -> requested frequencies
    # --------------------------------------------------------

    lams_diag = (
        wavelengths_nm
        * 1e-9
    )

    omegas_target = (
        2.0
        * np.pi
        * sc.c
        / lams_diag
    )

    # --------------------------------------------------------
    # Same index logic as original notebook
    # --------------------------------------------------------

    omega_indices = np.array(
        [
            np.argmin(
                np.abs(
                    omegas - om
                )
            )
            for om in omegas_target
        ],
        dtype=int,
    )

    # --------------------------------------------------------
    # Extract centered crops
    # --------------------------------------------------------

    phase_slices = []
    intensity_slices = []

    for idx in omega_indices:

        phase = np.angle(
            E_ff_om[:, :, idx]
        )

        amplitude = np.abs(
            E_ff_om[:, :, idx]
        )

        cropped_phase = center_crop_2d(
            phase,
            crop_size,
        )

        cropped_amp = center_crop_2d(
            amplitude,
            crop_size,
        )

        phase_slices.append(
            cropped_phase
        )

        intensity_slices.append(
            cropped_amp**2
        )

    phase_slices = np.stack(
        phase_slices,
        axis=0,
    )

    intensity_slices = np.stack(
        intensity_slices,
        axis=0,
    )

    # --------------------------------------------------------
    # Physical focal-plane coordinate
    #
    # Pixel spacing = dx_focus
    # --------------------------------------------------------

    x_cropped = (
        np.arange(crop_size)
        - (crop_size - 1) / 2.0
    ) * dx_focus * 1e6

    return {
        "wavelengths_nm": wavelengths_nm,
        "lams_diag": lams_diag,

        "omegas_target": omegas_target,
        "omega_indices": omega_indices,

        "phase_slices": phase_slices,
        "intensity_slices": intensity_slices,

        "x_cropped": x_cropped,
    }
def prepare_far_field_spectral_maps(
    E_ff_om,
    lams,
    dx_focus,
    crop_factor=4,
):
    """
    E_ff_om spectral axis follows the omega-sorted order produced
    by stc.get_phi().
    """

    # ========================================================
    # Reconstruct EXACT spectral order of E_ff_om
    # ========================================================

    omegas_raw = (
        2.0 * np.pi
        * sc.c
        / np.asarray(lams)
    )

    sort_idx_omega = np.argsort(
        omegas_raw
    )

    omegas = omegas_raw[
        sort_idx_omega
    ]

    # E_ff_om follows this order:
    # lambda = 836 -> 780 nm
    lams_Eff = np.asarray(lams)[
        sort_idx_omega
    ]

    # ========================================================
    # Crop
    # ========================================================

    Eff = stc.crop_array_3d(
        E_ff_om,
        crop_factor,
    )

    # ========================================================
    # Integrate over transverse dimensions
    # ========================================================

    Iff1 = np.sum(
        np.abs(Eff)**2,
        axis=0,
    )

    Iff3 = np.sum(
        np.abs(Eff)**2,
        axis=1,
    )

    # ========================================================
    # Sort from increasing lambda for plotting
    #
    # E_ff order: 836 -> 780
    # plot order : 780 -> 836
    # ========================================================

    sort_idx_lambda = np.argsort(
        lams_Eff
    )

    lams_sorted = lams_Eff[
        sort_idx_lambda
    ]

    Iff_sorted1 = Iff1[
        :,
        sort_idx_lambda,
    ]

    Iff_sorted3 = Iff3[
        :,
        sort_idx_lambda,
    ]

    lams_nm = (
        lams_sorted
        * 1e9
    )

    # ========================================================
    # Spatial axis — exact old notebook convention
    # ========================================================

    x_cropped = (
        np.linspace(
            -81,
            81,
            162,
        )
        * dx_focus
        * 1e6
    )

    extent_lambda = [
        lams_nm.min(),
        lams_nm.max(),
        x_cropped.min(),
        x_cropped.max(),
    ]

    # ========================================================
    # Integrated focal spot
    # ========================================================

    Iff2 = np.sum(
        np.abs(Eff)**2,
        axis=2,
    )

    return {
        "Eff": Eff,

        "omegas": omegas,
        "lams_Eff": lams_Eff,

        "Iff1": Iff1,
        "Iff3": Iff3,
        "Iff2": Iff2,

        "Iff_sorted1": Iff_sorted1,
        "Iff_sorted3": Iff_sorted3,

        "lams_sorted": lams_sorted,
        "lams_nm": lams_nm,

        "x_cropped": x_cropped,
        "extent_lambda": extent_lambda,
    }

# ============================================================
# Spectral -> temporal focal field
# ============================================================

def far_field_to_time(
    E_ff_om,
    omegas,
    spatial_crop_factor=8,
    Nw_uniform=512,
    pad_factor=8,
    time_crop=30,
    use_gpu=True,
):
    """
    Spectral -> temporal transform.

    E_ff_om and omegas are assumed to already have matching
    increasing-frequency ordering.

    Uses CuPy/CUDA when available and automatically falls back
    to SciPy/CPU otherwise.
    """

    cp = get_cupy(use_gpu)

    # ========================================================
    # Crop focal-plane field
    # ========================================================

    Effom = stc.crop_array_3d(
        E_ff_om,
        spatial_crop_factor,
    )

    Nx, Ny, Nw = Effom.shape

    # E_ff_om is already omega-sorted
    omegas = np.sort(
        np.asarray(omegas)
    )

    if len(omegas) != Nw:
        raise ValueError(
            f"Effom has {Nw} spectral slices "
            f"but omegas has {len(omegas)} values"
        )

    # ========================================================
    # Uniform omega grid
    # ========================================================

    omega_uniform = np.linspace(
        omegas[0],
        omegas[-1],
        Nw_uniform,
    )

    interp_func = interp1d(
        omegas,
        Effom,
        axis=-1,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    E_omega_uniform = interp_func(
        omega_uniform
    ).astype(np.complex64)

    # ========================================================
    # Spectral padding
    # ========================================================

    Nw_pad = (
        pad_factor
        * Nw_uniform
    )

    start_idx = (
        Nw_pad - Nw_uniform
    ) // 2

    E_padded_cpu = np.zeros(
        (Nx, Ny, Nw_pad),
        dtype=np.complex64,
    )

    E_padded_cpu[
        :,
        :,
        start_idx:start_idx + Nw_uniform,
    ] = E_omega_uniform

    del E_omega_uniform
    del Effom

    # ========================================================
    # GPU path
    # ========================================================

    if cp is not None:

        print("Temporal FFT: using GPU (CuPy)")

        E_gpu = cp.empty(
            E_padded_cpu.shape,
            dtype=cp.complex64,
        )

        cp.cuda.runtime.memcpy(
            E_gpu.data.ptr,
            E_padded_cpu.ctypes.data,
            E_padded_cpu.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
        )

        E_t_gpu = cp.fft.fftshift(
            cp.fft.ifft(
                cp.fft.ifftshift(
                    E_gpu,
                    axes=-1,
                ),
                axis=-1,
            ),
            axes=-1,
        )

        del E_gpu

        E_t = cp.asnumpy(
            E_t_gpu
        )

        del E_t_gpu

        clear_cupy_memory()

        backend = "GPU (CuPy)"

    # ========================================================
    # CPU fallback
    # ========================================================

    else:

        print("Temporal FFT: using CPU (SciPy)")

        E_t = np.fft.fftshift(
            spfft.ifft(
                np.fft.ifftshift(
                    E_padded_cpu,
                    axes=-1,
                ),
                axis=-1,
                workers=-1,
            ),
            axes=-1,
        )

        E_t = np.asarray(
            E_t,
            dtype=np.complex64,
        )

        backend = "CPU (SciPy)"

    del E_padded_cpu

    # ========================================================
    # Intensity
    # ========================================================

    I_t = np.abs(E_t)**2

    # ========================================================
    # Time axis
    # ========================================================

    domega = (
        omega_uniform[1]
        - omega_uniform[0]
    )

    t_axis = np.fft.fftshift(
        np.fft.fftfreq(
            Nw_pad,
            d=domega / (2.0 * np.pi),
        )
    )

    dt = (
        t_axis[1]
        - t_axis[0]
    )

    # ========================================================
    # Temporal crop
    # ========================================================

    Itt = stc.crop_array_omega(
        I_t,
        time_crop,
    )

    Ett = stc.crop_array_omega(
        E_t,
        time_crop,
    )

    t_axis_new = (
        np.arange(Itt.shape[2])
        * dt
    )

    t_axis_new -= np.mean(
        t_axis_new
    )

    # ========================================================
    # Integrated temporal maps
    # ========================================================

    A = np.sum(
        Itt,
        axis=0,
    )

    B = np.sum(
        Itt,
        axis=1,
    )
    total_t = np.sum(Itt, axis=(0,1))  # shape: (Nw_pad,)
    AA = np.sum(Ett, axis=(0,1))
    phase = np.unwrap(np.angle(AA))


    print(f"Backend: {backend}")

    return {
        "omegas": omegas,
        "omega_uniform": omega_uniform,

        "E_t": E_t,
        "I_t": I_t,

        "Itt": Itt,
        "Ett": Ett,

        "t_axis": t_axis,
        "t_axis_new": t_axis_new,
        "dt": dt,

        "A": A,
        "B": B,
        "total_t":total_t,
        "phase":phase,

        "Nx": Nx,
        "Ny": Ny,

        "backend": backend,
    }