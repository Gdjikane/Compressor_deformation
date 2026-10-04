import numpy as np
import scipy.constants as sc
from joblib import Parallel, delayed
from numba import njit,prange
C=sc.c
# -------------------------
# 1) Bilinear interpolation (Numba)
# -------------------------
@njit(cache=True)
def bilinear_interpolate(x, y, grid, x_arr, y_arr):
    """
    Bilinear interpolation of 2D grid at point (x, y).

    input:
    ----------
    x,y: float (m) point at which the interpolation is evaluated

    x_arr,y_arr: 1D arrays coordinates of the gratings.

    grid: 2D arrays, shape (ny, nx) deformation and derivative map.

    output: 
    ----------
    interpolated value at point (x,y)
    """

    nx = x_arr.size
    ny = y_arr.size

    # find indices in x
    if x <= x_arr[0]:
        i = 0
        dx = 0.0
    elif x >= x_arr[-1]:
        i = nx - 2
        dx = 1.0
    else:
        # locate x index
        for idx in range(nx-1):
            if x_arr[idx] <= x <= x_arr[idx+1]:
                i = idx
                break
        dx = (x - x_arr[i]) / (x_arr[i+1] - x_arr[i])
    # find indices in y
    if y <= y_arr[0]:
        j = 0
        dy = 0.0
    elif y >= y_arr[-1]:
        j = ny - 2
        dy = 1.0
    else:
        for idy in range(ny-1):
            if y_arr[idy] <= y <= y_arr[idy+1]:
                j = idy
                break
        dy = (y - y_arr[j]) / (y_arr[j+1] - y_arr[j])
    # four surrounding values
    f00 = grid[j, i]
    f10 = grid[j, i+1]
    f01 = grid[j+1, i]
    f11 = grid[j+1, i+1]
    # bilinear interpolation formula
    f = f00 * (1-dx)*(1-dy) + f10 * dx*(1-dy) + f01 * (1-dx)*dy + f11 * dx*dy
    return f
@njit(cache=True)
def Modified_grating_eq_numba(
    alpha, gamma, om,
    dz_x, dz_y,
    m,
    dux=0.0, duy=0.0,
    d=1.0/1.8e6
):
    """
    Modified grating equation including:
      - out-of-plane surface slopes dz_x, dz_y
      - horizontal-displacement-induced pitch variation dux = du/dx
      - the first-order y-coupling duy = du/dy

    The equations implemented are
    sin(theta_y) =sin(gamma)- dz_y * (cos(beta) cos(theta_y) + cos(alpha) cos(gamma)) -m * lambda/d * duy
    sin(beta) cos(theta_y) - sin(alpha) cos(gamma)+ dz_x * (cos(beta) cos(theta_y) + cos(alpha) cos(gamma))= m * lambda / [d (1 + dux)]
    with u(0,y) assumed negligible in the y-coupling term.
    """
    theta_y = gamma
    lambda_over_d = 2.0 * np.pi * C / (om * d)
    pitch_factor = 1.0 + dux
    if pitch_factor <= 0.0:
        raise ValueError("Invalid horizontal deformation: 1 + du/dx <= 0")
    p = dz_x
    A = (
        np.sin(alpha) * np.cos(gamma)
        - p * np.cos(alpha) * np.cos(gamma)
        + m * lambda_over_d / pitch_factor
    )
    denom = np.sqrt(1.0 + p * p) * np.cos(theta_y)
    arg = A / denom
    if np.abs(arg) > 1.0:
        raise ValueError("Invalid geometry (|sinβ'|>1)")
    beta = np.arcsin(arg) - np.arctan(p)
    tol = 1e-12
    max_iter = 100
    for _ in range(max_iter):
        theta_old = theta_y
        beta_old = beta
        arg_theta = (
            np.sin(gamma)
            - dz_y * (
                np.cos(beta_old) * np.cos(theta_old)
                + np.cos(alpha) * np.cos(gamma)
            )
            - m * lambda_over_d * duy
        )
        theta_y = np.arcsin(arg_theta)
        A = (
            np.sin(alpha) * np.cos(gamma)
            - p * np.cos(alpha) * np.cos(gamma)
            + m * lambda_over_d / pitch_factor
        )
        denom = np.sqrt(1.0 + p * p) * np.cos(theta_y)
        arg = A / denom
        if np.abs(arg) > 1.0:
            raise ValueError("Invalid geometry (|sinβ'|>1)")
        beta = np.arcsin(arg) - np.arctan(p)
        if max(abs(beta - beta_old), abs(theta_y - theta_old)) < tol:
            break
    return beta, theta_y
@njit(cache=True)
def path_from_incident_plane_to_G1_forward(x1, y1, s_in,z1_map, xarr1, yarr1,z1_dist=1.0):
    """
    Compute intersection distance t from incident plane to G1 surface.
    input:
    ----------
    x1, y1 : float (m) Coordinates on the grating plane
    s_in : array(3,) (unit vector) Incoming ray vector on G1 
    z1_map : 2D array (m) G1 surface height map
    xarr1, yarr1 : 1D arrays (m) coordinates of the gratings.
    z1_dist : float (m) Distance from the incident plane reference to G1 reference along s_in


    output:
    -------
    t : float (m) Distance along s_in to G1
    r_planes: floats (m) x,y,z cordinate  on the incident plane
    """
    # Reference point on the flat G1 reference plane
    ref_point_G1 = np.array([0.15, 0.12, 0.0])
    # Actual physical point on the deformed G1
    z1 = bilinear_interpolate(
        x1, y1,
        z1_map,
        xarr1, yarr1
    )
    r_G1 = np.array([
        x1,
        y1,
        z1
    ])
    # Reference point on the incident plane
    ref_point_plane = ref_point_G1 - z1_dist * s_in
    # Since s_in is unit length and the plane is perpendicular to s_in,
    # this is the signed propagation distance from the plane to G1.
    t_in = np.dot(
        r_G1 - ref_point_plane,
        s_in
    )
    # Corresponding point on the incident plane
    r_plane = r_G1 - t_in * s_in
    return t_in, r_plane[0], r_plane[1], r_plane[2]
@njit(cache=True)
def path_from_G1_to_incident_plane_return(s_in, s_out, r_G1_out,z1_dist=1.0):
    """
    Compute intersection from G1 surface back to incident plane.

    input:
    ----------
    z1_dist : float (m) Distance from the incident plane reference to G1 reference along s_in
    s_in : array(3,) (unit vector) Incoming ray vector 
    s_out : array(3,) (unit vector) Output ray direction after G1
    r_G1_out : array(3,) (m) Position on G1 surface

    output:
    -------
    t1 : float (m) Distance along s_out to reach the incident plane
    r1 : array(3,) (m) Intersection point on the incident plane
    """
    # Reference point on G1
    ref_point_G1 = np.array([0.15, 0.12, 0.0])
    # Compute reference point on incident plane
    ref_point_plane = ref_point_G1 - z1_dist * s_in
    denom1 = np.dot(s_in, s_out)
    t1 = np.dot(ref_point_plane - r_G1_out, s_in) / denom1
    r1 = np.empty(3)
    r1[0] = r_G1_out[0] + t1 * s_out[0]
    r1[1] = r_G1_out[1] + t1 * s_out[1]
    r1[2] = r_G1_out[2] + t1 * s_out[2]
    return t1, r1

@njit(cache=True)
def diffract_reflect_vector_numba(
    s_in, x, y,
    dzx1_map, dzy1_map, xarr1,
    dzx2_map, dzy2_map, xarr2,
    yarr1, yarr2,
    d=1.0/1.8e6,
    om=2.0*np.pi*C/810e-9,
    G1_diff=True,
    dux1_map=None, duy1_map=None,
    dux2_map=None, duy2_map=None
):
    """
    Output direction after diffraction on G1 or G2.

    dux*_map = du/dx and duy*_map = du/dy for the horizontal displacement field u(x,y). If omitted, pitch variation is disabled.

    """
    sy = s_in[1]
    if G1_diff:
        dzx = bilinear_interpolate(x, y, dzx1_map, xarr1, yarr1)
        dzy = bilinear_interpolate(x, y, dzy1_map, xarr1, yarr1)
        if dux1_map is None:
            dux = 0.0
            duy = 0.0
        else:
            dux = bilinear_interpolate(x, y, dux1_map, xarr1, yarr1)
            duy = bilinear_interpolate(x, y, duy1_map, xarr1, yarr1)
        ux = 1.0
        uy = 1.0
        uz = 1.0
        m = -1.0
    else:
        dzx = bilinear_interpolate(x, y, dzx2_map, xarr2, yarr2)
        dzy = bilinear_interpolate(x, y, dzy2_map, xarr2, yarr2)
        if dux2_map is None:
            dux = 0.0
            duy = 0.0
        else:
            dux = bilinear_interpolate(x, y, dux2_map, xarr2, yarr2)
            duy = bilinear_interpolate(x, y, duy2_map, xarr2, yarr2)
        ux = 1.0
        uy = 1.0
        uz = -1.0
        m = 1.0
    gamma = np.arcsin(sy)
    alpha_deg = np.arcsin(s_in[0] / np.cos(gamma))
    beta_loc, theta_y = Modified_grating_eq_numba(
        alpha_deg, gamma, om,
        dzx, dzy,
        m,
        dux=dux, duy=duy,
        d=d
    )
    s_out = np.zeros(3, dtype=np.float64)
    s_out[0] = np.sin(beta_loc) * np.cos(theta_y) * ux
    s_out[1] = np.sin(theta_y) * uy
    s_out[2] = np.cos(beta_loc) * np.cos(theta_y) * uz
    return s_out
# ============================================================
# Bilinear interpolation + exact gradient of same interpolant
# ============================================================
@njit(cache=True)
def bilinear_value_grad_numba(x, y, grid, x_arr, y_arr):
    """
    Bilinear interpolation of grid[y,x] and exact spatial derivatives of that SAME bilinear interpolant
    The boundary behaviour matches bilinear_interpolate: coordinates outside the map are clamped to the edge.

    Returns
    -------
    value : float
    dfdx  : float
    dfdy  : float
    """
    nx = x_arr.size
    ny = y_arr.size
    # --------------------------------------------------------
    # Locate x
    # --------------------------------------------------------
    clamp_x = False
    if x <= x_arr[0]:
        i = 0
        tx = 0.0
        clamp_x = True
    elif x >= x_arr[-1]:
        i = nx - 2
        tx = 1.0
        clamp_x = True
    else:
        i = 0
        for idx in range(nx - 1):
            if x_arr[idx] <= x <= x_arr[idx + 1]:
                i = idx
                break
        tx = (
            (x - x_arr[i])
            /
            (x_arr[i + 1] - x_arr[i])
        )
    # --------------------------------------------------------
    # Locate y
    # --------------------------------------------------------
    clamp_y = False
    if y <= y_arr[0]:
        j = 0
        ty = 0.0
        clamp_y = True
    elif y >= y_arr[-1]:
        j = ny - 2
        ty = 1.0
        clamp_y = True
    else:
        j = 0
        for idy in range(ny - 1):
            if y_arr[idy] <= y <= y_arr[idy + 1]:
                j = idy
                break
        ty = (
            (y - y_arr[j])
            /
            (y_arr[j + 1] - y_arr[j])
        )
    # --------------------------------------------------------
    # Cell values
    # grid convention = grid[y_index, x_index]
    # --------------------------------------------------------
    f00 = grid[j,     i]
    f10 = grid[j,     i + 1]
    f01 = grid[j + 1, i]
    f11 = grid[j + 1, i + 1]
    dx = x_arr[i + 1] - x_arr[i]
    dy = y_arr[j + 1] - y_arr[j]
    # --------------------------------------------------------
    # Bilinear value
    # --------------------------------------------------------
    value = (
        f00 * (1.0 - tx) * (1.0 - ty)
        + f10 * tx * (1.0 - ty)
        + f01 * (1.0 - tx) * ty
        + f11 * tx * ty
    )
    # --------------------------------------------------------
    # Exact derivatives of the bilinear interpolant
    # --------------------------------------------------------
    if clamp_x:
        dfdx = 0.0
    else:
        dfdx = (
            (f10 - f00) * (1.0 - ty)
            + (f11 - f01) * ty
        ) / dx
    if clamp_y:
        dfdy = 0.0
    else:
        dfdy = (
            (f01 - f00) * (1.0 - tx)
            + (f11 - f10) * tx
        ) / dy
    return value, dfdx, dfdy
# ============================================================
# Exact ray-surface intersection
# DROP-IN replacement for old function
# ============================================================
@njit(cache=True)
def path_length_exact_numba(
    x1, y1, s,
    z1_map, z2_map,
    dzx2_map, dzy2_map,
    xarr1, xarr2,
    yarr1, yarr2,
    L,
    t0=1.0,
    tol=1e-12,
    max_iter=50,
    G2_G1=False,
    line_search=True
):
    """
    Compute geometric propagation distance between two surfaces.
    DROP-IN replacement for the previous path_length_exact_numba.
    IMPORTANT
    --------
    dzx2_map and dzy2_map are retained in the function signature
    for backward compatibility with the rest of compressor.py,
    but are intentionally NOT used for Newton convergence.
    Newton instead uses the exact derivative of the same bilinear
    interpolation of z2_map that appears in the residual.

    Geometry
    --------
    G2_G1 = False:
        start:  z = z1(x,y)
        target: z = L - z2(x,y)
        residual:
            f = z1 + t*sz - (L-z2)

    G2_G1 = True:
        start:  z = L - z1(x,y)
        target: z = z2(x,y)
        residual:
            f = (L-z1) + t*sz - z2
    Returns
    -------
    """
    sx = np.float64(s[0])
    sy = np.float64(s[1])
    sz = np.float64(s[2])
    L = np.float64(L)
    # ========================================================
    # Starting-surface height
    # Constant throughout Newton iteration
    # ========================================================
    use_fixed_z1 = (
        z1_map is None
        or (
            z1_map.shape[0] == 1
            and z1_map.shape[1] == 1
        )
    )
    if use_fixed_z1:
        if z1_map is None:
            z1_val = 0.0
        else:
            z1_val = np.float64(z1_map[0, 0])
    else:
        z1_val = np.float64(
            bilinear_interpolate(
                x1,
                y1,
                z1_map,
                xarr1,
                yarr1
            )
        )
    # ========================================================
    # Initial guess
    # ========================================================
    if t0 is None:
        if G2_G1:
            # Approximate target z = 0
            #
            # (L-z1) + t sz = 0
            t = -(L - z1_val) / sz
        else:
            # Approximate target z = L
            #
            # z1 + t sz = L
            t = (L - z1_val) / sz
    else:
        t = np.float64(t0)
    # ========================================================
    # Newton iterations
    # ========================================================
    for it in range(max_iter):
        # ----------------------------------------------------
        # Current ray position
        # ----------------------------------------------------
        x2 = x1 + t * sx
        y2 = y1 + t * sy
        # ----------------------------------------------------
        # Target surface value AND exact derivative
        # of the same bilinear interpolant
        # ----------------------------------------------------
        z2_val, dzdx, dzdy = bilinear_value_grad_numba(
            x2,
            y2,
            z2_map,
            xarr2,
            yarr2
        )
        # ----------------------------------------------------
        # Residual and exact derivative df/dt
        # ----------------------------------------------------
        if G2_G1:
            # --------------------------------------------
            # G2 -> G1
            #
            # (L-z1) + t*sz = z2
            # --------------------------------------------
            f = (
                (L - z1_val)
                + t * sz
                - z2_val
            )
            df = (
                sz
                - sx * dzdx
                - sy * dzdy
            )
        else:
            # --------------------------------------------
            # G1 -> G2
            #
            # z1 + t*sz = L-z2
            # --------------------------------------------
            f = (
                z1_val
                + t * sz
                - (L - z2_val)
            )
            df = (
                sz
                + sx * dzdx
                + sy * dzdy
            )
        # ----------------------------------------------------
        # Residual convergence
        # ----------------------------------------------------
        if abs(f) < tol:
            return (
                t,
                x2,
                y2,
                z1_val,
                z2_val
            )
        # ----------------------------------------------------
        # Safeguard
        # ----------------------------------------------------
        if abs(df) < 1e-15:
            if sz >= 0.0:
                df = 1e-15
            else:
                df = -1e-15
        # ----------------------------------------------------
        # Full Newton step
        # ----------------------------------------------------
        dt = f / df
        alpha = 1.0
        # ====================================================
        # Backtracking line search
        #
        # Only damp Newton when the full step makes
        # |f| worse.
        # ====================================================
        if line_search:
            f_abs = abs(f)
            while alpha > (1.0 / 1024.0):
                t_trial = t - alpha * dt
                x_trial = x1 + t_trial * sx
                y_trial = y1 + t_trial * sy
                z_trial, _, _ = bilinear_value_grad_numba(
                    x_trial,
                    y_trial,
                    z2_map,
                    xarr2,
                    yarr2
                )
                if G2_G1:
                    f_trial = (
                        (L - z1_val)
                        + t_trial * sz
                        - z_trial
                    )
                else:
                    f_trial = (
                        z1_val
                        + t_trial * sz
                        - (L - z_trial)
                    )
                # Accept as soon as residual improves
                if abs(f_trial) < f_abs:
                    break
                alpha *= 0.5
        # ----------------------------------------------------
        # Apply Newton step
        # ----------------------------------------------------
        step = alpha * dt
        t -= step
        # ----------------------------------------------------
        # Numerical failure
        # ----------------------------------------------------
        if np.isnan(t) or np.isinf(t):
            raise RuntimeError(
                "Intersection Newton solver produced NaN/Inf"
            )
        # ----------------------------------------------------
        # Step-size convergence
        #
        # Re-evaluate residual after the step.
        # ----------------------------------------------------
        if abs(step) < tol:
            x2 = x1 + t * sx
            y2 = y1 + t * sy
            z2_val, _, _ = bilinear_value_grad_numba(
                x2,
                y2,
                z2_map,
                xarr2,
                yarr2
            )
            if G2_G1:
                f_final = (
                    (L - z1_val)
                    + t * sz
                    - z2_val
                )
            else:
                f_final = (
                    z1_val
                    + t * sz
                    - (L - z2_val)
                )
            if abs(f_final) < tol:
                return (
                    t,
                    x2,
                    y2,
                    z1_val,
                    z2_val
                )
    # ========================================================
    # Failure
    # ========================================================
    raise RuntimeError(
        "Intersection did not converge after max_iter iterations"
    )
@njit(cache=True)#,fastmath=True)
def path_length_roof_mirror_general_numba(x2, y2, z2, s2,
                                       z2_map,dzx2_map,dzy2_map,xarr2, yarr1,yarr2,
                                       theta=np.pi/4.0,
                                       L_max=0.58, s2_ref=None,
                                       pos_ref=None, L_rm=1):
    """
    Compute the ray propagation through a generalized roof mirror and return the
    intersection back onto the G2 surface.

    input:
    ----------
    x2, y2, z2 : float (m) Starting ray coordinates on the G2 surface.
    s2 : array(3,) (unit vector) Incoming ray direction from G2 toward the roof mirror.
    z2_map : 2D array (m) Height map of the G2 surface.
    dzx2_map, dzy2_map : 2D arrays Surface slope maps of G2 along x and y, used for the final Newton-based intersection.

    xarr2 : 1D array (m) x-coordinate array corresponding to z2_map.
    yarr1, yarr2 : 1D arrays (m) y-coordinate arrays. Only yarr2 is used for interpolation on G2.
    theta : float (rad), optional Half-angle between each roof mirror facet and the roof mirror axis.

    L_max : float (m), optional Nominal separation used as the reference distance for the final ray-surface intersection.
    s2_ref : array(3,) (unit vector) Reference propagation direction used to define the roof mirror orientation.
    pos_ref : array(3,) (m) Reference position on G2 used to locate the roof mirror apex.
    L_rm : float (m), optional Distance from the reference position to the roof mirror apex.

    output:
    ----------
    L_total : float (m) Total geometric propagation distance through the roof mirror.
    x2b, y2b : float (m) Coordinates of the final intersection point on G2.
    s2b : array(3,) (unit vector) Ray direction after the two roof mirror reflections.
    r1 : array(3,) (m) First reflection point on the roof mirror.
    r2 : array(3,) (m) Second reflection point on the roof mirror.
    """

    s2_ = s2
    y_vector = np.array((0.0, 1.0, 0.0))
    apex_dir = np.cross(s2_ref, y_vector)
    apex_dir = apex_dir / np.sqrt(np.dot(apex_dir, apex_dir))
    r_g = pos_ref
    u = s2_ref / np.sqrt(np.dot(s2_ref, s2_ref))
    H_y = 0.07    #To adapt with geometry and here should have used 0.07used to be 0.035
    apex_point = r_g + L_rm * u + np.array((0.0, H_y, 0.0))
    e1 = np.array((0.0, 1.0, 0.0))
    e2 = u
    n1 = -np.cos(theta) * e1 - np.sin(theta) * e2
    n2 = np.cos(theta) * e1 - np.sin(theta) * e2
    # --- Step 1: minimal logic for first-hit selection ---
    p0 = np.array((x2, y2, z2))
    den1 = np.dot(s2_, n1)
    den2 = np.dot(s2_, n2)
    t1_cand = np.dot(apex_point - p0, n1) / den1
    t2_cand = np.dot(apex_point - p0, n2) / den2
    # pick the first positive intersection along ray direction
    if t1_cand > 0.0 and (t2_cand <= 0.0 or t1_cand <= t2_cand):
        chosen_n = n1
        t1 = t1_cand
    elif t2_cand > 0.0:
        chosen_n = n2
        t1 = t2_cand
    else:
        chosen_n = n1
        t1 = t1_cand
    r1 = p0 + t1 * s2_
    s1 = s2_ - 2.0 * np.dot(s2_, chosen_n) * chosen_n
    # Step 2: second reflection on the other plane
    n_second = n2 if chosen_n is n1 else n1
    denom2 = np.dot(s1, n_second)
    t2 = np.dot(apex_point - r1, n_second) / denom2
    r2 = r1 + t2 * s1
    s2b = s1 - 2.0 * np.dot(s1, n_second) * n_second
    # Step 3: intersect back with G2 surface
    t0 = (z2 - r2[2]) / s2b[2]  # initial guess
    t3, x2b, y2b, _, _ = path_length_exact_numba(
        r2[0], r2[1], s2b,
        z1_map=np.array([[r2[2]]]),  # dummy, not used here
        z2_map=z2_map,
        dzx2_map=dzx2_map,
        dzy2_map=dzy2_map,
        xarr1=np.zeros(1),
        xarr2=xarr2,
        yarr1=np.zeros(1),
        yarr2=yarr2,
        L=L_max,
        t0=t0
    )
    L_total = np.abs(t1) + np.abs(t2) + np.abs(t3)
    return L_total, x2b, y2b, s2b, r1, r2
@njit(cache=True)
def total_path_length_numba(lam=810e-9, d=1.0/1.8e6, L=0.58,
                         alpha_deg=56.0,
                         s_in_user=None,
                         z1_map=None, z2_map=None,
                         dzx1_map=None, dzy1_map=None, xarr1=None,
                         dzx2_map=None, dzy2_map=None, xarr2=None,
                         yarr1=None, yarr2=None,
                         dux1_map=None, duy1_map=None,
                         dux2_map=None, duy2_map=None,
                         s2_ref=None, pos_ref=None,
                         x1=0.07, y1=0.025, z1_dist=1):
    """
    Compute the total optical path length including:
    Incident plane → G1 → G2 → Roof mirror → G2 → G1 → Incident plane.
    input
    ----------
    lam : float  Wavelength [m]
    d : float Grating spacing [m]
    L : float Separation between G1 and G2 [m]
    alpha_deg : float Incident angle (deg)
    s_in_user : array(3,), optional Custom incoming ray direction
    z1_map, z2_map : 2D arrays Surface height maps for G1 and G2
    dzx1_map, dzy1_map, dzx2_map, dzy2_map : 2D arrays Local surface slopes for G1 and G2
    dux1_map, duy1_map, dux2_map, duy2_map : 2D arrays, optional Derivatives of the horizontal displacement u(x,y). If omitted, grating-pitch variation is disabled.
    xarr1, xarr2, yarr : 1D arrays Grid coordinate arrays
    s2_ref, pos_ref : arrays(3,) Reference direction and position for roof mirror
    x1, y1 : float Starting coordinate on G1
    z1_dist : float Distance between the incident plane reference and the G1 reference

    output
    -------
    total_geom : float Total geometric path length [m]
    r_out array: gives the positions of intersections for each ray, usefull for visualisation  and calculating the grating phase.  
    """
    om = 2.0 * np.pi * C / lam
    alpha = np.deg2rad(alpha_deg)
    # --- Incoming direction ---
    if s_in_user is None:
        s_in = np.zeros(3, dtype=np.float64)
        s_in[0] = np.sin(alpha)
        s_in[1] = 0.0
        s_in[2] = -np.cos(alpha)
    else:
        s_in = s_in_user
    # ======================================================================
    # (1) Incident plane → G1 not counted in the code only used for visualisation
    # ======================================================================
    t_in,x_0,y_0,z_0 = path_from_incident_plane_to_G1_forward(x1, y1, s_in, z1_map,xarr1, yarr1,z1_dist=1.0)
    r_0=np.array([x_0,y_0,z_0])
    #r_1=np.array([x_1,y_1,z1_val])
    # ======================================================================
    # (2) G1 → G2
    # ======================================================================
    s_after_G1 = diffract_reflect_vector_numba(
        s_in, x1, y1,
        dzx1_map, dzy1_map, xarr1,
        dzx2_map, dzy2_map, xarr2,
        yarr1, yarr2, d=d, om=om, G1_diff=True,
        dux1_map=dux1_map, duy1_map=duy1_map,
        dux2_map=dux2_map, duy2_map=duy2_map
    )
    #print("intersection",x1,y1,s_in)
    t12, x2, y2, z1_val, z2_val = path_length_exact_numba(
        x1, y1, s_after_G1,
        z1_map=z1_map,
        z2_map=z2_map,
        dzx2_map=dzx2_map,
        dzy2_map=dzy2_map,
        xarr1=xarr1,
        xarr2=xarr2,
        yarr1=yarr1,
        yarr2=yarr2,
        L=L
    )
    r_1=np.array([x1,y1,z1_val])
    r_2=np.array([x2,y2,L-z2_val])## use to be L+z2_val
    #print("intersection_G2:",x2,y2,s_after_G1)
    L_G1_G2 = np.abs(t12)
    # ======================================================================
    # (3) G2 -> Roof mirror->G2
    # ======================================================================
    s_after_G2 = diffract_reflect_vector_numba(
        s_after_G1, x2, y2,
        dzx1_map, dzy1_map, xarr1,
        dzx2_map, dzy2_map, xarr2,
        yarr1, yarr2, d=d, om=om, G1_diff=False,
        dux1_map=dux1_map, duy1_map=duy1_map,
        dux2_map=dux2_map, duy2_map=duy2_map
    )
    z2_val_total = L-z2_val
    L_rm_total, x2b, y2b, s2b,r_3,r_4 = path_length_roof_mirror_general_numba(
        x2, y2, z2_val_total, s_after_G2,
        z2_map,dzx2_map,dzy2_map, xarr2, yarr1,yarr2,
        theta=np.pi/4.0, L_max=L,
        s2_ref=s2_ref, pos_ref=pos_ref, L_rm=1.5
    )
   # print("intersection_G2_back:",x2b,y2b,s2b)
    # ======================================================================
    # (4) G2 -> G1
    # ======================================================================
    s_after_G2_return = diffract_reflect_vector_numba(
        s2b, x2b, y2b,
        dzx1_map, dzy1_map, xarr1,
        dzx2_map, dzy2_map, xarr2,
        yarr1, yarr2, d=d, om=om, G1_diff=False,
        dux1_map=dux1_map, duy1_map=duy1_map,
        dux2_map=dux2_map, duy2_map=duy2_map
    )
    t_back, x1b, y1b, z2_val_back, z1_val_back = path_length_exact_numba(
        x2b, y2b, s_after_G2_return,
        z1_map=z2_map,
        z2_map=z1_map,
        dzx2_map=dzx1_map,
        dzy2_map=dzy1_map,
        xarr1=xarr2,
        xarr2=xarr1,
        yarr1=yarr2,
        yarr2=yarr1,
        L=L,
        G2_G1=True
    )
    r_5=np.array([x2b,y2b,L-z2_val_back])###use to be L+0
    r_6=np.array([x1b,y1b,z1_val_back])
 #   print("intersection_G1_back:",x2b,y2b,s_after_G2_return)
    L_G2_G1 = np.abs(t_back)
   # print(L_G2_G1)
    # ======================================================================
    # (5) G1 → Incident plane (Return path)
    # ======================================================================
    s_after_G1_return = diffract_reflect_vector_numba(
        s_after_G2_return, x1b, y1b,
        dzx1_map, dzy1_map, xarr1,
        dzx2_map, dzy2_map, xarr2,
        yarr1, yarr2, d=d, om=om, G1_diff=True,
        dux1_map=dux1_map, duy1_map=duy1_map,
        dux2_map=dux2_map, duy2_map=duy2_map
    )
    r_G1_out = np.array([x1b, y1b, z1_val_back], dtype=np.float64)
    t_out, r_plane_out = path_from_G1_to_incident_plane_return(s_in,s_out=s_after_G1_return, r_G1_out =  r_G1_out ,z1_dist=1.0)
 #   print("intersection_rg1out",r_G1_out)
  #  print("r_landing",r_plane_out)
    # ======================================================================
    # (6) Total
    # ======================================================================
#    print("lengths",L_G1_G2,L_rm_total,L_G2_G1,t_out)
    total_geom = t_in+L_G1_G2 + L_rm_total + L_G2_G1 + t_out# t_in +
    #r_out=np.array((r_0,r_1,r_2,r_3,r_4,r_5,r_6,r_plane_out))
    r_out = np.zeros((8, 3), dtype=np.float64)
    r_out[0, :] = r_0
    r_out[1, :] = r_1
    r_out[2, :] = r_2
    r_out[3, :] = r_3
    r_out[4, :] = r_4
    r_out[5, :] = r_5
    r_out[6, :] = r_6
    r_out[7, :] = r_plane_out
    return total_geom,r_out,L_G1_G2, L_rm_total, L_G2_G1, t_out
@njit(parallel=True, cache=True)
def compute_OPL_grid_numba(x_arr, y_arr, lam_arr,
                           z1_map, z2_map, dzx1_map, dzy1_map, xarr1,
                           dzx2_map, dzy2_map, xarr2,
                           yarr1, yarr2,
                           s2_ref, pos_ref,
                           mask,
                           dux1_map=None, duy1_map=None,
                           dux2_map=None, duy2_map=None,
                           d=1.0/1.8e6,
                           alpha_deg=56.0,s_in_user=None):
    """
    Compute OPL over (x, y, λ) grid using interpolated slope maps and external reference for roof mirror.
    mask : 2D bool array, shape (nx, ny)
    True  -> ray-trace this (x, y) point for all wavelengths

    False -> skip it, OPL / r_out / debug stay at 0
    dux1_map, duy1_map, dux2_map, duy2_map : 2D arrays, optional Derivatives of horizontal displacement u(x,y).

    """
    nx = x_arr.shape[0]
    ny = y_arr.shape[0]
    nl = lam_arr.shape[0]
    OPL = np.zeros((nx, ny, nl), dtype=np.float64)
    r_out = np.zeros((nx, ny, nl, 8, 3), dtype=np.float64)
    debug_out = np.zeros((nx, ny, nl, 4), dtype=np.float64)   # <-- NEW
    for i in prange(nx):
        xi = x_arr[i]
        for j in range(ny):
            if not mask[i, j]:
                continue
            yj = y_arr[j]
            for k in range(nl):
                lam = lam_arr[k]
                val, r_plane_out, L_G1_G2, L_rm_total, L_G2_G1, t_out = total_path_length_numba(
                    lam=lam,
                    d=d,
                    L=0.58,
                    alpha_deg=alpha_deg,
                    z1_map=z1_map,
                    z2_map=z2_map,
                    dzx1_map=dzx1_map,
                    dzy1_map=dzy1_map,
                    xarr1=xarr1,
                    dzx2_map=dzx2_map,
                    dzy2_map=dzy2_map,
                    xarr2=xarr2,
                    yarr1=yarr1,
                    yarr2=yarr2,
                    dux1_map=dux1_map,
                    duy1_map=duy1_map,
                    dux2_map=dux2_map,
                    duy2_map=duy2_map,
                    s2_ref=s2_ref,
                    pos_ref=pos_ref,
                    x1=xi,
                    y1=yj,
                    s_in_user=s_in_user
                )
                OPL[i, j, k] = val
                r_out[i, j, k, :] = r_plane_out
                debug_out[i, j, k, 0] = L_G1_G2
                debug_out[i, j, k, 1] = L_rm_total
                debug_out[i, j, k, 2] = L_G2_G1
                debug_out[i, j, k, 3] = t_out
    return OPL, r_out, debug_out
def build_groove_number_map(dux_map, x_arr, d=1.0/1.8e6):
    """

    Build the continuous groove-number field N(x,y) = integral dx / [d * (1 + du/dx)]
    using a trapezoidal integration along x.
    dux_map must use the same [y, x] convention as the surface maps.
    The arbitrary reference is N(x_arr[0], y) = 0.

    """
    dux_map = np.asarray(dux_map, dtype=np.float64)
    x_arr = np.asarray(x_arr, dtype=np.float64)
    if dux_map.ndim != 2:
        raise ValueError("dux_map must be a 2D array with shape (ny, nx)")
    if dux_map.shape[1] != x_arr.size:
        raise ValueError("dux_map.shape[1] must match x_arr.size")
    pitch_factor = 1.0 + dux_map
    if np.any(pitch_factor <= 0.0):
        raise ValueError("Invalid horizontal deformation: 1 + du/dx <= 0")
    rho = 1.0 / (d * pitch_factor)
    N_map = np.zeros_like(dux_map, dtype=np.float64)
    dx = np.diff(x_arr)
    N_map[:, 1:] = np.cumsum(
        0.5 * (rho[:, :-1] + rho[:, 1:]) * dx[None, :],
        axis=1
    )
    return N_map
@njit(parallel=True, cache=True)
def compute_grating_phase_numba(
    r_out,
    N1_map, N2_map,
    xarr1, yarr1,
    xarr2, yarr2,
    mask
):
    """

    Grating phase for the four diffraction events.



    The diffraction orders are

        G1 first pass : m = -1

        G2 first pass : m = +1

        G2 return     : m = +1

        G1 return     : m = -1



    N1_map and N2_map are continuous groove-number maps, not integer

    groove indices. For a uniform grating, N = (x - x_ref)/d.

    """
    nx = r_out.shape[0]
    ny = r_out.shape[1]
    nl = r_out.shape[2]
    phi_g = np.zeros((nx, ny, nl), dtype=np.float64)
    for i in prange(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            for k in range(nl):
                x1a = r_out[i, j, k, 1, 0]
                y1a = r_out[i, j, k, 1, 1]
                x2a = r_out[i, j, k, 2, 0]
                y2a = r_out[i, j, k, 2, 1]
                x2b = r_out[i, j, k, 5, 0]
                y2b = r_out[i, j, k, 5, 1]
                x1b = r_out[i, j, k, 6, 0]
                y1b = r_out[i, j, k, 6, 1]
                N1a = bilinear_interpolate(
                    x1a, y1a, N1_map, xarr1, yarr1
                )
                N2a = bilinear_interpolate(
                    x2a, y2a, N2_map, xarr2, yarr2
                )
                N2b = bilinear_interpolate(
                    x2b, y2b, N2_map, xarr2, yarr2
                )
                N1b = bilinear_interpolate(
                    x1b, y1b, N1_map, xarr1, yarr1
                )
                phi_g[i, j, k] = 2.0 * np.pi * (
                    -N1a + N2a + N2b - N1b
                )
    return phi_g
class PulseCompressor:
    def __init__(self, d=1.0/1.8e6, L=0.58, alpha_deg=56.0):
        self.d = d
        self.L = L
        self.alpha_deg = alpha_deg
    def compute_opl(self, *args, **kwargs):
        OPL, r_out, debug_out = compute_OPL_grid_numba(*args, **kwargs)
        # Same argument order as compute_OPL_grid_numba.
        # Positional arguments are used when present; otherwise keywords/defaults.
        xarr1 = args[7] if len(args) > 7 else kwargs["xarr1"]
        xarr2 = args[10] if len(args) > 10 else kwargs["xarr2"]
        yarr1 = args[11] if len(args) > 11 else kwargs["yarr1"]
        yarr2 = args[12] if len(args) > 12 else kwargs["yarr2"]
        mask   = args[15] if len(args) > 15 else kwargs["mask"]
        dux1_map = args[16] if len(args) > 16 else kwargs.get("dux1_map", None)
        dux2_map = args[18] if len(args) > 18 else kwargs.get("dux2_map", None)
        d = args[20] if len(args) > 20 else kwargs.get("d", self.d)
        # No horizontal displacement -> uniform groove-number field N=(x-x_ref)/d.
        if dux1_map is None:
            dux1_for_N = np.zeros((yarr1.size, xarr1.size), dtype=np.float64)
        else:
            dux1_for_N = dux1_map
        if dux2_map is None:
            dux2_for_N = np.zeros((yarr2.size, xarr2.size), dtype=np.float64)
        else:
            dux2_for_N = dux2_map
        N1_map = build_groove_number_map(dux1_for_N, xarr1, d=d)
        N2_map = build_groove_number_map(dux2_for_N, xarr2, d=d)
        phi_grating = compute_grating_phase_numba(
            r_out,
            N1_map, N2_map,
            xarr1, yarr1,
            xarr2, yarr2,
            mask
        )
        return OPL, r_out, debug_out, phi_grating
