import numpy as np
import scipy.constants as sc
from scipy.interpolate import RectBivariateSpline
from numba import njit

def get_phi(OPL0,phi_grating_0,OPLstc,phi_grating_stc,lams):
    "return phi omega from OPL"
    AAA=OPL0-OPLstc
    phi_lamb=2*np.pi/(lams)*AAA
    phi_grating=phi_grating_0-phi_grating_stc
    omegas = 2 * np.pi * sc.c / lams  # [rad/s]
    sort_idx = np.argsort(omegas)
    omegas = omegas[sort_idx]
    phi_omega_sorted = phi_lamb[:,:,sort_idx]
    phi_grating_omega=phi_grating[:,:,sort_idx]
    phi_omega=phi_omega_sorted+phi_grating_omega
    return phi_omega

def gauss2D(x, y, wx=0.018 /np.cos(np.deg2rad(56)), wy=0.018, x0=0.07, y0=0.025, offset=0, order=1):
    "return amplitude envelope "
    gauss = np.exp(
                   -(((x - x0) / wx) ** 2 +
                    ((y - y0) / wy)**2)**order) + offset
    return gauss
def recenter(image,n_x0,n_y0):
    intensity_data=np.zeros_like(image)
    n_y, n_x = np.shape(intensity_data)
    intensity_data = np.roll(
            np.roll(image, -int(n_x0 - n_x / 2), axis=1),
                -int(n_y0 - n_y / 2),
                axis=0,
            )
    return intensity_data

def spectum(lams, fwhm, lambda0=810e-9, offset=0, order=1, int_FWHM=True):
    """
    """
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    gauss = np.exp(-np.log(2) * coeff *
                   ((2 * (lams - lambda0) / fwhm)**2)**order) + offset
    #gauss=np.exp(-0.5*((lams - lambda0)/fwhm)**4) 
    omegas = 2*np.pi*sc.c / lams
    sort_idx = np.argsort(omegas)
    omegas = omegas[sort_idx]
    # Reorder S_lambda along omega
    S_omega = gauss[sort_idx]
    return S_omega,gauss

@njit(cache=True)
def pad_array_3d(arr, pad_factor):
    nx, ny, nλ = arr.shape
    nx_pad = int(pad_factor * nx)
    ny_pad = int(pad_factor * ny)
    padded = np.zeros((nx_pad, ny_pad, nλ), dtype=arr.dtype)
    startx = (nx_pad - nx)//2
    starty = (ny_pad - ny)//2
    padded[startx:startx+nx, starty:starty+ny, :] = arr
    return padded
@njit(cache=True)
def crop_array_3d(arr, pad_factor):
    """
    Crop a 3D array that was padded by pad_array_3d().

    Parameters
    ----------
    arr : np.ndarray or cp.ndarray
        Input array of shape (nx_pad, ny_pad, nλ).
    pad_factor : float
        The same pad factor used in pad_array_3d().

    Returns
    -------
    cropped : same type as arr
        Cropped array of shape (nx, ny, nλ).
    """

    nx_pad, ny_pad, nλ = arr.shape
    nx = int(nx_pad / pad_factor)
    ny = int(ny_pad / pad_factor)
    startx = (nx_pad - nx)//2
    starty = (ny_pad - ny)//2
    cropped = arr[startx:startx+nx, starty:starty+ny, :]
    return cropped
    
def crop_array_2d(arr, pad_factor):
    """
    Crop a 2D array that was padded by a corresponding pad function.

    Parameters
    ----------
    arr : np.ndarray or cp.ndarray
        Input array of shape (nx_pad, ny_pad).
    pad_factor : float
        The same pad factor used when padding (e.g., 2, 4, etc.).

    Returns
    -------
    cropped : same type as arr
        Cropped array of shape (nx, ny).
    """
    nx_pad, ny_pad = arr.shape
    nx = int(nx_pad / pad_factor)
    ny = int(ny_pad / pad_factor)

    startx = (nx_pad - nx) // 2
    starty = (ny_pad - ny) // 2

    cropped = arr[startx:startx+nx, starty:starty+ny]
    return cropped

@njit(cache=True)
def pad_array_omega(arr, pad_factor):
    """
    Pad a 3D array (Nx, Ny, Nw) along the omega axis for finer temporal resolution.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (Nx, Ny, Nw)
    pad_factor : int
        Factor by which to extend the omega axis (e.g., 4)

    Returns
    -------
    padded : np.ndarray
        Zero-padded array of shape (Nx, Ny, pad_factor*Nw)
    start_idx : int
        Starting index of the inserted spectrum (useful for cropping)
    """
    Nx, Ny, Nw = arr.shape
    Nw_pad = pad_factor * Nw
    padded = np.zeros((Nx, Ny, Nw_pad), dtype=np.complex128)
    start_idx = (Nw_pad - Nw) // 2
    for i in range(Nx):
        for j in range(Ny):
            padded[i, j, start_idx:start_idx + Nw] = arr[i, j, :]
    return padded, start_idx


@njit(cache=True)
def crop_array_omega(arr, pad_factor):
    """
    Reverse the omega-axis padding from pad_array_omega().

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (Nx, Ny, Nw_pad)
    pad_factor : int
        Same pad factor used for padding

    Returns
    -------
    cropped : np.ndarray
        Cropped array of shape (Nx, Ny, Nw_pad/pad_factor)
    """
    Nx, Ny, Nw_pad = arr.shape
    Nw = Nw_pad // pad_factor
    start_idx = (Nw_pad - Nw) // 2
    cropped = np.zeros((Nx, Ny, Nw), dtype=arr.dtype)
    for i in range(Nx):
        for j in range(Ny):
            cropped[i, j, :] = arr[i, j, start_idx:start_idx + Nw]
    return cropped