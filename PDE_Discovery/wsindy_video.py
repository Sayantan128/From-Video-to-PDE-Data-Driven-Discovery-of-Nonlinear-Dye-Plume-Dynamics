"""
wsindy_video
============

Single-file library for weak-form SINDy on spatio-temporal scalar fields
extracted from video. GPU-accelerated via PyTorch.

Contents
--------
Section 1. Configuration dataclasses
Section 2. Video I/O                 (was video_processing.py)
Section 3. Grid utilities            (was grid_utils.py)
Section 4. Drift estimation          (was drift_estimation.py)
Section 5. Weak-form feature library (was weak_form_library.py)  [GPU]
Section 6. Sparse regression         (was sindy_optimizer.py)
Section 7. PDE simulator             (was pde_simulator.py)      [GPU]
Section 8. Diagnostics               (was diagnostics.py)

Public API
----------
VideoConfig, load_video, infer_grid, gaussian_widths_from_grid,
estimate_velocities, center_of_mass,
Term, default_terms, build_weak_system, WeakSystem,
stlsq, fit_weak_sindy, SindyFit, stability_study,
rollout, one_step,
mse_over_time, relative_rmse, front_radius_error, com_error,
plot_snapshots, plot_mse_over_time
"""
from __future__ import annotations

# ---- stdlib ---------------------------------------------------------------
import os
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterable, List, Optional,
                    Sequence, Tuple)

# ---- third-party ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize as skresize
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
import matplotlib.animation as animation
import time
# ============================================================================
# Section 1. Configuration dataclasses
# ============================================================================
@dataclass
class VideoConfig:
    """Container for all preprocessing hyperparameters.

    Attributes
    ----------
    crop_box : (x0, y0, w, h) or None
        Pixel-space crop window. If None, no crop is performed.
    target_size : (H, W) or None
        Target height / width after resizing. If None, native size is kept.
    border_margin : int
        Pixels stripped from each side *after* cropping and *before* resizing.
    invert : bool
        If True, output = 1 - I/255 (dark -> high intensity).
    smoothing_sigma : float
        Gaussian smoothing sigma (pixels) applied frame-by-frame; 0 disables.
    dtype : numpy dtype
        Output dtype.
    """
    crop_box: Optional[Tuple[int, int, int, int]] = None
    target_size: Optional[Tuple[int, int]] = None
    border_margin: int = 0
    invert: bool = True
    smoothing_sigma: float = 1.0
    dtype: np.dtype = np.float32


@dataclass
class Grid:
    """Physical grid attached to a ``U: (ny, nx, nt)`` scalar field."""
    ny: int
    nx: int
    nt: int
    dy: float
    dx: float
    dt: float
    y: np.ndarray      # (ny,)
    x: np.ndarray      # (nx,)
    t: np.ndarray      # (nt,)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.ny, self.nx, self.nt)


# ============================================================================
# Section 2. Video I/O
# ============================================================================
def list_frame_files(frames_dir: str,
                     exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp",
                                              ".tif", ".tiff")) -> List[str]:
    """Return a sorted list of absolute paths to image files in ``frames_dir``."""
    names = sorted(f for f in os.listdir(frames_dir)
                   if f.lower().endswith(exts))
    return [os.path.join(frames_dir, n) for n in names]


def load_grayscale_stack(frame_paths: List[str]) -> np.ndarray:
    """Load a list of images into a (nt, H, W) uint8 array in grayscale."""
    frames = []
    for p in frame_paths:
        with Image.open(p) as img:
            frames.append(np.asarray(img.convert("L"), dtype=np.uint8))
    arr = np.stack(frames, axis=0)                 # (nt, H, W)
    return arr


def preprocess(stack_uint8: np.ndarray,
               cfg: VideoConfig) -> np.ndarray:
    """
    Apply crop → border-strip → normalize/invert → resize → denoise.

    Parameters
    ----------
    stack_uint8 : (nt, H, W) uint8
    cfg : VideoConfig

    Returns
    -------
    U : (ny, nx, nt) float array (axis 0 = rows/y, axis 1 = cols/x, axis 2 = t)
    """
    if stack_uint8.ndim != 3:
        raise ValueError("Expected (nt, H, W); got shape " +
                         str(stack_uint8.shape))

    nt, H, W = stack_uint8.shape

    # ---- 1. crop --------------------------------------------------------
    if cfg.crop_box is not None:
        x0, y0, w, h = cfg.crop_box
        if x0 + w > W or y0 + h > H:
            raise ValueError(f"Crop window {cfg.crop_box} exceeds frame {H}x{W}")
        stack_uint8 = stack_uint8[:, y0:y0 + h, x0:x0 + w]

    # ---- 2. strip border ------------------------------------------------
    m = int(cfg.border_margin)
    if m > 0:
        stack_uint8 = stack_uint8[:, m:-m, m:-m]

    # ---- 3. normalize + optional invert --------------------------------
    I = stack_uint8.astype(cfg.dtype) / np.float32(255.0)
    if cfg.invert:
        I = np.float32(1.0) - I

    # ---- 4. resize ------------------------------------------------------
    if cfg.target_size is not None:
        H_out, W_out = cfg.target_size
        out = np.empty((nt, H_out, W_out), dtype=cfg.dtype)
        for k in range(nt):
            out[k] = skresize(I[k], (H_out, W_out), anti_aliasing=True,
                              preserve_range=True).astype(cfg.dtype)
        I = out

    # ---- 5. denoise -----------------------------------------------------
    if cfg.smoothing_sigma > 0:
        for k in range(I.shape[0]):
            I[k] = gaussian_filter(I[k], sigma=float(cfg.smoothing_sigma))

    # ---- 6. transpose to (ny, nx, nt) ----------------------------------
    U = np.transpose(I, (1, 2, 0)).astype(cfg.dtype)
    return U


def load_video(frames_dir: str,
               cfg: Optional[VideoConfig] = None) -> np.ndarray:
    """Convenience entry point: list → load → preprocess."""
    cfg = cfg if cfg is not None else VideoConfig()
    paths = list_frame_files(frames_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"No image files in {frames_dir}")
    stack = load_grayscale_stack(paths)
    return preprocess(stack, cfg)

# -----------------------------------------------------------------------------
# Section 2b. Preprocessing diagnostics
# -----------------------------------------------------------------------------
def get_processed_single_frame(frame_path: str,
                               cfg: VideoConfig) -> np.ndarray:
    """
    Load one frame and apply the same preprocessing steps used before smoothing.

    Steps
    -----
    grayscale conversion -> crop -> border removal -> intensity inversion
    -> resizing.

    This helper is intended for preprocessing diagnostics, especially for
    previewing the effect of different Gaussian smoothing strengths before
    committing to a final ``VideoConfig.smoothing_sigma``.
    """
    with Image.open(frame_path) as img:
        arr = np.asarray(img.convert("L"), dtype=cfg.dtype) / np.float32(255.0)

    # Crop
    if cfg.crop_box is not None:
        x0, y0, w, h = cfg.crop_box
        arr = arr[y0:y0 + h, x0:x0 + w]

    # Remove border
    m = int(cfg.border_margin)
    if m > 0:
        arr = arr[m:-m, m:-m]

    # Invert intensity if dye is dark
    if cfg.invert:
        arr = np.float32(1.0) - arr

    # Resize
    if cfg.target_size is not None:
        H_out, W_out = cfg.target_size
        arr = skresize(
            arr,
            (H_out, W_out),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(cfg.dtype)

    return arr.astype(cfg.dtype, copy=False)


def preview_smoothing_effect(frame_paths: Sequence[str],
                             cfg: VideoConfig,
                             frame_index: Optional[int] = None,
                             sigmas: Sequence[float] = (0.0, 0.5, 1.0, 1.5,
                                                        2.0, 3.0),
                             cmap: str = "viridis") -> Dict[str, object]:
    """
    Visualize the effect of Gaussian smoothing on one representative frame.

    Important
    ---------
    This diagnostic does not modify the actual data array ``U``. It is only
    used to choose and justify ``VideoConfig.smoothing_sigma``.

    Returns
    -------
    dict
        Contains ``frame_index``, the unsmoothed frame ``arr0``, the list of
        ``sigmas``, the corresponding ``smoothed_frames``, and a diagnostics
        DataFrame with RMS / max differences relative to sigma = 0.
    """
    if len(frame_paths) == 0:
        raise ValueError("frame_paths is empty")

    if frame_index is None:
        frame_index = len(frame_paths) // 2
    frame_index = int(np.clip(frame_index, 0, len(frame_paths) - 1))

    sigmas = tuple(float(s) for s in sigmas)
    arr0 = get_processed_single_frame(frame_paths[frame_index], cfg)

    smoothed_frames = []
    for sigma in sigmas:
        if sigma > 0:
            arr_smooth = gaussian_filter(arr0, sigma=sigma)
        else:
            arr_smooth = arr0.copy()
        smoothed_frames.append(arr_smooth)

    # Plot 1: smoothed images with fixed color scale
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4))
    if len(sigmas) == 1:
        axes = [axes]

    for ax, sigma, arr_smooth in zip(axes, sigmas, smoothed_frames):
        im = ax.imshow(arr_smooth, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(f"$\\sigma = {sigma}$")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Smoothed frames with fixed color scale", y=1.03)
    plt.tight_layout()
    plt.show()

    # Plot 2: difference from unsmoothed frame
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4))
    if len(sigmas) == 1:
        axes = [axes]

    max_abs_diff = max(
        float(np.max(np.abs(arr_smooth - arr0))) for arr_smooth in smoothed_frames
    )
    if max_abs_diff == 0:
        max_abs_diff = 1e-12

    for ax, sigma, arr_smooth in zip(axes, sigmas, smoothed_frames):
        diff = arr_smooth - arr0
        im = ax.imshow(diff, cmap="coolwarm",
                       vmin=-max_abs_diff, vmax=max_abs_diff)
        ax.set_title(f"$u_\\sigma - u_0$, $\\sigma={sigma}$")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Difference from unsmoothed frame", y=1.03)
    plt.tight_layout()
    plt.show()

    # Plot 3: center-line intensity profile
    center_row = arr0.shape[0] // 2
    plt.figure(figsize=(8, 5))
    for sigma, arr_smooth in zip(sigmas, smoothed_frames):
        plt.plot(arr_smooth[center_row, :], label=f"$\\sigma={sigma}$")

    plt.xlabel("x-index")
    plt.ylabel("Intensity")
    plt.title("Center-line intensity profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Quantitative diagnostics
    rows = []
    for sigma, arr_smooth in zip(sigmas, smoothed_frames):
        diff = arr_smooth - arr0
        rows.append({
            "sigma": sigma,
            "rms_difference": float(np.sqrt(np.mean(diff ** 2))),
            "max_difference": float(np.max(np.abs(diff))),
        })
    df_diag = pd.DataFrame(rows)

    print(f"Frame index: {frame_index}")
    print(f"Crop box: {cfg.crop_box}")
    print(f"Target size: {cfg.target_size}")
    print(f"Invert: {cfg.invert}")
    print()
    print("Smoothing diagnostics relative to sigma = 0:")
    for row in rows:
        print(
            f"sigma = {row['sigma']:>4}: "
            f"RMS difference = {row['rms_difference']:.6e}, "
            f"max difference = {row['max_difference']:.6e}"
        )

    return {
        "frame_index": frame_index,
        "arr0": arr0,
        "sigmas": sigmas,
        "smoothed_frames": smoothed_frames,
        "diagnostics": df_diag,
    }


# ============================================================================
# Section 3. Grid utilities
# ============================================================================
def infer_grid(U: np.ndarray,
               Lx: Optional[float] = None,
               Ly: Optional[float] = None,
               total_time: Optional[float] = None,
               dx: Optional[float] = None,
               dy: Optional[float] = None,
               dt: Optional[float] = None) -> Grid:
    """
    Build a Grid from the shape of ``U`` and (at most) one of
    ``(L, total_time)`` or ``(spacing)`` per axis.

    If no physical extent is given on an axis, a unit spacing is assumed.
    """
    if U.ndim != 3:
        raise ValueError("U must be 3-D (ny, nx, nt); got " + str(U.shape))
    ny, nx, nt = U.shape

    if dx is None:
        dx = Lx / (nx - 1) if Lx else 1.0
    if dy is None:
        dy = Ly / (ny - 1) if Ly else 1.0
    if dt is None:
        dt = total_time / (nt - 1) if total_time else 1.0

    x = np.arange(nx, dtype=np.float64) * float(dx)
    y = np.arange(ny, dtype=np.float64) * float(dy)
    t = np.arange(nt, dtype=np.float64) * float(dt)

    return Grid(ny=ny, nx=nx, nt=nt,
                dy=float(dy), dx=float(dx), dt=float(dt),
                y=y, x=x, t=t)


def gaussian_widths_from_grid(grid: Grid,
                              frac_y: float = 0.06,
                              frac_x: float = 0.06,
                              frac_t: float = 0.025
                              ) -> Tuple[float, float, float]:
    """
    Default test-function scales (sigma) from the grid size.

    Defaults ≈6% of each spatial axis, ≈2.5% of the time axis; tune
    empirically on your own data.
    """
    sy = max(1.0, frac_y * grid.ny)
    sx = max(1.0, frac_x * grid.nx)
    st = max(1.0, frac_t * grid.nt)
    return sy, sx, st


# ============================================================================
# Section 4. Drift estimation
# ============================================================================
def center_of_mass(U: np.ndarray,
                   grid: Grid,
                   eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intensity-weighted center of mass for each frame.

    Returns
    -------
    xc, yc : (nt,) arrays
        Physical-unit (not pixel) positions. NaN for frames with total
        mass <= eps.
    """
    ny, nx, nt = U.shape
    if (ny, nx, nt) != grid.shape:
        raise ValueError("U shape does not match grid")

    Y, X = np.meshgrid(grid.y, grid.x, indexing="ij")   # (ny, nx)
    mass = U.sum(axis=(0, 1))                           # (nt,)

    safe = mass > eps
    xc = np.full(nt, np.nan, dtype=np.float64)
    yc = np.full(nt, np.nan, dtype=np.float64)

    num_x = (U * X[:, :, None]).sum(axis=(0, 1))
    num_y = (U * Y[:, :, None]).sum(axis=(0, 1))

    xc[safe] = num_x[safe] / mass[safe]
    yc[safe] = num_y[safe] / mass[safe]
    return xc, yc


def smooth_and_differentiate(xc: np.ndarray,
                             dt: float,
                             window_frac: float = 0.08,
                             polyorder: int = 3
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Savitzky-Golay smoothing followed by numerical differentiation.

    ``window_frac`` is the fraction of ``len(xc)`` used as the SG window
    (rounded to the nearest odd integer ≥ polyorder+2).
    """
    n = len(xc)
    win = max(polyorder + 2, int(round(window_frac * n)))
    if win % 2 == 0:
        win += 1
    win = min(win, n - (n + 1) % 2)         # ensure win <= n and odd

    # interpolate any NaN frames linearly before filtering
    good = np.isfinite(xc)
    if not good.all():
        idx = np.arange(n)
        xc = np.interp(idx, idx[good], xc[good])

    xs = savgol_filter(xc, window_length=win, polyorder=polyorder)
    vx = np.gradient(xs, dt)
    return xs, vx


def estimate_velocities(U: np.ndarray,
                        grid: Grid,
                        window_frac: float = 0.08,
                        polyorder: int = 3
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: COM → smooth → differentiate for both x and y."""
    xc, yc = center_of_mass(U, grid)
    _, vx = smooth_and_differentiate(xc, grid.dt, window_frac, polyorder)
    _, vy = smooth_and_differentiate(yc, grid.dt, window_frac, polyorder)
    return vx.astype(np.float32), vy.astype(np.float32)


# ============================================================================
# Section 5. Weak-form feature library (GPU)
# ============================================================================
def _gauss_kernel(sigma: float, half_width: int,
                  device: torch.device, dtype: torch.dtype,
                  order: int = 0) -> torch.Tensor:
    """
    1-D discrete Gaussian (or its k-th derivative w.r.t. its argument).

    The 0-th-order kernel is normalized to sum to 1; higher-order kernels
    inherit that normalization via the analytical derivative formula.
    """
    r = torch.arange(-half_width, half_width + 1,
                     device=device, dtype=dtype)
    g = torch.exp(-0.5 * (r / sigma) ** 2)
    g = g / g.sum()                              # normalize: sum(g) = 1

    if order == 0:
        return g
    elif order == 1:
        # d/dc of g(x - c) = (x - c)/sigma^2 * g ; sign flips for d/dx.
        return -(r / sigma ** 2) * g
    elif order == 2:
        return ((r ** 2 - sigma ** 2) / sigma ** 4) * g
    else:
        raise ValueError(f"order {order} not supported")


def _kernel_half_width(sigma: float, k: float = 4.0) -> int:
    """Half-width that captures >=99.99% of Gaussian mass (k-sigma rule)."""
    return max(1, int(np.ceil(k * sigma)))


def _conv1d_along(x: torch.Tensor, kernel: torch.Tensor, axis: int
                  ) -> torch.Tensor:
    """
    Apply a 1-D Gaussian (or its derivative) along a single axis of a 3-D
    tensor using ``torch.nn.functional.conv1d`` with 'same'-style edge
    replication.
    """
    if x.ndim != 3:
        raise ValueError("expected 3-D tensor")
    # bring the target axis to the last position
    perm = [a for a in range(3) if a != axis] + [axis]
    inv  = [perm.index(a) for a in range(3)]
    xp   = x.permute(*perm).contiguous()
    shape_lead = xp.shape[:-1]
    N = xp.shape[-1]

    # (B, 1, N)
    xb = xp.reshape(-1, 1, N)
    K = kernel.numel()
    pad = K // 2
    w = kernel.view(1, 1, K).to(xb.dtype).to(xb.device)
    xb = F.pad(xb, (pad, pad), mode="replicate")
    yb = F.conv1d(xb, w)
    y  = yb.reshape(*shape_lead, N).permute(*inv).contiguous()
    return y


def gaussian_smooth_3d(A: torch.Tensor,
                       sigmas: Tuple[float, float, float],
                       orders: Tuple[int, int, int] = (0, 0, 0),
                       half_widths: Optional[Tuple[int, int, int]] = None,
                       ) -> torch.Tensor:
    """
    Separable 3-D Gaussian convolution on a (ny, nx, nt) tensor.

    ``orders[i]`` lets you take derivative-of-Gaussian along each axis.
    """
    sy, sx, st = sigmas
    oy, ox, ot = orders
    if half_widths is None:
        hy = _kernel_half_width(sy)
        hx = _kernel_half_width(sx)
        ht = _kernel_half_width(st)
    else:
        hy, hx, ht = half_widths

    dtype, device = A.dtype, A.device
    gy = _gauss_kernel(sy, hy, device, dtype, order=oy)
    gx = _gauss_kernel(sx, hx, device, dtype, order=ox)
    gt = _gauss_kernel(st, ht, device, dtype, order=ot)

    A = _conv1d_along(A, gy, axis=0)
    A = _conv1d_along(A, gx, axis=1)
    A = _conv1d_along(A, gt, axis=2)
    return A


def central_dx(U: torch.Tensor, dx: float) -> torch.Tensor:
    """2nd-order central ∂U/∂x with replicate edges. U: (ny, nx, nt)."""
    pad = torch.nn.functional.pad
    Up = pad(U.permute(0, 2, 1), (1, 1), mode="replicate").permute(0, 2, 1)
    return (Up[:, 2:, :] - Up[:, :-2, :]) / (2.0 * dx)


def central_dy(U: torch.Tensor, dy: float) -> torch.Tensor:
    """2nd-order central ∂U/∂y with replicate edges. U: (ny, nx, nt)."""
    pad = torch.nn.functional.pad
    Up = pad(U.permute(1, 2, 0), (1, 1), mode="replicate").permute(2, 0, 1)
    return (Up[2:, :, :] - Up[:-2, :, :]) / (2.0 * dy)


@dataclass
class Term:
    """
    A single candidate feature.

    ``name``    : human-readable label
    ``kind``    : 'strong' (Gaussian smooths the given field),
                  'ibp_lap' (weak Laplacian via -<grad phi, grad U>),
                  or 'ibp_adv' (weak advection via -<v . grad phi, U>).
    ``builder`` : callable (U, Ux, Uy) -> (ny, nx, nt) tensor; used only for
                  'strong' kind.
    """
    name: str
    kind: str = "strong"
    builder: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                               torch.Tensor]] = None


def default_terms() -> List[Term]:
    """
    Candidate library for a dye-plume / reaction-diffusion-like system.

    Includes: constant, u, u^2, |grad u|^2, u|grad u|^2, Laplacian (IBP).
    Advection terms are added separately by ``build_weak_system`` when
    velocities are supplied.
    """
    return [
        Term("1",           "strong",  lambda U, Ux, Uy: torch.ones_like(U)),
        Term("u",           "strong",  lambda U, Ux, Uy: U),
        Term("u^2",         "strong",  lambda U, Ux, Uy: U * U),
        Term("|grad u|^2",  "strong",  lambda U, Ux, Uy: Ux * Ux + Uy * Uy),
        Term("u|grad u|^2", "strong",  lambda U, Ux, Uy: U * (Ux * Ux + Uy * Uy)),
        Term("Delta u",     "ibp_lap"),
    ]


def _sample_centers(ny: int, nx: int, nt: int,
                    sy: float, sx: float, st: float,
                    M: int, k_sigma: float,
                    device: torch.device,
                    generator: Optional[torch.Generator] = None
                    ) -> torch.Tensor:
    """
    Sample M centers uniformly from the interior rectangle

        [k*s_y, ny-k*s_y) x [k*s_x, nx-k*s_x) x [k*s_t, nt-k*s_t)

    so that the Gaussian support is contained in [0, ny-1] x ...  (and thus
    the IBP identity has no boundary error).

    Returns (M, 3) int64 tensor of (iy, ix, it) grid indices.
    """
    ly = int(np.ceil(k_sigma * sy));  hy = ny - ly
    lx = int(np.ceil(k_sigma * sx));  hx = nx - lx
    lt = int(np.ceil(k_sigma * st));  ht = nt - lt
    if not (hy > ly and hx > lx and ht > lt):
        raise ValueError(
            f"Interior too small for s=({sy},{sx},{st}), k={k_sigma}; "
            f"grid={ny}x{nx}x{nt}. Lower widths or k_sigma.")
    iy = torch.randint(ly, hy, (M,), generator=generator, device=device)
    ix = torch.randint(lx, hx, (M,), generator=generator, device=device)
    it = torch.randint(lt, ht, (M,), generator=generator, device=device)
    return torch.stack([iy, ix, it], dim=1)     # (M, 3)


@dataclass
class WeakSystem:
    Theta: np.ndarray            # (M, nT)
    b: np.ndarray                # (M,)
    feature_names: List[str]
    centers: np.ndarray          # (M, 3)
    sigmas: Tuple[float, float, float]
    dx: float
    dy: float
    dt: float


def build_weak_system(U: np.ndarray,
                      grid: Grid,
                      *,
                      sigmas: Tuple[float, float, float],
                      M: int = 500,
                      k_sigma: float = 4.0,
                      vx: Optional[np.ndarray] = None,
                      vy: Optional[np.ndarray] = None,
                      terms: Optional[Sequence[Term]] = None,
                      include_advection: bool = True,
                      device: Optional[str] = None,
                      dtype: torch.dtype = torch.float32,
                      seed: int = 0) -> WeakSystem:
    """
    Build Theta (feature matrix), b (u_t RHS via IBP), and feature names.

    Parameters
    ----------
    U : (ny, nx, nt) float array
    grid : Grid
    sigmas : (s_y, s_x, s_t) Gaussian test-function widths (grid cells).
    M : number of test functions to sample.
    k_sigma : interior-buffer in multiples of sigma (default 4).
    vx, vy : (nt,) velocities. If given and ``include_advection``, two
             advection features are added; otherwise they are treated as
             known and subtracted from the RHS.
    terms : custom term list (default = default_terms()).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    sy, sx, st = sigmas
    dy, dx, dt = grid.dy, grid.dx, grid.dt

    # Move U to GPU once
    U_t = torch.as_tensor(U, dtype=dtype, device=dev)

    # Spatial derivatives (still needed for strong nonlinear terms + IBP Lap)
    Ux = central_dx(U_t, dx)
    Uy = central_dy(U_t, dy)

    # ---- 1. sample centers ---------------------------------------------
    gen = torch.Generator(device=dev).manual_seed(seed)
    centers = _sample_centers(grid.ny, grid.nx, grid.nt,
                              sy, sx, st, M, k_sigma, dev, gen)
    iy, ix, it = centers[:, 0], centers[:, 1], centers[:, 2]

    # ---- 2. RHS:  b = -< d_t phi, u > ----------------------------------
    # Convolve U with (g_y, g_x, g_t') and sample; minus sign gives <phi, u_t>.
    conv_dtU = gaussian_smooth_3d(U_t, (sy, sx, st), orders=(0, 0, 1))
    b = -conv_dtU[iy, ix, it]

    # If velocities are provided *and* user wants them treated as known,
    # move the advection term onto the RHS: b <- b - <phi, v . grad u>
    if (vx is not None) and (vy is not None) and (not include_advection):
        vx_t = torch.as_tensor(vx, dtype=dtype, device=dev)     # (nt,)
        vy_t = torch.as_tensor(vy, dtype=dtype, device=dev)
        vxU = vx_t[None, None, :] * Ux
        vyU = vy_t[None, None, :] * Uy
        adv = gaussian_smooth_3d(vxU, (sy, sx, st), orders=(0, 0, 0)) + \
              gaussian_smooth_3d(vyU, (sy, sx, st), orders=(0, 0, 0))
        b = b - adv[iy, ix, it]

    # ---- 3. Theta columns ----------------------------------------------
    term_list = list(terms) if terms is not None else default_terms()
    feat_names: List[str] = []
    cols: List[torch.Tensor] = []

    for tm in term_list:
        if tm.kind == "strong":
            f = tm.builder(U_t, Ux, Uy)
            conv_f = gaussian_smooth_3d(f, (sy, sx, st), orders=(0, 0, 0))
            cols.append(conv_f[iy, ix, it])
            feat_names.append(tm.name)
        elif tm.kind == "ibp_lap":
            # <phi, Delta u> = -<d_x phi, u_x> - <d_y phi, u_y>
            cx = gaussian_smooth_3d(Ux, (sy, sx, st), orders=(0, 1, 0))
            cy = gaussian_smooth_3d(Uy, (sy, sx, st), orders=(1, 0, 0))
            lap = -(cx + cy)
            cols.append(lap[iy, ix, it])
            feat_names.append(tm.name)
        else:
            raise ValueError(f"unknown term kind: {tm.kind}")

    # ---- 4. Advection columns (when included as unknown terms) ---------
    if (vx is not None) and (vy is not None) and include_advection:
        vx_t = torch.as_tensor(vx, dtype=dtype, device=dev)
        vy_t = torch.as_tensor(vy, dtype=dtype, device=dev)
        # <phi, vx u_x> = -<d_x phi, vx u>  (vx depends only on t)
        f_x = vx_t[None, None, :] * U_t
        f_y = vy_t[None, None, :] * U_t
        cx = gaussian_smooth_3d(f_x, (sy, sx, st), orders=(0, 1, 0))
        cy = gaussian_smooth_3d(f_y, (sy, sx, st), orders=(1, 0, 0))
        cols.append(-cx[iy, ix, it]); feat_names.append("v_x · u_x")
        cols.append(-cy[iy, ix, it]); feat_names.append("v_y · u_y")

    Theta = torch.stack(cols, dim=1)                    # (M, nT)

    return WeakSystem(
        Theta=Theta.detach().cpu().numpy().astype(np.float64),
        b=b.detach().cpu().numpy().astype(np.float64),
        feature_names=feat_names,
        centers=centers.detach().cpu().numpy(),
        sigmas=(sy, sx, st),
        dx=dx, dy=dy, dt=dt,
    )


# ============================================================================
# Section 6. Sparse regression
# ============================================================================
def stlsq(Theta: np.ndarray, b: np.ndarray,
          *, threshold: float = 1e-3, alpha: float = 1e-6,
          max_iter: int = 50, normalize_columns: bool = True
          ) -> np.ndarray:
    """
    Classical Brunton-Proctor-Kutz sequentially-thresholded least squares.

    Parameters
    ----------
    Theta : (M, K)
    b     : (M,) or (M, 1)
    threshold : coefficients with |coef| < threshold are zeroed each iteration
    alpha : Ridge regularizer (L2) used at every least-squares solve
    normalize_columns : rescale Theta columns to unit norm before solving;
                        coefficients are rescaled back at the end.
    """
    b = np.asarray(b).reshape(-1)
    Theta = np.asarray(Theta)
    K = Theta.shape[1]

    if normalize_columns:
        col_norms = np.linalg.norm(Theta, axis=0)
        col_norms[col_norms == 0] = 1.0
        T = Theta / col_norms
    else:
        T = Theta
        col_norms = np.ones(K)

    active = np.ones(K, dtype=bool)
    coef = np.zeros(K)

    for _ in range(max_iter):
        if not active.any():
            break
        sub = T[:, active]
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(sub, b)
        c = ridge.coef_.copy()
        kill = np.abs(c) < threshold
        if not kill.any():
            coef[active] = c
            break
        # zero out below-threshold columns and loop
        new_active = active.copy()
        idx_active = np.where(active)[0]
        new_active[idx_active[kill]] = False
        if np.array_equal(new_active, active):
            coef[active] = c
            break
        active = new_active

    # one final clean fit
    if active.any():
        sub = T[:, active]
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(sub, b)
        coef = np.zeros(K)
        coef[active] = ridge.coef_

    # undo column normalization
    coef = coef / col_norms
    return coef


@dataclass
class SindyFit:
    coef: np.ndarray
    feature_names: List[str]

    def print(self, tol: float = 0.0) -> None:
        print("u_t =")
        parts = []
        for c, name in zip(self.coef, self.feature_names):
            if abs(c) > tol:
                parts.append(f"  {c:+.4g} * {name}")
        if not parts:
            print("  0")
        else:
            print("\n".join(parts))


def fit_weak_sindy(system: WeakSystem,
                   threshold: float = 1e-3,
                   alpha: float = 1e-6,
                   max_iter: int = 50) -> SindyFit:
    coef = stlsq(system.Theta, system.b,
                 threshold=threshold, alpha=alpha, max_iter=max_iter)
    return SindyFit(coef=coef, feature_names=list(system.feature_names))


def stability_study(U: np.ndarray,
                    grid: Grid,
                    *,
                    sigmas: Tuple[float, float, float],
                    M: int = 500,
                    runs: int = 100,
                    vx: Optional[np.ndarray] = None,
                    vy: Optional[np.ndarray] = None,
                    include_advection: bool = True,
                    terms: Optional[Sequence[Term]] = None,
                    threshold: float = 1e-3,
                    alpha: float = 1e-6,
                    device: Optional[str] = None,
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeat the full (sample-centers -> STLSQ) pipeline ``runs`` times with
    different seeds, reporting how often each feature is selected and its
    coefficient's mean / std across selecting runs.
    """
    per_run = []
    names: Optional[List[str]] = None
    for k in range(runs):
        system = build_weak_system(U, grid,
                                   sigmas=sigmas, M=M,
                                   vx=vx, vy=vy,
                                   include_advection=include_advection,
                                   terms=terms, device=device,
                                   seed=k)
        if names is None:
            names = list(system.feature_names)
        fit = fit_weak_sindy(system, threshold=threshold, alpha=alpha)
        rec = {"run": k}
        for n, c in zip(names, fit.coef):
            rec[f"coef::{n}"] = float(c)
            rec[f"sel::{n}"]  = bool(abs(c) > 0)
        per_run.append(rec)

    df_runs = pd.DataFrame(per_run)

    rows = []
    for n in names:
        sel = df_runs[f"sel::{n}"].values.astype(bool)
        coefs = df_runs[f"coef::{n}"].values
        rows.append({
            "term": n,
            "selection_freq": float(sel.mean()),
            "coef_mean_selected": (float(coefs[sel].mean())
                                   if sel.any() else np.nan),
            "coef_std_selected":  (float(coefs[sel].std(ddof=1))
                                   if sel.sum() > 1 else 0.0),
            "n_selected": int(sel.sum()),
            "n_runs": int(runs),
        })
    summary = pd.DataFrame(rows).sort_values("selection_freq",
                                             ascending=False).reset_index(drop=True)
    return df_runs, summary


# ============================================================================
# Section 7. PDE simulator (GPU)
# ============================================================================
def _pad2d(U: torch.Tensor) -> torch.Tensor:
    """Replicate-pad 2-D field by 1 on every side. U: (ny, nx)."""
    return torch.nn.functional.pad(U.unsqueeze(0).unsqueeze(0),
                                    (1, 1, 1, 1), mode="replicate"
                                    ).squeeze(0).squeeze(0)


def central_grad(U: torch.Tensor, dx: float, dy: float):
    Up = _pad2d(U)
    Ux = (Up[1:-1, 2:] - Up[1:-1, :-2]) / (2.0 * dx)
    Uy = (Up[2:, 1:-1] - Up[:-2, 1:-1]) / (2.0 * dy)
    return Ux, Uy


def central_laplacian(U: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    Up = _pad2d(U)
    Uxx = (Up[1:-1, 2:] - 2 * Up[1:-1, 1:-1] + Up[1:-1, :-2]) / (dx * dx)
    Uyy = (Up[2:, 1:-1] - 2 * Up[1:-1, 1:-1] + Up[:-2, 1:-1]) / (dy * dy)
    return Uxx + Uyy


def upwind_advection(U: torch.Tensor, vx: float, vy: float,
                     dx: float, dy: float) -> torch.Tensor:
    """Return -(v_x u_x + v_y u_y) using 1st-order upwind."""
    Up = _pad2d(U)
    Uc = Up[1:-1, 1:-1]
    Ux_plus  = (Up[1:-1, 2:]  - Uc) / dx
    Ux_minus = (Uc - Up[1:-1, :-2]) / dx
    Uy_plus  = (Up[2:,  1:-1] - Uc) / dy
    Uy_minus = (Uc - Up[:-2, 1:-1]) / dy
    Ux = Ux_minus if vx >= 0 else Ux_plus
    Uy = Uy_minus if vy >= 0 else Uy_plus
    return -(vx * Ux + vy * Uy)


def rhs(U: torch.Tensor,
        vx: float, vy: float,
        dx: float, dy: float,
        coefs: Dict[str, float],
        eps_visc: float = 0.0) -> torch.Tensor:
    """
    Generic RHS. ``coefs`` may contain any subset of the keys below:

        'u'            : linear source  c * u
        '1'            : constant       c
        'grad2'        : c * |grad u|^2
        'u_grad2'      : c * u * |grad u|^2
        'lap'          : c * Delta u
        'u_lap'        : c * u * Delta u
        'adv'          : include upwind advection term -(v.grad u) with
                         unit coefficient.  Pass c=1.0 to enable, 0 to disable.

    ``eps_visc`` adds a small artificial diffusion term (numerical stabilizer).
    """
    Ux, Uy = central_grad(U, dx, dy)
    out = torch.zeros_like(U)
    grad2 = Ux * Ux + Uy * Uy
    lap = None

    if coefs.get("adv", 1.0):
        out = out + coefs.get("adv", 1.0) * upwind_advection(U, vx, vy, dx, dy)
    if "1" in coefs and coefs["1"] != 0.0:
        out = out + coefs["1"]
    if "u" in coefs and coefs["u"] != 0.0:
        out = out + coefs["u"] * U
    if "grad2" in coefs and coefs["grad2"] != 0.0:
        out = out + coefs["grad2"] * grad2
    if "u_grad2" in coefs and coefs["u_grad2"] != 0.0:
        out = out + coefs["u_grad2"] * U * grad2
    if "lap" in coefs and coefs["lap"] != 0.0:
        if lap is None:
            lap = central_laplacian(U, dx, dy)
        out = out + coefs["lap"] * lap
    if "u_lap" in coefs and coefs["u_lap"] != 0.0:
        if lap is None:
            lap = central_laplacian(U, dx, dy)
        out = out + coefs["u_lap"] * U * lap
    if eps_visc != 0.0:
        if lap is None:
            lap = central_laplacian(U, dx, dy)
        out = out + eps_visc * lap
    return out


def _cfl_dt(U: torch.Tensor, vx: float, vy: float,
            dx: float, dy: float,
            coefs: Dict[str, float], eps_visc: float, safety: float) -> float:
    umax = float(torch.max(torch.abs(U)).item())
    # diffusion bound
    D_lin = abs(coefs.get("lap", 0.0)) + abs(eps_visc)
    D_nl  = abs(coefs.get("u_lap", 0.0)) * umax
    D_eff = max(D_lin + D_nl, 1e-12)
    dt_diff = 0.5 / (D_eff * (1.0 / dx ** 2 + 1.0 / dy ** 2))
    # advection bound
    dt_adv = np.inf
    if abs(vx) > 1e-12: dt_adv = min(dt_adv, dx / abs(vx))
    if abs(vy) > 1e-12: dt_adv = min(dt_adv, dy / abs(vy))
    # nonlinear gradient^2 - heuristic bound via ||grad u||_inf
    if coefs.get("grad2", 0.0) or coefs.get("u_grad2", 0.0):
        Ux, Uy = central_grad(U, dx, dy)
        gmax = float(torch.max(torch.sqrt(Ux ** 2 + Uy ** 2)).item())
        c = abs(coefs.get("grad2", 0.0)) + abs(coefs.get("u_grad2", 0.0)) * umax
        if c * gmax > 1e-12:
            dt_adv = min(dt_adv, 0.5 / (c * gmax + 1e-12))
    return safety * float(min(dt_diff, dt_adv))


def rollout(U0: np.ndarray,
            t_grid: np.ndarray,
            vx_t: np.ndarray,
            vy_t: np.ndarray,
            dx: float, dy: float,
            coefs: Dict[str, float],
            eps_visc: float = 0.0,
            safety: float = 0.25,
            max_substeps: int = 2000,
            clip: Optional[tuple] = None,
            device: Optional[str] = None,
            dtype: torch.dtype = torch.float32,
            ) -> np.ndarray:
    """
    Heun (RK2) rollout from U0. Returns the spatio-temporal field on
    ``t_grid`` as a NumPy array of shape (ny, nx, nt).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    U = torch.as_tensor(U0, dtype=dtype, device=dev)
    nt = len(t_grid)
    ny, nx = U.shape
    U_hist = torch.zeros((ny, nx, nt), dtype=dtype, device=dev)
    U_hist[:, :, 0] = U

    for k in range(nt - 1):
        Dt = float(t_grid[k + 1] - t_grid[k])
        vx = float(vx_t[k]); vy = float(vy_t[k])
        dt_max = _cfl_dt(U, vx, vy, dx, dy, coefs, eps_visc, safety)
        nsub = max(1, min(max_substeps, int(np.ceil(Dt / max(dt_max, 1e-12)))))
        dt = Dt / nsub
        for _ in range(nsub):
            k1 = rhs(U, vx, vy, dx, dy, coefs, eps_visc)
            U1 = U + dt * k1
            k2 = rhs(U1, vx, vy, dx, dy, coefs, eps_visc)
            U = U + 0.5 * dt * (k1 + k2)
            if clip is not None:
                lo, hi = clip
                U = torch.clamp(U, lo, hi)
        U_hist[:, :, k + 1] = U

    return U_hist.detach().cpu().numpy()


def one_step(U_data: np.ndarray,
             t_grid: np.ndarray,
             vx_t: np.ndarray, vy_t: np.ndarray,
             dx: float, dy: float,
             coefs: Dict[str, float],
             eps_visc: float = 0.0,
             safety: float = 0.25,
             clip: Optional[tuple] = None,
             device: Optional[str] = None,
             dtype: torch.dtype = torch.float32,
             ) -> np.ndarray:
    """
    For each k, integrate from the *data* at time k forward by one frame
    interval and record the result at time k+1.  Diagnostic only (does not
    accumulate error across time).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    U_data_t = torch.as_tensor(U_data, dtype=dtype, device=dev)
    nt = U_data.shape[2]
    U_pred = torch.empty_like(U_data_t)
    U_pred[:, :, 0] = U_data_t[:, :, 0]
    for k in range(nt - 1):
        Dt = float(t_grid[k + 1] - t_grid[k])
        vx = float(vx_t[k]); vy = float(vy_t[k])
        U = U_data_t[:, :, k].clone()
        dt_max = _cfl_dt(U, vx, vy, dx, dy, coefs, eps_visc, safety)
        nsub = max(1, int(np.ceil(Dt / max(dt_max, 1e-12))))
        dt = Dt / nsub
        for _ in range(nsub):
            k1 = rhs(U, vx, vy, dx, dy, coefs, eps_visc)
            U1 = U + dt * k1
            k2 = rhs(U1, vx, vy, dx, dy, coefs, eps_visc)
            U = U + 0.5 * dt * (k1 + k2)
            if clip is not None:
                U = torch.clamp(U, *clip)
        U_pred[:, :, k + 1] = U
    return U_pred.detach().cpu().numpy()


# ============================================================================
# Section 8. Diagnostics
# ============================================================================
def mse_over_time(U_true: np.ndarray, U_pred: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    nt = U_true.shape[2]
    out = np.empty(nt)
    for k in range(nt):
        d = U_true[:, :, k] - U_pred[:, :, k]
        if mask is None:
            out[k] = np.mean(d * d)
        else:
            m = mask[:, :, k] if mask.ndim == 3 else mask
            out[k] = np.mean((d[m]) ** 2) if m.any() else np.nan
    return out


def relative_rmse(U_true: np.ndarray, U_pred: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        num = np.sum((U_true - U_pred) ** 2)
        den = np.sum(U_true ** 2) + 1e-14
    else:
        m = mask if mask.ndim == 3 else mask[:, :, None]
        num = np.sum(((U_true - U_pred) * m) ** 2)
        den = np.sum((U_true * m) ** 2) + 1e-14
    return float(np.sqrt(num / den))


def front_radius_series(U: np.ndarray, *, level: float, grid: Grid
                        ) -> np.ndarray:
    """Equivalent radius sqrt(area/pi) of {U >= level} per frame."""
    cell_area = grid.dx * grid.dy
    nt = U.shape[2]
    radii = np.empty(nt)
    for k in range(nt):
        radii[k] = np.sqrt(np.sum(U[:, :, k] >= level) * cell_area / np.pi)
    return radii


def front_radius_error(U_true: np.ndarray, U_pred: np.ndarray, *,
                       levels: Sequence[float], grid: Grid) -> Dict[str, np.ndarray]:
    """Multi-level front-radius MAE / RMSE."""
    per_level = {}
    mae, rmse = [], []
    for lev in levels:
        rT = front_radius_series(U_true, level=lev, grid=grid)
        rP = front_radius_series(U_pred, level=lev, grid=grid)
        signed = rP - rT
        mae.append(float(np.mean(np.abs(signed))))
        rmse.append(float(np.sqrt(np.mean(signed ** 2))))
        per_level[f"level_{lev:.3f}"] = dict(
            r_true=rT, r_pred=rP, signed_err=signed)
    return {
        "levels": np.array(levels, dtype=float),
        "per_level": per_level,
        "mae_mean":  float(np.mean(mae)),
        "rmse_mean": float(np.mean(rmse)),
    }


def com_series(U: np.ndarray, grid: Grid, eps: float = 1e-12):
    Y, X = np.meshgrid(grid.y, grid.x, indexing="ij")
    mass = U.sum(axis=(0, 1))
    safe = mass > eps
    xc = np.where(safe, (U * X[:, :, None]).sum(axis=(0, 1)) / (mass + eps), np.nan)
    yc = np.where(safe, (U * Y[:, :, None]).sum(axis=(0, 1)) / (mass + eps), np.nan)
    return xc, yc


def com_error(U_true: np.ndarray, U_pred: np.ndarray, grid: Grid) -> Dict[str, float]:
    xT, yT = com_series(U_true, grid)
    xP, yP = com_series(U_pred, grid)
    err = np.sqrt((xP - xT) ** 2 + (yP - yT) ** 2)
    return dict(mae=float(np.nanmean(err)),
                rmse=float(np.sqrt(np.nanmean(err ** 2))),
                err_t=err, xT=xT, yT=yT, xP=xP, yP=yP)


def plot_snapshots(U_true: np.ndarray, U_pred: np.ndarray,
                   t_grid: np.ndarray, n_show: int = 4,
                   cmap: str = "viridis"):
    nt = U_true.shape[2]
    idxs = np.linspace(0, nt - 1, n_show, dtype=int)
    for k in idxs:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(U_true[:, :, k], cmap=cmap)
        axes[0].set_title(f"True t={t_grid[k]:.2f}"); plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(U_pred[:, :, k], cmap=cmap,
                              vmin=U_true[:, :, k].min(), vmax=U_true[:, :, k].max())
        axes[1].set_title(f"Pred t={t_grid[k]:.2f}"); plt.colorbar(im1, ax=axes[1])
        err = U_pred[:, :, k] - U_true[:, :, k]
        m = np.max(np.abs(err)) + 1e-12
        im2 = axes[2].imshow(err, cmap="bwr", vmin=-m, vmax=m)
        axes[2].set_title("Pred - True");           plt.colorbar(im2, ax=axes[2])
        for a in axes: a.set_axis_off()
        plt.tight_layout(); plt.show()


def plot_mse_over_time(U_true: np.ndarray, U_pred: np.ndarray,
                       t_grid: np.ndarray, mask: Optional[np.ndarray] = None,
                       label: str = "rollout"):
    m = mse_over_time(U_true, U_pred, mask)
    plt.figure()
    plt.semilogy(t_grid, m, label=f"{label} MSE(t)")
    plt.xlabel("t"); plt.ylabel("MSE"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()
    return m



# ============================================================================
# Section 9. Parameter bootstrap (rollout-MSE minimization)
# ============================================================================
#
# Purpose: given a *fixed* PDE structure (keys in `param_names` below),
# quantify how tightly the data constrain the continuous coefficients by
# refitting them on chronologically-resampled time blocks.
#
# Differences from the original notebook:
#   * True block bootstrap -- blocks stay contiguous (no idx.sort()).
#   * Uses the GPU rollout() from Section 7, not a CPU Numba kernel.
#   * Initial guess defaults to the STLSQ estimate, not a hardcoded value.
#   * Returns percentile CIs and the full parameter-correlation matrix.
#   * Distinguishes "converged" from "iter-cap reached" via res.success.

from scipy.optimize import minimize


def time_block_indices(nt: int,
                       block_len: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Chronological block bootstrap: sample ceil(nt/block_len) blocks of
    length `block_len` with random start positions, concatenate them,
    and truncate to `nt` frames.

    CRITICAL: the returned indices are NOT sorted. Sorting would collapse
    the block structure into subsampling (a silent bug in the original
    notebook's `time_block_indices`).

    Returns
    -------
    idx : (nt,) int array, may contain duplicates and non-monotonic runs.
    """
    n_blocks = int(np.ceil(nt / block_len))
    starts = rng.integers(0, nt - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:nt]


def _coefs_from_vector(x: np.ndarray,
                       param_names: Sequence[str],
                       positive_log: Sequence[str]) -> Dict[str, float]:
    """
    Map the optimizer's unconstrained vector back to a coefficient dict.
    Parameters named in `positive_log` are treated as log-transformed
    (enforces positivity without bound constraints).
    """
    out = {}
    for xi, n in zip(x, param_names):
        out[n] = float(np.exp(xi)) if n in positive_log else float(xi)
    return out


def _vector_from_coefs(coefs: Dict[str, float],
                       param_names: Sequence[str],
                       positive_log: Sequence[str]) -> np.ndarray:
    return np.array(
        [np.log(max(coefs[n], 1e-12)) if n in positive_log else coefs[n]
         for n in param_names],
        dtype=np.float64)


def masked_subsampled_mse(U_true: np.ndarray, U_pred: np.ndarray,
                          mask: Optional[np.ndarray],
                          max_points: int,
                          rng: np.random.Generator) -> float:
    """
    Sampled MSE used as the bootstrap objective. Subsampling makes the
    objective cheaper without materially changing its minimizer.
    """
    if mask is None:
        diff = (U_true - U_pred).ravel()
    else:
        diff = (U_true[mask] - U_pred[mask]).ravel()
    n = diff.size
    if max_points and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        diff = diff[idx]
    return float(np.mean(diff * diff))


def fit_coefficients_by_rollout_mse(
    U: np.ndarray,
    t_grid: np.ndarray,
    vx_t: np.ndarray,
    vy_t: np.ndarray,
    grid: Grid,
    init: Dict[str, float],
    *,
    param_names: Sequence[str],
    positive_log: Sequence[str] = (),
    fixed: Optional[Dict[str, float]] = None,
    mask: Optional[np.ndarray] = None,
    n_sub: int = 100,
    eps_visc: float = 0.01,
    max_points: int = 80_000,
    maxiter: int = 50,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = False,
    nm_verbose: bool = False,
    nm_print_every: int = 10,
    nm_label: str = "",
) -> Dict[str, object]:
    """
    Fit coefficients in `param_names` by minimizing rollout MSE.

    Parameters
    ----------
    U            : (ny, nx, nt) data tensor, possibly a resampled slice.
    t_grid, vx_t, vy_t : matching time axis and velocity series.
    grid         : Grid object, used for dx and dy.
    init         : {name: value} initial guess, typically Weak-SINDy or iPINN.
    param_names  : Which keys in `init` are being optimized.
    positive_log : Subset of `param_names` forced positive via log-transform.
    fixed        : {name: value} coefficients held fixed, e.g. {"adv": 1.0}.
    mask         : Optional boolean mask for error evaluation.
    n_sub        : substeps per frame in the rollout simulator.
    max_points   : MSE is evaluated on this many random pixel-time samples.
    maxiter      : Nelder-Mead iteration cap.
    nm_verbose   : If True, print inner Nelder-Mead progress.
    nm_print_every : Print every this many Nelder-Mead iterations.
    nm_label     : Label printed in Nelder-Mead progress messages.

    Returns
    -------
    dict with keys:
        coefs, mse, success, n_iter, init_mse, n_eval, optimizer_message.
    """

    rng = np.random.default_rng(seed)
    fixed = dict(fixed or {})
    U0 = U[:, :, 0]

    sample_indices = make_fixed_mse_sample_indices(
    U.shape,
    mask=mask,
    max_points=max_points,
    seed=seed,)

    t0_wall = time.perf_counter()

    def _format_duration(seconds: float) -> str:
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds < 0:
            return "--h --m --s"
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}h {int(m):02d}m {s:05.2f}s"

    def build_coefs(x):
        c = dict(fixed)
        c.update(_coefs_from_vector(x, param_names, positive_log))
        return c

    def _format_coefs_from_x(x):
        coefs_now = build_coefs(x)
        return "  ".join(
            f"{name}={float(coefs_now[name]):+.4g}"
            for name in param_names
            if name in coefs_now
        )

    nm_state = {
        "iter": 0,
        "eval": 0,
        "best_mse": np.inf,
        "last_mse": np.nan,
        "last_x": None,
    }

    def objective(x):
        coefs = build_coefs(x)

        U_sim = rollout(
            U0,
            t_grid,
            vx_t,
            vy_t,
            dx=grid.dx,
            dy=grid.dy,
            coefs=coefs,
            eps_visc=eps_visc,
            safety=0.25,
            max_substeps=n_sub * 4,
            clip=clip,
            device=device,
        )

        # mse = masked_subsampled_mse(U, U_sim, mask, max_points, rng)
        mse = fixed_sample_mse(U, U_sim, sample_indices)
        mse = float(mse)

        nm_state["eval"] += 1
        nm_state["last_mse"] = mse
        nm_state["last_x"] = np.asarray(x, dtype=float).copy()

        if mse < nm_state["best_mse"]:
            nm_state["best_mse"] = mse

        return mse

    def nm_callback(xk):
        """
        Called by scipy.optimize.minimize after a Nelder-Mead iteration.
        """
        nm_state["iter"] += 1

        if not nm_verbose:
            return

        if nm_print_every is None or nm_print_every <= 0:
            return

        should_print = (
            nm_state["iter"] == 1
            or nm_state["iter"] % nm_print_every == 0
            or nm_state["iter"] >= maxiter
        )

        if not should_print:
            return

        elapsed = time.perf_counter() - t0_wall
        coef_str = _format_coefs_from_x(xk)

        label = f"{nm_label} | " if nm_label else ""

        print(
            f"    [NM {label}iter {nm_state['iter']:4d}/{maxiter} | "
            f"eval {nm_state['eval']:5d}] "
            f"last_mse={nm_state['last_mse']:.3e}  "
            f"best_mse={nm_state['best_mse']:.3e}  "
            f"{coef_str}  "
            f"elapsed={_format_duration(elapsed)}"
        )

    x0 = _vector_from_coefs(init, param_names, positive_log)

    init_mse = objective(x0)

    if nm_verbose:
        label = f"{nm_label} | " if nm_label else ""
        print(
            f"    [NM {label}start] "
            f"init_mse={init_mse:.3e}  "
            f"{_format_coefs_from_x(x0)}"
        )

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        callback=nm_callback,
        options=dict(
            maxiter=maxiter,
            xatol=1e-2,
            fatol=1e-4,
            disp=False,
        ),
    )

    fitted = build_coefs(res.x)
    total_elapsed = time.perf_counter() - t0_wall

    if verbose or nm_verbose:
        print(
            f"  init MSE={init_mse:.3e} -> final MSE={float(res.fun):.3e}  "
            f"iters={res.nit}  evals={res.nfev}  "
            f"converged={res.success}  "
            f"time={_format_duration(total_elapsed)}"
        )

        if not res.success:
            print(f"  optimizer message: {res.message}")

    return dict(
        coefs=fitted,
        mse=float(res.fun),
        success=bool(res.success),
        n_iter=int(res.nit),
        n_eval=int(res.nfev),
        init_mse=float(init_mse),
        optimizer_message=str(res.message),
    )


# ============================================================
# Front-radius-aware rollout calibration
# ============================================================

def _sigmoid_np(z):
    """
    Numerically stable sigmoid for numpy arrays.
    """
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def front_radius_curves_for_calibration(
    U_field,
    levels,
    grid,
    *,
    time_indices=None,
    soft=True,
    softness=0.02,
):
    """
    Compute equivalent front-radius curves for several threshold levels.

    If soft=True, the superlevel set indicator 1_{u >= level} is replaced by

        sigmoid((u - level) / softness)

    This makes the front-radius objective smoother for Nelder-Mead.
    """
    U_arr = np.asarray(U_field)

    if time_indices is None:
        U_sub = U_arr
    else:
        U_sub = U_arr[:, :, time_indices]

    dx = float(grid.dx)
    dy = float(grid.dy)
    area_cell = dx * dy

    curves = {}

    for lev in levels:
        lev = float(lev)

        if soft:
            weights = _sigmoid_np((U_sub - lev) / float(softness))
            area = np.sum(weights, axis=(0, 1)) * area_cell
        else:
            area = np.sum(U_sub >= lev, axis=(0, 1)) * area_cell

        radius = np.sqrt(np.maximum(area, 0.0) / np.pi)
        curves[lev] = radius

    return curves


def front_radius_calibration_loss(
    U_true,
    U_pred,
    levels,
    grid,
    *,
    time_indices=None,
    soft=True,
    softness=0.02,
    eps=1e-12,
):
    """
    Relative front-radius loss between true and predicted fields.

    The loss is dimensionless:

        mean_{levels,t} [ (r_pred - r_true) / (mean(r_true)+eps) ]^2
    """
    true_curves = front_radius_curves_for_calibration(
        U_true,
        levels,
        grid,
        time_indices=time_indices,
        soft=soft,
        softness=softness,
    )

    pred_curves = front_radius_curves_for_calibration(
        U_pred,
        levels,
        grid,
        time_indices=time_indices,
        soft=soft,
        softness=softness,
    )

    losses = []

    for lev in levels:
        lev = float(lev)

        r_true = true_curves[lev]
        r_pred = pred_curves[lev]

        scale = np.mean(np.abs(r_true)) + eps
        losses.append(np.mean(((r_pred - r_true) / scale) ** 2))

    return float(np.mean(losses))


def front_growth_calibration_loss(
    U_true,
    U_pred,
    levels,
    grid,
    *,
    time_indices=None,
    soft=True,
    softness=0.02,
    eps=1e-12,
):
    """
    Penalize mismatch in total front growth over the calibration window.

    This targets the failure mode where the predicted front radius is too flat.
    """
    true_curves = front_radius_curves_for_calibration(
        U_true,
        levels,
        grid,
        time_indices=time_indices,
        soft=soft,
        softness=softness,
    )

    pred_curves = front_radius_curves_for_calibration(
        U_pred,
        levels,
        grid,
        time_indices=time_indices,
        soft=soft,
        softness=softness,
    )

    losses = []

    for lev in levels:
        lev = float(lev)

        r_true = true_curves[lev]
        r_pred = pred_curves[lev]

        true_growth = r_true[-1] - r_true[0]
        pred_growth = r_pred[-1] - r_pred[0]

        scale = abs(true_growth) + eps
        losses.append(((pred_growth - true_growth) / scale) ** 2)

    return float(np.mean(losses))


def fit_coefficients_by_rollout_front_aware(
    U: np.ndarray,
    t_grid: np.ndarray,
    vx_t: np.ndarray,
    vy_t: np.ndarray,
    grid: Grid,
    init: Dict[str, float],
    *,
    param_names: Sequence[str],
    positive_log: Sequence[str] = (),
    fixed: Optional[Dict[str, float]] = None,
    mask: Optional[np.ndarray] = None,
    n_sub: int = 100,
    eps_visc: float = 0.01,
    max_points: int = 80_000,
    maxiter: int = 200,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    device: Optional[str] = None,
    seed: int = 0,
    front_levels: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    n_front_times: int = 40,
    front_weight: float = 5.0,
    growth_weight: float = 0.05,
    soft_front: bool = True,
    front_softness: float = 0.02,
    verbose: bool = True,
    nm_verbose: bool = False,
    nm_print_every: int = 10,
    nm_label: str = "",
) -> Dict[str, object]:
    """
    Fit coefficients by minimizing a front-radius-aware rollout objective.

    Objective:

        J = data_mse
            + front_weight * front_radius_loss
            + growth_weight * front_growth_loss

    This is useful when pixel-wise RMSE is acceptable but the predicted plume
    front under-expands or remains nearly flat.
    """
    fixed = dict(fixed or {})
    U0 = U[:, :, 0]

    t0_wall = time.perf_counter()

    def _format_duration(seconds: float) -> str:
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds < 0:
            return "--h --m --s"
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}h {int(m):02d}m {s:05.2f}s"

    # Fixed sampled MSE points inside this optimization.
    sample_indices = make_fixed_mse_sample_indices(
        U.shape,
        mask=mask,
        max_points=max_points,
        seed=seed,
    )

    # Fixed time indices for front-radius calibration.
    nt_loc = U.shape[2]
    n_front_times_eff = min(int(n_front_times), nt_loc)
    front_time_indices = np.linspace(
        0,
        nt_loc - 1,
        n_front_times_eff,
        dtype=int,
    )

    def build_coefs(x):
        c = dict(fixed)
        c.update(_coefs_from_vector(x, param_names, positive_log))
        return c

    def _format_coefs_from_x(x):
        coefs_now = build_coefs(x)
        return "  ".join(
            f"{name}={float(coefs_now[name]):+.4g}"
            for name in param_names
            if name in coefs_now
        )

    state = {
        "iter": 0,
        "eval": 0,
        "best_J": np.inf,
        "last_J": np.nan,
        "last_data": np.nan,
        "last_front": np.nan,
        "last_growth": np.nan,
    }

    def objective(x):
        coefs = build_coefs(x)

        U_sim = rollout(
            U0,
            t_grid,
            vx_t,
            vy_t,
            dx=grid.dx,
            dy=grid.dy,
            coefs=coefs,
            eps_visc=eps_visc,
            safety=0.25,
            max_substeps=n_sub * 4,
            clip=clip,
            device=device,
        )

        data_mse = fixed_sample_mse(U, U_sim, sample_indices)

        front_loss = front_radius_calibration_loss(
            U,
            U_sim,
            levels=front_levels,
            grid=grid,
            time_indices=front_time_indices,
            soft=soft_front,
            softness=front_softness,
        )

        growth_loss = front_growth_calibration_loss(
            U,
            U_sim,
            levels=front_levels,
            grid=grid,
            time_indices=front_time_indices,
            soft=soft_front,
            softness=front_softness,
        )

        J = (
            float(data_mse)
            + float(front_weight) * float(front_loss)
            + float(growth_weight) * float(growth_loss)
        )

        state["eval"] += 1
        state["last_J"] = J
        state["last_data"] = float(data_mse)
        state["last_front"] = float(front_loss)
        state["last_growth"] = float(growth_loss)

        if J < state["best_J"]:
            state["best_J"] = J

        return J

    def nm_callback(xk):
        state["iter"] += 1

        if not nm_verbose:
            return

        if nm_print_every is None or nm_print_every <= 0:
            return

        should_print = (
            state["iter"] == 1
            or state["iter"] % nm_print_every == 0
            or state["iter"] >= maxiter
        )

        if not should_print:
            return

        elapsed = time.perf_counter() - t0_wall
        label = f"{nm_label} | " if nm_label else ""

        print(
            f"    [Front-NM {label}iter {state['iter']:4d}/{maxiter} | "
            f"eval {state['eval']:5d}] "
            f"J={state['last_J']:.3e}  "
            f"best_J={state['best_J']:.3e}  "
            f"data={state['last_data']:.3e}  "
            f"front={state['last_front']:.3e}  "
            f"growth={state['last_growth']:.3e}  "
            f"{_format_coefs_from_x(xk)}  "
            f"elapsed={_format_duration(elapsed)}"
        )

    x0 = _vector_from_coefs(init, param_names, positive_log)

    init_J = objective(x0)

    if verbose:
        print(
            f"Front-aware init objective: J={init_J:.6e}  "
            f"data={state['last_data']:.6e}  "
            f"front={state['last_front']:.6e}  "
            f"growth={state['last_growth']:.6e}"
        )
        print("Initial coefficients:", build_coefs(x0))

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        callback=nm_callback,
        options=dict(
            maxiter=maxiter,
            xatol=1e-2,
            fatol=1e-4,
            disp=False,
        ),
    )

    fitted = build_coefs(res.x)
    final_J = float(res.fun)

    # Recompute final component losses cleanly.
    U_sim_final = rollout(
        U0,
        t_grid,
        vx_t,
        vy_t,
        dx=grid.dx,
        dy=grid.dy,
        coefs=fitted,
        eps_visc=eps_visc,
        safety=0.25,
        max_substeps=n_sub * 4,
        clip=clip,
        device=device,
    )

    final_data_mse = fixed_sample_mse(U, U_sim_final, sample_indices)

    final_front_loss = front_radius_calibration_loss(
        U,
        U_sim_final,
        levels=front_levels,
        grid=grid,
        time_indices=front_time_indices,
        soft=soft_front,
        softness=front_softness,
    )

    final_growth_loss = front_growth_calibration_loss(
        U,
        U_sim_final,
        levels=front_levels,
        grid=grid,
        time_indices=front_time_indices,
        soft=soft_front,
        softness=front_softness,
    )

    total_elapsed = time.perf_counter() - t0_wall

    if verbose or nm_verbose:
        print(
            f"Front-aware final objective: J={final_J:.6e}  "
            f"data={final_data_mse:.6e}  "
            f"front={final_front_loss:.6e}  "
            f"growth={final_growth_loss:.6e}"
        )
        print(
            f"iters={res.nit}  evals={res.nfev}  "
            f"converged={res.success}  "
            f"time={_format_duration(total_elapsed)}"
        )
        print("Final coefficients:", fitted)

        if not res.success:
            print("Optimizer message:", res.message)

    return dict(
        coefs=fitted,
        J=final_J,
        data_mse=float(final_data_mse),
        front_loss=float(final_front_loss),
        growth_loss=float(final_growth_loss),
        success=bool(res.success),
        n_iter=int(res.nit),
        n_eval=int(res.nfev),
        init_J=float(init_J),
        optimizer_message=str(res.message),
        U_sim=U_sim_final,
    )



def format_duration(seconds: float) -> str:
    """
    Format seconds as HHh MMm SSs.
    Useful for progress/ETA reporting.
    """
    seconds = float(seconds)

    if not np.isfinite(seconds) or seconds < 0:
        return "--h --m --s"

    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    return f"{int(h):02d}h {int(m):02d}m {s:05.2f}s"



def bootstrap_rollout_parameters(
    U: np.ndarray,
    t_grid: np.ndarray,
    vx_t: np.ndarray,
    vy_t: np.ndarray,
    grid: Grid,
    init: Dict[str, float],
    *,
    param_names: Sequence[str],
    positive_log: Sequence[str] = (),
    fixed: Optional[Dict[str, float]] = None,
    B: int = 50,
    block_len: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    n_sub: int = 100,
    eps_visc: float = 0.01,
    max_points: int = 80_000,
    maxiter: int = 30,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
    nm_verbose: bool = False,
    nm_print_every: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Block-bootstrap parameter estimation.

    For each of B replicates:
      1. Draw chronologically-contiguous time blocks with replacement.
      2. Slice U, t, vx, vy, and mask to those indices.
      3. Refit param_names on the slice via fit_coefficients_by_rollout_mse.

    This version includes:
      - bootstrap-level elapsed-time and ETA reporting;
      - optional inner Nelder-Mead progress reporting.
    """
    rng = np.random.default_rng(seed)
    nt = U.shape[2]

    if block_len is None:
        block_len = max(1, int(round(np.sqrt(nt))))

    rows = []
    t_start_wall = time.perf_counter()

    for b in range(B):
        tidx = time_block_indices(nt, block_len, rng)

        U_b = U[:, :, tidx]
        t_b = t_grid[tidx]
        vx_b = vx_t[tidx]
        vy_b = vy_t[tidx]
        mask_b = None if mask is None else mask[:, :, tidx]

        if verbose:
            print()
            print("-" * 80)
            print(f"Bootstrap replicate {b + 1}/{B}")
            print("-" * 80)

        out = fit_coefficients_by_rollout_mse(
            U_b,
            t_b,
            vx_b,
            vy_b,
            grid,
            init=init,
            param_names=param_names,
            positive_log=positive_log,
            fixed=fixed,
            mask=mask_b,
            n_sub=n_sub,
            eps_visc=eps_visc,
            max_points=max_points,
            maxiter=maxiter,
            clip=clip,
            device=device,
            seed=seed + 1000 + b,
            verbose=False,

            # Inner Nelder-Mead progress
            nm_verbose=nm_verbose,
            nm_print_every=nm_print_every,
            nm_label=f"rep {b + 1}/{B}",
        )

        row = {n: out["coefs"][n] for n in param_names}
        row.update({
            "mse": out["mse"],
            "success": out["success"],
            "n_iter": out["n_iter"],
            "n_eval": out.get("n_eval", np.nan),
            "init_mse": out.get("init_mse", np.nan),
            "optimizer_message": out.get("optimizer_message", ""),
        })

        rows.append(row)

        if verbose:
            elapsed = time.perf_counter() - t_start_wall
            done = b + 1
            avg_per_rep = elapsed / done
            remaining = avg_per_rep * max(B - done, 0)

            summary_str = "  ".join(
                f"{n}={row[n]:+.3g}" for n in param_names
            )

            status = "✓" if out["success"] else "·"

            print(
                f"[{done:>3}/{B}] {status}  "
                f"mse={row['mse']:.3e}  "
                f"{summary_str}  "
                f"iters={row['n_iter']}  "
                f"evals={row['n_eval']}  "
                f"elapsed={format_duration(elapsed)}  "
                f"ETA={format_duration(remaining)}"
            )

    total_elapsed = time.perf_counter() - t_start_wall

    df = pd.DataFrame(rows)

    conv = df["success"].values
    df_ok = df[conv] if conv.any() else df

    percentiles = {
        n: np.percentile(df_ok[n], [2.5, 50, 97.5]).tolist()
        for n in param_names
    }

    summary = dict(
        n_replicates=int(B),
        n_converged=int(conv.sum()),
        mean={
            n: float(df_ok[n].mean())
            for n in param_names
        },
        std={
            n: float(df_ok[n].std(ddof=1)) if len(df_ok) > 1 else 0.0
            for n in param_names
        },
        percentiles=percentiles,
        corr=df_ok[list(param_names)].corr().to_dict(),
        total_elapsed_seconds=float(total_elapsed),
        total_elapsed=format_duration(total_elapsed),
        avg_seconds_per_replicate=float(total_elapsed / max(B, 1)),
    )

    if verbose:
        print()
        print("Bootstrap complete.")
        print(f"Total elapsed time        : {format_duration(total_elapsed)}")
        print(f"Average time per replicate: {format_duration(total_elapsed / max(B, 1))}")
        print(f"Converged replicates      : {int(conv.sum())}/{B}")

    return df, summary


# --- small diagnostic helpers ------------------------------------------------
def plot_bootstrap_pairs(df_boot: pd.DataFrame,
                         param_names: Sequence[str]) -> None:
    """
    Scatter-matrix of bootstrap parameter draws colored by rollout MSE.
    Strong linear trends in a panel indicate parameter non-identifiability
    (the data constrain only a combination, not each coefficient).
    """
    df_ok = df_boot[df_boot["success"]] if df_boot["success"].any() else df_boot
    k = len(param_names)
    if k < 2: return
    mse = df_ok["mse"].values
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4),
                              squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        ni, nj = param_names[i], param_names[j]
        sc = ax.scatter(df_ok[ni], df_ok[nj], c=mse, s=35, cmap="viridis")
        ax.set_xlabel(ni); ax.set_ylabel(nj)
        ax.set_title(f"{ni} vs {nj}  (r = {df_ok[[ni,nj]].corr().iloc[0,1]:+.2f})")
        ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.9, label="rollout MSE")
    plt.tight_layout(); plt.show()


def bootstrap_effective_law(df_boot: pd.DataFrame,
                            linear_key: str, nonlinear_key: str,
                            u_grid: Optional[np.ndarray] = None
                            ) -> pd.DataFrame:
    """
    For a PDE term of the form  (a0 + a1 * u) * |grad u|^2,  compute the
    pointwise posterior of a(u) = a0 + a1 * u across the bootstrap draws.

    Parameters
    ----------
    linear_key    : e.g. "grad2"   (coefficient of |grad u|^2)
    nonlinear_key : e.g. "u_grad2" (coefficient of u|grad u|^2)
    u_grid        : evaluation grid in u. Defaults to 201 points in [0, 1].
    """
    if u_grid is None:
        u_grid = np.linspace(0.0, 1.0, 201)
    df_ok = df_boot[df_boot["success"]] if df_boot["success"].any() else df_boot
    a0 = df_ok[linear_key].values[:, None]
    a1 = df_ok[nonlinear_key].values[:, None]
    A  = a0 + a1 * u_grid[None, :]
    return pd.DataFrame({
        "u": u_grid,
        "mean":    A.mean(axis=0),
        "median":  np.median(A, axis=0),
        "p2_5":    np.percentile(A, 2.5, axis=0),
        "p97_5":   np.percentile(A, 97.5, axis=0),
        "frac_pos": (A > 0).mean(axis=0),
    })

# ============================================================================
# Section 10. Multi-library weak-SINDy fitting, diagnostics, and export
# ============================================================================
def make_candidate_libraries() -> Dict[str, List[Term]]:
    """
    Create the standard candidate PDE libraries used in the notebook workflow.

    Notes
    -----
    Advection terms ``v_x u_x`` and ``v_y u_y`` are not included directly in
    these lists. They are added automatically by ``build_weak_system`` when
    ``vx`` and ``vy`` are supplied and ``include_advection=True``.
    """
    full_previous = default_terms()

    adv_diff = [
        Term("Delta u", "ibp_lap"),
    ]

    adv_diff_linear = [
        Term("u", "strong", lambda U, Ux, Uy: U),
        Term("Delta u", "ibp_lap"),
    ]

    nonlinear_grad = [
        Term("|grad u|^2", "strong", lambda U, Ux, Uy: Ux * Ux + Uy * Uy),
        Term("Delta u", "ibp_lap"),
    ]

    nonlinear_u_grad = [
        Term("u|grad u|^2", "strong",
             lambda U, Ux, Uy: U * (Ux * Ux + Uy * Uy)),
        Term("Delta u", "ibp_lap"),
    ]

    nonlinear_grad_both = [
        Term("|grad u|^2", "strong", lambda U, Ux, Uy: Ux * Ux + Uy * Uy),
        Term("u|grad u|^2", "strong",
             lambda U, Ux, Uy: U * (Ux * Ux + Uy * Uy)),
        Term("Delta u", "ibp_lap"),
    ]

    return {
        "Full previous library": full_previous,
        "A: advection-diffusion": adv_diff,
        "B: advection-diffusion + u": adv_diff_linear,
        "C: advection-diffusion + |grad u|^2": nonlinear_grad,
        "C-alt: advection-diffusion + u|grad u|^2": nonlinear_u_grad,
        "C-both: advection-diffusion + both gradient terms": nonlinear_grad_both,
    }


def safe_filename(name: str) -> str:
    """Convert a display name into a filesystem-safe stem."""
    return (
        str(name).replace(":", "")
        .replace("|", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("^", "pow")
        .replace("·", "dot")
    )


def theta_condition_number(Theta: np.ndarray) -> float:
    """Compute the numerical condition number of the weak feature matrix."""
    return float(np.linalg.cond(Theta))


def theta_column_correlation(Theta: np.ndarray,
                             eps: float = 1e-12) -> np.ndarray:
    """
    Compute normalized feature-column cosine/correlation matrix.

    ``R[i, j] = <Theta_i, Theta_j> / (||Theta_i|| ||Theta_j||)``.
    """
    C = Theta / (np.linalg.norm(Theta, axis=0, keepdims=True) + eps)
    return C.T @ C


def plot_theta_correlation(R: np.ndarray,
                           feature_names: Sequence[str],
                           title: str,
                           save_path: Optional[os.PathLike] = None,
                           show: bool = True,
                           cmap: str = "RdBu_r") -> None:
    """Plot and optionally save a feature-column correlation heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)

    plt.colorbar(im, ax=ax, label="column cosine")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def fit_and_diagnose_library(
    library_name: str,
    terms: Sequence[Term],
    U: np.ndarray,
    grid: Grid,
    sigmas: Tuple[float, float, float],
    vx: Optional[np.ndarray],
    vy: Optional[np.ndarray],
    M: int = 2000,
    k_sigma: float = 4.0,
    include_advection: bool = True,
    device: Optional[str] = None,
    seed: int = 0,
    threshold: float = 1e-3,
    alpha: float = 1e-6,
    max_iter: int = 100,
    make_plot: bool = True,
    show_plot: bool = True,
) -> Dict[str, object]:
    """
    Build the weak system, compute diagnostics, and fit one candidate library.
    """
    print("=" * 80)
    print(f"Library: {library_name}")
    print("=" * 80)

    system = build_weak_system(
        U,
        grid,
        sigmas=sigmas,
        M=M,
        k_sigma=k_sigma,
        vx=vx,
        vy=vy,
        terms=terms,
        include_advection=include_advection,
        device=device,
        seed=seed,
    )

    cond = theta_condition_number(system.Theta)
    R = theta_column_correlation(system.Theta)

    print(f"Theta shape      : {system.Theta.shape}")
    print(f"cond(Theta)      : {cond:.3e}")
    print("Feature names:")
    for name in system.feature_names:
        print(f"  - {name}")

    if make_plot:
        plot_theta_correlation(
            R,
            system.feature_names,
            title=f"Theta column correlations\n{library_name}",
            save_path=None,
            show=show_plot,
        )

    fit = fit_weak_sindy(system, threshold=threshold,
                         alpha=alpha, max_iter=max_iter)

    print()
    print("Discovered PDE:")
    fit.print(tol=0.0)

    return {
        "library_name": library_name,
        "system": system,
        "fit": fit,
        "condition_number": cond,
        "correlation_matrix": R,
        "feature_names": system.feature_names,
    }


def summarize_library_results(all_results: Dict[str, Dict[str, object]],
                              select_tol: float = 1e-12) -> pd.DataFrame:
    """Create a compact summary table across fitted candidate libraries."""
    summary_rows = []

    for library_name, result in all_results.items():
        fit = result["fit"]
        feature_names = result["feature_names"]
        coef = fit.coef

        active_terms = [
            f"{name}: {c:+.4e}"
            for name, c in zip(feature_names, coef)
            if abs(c) > select_tol
        ]
        active_string = ", ".join(active_terms) if active_terms else "None"

        summary_rows.append({
            "library": library_name,
            "Theta_shape": result["system"].Theta.shape,
            "cond_Theta": result["condition_number"],
            "n_features": len(feature_names),
            "active_terms": active_string,
        })

    return pd.DataFrame(summary_rows)


def run_candidate_libraries(
    U: np.ndarray,
    grid: Grid,
    vx: Optional[np.ndarray],
    vy: Optional[np.ndarray],
    *,
    libraries: Optional[Dict[str, Sequence[Term]]] = None,
    sigmas: Optional[Tuple[float, float, float]] = None,
    frac_y: float = 0.06,
    frac_x: float = 0.06,
    frac_t: float = 0.025,
    M: int = 2000,
    k_sigma: float = 4.0,
    include_advection: bool = True,
    device: Optional[str] = None,
    seed: int = 0,
    threshold: float = 1e-3,
    alpha: float = 1e-6,
    max_iter: int = 100,
    select_tol: float = 1e-12,
    make_plots: bool = True,
    show_plots: bool = True,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """
    Fit and diagnose all candidate PDE libraries.

    Returns
    -------
    all_results : dict
        One result dictionary per library.
    df_summary : pandas.DataFrame
        Summary table with condition numbers and selected terms.
    """
    if libraries is None:
        libraries = make_candidate_libraries()
    if sigmas is None:
        sigmas = gaussian_widths_from_grid(grid, frac_y=frac_y,
                                           frac_x=frac_x, frac_t=frac_t)

    print(
        f"Gaussian widths (grid cells): "
        f"sy={sigmas[0]:.2f}, sx={sigmas[1]:.2f}, st={sigmas[2]:.2f}"
    )
    if device is not None:
        print(f"Using device: {device}")

    all_results = {}
    for library_name, terms in libraries.items():
        all_results[library_name] = fit_and_diagnose_library(
            library_name=library_name,
            terms=terms,
            U=U,
            grid=grid,
            sigmas=sigmas,
            vx=vx,
            vy=vy,
            M=M,
            k_sigma=k_sigma,
            include_advection=include_advection,
            device=device,
            seed=seed,
            threshold=threshold,
            alpha=alpha,
            max_iter=max_iter,
            make_plot=make_plots,
            show_plot=show_plots,
        )

    df_summary = summarize_library_results(all_results, select_tol=select_tol)
    return all_results, df_summary


def library_coefficients_dataframe(all_results: Dict[str, Dict[str, object]],
                                   select_tol: float = 1e-12) -> pd.DataFrame:
    """Return one long-form DataFrame of coefficients from all libraries."""
    coef_rows = []
    for library_name, result in all_results.items():
        fit_i = result["fit"]
        for name, c in zip(result["feature_names"], fit_i.coef):
            coef_rows.append({
                "library": library_name,
                "feature": name,
                "coefficient": float(c),
                "abs_coefficient": float(abs(c)),
                "selected": bool(abs(c) > select_tol),
                "condition_number": float(result["condition_number"]),
            })
    return pd.DataFrame(coef_rows)


def write_discovered_pdes_text(all_results: Dict[str, Dict[str, object]],
                               txt_path: os.PathLike,
                               select_tol: float = 1e-12) -> Path:
    """Write readable discovered PDEs for all candidate libraries."""
    txt_path = Path(txt_path)
    with open(txt_path, "w") as f_txt:
        for library_name, result in all_results.items():
            fit_i = result["fit"]
            f_txt.write("=" * 80 + "\n")
            f_txt.write(f"Library: {library_name}\n")
            f_txt.write(f"Condition number: {result['condition_number']:.6e}\n")
            f_txt.write("=" * 80 + "\n")
            f_txt.write("u_t =\n")

            any_term = False
            for c, name in zip(fit_i.coef, fit_i.feature_names):
                if abs(c) > select_tol:
                    f_txt.write(f"  {c:+.8e} * {name}\n")
                    any_term = True
            if not any_term:
                f_txt.write("  0\n")
            f_txt.write("\n\n")
    return txt_path


def save_library_results(
    all_results: Dict[str, Dict[str, object]],
    output_dir: os.PathLike,
    *,
    select_tol: float = 1e-12,
    metadata: Optional[Dict[str, object]] = None,
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
) -> Dict[str, Path]:
    """
    Save multi-library weak-SINDy results to CSV, PNG, NPZ, TXT, and ZIP.

    Parameters
    ----------
    all_results
        Output from ``run_candidate_libraries``.
    output_dir
        Directory where all files will be saved.
    select_tol
        Coefficient magnitude threshold used only for selected/unselected
        reporting.
    metadata
        Optional dictionary written to ``run_metadata.txt``. This is where the
        notebook can pass ``M_TEST``, ``K_SIGMA``, ``DEVICE``, ``U.shape``, etc.
    make_zip
        If True, create a zip archive of all saved outputs.
    zip_path
        Optional explicit location for the zip file. If omitted, the zip file
        is placed next to ``output_dir`` with ``.zip`` suffix.
    download_zip
        If True and running in Google Colab, call ``files.download`` on the zip.

    Returns
    -------
    dict
        Paths for the output directory and, if created, the zip archive.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results to:", output_dir)

    # Summary table
    df_summary = summarize_library_results(all_results, select_tol=select_tol)
    summary_path = output_dir / "library_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    # Per-library coefficient CSVs and combined coefficient CSV
    for library_name, result in all_results.items():
        fit_i = result["fit"]
        feature_names = result["feature_names"]
        coef = fit_i.coef

        df_coef = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "selected": np.abs(coef) > select_tol,
        })

        safe_name = safe_filename(library_name)
        csv_path = output_dir / f"coefficients_{safe_name}.csv"
        df_coef.to_csv(csv_path, index=False)
        print("Saved:", csv_path)

    df_all_coef = library_coefficients_dataframe(all_results,
                                                 select_tol=select_tol)
    all_coef_path = output_dir / "all_library_coefficients.csv"
    df_all_coef.to_csv(all_coef_path, index=False)
    print("Saved:", all_coef_path)

    # Correlation heatmaps
    for library_name, result in all_results.items():
        safe_name = safe_filename(library_name)
        plot_path = output_dir / f"correlation_heatmap_{safe_name}.png"
        plot_theta_correlation(
            result["correlation_matrix"],
            result["feature_names"],
            title=f"Theta column correlations\n{library_name}",
            save_path=plot_path,
            show=False,
        )
        print("Saved:", plot_path)

    # Numerical arrays
    for library_name, result in all_results.items():
        system_i = result["system"]
        fit_i = result["fit"]
        safe_name = safe_filename(library_name)
        npz_path = output_dir / f"numeric_results_{safe_name}.npz"

        np.savez_compressed(
            npz_path,
            Theta=system_i.Theta,
            b=system_i.b,
            correlation_matrix=result["correlation_matrix"],
            coefficients=fit_i.coef,
            feature_names=np.array(system_i.feature_names, dtype=object),
            centers=system_i.centers,
            sigmas=np.array(system_i.sigmas),
            dx=np.array(system_i.dx),
            dy=np.array(system_i.dy),
            dt=np.array(system_i.dt),
            condition_number=np.array(result["condition_number"]),
        )
        print("Saved:", npz_path)

    # Readable PDEs
    txt_path = output_dir / "discovered_pdes.txt"
    write_discovered_pdes_text(all_results, txt_path, select_tol=select_tol)
    print("Saved:", txt_path)

    # Metadata
    if metadata is not None:
        metadata_path = output_dir / "run_metadata.txt"
        with open(metadata_path, "w") as f_meta:
            f_meta.write("Video-to-PDE multi-library weak-SINDy run metadata\n")
            f_meta.write("=" * 80 + "\n")
            for key, value in metadata.items():
                f_meta.write(f"{key} = {value}\n")
        print("Saved:", metadata_path)

    out_paths: Dict[str, Path] = {"output_dir": output_dir}

    # Zip all outputs
    if make_zip:
        if zip_path is None:
            zip_path = output_dir.with_suffix(".zip")
        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob("*"):
                zipf.write(file_path, arcname=file_path.relative_to(output_dir))
        print("Created zip file:", zip_path)
        out_paths["zip_path"] = zip_path

        if download_zip:
            try:
                from google.colab import files  # type: ignore
                files.download(str(zip_path))
            except ImportError:
                print("Not running in Colab. Zip file saved at:", zip_path)

    return out_paths


# ============================================================================
# Section 11. STLSQ threshold sweeps for fitted libraries
# ============================================================================
def run_threshold_sweep_for_system(
    system: WeakSystem,
    library_name: str,
    thresholds: Sequence[float],
    *,
    alpha: float = 1e-6,
    max_iter: int = 100,
    active_tol: float = 1e-12,
) -> pd.DataFrame:
    """
    Run STLSQ repeatedly over a range of threshold values for a fixed weak
    system.

    Notes
    -----
    The weak system itself may already have been built on GPU via
    ``build_weak_system``. The STLSQ solve uses scikit-learn Ridge on CPU,
    because the existing sparse-regression backend is CPU-based.
    """
    rows = []
    for thr in thresholds:
        fit_thr = fit_weak_sindy(
            system,
            threshold=float(thr),
            alpha=alpha,
            max_iter=max_iter,
        )
        coef = np.array(fit_thr.coef, dtype=float)

        row = {
            "library": library_name,
            "threshold": float(thr),
            "n_active": int(np.sum(np.abs(coef) > active_tol)),
        }
        for name, c in zip(system.feature_names, coef):
            row[f"c[{name}]"] = float(c)

        active_terms = [
            f"{name}: {c:+.4e}"
            for name, c in zip(system.feature_names, coef)
            if abs(c) > active_tol
        ]
        row["active_terms"] = ", ".join(active_terms) if active_terms else "None"
        rows.append(row)

    return pd.DataFrame(rows)


def plot_threshold_sweep_coefficients(
    df_sweep: pd.DataFrame,
    feature_names: Sequence[str],
    library_name: str,
    *,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
) -> None:
    """Plot coefficient paths as a function of STLSQ threshold."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in feature_names:
        col = f"c[{name}]"
        if col not in df_sweep.columns:
            continue
        ax.plot(
            df_sweep["threshold"].values,
            df_sweep[col].values,
            "-o",
            ms=3,
            label=name,
        )

    ax.set_xscale("log")
    ax.set_xlabel("STLSQ threshold")
    ax.set_ylabel("coefficient")
    ax.set_title(f"STLSQ Pareto sweep: coefficient vs threshold\n{library_name}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_threshold_sweep_active_count(
    df_sweep: pd.DataFrame,
    library_name: str,
    *,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
) -> None:
    """Plot number of active terms as a function of STLSQ threshold."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(
        df_sweep["threshold"].values,
        df_sweep["n_active"].values,
        "-o",
        ms=4,
    )
    ax.set_xscale("log")
    ax.set_xlabel("STLSQ threshold")
    ax.set_ylabel("number of active terms")
    ax.set_title(f"Active terms vs threshold\n{library_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def pareto_summary_from_sweeps(
    all_sweep_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create a compact Pareto summary table from threshold-sweep outputs."""
    rows = []
    for library_name, df_sweep_i in all_sweep_results.items():
        for _, row in df_sweep_i.iterrows():
            rows.append({
                "library": library_name,
                "threshold": float(row["threshold"]),
                "n_active": int(row["n_active"]),
                "active_terms": row["active_terms"],
            })
    return pd.DataFrame(rows)


def run_threshold_sweeps_for_libraries(
    all_results: Dict[str, Dict[str, object]],
    *,
    thresholds: Optional[Sequence[float]] = None,
    alpha: float = 1e-6,
    max_iter: int = 100,
    active_tol: float = 1e-12,
    output_dir: Optional[os.PathLike] = None,
    save_results: bool = True,
    make_plots: bool = True,
    show_plots: bool = True,
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    """
    Run STLSQ threshold sweeps for every fitted library in ``all_results``.

    Parameters
    ----------
    all_results
        Output from ``run_candidate_libraries``.
    thresholds
        Threshold values. Defaults to ``np.geomspace(1e-5, 1.0, 21)``.
    save_results
        If True, save per-library CSV files, combined CSV files, and plots.
    make_plots
        If True, create coefficient-path and active-count plots.
    show_plots
        If True, display plots in the notebook. If False, plots are saved and
        closed silently.
    make_zip
        If True and ``save_results`` is True, zip all saved sweep outputs.
    download_zip
        If True and running in Colab, download the zip file. Set this to False
        to save outputs without downloading anything.

    Returns
    -------
    all_sweep_results, df_all_sweeps, df_pareto_summary, paths
    """
    if thresholds is None:
        thresholds = np.geomspace(1e-5, 1.0, 21)
    thresholds = np.asarray(thresholds, dtype=float)

    paths: Dict[str, Path] = {}
    if save_results:
        if output_dir is None:
            output_dir = Path("/content/video_to_pde_threshold_sweeps")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths["output_dir"] = output_dir
    else:
        output_dir = None

    all_sweep_results: Dict[str, pd.DataFrame] = {}
    combined_sweep_rows = []

    for library_name, result in all_results.items():
        print("=" * 100)
        print(f"Threshold sweep for library: {library_name}")
        print("=" * 100)

        system_i: WeakSystem = result["system"]
        df_sweep_i = run_threshold_sweep_for_system(
            system=system_i,
            library_name=library_name,
            thresholds=thresholds,
            alpha=alpha,
            max_iter=max_iter,
            active_tol=active_tol,
        )
        all_sweep_results[library_name] = df_sweep_i
        combined_sweep_rows.append(df_sweep_i)

        cols_to_show = ["threshold", "n_active", "active_terms"]
        print(df_sweep_i[cols_to_show].to_string(index=False))

        safe_name = safe_filename(library_name)
        if save_results and output_dir is not None:
            csv_path = output_dir / f"threshold_sweep_{safe_name}.csv"
            df_sweep_i.to_csv(csv_path, index=False)
            print("Saved:", csv_path)

        if make_plots:
            coef_plot_path = None
            active_plot_path = None
            if save_results and output_dir is not None:
                coef_plot_path = output_dir / f"threshold_sweep_coefficients_{safe_name}.png"
                active_plot_path = output_dir / f"threshold_sweep_active_count_{safe_name}.png"

            plot_threshold_sweep_coefficients(
                df_sweep=df_sweep_i,
                feature_names=system_i.feature_names,
                library_name=library_name,
                save_path=coef_plot_path,
                show=show_plots,
            )
            if coef_plot_path is not None:
                print("Saved:", coef_plot_path)

            plot_threshold_sweep_active_count(
                df_sweep=df_sweep_i,
                library_name=library_name,
                save_path=active_plot_path,
                show=show_plots,
            )
            if active_plot_path is not None:
                print("Saved:", active_plot_path)

    df_all_sweeps = pd.concat(combined_sweep_rows, ignore_index=True)
    df_pareto_summary = pareto_summary_from_sweeps(all_sweep_results)

    if save_results and output_dir is not None:
        combined_csv_path = output_dir / "all_threshold_sweeps.csv"
        df_all_sweeps.to_csv(combined_csv_path, index=False)
        print("Saved combined threshold sweep table:", combined_csv_path)

        pareto_summary_path = output_dir / "pareto_summary_active_terms.csv"
        df_pareto_summary.to_csv(pareto_summary_path, index=False)
        print("Saved Pareto summary:", pareto_summary_path)

        if make_zip:
            if zip_path is None:
                zip_path = Path("/content/video_to_pde_threshold_sweeps.zip")
            zip_path = Path(zip_path)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.rglob("*"):
                    zipf.write(file_path, arcname=file_path.relative_to(output_dir))
            print("Created zip file:", zip_path)
            paths["zip_path"] = zip_path

            if download_zip:
                try:
                    from google.colab import files  # type: ignore
                    files.download(str(zip_path))
                except ImportError:
                    print("Not running in Colab. Zip file saved at:", zip_path)

    return all_sweep_results, df_all_sweeps, df_pareto_summary, paths


# ============================================================================
# Section 12. Stability studies for all PDE libraries
# ============================================================================
def plot_stability_selection_frequency(
    summary: pd.DataFrame,
    library_name: str,
    *,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
) -> None:
    """Horizontal bar plot of selection frequency for one library."""
    summary_plot = summary.copy().sort_values("selection_freq", ascending=True)
    terms = summary_plot["term"].values
    freqs = summary_plot["selection_freq"].values

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(terms))))
    ax.barh(terms, freqs)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("selection frequency over random weak-center samples")
    ax.set_ylabel("term")
    ax.set_title(f"Term-selection stability\n{library_name}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_stability_coefficients(
    summary: pd.DataFrame,
    library_name: str,
    *,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
) -> None:
    """Plot coefficient mean among selected runs with standard deviation."""
    summary_plot = summary.copy().sort_values("selection_freq", ascending=True)
    terms = summary_plot["term"].values
    means = summary_plot["coef_mean_selected"].values
    stds = summary_plot["coef_std_selected"].values
    freqs = summary_plot["selection_freq"].values

    finite_means = np.asarray(means, dtype=float)
    finite_means = finite_means[np.isfinite(finite_means)]
    if len(finite_means) > 0:
        text_x = 1.05 * np.max(np.abs(finite_means))
        if text_x == 0:
            text_x = 0.05
    else:
        text_x = 0.05

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(terms))))
    ax.errorbar(means, terms, xerr=stds, fmt="o", capsize=3)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("coefficient mean among selected runs ± std")
    ax.set_ylabel("term")
    ax.set_title(f"Coefficient stability\n{library_name}")
    ax.grid(True, axis="x", alpha=0.3)

    for y, f in zip(terms, freqs):
        ax.text(
            x=text_x,
            y=y,
            s=f"freq={f:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def stability_comparison_dataframe(
    all_stability_results: Dict[str, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Create a compact long-form comparison table across stability studies."""
    rows = []
    for library_name, result in all_stability_results.items():
        summary = result["summary"]
        for _, row in summary.iterrows():
            rows.append({
                "library": library_name,
                "term": row["term"],
                "selection_freq": row["selection_freq"],
                "coef_mean_selected": row["coef_mean_selected"],
                "coef_std_selected": row["coef_std_selected"],
                "n_selected": row["n_selected"],
                "n_runs": row["n_runs"],
            })
    return pd.DataFrame(rows)


def run_stability_studies_for_libraries(
    U: np.ndarray,
    grid: Grid,
    vx: Optional[np.ndarray],
    vy: Optional[np.ndarray],
    *,
    libraries: Optional[Dict[str, Sequence[Term]]] = None,
    sigmas: Optional[Tuple[float, float, float]] = None,
    frac_y: float = 0.06,
    frac_x: float = 0.06,
    frac_t: float = 0.025,
    M: int = 1000,
    runs: int = 100,
    threshold: float = 1e-3,
    alpha: float = 1e-6,
    include_advection: bool = True,
    device: Optional[str] = None,
    output_dir: Optional[os.PathLike] = None,
    save_results: bool = True,
    make_plots: bool = True,
    show_plots: bool = True,
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
    metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    """
    Run the random-center stability study for all candidate PDE libraries.

    Parameters
    ----------
    U, grid, vx, vy
        Data and drift velocity inputs.
    libraries
        Candidate libraries. Defaults to ``make_candidate_libraries()``.
    sigmas
        Weak-test Gaussian widths. If omitted, they are computed from
        ``gaussian_widths_from_grid`` using ``frac_y``, ``frac_x``, ``frac_t``.
    device
        Passed to ``stability_study`` / ``build_weak_system``. Use ``"cuda"``
        in Colab when available. If None, the lower-level code auto-selects CUDA
        when available.
    save_results
        If True, save per-library and combined CSV files, plots, and metadata.
    make_plots
        If True, create stability plots.
    show_plots
        If True, display plots in the notebook. If False, plots are saved and
        closed silently.
    download_zip
        If True and running in Colab, download the zip file. Set to False to
        avoid automatic downloading.

    Returns
    -------
    all_stability_results, df_runs_all, df_summary_all, df_comparison, paths
    """
    if libraries is None:
        libraries = make_candidate_libraries()
    if sigmas is None:
        sigmas = gaussian_widths_from_grid(
            grid, frac_y=frac_y, frac_x=frac_x, frac_t=frac_t
        )

    print(
        f"Gaussian widths (grid cells): "
        f"sy={sigmas[0]:.2f}, sx={sigmas[1]:.2f}, st={sigmas[2]:.2f}"
    )
    if device is not None:
        print(f"Using device: {device}")

    paths: Dict[str, Path] = {}
    if save_results:
        if output_dir is None:
            output_dir = Path("/content/video_to_pde_stability_results")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths["output_dir"] = output_dir
    else:
        output_dir = None

    all_stability_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    combined_summary_list = []
    combined_runs_list = []

    for library_name, terms in libraries.items():
        print("=" * 100)
        print(f"Stability study for library: {library_name}")
        print("=" * 100)

        df_runs, summary = stability_study(
            U,
            grid,
            sigmas=sigmas,
            M=M,
            runs=runs,
            vx=vx,
            vy=vy,
            include_advection=include_advection,
            terms=terms,
            threshold=threshold,
            alpha=alpha,
            device=device,
        )

        df_runs = df_runs.copy()
        summary = summary.copy()
        df_runs.insert(0, "library", library_name)
        summary.insert(0, "library", library_name)

        all_stability_results[library_name] = {
            "df_runs": df_runs,
            "summary": summary,
        }
        combined_runs_list.append(df_runs)
        combined_summary_list.append(summary)

        print()
        print(summary.to_string(index=False))

        safe_name = safe_filename(library_name)
        if save_results and output_dir is not None:
            runs_path = output_dir / f"stability_runs_{safe_name}.csv"
            summary_path = output_dir / f"stability_summary_{safe_name}.csv"
            df_runs.to_csv(runs_path, index=False)
            summary.to_csv(summary_path, index=False)
            print("Saved:", runs_path)
            print("Saved:", summary_path)

        if make_plots:
            freq_plot_path = None
            coef_plot_path = None
            if save_results and output_dir is not None:
                freq_plot_path = output_dir / f"stability_selection_frequency_{safe_name}.png"
                coef_plot_path = output_dir / f"stability_coefficients_{safe_name}.png"

            plot_stability_selection_frequency(
                summary,
                library_name=library_name,
                save_path=freq_plot_path,
                show=show_plots,
            )
            if freq_plot_path is not None:
                print("Saved:", freq_plot_path)

            plot_stability_coefficients(
                summary,
                library_name=library_name,
                save_path=coef_plot_path,
                show=show_plots,
            )
            if coef_plot_path is not None:
                print("Saved:", coef_plot_path)

    df_runs_all = pd.concat(combined_runs_list, ignore_index=True)
    df_summary_all = pd.concat(combined_summary_list, ignore_index=True)
    df_comparison = stability_comparison_dataframe(all_stability_results)

    if save_results and output_dir is not None:
        runs_all_path = output_dir / "stability_runs_all_libraries.csv"
        summary_all_path = output_dir / "stability_summary_all_libraries.csv"
        comparison_path = output_dir / "stability_comparison_terms.csv"

        df_runs_all.to_csv(runs_all_path, index=False)
        df_summary_all.to_csv(summary_all_path, index=False)
        df_comparison.to_csv(comparison_path, index=False)

        print("Saved combined runs:", runs_all_path)
        print("Saved combined summary:", summary_all_path)
        print("Saved comparison table:", comparison_path)

        metadata_path = output_dir / "stability_run_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write("Video-to-PDE stability study metadata\n")
            f.write("=" * 80 + "\n")
            default_metadata = {
                "FRAC_Y": frac_y,
                "FRAC_X": frac_x,
                "FRAC_T": frac_t,
                "sy": sigmas[0],
                "sx": sigmas[1],
                "st": sigmas[2],
                "STAB_M": M,
                "STAB_RUNS": runs,
                "STAB_THRESHOLD": threshold,
                "STAB_ALPHA": alpha,
                "INCLUDE_ADVECTION": include_advection,
                "DEVICE": device,
                "U.shape": U.shape,
                "grid.dx": grid.dx,
                "grid.dy": grid.dy,
                "grid.dt": grid.dt,
            }
            if metadata:
                default_metadata.update(metadata)
            for key, value in default_metadata.items():
                f.write(f"{key} = {value}\n")
        print("Saved metadata:", metadata_path)

        if make_zip:
            if zip_path is None:
                zip_path = Path("/content/video_to_pde_stability_results.zip")
            zip_path = Path(zip_path)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.rglob("*"):
                    zipf.write(file_path, arcname=file_path.relative_to(output_dir))
            print("Created zip file:", zip_path)
            paths["zip_path"] = zip_path

            if download_zip:
                try:
                    from google.colab import files  # type: ignore
                    files.download(str(zip_path))
                except ImportError:
                    print("Not running in Colab. Zip file saved at:", zip_path)

    return all_stability_results, df_runs_all, df_summary_all, df_comparison, paths


# ============================================================================
# Section 13. Rollout / one-step validation for discovered PDE libraries
# ============================================================================
def validation_split(
    U: np.ndarray,
    grid: Grid,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    train_end_frac: float = 0.60,
    test_end_frac: float = 0.80,
) -> Dict[str, object]:
    """
    Create the chronological train/test/validation split used in the notebooks.

    By default, this returns:
      train: [0, 60%),
      test : [60%, 80%),
      val  : [80%, 100%).

    Returns a dictionary containing the slices and the validation arrays.
    """
    nt = int(grid.nt)
    i1 = int(round(train_end_frac * nt))
    i2 = int(round(test_end_frac * nt))
    i1 = int(np.clip(i1, 0, nt))
    i2 = int(np.clip(i2, i1, nt))

    slice_train = slice(0, i1)
    slice_test = slice(i1, i2)
    slice_val = slice(i2, nt)

    return {
        "slice_train": slice_train,
        "slice_test": slice_test,
        "slice_val": slice_val,
        "i_train_end": i1,
        "i_test_end": i2,
        "U_val": U[:, :, slice_val],
        "t_val": grid.t[slice_val],
        "vx_val": vx[slice_val],
        "vy_val": vy[slice_val],
    }

# ============================================================
# Post-bootstrap diagnostics:
# front radius, center of mass, snapshots, and comparison plots
# ============================================================
# ============================================================
# Patch post-bootstrap plotting functions
# Fix: front levels may be strings, so cast them to float for labels
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import wsindy_video

# ============================================================
# Robust post-bootstrap diagnostics
# Safe replacement version
# Handles front keys like:
#   0.05, "0.05", "level_0.050"
# Also computes COM error-over-time locally.
# ============================================================

import re as _re_postboot
import numpy as _np_postboot
import pandas as _pd_postboot
import matplotlib.pyplot as _plt_postboot
from pathlib import Path as _Path_postboot


def _front_level_to_float(level_key):
    """
    Convert front-radius level keys to float.

    Handles:
        0.05
        "0.05"
        "level_0.050"
    """
    if isinstance(level_key, (int, float, _np_postboot.integer, _np_postboot.floating)):
        return float(level_key)

    s = str(level_key)

    try:
        return float(s)
    except ValueError:
        pass

    match = _re_postboot.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if match is None:
        raise ValueError(f"Could not parse numeric level from key: {level_key!r}")

    return float(match.group(0))


def _postboot_safe_filename(name):
    """
    Safe filename helper for diagnostic outputs.
    """
    return (
        str(name)
        .replace(":", "")
        .replace("|", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("^", "pow")
        .replace("·", "dot")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def _compute_com_error_over_time_postboot(U_true, U_pred, grid):
    """
    Compute center-of-mass distance error at each time frame.
    """
    ny, nx, nt_loc = U_true.shape

    if hasattr(grid, "x"):
        x = _np_postboot.asarray(grid.x)
    else:
        x = _np_postboot.arange(nx) * float(grid.dx)

    if hasattr(grid, "y"):
        y = _np_postboot.asarray(grid.y)
    else:
        y = _np_postboot.arange(ny) * float(grid.dy)

    X, Y = _np_postboot.meshgrid(x, y)

    err = []

    for k in range(nt_loc):
        A = U_true[:, :, k]
        B = U_pred[:, :, k]

        mA = _np_postboot.sum(A) + 1e-12
        mB = _np_postboot.sum(B) + 1e-12

        cxA = _np_postboot.sum(A * X) / mA
        cyA = _np_postboot.sum(A * Y) / mA

        cxB = _np_postboot.sum(B * X) / mB
        cyB = _np_postboot.sum(B * Y) / mB

        err.append(_np_postboot.sqrt((cxB - cxA) ** 2 + (cyB - cyA) ** 2))

    return _np_postboot.asarray(err)


def plot_front_radius_diagnostic(
    t_val,
    front,
    model_name,
    *,
    save_path=None,
    show=True,
):
    """
    Plot true vs predicted equivalent front radius for one model.
    Robust to front['per_level'] keys like 0.05, '0.05', or 'level_0.050'.
    """
    fig = _plt_postboot.figure(figsize=(8, 4.8))

    for lev, d in front["per_level"].items():
        lev_float = _front_level_to_float(lev)

        _plt_postboot.plot(
            t_val,
            d["r_true"],
            "-",
            label=f"true level {lev_float:.3f}",
        )

        _plt_postboot.plot(
            t_val,
            d["r_pred"],
            "--",
            label=f"pred level {lev_float:.3f}",
        )

    _plt_postboot.xlabel("t")
    _plt_postboot.ylabel("equivalent front radius")
    _plt_postboot.title(f"Front-radius diagnostic\n{model_name}")
    _plt_postboot.grid(True, alpha=0.3)
    _plt_postboot.legend(fontsize=7, ncol=2)
    _plt_postboot.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        _plt_postboot.show()
    else:
        _plt_postboot.close(fig)


def plot_com_error_diagnostic(
    t_val,
    U_true,
    U_pred,
    grid,
    model_name,
    *,
    save_path=None,
    show=True,
):
    """
    Plot center-of-mass error over time.
    """
    err_t = _compute_com_error_over_time_postboot(U_true, U_pred, grid)

    fig = _plt_postboot.figure(figsize=(7, 4))

    _plt_postboot.plot(t_val, err_t, "-o", ms=3)

    _plt_postboot.xlabel("t")
    _plt_postboot.ylabel("center-of-mass error")
    _plt_postboot.title(f"Center-of-mass error\n{model_name}")
    _plt_postboot.grid(True, alpha=0.3)
    _plt_postboot.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        _plt_postboot.show()
    else:
        _plt_postboot.close(fig)

    return err_t


def plot_snapshot_diagnostic(
    U_true,
    U_pred,
    t_val,
    model_name,
    *,
    n_show=6,
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Plot true/predicted snapshots for one model.
    """
    nt_loc = U_true.shape[2]
    ids = _np_postboot.linspace(0, nt_loc - 1, min(n_show, nt_loc), dtype=int)

    fig, axes = _plt_postboot.subplots(
        2,
        len(ids),
        figsize=(3.0 * len(ids), 6.0),
        squeeze=False,
    )

    vmin = min(float(U_true.min()), float(U_pred.min()))
    vmax = max(float(U_true.max()), float(U_pred.max()))

    for j, k in enumerate(ids):
        ax = axes[0, j]
        ax.imshow(
            U_true[:, :, k],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"true\nt={t_val[k]:.2f}")
        ax.axis("off")

        ax = axes[1, j]
        ax.imshow(
            U_pred[:, :, k],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"pred\nt={t_val[k]:.2f}")
        ax.axis("off")

    fig.suptitle(f"Snapshots\n{model_name}", fontsize=14)
    _plt_postboot.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        _plt_postboot.show()
    else:
        _plt_postboot.close(fig)


def plot_selected_front_radius_comparison(
    t_val,
    fronts_by_model,
    *,
    selected_level=0.10,
    save_path=None,
    show=True,
):
    """
    Compare predicted front-radius curves across models at one selected threshold.
    Robust to front['per_level'] keys like 0.05, '0.05', or 'level_0.050'.
    """
    fig = _plt_postboot.figure(figsize=(9, 5))

    true_plotted = False

    for model_name, front in fronts_by_model.items():
        keys = list(front["per_level"].keys())

        key_float_pairs = [
            (k, _front_level_to_float(k))
            for k in keys
        ]

        lev_key, lev_float = min(
            key_float_pairs,
            key=lambda pair: abs(pair[1] - float(selected_level)),
        )

        d = front["per_level"][lev_key]

        if not true_plotted:
            _plt_postboot.plot(
                t_val,
                d["r_true"],
                "k-",
                linewidth=2.5,
                label=f"true level {lev_float:.3f}",
            )
            true_plotted = True

        _plt_postboot.plot(
            t_val,
            d["r_pred"],
            "--",
            linewidth=2,
            label=model_name,
        )

    _plt_postboot.xlabel("t")
    _plt_postboot.ylabel("equivalent front radius")
    _plt_postboot.title(
        f"Post-bootstrap front-radius comparison at level {float(selected_level):.3f}"
    )
    _plt_postboot.grid(True, alpha=0.3)
    _plt_postboot.legend(fontsize=7)
    _plt_postboot.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        _plt_postboot.show()
    else:
        _plt_postboot.close(fig)


def run_post_bootstrap_diagnostics(
    all_boot_results,
    U,
    grid,
    vx,
    vy,
    *,
    output_dir=None,
    front_levels=(0.05, 0.10, 0.15, 0.20, 0.25),
    selected_front_level=0.10,
    train_end_frac=0.60,
    test_end_frac=0.80,
    model_order=None,
    save_results=True,
    make_plots=True,
    show_plots=True,
    n_snapshots=6,
    cmap="viridis",
):
    """
    Robust post-bootstrap diagnostics for bootstrap-refitted models.

    Computes:
      - validation rollout RMSE,
      - front-radius MAE/RMSE,
      - center-of-mass MAE/RMSE,
      - front-radius plots,
      - COM plots,
      - snapshots,
      - comparison plots.
    """

    split = validation_split(
        U,
        grid,
        vx,
        vy,
        train_end_frac=train_end_frac,
        test_end_frac=test_end_frac,
    )

    U_val = split["U_val"]
    t_val = split["t_val"]
    vx_val = split["vx_val"]
    vy_val = split["vy_val"]

    if output_dir is not None:
        output_path = _Path_postboot(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    paths = {}
    if output_path is not None:
        paths["output_dir"] = output_path

    if model_order is None:
        model_names = list(all_boot_results.keys())
    else:
        model_names = [m for m in model_order if m in all_boot_results]

    missing_models = []
    if model_order is not None:
        missing_models = [m for m in model_order if m not in all_boot_results]

    if missing_models:
        print("Warning: these requested models are missing:")
        for m in missing_models:
            print("  -", m)

    summary_rows = []
    level_rows = []
    fronts_by_model = {}

    for model_name in model_names:
        res = all_boot_results[model_name]

        print("\n" + "=" * 100)
        print(f"Post-bootstrap diagnostics for: {model_name}")
        print("=" * 100)

        if "U_pred_val" in res:
            U_pred = res["U_pred_val"]
        else:
            raise KeyError(
                f"{model_name} does not contain 'U_pred_val'. "
                "Run bootstrap validation first or store prediction arrays."
            )

        if "mse_val_t" in res:
            mse_t = res["mse_val_t"]
        else:
            mse_t = mse_over_time(U_val, U_pred)

        if "rmse_val_percent" in res:
            rmse_percent = float(res["rmse_val_percent"])
        else:
            rmse_percent = 100.0 * relative_rmse(U_val, U_pred)

        front = front_radius_error(
            U_val,
            U_pred,
            levels=list(front_levels),
            grid=grid,
        )

        com_error_over_time = _compute_com_error_over_time_postboot(
            U_val,
            U_pred,
            grid,
        )

        com_mae = float(_np_postboot.mean(_np_postboot.abs(com_error_over_time)))
        com_rmse = float(_np_postboot.sqrt(_np_postboot.mean(com_error_over_time ** 2)))

        fronts_by_model[model_name] = front

        front_mae = float(front["mae_mean"])
        front_rmse = float(front["rmse_mean"])

        print(f"Validation RMSE (%) : {rmse_percent:.6f}")
        print(f"Front-radius MAE   : {front_mae:.6f}")
        print(f"Front-radius RMSE  : {front_rmse:.6f}")
        print(f"COM MAE            : {com_mae:.6f}")
        print(f"COM RMSE           : {com_rmse:.6f}")

        if "median_coefs" in res:
            print("Median coefficients:")
            for k, v in res["median_coefs"].items():
                print(f"  {k:12s} = {v:+.6e}")

        safe_name = _postboot_safe_filename(model_name)

        if output_path is not None:
            front_plot_path = output_path / f"front_radius_{safe_name}.png"
            com_plot_path = output_path / f"com_error_{safe_name}.png"
            snap_plot_path = output_path / f"snapshots_{safe_name}.png"
            mse_plot_path = output_path / f"mse_over_time_{safe_name}.png"
        else:
            front_plot_path = None
            com_plot_path = None
            snap_plot_path = None
            mse_plot_path = None

        if make_plots:
            plot_front_radius_diagnostic(
                t_val,
                front,
                model_name,
                save_path=front_plot_path if save_results else None,
                show=show_plots,
            )

            plot_com_error_diagnostic(
                t_val,
                U_val,
                U_pred,
                grid,
                model_name,
                save_path=com_plot_path if save_results else None,
                show=show_plots,
            )

            plot_snapshot_diagnostic(
                U_val,
                U_pred,
                t_val,
                model_name,
                n_show=n_snapshots,
                cmap=cmap,
                save_path=snap_plot_path if save_results else None,
                show=show_plots,
            )

            fig = _plt_postboot.figure(figsize=(7, 4))
            _plt_postboot.semilogy(t_val, mse_t)
            _plt_postboot.xlabel("t")
            _plt_postboot.ylabel("MSE")
            _plt_postboot.title(f"Post-bootstrap validation MSE\n{model_name}")
            _plt_postboot.grid(True, alpha=0.3)
            _plt_postboot.tight_layout()

            if save_results and mse_plot_path is not None:
                fig.savefig(mse_plot_path, dpi=1200, bbox_inches="tight")

            if show_plots:
                _plt_postboot.show()
            else:
                _plt_postboot.close(fig)

        summary_rows.append({
            "model": model_name,
            "model_family": res["spec"]["model_family"],
            "init_source": res["spec"]["init_source"],
            "init_source_label": res["spec"]["init_source_label"],
            "validation_rmse_percent": rmse_percent,
            "front_mae": front_mae,
            "front_rmse": front_rmse,
            "com_mae": com_mae,
            "com_rmse": com_rmse,
            "median_coefs": ", ".join(
                f"{k}: {v:+.6e}"
                for k, v in res.get("median_coefs", {}).items()
            ),
            "front_plot": str(front_plot_path) if front_plot_path is not None else "",
            "com_plot": str(com_plot_path) if com_plot_path is not None else "",
            "snapshot_plot": str(snap_plot_path) if snap_plot_path is not None else "",
            "mse_plot": str(mse_plot_path) if mse_plot_path is not None else "",
        })

        for lev, d in front["per_level"].items():
            lev_float = _front_level_to_float(lev)

            r_true = _np_postboot.asarray(d["r_true"])
            r_pred = _np_postboot.asarray(d["r_pred"])

            true_growth = float(r_true[-1] - r_true[0])
            pred_growth = float(r_pred[-1] - r_pred[0])

            level_rows.append({
                "model": model_name,
                "model_family": res["spec"]["model_family"],
                "init_source": res["spec"]["init_source"],
                "level": lev_float,
                "front_mae_level": float(d["mae"]),
                "front_rmse_level": float(d["rmse"]),
                "r_true_start": float(r_true[0]),
                "r_true_end": float(r_true[-1]),
                "r_pred_start": float(r_pred[0]),
                "r_pred_end": float(r_pred[-1]),
                "true_growth": true_growth,
                "pred_growth": pred_growth,
                "growth_error": pred_growth - true_growth,
            })

    df_summary = _pd_postboot.DataFrame(summary_rows)

    if len(df_summary) > 0:
        df_summary = df_summary.sort_values(
            ["front_rmse", "validation_rmse_percent"]
        ).reset_index(drop=True)

    df_front_levels = _pd_postboot.DataFrame(level_rows)

    if len(df_front_levels) > 0:
        df_front_levels = df_front_levels.sort_values(
            ["level", "front_rmse_level", "model"]
        ).reset_index(drop=True)

    if save_results and output_path is not None:
        summary_path = output_path / "post_bootstrap_diagnostics_summary.csv"
        levels_path = output_path / "post_bootstrap_front_radius_by_level.csv"

        df_summary.to_csv(summary_path, index=False)
        df_front_levels.to_csv(levels_path, index=False)

        paths["summary_csv"] = summary_path
        paths["front_levels_csv"] = levels_path

        print("\nSaved post-bootstrap diagnostic summary:")
        print(summary_path)
        print("Saved front-radius-by-level table:")
        print(levels_path)

    if make_plots and len(df_summary) > 0:
        if output_path is not None:
            front_compare_path = output_path / f"front_radius_comparison_level_{selected_front_level:.3f}.png"
            rmse_compare_path = output_path / "post_bootstrap_validation_rmse_comparison.png"
            front_rmse_compare_path = output_path / "post_bootstrap_front_rmse_comparison.png"
            com_rmse_compare_path = output_path / "post_bootstrap_com_rmse_comparison.png"
        else:
            front_compare_path = None
            rmse_compare_path = None
            front_rmse_compare_path = None
            com_rmse_compare_path = None

        plot_selected_front_radius_comparison(
            t_val,
            fronts_by_model,
            selected_level=selected_front_level,
            save_path=front_compare_path if save_results else None,
            show=show_plots,
        )

        fig = _plt_postboot.figure(figsize=(10, 5))
        plot_df = df_summary.sort_values("validation_rmse_percent")
        _plt_postboot.barh(plot_df["model"], plot_df["validation_rmse_percent"])
        _plt_postboot.xlabel("validation rollout relative RMSE (%)")
        _plt_postboot.ylabel("model")
        _plt_postboot.title("Post-bootstrap validation RMSE comparison")
        _plt_postboot.grid(True, axis="x", alpha=0.3)
        _plt_postboot.tight_layout()

        if save_results and rmse_compare_path is not None:
            fig.savefig(rmse_compare_path, dpi=1200, bbox_inches="tight")

        if show_plots:
            _plt_postboot.show()
        else:
            _plt_postboot.close(fig)

        fig = _plt_postboot.figure(figsize=(10, 5))
        plot_df = df_summary.sort_values("front_rmse")
        _plt_postboot.barh(plot_df["model"], plot_df["front_rmse"])
        _plt_postboot.xlabel("front-radius RMSE")
        _plt_postboot.ylabel("model")
        _plt_postboot.title("Post-bootstrap front-radius RMSE comparison")
        _plt_postboot.grid(True, axis="x", alpha=0.3)
        _plt_postboot.tight_layout()

        if save_results and front_rmse_compare_path is not None:
            fig.savefig(front_rmse_compare_path, dpi=1200, bbox_inches="tight")

        if show_plots:
            _plt_postboot.show()
        else:
            _plt_postboot.close(fig)

        fig = _plt_postboot.figure(figsize=(10, 5))
        plot_df = df_summary.sort_values("com_rmse")
        _plt_postboot.barh(plot_df["model"], plot_df["com_rmse"])
        _plt_postboot.xlabel("center-of-mass RMSE")
        _plt_postboot.ylabel("model")
        _plt_postboot.title("Post-bootstrap center-of-mass RMSE comparison")
        _plt_postboot.grid(True, axis="x", alpha=0.3)
        _plt_postboot.tight_layout()

        if save_results and com_rmse_compare_path is not None:
            fig.savefig(com_rmse_compare_path, dpi=1200, bbox_inches="tight")

        if show_plots:
            _plt_postboot.show()
        else:
            _plt_postboot.close(fig)

    if len(df_summary) > 0:
        best_front = df_summary.sort_values(
            ["front_rmse", "validation_rmse_percent"]
        ).iloc[0]

        best_rmse = df_summary.sort_values(
            ["validation_rmse_percent", "front_rmse"]
        ).iloc[0]

        print("\n" + "=" * 100)
        print("POST-BOOTSTRAP MODEL SELECTION HINTS")
        print("=" * 100)

        print("Best by front-radius RMSE:")
        print(f"  model      : {best_front['model']}")
        print(f"  front RMSE : {best_front['front_rmse']:.6f}")
        print(f"  val RMSE % : {best_front['validation_rmse_percent']:.6f}")

        print("\nBest by validation RMSE:")
        print(f"  model      : {best_rmse['model']}")
        print(f"  val RMSE % : {best_rmse['validation_rmse_percent']:.6f}")
        print(f"  front RMSE : {best_rmse['front_rmse']:.6f}")

        if best_front["model"] == best_rmse["model"]:
            print("\nRecommendation: this model is best by both front-radius and global RMSE.")
        else:
            print(
                "\nRecommendation: compare these two models visually. "
                "If front propagation is central, prefer the front-radius winner. "
                "If pixel-wise rollout accuracy is central, prefer the validation-RMSE winner."
            )

    return df_summary, df_front_levels, paths

# def plot_front_radius_diagnostic_patched(
#     t_val,
#     front,
#     model_name,
#     *,
#     save_path=None,
#     show=True,
# ):
#     """
#     Plot true vs predicted equivalent front radius for one model.
#     Robust to front["per_level"] keys being strings or floats.
#     """
#     fig = plt.figure(figsize=(8, 4.8))

#     for lev, d in front["per_level"].items():
#         lev_float = float(lev)

#         plt.plot(
#             t_val,
#             d["r_true"],
#             "-",
#             label=f"true level {lev_float:.3f}",
#         )
#         plt.plot(
#             t_val,
#             d["r_pred"],
#             "--",
#             label=f"pred level {lev_float:.3f}",
#         )

#     plt.xlabel("t")
#     plt.ylabel("equivalent front radius")
#     plt.title(f"Front-radius diagnostic\n{model_name}")
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=7, ncol=2)
#     plt.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=1200, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)


def plot_selected_front_radius_comparison_patched(
    t_val,
    fronts_by_model,
    *,
    selected_level=0.10,
    save_path=None,
    show=True,
):
    """
    Compare predicted front-radius curves across models at one selected threshold.
    Robust to front["per_level"] keys being strings or floats.
    """
    fig = plt.figure(figsize=(9, 5))

    true_plotted = False

    for model_name, front in fronts_by_model.items():
        keys = list(front["per_level"].keys())

        # Find nearest available level, allowing string keys
        key_float_pairs = [(k, float(k)) for k in keys]
        lev_key, lev_float = min(
            key_float_pairs,
            key=lambda pair: abs(pair[1] - float(selected_level)),
        )

        d = front["per_level"][lev_key]

        if not true_plotted:
            plt.plot(
                t_val,
                d["r_true"],
                "k-",
                linewidth=2.5,
                label=f"true level {lev_float:.3f}",
            )
            true_plotted = True

        plt.plot(
            t_val,
            d["r_pred"],
            "--",
            linewidth=2,
            label=f"{model_name}",
        )

    plt.xlabel("t")
    plt.ylabel("equivalent front radius")
    plt.title(f"Post-bootstrap front-radius comparison at level {float(selected_level):.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
# def plot_front_radius_diagnostic(
#     t_val,
#     front,
#     model_name,
#     *,
#     save_path=None,
#     show=True,
# ):
#     """
#     Plot true vs predicted equivalent front radius for one model.
#     """
#     fig = plt.figure(figsize=(8, 4.8))

#     for lev, d in front["per_level"].items():
#         plt.plot(
#             t_val,
#             d["r_true"],
#             "-",
#             label=f"true level {lev:.3f}",
#         )
#         plt.plot(
#             t_val,
#             d["r_pred"],
#             "--",
#             label=f"pred level {lev:.3f}",
#         )

#     plt.xlabel("t")
#     plt.ylabel("equivalent front radius")
#     plt.title(f"Front-radius diagnostic\n{model_name}")
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=7, ncol=2)
#     plt.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=1200, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)


def plot_com_error_diagnostic(
    t_val,
    com,
    model_name,
    *,
    save_path=None,
    show=True,
):
    """
    Plot center-of-mass error over time for one model.
    """
    fig = plt.figure(figsize=(7, 4))

    plt.plot(t_val, com["error_over_time"], "-o", ms=3)

    plt.xlabel("t")
    plt.ylabel("center-of-mass error")
    plt.title(f"Center-of-mass error\n{model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_snapshot_diagnostic(
    U_true,
    U_pred,
    t_val,
    model_name,
    *,
    n_show=6,
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Plot true/predicted snapshots for one model.
    This is self-contained and does not rely on plot_snapshots supporting save_path.
    """
    nt_loc = U_true.shape[2]
    ids = np.linspace(0, nt_loc - 1, min(n_show, nt_loc), dtype=int)

    fig, axes = plt.subplots(
        2,
        len(ids),
        figsize=(3.0 * len(ids), 6.0),
        squeeze=False,
    )

    vmin = min(float(U_true.min()), float(U_pred.min()))
    vmax = max(float(U_true.max()), float(U_pred.max()))

    for j, k in enumerate(ids):
        ax = axes[0, j]
        ax.imshow(
            U_true[:, :, k],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"true\nt={t_val[k]:.2f}")
        ax.axis("off")

        ax = axes[1, j]
        ax.imshow(
            U_pred[:, :, k],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"pred\nt={t_val[k]:.2f}")
        ax.axis("off")

    fig.suptitle(f"Snapshots\n{model_name}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# def plot_selected_front_radius_comparison(
#     t_val,
#     fronts_by_model,
#     *,
#     selected_level=0.10,
#     save_path=None,
#     show=True,
# ):
#     """
#     Compare predicted front-radius curves across bootstrap-refitted models
#     at one selected threshold level.

#     The true curve is plotted once. Each model contributes one predicted curve.
#     """
#     fig = plt.figure(figsize=(9, 5))

#     true_plotted = False

#     for model_name, front in fronts_by_model.items():
#         levels_available = list(front["per_level"].keys())

#         # Use selected_level if available exactly; otherwise use nearest.
#         if selected_level in front["per_level"]:
#             lev = selected_level
#         else:
#             lev = min(levels_available, key=lambda x: abs(float(x) - float(selected_level)))

#         d = front["per_level"][lev]

#         if not true_plotted:
#             plt.plot(
#                 t_val,
#                 d["r_true"],
#                 "k-",
#                 linewidth=2.5,
#                 label=f"true level {lev:.3f}",
#             )
#             true_plotted = True

#         plt.plot(
#             t_val,
#             d["r_pred"],
#             "--",
#             linewidth=2,
#             label=f"{model_name}",
#         )

#     plt.xlabel("t")
#     plt.ylabel("equivalent front radius")
#     plt.title(f"Post-bootstrap front-radius comparison at level {selected_level:.3f}")
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=7)
#     plt.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=1200, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)


def run_post_bootstrap_diagnostics(
    all_boot_results,
    U,
    grid,
    vx,
    vy,
    *,
    output_dir=None,
    front_levels=(0.05, 0.10, 0.15, 0.20, 0.25),
    selected_front_level=0.10,
    train_end_frac=0.60,
    test_end_frac=0.80,
    model_order=None,
    save_results=True,
    make_plots=True,
    show_plots=True,
    n_snapshots=6,
    cmap="viridis",
):
    """
    Run post-bootstrap validation diagnostics for bootstrap-refitted models.

    This function compares:
      - rollout relative RMSE,
      - front-radius MAE/RMSE,
      - center-of-mass MAE/RMSE,
      - front-radius plots,
      - COM error plots,
      - snapshot plots,
      - compact comparison plots.

    Parameters
    ----------
    all_boot_results
        Dictionary returned by run_bootstrap_C_Calt_by_init_source.
    U, grid, vx, vy
        Full data and grid/velocity objects.
    output_dir
        Directory for saving diagnostic figures and CSV files.
    front_levels
        Threshold levels used to define superlevel-set fronts.
    selected_front_level
        One threshold level used for compact across-model comparison.
    model_order
        Optional list of model names to control plotting/order/filtering.

    Returns
    -------
    df_summary, df_front_levels, paths
    """
    split = validation_split(
        U,
        grid,
        vx,
        vy,
        train_end_frac=train_end_frac,
        test_end_frac=test_end_frac,
    )

    U_val = split["U_val"]
    t_val = split["t_val"]
    vx_val = split["vx_val"]
    vy_val = split["vy_val"]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    paths = {}

    if output_path is not None:
        paths["output_dir"] = output_path

    if model_order is None:
        model_names = list(all_boot_results.keys())
    else:
        model_names = [m for m in model_order if m in all_boot_results]

    missing_models = []
    if model_order is not None:
        missing_models = [m for m in model_order if m not in all_boot_results]

    if missing_models:
        print("Warning: the following requested models are missing:")
        for m in missing_models:
            print("  -", m)

    summary_rows = []
    level_rows = []
    fronts_by_model = {}

    for model_name in model_names:
        res = all_boot_results[model_name]

        print("\n" + "=" * 100)
        print(f"Post-bootstrap diagnostics for: {model_name}")
        print("=" * 100)

        U_pred = res["U_pred_val"]
        mse_t = res["mse_val_t"]
        rmse_percent = float(res["rmse_val_percent"])

        front = front_radius_error(
            U_val,
            U_pred,
            levels=list(front_levels),
            grid=grid,
        )

        com = com_error(
            U_val,
            U_pred,
            grid,
        )

        fronts_by_model[model_name] = front

        print(f"Validation RMSE (%) : {rmse_percent:.6f}")
        print(f"Front-radius MAE   : {front['mae_mean']:.6f}")
        print(f"Front-radius RMSE  : {front['rmse_mean']:.6f}")
        print(f"COM MAE            : {com['mae']:.6f}")
        print(f"COM RMSE           : {com['rmse']:.6f}")

        print("Median coefficients:")
        for k, v in res["median_coefs"].items():
            print(f"  {k:12s} = {v:+.6e}")

        safe_name = safe_filename(model_name)

        if output_path is not None:
            front_plot_path = output_path / f"front_radius_{safe_name}.png"
            com_plot_path = output_path / f"com_error_{safe_name}.png"
            snap_plot_path = output_path / f"snapshots_{safe_name}.png"
            mse_plot_path = output_path / f"mse_over_time_{safe_name}.png"
        else:
            front_plot_path = None
            com_plot_path = None
            snap_plot_path = None
            mse_plot_path = None

        if make_plots:
            plot_front_radius_diagnostic(
                t_val,
                front,
                model_name,
                save_path=front_plot_path if save_results else None,
                show=show_plots,
            )

            plot_com_error_diagnostic(
                t_val,
                com,
                model_name,
                save_path=com_plot_path if save_results else None,
                show=show_plots,
            )

            plot_snapshot_diagnostic(
                U_val,
                U_pred,
                t_val,
                model_name,
                n_show=n_snapshots,
                cmap=cmap,
                save_path=snap_plot_path if save_results else None,
                show=show_plots,
            )

            fig = plt.figure(figsize=(7, 4))
            plt.semilogy(t_val, mse_t)
            plt.xlabel("t")
            plt.ylabel("MSE")
            plt.title(f"Post-bootstrap validation MSE\n{model_name}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_results and mse_plot_path is not None:
                fig.savefig(mse_plot_path, dpi=1200, bbox_inches="tight")

            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        summary_rows.append({
            "model": model_name,
            "model_family": res["spec"]["model_family"],
            "init_source": res["spec"]["init_source"],
            "init_source_label": res["spec"]["init_source_label"],
            "validation_rmse_percent": rmse_percent,
            "front_mae": float(front["mae_mean"]),
            "front_rmse": float(front["rmse_mean"]),
            "com_mae": float(com["mae"]),
            "com_rmse": float(com["rmse"]),
            "median_coefs": ", ".join(
                f"{k}: {v:+.6e}" for k, v in res["median_coefs"].items()
            ),
            "front_plot": str(front_plot_path) if front_plot_path is not None else "",
            "com_plot": str(com_plot_path) if com_plot_path is not None else "",
            "snapshot_plot": str(snap_plot_path) if snap_plot_path is not None else "",
            "mse_plot": str(mse_plot_path) if mse_plot_path is not None else "",
        })

        for lev, d in front["per_level"].items():
            level_rows.append({
                "model": model_name,
                "model_family": res["spec"]["model_family"],
                "init_source": res["spec"]["init_source"],
                "level": float(lev),
                "front_mae_level": float(d["mae"]),
                "front_rmse_level": float(d["rmse"]),
                "r_true_start": float(d["r_true"][0]),
                "r_true_end": float(d["r_true"][-1]),
                "r_pred_start": float(d["r_pred"][0]),
                "r_pred_end": float(d["r_pred"][-1]),
                "true_growth": float(d["r_true"][-1] - d["r_true"][0]),
                "pred_growth": float(d["r_pred"][-1] - d["r_pred"][0]),
                "growth_error": float(
                    (d["r_pred"][-1] - d["r_pred"][0])
                    - (d["r_true"][-1] - d["r_true"][0])
                ),
            })

    df_summary = pd.DataFrame(summary_rows)

    if len(df_summary) > 0:
        df_summary = df_summary.sort_values(
            ["front_rmse", "validation_rmse_percent"]
        ).reset_index(drop=True)

    df_front_levels = pd.DataFrame(level_rows)

    if len(df_front_levels) > 0:
        df_front_levels = df_front_levels.sort_values(
            ["level", "front_rmse_level", "model"]
        ).reset_index(drop=True)

    # ========================================================
    # Combined comparison plots
    # ========================================================

    if make_plots and len(df_summary) > 0:

        if output_path is not None:
            front_compare_path = output_path / f"front_radius_comparison_level_{selected_front_level:.3f}.png"
            rmse_compare_path = output_path / "post_bootstrap_validation_rmse_comparison.png"
            front_rmse_compare_path = output_path / "post_bootstrap_front_rmse_comparison.png"
            com_rmse_compare_path = output_path / "post_bootstrap_com_rmse_comparison.png"
        else:
            front_compare_path = None
            rmse_compare_path = None
            front_rmse_compare_path = None
            com_rmse_compare_path = None

        plot_selected_front_radius_comparison(
            t_val,
            fronts_by_model,
            selected_level=selected_front_level,
            save_path=front_compare_path if save_results else None,
            show=show_plots,
        )

        # Validation RMSE comparison.
        fig = plt.figure(figsize=(10, 5))
        plot_df = df_summary.sort_values("validation_rmse_percent")
        plt.barh(plot_df["model"], plot_df["validation_rmse_percent"])
        plt.xlabel("validation rollout relative RMSE (%)")
        plt.ylabel("model")
        plt.title("Post-bootstrap validation RMSE comparison")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_results and rmse_compare_path is not None:
            fig.savefig(rmse_compare_path, dpi=1200, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # Front RMSE comparison.
        fig = plt.figure(figsize=(10, 5))
        plot_df = df_summary.sort_values("front_rmse")
        plt.barh(plot_df["model"], plot_df["front_rmse"])
        plt.xlabel("front-radius RMSE")
        plt.ylabel("model")
        plt.title("Post-bootstrap front-radius RMSE comparison")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_results and front_rmse_compare_path is not None:
            fig.savefig(front_rmse_compare_path, dpi=1200, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # COM RMSE comparison.
        fig = plt.figure(figsize=(10, 5))
        plot_df = df_summary.sort_values("com_rmse")
        plt.barh(plot_df["model"], plot_df["com_rmse"])
        plt.xlabel("center-of-mass RMSE")
        plt.ylabel("model")
        plt.title("Post-bootstrap center-of-mass RMSE comparison")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_results and com_rmse_compare_path is not None:
            fig.savefig(com_rmse_compare_path, dpi=1200, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # ========================================================
    # Save tables
    # ========================================================

    if save_results and output_path is not None:
        summary_path = output_path / "post_bootstrap_diagnostics_summary.csv"
        levels_path = output_path / "post_bootstrap_front_radius_by_level.csv"

        df_summary.to_csv(summary_path, index=False)
        df_front_levels.to_csv(levels_path, index=False)

        paths["summary_csv"] = summary_path
        paths["front_levels_csv"] = levels_path

        print("\nSaved post-bootstrap diagnostic summary:")
        print(summary_path)
        print("Saved front-radius-by-level table:")
        print(levels_path)

    # ========================================================
    # Print final model suggestions
    # ========================================================

    if len(df_summary) > 0:
        best_front = df_summary.sort_values(
            ["front_rmse", "validation_rmse_percent"]
        ).iloc[0]

        best_rmse = df_summary.sort_values(
            ["validation_rmse_percent", "front_rmse"]
        ).iloc[0]

        print("\n" + "=" * 100)
        print("POST-BOOTSTRAP MODEL SELECTION HINTS")
        print("=" * 100)
        print("Best by front-radius RMSE:")
        print(f"  model      : {best_front['model']}")
        print(f"  front RMSE : {best_front['front_rmse']:.6f}")
        print(f"  val RMSE % : {best_front['validation_rmse_percent']:.6f}")

        print("\nBest by validation RMSE:")
        print(f"  model      : {best_rmse['model']}")
        print(f"  val RMSE % : {best_rmse['validation_rmse_percent']:.6f}")
        print(f"  front RMSE : {best_rmse['front_rmse']:.6f}")

        if best_front["model"] == best_rmse["model"]:
            print("\nRecommendation: this model is best by both front-radius and global RMSE.")
        else:
            print(
                "\nRecommendation: compare these two models visually. "
                "If front propagation is central to the paper, prefer the front-radius winner. "
                "If pixel-wise rollout accuracy is central, prefer the validation-RMSE winner."
            )

    return df_summary, df_front_levels, paths

def auto_clip_from_data(U: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Return ``(0, 1)`` when the supplied data already lie in [0, 1],
    otherwise return None.
    """
    return (0.0, 1.0) if (float(np.nanmin(U)) >= 0.0 and float(np.nanmax(U)) <= 1.0) else None


def get_active_fit_coefs(system: WeakSystem,
                         fit: SindyFit,
                         tol: float = 0.0) -> Dict[str, float]:
    """
    Return a dictionary of active fitted coefficients keyed by feature name.
    """
    coefs: Dict[str, float] = {}
    for name, c in zip(system.feature_names, fit.coef):
        if abs(float(c)) > tol:
            coefs[name] = float(c)
    return coefs


def map_fit_to_rhs_coefs(
    coefs: Dict[str, float],
    *,
    use_measured_advection: bool = True,
    use_identified_advection_coefs: bool = False,
) -> Dict[str, float]:
    """
    Convert weak-SINDy feature names to simulator coefficient keys.

    This is the Section-5 convention:
      * learned reaction/diffusion/nonlinear-gradient terms are retained;
      * fitted ``v_x · u_x`` and ``v_y · u_y`` terms are ignored by default;
      * measured drift velocities are used with unit advection coefficient
        through ``rhs_coefs["adv"] = 1.0``.

    Set ``use_identified_advection_coefs=True`` only if you plan to use a
    simulator that supports ``adv_x`` and ``adv_y``.
    """
    rhs_coefs: Dict[str, float] = {}

    for name, value in coefs.items():
        if name == "|grad u|^2":
            rhs_coefs["grad2"] = value
        elif name == "u|grad u|^2":
            rhs_coefs["u_grad2"] = value
        elif name == "Delta u":
            rhs_coefs["lap"] = value
        elif name == "u":
            rhs_coefs["u"] = value
        elif name == "u^2":
            rhs_coefs["u_sq"] = value
        elif name == "1":
            rhs_coefs["const"] = value
        elif name == "v_x · u_x":
            if use_identified_advection_coefs:
                rhs_coefs["adv_x"] = value
        elif name == "v_y · u_y":
            if use_identified_advection_coefs:
                rhs_coefs["adv_y"] = value
        else:
            print(f"Warning: unrecognized feature name ignored: {name}")

    if use_measured_advection:
        rhs_coefs["adv"] = 1.0

    return rhs_coefs


def map_fit_to_rhs_coefs_learned_advection(
    coefs: Dict[str, float],
) -> Dict[str, float]:
    """
    Convert weak-SINDy feature names to simulator coefficient keys using
    the learned advection coefficients.

    This is the Section-6 / Validation-A convention:
      u_t = learned terms + c_x v_x(t) u_x + c_y v_y(t) u_y.

    It does not impose measured advection with coefficient 1.
    """
    rhs_coefs: Dict[str, float] = {}

    for name, value in coefs.items():
        if name == "1":
            rhs_coefs["const"] = value
        elif name == "u":
            rhs_coefs["u"] = value
        elif name == "u^2":
            rhs_coefs["u_sq"] = value
        elif name == "|grad u|^2":
            rhs_coefs["grad2"] = value
        elif name == "u|grad u|^2":
            rhs_coefs["u_grad2"] = value
        elif name == "Delta u":
            rhs_coefs["lap"] = value
        elif name == "v_x · u_x":
            rhs_coefs["adv_x"] = value
        elif name == "v_y · u_y":
            rhs_coefs["adv_y"] = value
        else:
            print(f"Warning: unrecognized feature name ignored: {name}")

    return rhs_coefs


def print_rhs_coefs(library_name: str,
                    rhs_coefs: Dict[str, float],
                    *,
                    title_prefix: str = "Simulator coefficients") -> None:
    """Pretty-print simulator coefficients for one library."""
    print("=" * 90)
    print(f"{title_prefix} for: {library_name}")
    print("=" * 90)

    if len(rhs_coefs) == 0:
        print("  No active terms.")
        return

    for key, value in rhs_coefs.items():
        print(f"  {key:15s} = {value:+.8e}")


def _common_validation_terms(
    U2: torch.Tensor,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    eps_visc: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute non-advection RHS terms for validation simulators.

    Supported keys:
      const / 1, u, u_sq / u^2, grad2, u_grad2, lap, u_lap.
    """
    Ux, Uy = central_grad(U2, dx, dy)
    lap = central_laplacian(U2, dx, dy)
    out = torch.zeros_like(U2)

    c_const = coefs.get("const", coefs.get("1", 0.0))
    if c_const != 0.0:
        out = out + float(c_const)

    if coefs.get("u", 0.0) != 0.0:
        out = out + float(coefs["u"]) * U2

    c_u2 = coefs.get("u_sq", coefs.get("u^2", 0.0))
    if c_u2 != 0.0:
        out = out + float(c_u2) * U2 * U2

    need_grad2 = (coefs.get("grad2", 0.0) != 0.0) or (coefs.get("u_grad2", 0.0) != 0.0)
    if need_grad2:
        grad2 = Ux * Ux + Uy * Uy
        if coefs.get("grad2", 0.0) != 0.0:
            out = out + float(coefs["grad2"]) * grad2
        if coefs.get("u_grad2", 0.0) != 0.0:
            out = out + float(coefs["u_grad2"]) * U2 * grad2

    if coefs.get("lap", 0.0) != 0.0:
        out = out + float(coefs["lap"]) * lap

    if coefs.get("u_lap", 0.0) != 0.0:
        out = out + float(coefs["u_lap"]) * U2 * lap

    if eps_visc is not None and eps_visc != 0.0:
        out = out + float(eps_visc) * lap

    return out, Ux, Uy


def rhs_measured_advection_validation(
    U2: torch.Tensor,
    vx_t: float,
    vy_t: float,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    eps_visc: float = 0.0,
) -> torch.Tensor:
    """
    RHS used for Section-5 validation.

    Measured advection is imposed through the existing convention:
        adv * ( -v_x u_x - v_y u_y )
    using the upwind discretization from ``upwind_advection``.
    """
    out, _, _ = _common_validation_terms(U2, dx, dy, coefs, eps_visc=eps_visc)

    if coefs.get("adv", 0.0) != 0.0:
        out = out + float(coefs.get("adv", 0.0)) * upwind_advection(
            U2, float(vx_t), float(vy_t), dx, dy
        )

    return out


def rhs_learned_advection(
    U2: torch.Tensor,
    vx_t: float,
    vy_t: float,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    eps_visc: float = 0.0,
) -> torch.Tensor:
    """
    RHS used for Section-6 / Validation-A.

    The learned advection model is:
        u_t = learned terms + c_x v_x(t) u_x + c_y v_y(t) u_y.
    """
    out, Ux, Uy = _common_validation_terms(U2, dx, dy, coefs, eps_visc=eps_visc)

    if coefs.get("adv_x", 0.0) != 0.0:
        out = out + float(coefs["adv_x"]) * float(vx_t) * Ux

    if coefs.get("adv_y", 0.0) != 0.0:
        out = out + float(coefs["adv_y"]) * float(vy_t) * Uy

    return out


def _choose_validation_substeps(
    U2: torch.Tensor,
    dt_total: float,
    vx_t: float,
    vy_t: float,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    eps_visc: float,
    *,
    advection_mode: str,
    safety: float = 0.25,
    max_substeps: int = 2000,
) -> int:
    """
    Conservative CFL-style substep selector for explicit validation rollouts.
    """
    dt_abs = abs(float(dt_total))
    if dt_abs == 0.0:
        return 1

    umax = float(torch.max(torch.abs(U2)).item())
    hmin = min(float(dx), float(dy))

    D_eff = (
        abs(float(coefs.get("lap", 0.0)))
        + abs(float(coefs.get("u_lap", 0.0))) * umax
        + abs(float(eps_visc or 0.0))
    )
    if D_eff > 1e-14:
        dt_diff = 0.5 / (D_eff * (1.0 / dx ** 2 + 1.0 / dy ** 2))
    else:
        dt_diff = np.inf

    if advection_mode == "measured":
        adv_rate = abs(float(coefs.get("adv", 0.0))) * (
            abs(float(vx_t)) / float(dx) + abs(float(vy_t)) / float(dy)
        )
    elif advection_mode == "learned":
        adv_rate = (
            abs(float(coefs.get("adv_x", 0.0)) * float(vx_t)) / float(dx)
            + abs(float(coefs.get("adv_y", 0.0)) * float(vy_t)) / float(dy)
        )
    else:
        raise ValueError("advection_mode must be 'measured' or 'learned'")

    dt_adv = safety / adv_rate if adv_rate > 1e-14 else np.inf

    # Mild reaction/nonlinear heuristic.
    reaction_rate = abs(float(coefs.get("u", 0.0))) + 2.0 * abs(float(coefs.get("u_sq", 0.0))) * umax
    dt_react = safety / reaction_rate if reaction_rate > 1e-14 else np.inf

    if coefs.get("grad2", 0.0) or coefs.get("u_grad2", 0.0):
        Ux, Uy = central_grad(U2, dx, dy)
        gmax = float(torch.max(torch.sqrt(Ux ** 2 + Uy ** 2)).item())
        c = abs(float(coefs.get("grad2", 0.0))) + abs(float(coefs.get("u_grad2", 0.0))) * umax
        dt_grad = safety / (c * gmax + 1e-12) if c * gmax > 1e-14 else np.inf
    else:
        dt_grad = np.inf

    dt_allowed = min(dt_diff, dt_adv, dt_react, dt_grad)
    if not np.isfinite(dt_allowed) or dt_allowed <= 0:
        return 1

    n_sub = int(np.ceil(dt_abs / max(dt_allowed, 1e-12)))
    return max(1, min(n_sub, int(max_substeps)))


def advance_one_interval_validation(
    U2: torch.Tensor,
    dt_total: float,
    vx_t: float,
    vy_t: float,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    *,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    advection_mode: str = "measured",
    integrator: str = "heun",
) -> torch.Tensor:
    """
    Advance one frame interval for validation.

    Parameters
    ----------
    advection_mode
        ``"measured"`` uses ``adv * ( -v_x u_x - v_y u_y )``.
        ``"learned"`` uses ``adv_x * v_x u_x + adv_y * v_y u_y``.
    integrator
        ``"heun"`` gives RK2/Heun substepping.
        ``"euler"`` reproduces the simpler Validation-A notebook flow.
    """
    n_sub = _choose_validation_substeps(
        U2,
        dt_total,
        vx_t,
        vy_t,
        dx,
        dy,
        coefs,
        eps_visc,
        advection_mode=advection_mode,
        safety=safety,
        max_substeps=max_substeps,
    )
    dt = float(dt_total) / n_sub
    Ucur = U2

    rhs_fun = rhs_measured_advection_validation if advection_mode == "measured" else rhs_learned_advection

    for _ in range(n_sub):
        if integrator == "euler":
            Ucur = Ucur + dt * rhs_fun(Ucur, vx_t, vy_t, dx, dy, coefs, eps_visc)
        elif integrator == "heun":
            k1 = rhs_fun(Ucur, vx_t, vy_t, dx, dy, coefs, eps_visc)
            U1 = Ucur + dt * k1
            k2 = rhs_fun(U1, vx_t, vy_t, dx, dy, coefs, eps_visc)
            Ucur = Ucur + 0.5 * dt * (k1 + k2)
        else:
            raise ValueError("integrator must be 'euler' or 'heun'")

        if clip is not None:
            Ucur = torch.clamp(Ucur, float(clip[0]), float(clip[1]))

    return Ucur


def rollout_validation_model(
    U0: np.ndarray,
    t: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    *,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    advection_mode: str = "measured",
    integrator: str = "heun",
) -> np.ndarray:
    """
    GPU-enabled rollout over a validation window.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    t_np = np.asarray(t, dtype=float)
    vx_np = np.asarray(vx, dtype=float)
    vy_np = np.asarray(vy, dtype=float)

    Ucur = torch.as_tensor(U0, dtype=dtype, device=dev).clone()
    ny, nx = Ucur.shape
    nt_loc = len(t_np)

    Uout = torch.empty((ny, nx, nt_loc), dtype=dtype, device=dev)
    Uout[:, :, 0] = Ucur

    for k in range(nt_loc - 1):
        Ucur = advance_one_interval_validation(
            Ucur,
            dt_total=float(t_np[k + 1] - t_np[k]),
            vx_t=float(vx_np[k]),
            vy_t=float(vy_np[k]),
            dx=dx,
            dy=dy,
            coefs=coefs,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip,
            advection_mode=advection_mode,
            integrator=integrator,
        )
        Uout[:, :, k + 1] = Ucur

    return Uout.detach().cpu().numpy()


def one_step_validation_model(
    U_true: np.ndarray,
    t: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    *,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    advection_mode: str = "measured",
    integrator: str = "heun",
) -> np.ndarray:
    """
    GPU-enabled one-step prediction over a validation window.

    For each interval, the model starts from the true frame at time k and
    predicts only frame k+1.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    U_t = torch.as_tensor(U_true, dtype=dtype, device=dev)
    t_np = np.asarray(t, dtype=float)
    vx_np = np.asarray(vx, dtype=float)
    vy_np = np.asarray(vy, dtype=float)

    ny, nx, nt_loc = U_t.shape
    Uout = torch.empty_like(U_t)
    Uout[:, :, 0] = U_t[:, :, 0]

    for k in range(nt_loc - 1):
        Unext = advance_one_interval_validation(
            U_t[:, :, k].clone(),
            dt_total=float(t_np[k + 1] - t_np[k]),
            vx_t=float(vx_np[k]),
            vy_t=float(vy_np[k]),
            dx=dx,
            dy=dy,
            coefs=coefs,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip,
            advection_mode=advection_mode,
            integrator=integrator,
        )
        Uout[:, :, k + 1] = Unext

    return Uout.detach().cpu().numpy()


# Backward-compatible names for the Section-6 notebook helpers.
def rollout_learned_advection(
    U0: np.ndarray,
    t: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compatibility wrapper for Validation-A learned-advection rollout."""
    return rollout_validation_model(
        U0, t, vx, vy, dx, dy, coefs,
        eps_visc=eps_visc, safety=safety, max_substeps=max_substeps,
        clip=clip, device=device, advection_mode="learned", integrator="euler",
    )


def one_step_learned_advection(
    U_true: np.ndarray,
    t: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    coefs: Dict[str, float],
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compatibility wrapper for Validation-A learned-advection one-step prediction."""
    return one_step_validation_model(
        U_true, t, vx, vy, dx, dy, coefs,
        eps_visc=eps_visc, safety=safety, max_substeps=max_substeps,
        clip=clip, device=device, advection_mode="learned", integrator="euler",
    )


def plot_mse_validation(
    t_val: np.ndarray,
    m_roll: np.ndarray,
    m_one: np.ndarray,
    library_name: str,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
    title_prefix: str = "MSE over validation window",
) -> None:
    """Plot rollout and one-step MSE over the validation window."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(t_val, m_roll, label="rollout")
    ax.semilogy(t_val, m_one, label="one-step")
    ax.set_xlabel("t")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{title_prefix}\n{library_name}")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_front_radius_validation(
    t_val: np.ndarray,
    front: Dict[str, object],
    library_name: str,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
    title_prefix: str = "Front radius",
) -> None:
    """Plot true vs predicted equivalent front radius for all threshold levels."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for lev, d in front["per_level"].items():
        label_level = str(lev)
        ax.plot(t_val, d["r_true"], "-", label=f"true {label_level}")
        ax.plot(t_val, d["r_pred"], "--", label=f"pred {label_level}")

    ax.set_xlabel("t")
    ax.set_ylabel("equivalent front radius")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title_prefix}\n{library_name}")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_validation_snapshots(
    U_true: np.ndarray,
    U_pred: np.ndarray,
    t: np.ndarray,
    library_name: str,
    n_show: int = 6,
    cmap: str = "viridis",
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
    title_prefix: str = "Validation snapshots",
) -> None:
    """
    Plot true/predicted snapshots in a single saveable figure.
    """
    nt_loc = U_true.shape[2]
    ids = np.linspace(0, nt_loc - 1, min(n_show, nt_loc), dtype=int)

    fig, axes = plt.subplots(2, len(ids), figsize=(3.0 * len(ids), 6.0))
    if len(ids) == 1:
        axes = np.array(axes).reshape(2, 1)

    vmin = min(float(np.nanmin(U_true)), float(np.nanmin(U_pred)))
    vmax = max(float(np.nanmax(U_true)), float(np.nanmax(U_pred)))

    for j, k in enumerate(ids):
        ax0 = axes[0, j]
        ax0.imshow(U_true[:, :, k], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax0.set_title(f"true\nt={t[k]:.2f}")
        ax0.axis("off")

        ax1 = axes[1, j]
        ax1.imshow(U_pred[:, :, k], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax1.set_title(f"pred\nt={t[k]:.2f}")
        ax1.axis("off")

    fig.suptitle(f"{title_prefix}\n{library_name}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_validation_rmse_comparison(
    df_summary: pd.DataFrame,
    *,
    save_path: Optional[os.PathLike] = None,
    show: bool = True,
    title: str = "Validation comparison across discovered PDE models",
    x_label: str = "rollout relative RMSE (%)",
) -> None:
    """Horizontal bar plot comparing validation RMSE across libraries."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(df_summary["library"], df_summary["relative_rmse_percent"])
    ax.set_xlabel(x_label)
    ax.set_ylabel("library")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _save_validation_numeric_outputs(
    output_dir: Path,
    safe_name: str,
    prefix: str,
    rhs_coefs: Dict[str, float],
    U_val: np.ndarray,
    U_pred: np.ndarray,
    U_pred_one_step: np.ndarray,
    t_val: np.ndarray,
    vx_val: np.ndarray,
    vy_val: np.ndarray,
    m_roll: np.ndarray,
    m_one: np.ndarray,
) -> None:
    """Save per-library validation CSV and NPZ outputs."""
    df_mse = pd.DataFrame({
        "t": t_val,
        "mse_rollout": m_roll,
        "mse_one_step": m_one,
    })
    mse_csv_path = output_dir / f"{prefix}_mse_{safe_name}.csv"
    df_mse.to_csv(mse_csv_path, index=False)

    df_coef = pd.DataFrame({
        "rhs_key": list(rhs_coefs.keys()),
        "coefficient": list(rhs_coefs.values()),
    })
    coef_csv_path = output_dir / f"{prefix}_rhs_coefficients_{safe_name}.csv"
    df_coef.to_csv(coef_csv_path, index=False)

    arrays_path = output_dir / f"{prefix}_arrays_{safe_name}.npz"
    np.savez_compressed(
        arrays_path,
        U_val=U_val,
        U_pred=U_pred,
        U_pred_one_step=U_pred_one_step,
        t_val=t_val,
        vx_val=vx_val,
        vy_val=vy_val,
        mse_rollout=m_roll,
        mse_one_step=m_one,
    )

    print("Saved:", mse_csv_path)
    print("Saved:", coef_csv_path)
    print("Saved:", arrays_path)


def validate_one_model(
    library_name: str,
    system: WeakSystem,
    fit: SindyFit,
    U_val: np.ndarray,
    t_val: np.ndarray,
    vx_val: np.ndarray,
    vy_val: np.ndarray,
    grid: Grid,
    *,
    device: Optional[str] = None,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    active_tol: float = 0.0,
    use_measured_advection: bool = True,
    use_identified_advection_coefs: bool = False,
    front_levels: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    output_dir: Optional[os.PathLike] = None,
    make_plots: bool = True,
    show_plots: bool = True,
    n_snapshot_show: int = 6,
    cmap: str = "viridis",
    integrator: str = "heun",
) -> Dict[str, object]:
    """
    Validate one model using the Section-5 measured-advection convention.

    This uses measured ``vx(t), vy(t)`` with coefficient 1 by default.
    """
    safe_name = safe_filename(library_name)
    output_path = Path(output_dir) if output_dir is not None else None

    fit_coefs = get_active_fit_coefs(system, fit, tol=active_tol)
    rhs_coefs = map_fit_to_rhs_coefs(
        fit_coefs,
        use_measured_advection=use_measured_advection,
        use_identified_advection_coefs=use_identified_advection_coefs,
    )

    print_rhs_coefs(library_name, rhs_coefs)

    U_pred = rollout_validation_model(
        U_val[:, :, 0],
        t_val,
        vx_val,
        vy_val,
        dx=grid.dx,
        dy=grid.dy,
        coefs=rhs_coefs,
        eps_visc=eps_visc,
        safety=safety,
        max_substeps=max_substeps,
        clip=clip,
        device=device,
        advection_mode="measured",
        integrator=integrator,
    )

    U_pred1 = one_step_validation_model(
        U_val,
        t_val,
        vx_val,
        vy_val,
        dx=grid.dx,
        dy=grid.dy,
        coefs=rhs_coefs,
        eps_visc=eps_visc,
        safety=safety,
        max_substeps=max_substeps,
        clip=clip,
        device=device,
        advection_mode="measured",
        integrator=integrator,
    )

    rel_rmse = relative_rmse(U_val, U_pred)
    m_roll = mse_over_time(U_val, U_pred)
    m_one = mse_over_time(U_val, U_pred1)

    front = front_radius_error(U_val, U_pred, levels=front_levels, grid=grid)
    com = com_error(U_val, U_pred, grid)

    print()
    print(f"Rollout relative RMSE = {100.0 * rel_rmse:.2f} %")
    print(
        f"front-radius multi-level MAE = {front['mae_mean']:.4f}"
        f"   RMSE = {front['rmse_mean']:.4f}"
    )
    print(
        f"center-of-mass              MAE = {com['mae']:.4f}"
        f"   RMSE = {com['rmse']:.4f}"
    )

    if make_plots:
        mse_path = output_path / f"validation_mse_{safe_name}.png" if output_path else None
        front_path = output_path / f"validation_front_radius_{safe_name}.png" if output_path else None
        snap_path = output_path / f"validation_snapshots_{safe_name}.png" if output_path else None

        plot_mse_validation(
            t_val, m_roll, m_one, library_name,
            save_path=mse_path, show=show_plots,
        )
        plot_validation_snapshots(
            U_val, U_pred, t_val, library_name,
            n_show=n_snapshot_show, cmap=cmap,
            save_path=snap_path, show=show_plots,
        )
        plot_front_radius_validation(
            t_val, front, library_name,
            save_path=front_path, show=show_plots,
        )

    if output_path is not None:
        _save_validation_numeric_outputs(
            output_path, safe_name, "validation",
            rhs_coefs, U_val, U_pred, U_pred1,
            t_val, vx_val, vy_val, m_roll, m_one,
        )

    return {
        "library": library_name,
        "fit_coefs": fit_coefs,
        "rhs_coefs": rhs_coefs,
        "U_pred": U_pred,
        "U_pred_one_step": U_pred1,
        "mse_rollout": m_roll,
        "mse_one_step": m_one,
        "relative_rmse": rel_rmse,
        "front_mae": front["mae_mean"],
        "front_rmse": front["rmse_mean"],
        "com_mae": com["mae"],
        "com_rmse": com["rmse"],
        "front": front,
        "com": com,
    }


def validate_one_model_A_learned_advection(
    library_name: str,
    system: WeakSystem,
    fit: SindyFit,
    U_val: np.ndarray,
    t_val: np.ndarray,
    vx_val: np.ndarray,
    vy_val: np.ndarray,
    grid: Grid,
    *,
    device: Optional[str] = None,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] = None,
    active_tol: float = 1e-12,
    front_levels: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    output_dir: Optional[os.PathLike] = None,
    make_plots: bool = True,
    show_plots: bool = True,
    n_snapshot_show: int = 6,
    cmap: str = "viridis",
    integrator: str = "euler",
) -> Dict[str, object]:
    """
    Validate one model using learned advection coefficients.

    This is the reusable version of Section 6 / Validation A.
    """
    safe_name = safe_filename(library_name)
    output_path = Path(output_dir) if output_dir is not None else None

    fit_coefs = get_active_fit_coefs(system, fit, tol=active_tol)
    rhs_coefs = map_fit_to_rhs_coefs_learned_advection(fit_coefs)

    print_rhs_coefs(
        library_name,
        rhs_coefs,
        title_prefix="Validation A simulator coefficients",
    )

    U_pred = rollout_validation_model(
        U_val[:, :, 0],
        t_val,
        vx_val,
        vy_val,
        dx=grid.dx,
        dy=grid.dy,
        coefs=rhs_coefs,
        eps_visc=eps_visc,
        safety=safety,
        max_substeps=max_substeps,
        clip=clip,
        device=device,
        advection_mode="learned",
        integrator=integrator,
    )

    U_pred1 = one_step_validation_model(
        U_val,
        t_val,
        vx_val,
        vy_val,
        dx=grid.dx,
        dy=grid.dy,
        coefs=rhs_coefs,
        eps_visc=eps_visc,
        safety=safety,
        max_substeps=max_substeps,
        clip=clip,
        device=device,
        advection_mode="learned",
        integrator=integrator,
    )

    rel_rmse = relative_rmse(U_val, U_pred)
    m_roll = mse_over_time(U_val, U_pred)
    m_one = mse_over_time(U_val, U_pred1)
    front = front_radius_error(U_val, U_pred, levels=front_levels, grid=grid)
    com = com_error(U_val, U_pred, grid)

    print()
    print(f"Validation A rollout relative RMSE = {100.0 * rel_rmse:.2f} %")
    print(
        f"front-radius multi-level MAE = {front['mae_mean']:.4f}"
        f"   RMSE = {front['rmse_mean']:.4f}"
    )
    print(
        f"center-of-mass              MAE = {com['mae']:.4f}"
        f"   RMSE = {com['rmse']:.4f}"
    )

    if make_plots:
        mse_path = output_path / f"validation_A_mse_{safe_name}.png" if output_path else None
        front_path = output_path / f"validation_A_front_radius_{safe_name}.png" if output_path else None
        snap_path = output_path / f"validation_A_snapshots_{safe_name}.png" if output_path else None

        plot_mse_validation(
            t_val, m_roll, m_one, library_name,
            save_path=mse_path, show=show_plots,
            title_prefix="Validation A: MSE over validation window",
        )
        plot_validation_snapshots(
            U_val, U_pred, t_val, library_name,
            n_show=n_snapshot_show, cmap=cmap,
            save_path=snap_path, show=show_plots,
            title_prefix="Validation A: snapshots",
        )
        plot_front_radius_validation(
            t_val, front, library_name,
            save_path=front_path, show=show_plots,
            title_prefix="Validation A: Front radius",
        )

    if output_path is not None:
        _save_validation_numeric_outputs(
            output_path, safe_name, "validation_A",
            rhs_coefs, U_val, U_pred, U_pred1,
            t_val, vx_val, vy_val, m_roll, m_one,
        )

    return {
        "library": library_name,
        "fit_coefs": fit_coefs,
        "rhs_coefs": rhs_coefs,
        "U_pred": U_pred,
        "U_pred_one_step": U_pred1,
        "mse_rollout": m_roll,
        "mse_one_step": m_one,
        "relative_rmse": rel_rmse,
        "front_mae": front["mae_mean"],
        "front_rmse": front["rmse_mean"],
        "com_mae": com["mae"],
        "com_rmse": com["rmse"],
        "front": front,
        "com": com,
    }


def _validation_summary_dataframe(
    all_validation_results: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """Create a compact validation summary DataFrame."""
    rows = []
    for library_name, result in all_validation_results.items():
        rhs_string = ", ".join(
            f"{k}: {v:+.4e}" for k, v in result["rhs_coefs"].items()
        )
        rows.append({
            "library": library_name,
            "relative_rmse_percent": 100.0 * float(result["relative_rmse"]),
            "front_mae": float(result["front_mae"]),
            "front_rmse": float(result["front_rmse"]),
            "com_mae": float(result["com_mae"]),
            "com_rmse": float(result["com_rmse"]),
            "rhs_coefs": rhs_string,
        })
    return pd.DataFrame(rows).sort_values("relative_rmse_percent").reset_index(drop=True)


def _write_validation_metadata(
    output_dir: Path,
    filename: str,
    title: str,
    *,
    U_val: np.ndarray,
    t_val: np.ndarray,
    grid: Grid,
    eps_visc: float,
    safety: float,
    max_substeps: int,
    clip: Optional[Tuple[float, float]],
    active_tol: float,
    front_levels: Sequence[float],
    device: Optional[str],
    metadata: Optional[Dict[str, object]] = None,
    extra_lines: Optional[Sequence[str]] = None,
) -> Path:
    """Save validation run metadata."""
    metadata_path = output_dir / filename
    with open(metadata_path, "w") as f:
        f.write(title + "\n")
        f.write("=" * 80 + "\n")
        if extra_lines:
            for line in extra_lines:
                f.write(str(line) + "\n")
            f.write("\n")
        default_metadata = {
            "EPS_VISC": eps_visc,
            "SAFETY": safety,
            "MAX_SUBSTEPS": max_substeps,
            "CLIP": clip,
            "ACTIVE_TOL": active_tol,
            "FRONT_LEVELS": list(front_levels),
            "DEVICE": device,
            "U_val.shape": U_val.shape,
            "grid.dx": grid.dx,
            "grid.dy": grid.dy,
            "grid.dt": grid.dt,
            "t_val[0]": t_val[0] if len(t_val) else None,
            "t_val[-1]": t_val[-1] if len(t_val) else None,
        }
        if metadata:
            default_metadata.update(metadata)
        for key, value in default_metadata.items():
            f.write(f"{key} = {value}\n")
    print("Saved:", metadata_path)
    return metadata_path


def _zip_and_download_dir(
    output_dir: Path,
    zip_path: Path,
    *,
    download_zip: bool = True,
) -> None:
    """Zip an output directory and optionally download it in Colab."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_dir.rglob("*"):
            zipf.write(file_path, arcname=file_path.relative_to(output_dir))

    print("Created zip file:", zip_path)

    if download_zip:
        try:
            from google.colab import files  # type: ignore
            files.download(str(zip_path))
        except ImportError:
            print("Not running in Colab. Zip file saved at:", zip_path)


def run_validation_for_libraries(
    all_results: Dict[str, Dict[str, object]],
    U: np.ndarray,
    grid: Grid,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    device: Optional[str] = None,
    train_end_frac: float = 0.60,
    test_end_frac: float = 0.80,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] | str = "auto",
    active_tol: float = 0.0,
    use_measured_advection: bool = True,
    use_identified_advection_coefs: bool = False,
    front_levels: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    output_dir: Optional[os.PathLike] = None,
    save_results: bool = True,
    make_plots: bool = True,
    show_plots: bool = True,
    n_snapshot_show: int = 6,
    cmap: str = "viridis",
    integrator: str = "heun",
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
    metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame, Dict[str, Path]]:
    """
    Validate all discovered PDE models with the Section-5 measured-advection
    convention.

    ``download_zip`` defaults to True for Colab convenience. Set it to False
    to save files without downloading anything.
    """
    split = validation_split(
        U, grid, vx, vy,
        train_end_frac=train_end_frac,
        test_end_frac=test_end_frac,
    )
    U_val = split["U_val"]
    t_val = split["t_val"]
    vx_val = split["vx_val"]
    vy_val = split["vy_val"]

    if clip == "auto":
        clip_use = auto_clip_from_data(U_val)
    else:
        clip_use = clip

    paths: Dict[str, Path] = {}
    if save_results:
        if output_dir is None:
            output_dir = Path("/content/video_to_pde_validation_results")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths["output_dir"] = output_path
    else:
        output_path = None

    print("Validation window")
    print("-----------------")
    print(f"nt total       = {grid.nt}")
    print(f"validation idx = [{split['i_test_end']}, {grid.nt})")
    print(f"U_val shape    = {U_val.shape}")
    if len(t_val) > 0:
        print(f"t range        = [{t_val[0]:.4f}, {t_val[-1]:.4f}]")

    all_validation_results: Dict[str, Dict[str, object]] = {}

    for library_name, result in all_results.items():
        validation_result = validate_one_model(
            library_name=library_name,
            system=result["system"],
            fit=result["fit"],
            U_val=U_val,
            t_val=t_val,
            vx_val=vx_val,
            vy_val=vy_val,
            grid=grid,
            device=device,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip_use,
            active_tol=active_tol,
            use_measured_advection=use_measured_advection,
            use_identified_advection_coefs=use_identified_advection_coefs,
            front_levels=front_levels,
            output_dir=output_path,
            make_plots=make_plots,
            show_plots=show_plots,
            n_snapshot_show=n_snapshot_show,
            cmap=cmap,
            integrator=integrator,
        )
        all_validation_results[library_name] = validation_result

    df_summary = _validation_summary_dataframe(all_validation_results)

    if save_results and output_path is not None:
        summary_path = output_path / "validation_summary_all_models.csv"
        df_summary.to_csv(summary_path, index=False)
        print("Saved validation summary:", summary_path)

        if make_plots:
            comparison_plot_path = output_path / "validation_relative_rmse_comparison.png"
            plot_validation_rmse_comparison(
                df_summary,
                save_path=comparison_plot_path,
                show=show_plots,
                title="Validation comparison across discovered PDE models",
                x_label="rollout relative RMSE (%)",
            )
            print("Saved:", comparison_plot_path)

        _write_validation_metadata(
            output_path,
            "validation_metadata.txt",
            "Validation: measured advection coefficient convention",
            U_val=U_val,
            t_val=t_val,
            grid=grid,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip_use,
            active_tol=active_tol,
            front_levels=front_levels,
            device=device,
            metadata=metadata,
            extra_lines=[
                "This validation uses measured drift velocities with coefficient adv=1.0 by default.",
                "Learned v_x · u_x and v_y · u_y coefficients are ignored unless explicitly requested.",
            ],
        )

        if make_zip:
            if zip_path is None:
                zip_path = Path("/content/video_to_pde_validation_results.zip")
            zip_path = Path(zip_path)
            _zip_and_download_dir(output_path, zip_path, download_zip=download_zip)
            paths["zip_path"] = zip_path

    return all_validation_results, df_summary, paths


def run_validation_A_learned_advection_for_libraries(
    all_results: Dict[str, Dict[str, object]],
    U: np.ndarray,
    grid: Grid,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    device: Optional[str] = None,
    train_end_frac: float = 0.60,
    test_end_frac: float = 0.80,
    eps_visc: float = 0.01,
    safety: float = 0.25,
    max_substeps: int = 2000,
    clip: Optional[Tuple[float, float]] | str = "auto",
    active_tol: float = 1e-12,
    front_levels: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    output_dir: Optional[os.PathLike] = None,
    save_results: bool = True,
    make_plots: bool = True,
    show_plots: bool = True,
    n_snapshot_show: int = 6,
    cmap: str = "viridis",
    integrator: str = "euler",
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
    metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame, Dict[str, Path]]:
    """
    Validate all discovered PDE models using learned advection coefficients.

    This is the high-level reusable version of Section 6 / Validation A.
    ``download_zip`` defaults to True for Colab convenience. Set it to False
    to save files without downloading anything.
    """
    split = validation_split(
        U, grid, vx, vy,
        train_end_frac=train_end_frac,
        test_end_frac=test_end_frac,
    )
    U_val = split["U_val"]
    t_val = split["t_val"]
    vx_val = split["vx_val"]
    vy_val = split["vy_val"]

    if clip == "auto":
        clip_use = auto_clip_from_data(U_val)
    else:
        clip_use = clip

    paths: Dict[str, Path] = {}
    if save_results:
        if output_dir is None:
            output_dir = Path("/content/video_to_pde_validation_A_learned_advection")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths["output_dir"] = output_path
    else:
        output_path = None

    print("Validation A window")
    print("-------------------")
    print(f"nt total       = {grid.nt}")
    print(f"validation idx = [{split['i_test_end']}, {grid.nt})")
    print(f"U_val shape    = {U_val.shape}")
    if len(t_val) > 0:
        print(f"t range        = [{t_val[0]:.4f}, {t_val[-1]:.4f}]")

    all_validation_results: Dict[str, Dict[str, object]] = {}

    for library_name, result in all_results.items():
        validation_result = validate_one_model_A_learned_advection(
            library_name=library_name,
            system=result["system"],
            fit=result["fit"],
            U_val=U_val,
            t_val=t_val,
            vx_val=vx_val,
            vy_val=vy_val,
            grid=grid,
            device=device,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip_use,
            active_tol=active_tol,
            front_levels=front_levels,
            output_dir=output_path,
            make_plots=make_plots,
            show_plots=show_plots,
            n_snapshot_show=n_snapshot_show,
            cmap=cmap,
            integrator=integrator,
        )
        all_validation_results[library_name] = validation_result

    df_summary = _validation_summary_dataframe(all_validation_results)

    if save_results and output_path is not None:
        summary_path = output_path / "validation_A_summary_all_models.csv"
        df_summary.to_csv(summary_path, index=False)
        print("Saved Validation A summary:", summary_path)

        if make_plots:
            comparison_plot_path = output_path / "validation_A_relative_rmse_comparison.png"
            plot_validation_rmse_comparison(
                df_summary,
                save_path=comparison_plot_path,
                show=show_plots,
                title="Validation A comparison: learned advection coefficients",
                x_label="Validation A rollout relative RMSE (%)",
            )
            print("Saved:", comparison_plot_path)

        _write_validation_metadata(
            output_path,
            "validation_A_metadata.txt",
            "Validation A: learned advection coefficients",
            U_val=U_val,
            t_val=t_val,
            grid=grid,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip_use,
            active_tol=active_tol,
            front_levels=front_levels,
            device=device,
            metadata=metadata,
            extra_lines=[
                "This validation uses learned coefficients for v_x u_x and v_y u_y.",
                "It does NOT impose measured advection with coefficient 1.",
            ],
        )

        if make_zip:
            if zip_path is None:
                zip_path = Path("/content/video_to_pde_validation_A_learned_advection.zip")
            zip_path = Path(zip_path)
            _zip_and_download_dir(output_path, zip_path, download_zip=download_zip)
            paths["zip_path"] = zip_path

    return all_validation_results, df_summary, paths

def save_field_animation(
    U,
    grid,
    save_path,
    *,
    fps=30,
    cmap="gray_r",
    row_index=None,
    figsize=(10, 4),
    dpi=120,
    title_prefix="Frame",
    close_fig=True,
):
    """
    Save an animation of a scalar field U(y, x, t) together with a
    mid-row cross-section.

    Parameters
    ----------
    U : np.ndarray or torch.Tensor
        Scalar field with shape (ny, nx, nt). Can be a GPU tensor.
    grid : Grid
        Grid object containing x, t, ny, nt.
    save_path : str or Path
        Output path, usually ending in .gif.
    fps : int
        Frames per second for the GIF.
    cmap : str
        Matplotlib colormap.
    row_index : int or None
        Row used for the cross-section. If None, uses middle row.
    figsize : tuple
        Figure size.
    dpi : int
        Output resolution.
    title_prefix : str
        Prefix for frame title.
    close_fig : bool
        If True, closes the figure after saving.

    Returns
    -------
    save_path : str
        Path to the saved GIF.
    """

    # Convert torch tensor, including GPU tensor, to NumPy
    if torch.is_tensor(U):
        U_cpu = U.detach().cpu().numpy()
    else:
        U_cpu = np.asarray(U)

    if U_cpu.ndim != 3:
        raise ValueError(f"Expected U with shape (ny, nx, nt), got {U_cpu.shape}")

    ny, nx, nt = U_cpu.shape

    if nt != grid.nt:
        raise ValueError(f"U has nt={nt}, but grid.nt={grid.nt}")

    if row_index is None:
        row_index = ny // 2

    row_index = int(np.clip(row_index, 0, ny - 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: spatial field
    im = ax1.imshow(U_cpu[:, :, 0], cmap=cmap, animated=True)
    ax1.axis("off")
    title1 = ax1.set_title(f"{title_prefix} t={grid.t[0]:.2f}s")

    # Right: cross-section
    line, = ax2.plot(grid.x, U_cpu[row_index, :, 0])
    ax2.set_title(f"Row {row_index} cross-section")
    ax2.set_xlabel("x")
    ax2.set_ylabel("u")
    ax2.grid(True, alpha=0.3)

    u_min, u_max = float(np.min(U_cpu)), float(np.max(U_cpu))
    pad = 0.05 * max(u_max - u_min, 1e-12)
    ax2.set_ylim(u_min - pad, u_max + pad)

    plt.tight_layout()

    def update(k):
        im.set_array(U_cpu[:, :, k])
        title1.set_text(f"{title_prefix} t={grid.t[k]:.2f}s")
        line.set_ydata(U_cpu[row_index, :, k])
        return im, title1, line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        blit=True,
    )

    save_path = str(save_path)
    ani.save(save_path, writer="pillow", fps=fps, dpi=dpi)

    if close_fig:
        plt.close(fig)

    return save_path

# ============================================================================
# Section 15. iPINN coefficient refinement for Library C and C-alt
# ============================================================================
import time
from contextlib import contextmanager
import torch.nn as nn


LIBRARY_C_NAME = "C: advection-diffusion + |grad u|^2"
LIBRARY_C_ALT_NAME = "C-alt: advection-diffusion + u|grad u|^2"


@dataclass
class IPINNSettings:
    """Configuration for inverse-PINN coefficient refinement."""
    mode: str = "final"
    pretrain_steps: int = 5000
    joint_steps: int = 12000
    n_data: int = 8192
    n_pde: int = 4096
    hidden_dim: int = 128
    n_layers: int = 6
    lr_pretrain: float = 8e-4
    lr_joint: float = 2e-4
    print_every: int = 500
    data_weight: float = 1.0
    pde_weight: float = 1e-3
    coef_prior_weight: float = 0.0
    seed: int = 2026
    use_sigmoid_output: bool = True


def make_ipinn_settings(mode: str = "final", **overrides) -> IPINNSettings:
    """
    Create iPINN settings from a named preset, with optional field overrides.

    Presets
    -------
    smoke
        Fast syntax/runtime test.
    production
        Serious coefficient estimate.
    final
        Heavier run for final reported numbers, if runtime is acceptable.
    """
    presets = {
        "smoke": dict(
            pretrain_steps=500, joint_steps=1000, n_data=2048, n_pde=2048,
            hidden_dim=64, n_layers=4, lr_pretrain=1e-3, lr_joint=5e-4,
            print_every=100,
        ),
        "production": dict(
            pretrain_steps=2500, joint_steps=6000, n_data=4096, n_pde=4096,
            hidden_dim=96, n_layers=5, lr_pretrain=1e-3, lr_joint=3e-4,
            print_every=250,
        ),
        "final": dict(
            pretrain_steps=5000, joint_steps=12000, n_data=8192, n_pde=4096,
            hidden_dim=128, n_layers=6, lr_pretrain=8e-4, lr_joint=2e-4,
            print_every=500,
        ),
    }
    if mode not in presets:
        raise ValueError(f"Unknown iPINN mode {mode!r}; choose from {sorted(presets)}")
    data = dict(mode=mode)
    data.update(presets[mode])
    data.update(overrides)
    return IPINNSettings(**data)


def get_fit_coef_by_feature(system: WeakSystem,
                            fit: SindyFit,
                            feature_name: str,
                            default: float = 0.0) -> float:
    """Return the fitted coefficient for a named weak-SINDy feature."""
    for name, c in zip(system.feature_names, fit.coef):
        if name == feature_name:
            return float(c)
    return float(default)


class IPINNNet(nn.Module):
    """Small tanh MLP used for inverse-PINN coefficient refinement."""

    def __init__(self,
                 hidden_dim: int = 64,
                 n_layers: int = 4,
                 use_sigmoid: bool = True):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_sigmoid = bool(use_sigmoid)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


@dataclass
class IPINNDataContext:
    """Torch tensors and normalization constants used by the iPINN sampler."""
    U_torch: torch.Tensor
    xhat_torch: torch.Tensor
    yhat_torch: torch.Tensor
    that_torch: torch.Tensor
    vx_torch: torch.Tensor
    vy_torch: torch.Tensor
    sx_t: torch.Tensor
    sy_t: torch.Tensor
    st_t: torch.Tensor
    nx: int
    ny: int
    nt: int
    train_t_start: int
    train_t_end: int
    val_t_start: int
    val_t_end: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float
    device: torch.device


def _resolve_torch_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return dev


def _set_ipinn_seed(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _normalize_numpy(vals: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return 2.0 * (vals - vmin) / max(vmax - vmin, 1e-12) - 1.0


def prepare_ipinn_context(U: np.ndarray,
                          grid: Grid,
                          vx: np.ndarray,
                          vy: np.ndarray,
                          *,
                          device: Optional[str] = None,
                          train_end_frac: float = 0.60,
                          test_end_frac: float = 0.80) -> IPINNDataContext:
    """Move video data and normalized coordinates to the chosen torch device."""
    dev = _resolve_torch_device(device)
    ny, nx, nt = U.shape
    if int(grid.nt) != nt:
        raise ValueError(f"grid.nt={grid.nt}, but U.shape[2]={nt}")

    i1 = int(round(train_end_frac * nt))
    i2 = int(round(test_end_frac * nt))
    i1 = int(np.clip(i1, 1, nt))
    i2 = int(np.clip(i2, i1, nt - 1 if nt > 1 else nt))

    x_np = np.asarray(getattr(grid, "x", np.arange(nx) * grid.dx), dtype=np.float32)
    y_np = np.asarray(getattr(grid, "y", np.arange(ny) * grid.dy), dtype=np.float32)
    t_np = np.asarray(grid.t, dtype=np.float32)
    vx_np = np.asarray(vx, dtype=np.float32)
    vy_np = np.asarray(vy, dtype=np.float32)
    U_np = np.asarray(U, dtype=np.float32)

    x_min, x_max = float(x_np.min()), float(x_np.max())
    y_min, y_max = float(y_np.min()), float(y_np.max())
    t_min, t_max = float(t_np.min()), float(t_np.max())

    sx = 2.0 / max(x_max - x_min, 1e-12)
    sy = 2.0 / max(y_max - y_min, 1e-12)
    st = 2.0 / max(t_max - t_min, 1e-12)

    return IPINNDataContext(
        U_torch=torch.tensor(U_np, dtype=torch.float32, device=dev),
        xhat_torch=torch.tensor(_normalize_numpy(x_np, x_min, x_max), dtype=torch.float32, device=dev),
        yhat_torch=torch.tensor(_normalize_numpy(y_np, y_min, y_max), dtype=torch.float32, device=dev),
        that_torch=torch.tensor(_normalize_numpy(t_np, t_min, t_max), dtype=torch.float32, device=dev),
        vx_torch=torch.tensor(vx_np, dtype=torch.float32, device=dev),
        vy_torch=torch.tensor(vy_np, dtype=torch.float32, device=dev),
        sx_t=torch.tensor(sx, dtype=torch.float32, device=dev),
        sy_t=torch.tensor(sy, dtype=torch.float32, device=dev),
        st_t=torch.tensor(st, dtype=torch.float32, device=dev),
        nx=nx,
        ny=ny,
        nt=nt,
        train_t_start=0,
        train_t_end=i1,
        val_t_start=i2,
        val_t_end=nt,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        t_min=t_min,
        t_max=t_max,
        device=dev,
    )


def sample_ipinn_points(ctx: IPINNDataContext,
                        n_points: int,
                        t_start: int,
                        t_end: int,
                        rng: np.random.Generator,
                        *,
                        requires_grad: bool = True
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample random discrete video-grid points for iPINN data/PDE losses.

    Returns normalized coordinates ``z=[x_hat,y_hat,t_hat]``, observed data,
    and measured drift velocities at sampled times.
    """
    if t_end <= t_start:
        raise ValueError(f"Invalid time sampling range [{t_start}, {t_end})")
    ix = rng.integers(0, ctx.nx, size=int(n_points))
    iy = rng.integers(0, ctx.ny, size=int(n_points))
    it = rng.integers(t_start, t_end, size=int(n_points))

    ix_t = torch.tensor(ix, dtype=torch.long, device=ctx.device)
    iy_t = torch.tensor(iy, dtype=torch.long, device=ctx.device)
    it_t = torch.tensor(it, dtype=torch.long, device=ctx.device)

    z = torch.stack(
        [ctx.xhat_torch[ix_t], ctx.yhat_torch[iy_t], ctx.that_torch[it_t]],
        dim=1,
    )
    z.requires_grad_(requires_grad)

    u_data = ctx.U_torch[iy_t, ix_t, it_t].reshape(-1, 1)
    vx_batch = ctx.vx_torch[it_t].reshape(-1, 1)
    vy_batch = ctx.vy_torch[it_t].reshape(-1, 1)
    return z, u_data, vx_batch, vy_batch


def compute_ipinn_derivatives(model: nn.Module,
                              z: torch.Tensor,
                              ctx: IPINNDataContext) -> Dict[str, torch.Tensor]:
    """
    Compute u, physical derivatives, Laplacian, and |grad u|^2 by autodiff.
    """
    u = model(z)
    grad_u = torch.autograd.grad(
        u,
        z,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xhat = grad_u[:, 0:1]
    u_yhat = grad_u[:, 1:2]
    u_that = grad_u[:, 2:3]

    u_x = ctx.sx_t * u_xhat
    u_y = ctx.sy_t * u_yhat
    u_t = ctx.st_t * u_that

    grad_u_xhat = torch.autograd.grad(
        u_xhat,
        z,
        grad_outputs=torch.ones_like(u_xhat),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_u_yhat = torch.autograd.grad(
        u_yhat,
        z,
        grad_outputs=torch.ones_like(u_yhat),
        create_graph=True,
        retain_graph=True,
    )[0]

    lap = (ctx.sx_t ** 2) * grad_u_xhat[:, 0:1] + (ctx.sy_t ** 2) * grad_u_yhat[:, 1:2]
    grad2 = u_x ** 2 + u_y ** 2
    return {"u": u, "u_t": u_t, "u_x": u_x, "u_y": u_y, "lap": lap, "grad2": grad2}


def ipinn_pde_residual(model: nn.Module,
                       z: torch.Tensor,
                       vx_batch: torch.Tensor,
                       vy_batch: torch.Tensor,
                       coef_a: torch.Tensor,
                       coef_beta: torch.Tensor,
                       *,
                       model_type: str,
                       ctx: IPINNDataContext) -> torch.Tensor:
    """
    PDE residual for C and C-alt iPINN refinement.

    C:
        u_t + v·grad u - a |grad u|^2 - beta Δu
    Calt:
        u_t + v·grad u - a u |grad u|^2 - beta Δu
    """
    d = compute_ipinn_derivatives(model, z, ctx)
    adv = vx_batch * d["u_x"] + vy_batch * d["u_y"]
    if model_type == "C":
        intrinsic = coef_a * d["grad2"] + coef_beta * d["lap"]
    elif model_type in {"Calt", "C-alt", "C_alt"}:
        intrinsic = coef_a * d["u"] * d["grad2"] + coef_beta * d["lap"]
    else:
        raise ValueError("model_type must be 'C' or 'Calt'")
    return d["u_t"] + adv - intrinsic


def train_ipinn_coefficients(model_type: str,
                             init_a: float,
                             init_beta: float,
                             label: str,
                             ctx: IPINNDataContext,
                             settings: IPINNSettings,
                             *,
                             seed_offset: int = 0,
                             verbose: bool = True) -> Dict[str, object]:
    """Train one inverse PINN and return fitted coefficients plus history."""
    rng = np.random.default_rng(settings.seed + seed_offset)
    if verbose:
        print("\n" + "=" * 100)
        print(f"Training iPINN: {label}")
        print("=" * 100)
        print(f"Initial a     = {init_a:+.6e}")
        print(f"Initial beta  = {init_beta:+.6e}")

    model = IPINNNet(
        hidden_dim=settings.hidden_dim,
        n_layers=settings.n_layers,
        use_sigmoid=settings.use_sigmoid_output,
    ).to(ctx.device)

    coef_a = nn.Parameter(torch.tensor(float(init_a), dtype=torch.float32, device=ctx.device))
    coef_beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32, device=ctx.device))
    init_a_t = torch.tensor(float(init_a), dtype=torch.float32, device=ctx.device)
    init_beta_t = torch.tensor(float(init_beta), dtype=torch.float32, device=ctx.device)

    mse = nn.MSELoss()
    history = {
        "step": [], "phase": [], "loss_total": [], "loss_data": [], "loss_pde": [],
        "coef_a": [], "coef_beta": [],
    }

    def log_record(step: int, phase: str, loss_total, loss_data, loss_pde) -> None:
        history["step"].append(int(step))
        history["phase"].append(str(phase))
        history["loss_total"].append(float(loss_total))
        history["loss_data"].append(float(loss_data))
        history["loss_pde"].append(float(loss_pde) if loss_pde is not None else np.nan)
        history["coef_a"].append(float(coef_a.detach().cpu().item()))
        history["coef_beta"].append(float(coef_beta.detach().cpu().item()))

    t0 = time.perf_counter()

    # Phase 1: fit the neural representation to data.
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=settings.lr_pretrain)
    for step in range(1, settings.pretrain_steps + 1):
        z_data, u_data, _, _ = sample_ipinn_points(
            ctx, settings.n_data, ctx.train_t_start, ctx.train_t_end, rng, requires_grad=True
        )
        u_pred = model(z_data)
        loss_data = mse(u_pred, u_data)
        optimizer_pre.zero_grad(set_to_none=True)
        loss_data.backward()
        optimizer_pre.step()

        if step % settings.print_every == 0 or step == 1 or step == settings.pretrain_steps:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(
                    f"[{label}] pretrain step {step:6d}/{settings.pretrain_steps} | "
                    f"data={loss_data.item():.6e} | "
                    f"a={coef_a.item():+.6e}, beta={coef_beta.item():+.6e} | "
                    f"time={elapsed:.1f}s"
                )
            log_record(step, "pretrain", loss_data.item(), loss_data.item(), None)

    # Phase 2: fit data and PDE residual, including coefficients.
    optimizer_joint = torch.optim.Adam(
        list(model.parameters()) + [coef_a, coef_beta], lr=settings.lr_joint
    )
    for step in range(1, settings.joint_steps + 1):
        z_data, u_data, _, _ = sample_ipinn_points(
            ctx, settings.n_data, ctx.train_t_start, ctx.train_t_end, rng, requires_grad=True
        )
        loss_data = mse(model(z_data), u_data)

        z_pde, _, vx_batch, vy_batch = sample_ipinn_points(
            ctx, settings.n_pde, ctx.train_t_start, ctx.train_t_end, rng, requires_grad=True
        )
        res = ipinn_pde_residual(
            model, z_pde, vx_batch, vy_batch, coef_a, coef_beta,
            model_type=model_type, ctx=ctx,
        )
        loss_pde = torch.mean(res ** 2)
        loss_prior = (coef_a - init_a_t) ** 2 + (coef_beta - init_beta_t) ** 2
        loss_total = (
            settings.data_weight * loss_data
            + settings.pde_weight * loss_pde
            + settings.coef_prior_weight * loss_prior
        )

        optimizer_joint.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer_joint.step()

        if step % settings.print_every == 0 or step == 1 or step == settings.joint_steps:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(
                    f"[{label}] joint step {step:6d}/{settings.joint_steps} | "
                    f"total={loss_total.item():.6e} | data={loss_data.item():.6e} | "
                    f"pde={loss_pde.item():.6e} | "
                    f"a={coef_a.item():+.6e}, beta={coef_beta.item():+.6e} | "
                    f"time={elapsed:.1f}s"
                )
            log_record(step, "joint", loss_total.item(), loss_data.item(), loss_pde.item())

    model.eval()
    with torch.no_grad():
        n_val = min(20000, ctx.nx * ctx.ny * max(ctx.val_t_end - ctx.val_t_start, 1))
        z_val, u_val_sample, _, _ = sample_ipinn_points(
            ctx, n_val, ctx.val_t_start, ctx.val_t_end, rng, requires_grad=False
        )
        val_data_mse = torch.mean((model(z_val) - u_val_sample) ** 2).item()
        val_data_rmse = float(np.sqrt(val_data_mse))

    coef_a_final = float(coef_a.detach().cpu().item())
    coef_beta_final = float(coef_beta.detach().cpu().item())

    if verbose:
        print("\nFinal iPINN coefficients")
        print("------------------------")
        print(f"{label}")
        print(f"  a / nonlinear coefficient = {coef_a_final:+.8e}")
        print(f"  beta / Laplacian coeff    = {coef_beta_final:+.8e}")
        print(f"  validation data RMSE      = {val_data_rmse:.6e}")

    return {
        "label": label,
        "model_type": model_type,
        "model": model,
        "coef_a": coef_a_final,
        "coef_beta": coef_beta_final,
        "val_data_rmse": val_data_rmse,
        "history": pd.DataFrame(history),
        "init_a": float(init_a),
        "init_beta": float(init_beta),
    }


def _plot_ipinn_histories(ipinn_results: Dict[str, Dict[str, object]],
                          output_dir: Optional[Path] = None,
                          *,
                          show: bool = True) -> None:
    """Save/show iPINN loss and coefficient-history plots."""
    for key, result in ipinn_results.items():
        df_hist = result["history"]
        if not isinstance(df_hist, pd.DataFrame) or df_hist.empty:
            continue

        # Loss history.
        fig = plt.figure(figsize=(8, 4))
        df_joint = df_hist[df_hist["phase"] == "joint"]
        plt.semilogy(df_hist.index, df_hist["loss_total"], label="total")
        plt.semilogy(df_hist.index, df_hist["loss_data"], label="data")
        if len(df_joint) > 0 and df_joint["loss_pde"].notna().any():
            plt.semilogy(df_joint.index, df_joint["loss_pde"], label="PDE")
        plt.xlabel("logged iteration index")
        plt.ylabel("loss")
        plt.title(f"iPINN loss history: {key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if output_dir is not None:
            loss_path = output_dir / f"{safe_filename(key)}_loss_history.png"
            fig.savefig(loss_path, dpi=1200, bbox_inches="tight")
            print("Saved:", loss_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

        # Coefficient history.
        fig = plt.figure(figsize=(8, 4))
        plt.plot(df_hist.index, df_hist["coef_a"], label="a")
        plt.plot(df_hist.index, df_hist["coef_beta"], label="beta")
        plt.xlabel("logged iteration index")
        plt.ylabel("coefficient value")
        plt.title(f"iPINN coefficient history: {key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if output_dir is not None:
            coef_path = output_dir / f"{safe_filename(key)}_coefficient_history.png"
            fig.savefig(coef_path, dpi=1200, bbox_inches="tight")
            print("Saved:", coef_path)
        if show:
            plt.show()
        else:
            plt.close(fig)


def run_ipinn_refinement_C_Calt(
    all_results: Dict[str, Dict[str, object]],
    U: np.ndarray,
    grid: Grid,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    mode: str = "final",
    settings: Optional[IPINNSettings] = None,
    device: Optional[str] = None,
    train_end_frac: float = 0.60,
    test_end_frac: float = 0.80,
    output_dir: Optional[os.PathLike] = None,
    save_results: bool = True,
    make_plots: bool = True,
    show_plots: bool = True,
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
    verbose: bool = True,
    **settings_overrides,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, float], Dict[str, float], pd.DataFrame, Dict[str, Path]]:
    """
    Run iPINN coefficient refinement for Library C and C-alt only.

    Returns
    -------
    ipinn_results, ipinn_init_C, ipinn_init_Calt, df_summary, paths
    """
    settings = settings if settings is not None else make_ipinn_settings(mode, **settings_overrides)
    dev = _resolve_torch_device(device)
    _set_ipinn_seed(settings.seed, dev)

    if LIBRARY_C_NAME not in all_results:
        raise RuntimeError(f"Missing library in all_results: {LIBRARY_C_NAME}")
    if LIBRARY_C_ALT_NAME not in all_results:
        raise RuntimeError(f"Missing library in all_results: {LIBRARY_C_ALT_NAME}")

    if save_results:
        if output_dir is None:
            output_dir = Path(f"/content/ipinn_refinement_C_Calt_{settings.mode}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    paths: Dict[str, Path] = {}
    if output_path is not None:
        paths["output_dir"] = output_path

    if verbose:
        print("Using torch device:", dev)
        if dev.type == "cuda":
            try:
                print("GPU:", torch.cuda.get_device_name(0))
            except Exception:
                pass
        print("\niPINN configuration")
        print("-------------------")
        for key, value in settings.__dict__.items():
            print(f"{key} = {value}")

    ctx = prepare_ipinn_context(
        U, grid, vx, vy, device=str(dev), train_end_frac=train_end_frac,
        test_end_frac=test_end_frac,
    )
    if verbose:
        print("\nData split")
        print("----------")
        print(f"U shape            : {U.shape}")
        print(f"Training time idx  : [{ctx.train_t_start}, {ctx.train_t_end})")
        print(f"Validation time idx: [{ctx.val_t_start}, {ctx.val_t_end})")
        print("\nCoordinate ranges")
        print("-----------------")
        print(f"x in [{ctx.x_min:.6g}, {ctx.x_max:.6g}]")
        print(f"y in [{ctx.y_min:.6g}, {ctx.y_max:.6g}]")
        print(f"t in [{ctx.t_min:.6g}, {ctx.t_max:.6g}]")

    system_C = all_results[LIBRARY_C_NAME]["system"]
    fit_C = all_results[LIBRARY_C_NAME]["fit"]
    system_Calt = all_results[LIBRARY_C_ALT_NAME]["system"]
    fit_Calt = all_results[LIBRARY_C_ALT_NAME]["fit"]

    init_C_a = get_fit_coef_by_feature(system_C, fit_C, "|grad u|^2", default=0.8)
    init_C_beta = get_fit_coef_by_feature(system_C, fit_C, "Delta u", default=-0.1)
    init_Calt_a = get_fit_coef_by_feature(system_Calt, fit_Calt, "u|grad u|^2", default=2.5)
    init_Calt_beta = get_fit_coef_by_feature(system_Calt, fit_Calt, "Delta u", default=0.0)

    if verbose:
        print("\nWeak-SINDy initialization")
        print("-------------------------")
        print(f"C     : grad2    = {init_C_a:+.6e}, lap = {init_C_beta:+.6e}")
        print(f"C-alt : u_grad2 = {init_Calt_a:+.6e}, lap = {init_Calt_beta:+.6e}")

    ipinn_results: Dict[str, Dict[str, object]] = {}
    ipinn_results["C_ipinn"] = train_ipinn_coefficients(
        model_type="C",
        init_a=init_C_a,
        init_beta=init_C_beta,
        label="C_ipinn: |grad u|^2 + lap",
        ctx=ctx,
        settings=settings,
        seed_offset=0,
        verbose=verbose,
    )
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    ipinn_results["Calt_ipinn"] = train_ipinn_coefficients(
        model_type="Calt",
        init_a=init_Calt_a,
        init_beta=init_Calt_beta,
        label="Calt_ipinn: u|grad u|^2 + lap",
        ctx=ctx,
        settings=settings,
        seed_offset=10_000,
        verbose=verbose,
    )
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    ipinn_init_C = {
        "grad2": float(ipinn_results["C_ipinn"]["coef_a"]),
        "lap": float(ipinn_results["C_ipinn"]["coef_beta"]),
        "adv": 1.0,
    }
    ipinn_init_Calt = {
        "u_grad2": float(ipinn_results["Calt_ipinn"]["coef_a"]),
        "lap": float(ipinn_results["Calt_ipinn"]["coef_beta"]),
        "adv": 1.0,
    }

    if verbose:
        print("\n" + "=" * 100)
        print("iPINN INITIALIZATIONS FOR BOOTSTRAP")
        print("=" * 100)
        print("\nipinn_init_C")
        print(ipinn_init_C)
        print("\nipinn_init_Calt")
        print(ipinn_init_Calt)

    df_summary = pd.DataFrame([
        {
            "model": "C",
            "pde_form": "u_t + v·grad u = a |grad u|^2 + beta Delta u",
            "init_source": "Weak-SINDy",
            "weak_sindy_a": init_C_a,
            "weak_sindy_beta": init_C_beta,
            "ipinn_a": ipinn_init_C["grad2"],
            "ipinn_beta": ipinn_init_C["lap"],
            "val_data_rmse": ipinn_results["C_ipinn"]["val_data_rmse"],
        },
        {
            "model": "C-alt",
            "pde_form": "u_t + v·grad u = a u|grad u|^2 + beta Delta u",
            "init_source": "Weak-SINDy",
            "weak_sindy_a": init_Calt_a,
            "weak_sindy_beta": init_Calt_beta,
            "ipinn_a": ipinn_init_Calt["u_grad2"],
            "ipinn_beta": ipinn_init_Calt["lap"],
            "val_data_rmse": ipinn_results["Calt_ipinn"]["val_data_rmse"],
        },
    ])

    if save_results and output_path is not None:
        summary_path = output_path / "ipinn_coefficient_summary_C_Calt.csv"
        df_summary.to_csv(summary_path, index=False)
        print("Saved:", summary_path)

        for key, result in ipinn_results.items():
            hist_path = output_path / f"{safe_filename(key)}_training_history.csv"
            result["history"].to_csv(hist_path, index=False)
            print("Saved:", hist_path)

        df_bootstrap_init = pd.DataFrame([
            {"bootstrap_model": "C_unconstrained_lap", "rhs_key": "grad2", "coefficient": ipinn_init_C["grad2"]},
            {"bootstrap_model": "C_unconstrained_lap", "rhs_key": "lap", "coefficient": ipinn_init_C["lap"]},
            {"bootstrap_model": "C_unconstrained_lap", "rhs_key": "adv", "coefficient": ipinn_init_C["adv"]},
            {"bootstrap_model": "Calt_unconstrained_lap", "rhs_key": "u_grad2", "coefficient": ipinn_init_Calt["u_grad2"]},
            {"bootstrap_model": "Calt_unconstrained_lap", "rhs_key": "lap", "coefficient": ipinn_init_Calt["lap"]},
            {"bootstrap_model": "Calt_unconstrained_lap", "rhs_key": "adv", "coefficient": ipinn_init_Calt["adv"]},
        ])
        init_path = output_path / "ipinn_bootstrap_initializations.csv"
        df_bootstrap_init.to_csv(init_path, index=False)
        print("Saved:", init_path)

        metadata_path = output_path / "ipinn_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write("iPINN coefficient refinement for Video-to-PDE discovery\n")
            f.write("=" * 80 + "\n")
            for key, value in settings.__dict__.items():
                f.write(f"{key} = {value}\n")
            f.write(f"device = {dev}\n")
            f.write(f"U.shape = {U.shape}\n")
            f.write(f"grid.dx = {grid.dx}\n")
            f.write(f"grid.dy = {grid.dy}\n")
            f.write(f"grid.nt = {grid.nt}\n")
            f.write("\nModel C:\n")
            f.write("u_t + v(t)·grad u = a |grad u|^2 + beta Delta u\n")
            f.write(f"Weak-SINDy init: a={init_C_a}, beta={init_C_beta}\n")
            f.write(f"iPINN result: a={ipinn_init_C['grad2']}, beta={ipinn_init_C['lap']}\n")
            f.write("\nModel C-alt:\n")
            f.write("u_t + v(t)·grad u = a u|grad u|^2 + beta Delta u\n")
            f.write(f"Weak-SINDy init: a={init_Calt_a}, beta={init_Calt_beta}\n")
            f.write(f"iPINN result: a={ipinn_init_Calt['u_grad2']}, beta={ipinn_init_Calt['lap']}\n")
        print("Saved:", metadata_path)

        if make_plots:
            _plot_ipinn_histories(ipinn_results, output_path, show=show_plots)

        if make_zip:
            if zip_path is None:
                zip_path = Path(f"/content/ipinn_refinement_C_Calt_{settings.mode}.zip")
            zip_path = Path(zip_path)
            _zip_and_download_dir(output_path, zip_path, download_zip=download_zip)
            paths["zip_path"] = zip_path

    elif make_plots:
        _plot_ipinn_histories(ipinn_results, None, show=show_plots)

    return ipinn_results, ipinn_init_C, ipinn_init_Calt, df_summary, paths


# ============================================================================
# Section 16. Bootstrap C/C-alt with Weak-SINDy and/or iPINN initialization
# ============================================================================
@dataclass
class BootstrapCCaltSettings:
    """Configuration for C/C-alt rollout-parameter bootstrap."""
    mode: str = "smoke"
    B: int = 10
    maxiter: int = 100
    n_sub_fit: int = 100
    max_points: int = 50_000


def make_bootstrap_ccalt_settings(mode: str = "smoke", **overrides) -> BootstrapCCaltSettings:
    """Create bootstrap settings from smoke/production/final presets."""
    presets = {
        "smoke": dict(B=10, maxiter=100, n_sub_fit=100, max_points=50_000),
        "production": dict(B=50, maxiter=150, n_sub_fit=100, max_points=80_000),
        "final": dict(B=100, maxiter=200, n_sub_fit=150, max_points=100_000),
    }
    if mode not in presets:
        raise ValueError(f"Unknown bootstrap mode {mode!r}; choose from {sorted(presets)}")
    data = dict(mode=mode)
    data.update(presets[mode])
    data.update(overrides)
    return BootstrapCCaltSettings(**data)


@contextmanager
def timed(label: str):
    """Print wall-clock time for a block."""
    t0 = time.perf_counter()
    print(f"\n{'=' * 80}\n>>> START: {label}\n{'=' * 80}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        h, rem = divmod(dt, 3600)
        m, s = divmod(rem, 60)
        print(f"\n>>> END  : {label}")
        print(f">>> Wall time: {int(h):d}h {int(m):02d}m {s:05.2f}s ({dt:.1f} s total)")


def format_ci(p: Sequence[float]) -> str:
    """Format a [2.5, 50, 97.5] percentile triplet."""
    return f"[{p[0]:+.4g}, {p[1]:+.4g}, {p[2]:+.4g}]"


def validate_init_dict(init: Dict[str, float],
                       required_keys: Sequence[str],
                       label: str) -> Dict[str, float]:
    """Check that an initialization dictionary contains required keys."""
    missing = [k for k in required_keys if k not in init]
    if missing:
        raise ValueError(
            f"{label} is missing required keys {missing}. Available keys: {list(init.keys())}"
        )
    return {str(k): float(v) for k, v in init.items()}


def weak_sindy_initializations_C_Calt(all_results: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extract bootstrap-ready Weak-SINDy initializations for C and C-alt."""
    if LIBRARY_C_NAME not in all_results:
        raise RuntimeError(f"Missing library in all_results: {LIBRARY_C_NAME}")
    if LIBRARY_C_ALT_NAME not in all_results:
        raise RuntimeError(f"Missing library in all_results: {LIBRARY_C_ALT_NAME}")

    system_C = all_results[LIBRARY_C_NAME]["system"]
    fit_C = all_results[LIBRARY_C_NAME]["fit"]
    weak_init_C = {
        "grad2": get_fit_coef_by_feature(system_C, fit_C, "|grad u|^2", default=0.8),
        "lap": get_fit_coef_by_feature(system_C, fit_C, "Delta u", default=-0.1),
        "adv": 1.0,
    }

    system_Calt = all_results[LIBRARY_C_ALT_NAME]["system"]
    fit_Calt = all_results[LIBRARY_C_ALT_NAME]["fit"]
    weak_init_Calt = {
        "u_grad2": get_fit_coef_by_feature(system_Calt, fit_Calt, "u|grad u|^2", default=2.5),
        "lap": get_fit_coef_by_feature(system_Calt, fit_Calt, "Delta u", default=0.0),
        "adv": 1.0,
    }
    return weak_init_C, weak_init_Calt


def build_bootstrap_models_C_Calt(
    all_results: Dict[str, Dict[str, object]],
    *,
    init_sources: Sequence[str] = ("weak_sindy", "ipinn"),
    ipinn_init_C: Optional[Dict[str, float]] = None,
    ipinn_init_Calt: Optional[Dict[str, float]] = None,
    fixed: Optional[Dict[str, float]] = None,
    positive_log: Sequence[str] = (),
) -> Dict[str, Dict[str, object]]:
    """Build C/C-alt bootstrap model specifications."""
    fixed = dict({"adv": 1.0} if fixed is None else fixed)
    weak_init_C, weak_init_Calt = weak_sindy_initializations_C_Calt(all_results)

    bootstrap_models: Dict[str, Dict[str, object]] = {}
    for source in init_sources:
        if source == "weak_sindy":
            init_C_current = weak_init_C
            init_Calt_current = weak_init_Calt
            source_label = "Weak-SINDy"
        elif source == "ipinn":
            if ipinn_init_C is None or ipinn_init_Calt is None:
                raise RuntimeError(
                    "init_sources includes 'ipinn', but ipinn_init_C and/or ipinn_init_Calt "
                    "were not supplied. Run run_ipinn_refinement_C_Calt first, or use init_sources=('weak_sindy',)."
                )
            init_C_current = validate_init_dict(ipinn_init_C, ["grad2", "lap", "adv"], "ipinn_init_C")
            init_Calt_current = validate_init_dict(ipinn_init_Calt, ["u_grad2", "lap", "adv"], "ipinn_init_Calt")
            source_label = "iPINN"
        else:
            raise ValueError("init_sources entries must be 'weak_sindy' or 'ipinn'")

        bootstrap_models[f"C_unconstrained_lap__init_{source}"] = {
            "library": LIBRARY_C_NAME,
            "model_family": "C",
            "init_source": source,
            "init_source_label": source_label,
            "description": (
                f"Measured-advection model initialized from {source_label}: "
                "u_t + v(t)·grad u = a|grad u|^2 + beta Delta u; beta unconstrained"
            ),
            "init": dict(init_C_current),
            "param_names": ["grad2", "lap"],
            "fixed": dict(fixed),
            "positive_log": list(positive_log),
        }
        bootstrap_models[f"Calt_unconstrained_lap__init_{source}"] = {
            "library": LIBRARY_C_ALT_NAME,
            "model_family": "C-alt",
            "init_source": source,
            "init_source_label": source_label,
            "description": (
                f"Measured-advection model initialized from {source_label}: "
                "u_t + v(t)·grad u = a u|grad u|^2 + beta Delta u; beta unconstrained"
            ),
            "init": dict(init_Calt_current),
            "param_names": ["u_grad2", "lap"],
            "fixed": dict(fixed),
            "positive_log": list(positive_log),
        }
    return bootstrap_models


def median_coefs(df: pd.DataFrame,
                 param_names: Sequence[str],
                 fixed: Dict[str, float],
                 *,
                 min_success_fraction: float = 0.50,
                 verbose: bool = True) -> Dict[str, float]:
    """
    Compute median bootstrap coefficients with a robust fallback.

    Uses converged replicates when enough converged; otherwise uses the best-MSE
    half of all replicates so the median is not based on too few successful runs.
    """
    out = dict(fixed)
    n_total = len(df)
    n_success = int(df["success"].sum()) if "success" in df.columns else 0
    success_fraction = n_success / max(n_total, 1)

    if n_success > 0 and success_fraction >= min_success_fraction:
        ok = df[df["success"]].copy()
        source = "converged replicates"
    else:
        if "mse" in df.columns:
            ok = df.sort_values("mse").head(max(3, n_total // 2)).copy()
            source = "best-MSE half of all replicates"
        else:
            ok = df.copy()
            source = "all replicates"

    if verbose:
        print(f"\nMedian coefficient source: {source}")
        print(f"Successful replicates    : {n_success}/{n_total}")

    for n in param_names:
        out[n] = float(ok[n].median())
    return out


def print_bootstrap_convergence_report(df_boot: pd.DataFrame,
                                       summ_boot: Dict[str, object],
                                       model_name: str) -> None:
    """Print convergence diagnostics for one bootstrap model."""
    n_reps = int(summ_boot.get("n_replicates", len(df_boot)))
    n_conv_default = int(df_boot["success"].sum()) if "success" in df_boot.columns else 0
    n_conv = int(summ_boot.get("n_converged", n_conv_default))
    conv_rate = n_conv / max(n_reps, 1)

    print("\n" + "=" * 80)
    print(f"Bootstrap convergence report: {model_name}")
    print("=" * 80)
    print(f"Converged replicates : {n_conv} / {n_reps}")
    print(f"Convergence rate     : {100 * conv_rate:.1f}%")
    if "mse" in df_boot.columns:
        print(f"Median MSE, all reps       : {df_boot['mse'].median():.6e}")
        if "success" in df_boot.columns and df_boot["success"].any():
            print(f"Median MSE, converged reps : {df_boot[df_boot['success']]['mse'].median():.6e}")
    if conv_rate < 0.50:
        print(
            "\nWARNING: Fewer than 50% of bootstrap replicates converged. "
            "Treat confidence intervals as preliminary. Consider increasing maxiter."
        )
    elif conv_rate < 0.80:
        print("\nNOTE: Convergence is moderate. Interpret bootstrap intervals cautiously.")
    else:
        print("\nGood: bootstrap convergence rate is reasonably high.")


def validate_model_from_coefs(coefs: Dict[str, float],
                              U_val: np.ndarray,
                              t_val: np.ndarray,
                              vx_val: np.ndarray,
                              vy_val: np.ndarray,
                              grid: Grid,
                              *,
                              eps_visc: float = 0.01,
                              safety: float = 0.25,
                              max_substeps: int = 2000,
                              clip: Optional[Tuple[float, float]] = (0.0, 1.0),
                              device: Optional[str] = None,
                              label: str = "model") -> Tuple[np.ndarray, float, np.ndarray]:
    """Validate one coefficient dictionary on a held-out window."""
    with timed(f"Validation rollout: {label}"):
        U_pred = rollout(
            U_val[:, :, 0],
            t_val,
            vx_val,
            vy_val,
            dx=grid.dx,
            dy=grid.dy,
            coefs=coefs,
            eps_visc=eps_visc,
            safety=safety,
            max_substeps=max_substeps,
            clip=clip,
            device=device,
        )
    rmse_percent = 100.0 * relative_rmse(U_val, U_pred)
    mse_t = mse_over_time(U_val, U_pred)
    return U_pred, float(rmse_percent), mse_t


def plot_bootstrap_parameter_histograms(df_boot: pd.DataFrame,
                                        param_names: Sequence[str],
                                        title: str,
                                        save_path: Optional[os.PathLike] = None,
                                        *,
                                        show: bool = True) -> None:
    """Plot histograms of bootstrap parameter values."""
    ok = df_boot[df_boot["success"]].copy() if "success" in df_boot.columns and df_boot["success"].any() else df_boot.copy()
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, param_names):
        vals = ok[p].dropna().values
        ax.hist(vals, bins=20)
        if len(vals) > 0:
            ax.axvline(np.median(vals), linestyle="--", linewidth=2, label="median")
        ax.set_title(p)
        ax.set_xlabel("coefficient")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bootstrap_mse_vs_parameters(df_boot: pd.DataFrame,
                                     param_names: Sequence[str],
                                     title: str,
                                     save_path: Optional[os.PathLike] = None,
                                     *,
                                     show: bool = True) -> None:
    """Plot bootstrap parameter values against training MSE."""
    if "mse" not in df_boot.columns:
        print("No mse column found. Skipping MSE-vs-parameter plot.")
        return
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    success = df_boot["success"].values.astype(bool) if "success" in df_boot.columns else np.ones(len(df_boot), dtype=bool)
    for ax, p in zip(axes, param_names):
        ax.scatter(df_boot.loc[~success, p], df_boot.loc[~success, "mse"], alpha=0.5, label="not converged", marker="x")
        ax.scatter(df_boot.loc[success, p], df_boot.loc[success, "mse"], alpha=0.7, label="converged")
        ax.set_yscale("log")
        ax.set_xlabel(p)
        ax.set_ylabel("training MSE")
        ax.set_title(f"{p} vs training MSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_validation_mse_series(t_val: np.ndarray,
                               mse_t: np.ndarray,
                               title: str,
                               save_path: Optional[os.PathLike] = None,
                               *,
                               show: bool = True) -> None:
    """Plot validation MSE over time."""
    fig = plt.figure(figsize=(8, 4))
    plt.semilogy(t_val, mse_t)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _write_bootstrap_metadata(path: Path,
                              *,
                              settings: BootstrapCCaltSettings,
                              init_sources: Sequence[str],
                              eps_visc: float,
                              seed: int,
                              clip,
                              fixed: Dict[str, float],
                              positive_log: Sequence[str],
                              device: Optional[str],
                              U_train: np.ndarray,
                              U_val: np.ndarray,
                              grid: Grid,
                              bootstrap_models: Dict[str, Dict[str, object]],
                              weak_init_C: Dict[str, float],
                              weak_init_Calt: Dict[str, float],
                              ipinn_init_C: Optional[Dict[str, float]],
                              ipinn_init_Calt: Optional[Dict[str, float]],
                              metadata: Optional[Dict[str, object]] = None) -> Path:
    metadata_path = path / "bootstrap_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("Bootstrap rollout-parameter fitting: C vs C-alt\n")
        f.write("Initialization sources: Weak-SINDy and/or iPINN\n")
        f.write("Laplacian coefficient unconstrained unless positive_log includes it\n")
        f.write("=" * 80 + "\n")
        f.write(f"mode = {settings.mode}\n")
        f.write(f"init_sources = {list(init_sources)}\n")
        f.write(f"B = {settings.B}\n")
        f.write(f"n_sub_fit = {settings.n_sub_fit}\n")
        f.write(f"maxiter = {settings.maxiter}\n")
        f.write(f"max_points = {settings.max_points}\n")
        f.write(f"eps_visc = {eps_visc}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"clip = {clip}\n")
        f.write(f"fixed = {fixed}\n")
        f.write(f"positive_log = {list(positive_log)}\n")
        f.write(f"device = {device}\n")
        f.write(f"U_train.shape = {U_train.shape}\n")
        f.write(f"U_val.shape = {U_val.shape}\n")
        f.write(f"grid.dx = {grid.dx}\n")
        f.write(f"grid.dy = {grid.dy}\n")
        f.write(f"grid.dt = {grid.dt}\n")
        f.write("\nInterpretation:\n")
        f.write("Weak-SINDy is used for structural discovery.\n")
        f.write("iPINN can provide refined coefficient initialization.\n")
        f.write("Bootstrap rollout fitting assesses predictive calibration and stability.\n")
        f.write("Measured center-of-mass advection is imposed through adv = 1.0 by default.\n")
        f.write("\nInitializations:\n")
        f.write(f"weak_init_C = {weak_init_C}\n")
        f.write(f"weak_init_Calt = {weak_init_Calt}\n")
        if ipinn_init_C is not None:
            f.write(f"ipinn_init_C = {ipinn_init_C}\n")
        if ipinn_init_Calt is not None:
            f.write(f"ipinn_init_Calt = {ipinn_init_Calt}\n")
        if metadata:
            f.write("\nExtra metadata:\n")
            for key, value in metadata.items():
                f.write(f"{key} = {value}\n")
        f.write("\nModels:\n")
        for model_name, spec in bootstrap_models.items():
            f.write(f"\n{model_name}\n")
            f.write(f"  library: {spec['library']}\n")
            f.write(f"  model_family: {spec['model_family']}\n")
            f.write(f"  init_source: {spec['init_source_label']}\n")
            f.write(f"  description: {spec['description']}\n")
            f.write(f"  init: {spec['init']}\n")
            f.write(f"  param_names: {spec['param_names']}\n")
            f.write(f"  fixed: {spec['fixed']}\n")
            f.write(f"  positive_log: {spec['positive_log']}\n")
    print("Saved:", metadata_path)
    return metadata_path


def run_bootstrap_C_Calt_by_init_source(
    all_results: Dict[str, Dict[str, object]],
    U: np.ndarray,
    grid: Grid,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    init_sources: Sequence[str] = ("weak_sindy", "ipinn"),
    ipinn_init_C: Optional[Dict[str, float]] = None,
    ipinn_init_Calt: Optional[Dict[str, float]] = None,
    mode: str = "smoke",
    settings: Optional[BootstrapCCaltSettings] = None,
    device: Optional[str] = None,
    eps_visc: float = 0.01,
    seed: int = 2026,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    fixed: Optional[Dict[str, float]] = None,
    positive_log: Sequence[str] = (),
    train_end_frac: float = 0.60,
    test_end_frac: float = 0.80,
    output_dir: Optional[os.PathLike] = None,
    save_results: bool = True,
    make_plots: bool = True,
    show_plots: bool = True,
    make_zip: bool = True,
    zip_path: Optional[os.PathLike] = None,
    download_zip: bool = True,
    min_success_fraction: float = 0.50,
    validation_safety: float = 0.25,
    validation_max_substeps: int = 2000,
    verbose: bool = True,

    # ------------------------------------------------------------
    # Inner Nelder-Mead progress controls
    # ------------------------------------------------------------
    nm_verbose: bool = False,
    nm_print_every: int = 10,

    metadata: Optional[Dict[str, object]] = None,
    **settings_overrides,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    """
    Bootstrap C and C-alt rollout parameters using Weak-SINDy and/or iPINN initialization.

    Parameters
    ----------
    all_results
        Dictionary returned by run_candidate_libraries.
    U
        Data array with shape (ny, nx, nt).
    grid
        Grid object.
    vx, vy
        Measured advection velocity time series.
    init_sources
        Any subset/order of ("weak_sindy", "ipinn"). If "ipinn" is included,
        pass ipinn_init_C and ipinn_init_Calt from run_ipinn_refinement_C_Calt.
    mode
        Preset mode: "smoke", "production", or "final", unless settings is passed.
    settings
        Optional BootstrapCCaltSettings object. If None, constructed from mode.
    nm_verbose
        If True, print inner Nelder-Mead progress inside each bootstrap replicate.
    nm_print_every
        Print Nelder-Mead status every this many iterations.
    download_zip
        Defaults to True for Colab convenience. Set False to save without download.

    Returns
    -------
    all_boot_results, df_boot_summary, df_boot_intervals, paths
    """

    # ============================================================
    # Settings and defaults
    # ============================================================

    settings = (
        settings
        if settings is not None
        else make_bootstrap_ccalt_settings(mode, **settings_overrides)
    )

    fixed = dict({"adv": 1.0} if fixed is None else fixed)

    # ============================================================
    # Data split
    # ============================================================

    split = validation_split(
        U,
        grid,
        vx,
        vy,
        train_end_frac=train_end_frac,
        test_end_frac=test_end_frac,
    )

    U_train = U[:, :, split["slice_train"]]
    t_train = grid.t[split["slice_train"]]
    vx_train = vx[split["slice_train"]]
    vy_train = vy[split["slice_train"]]

    U_val = split["U_val"]
    t_val = split["t_val"]
    vx_val = split["vx_val"]
    vy_val = split["vy_val"]

    # ============================================================
    # Initializations and model specs
    # ============================================================

    weak_init_C, weak_init_Calt = weak_sindy_initializations_C_Calt(all_results)

    bootstrap_models = build_bootstrap_models_C_Calt(
        all_results,
        init_sources=init_sources,
        ipinn_init_C=ipinn_init_C,
        ipinn_init_Calt=ipinn_init_Calt,
        fixed=fixed,
        positive_log=positive_log,
    )

    # ============================================================
    # Output paths
    # ============================================================

    if save_results:
        if output_dir is None:
            init_label = "_".join(init_sources)
            output_dir = Path(
                f"/content/bootstrap_C_Calt_unconstrained_lap_{init_label}_"
                f"{settings.mode}_B{settings.B}_iter{settings.maxiter}"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    else:
        output_path = None

    paths: Dict[str, Path] = {}

    if output_path is not None:
        paths["output_dir"] = output_path

    # ============================================================
    # Print configuration
    # ============================================================

    if verbose:
        print("Data split")
        print("----------")
        print(f"Total frames       : {grid.nt}")
        print(f"Training frames    : [0, {split['i_train_end']})")
        print(f"Validation frames  : [{split['i_test_end']}, {grid.nt})")
        print(f"U_train shape      : {U_train.shape}")
        print(f"U_val shape        : {U_val.shape}")

        print("\nBootstrap configuration")
        print("-----------------------")
        print(f"settings       = {settings}")
        print(f"init_sources   = {list(init_sources)}")
        print(f"eps_visc       = {eps_visc}")
        print(f"seed           = {seed}")
        print(f"clip           = {clip}")
        print(f"fixed          = {fixed}")
        print(f"positive_log   = {list(positive_log)}")
        print(f"device         = {device}")
        print(f"nm_verbose     = {nm_verbose}")
        print(f"nm_print_every = {nm_print_every}")

        if output_path is not None:
            print(f"output_dir     = {output_path}")

        print("\nBootstrap models to run")
        print("-----------------------")

        for model_name, spec in bootstrap_models.items():
            print(f"\n{model_name}")
            print(f"  library       : {spec['library']}")
            print(f"  model_family  : {spec['model_family']}")
            print(f"  init_source   : {spec['init_source_label']}")
            print(f"  init          : {spec['init']}")
            print(f"  free params   : {spec['param_names']}")
            print(f"  fixed         : {spec['fixed']}")
            print(f"  positive_log  : {spec['positive_log']}")

    # ============================================================
    # Run bootstrap for each model
    # ============================================================

    all_boot_results: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    interval_rows: List[Dict[str, object]] = []

    for model_name, spec in bootstrap_models.items():

        print("\n" + "#" * 100)
        print(f"# BOOTSTRAP MODEL: {model_name}")
        print("#" * 100)
        print(f"Description : {spec['description']}")
        print(f"Init source : {spec['init_source_label']}")
        print(f"Init        : {spec['init']}")
        print(f"Free params : {spec['param_names']}")
        print(f"Fixed       : {spec['fixed']}")
        print(f"Positive-log constraints: {spec['positive_log']}")

        with timed(
            f"Bootstrap {model_name} "
            f"(B={settings.B}, maxiter={settings.maxiter})"
        ):
            df_boot, summ_boot = bootstrap_rollout_parameters(
                U_train,
                t_train,
                vx_train,
                vy_train,
                grid,
                init=spec["init"],
                param_names=spec["param_names"],
                positive_log=spec["positive_log"],
                fixed=spec["fixed"],
                B=settings.B,
                block_len=None,
                mask=None,
                n_sub=settings.n_sub_fit,
                eps_visc=eps_visc,
                max_points=settings.max_points,
                maxiter=settings.maxiter,
                clip=clip,
                device=device,
                seed=seed,
                verbose=verbose,

                # Inner Nelder-Mead progress
                nm_verbose=nm_verbose,
                nm_print_every=nm_print_every,
            )

        print_bootstrap_convergence_report(df_boot, summ_boot, model_name)

        med_coefs = median_coefs(
            df_boot,
            spec["param_names"],
            spec["fixed"],
            min_success_fraction=min_success_fraction,
            verbose=verbose,
        )

        if verbose:
            print("\nMedian bootstrap coefficients:")
            for k, v in med_coefs.items():
                print(f"  {k:12s} = {v:+.6g}")

        # ------------------------------------------------------------
        # Validation rollout using median bootstrap coefficients
        # ------------------------------------------------------------

        U_pred, rmse_val, mse_t = validate_model_from_coefs(
            med_coefs,
            U_val,
            t_val,
            vx_val,
            vy_val,
            grid,
            eps_visc=eps_visc,
            safety=validation_safety,
            max_substeps=validation_max_substeps,
            clip=clip,
            device=device,
            label=model_name,
        )

        print(f"\nValidation relative RMSE for {model_name}: {rmse_val:.2f}%")

        # ------------------------------------------------------------
        # Summary stats
        # ------------------------------------------------------------

        train_mse_median = (
            float(df_boot[df_boot["success"]]["mse"].median())
            if "success" in df_boot.columns and df_boot["success"].any()
            else float(df_boot["mse"].median())
        )

        n_conv_default = (
            int(df_boot["success"].sum())
            if "success" in df_boot.columns
            else 0
        )

        n_converged = int(summ_boot.get("n_converged", n_conv_default))
        n_replicates = int(summ_boot.get("n_replicates", len(df_boot)))
        convergence_rate_percent = 100.0 * n_converged / max(n_replicates, 1)

        all_boot_results[model_name] = {
            "df": df_boot,
            "summary": summ_boot,
            "median_coefs": med_coefs,
            "U_pred_val": U_pred,
            "mse_val_t": mse_t,
            "rmse_val_percent": rmse_val,
            "train_mse_median": train_mse_median,
            "n_converged": n_converged,
            "n_replicates": n_replicates,
            "convergence_rate_percent": convergence_rate_percent,
            "spec": spec,
        }

        summary_rows.append({
            "model": model_name,
            "model_family": spec["model_family"],
            "library": spec["library"],
            "description": spec["description"],
            "init_source": spec["init_source"],
            "init_source_label": spec["init_source_label"],
            "bootstrap_mode": settings.mode,
            "free_params": ", ".join(spec["param_names"]),
            "positive_log": (
                ", ".join(spec["positive_log"])
                if spec["positive_log"]
                else "None"
            ),
            "B": settings.B,
            "maxiter": settings.maxiter,
            "n_sub_fit": settings.n_sub_fit,
            "max_points": settings.max_points,
            "n_converged": n_converged,
            "n_replicates": n_replicates,
            "convergence_rate_percent": convergence_rate_percent,
            "train_mse_median": train_mse_median,
            "validation_rmse_percent": rmse_val,
            "init_coefs": ", ".join(
                f"{k}: {v:+.6e}" for k, v in spec["init"].items()
            ),
            "median_coefs": ", ".join(
                f"{k}: {v:+.6e}" for k, v in med_coefs.items()
            ),
        })

        # ------------------------------------------------------------
        # Parameter interval rows
        # ------------------------------------------------------------

        for p in spec["param_names"]:
            if "percentiles" in summ_boot and p in summ_boot["percentiles"]:
                ci = summ_boot["percentiles"][p]

                interval_rows.append({
                    "model": model_name,
                    "model_family": spec["model_family"],
                    "init_source": spec["init_source"],
                    "init_source_label": spec["init_source_label"],
                    "parameter": p,
                    "init_value": spec["init"].get(p, np.nan),
                    "p2_5": ci[0],
                    "median": ci[1],
                    "p97_5": ci[2],
                    "mean": summ_boot.get("mean", {}).get(p, np.nan),
                    "std": summ_boot.get("std", {}).get(p, np.nan),
                    "n_converged": n_converged,
                    "n_replicates": n_replicates,
                    "convergence_rate_percent": convergence_rate_percent,
                })

        # ------------------------------------------------------------
        # Plots for this model
        # ------------------------------------------------------------

        if make_plots:
            safe_name = safe_filename(model_name)

            hist_path = (
                output_path / f"bootstrap_histograms_{safe_name}.png"
                if output_path is not None and save_results
                else None
            )

            mse_param_path = (
                output_path / f"bootstrap_mse_vs_parameters_{safe_name}.png"
                if output_path is not None and save_results
                else None
            )

            mse_path = (
                output_path / f"validation_mse_{safe_name}.png"
                if output_path is not None and save_results
                else None
            )

            plot_bootstrap_parameter_histograms(
                df_boot,
                spec["param_names"],
                title=f"Bootstrap parameter distributions\n{model_name}",
                save_path=hist_path,
                show=show_plots,
            )

            plot_bootstrap_mse_vs_parameters(
                df_boot,
                spec["param_names"],
                title=f"Bootstrap MSE vs parameters\n{model_name}",
                save_path=mse_param_path,
                show=show_plots,
            )

            plot_validation_mse_series(
                t_val,
                mse_t,
                title=(
                    f"Validation MSE over time\n"
                    f"{model_name}, RMSE={rmse_val:.2f}%"
                ),
                save_path=mse_path,
                show=show_plots,
            )

        # ------------------------------------------------------------
        # Save individual model outputs
        # ------------------------------------------------------------

        if save_results and output_path is not None:
            safe_name = safe_filename(model_name)

            df_path = output_path / f"bootstrap_runs_{safe_name}.csv"
            df_boot.to_csv(df_path, index=False)

            param_rows = []

            for p in spec["param_names"]:
                if "percentiles" in summ_boot and p in summ_boot["percentiles"]:
                    ci = summ_boot["percentiles"][p]

                    param_rows.append({
                        "model": model_name,
                        "model_family": spec["model_family"],
                        "init_source": spec["init_source"],
                        "init_source_label": spec["init_source_label"],
                        "parameter": p,
                        "init_value": spec["init"].get(p, np.nan),
                        "mean": summ_boot.get("mean", {}).get(p, np.nan),
                        "std": summ_boot.get("std", {}).get(p, np.nan),
                        "p2_5": ci[0],
                        "median": ci[1],
                        "p97_5": ci[2],
                        "n_converged": n_converged,
                        "n_replicates": n_replicates,
                        "convergence_rate_percent": convergence_rate_percent,
                    })

            param_summary_path = output_path / f"bootstrap_parameter_summary_{safe_name}.csv"
            pd.DataFrame(param_rows).to_csv(param_summary_path, index=False)

            val_mse_path = output_path / f"validation_mse_{safe_name}.csv"
            pd.DataFrame({
                "t": t_val,
                "mse": mse_t,
            }).to_csv(val_mse_path, index=False)

            arr_path = output_path / f"validation_arrays_{safe_name}.npz"

            np.savez_compressed(
                arr_path,
                U_val=U_val,
                U_pred=U_pred,
                t_val=t_val,
                vx_val=vx_val,
                vy_val=vy_val,
                mse_val_t=mse_t,
                init_source=spec["init_source"],
                init_coefs=np.array(list(spec["init"].items()), dtype=object),
                median_coefs=np.array(list(med_coefs.items()), dtype=object),
            )

            print("Saved:", df_path)
            print("Saved:", param_summary_path)
            print("Saved:", val_mse_path)
            print("Saved:", arr_path)

    # ============================================================
    # Combined summaries
    # ============================================================

    df_boot_summary = pd.DataFrame(summary_rows).sort_values(
        ["model_family", "validation_rmse_percent", "init_source"]
    ).reset_index(drop=True)

    df_boot_intervals = pd.DataFrame(interval_rows)

    print("\n" + "=" * 100)
    print("BOOTSTRAP SUMMARY: C vs C-alt, weak-SINDy vs iPINN initialization")
    print("=" * 100)
    print(df_boot_summary.to_string(index=False))

    if save_results and output_path is not None:
        summary_path = output_path / "bootstrap_summary_C_vs_Calt_by_init_source.csv"
        intervals_path = output_path / "bootstrap_parameter_intervals_C_vs_Calt_by_init_source.csv"

        df_boot_summary.to_csv(summary_path, index=False)
        df_boot_intervals.to_csv(intervals_path, index=False)

        print("Saved:", summary_path)
        print("Saved:", intervals_path)

    # ============================================================
    # Combined plots
    # ============================================================

    if make_plots:

        # --------------------------------------------------------
        # Validation RMSE comparison
        # --------------------------------------------------------

        fig = plt.figure(figsize=(10, 5))

        plot_df = df_boot_summary.sort_values("validation_rmse_percent")

        plt.barh(
            plot_df["model"],
            plot_df["validation_rmse_percent"],
        )

        plt.xlabel("Validation rollout relative RMSE (%)")
        plt.ylabel("model")
        plt.title("Bootstrap-refitted validation RMSE: model and initialization source")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_results and output_path is not None:
            rmse_plot_path = output_path / "bootstrap_validation_rmse_by_init_source.png"
            fig.savefig(rmse_plot_path, dpi=1200, bbox_inches="tight")
            print("Saved:", rmse_plot_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # --------------------------------------------------------
        # Validation MSE over time
        # --------------------------------------------------------

        fig = plt.figure(figsize=(9, 5))

        for model_name, res in all_boot_results.items():
            plt.semilogy(
                t_val,
                res["mse_val_t"],
                label=(
                    f"{res['spec']['model_family']} | "
                    f"{res['spec']['init_source_label']} | "
                    f"RMSE={res['rmse_val_percent']:.2f}%"
                ),
            )

        plt.xlabel("t")
        plt.ylabel("MSE")
        plt.title("Validation MSE over time: C vs C-alt, weak-SINDy vs iPINN init")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        if save_results and output_path is not None:
            mse_all_path = output_path / "bootstrap_validation_mse_by_init_source.png"
            fig.savefig(mse_all_path, dpi=1200, bbox_inches="tight")
            print("Saved:", mse_all_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # --------------------------------------------------------
        # Convergence rate comparison
        # --------------------------------------------------------

        fig = plt.figure(figsize=(10, 5))

        conv_df = df_boot_summary.sort_values("convergence_rate_percent")

        plt.barh(
            conv_df["model"],
            conv_df["convergence_rate_percent"],
        )

        plt.xlabel("Bootstrap convergence rate (%)")
        plt.ylabel("model")
        plt.title("Bootstrap convergence rate by initialization source")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_results and output_path is not None:
            conv_plot_path = output_path / "bootstrap_convergence_rate_by_init_source.png"
            fig.savefig(conv_plot_path, dpi=1200, bbox_inches="tight")
            print("Saved:", conv_plot_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # ============================================================
    # Metadata and zip
    # ============================================================

    if save_results and output_path is not None:
        _write_bootstrap_metadata(
            output_path,
            settings=settings,
            init_sources=init_sources,
            eps_visc=eps_visc,
            seed=seed,
            clip=clip,
            fixed=fixed,
            positive_log=positive_log,
            device=device,
            U_train=U_train,
            U_val=U_val,
            grid=grid,
            bootstrap_models=bootstrap_models,
            weak_init_C=weak_init_C,
            weak_init_Calt=weak_init_Calt,
            ipinn_init_C=ipinn_init_C,
            ipinn_init_Calt=ipinn_init_Calt,
            metadata=metadata,
        )

        if make_zip:
            if zip_path is None:
                init_label = "_".join(init_sources)
                zip_path = Path(
                    f"/content/bootstrap_C_Calt_unconstrained_lap_{init_label}_"
                    f"{settings.mode}_B{settings.B}_iter{settings.maxiter}.zip"
                )

            zip_path = Path(zip_path)

            _zip_and_download_dir(
                output_path,
                zip_path,
                download_zip=download_zip,
            )

            paths["zip_path"] = zip_path

    return all_boot_results, df_boot_summary, df_boot_intervals, paths

def make_fixed_mse_sample_indices(
    U_shape,
    mask=None,
    max_points=80_000,
    seed=0,
):
    """
    Create fixed sampled indices for MSE evaluation.

    Returns arrays iy, ix, it such that the MSE is always evaluated
    on the same sampled space-time points.
    """
    rng = np.random.default_rng(seed)

    ny, nx, nt = U_shape

    if mask is not None:
        valid = np.asarray(mask, dtype=bool)

        if valid.shape != U_shape:
            raise ValueError(
                f"mask shape {valid.shape} does not match U shape {U_shape}"
            )

        iy_all, ix_all, it_all = np.where(valid)

        n_available = len(iy_all)

        if n_available == 0:
            raise ValueError("Mask contains no valid points.")

        n_sample = min(int(max_points), n_available)

        choice = rng.choice(n_available, size=n_sample, replace=False)

        iy = iy_all[choice]
        ix = ix_all[choice]
        it = it_all[choice]

    else:
        n_total = ny * nx * nt
        n_sample = min(int(max_points), n_total)

        flat = rng.choice(n_total, size=n_sample, replace=False)

        iy, ix, it = np.unravel_index(flat, U_shape)

    return iy, ix, it


def fixed_sample_mse(U_true, U_pred, sample_indices):
    """
    Compute MSE on fixed sampled space-time indices.
    """
    iy, ix, it = sample_indices

    diff = U_true[iy, ix, it] - U_pred[iy, ix, it]

    return float(np.mean(diff ** 2))