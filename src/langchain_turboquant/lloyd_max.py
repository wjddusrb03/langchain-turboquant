"""Lloyd-Max optimal scalar quantizer for the unit hypersphere distribution.

TurboQuant quantizes each coordinate of a randomly-rotated vector.  After
rotation by a random orthogonal matrix, each coordinate follows a distribution
derived from the uniform measure on the unit hypersphere.  The marginal PDF is:

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

for x in [-1, 1], where d is the vector dimension.

This module pre-computes the Lloyd-Max optimal codebook (centroids and
decision boundaries) for that distribution at a given bit-width, so the
quantizer can simply look up the nearest centroid at runtime.

Reference: TurboQuant (ICLR 2026), Algorithm 1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn, gammaln as lgamma_fn
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Marginal PDF of a single coordinate on the unit hypersphere
# ---------------------------------------------------------------------------

def _hypersphere_pdf(x: float, dim: int) -> float:
    """PDF of a single coordinate of a point drawn uniformly on S^{d-1}."""
    if dim <= 2:
        # d=2: uniform on circle, PDF = 1/(pi*sqrt(1-x^2))
        denom = np.sqrt(1.0 - np.clip(x * x, 0.0, 1.0 - 1e-15))
        return 1.0 / (np.pi * denom) if denom > 1e-15 else 0.0
    # Use log-gamma for numerical stability at high dimensions
    log_coeff = (
        lgamma_fn(dim / 2.0)
        - 0.5 * np.log(np.pi)
        - lgamma_fn((dim - 1) / 2.0)
    )
    val = 1.0 - x * x
    if val <= 0.0:
        return 0.0
    return np.exp(log_coeff + ((dim - 3) / 2.0) * np.log(val))


def _hypersphere_pdf_vec(x: np.ndarray, dim: int) -> np.ndarray:
    """Vectorised version of _hypersphere_pdf."""
    if dim <= 2:
        denom = np.sqrt(np.clip(1.0 - x * x, 1e-30, None))
        return 1.0 / (np.pi * denom)
    coeff = gamma_fn(dim / 2.0) / (np.sqrt(np.pi) * gamma_fn((dim - 1) / 2.0))
    val = np.clip(1.0 - x * x, 0.0, None)
    return coeff * np.power(val, (dim - 3) / 2.0)


# ---------------------------------------------------------------------------
# Lloyd-Max algorithm
# ---------------------------------------------------------------------------

def _lloyd_max_iteration(
    boundaries: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One Lloyd-Max iteration: recompute centroids then boundaries.

    Parameters
    ----------
    boundaries : array of shape (n_levels + 1,)
        Decision boundaries including -1 and +1 at the edges.
    dim : int
        Vector dimension (controls the PDF shape).

    Returns
    -------
    centroids : array of shape (n_levels,)
    new_boundaries : array of shape (n_levels + 1,)
    """
    n_levels = len(boundaries) - 1
    centroids = np.empty(n_levels)

    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        # centroid = E[x | lo <= x <= hi]
        num, _ = quad(lambda x: x * _hypersphere_pdf(x, dim), lo, hi)
        den, _ = quad(lambda x: _hypersphere_pdf(x, dim), lo, hi)
        centroids[i] = num / den if abs(den) > 1e-30 else (lo + hi) / 2.0

    # New boundaries = midpoints of adjacent centroids
    new_boundaries = np.empty(n_levels + 1)
    new_boundaries[0] = -1.0
    new_boundaries[-1] = 1.0
    for i in range(1, n_levels):
        new_boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return centroids, new_boundaries


def compute_codebook(bits: int, dim: int, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Compute Lloyd-Max optimal centroids for the hypersphere marginal.

    Parameters
    ----------
    bits : int
        Number of quantization bits per coordinate (e.g. 2 or 3).
    dim : int
        Vector dimension.  Controls the shape of the marginal PDF.
    max_iter : int
        Maximum Lloyd-Max iterations.
    tol : float
        Convergence tolerance on centroid movement.

    Returns
    -------
    centroids : np.ndarray of shape (2**bits,)
        Sorted centroid values in [-1, 1].
    """
    n_levels = 1 << bits  # 2^bits

    # Initialise boundaries uniformly in [-1, 1]
    boundaries = np.linspace(-1.0, 1.0, n_levels + 1)

    centroids = None
    for _ in range(max_iter):
        centroids, new_boundaries = _lloyd_max_iteration(boundaries, dim)
        if np.max(np.abs(new_boundaries - boundaries)) < tol:
            break
        boundaries = new_boundaries

    return np.sort(centroids)


# ---------------------------------------------------------------------------
# Caching helper
# ---------------------------------------------------------------------------

_codebook_cache: dict[tuple[int, int], np.ndarray] = {}


def get_codebook(bits: int, dim: int) -> np.ndarray:
    """Return cached Lloyd-Max codebook (compute once, reuse forever)."""
    key = (bits, dim)
    if key not in _codebook_cache:
        _codebook_cache[key] = compute_codebook(bits, dim)
    return _codebook_cache[key]
