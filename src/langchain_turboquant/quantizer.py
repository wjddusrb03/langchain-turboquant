"""TurboQuant quantizer — PolarQuant (Stage 1) + QJL (Stage 2).

Implements the two-stage compression algorithm from the TurboQuant paper
(ICLR 2026).  All operations are pure NumPy — no GPU required.

Stage 1 (TurboQuant_mse):
    Rotate the vector with a random orthogonal matrix, then scalar-quantize
    each coordinate using a Lloyd-Max codebook.

Stage 2 (QJL residual correction):
    Compute the quantization residual, project it through a random Gaussian
    matrix, and store only the sign bits (1 bit per projection dimension).
    This enables an unbiased inner-product estimator at query time.

Reference: TurboQuant (ICLR 2026), Algorithms 1-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from langchain_turboquant.lloyd_max import get_codebook


# ---------------------------------------------------------------------------
# Compressed representation
# ---------------------------------------------------------------------------

@dataclass
class CompressedVectors:
    """Container for a batch of TurboQuant-compressed vectors.

    Attributes
    ----------
    indices : np.ndarray, uint8, shape (n, dim)
        Lloyd-Max centroid indices (Stage 1).
    qjl_bits : np.ndarray, int8 (+1/-1), shape (n, qjl_dim)
        QJL sign-bit sketches of the residual (Stage 2).
    gammas : np.ndarray, float32, shape (n,)
        L2 norms of the residuals.
    norms : np.ndarray, float32, shape (n,)
        L2 norms of the original vectors (for cosine similarity).
    """
    indices: np.ndarray
    qjl_bits: np.ndarray
    gammas: np.ndarray
    norms: np.ndarray


# ---------------------------------------------------------------------------
# TurboQuantizer
# ---------------------------------------------------------------------------

class TurboQuantizer:
    """Two-stage vector quantizer: PolarQuant + QJL.

    Parameters
    ----------
    dim : int
        Dimensionality of the input vectors.
    bits : int
        Bits per coordinate for Stage 1 (default 3 → 8 centroids).
    qjl_dim : int or None
        Number of QJL projection dimensions.  Defaults to *dim* (full).
    seed : int
        Random seed for reproducible rotation / projection matrices.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.qjl_dim = qjl_dim or dim
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Stage 1: random orthogonal rotation matrix Pi (d x d)
        gauss = rng.randn(dim, dim).astype(np.float32)
        self.rotation, _ = np.linalg.qr(gauss)  # (d, d) orthogonal

        # Stage 1: Lloyd-Max codebook
        self.codebook = get_codebook(bits, dim).astype(np.float32)  # (2^b,)

        # Stage 2: random Gaussian projection matrix S (qjl_dim x d)
        self.projection = (
            rng.randn(self.qjl_dim, dim).astype(np.float32)
            / np.sqrt(self.qjl_dim)
        )

    # ----- Stage 1: quantize / dequantize (MSE-optimal) -----

    def _quantize_mse(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate then scalar-quantize each coordinate.

        Parameters
        ----------
        vectors : (n, dim) float32

        Returns
        -------
        indices : (n, dim) uint8
        """
        rotated = vectors @ self.rotation.T  # (n, d)
        # nearest centroid per coordinate
        # |rotated[:,:,None] - codebook[None,None,:]| → argmin over last axis
        diffs = np.abs(rotated[:, :, np.newaxis] - self.codebook[np.newaxis, np.newaxis, :])
        return diffs.argmin(axis=2).astype(np.uint8)

    def _dequantize_mse(self, indices: np.ndarray) -> np.ndarray:
        """Restore approximate vectors from centroid indices.

        Parameters
        ----------
        indices : (n, dim) uint8

        Returns
        -------
        vectors_hat : (n, dim) float32
        """
        rotated_hat = self.codebook[indices]  # (n, d)
        return rotated_hat @ self.rotation  # inverse rotation (Pi^T = Pi^{-1})

    # ----- Stage 2: QJL residual sketch -----

    def _qjl_sketch(self, residuals: np.ndarray) -> np.ndarray:
        """Project residuals and keep only the sign bits.

        Parameters
        ----------
        residuals : (n, dim) float32

        Returns
        -------
        sign_bits : (n, qjl_dim) int8  (+1 or -1)
        """
        projected = residuals @ self.projection.T  # (n, qjl_dim)
        return np.sign(projected).astype(np.int8)

    # ----- Public API -----

    def quantize(self, vectors: np.ndarray) -> CompressedVectors:
        """Compress a batch of vectors.

        Parameters
        ----------
        vectors : (n, dim) float32

        Returns
        -------
        CompressedVectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        norms = np.linalg.norm(vectors, axis=1)  # (n,)

        # Normalise to unit sphere for quantization
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_vecs = vectors / safe_norms[:, np.newaxis]

        # Stage 1
        indices = self._quantize_mse(unit_vecs)
        unit_hat = self._dequantize_mse(indices)

        # Stage 2
        residuals = unit_vecs - unit_hat
        gammas = np.linalg.norm(residuals, axis=1)
        qjl_bits = self._qjl_sketch(residuals)

        return CompressedVectors(
            indices=indices,
            qjl_bits=qjl_bits,
            gammas=gammas,
            norms=norms,
        )

    def dequantize(self, compressed: CompressedVectors) -> np.ndarray:
        """Reconstruct approximate vectors (for debugging / evaluation).

        Parameters
        ----------
        compressed : CompressedVectors

        Returns
        -------
        vectors_hat : (n, dim) float32
        """
        unit_hat = self._dequantize_mse(compressed.indices)

        # QJL reconstruction: x_qjl = sqrt(pi/2)/qjl_dim * gamma * S^T @ sign_bits
        coeff = np.sqrt(np.pi / 2.0) / self.qjl_dim
        qjl_recon = (
            coeff
            * compressed.gammas[:, np.newaxis]
            * (compressed.qjl_bits.astype(np.float32) @ self.projection)
        )

        unit_recon = unit_hat + qjl_recon
        return unit_recon * compressed.norms[:, np.newaxis]

    def asymmetric_scores(
        self,
        query: np.ndarray,
        compressed: CompressedVectors,
    ) -> np.ndarray:
        """Compute approximate inner products WITHOUT full dequantization.

        This is the key advantage: the query is in full precision, but the
        database vectors stay compressed.  The estimator is *unbiased*.

        Parameters
        ----------
        query : (dim,) float32  — a single query vector.
        compressed : CompressedVectors  — the stored database.

        Returns
        -------
        scores : (n,) float32  — estimated <query, x_i> for each stored vector.
        """
        query = np.asarray(query, dtype=np.float32).ravel()

        # --- Stage 1 contribution ---
        q_rot = self.rotation @ query  # (d,)
        # inner product between rotated query and quantized centroids
        centroids_per_vec = self.codebook[compressed.indices]  # (n, d)
        score_mse = centroids_per_vec @ q_rot  # (n,)
        # Scale by original norm (we quantized unit vectors)
        score_mse = score_mse * compressed.norms

        # --- Stage 2 (QJL) contribution ---
        q_proj = self.projection @ query  # (qjl_dim,)
        coeff = np.sqrt(np.pi / 2.0) / self.qjl_dim
        score_qjl = coeff * compressed.gammas * (
            compressed.qjl_bits.astype(np.float32) @ q_proj
        )
        score_qjl = score_qjl * compressed.norms

        return score_mse + score_qjl

    def cosine_scores(
        self,
        query: np.ndarray,
        compressed: CompressedVectors,
    ) -> np.ndarray:
        """Approximate cosine similarity between query and compressed vectors.

        Parameters
        ----------
        query : (dim,) float32
        compressed : CompressedVectors

        Returns
        -------
        cosine_similarities : (n,) float32
        """
        inner = self.asymmetric_scores(query, compressed)
        query_norm = np.linalg.norm(query)
        safe_denom = np.where(
            compressed.norms * query_norm > 1e-10,
            compressed.norms * query_norm,
            1.0,
        )
        return inner / safe_denom

    # ----- Memory stats -----

    def compressed_bytes_per_vector(self) -> float:
        """Estimate storage cost per vector in bytes."""
        bits_stage1 = self.dim * self.bits
        bits_stage2 = self.qjl_dim * 1  # 1 bit per QJL dimension
        bits_gamma = 32  # float32 for gamma
        bits_norm = 32   # float32 for norm
        total_bits = bits_stage1 + bits_stage2 + bits_gamma + bits_norm
        return total_bits / 8.0

    def original_bytes_per_vector(self) -> float:
        """Storage cost of an uncompressed float32 vector."""
        return self.dim * 4.0  # 4 bytes per float32

    def compression_ratio(self) -> float:
        """Ratio of original to compressed size."""
        return self.original_bytes_per_vector() / self.compressed_bytes_per_vector()
