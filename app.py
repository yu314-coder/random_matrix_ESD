import json
import os
import io
import sys
import base64
import atexit
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, minimize_scalar
from PIL import Image
import sympy as sp
import webview
import threading
import time

# GPU Detection - Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

# Create xp as either cupy or numpy depending on availability
xp = cp if GPU_AVAILABLE else np

def to_gpu(arr):
    """Move numpy array to GPU if available"""
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move array back to CPU (numpy)"""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


# --------- Core math / plotting helpers (pure Python, no C++) --------- #
def solve_cubic_numpy(a, b, c, d):
    """Solve ax^3 + bx^2 + cx + d = 0 using NumPy polynomial solver."""
    epsilon = 1e-12

    # Handle degenerate cases
    if abs(a) < epsilon:
        if abs(b) < epsilon:
            if abs(c) < epsilon:
                return np.array([complex(np.nan), complex(np.nan), complex(np.nan)])
            # Linear equation: cx + d = 0
            return np.array([complex(-d/c), complex(np.inf), complex(np.inf)])
        # Quadratic equation: bx^2 + cx + d = 0
        roots_2d = np.roots([b, c, d])
        return np.concatenate([roots_2d, [complex(np.inf)]])

    # Use numpy.roots for cubic polynomial
    # numpy.roots expects coefficients in descending order: [a, b, c, d]
    roots = np.roots([a, b, c, d])

    # Ensure we have exactly 3 roots (should always be the case for cubic)
    if len(roots) < 3:
        roots = np.concatenate([roots, [complex(0.0)] * (3 - len(roots))])

    return roots


def compute_ImS_vs_Z(a, y, beta, num_points, z_min, z_max, progress_cb=None):
    # Use linear spacing for z values
    z_values = np.linspace(z_min, z_max, num_points)
    ims_values = np.zeros((num_points, 3))
    real_values = np.zeros((num_points, 3))

    for i, z_val in enumerate(z_values):
        coef_a = z_val * a
        coef_b = z_val * (a + 1) + a * (1 - y)
        coef_c = z_val + (a + 1) - y - y * beta * (a - 1)
        coef_d = 1.0
        roots = solve_cubic_numpy(coef_a, coef_b, coef_c, coef_d)
        for j in range(3):
            # Extract imaginary part (keep sign, don't use abs)
            ims_values[i, j] = roots[j].imag
            real_values[i, j] = roots[j].real
        if progress_cb:
            progress_cb(i + 1, num_points)

    return {
        "z_values": z_values.tolist(),
        "ims_values": ims_values.tolist(),
        "real_values": real_values.tolist(),
    }


def compute_cubic_roots(z, beta, a, y):
    a_coef = z * a
    b_coef = z * (a + 1) + a * (1 - y)
    c_coef = z + (a + 1) - y - y * beta * (a - 1)
    d_coef = 1
    return solve_cubic_numpy(a_coef, b_coef, c_coef, d_coef)


def track_roots_consistently(grid_values, all_roots):
    n_points = len(grid_values)
    n_roots = all_roots[0].shape[0]
    tracked_roots = np.zeros((n_points, n_roots), dtype=complex)
    tracked_roots[0] = all_roots[0]
    for i in range(1, n_points):
        prev_roots = tracked_roots[i - 1]
        current_roots = all_roots[i]
        assigned = np.zeros(n_roots, dtype=bool)
        assignments = np.zeros(n_roots, dtype=int)
        for j in range(n_roots):
            distances = np.abs(current_roots - prev_roots[j])
            while True:
                best_idx = np.min(np.where(distances == distances.min()))
                if not assigned[best_idx]:
                    assignments[j] = best_idx
                    assigned[best_idx] = True
                    break
                distances[best_idx] = np.inf
                if np.all(distances == np.inf):
                    assignments[j] = j
                    break
        tracked_roots[i] = current_roots[assignments]
    return tracked_roots


def generate_cubic_discriminant(z, beta, a, y):
    a_coef = z * a
    b_coef = z * (a + 1) + a * (1 - y)
    c_coef = z + (a + 1) - y - y * beta * (a - 1)
    d_coef = 1
    return (
        18 * a_coef * b_coef * c_coef * d_coef
        - 27 * a_coef ** 2 * d_coef ** 2
        + b_coef ** 2 * c_coef ** 2
        - 2 * b_coef ** 3 * d_coef
        - 9 * a_coef * c_coef ** 3
    )


# All plotting is now handled by JavaScript in the frontend


# --------- Image Processing Helper Functions --------- #
def compute_ssim(img1, img2, use_gpu=True):
    """
    Compute Structural Similarity Index (SSIM) between two grayscale images.
    GPU-accelerated when available.

    Args:
        img1, img2: numpy or cupy arrays of same shape, values in [0, 1]
        use_gpu: whether to use GPU if available

    Returns:
        ssim_value: float between -1 and 1 (higher is better, 1 = identical)
    """
    # Use GPU if available and requested
    if use_gpu and GPU_AVAILABLE:
        img1 = to_gpu(img1)
        img2 = to_gpu(img2)
        xp_local = cp
    else:
        xp_local = np

    # Constants for stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Compute means
    mu1 = xp_local.mean(img1)
    mu2 = xp_local.mean(img2)

    # Compute variances and covariance
    sigma1_sq = xp_local.var(img1)
    sigma2_sq = xp_local.var(img2)
    sigma12 = xp_local.mean((img1 - mu1) * (img2 - mu2))

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator

    # Return as Python float
    return float(to_cpu(ssim))

def compute_mean_ssim(images1, images2, use_gpu=True):
    """
    Compute mean SSIM across multiple image pairs.
    GPU-accelerated when available.

    Args:
        images1, images2: numpy or cupy arrays of shape (n, H, W)
        use_gpu: whether to use GPU if available

    Returns:
        mean_ssim: average SSIM across all image pairs
    """
    n = images1.shape[0]

    if use_gpu and GPU_AVAILABLE:
        # GPU-accelerated batch processing
        images1_gpu = to_gpu(images1)
        images2_gpu = to_gpu(images2)

        ssim_values = []
        for i in range(n):
            ssim = compute_ssim(images1_gpu[i], images2_gpu[i], use_gpu=True)
            ssim_values.append(ssim)
    else:
        # CPU processing
        ssim_values = []
        for i in range(n):
            ssim = compute_ssim(images1[i], images2[i], use_gpu=False)
            ssim_values.append(ssim)

    return float(np.mean(ssim_values))

def get_random_matrix_folder():
    """Get the .random_matrix folder path in user's home directory"""
    home = Path.home()
    folder = home / ".random_matrix"
    folder.mkdir(exist_ok=True)
    return folder

def get_temp_folder():
    """Get the temp folder path"""
    temp = get_random_matrix_folder() / "temp"
    temp.mkdir(exist_ok=True)
    return temp

def cleanup_temp_folder():
    """Clean up temp folder on exit - DISABLED to let user manage files"""
    # User will manage temp files manually through File Management tab
    pass

def add_laplacian_noise(images, scale=0.1, seed=None):
    """
    Add independent Laplacian (double exponential) noise to a batch of grayscale images.

    Laplacian noise has heavier tails than Gaussian and is zero-mean.
    PDF: p(x) = (1/2b) * exp(-|x|/b) where b is the scale parameter.

    Args:
        images: numpy array (n, H, W) or (H, W) in [0, 1]
        scale: scale parameter (b) of Laplacian distribution
               Related to standard deviation: std = scale * sqrt(2)
               Default: 0.1
        seed: random seed for reproducibility (default: None for random behavior)

    Returns:
        noisy_images: numpy array, clipped to [0, 1]
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Handle both single image and batch
    if images.ndim == 2:
        H, W = images.shape
        noise = np.random.laplace(0, scale, size=(H, W))
    else:
        n, H, W = images.shape
        noise = np.random.laplace(0, scale, size=(n, H, W))

    noisy = images + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)

def add_mixture_gaussian_noise(images, weights, means, sigmas, seed=None):
    """
    Add Mixture of Gaussians noise to a batch of grayscale images.

    Uses a mixture of Gaussian components with customizable weights, means, and variances
    to create a multi-modal noise distribution.

    Args:
        images: numpy array (n, H, W) or (H, W) in [0, 1]
        weights: list of weights for each component (should sum to 1.0)
        means: list of mean values for each component
        sigmas: list of standard deviations for each component
        seed: random seed for reproducibility (default: None for random behavior)

    Returns:
        noisy_images: numpy array, clipped to [0, 1]
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # Handle both single image and batch
    if images.ndim == 2:
        H, W = images.shape
        shape = (H, W)
        noise = np.zeros((H, W))
    else:
        n, H, W = images.shape
        shape = (n, H, W)
        noise = np.zeros((n, H, W))

    # Generate noise from mixture
    num_components = len(weights)
    flat_shape = np.prod(shape)

    # Randomly select component for each pixel
    component_choices = np.random.choice(num_components, size=flat_shape, p=weights)

    # Generate noise for each component
    noise_flat = np.zeros(flat_shape)
    for i in range(num_components):
        mask = (component_choices == i)
        count = np.sum(mask)
        if count > 0:
            noise_flat[mask] = np.random.normal(means[i], sigmas[i], size=count)

    # Reshape to original dimensions
    noise = noise_flat.reshape(shape)

    noisy = images + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)

def images_to_matrix(images):
    """Convert images (n, H, W) to data matrix X of shape (p, n)"""
    n, H, W = images.shape
    p = H * W
    X = images.reshape(n, p).T
    return X

def matrix_to_images(X, H, W):
    """Convert matrix X of shape (p, n) back to images (n, H, W)"""
    p, n = X.shape
    assert p == H * W
    images = X.T.reshape(n, H, W)
    return images

def mp_denoise(X, original_images=None, H=None, W=None, sigma_steps=50, optimize_ssim=True, use_gpu=True):
    """
    M-P law denoising with optional SSIM-based optimization.
    GPU-accelerated when available.
    
    Uses S = (1/n) X^T X (n×n matrix) for efficiency when p > n.
    Eigenvectors recovered via: v'_j = (X @ v_j) / sqrt(n * lambda_j)

    Args:
        X: data matrix (p, n)
        original_images: original clean images (n, H, W) for SSIM optimization
        H, W: image dimensions for reconstruction
        sigma_steps: number of steps to search for best sigma (default: 50)
        optimize_ssim: if True and original_images provided, optimize sigma using SSIM
        use_gpu: whether to use GPU if available

    Returns:
        X_denoised: denoised data matrix
        info: dictionary with denoising parameters
    """
    # Move to GPU if available
    if use_gpu and GPU_AVAILABLE:
        X_gpu = to_gpu(X)
        xp_local = cp
    else:
        X_gpu = X
        xp_local = np

    p, n = X_gpu.shape

    # Sample covariance S = (1/n) X^T X (n×n matrix, more efficient when p > n)
    S = (X_gpu.T @ X_gpu) / n

    # Eigen decomposition of n×n matrix
    eigvals, eigvecs_small = xp_local.linalg.eigh(S)
    idx = xp_local.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_small = eigvecs_small[:, idx]

    # Aspect ratio
    y = p / float(n)
    sqrt_y = float(xp_local.sqrt(y))

    # Positive eigenvalues
    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) < 3:
        return X, {"y": y, "sigma2_hat": None, "best_ssim": None, "optimized": False}

    # Use bottom half as noise bulk to estimate initial sigma
    half = len(pos_eigs) // 2
    bulk_eigs = pos_eigs[half:]
    bulk_min = float(xp_local.min(bulk_eigs))

    # Estimate initial sigma^2
    lambda_minus_factor = (1.0 - sqrt_y) ** 2
    sigma2_initial = bulk_min / lambda_minus_factor

    # SSIM-based optimization if requested
    best_sigma2 = sigma2_initial
    best_ssim = None

    if optimize_ssim and original_images is not None and H is not None and W is not None:
        # Search range: from 0.01 * initial to 2.0 * initial
        sigma2_min = max(0.01, sigma2_initial * 0.1)
        sigma2_max = sigma2_initial * 3.0
        sigma2_range = np.linspace(sigma2_min, sigma2_max, sigma_steps)

        best_ssim = -np.inf

        for sigma2_test in sigma2_range:
            lambda_plus_test = sigma2_test * (1.0 + sqrt_y) ** 2

            # Signal eigenvalues for this sigma
            signal_mask_test = eigvals > (lambda_plus_test + 1e-9)
            num_signal_test = int(xp_local.sum(signal_mask_test))

            if num_signal_test == 0:
                X_test = xp_local.zeros_like(X_gpu)
            else:
                # Get signal eigenvalues and small eigenvectors
                signal_eigvals = eigvals[signal_mask_test]
                V_small = eigvecs_small[:, signal_mask_test]
                
                # Recover p-dimensional eigenvectors: v'_j = (X @ v_j) / sqrt(n * lambda_j)
                Gamma_test = (X_gpu @ V_small) / xp_local.sqrt(float(n) * signal_eigvals)
                
                X_test = Gamma_test @ (Gamma_test.T @ X_gpu)

            # Convert to images and compute SSIM (move to CPU for image conversion)
            X_test_cpu = to_cpu(X_test)
            test_images = matrix_to_images(X_test_cpu, H, W)
            test_images = np.clip(test_images, 0, 1)

            ssim_score = compute_mean_ssim(original_images, test_images, use_gpu=use_gpu)

            if ssim_score > best_ssim:
                best_ssim = ssim_score
                best_sigma2 = sigma2_test

    # Apply denoising with best sigma
    lambda_plus = best_sigma2 * (1.0 + sqrt_y) ** 2

    # Signal eigenvalues
    signal_mask = eigvals > (lambda_plus + 1e-9)
    num_signal = int(xp_local.sum(signal_mask))

    if num_signal == 0:
        X_denoised = xp_local.zeros_like(X_gpu)
    else:
        # Get signal eigenvalues and small eigenvectors
        signal_eigvals = eigvals[signal_mask]
        V_small = eigvecs_small[:, signal_mask]
        
        # Recover p-dimensional eigenvectors: v'_j = (X @ v_j) / sqrt(n * lambda_j)
        Gamma = (X_gpu @ V_small) / xp_local.sqrt(float(n) * signal_eigvals)
        
        X_denoised = Gamma @ (Gamma.T @ X_gpu)

    # Move result back to CPU
    X_denoised = to_cpu(X_denoised)

    info = {
        "y": y,
        "sigma2_hat": best_sigma2,
        "lambda_plus": lambda_plus,
        "num_signal": num_signal,
        "best_ssim": best_ssim,
        "optimized": (optimize_ssim and original_images is not None)
    }
    return X_denoised, info

def optimize_generalized_params(X):
    """Optimize sigma2, a, beta to match lower bound (from random_matrix.py lines 778-883)
    
    Uses S = (1/n) X^T X (n×n matrix) for efficiency when p > n.
    """
    p, n = X.shape

    # Compute eigenvalues using S = (1/n) X^T X (n×n matrix)
    S = (X.T @ X) / n
    eigvals = np.linalg.eigvalsh(S)
    eigvals = eigvals[::-1]

    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) < 3:
        return 1.0, 2.0, 0.5

    y = p / float(n)

    # Use bottom half to estimate noise bulk
    half = len(pos_eigs) // 2
    bulk_eigs = pos_eigs[half:]
    bulk_min = np.min(bulk_eigs)
    target_lower_bound = bulk_min

    def compute_lower_bound(params):
        sigma2, a, beta = params
        if sigma2 <= 0 or a <= 1 or beta <= 0 or beta >= 1:
            return np.inf

        def g(t):
            numerator = y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)
            denominator = (a * t + 1) * (t**2 + t)
            if abs(denominator) < 1e-12:
                return np.inf
            return numerator / denominator

        try:
            result_max = minimize_scalar(lambda t: -g(t), bounds=(1e-6, 100), method='bounded')
            g_plus = -result_max.fun
            return sigma2 * g_plus
        except:
            return np.inf

    def objective(params):
        lower_bound = compute_lower_bound(params)
        if lower_bound == np.inf:
            return 1e10
        return (lower_bound - target_lower_bound) ** 2

    x0 = [1.0, 2.0, 0.5]
    bounds = [(0.01, 10.0), (1.01, 10.0), (0.01, 0.99)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
    return tuple(result.x)

def optimize_gen_params_sympy(X, a, beta):
    """
    Optimize sigma2 using sympy with manual a and beta inputs.
    Uses sympy to find max g(t) and optimizes σ² so that max g(t) * σ² ≈ bulk_min

    The approach:
    1. User provides manual a and beta
    2. Use sympy to find critical points of g(t) by solving dg/dt = 0
    3. Optimize sigma2 so that max g(t) * sigma2 ≈ bulk_min

    Args:
        X: data matrix (p, n)
        a: spike value parameter (manual input)
        beta: beta parameter (manual input)

    Returns:
        sigma2: optimized parameter
    """
    p, n = X.shape

    # Compute eigenvalues using S = (1/n) X^T X (n×n matrix)
    S = (X.T @ X) / n
    eigvals = np.linalg.eigvalsh(S)
    eigvals = eigvals[::-1]

    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) < 3:
        return 0.01

    y = p / float(n)

    # Use bottom half to estimate noise bulk
    half = len(pos_eigs) // 2
    bulk_eigs = pos_eigs[half:]
    bulk_min = np.min(bulk_eigs)

    # Find max g(t) for the given a and beta values
    def g_numeric(t):
        num = y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)
        den = (a * t + 1) * (t**2 + t)
        if abs(den) < 1e-12:
            return -np.inf
        return num / den

    # Find maximum g(t) in positive range
    try:
        result_max = minimize_scalar(lambda t: -g_numeric(t), bounds=(1e-6, 100), method='bounded')
        g_max = -result_max.fun
    except:
        g_max = 1.0

    # Optimize sigma2 so that: sigma2 * g_max ≈ bulk_min
    # Therefore: sigma2 ≈ bulk_min / g_max
    if abs(g_max) > 1e-12:
        sigma2_opt = bulk_min / g_max
    else:
        # Fallback to M-P law estimate
        sqrt_y = np.sqrt(y)
        lambda_minus_factor = (1.0 - sqrt_y) ** 2
        sigma2_opt = bulk_min / lambda_minus_factor

    return sigma2_opt

def mp_denoise_centralized(X, use_gpu=True):
    """
    M-P law denoising with data centralization.

    Steps:
    1. Compute row-wise mean X̄
    2. Center data: X_centered = X - X̄
    3. Apply M-P denoising on centered data
    4. Add back mean: X_denoised = X_denoised_centered + X̄

    Args:
        X: data matrix (p, n)
        use_gpu: whether to use GPU if available

    Returns:
        X_denoised: denoised data matrix
        info: dictionary with denoising parameters
    """
    if use_gpu and GPU_AVAILABLE:
        X_gpu = to_gpu(X)
        xp_local = cp
    else:
        X_gpu = X
        xp_local = np

    p, n = X_gpu.shape

    # Step 1: Compute row-wise mean
    X_mean = xp_local.mean(X_gpu, axis=1, keepdims=True)

    # Step 2: Center the data
    X_centered = X_gpu - X_mean

    # Step 3: Apply M-P denoising on centered data
    # Move to CPU for mp_denoise if needed
    X_centered_cpu = to_cpu(X_centered)
    X_denoised_centered, info = mp_denoise(X_centered_cpu, None, None, None, optimize_ssim=False, use_gpu=use_gpu)

    # Step 4: Add back the mean
    X_mean_cpu = to_cpu(X_mean)
    X_denoised = X_denoised_centered + X_mean_cpu

    # Add centralization info
    info['centralized'] = True
    info['mean_magnitude'] = float(np.mean(np.abs(X_mean_cpu)))

    return X_denoised, info

def gen_denoise_centralized(X, use_gpu=True):
    """
    Generalized covariance denoising with data centralization.

    Steps:
    1. Compute row-wise mean X̄
    2. Center data: X_centered = X - X̄
    3. Apply Gen Cov denoising on centered data (auto-optimize σ², a, β)
    4. Add back mean: X_denoised = X_denoised_centered + X̄

    Args:
        X: data matrix (p, n)
        use_gpu: whether to use GPU if available

    Returns:
        X_denoised: denoised data matrix
        info: dictionary with denoising parameters
    """
    if use_gpu and GPU_AVAILABLE:
        X_gpu = to_gpu(X)
        xp_local = cp
    else:
        X_gpu = X
        xp_local = np

    p, n = X_gpu.shape

    # Step 1: Compute row-wise mean
    X_mean = xp_local.mean(X_gpu, axis=1, keepdims=True)

    # Step 2: Center the data
    X_centered = X_gpu - X_mean

    # Step 3: Apply generalized denoising on centered data
    X_centered_cpu = to_cpu(X_centered)
    X_denoised_centered, info = generalized_denoise(X_centered_cpu, None, None, None, None, None, None, optimize_ssim=False, use_gpu=use_gpu)

    # Step 4: Add back the mean
    X_mean_cpu = to_cpu(X_mean)
    X_denoised = X_denoised_centered + X_mean_cpu

    # Add centralization info
    info['centralized'] = True
    info['mean_magnitude'] = float(np.mean(np.abs(X_mean_cpu)))

    return X_denoised, info

def gen_denoise_centralized_manual(X, a, beta, use_gpu=True):
    """
    Generalized covariance denoising with data centralization and manual a, β.
    σ² is auto-optimized from centered data's bulk_min using M-P law.

    Steps:
    1. Compute row-wise mean X̄
    2. Center data: X_centered = X - X̄
    3. Estimate σ² from centered data using M-P law: σ² = bulk_min / (1-√y)²
    4. Apply Gen Cov denoising with optimized σ² and manual a, β on centered data
    5. Add back mean: X_denoised = X_denoised_centered + X̄

    Args:
        X: data matrix (p, n)
        a: spike value parameter (manual)
        beta: beta parameter (manual)
        use_gpu: whether to use GPU if available

    Returns:
        X_denoised: denoised data matrix
        info: dictionary with denoising parameters
    """
    if use_gpu and GPU_AVAILABLE:
        X_gpu = to_gpu(X)
        xp_local = cp
    else:
        X_gpu = X
        xp_local = np

    p, n = X_gpu.shape

    # Step 1: Compute row-wise mean
    X_mean = xp_local.mean(X_gpu, axis=1, keepdims=True)

    # Step 2: Center the data
    X_centered = X_gpu - X_mean
    X_centered_cpu = to_cpu(X_centered)

    # Step 3: Estimate σ² from centered data using M-P law
    # Compute eigenvalues of centered data
    S = (X_centered_cpu.T @ X_centered_cpu) / n
    eigvals = np.linalg.eigvalsh(S)
    eigvals = eigvals[::-1]

    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) >= 3:
        y = p / float(n)
        sqrt_y = np.sqrt(y)

        # Use bottom half to estimate noise bulk
        half = len(pos_eigs) // 2
        bulk_eigs = pos_eigs[half:]
        bulk_min = np.min(bulk_eigs)

        # Estimate sigma2 using M-P law
        lambda_minus_factor = (1.0 - sqrt_y) ** 2
        sigma2 = bulk_min / lambda_minus_factor
    else:
        sigma2 = 0.01  # fallback value

    # Step 4: Apply generalized denoising with optimized σ² and manual a, β
    X_denoised_centered, info = generalized_denoise(X_centered_cpu, sigma2, a, beta, optimize_ssim=False, use_gpu=use_gpu)

    # Step 5: Add back the mean
    X_mean_cpu = to_cpu(X_mean)
    X_denoised = X_denoised_centered + X_mean_cpu

    # Add centralization info
    info['centralized'] = True
    info['mean_magnitude'] = float(np.mean(np.abs(X_mean_cpu)))
    info['sigma2_from_centered'] = sigma2

    return X_denoised, info

def mp_denoise_top20std(X, use_gpu=True):
    """
    M-P law denoising where support range matches std dev of top 5% eigenvalues.
    Range: σ²(1+√y)² - σ²(1-√y)² ≈ std(λ_{0.8n} to λ_n)
    
    Uses S = (1/n) X^T X (n×n matrix) for efficiency when p > n.
    Eigenvectors recovered via: v'_j = (X @ v_j) / sqrt(n * lambda_j)
    """
    if use_gpu and GPU_AVAILABLE:
        X_gpu = to_gpu(X)
        xp_local = cp
    else:
        X_gpu = X
        xp_local = np

    p, n = X_gpu.shape

    # Sample covariance S = (1/n) X^T X (n×n matrix)
    S = (X_gpu.T @ X_gpu) / n

    # Eigen decomposition of n×n matrix
    eigvals, eigvecs_small = xp_local.linalg.eigh(S)
    idx = xp_local.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_small = eigvecs_small[:, idx]

    # Aspect ratio
    y = p / float(n)
    sqrt_y = float(xp_local.sqrt(y))

    # Positive eigenvalues
    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) < 5:
        return X, {"y": y, "sigma2_hat": None, "optimized": False}

    # Get top 20% eigenvalues
    top20_count = max(1, int(len(pos_eigs) * 0.2))
    top20_eigs = pos_eigs[:top20_count]
    top20_std = float(xp_local.std(top20_eigs))

    # Calculate sigma² such that support range matches std dev
    # Range = σ²(1+√y)² - σ²(1-√y)² = σ² * 4√y ≈ top20_std
    sigma2_estimated = top20_std / (4 * sqrt_y) if sqrt_y > 0 else 0.01

    # Apply denoising
    lambda_plus = sigma2_estimated * (1.0 + sqrt_y) ** 2

    # Signal eigenvalues
    signal_mask = eigvals > (lambda_plus + 1e-9)
    num_signal = int(xp_local.sum(signal_mask))

    if num_signal == 0:
        X_denoised = xp_local.zeros_like(X_gpu)
    else:
        # Get signal eigenvalues and small eigenvectors
        signal_eigvals = eigvals[signal_mask]
        V_small = eigvecs_small[:, signal_mask]
        
        # Recover p-dimensional eigenvectors: v'_j = (X @ v_j) / sqrt(n * lambda_j)
        Gamma = (X_gpu @ V_small) / xp_local.sqrt(float(n) * signal_eigvals)
        
        X_denoised = Gamma @ (Gamma.T @ X_gpu)

    # Move result back to CPU
    X_denoised = to_cpu(X_denoised)

    info = {
        "y": y,
        "sigma2_hat": sigma2_estimated,
        "lambda_plus": lambda_plus,
        "num_signal": num_signal,
        "top20_std": top20_std,
        "support_range": sigma2_estimated * 4 * sqrt_y,
        "optimized": True
    }
    return X_denoised, info

def optimize_gen_params_top20std(X):
    """Optimize generalized covariance params where support range matches top 5% std dev
    
    Uses S = (1/n) X^T X (n×n matrix) for efficiency when p > n.
    """
    p, n = X.shape

    # Compute eigenvalues using S = (1/n) X^T X (n×n matrix)
    S = (X.T @ X) / n
    eigvals = np.linalg.eigvalsh(S)
    eigvals = eigvals[::-1]

    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) < 5:
        return 1.0, 2.0, 0.5

    y = p / float(n)

    # Get top 20% eigenvalues and their std dev
    top20_count = max(1, int(len(pos_eigs) * 0.2))
    top20_eigs = pos_eigs[:top20_count]
    target_std = np.std(top20_eigs)

    def compute_support_range(params):
        sigma2, a, beta = params
        if sigma2 <= 0 or a <= 1 or beta <= 0 or beta >= 1:
            return np.inf

        def g(t):
            numerator = y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)
            denominator = (a * t + 1) * (t**2 + t)
            if abs(denominator) < 1e-12:
                return np.inf
            return numerator / denominator

        try:
            # Find max g(t) for t > 0
            result_max = minimize_scalar(lambda t: -g(t), bounds=(1e-6, 100), method='bounded')
            g_plus = -result_max.fun

            # Find min g(t) for t in (-1/a, 0)
            t_lower = -1.0/a + 1e-6
            result_min = minimize_scalar(g, bounds=(t_lower, -1e-6), method='bounded')
            g_minus = result_min.fun

            # Support range
            support_range = sigma2 * (g_minus - g_plus)
            return support_range
        except:
            return np.inf

    def objective(params):
        support_range = compute_support_range(params)
        if support_range == np.inf or support_range <= 0:
            return 1e10
        return (support_range - target_std) ** 2

    x0 = [1.0, 2.0, 0.5]
    bounds = [(0.01, 10.0), (1.01, 10.0), (0.01, 0.99)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
    return tuple(result.x)

def generalized_denoise(X, sigma2=None, a=None, beta=None, original_images=None, H=None, W=None,
                        sigma_steps=100, a_steps=100, beta_steps=100,
                        sigma_step_size=0.01, a_step_size=0.01, beta_step_size=0.01,
                        optimize_ssim=True, use_gpu=True):
    """
    Generalized covariance denoising with optional SSIM-based optimization.
    GPU-accelerated when available.
    
    Uses S = (1/n) X^T X (n×n matrix) for efficiency when p > n.
    Eigenvectors recovered via: v'_j = (X @ v_j) / sqrt(n * lambda_j)

    Args:
        X: data matrix (p, n)
        sigma2, a, beta: parameters (if None, will auto-optimize)
        original_images: original clean images (n, H, W) for SSIM optimization
        H, W: image dimensions for reconstruction
        sigma_steps: number of steps for sigma search (default: 100)
        a_steps: number of steps for a search (default: 100)
        beta_steps: number of steps for beta search (default: 100)
        sigma_step_size: step size for sigma (default: 0.01)
        a_step_size: step size for a (default: 0.01)
        beta_step_size: step size for beta (default: 0.01)
        optimize_ssim: if True and original_images provided, optimize params using SSIM
        use_gpu: whether to use GPU if available

    Returns:
        X_denoised: denoised data matrix
        info: dictionary with denoising parameters
    """
    # Move to GPU if available
    if use_gpu and GPU_AVAILABLE:
        X_gpu = to_gpu(X)
        xp_local = cp
    else:
        X_gpu = X
        xp_local = np

    p, n = X_gpu.shape

    # Sample covariance S = (1/n) X^T X (n×n matrix)
    S = (X_gpu.T @ X_gpu) / n

    # Eigen decomposition of n×n matrix
    eigvals, eigvecs_small = xp_local.linalg.eigh(S)
    idx = xp_local.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_small = eigvecs_small[:, idx]

    # Aspect ratio
    y = p / float(n)

    # Auto-optimize if not provided
    if sigma2 is None or a is None or beta is None:
        if optimize_ssim and original_images is not None and H is not None and W is not None:
            # SSIM-based optimization using grid search (pass GPU arrays and flag)
            sigma2, a, beta = optimize_generalized_params_ssim(
                X_gpu, eigvals, eigvecs_small, y,
                original_images, H, W,
                sigma_steps, a_steps, beta_steps,
                sigma_step_size, a_step_size, beta_step_size,
                use_gpu=use_gpu, xp=xp_local
            )
        else:
            # Use original optimization method (CPU-based)
            X_cpu = to_cpu(X_gpu)
            sigma2, a, beta = optimize_generalized_params(X_cpu)

    # Define g(t) function
    def g(t):
        numerator = y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)
        denominator = (a * t + 1) * (t**2 + t)
        if abs(denominator) < 1e-12:
            return np.inf
        return numerator / denominator

    # Find g_plus and g_minus
    result_max = minimize_scalar(lambda t: -g(t), bounds=(1e-6, 100), method='bounded')
    g_plus = -result_max.fun

    result_min = minimize_scalar(g, bounds=(-1/a + 1e-6, -1e-6), method='bounded')
    g_minus = result_min.fun

    # Support bounds
    lambda_plus = sigma2 * g_plus
    lambda_minus = sigma2 * g_minus

    # Signal eigenvalues
    signal_mask = eigvals > (lambda_minus + 1e-9)
    num_signal = int(xp_local.sum(signal_mask))

    if num_signal == 0:
        X_denoised = xp_local.zeros_like(X_gpu)
    else:
        # Get signal eigenvalues and small eigenvectors
        signal_eigvals = eigvals[signal_mask]
        V_small = eigvecs_small[:, signal_mask]
        
        # Recover p-dimensional eigenvectors: v'_j = (X @ v_j) / sqrt(n * lambda_j)
        Gamma = (X_gpu @ V_small) / xp_local.sqrt(float(n) * signal_eigvals)
        
        X_denoised = Gamma @ (Gamma.T @ X_gpu)

    # Move result back to CPU
    X_denoised = to_cpu(X_denoised)

    info = {
        "y": y,
        "a": a,
        "beta": beta,
        "sigma2": sigma2,
        "num_signal": num_signal,
        "optimized": (optimize_ssim and original_images is not None)
    }
    return X_denoised, info

def optimize_generalized_params_ssim(X, eigvals, eigvecs_small, y,
                                     original_images, H, W,
                                     sigma_steps, a_steps, beta_steps,
                                     sigma_step_size=0.01, a_step_size=0.01, beta_step_size=0.01,
                                     use_gpu=True, xp=np):
    """
    Optimize sigma2, a, beta using SSIM metric with grid search.
    GPU-accelerated when available.
    Step sizes are user-configurable.
    
    Uses eigenvector recovery: v'_j = (X @ v_j) / sqrt(n * lambda_j)

    Args:
        X: data matrix (on GPU if use_gpu=True)
        eigvals, eigvecs_small: eigenvalue decomposition of X^T X / n (on GPU if use_gpu=True)
        y: aspect ratio
        original_images: original clean images (CPU)
        H, W: image dimensions
        sigma_steps, a_steps, beta_steps: number of steps for each parameter
        sigma_step_size: step size for sigma (default: 0.01)
        a_step_size: step size for a (default: 0.01)
        beta_step_size: step size for beta (default: 0.01)
        use_gpu: whether arrays are on GPU
        xp: numpy or cupy depending on GPU availability

    Returns:
        best_sigma2, best_a, best_beta
    """
    p, n = X.shape
    
    # Define search ranges using user-configurable step sizes
    # Sigma: 0.01 to 0.01 + (sigma_steps - 1) * sigma_step_size
    sigma_min = 0.01
    sigma_max = sigma_min + (sigma_steps - 1) * sigma_step_size

    # A: 1.01 to 1.01 + (a_steps - 1) * a_step_size
    a_min = 1.01
    a_max = a_min + (a_steps - 1) * a_step_size

    # Beta: 0.01 to 0.01 + (beta_steps - 1) * beta_step_size, max 0.99
    beta_min = 0.01
    beta_max = min(0.99, beta_min + (beta_steps - 1) * beta_step_size)

    # Create parameter grids
    sigma_range = np.linspace(sigma_min, sigma_max, sigma_steps)
    a_range = np.linspace(a_min, a_max, a_steps)
    beta_range = np.linspace(beta_min, beta_max, beta_steps)

    best_ssim = -np.inf
    best_sigma2 = sigma_range[0]
    best_a = a_range[0]
    best_beta = beta_range[0]

    # Grid search
    total_iterations = len(sigma_range) * len(a_range) * len(beta_range)
    current_iteration = 0

    for sigma2_test in sigma_range:
        for a_test in a_range:
            for beta_test in beta_range:
                current_iteration += 1

                # Define g(t) function for these parameters
                def g(t):
                    numerator = y * beta_test * (a_test - 1) * t + (a_test * t + 1) * ((y - 1) * t - 1)
                    denominator = (a_test * t + 1) * (t**2 + t)
                    if abs(denominator) < 1e-12:
                        return np.inf
                    return numerator / denominator

                try:
                    # Find g_minus
                    result_min = minimize_scalar(g, bounds=(-1/a_test + 1e-6, -1e-6), method='bounded')
                    g_minus = result_min.fun
                    lambda_minus = sigma2_test * g_minus

                    # Signal eigenvalues
                    signal_mask = eigvals > (lambda_minus + 1e-9)
                    num_signal = int(xp.sum(signal_mask))

                    if num_signal == 0:
                        X_test = xp.zeros_like(X)
                    else:
                        # Get signal eigenvalues and small eigenvectors
                        signal_eigvals = eigvals[signal_mask]
                        V_small = eigvecs_small[:, signal_mask]
                        
                        # Recover p-dimensional eigenvectors: v'_j = (X @ v_j) / sqrt(n * lambda_j)
                        Gamma_test = (X @ V_small) / xp.sqrt(float(n) * signal_eigvals)
                        
                        X_test = Gamma_test @ (Gamma_test.T @ X)

                    # Convert to images and compute SSIM (move to CPU for image processing)
                    X_test_cpu = to_cpu(X_test)
                    test_images = matrix_to_images(X_test_cpu, H, W)
                    test_images = np.clip(test_images, 0, 1)

                    ssim_score = compute_mean_ssim(original_images, test_images, use_gpu=use_gpu)

                    if ssim_score > best_ssim:
                        best_ssim = ssim_score
                        best_sigma2 = sigma2_test
                        best_a = a_test
                        best_beta = beta_test

                except Exception as e:
                    # Skip invalid parameter combinations
                    continue

    return best_sigma2, best_a, best_beta


# --------- Desktop (PyWebView) bridge --------- #
class Bridge:
    def __init__(self, window):
        self.window = window
        self.processed_results = []

    def _log(self, event, payload=None):
        pass

    def _update_progress_ui(self, bar_id, status_id, step, total):
        """Update progress bar and status text in real-time using evaluate_js"""
        if self.window and step > 0 and total > 0:
            pct = int((step / total) * 100)
            try:
                # Update progress bar width
                self.window.evaluate_js(f"document.getElementById('{bar_id}').style.width = '{pct}%';")
                # Update status text
                self.window.evaluate_js(f"document.getElementById('{status_id}').innerHTML = '{step}/{total} ({pct}%)';")
            except Exception as e:
                pass

    def _update_status_ui(self, element_id, text):
        """Update any UI element text in real-time using evaluate_js"""
        if self.window:
            try:
                # Escape single quotes in text
                text_escaped = text.replace("'", "\\'")
                self.window.evaluate_js(f"document.getElementById('{element_id}').innerHTML = '{text_escaped}';")
            except Exception as e:
                pass

    def im_vs_z(self, params):
        self._log("im_vs_z:start", params)
        progress = []

        def cb(step, total):
            progress.append({"step": int(step), "total": int(total)})
            self._update_progress_ui('bar1', 'status1', step, total)

        a = float(params.get("a", 2.0))
        y = float(params.get("y", 1.0))
        beta = float(params.get("beta", 0.5))
        z_min = float(params.get("z_min", 0.01))
        z_max = float(params.get("z_max", 10.0))
        points = int(params.get("points", 400))

        result = compute_ImS_vs_Z(a, y, beta, points, z_min, z_max, progress_cb=cb)
        explanation = (
            "Curves show the three roots of the cubic; dashed vticks mark the Im(s)=0 crossings "
            "where root structure flips between real and complex."
        )
        self._log("im_vs_z:done", {"progress": progress})
        return {
            "data": result,
            "params": {"a": a, "y": y, "beta": beta},
            "progress": progress,
            "explanation": explanation
        }

    def roots_vs_beta(self, params):
        self._log("roots_vs_beta:start", params)
        progress = []

        def cb(step, total):
            progress.append({"step": int(step), "total": int(total)})
            self._update_progress_ui('bar2', 'status2', step, total)

        z = float(params.get("z", 1.0))
        y = float(params.get("y", 1.0))
        a = float(params.get("a", 1.0))
        beta_min = float(params.get("beta_min", 0.0))
        beta_max = float(params.get("beta_max", 1.0))
        points = int(params.get("points", 400))

        # Compute roots
        beta_points = np.linspace(beta_min, beta_max, points)
        all_roots = []
        discriminants = []

        for idx, beta in enumerate(beta_points):
            roots = compute_cubic_roots(z, beta, a, y)
            roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
            all_roots.append(roots)
            discriminants.append(generate_cubic_discriminant(z, beta, a, y))
            if cb:
                cb(idx + 1, points)

        all_roots = np.array(all_roots)
        tracked_roots = track_roots_consistently(beta_points, all_roots)

        explanation = (
            "Im(s) highlights where roots gain imaginary parts; Re(s) shows ordering; the discriminant crosses zero at transitions between 3 real roots and 1 real + 2 complex."
        )
        self._log("roots_vs_beta:done", {"progress": progress})
        return {
            "data": {
                "beta_values": beta_points.tolist(),
                "ims_values": np.imag(tracked_roots).tolist(),
                "real_values": np.real(tracked_roots).tolist(),
                "discriminants": discriminants
            },
            "params": {"z": z, "y": y, "a": a},
            "progress": progress,
            "explanation": explanation,
        }

    def eigen_distribution(self, params):
        self._log("eigen:start", params)
        progress = []

        def cb(step, total):
            progress.append({"step": int(step), "total": int(total)})
            self._update_progress_ui('bar4', 'status4', step, total)

        beta = float(params.get("beta", 0.5))
        a = float(params.get("a", 2.0))
        n = int(params.get("n", 400))
        p = int(params.get("p", max(1, n)))
        p = max(1, p)
        y = p / n if n else 1.0
        seed = int(params.get("seed", 42))

        # Generate eigenvalues
        y_eff = y if y > 1 else 1 / y
        np.random.seed(seed)

        total_steps = 5
        p_dim = int(y_eff * n)
        k = int(np.floor(beta * p_dim))
        diag_entries = np.concatenate([np.full(k, a), np.full(p_dim - k, 1.0)])
        np.random.shuffle(diag_entries)
        T_n = np.diag(diag_entries)
        if cb:
            cb(1, total_steps)

        X = np.random.randn(p_dim, n)
        if cb:
            cb(2, total_steps)

        S_n = (1 / n) * (X @ X.T)
        if cb:
            cb(3, total_steps)

        B_n = S_n @ T_n
        eigenvalues = np.linalg.eigvalsh(B_n)
        if cb:
            cb(4, total_steps)

        kde = gaussian_kde(eigenvalues)
        x_vals = np.linspace(float(np.min(eigenvalues)), float(np.max(eigenvalues)), 400)
        kde_vals = kde(x_vals)
        if cb:
            cb(5, total_steps)

        stats = {
            "min": float(np.min(eigenvalues)),
            "max": float(np.max(eigenvalues)),
            "mean": float(np.mean(eigenvalues)),
            "std": float(np.std(eigenvalues)),
        }
        explanation = (
            "Histogram + KDE of eigenvalues for B_n = S_n T_n with mixed diagonal entries. Use stats to compare spread."
        )
        self._log("eigen:done", {"stats": stats})
        return {
            "data": {
                "eigenvalues": eigenvalues.tolist(),
                "kde_x": x_vals.tolist(),
                "kde_y": kde_vals.tolist()
            },
            "stats": stats,
            "progress": progress,
            "explanation": explanation
        }

    def list_random_matrix_folders(self):
        """List all folders in .random_matrix directory"""
        try:
            base_folder = get_random_matrix_folder()
            folders = [f.name for f in base_folder.iterdir() if f.is_dir() and f.name != "temp"]
            return {"folders": sorted(folders)}
        except Exception as e:
            return {"folders": []}

    def get_folder_contents(self, folder_name):
        """Get contents of a specific folder"""
        try:
            folder_path = get_random_matrix_folder() / folder_name
            files = []
            for f in sorted(folder_path.iterdir()):
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    with open(f, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    files.append({"name": f.name, "data": img_data})
            return {"files": files[:50]}  # Limit to 50 images
        except Exception as e:
            return {"files": []}

    def import_local_folder(self, params):
        """Import local folder with images to .random_matrix folder, preserving structure"""
        try:
            files = params.get("files", [])
            folder_name = params.get("folder", "imported")

            folder_path = get_random_matrix_folder() / folder_name
            folder_path.mkdir(exist_ok=True)

            for file_data in files:
                # Get relative path from the selected folder
                relative_path = file_data.get("path", file_data["name"])

                # Create full output path, preserving subfolder structure
                output_path = folder_path / relative_path

                # Create parent directories if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Decode and save image with original name
                img_bytes = base64.b64decode(file_data["data"])
                img = Image.open(io.BytesIO(img_bytes))

                # Save in original format (don't force convert to L)
                img.save(output_path)

            return {"success": True, "count": len(files), "folder": folder_name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_folder(self, params):
        """Create a new folder in .random_matrix directory"""
        try:
            folder_name = params.get("folder_name", "").strip()

            # Validate folder name
            if not folder_name:
                return {"success": False, "error": "Folder name cannot be empty"}

            # Check for invalid characters
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
            if any(char in folder_name for char in invalid_chars):
                return {"success": False, "error": "Folder name contains invalid characters"}

            # Create the folder
            folder_path = get_random_matrix_folder() / folder_name
            if folder_path.exists():
                return {"success": False, "error": f"Folder '{folder_name}' already exists"}

            folder_path.mkdir(parents=True, exist_ok=False)
            return {"success": True, "folder": folder_name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_folder(self, params):
        """Delete a folder from .random_matrix directory"""
        try:
            import shutil
            folder_name = params.get("folder_name", "").strip()

            if not folder_name:
                return {"success": False, "error": "Folder name cannot be empty"}

            # Prevent deletion of temp folder
            if folder_name == "temp":
                return {"success": False, "error": "Cannot delete temp folder"}

            folder_path = get_random_matrix_folder() / folder_name
            if not folder_path.exists():
                return {"success": False, "error": f"Folder '{folder_name}' does not exist"}

            if not folder_path.is_dir():
                return {"success": False, "error": f"'{folder_name}' is not a folder"}

            # Delete the folder and all its contents
            shutil.rmtree(folder_path)
            return {"success": True, "folder": folder_name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upload_files_to_folder(self, params):
        """Upload individual files to an existing folder in .random_matrix"""
        try:
            files = params.get("files", [])
            folder_name = params.get("folder", "").strip()

            if not folder_name:
                return {"success": False, "error": "Folder name cannot be empty"}

            folder_path = get_random_matrix_folder() / folder_name
            if not folder_path.exists():
                return {"success": False, "error": f"Folder '{folder_name}' does not exist"}

            if not folder_path.is_dir():
                return {"success": False, "error": f"'{folder_name}' is not a folder"}

            count = 0
            for file_data in files:
                filename = file_data.get("name", "")

                # Create full output path
                output_path = folder_path / filename

                # Decode and save image
                img_bytes = base64.b64decode(file_data["data"])
                img = Image.open(io.BytesIO(img_bytes))

                # Save in original format
                img.save(output_path)
                count += 1

            return {"success": True, "count": count, "folder": folder_name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def compute_image_eigenvalues(self, params):
        """Compute eigenvalue distribution for a specific image"""
        try:
            import scipy.stats as stats
            from sklearn.neighbors import KernelDensity

            # Get image data (base64 encoded)
            image_data = params.get("image_data", "")
            if not image_data:
                return {"success": False, "error": "No image data provided"}

            # Decode image
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Get dimensions
            H, W = img_array.shape

            # Compute eigenvalues of the image covariance matrix
            X_matrix = img_array.reshape(H, W)

            # Compute covariance matrix (use smaller dimension for efficiency)
            if H < W:
                # Compute X X^T / W
                C = np.dot(X_matrix, X_matrix.T) / W
            else:
                # Compute X^T X / H
                C = np.dot(X_matrix.T, X_matrix) / H

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(C)

            # Sort in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Filter near-zero eigenvalues but keep track of total count
            total_count = len(eigenvalues)
            eigenvalues_filtered = eigenvalues[eigenvalues > 1e-10]

            if len(eigenvalues_filtered) == 0:
                eigenvalues_filtered = eigenvalues[:10]  # Keep at least top 10

            # Compute improved KDE for smooth distribution
            if len(eigenvalues_filtered) > 1:
                # Use Scott's rule for bandwidth selection
                bandwidth = 1.06 * np.std(eigenvalues_filtered) * len(eigenvalues_filtered) ** (-1/5)
                bandwidth = max(bandwidth, (np.max(eigenvalues_filtered) - np.min(eigenvalues_filtered)) / 100)

                kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
                kde.fit(eigenvalues_filtered.reshape(-1, 1))

                # Extend range slightly for better visualization
                x_min = max(0, np.min(eigenvalues_filtered) - 0.1 * np.std(eigenvalues_filtered))
                x_max = np.max(eigenvalues_filtered) + 0.1 * np.std(eigenvalues_filtered)
                x_vals = np.linspace(x_min, x_max, 300)
                log_density = kde.score_samples(x_vals.reshape(-1, 1))
                kde_vals = np.exp(log_density)

                # Normalize KDE to match histogram scale (approximate)
                kde_vals = kde_vals * len(eigenvalues_filtered) * (x_max - x_min) / 50
            else:
                x_vals = np.array([0, 1])
                kde_vals = np.array([0, 0])

            # Compute comprehensive statistics
            if len(eigenvalues_filtered) > 0:
                # Percentiles
                percentiles = np.percentile(eigenvalues_filtered, [25, 50, 75, 90, 95, 99])

                stats_dict = {
                    "count": int(len(eigenvalues_filtered)),
                    "total_count": int(total_count),
                    "min": float(np.min(eigenvalues_filtered)),
                    "max": float(np.max(eigenvalues_filtered)),
                    "mean": float(np.mean(eigenvalues_filtered)),
                    "median": float(np.median(eigenvalues_filtered)),
                    "std": float(np.std(eigenvalues_filtered)),
                    "p25": float(percentiles[0]),
                    "p50": float(percentiles[1]),
                    "p75": float(percentiles[2]),
                    "p90": float(percentiles[3]),
                    "p95": float(percentiles[4]),
                    "p99": float(percentiles[5]),
                }
            else:
                stats_dict = {
                    "count": 0,
                    "total_count": int(total_count),
                    "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0,
                    "p25": 0, "p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0
                }

            return {
                "success": True,
                "eigenvalues": eigenvalues_filtered.tolist(),
                "eigenvalues_all": eigenvalues.tolist(),  # Include all eigenvalues
                "kde_x": x_vals.tolist(),
                "kde_y": kde_vals.tolist(),
                "stats": stats_dict,
                "image_size": {"height": H, "width": W}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download_images_playwright(self, params):
        """Download images using playwright (simplified version - requires playwright installation)"""
        try:
            from playwright.sync_api import sync_playwright
            import time
            import random

            url = params.get("url")
            count = params.get("count", 10)
            scale = params.get("scale", 100)
            folder_name = params.get("folder", "downloaded")

            folder_path = get_random_matrix_folder() / folder_name
            folder_path.mkdir(exist_ok=True)

            images = []
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                page = context.new_page()

                # Simple download: navigate to URL and screenshot
                page.goto(url, timeout=60000)
                time.sleep(random.uniform(2, 4))

                for i in range(count):
                    screenshot_bytes = page.screenshot()
                    img = Image.open(io.BytesIO(screenshot_bytes)).convert('L')
                    img = img.resize((scale, scale), Image.BICUBIC)

                    output_path = folder_path / f"{i:03d}.png"
                    img.save(output_path)
                    images.append(output_path)

                    if i < count - 1:
                        time.sleep(random.uniform(1, 2))

                browser.close()

            return {"success": True, "count": len(images), "folder": folder_name}
        except ImportError:
            return {"success": False, "error": "Playwright not installed. Please run: pip install playwright && playwright install chromium"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _apply_method(self, method, X_noisy, params, prefix):
        """Apply a single denoising method"""
        if method == "mp_lower_bulk":
            X_result, info = mp_denoise(X_noisy, None, None, None, optimize_ssim=False)
        elif method == "mp_top20_std":
            X_result, info = mp_denoise_top20std(X_noisy)
        elif method == "mp_centralized":
            X_result, info = mp_denoise_centralized(X_noisy)
        elif method == "gen_lower_bulk":
            X_result, info = generalized_denoise(X_noisy, None, None, None, None, None, None, optimize_ssim=False)
        elif method == "gen_top20_std":
            sigma2, a, beta = optimize_gen_params_top20std(X_noisy)
            X_result, info = generalized_denoise(X_noisy, sigma2, a, beta)
        elif method == "gen_centralized":
            X_result, info = gen_denoise_centralized(X_noisy)
        elif method == "gen_manual":
            sigma2 = params.get(f"{prefix}_sigma2", 0.01)
            a = params.get(f"{prefix}_a", 2.0)
            beta = params.get(f"{prefix}_beta", 0.5)
            X_result, info = generalized_denoise(X_noisy, sigma2, a, beta)
        elif method == "gen_centralized_manual":
            a = params.get(f"{prefix}_a", 2.0)
            beta = params.get(f"{prefix}_beta", 0.5)
            X_result, info = gen_denoise_centralized_manual(X_noisy, a, beta)
        elif method == "gen_sympy_optimal":
            a = params.get(f"{prefix}_a", 2.0)
            beta = params.get(f"{prefix}_beta", 0.5)
            sigma2 = optimize_gen_params_sympy(X_noisy, a, beta)
            X_result, info = generalized_denoise(X_noisy, sigma2, a, beta)
        else:
            X_result, info = mp_denoise(X_noisy, None, None, None, optimize_ssim=False)
        return X_result, info

    def _format_method_details(self, method, info, method_name):
        """Format method details for display"""
        num_eigvec = info.get('num_signal', 0)
        if method in ["mp_lower_bulk", "mp_top20_std", "mp_centralized"]:
            details = f"<strong>{method_name}</strong><br>"
            details += f'<strong style="color: #2563eb;">Eigenvectors used: {num_eigvec}</strong><br>'
            details += f"σ²={info.get('sigma2_hat', 0):.6f}, λ₊={info.get('lambda_plus', 0):.4f}"
            if method == "mp_top20_std" and 'top20_std' in info:
                details += f"<br>top5_std={info.get('top20_std', 0):.6f}"
            if info.get('centralized'):
                details += f"<br>Centralized (|X̄|={info.get('mean_magnitude', 0):.4f})"
        else:  # generalized covariance methods
            details = f"<strong>{method_name}</strong><br>"
            details += f'<strong style="color: #2563eb;">Eigenvectors used: {num_eigvec}</strong><br>'

            # For centralized manual, show auto-computed sigma2
            if method == "gen_centralized_manual" and 'sigma2_from_centered' in info:
                details += f"σ²={info.get('sigma2_from_centered', 0):.6f} (auto), a={info.get('a', 0):.4f} (manual), β={info.get('beta', 0):.4f} (manual)"
            else:
                details += f"σ²={info.get('sigma2', 0):.6f}, a={info.get('a', 0):.4f}, β={info.get('beta', 0):.4f}"

            if info.get('centralized'):
                details += f"<br>Centralized (|X̄|={info.get('mean_magnitude', 0):.4f})"
        return details

    def process_images(self, params):
        """Process images with denoising methods"""
        try:
            folder_name = params.get("folder")
            method1 = params.get("method1", "mp_lower_bulk")
            method2 = params.get("method2", "mp_top20_std")
            noise_type = params.get("noise_type", "laplacian")
            laplacian_scale = params.get("laplacian_scale", 0.1)
            random_seed = params.get("random_seed", None)

            # Step 1: Load images
            self._update_status_ui('status_process', 'Step 1/5: Loading images...')
            self._update_progress_ui('bar_process', 'status_process', 1, 5)

            folder_path = get_random_matrix_folder() / folder_name
            image_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

            if not image_files:
                return {"success": False, "error": "No images found in folder"}

            # Load images
            images = []
            for img_file in image_files:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)

            images = np.stack(images, axis=0)
            n, H, W = images.shape

            # Calculate y parameter
            p = H * W
            y = p / float(n)

            self._update_status_ui('processing_info', f'<strong>Loaded {n} images</strong> (Size: {H}×{W}, Pixels p={p}, y=p/n={y:.4f})')

            # Step 2: Add noise
            seed_text = f", seed={random_seed}" if random_seed is not None else " (random)"

            if noise_type == "laplacian":
                self._update_status_ui('status_process', f'Step 2/5: Adding Laplacian noise (scale={laplacian_scale}{seed_text})...')
                self._update_progress_ui('bar_process', 'status_process', 2, 5)
                noisy_images = add_laplacian_noise(images, scale=laplacian_scale, seed=random_seed)
            elif noise_type == "mixture_gaussian":
                # Get mixture parameters
                weights = params.get("mog_weights", [0.6, 0.3, 0.1])
                means = params.get("mog_means", [0.0, 0.0, 0.0])
                sigmas = params.get("mog_sigmas", [0.1, 0.05, 0.3])
                self._update_status_ui('status_process', f'Step 2/5: Adding Mixture of Gaussians noise ({len(weights)} components{seed_text})...')
                self._update_progress_ui('bar_process', 'status_process', 2, 5)
                noisy_images = add_mixture_gaussian_noise(images, weights, means, sigmas, seed=random_seed)
            else:
                # Default to Laplacian
                self._update_status_ui('status_process', f'Step 2/5: Adding Laplacian noise (scale={laplacian_scale}{seed_text})...')
                self._update_progress_ui('bar_process', 'status_process', 2, 5)
                noisy_images = add_laplacian_noise(images, scale=laplacian_scale, seed=random_seed)

            # Convert to matrix
            X_noisy = images_to_matrix(noisy_images)

            # Step 3: Apply Method 1
            method_names_display = {
                "mp_lower_bulk": "M-P Law (Lower Bulk)",
                "mp_centralized": "M-P Law (Centralized)",
                "mp_top20_std": "M-P Law (Top 20% Std)",
                "gen_lower_bulk": "Gen. Cov (Lower Bulk)",
                "gen_centralized": "Gen. Cov (Centralized)",
                "gen_top20_std": "Gen. Cov (Top 20% Std)",
                "gen_manual": "Gen. Cov (Manual)",
                "gen_centralized_manual": "Gen. Cov (Centralized Manual)",
                "gen_sympy_optimal": "Gen. Cov (Sympy Optimal)"
            }
            method1_name = method_names_display.get(method1, method1)
            self._update_status_ui('status_process', f'Step 3/5: Applying Method 1 ({method1_name})...')
            self._update_progress_ui('bar_process', 'status_process', 3, 5)

            X_method1, info1 = self._apply_method(method1, X_noisy, params, "method1")
            method1_details = self._format_method_details(method1, info1, method1_name)
            self._update_status_ui('method1_info', method1_details)

            # Step 4: Apply Method 2
            method2_name = method_names_display.get(method2, method2)
            self._update_status_ui('status_process', f'Step 4/5: Applying Method 2 ({method2_name})...')
            self._update_progress_ui('bar_process', 'status_process', 4, 5)

            X_method2, info2 = self._apply_method(method2, X_noisy, params, "method2")
            method2_details = self._format_method_details(method2, info2, method2_name)
            self._update_status_ui('method2_info', method2_details)

            # Step 5: Convert back to images
            self._update_status_ui('status_process', 'Step 5/5: Converting results back to images...')
            self._update_progress_ui('bar_process', 'status_process', 5, 5)

            method1_images = matrix_to_images(X_method1, H, W)
            method2_images = matrix_to_images(X_method2, H, W)

            # Create output folders in temp with timestamps
            temp_folder = get_temp_folder()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            noise_folder_name = f"noisy_{noise_type}_{timestamp}"
            noisy_folder = temp_folder / noise_folder_name

            # Map method codes to folder names with timestamps
            method_folder_names = {
                "mp_lower_bulk": "mp_lower_bulk",
                "mp_centralized": "mp_centralized",
                "mp_top20_std": "mp_top20std",
                "gen_lower_bulk": "gen_lower_bulk",
                "gen_centralized": "gen_centralized",
                "gen_top20_std": "gen_top20std",
                "gen_manual": "gen_manual",
                "gen_centralized_manual": "gen_centralized_manual",
                "gen_sympy_optimal": "gen_sympy_optimal"
            }
            method1_folder = temp_folder / f"{method_folder_names.get(method1, method1)}_{timestamp}"
            method2_folder = temp_folder / f"{method_folder_names.get(method2, method2)}_{timestamp}"

            noisy_folder.mkdir(exist_ok=True)
            method1_folder.mkdir(exist_ok=True)
            method2_folder.mkdir(exist_ok=True)

            # Create noise type label
            if noise_type == "laplacian":
                noise_label = f"Laplacian Noise (σ={laplacian_scale})"
            elif noise_type == "mixture_gaussian":
                noise_label = "Mixture of Gaussians Noise"
            else:
                noise_label = "With Noise"

            results = []
            for i in range(n):
                # Create PIL images
                orig_img = Image.fromarray((images[i] * 255).astype(np.uint8), mode='L')
                noisy_img = Image.fromarray((noisy_images[i] * 255).astype(np.uint8), mode='L')
                method1_img = Image.fromarray((np.clip(method1_images[i], 0, 1) * 255).astype(np.uint8), mode='L')
                method2_img = Image.fromarray((np.clip(method2_images[i], 0, 1) * 255).astype(np.uint8), mode='L')

                # Save to disk
                noisy_path = noisy_folder / f"{i:03d}.png"
                method1_path = method1_folder / f"{i:03d}.png"
                method2_path = method2_folder / f"{i:03d}.png"

                noisy_img.save(noisy_path)
                method1_img.save(method1_path)
                method2_img.save(method2_path)

                # Encode to base64
                def img_to_base64(img):
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')

                results.append({
                    "original": img_to_base64(orig_img),
                    "blurred": img_to_base64(noisy_img),
                    "method1": img_to_base64(method1_img),
                    "method2": img_to_base64(method2_img),
                    "method1_name": method1_name,
                    "method2_name": method2_name,
                    "method1_eigvec": info1.get('num_signal', 0),
                    "method2_eigvec": info2.get('num_signal', 0),
                    "noise_type_label": noise_label
                })

            # Store results in instance variable for navigation
            self.processed_results = results

            return {
                "success": True,
                "total_images": len(results),
                "results": results,
                "processing_info": {
                    "n": n,
                    "H": H,
                    "W": W,
                    "p": p,
                    "y": y,
                    "method1": method1_details,
                    "method2": method2_details
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_processed_image(self, index):
        """Get a specific processed image by index"""
        try:
            if hasattr(self, 'processed_results') and index < len(self.processed_results):
                return self.processed_results[index]
            return {"error": "Image not found"}
        except Exception as e:
            return {"error": str(e)}

    def list_random_matrix_folders(self):
        """List all folders in .random_matrix directory"""
        try:
            base_folder = get_random_matrix_folder()
            folders = [f.name for f in base_folder.iterdir() if f.is_dir() and f.name != "temp"]
            return {"folders": sorted(folders)}
        except Exception as e:
            return {"folders": []}

    def get_folder_contents(self, folder_name):
        """Get contents of a specific folder"""
        try:
            folder_path = get_random_matrix_folder() / folder_name
            files = []
            for f in sorted(folder_path.iterdir()):
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    with open(f, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    files.append({"name": f.name, "data": img_data})
            return {"files": files}
        except Exception as e:
            return {"files": []}

    def list_temp_folders(self):
        """List all folders in temp directory with metadata"""
        try:
            temp_folder = get_temp_folder()
            folders_info = []

            if not temp_folder.exists():
                return {"folders": []}

            for folder in sorted(temp_folder.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if folder.is_dir():
                    # Calculate folder size
                    total_size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
                    # Count files
                    file_count = sum(1 for f in folder.rglob('*') if f.is_file())
                    # Get modification time
                    mtime = folder.stat().st_mtime

                    folders_info.append({
                        "name": folder.name,
                        "size_bytes": total_size,
                        "size_mb": round(total_size / (1024 * 1024), 2),
                        "file_count": file_count,
                        "modified_time": mtime,
                        "modified_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })

            return {"folders": folders_info}
        except Exception as e:
            return {"folders": [], "error": str(e)}

    def get_temp_folder_info(self, folder_name):
        """Get detailed info about a specific temp folder"""
        try:
            temp_folder = get_temp_folder()
            folder_path = temp_folder / folder_name

            if not folder_path.exists():
                return {"error": "Folder not found"}

            files = []
            total_size = 0
            for f in sorted(folder_path.glob('*.png')):
                size = f.stat().st_size
                total_size += size
                files.append({
                    "name": f.name,
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 2)
                })

            return {
                "name": folder_name,
                "files": files,
                "total_files": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_temp_folder(self, folder_name):
        """Delete a specific temp folder"""
        try:
            temp_folder = get_temp_folder()
            folder_path = temp_folder / folder_name

            if not folder_path.exists():
                return {"success": False, "error": "Folder not found"}

            # Delete all files in folder
            import shutil
            shutil.rmtree(folder_path)

            return {"success": True, "message": f"Deleted folder: {folder_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_all_temp_folders(self):
        """Delete all folders in temp directory"""
        try:
            temp_folder = get_temp_folder()

            if not temp_folder.exists():
                return {"success": True, "deleted_count": 0, "message": "Temp folder is empty"}

            deleted_count = 0
            import shutil

            for folder in temp_folder.iterdir():
                if folder.is_dir():
                    try:
                        shutil.rmtree(folder)
                        deleted_count += 1
                    except Exception as e:
                        pass

            return {"success": True, "deleted_count": deleted_count, "message": f"Deleted {deleted_count} folders"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_temp_folder(self, folder_name):
        """Open a specific temp folder in file explorer"""
        try:
            temp_folder = get_temp_folder()
            folder_path = temp_folder / folder_name

            if not folder_path.exists():
                return {"success": False, "error": "Folder not found"}

            # Open folder in file explorer (cross-platform)
            import subprocess
            import sys

            if sys.platform == 'win32':
                # Windows
                subprocess.Popen(['explorer', str(folder_path)])
            elif sys.platform == 'darwin':
                # macOS
                subprocess.Popen(['open', str(folder_path)])
            else:
                # Linux
                subprocess.Popen(['xdg-open', str(folder_path)])

            return {"success": True, "message": f"Opened folder: {folder_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# --------- HTML / JS UI (desktop) --------- #
HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Matrix Analysis Lab</title>
  <script src="https://cdn.plot.ly/plotly-2.27.1.min.js"></script>
  <style>
    :root {
      --bg: #0c1224;
      --panel: #0f172a;
      --card: #ffffff;
      --border: #d9e1ef;
      --accent: #3b82f6;
      --accent2: #60a5fa;
      --text: #0f172a;
      --muted: #5b6475;
      --status: #0ea5e9;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
      background: radial-gradient(circle at 20% 15%, rgba(255,255,255,0.10), transparent 30%), linear-gradient(135deg, #0c1224, #0f172a 50%, #0b1020);
      color: #e2e8f0;
      overflow-x: hidden;
      scroll-behavior: smooth;
    }
    header {
      padding: 18px 24px;
      background: rgba(12,18,36,0.8);
      box-shadow: 0 6px 24px rgba(0,0,0,0.35);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: 24px; color: #fff; font-weight: 700; }
    p { margin: 6px 0 0; color: #cbd5e1; font-size: 14px; }
    .tabs { display: flex; gap: 8px; padding: 12px 16px 0; flex-wrap: wrap; background: var(--bg); }
    .tab {
      padding: 12px 18px;
      border-radius: 10px 10px 0 0;
      background: var(--panel);
      color: #cbd5e1;
      border: 1px solid #1e293b;
      border-bottom: none;
      cursor: pointer;
      user-select: none;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
      transition: all 0.2s ease;
      font-weight: 500;
    }
    .tab:hover {
      background: rgba(255,255,255,0.05);
      color: #e2e8f0;
    }
    .tab.active {
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      border-bottom: 1px solid var(--card);
      box-shadow: 0 -2px 14px rgba(15,23,42,0.25);
      transform: translateY(1px);
    }
    .tab-content { display: none; padding: 0; }
    .tab-content.active { display: block; }
    .container { padding: 20px; max-width: 1400px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 20px; }
    .grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
    .grid-full { grid-column: 1 / -1; }
    .card {
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 4px 12px rgba(15,23,42,0.15), 0 1px 3px rgba(15,23,42,0.1);
      padding: 20px;
      transition: all 0.3s ease;
      animation: fadeIn 0.5s ease;
    }
    .card:hover {
      box-shadow: 0 8px 20px rgba(15,23,42,0.2), 0 2px 6px rgba(15,23,42,0.15);
    }
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    .controls-panel {
      position: sticky;
      top: 90px;
      height: fit-content;
    }
    #result1:hover, #result2:hover, #result_original:hover, #result_blur:hover {
      border-color: #3b82f6 !important;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
      transform: scale(1.02);
    }
    .plot-area {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .section-title {
      font-size: 12px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin: 24px 0 12px 0;
      padding-bottom: 8px;
      border-bottom: 2px solid #e2e8f0;
      position: relative;
    }
    .section-title:first-of-type {
      margin-top: 16px;
    }
    h3 {
      margin: 0 0 16px;
      font-size: 18px;
      font-weight: 600;
      color: var(--text);
      padding-bottom: 12px;
      border-bottom: 1px solid #e2e8f0;
    }
    label { display: block; font-size: 13px; margin: 10px 0 5px; color: #475569; font-weight: 500; }
    input {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      font-size: 14px;
      transition: all 0.2s ease;
      box-sizing: border-box;
    }
    input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(255,106,61,0.1);
    }
    button {
      margin-top: 16px;
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: white;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(59,130,246,0.3);
      letter-spacing: 0.3px;
      font-size: 14px;
      transition: all 0.2s ease;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(59,130,246,0.4);
    }
    button:active {
      transform: translateY(0px);
    }
    .plot-box {
      margin-top: 0;
      background: #f8fafc;
      border-radius: 12px;
      padding: 0;
      border: 2px dashed #cbd5e1;
      min-height: 400px;
      height: 450px;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
      color: #94a3b8;
      font-size: 14px;
      overflow: hidden;
      position: relative;
      transition: all 0.3s ease;
    }
    .plot-box.empty {
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 500;
    }
    .plot-box.has-plot {
      border: 1px solid #cbd5e1;
      background: #ffffff;
      padding: 0;
    }
    .plot-box:hover {
      border-color: #94a3b8;
      background: #ffffff;
    }
    .plot-box > div {
      width: 100%;
      height: 100%;
    }
    .plot-box .main-svg {
      border-radius: 10px;
    }
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 0;
    }
    .metric {
      background: linear-gradient(135deg, #0f172a, #1e293b);
      color: #a5b4fc;
      padding: 12px;
      border-radius: 10px;
      font-size: 12px;
      border: 1px solid #334155;
      font-weight: 600;
      text-align: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metric:hover {
      background: linear-gradient(135deg, #1e293b, #334155);
      transform: translateY(-1px);
      transition: all 0.2s ease;
    }
    .small {
      font-size: 12px;
      color: #64748b;
      margin-top: 12px;
      line-height: 1.6;
      padding: 10px;
      background: #f8fafc;
      border-radius: 6px;
      border-left: 3px solid #94a3b8;
    }
    .status { color: var(--status); margin-top: 10px; font-size: 13px; min-height: 18px; font-weight: 500; }
    .explain {
      color: #475569;
      margin-top: 10px;
      padding: 10px;
      font-size: 13px;
      line-height: 1.6;
      background: #f1f5f9;
      border-radius: 8px;
      border-left: 3px solid var(--accent);
    }
    .bar-wrap {
      margin-top: 8px;
      background: #e2e8f0;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      height: 12px;
      overflow: hidden;
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
    }
    .bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #22d3ee, #38bdf8);
      transition: width 0.3s ease;
      box-shadow: 0 0 10px rgba(34,211,238,0.5);
    }
    .equation-panel {
      margin-bottom: 16px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      overflow: hidden;
      background: #f8fafc;
    }
    .equation-panel summary {
      padding: 12px 16px;
      cursor: pointer;
      user-select: none;
      font-weight: 600;
      font-size: 14px;
      color: #1e293b;
      background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
      transition: background 0.2s ease;
    }
    .equation-panel summary:hover {
      background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
    }
    .equation-content {
      padding: 16px;
      border-top: 1px solid #cbd5e1;
      background: white;
    }
    .equation-content h4 {
      margin: 12px 0 8px 0;
      font-size: 13px;
      font-weight: 600;
      color: #334155;
      border-bottom: none;
      padding-bottom: 4px;
    }
    .equation-content h4:first-child {
      margin-top: 0;
    }
    .equation-content code {
      display: block;
      padding: 8px 12px;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 12px;
      color: #0f172a;
      margin: 8px 0;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .equation-content p {
      margin: 6px 0;
      font-size: 13px;
      color: #475569;
      line-height: 1.6;
    }
    @media (max-width: 1024px) {
      .grid { grid-template-columns: 1fr; }
      .grid-2col { grid-template-columns: 1fr; }
      .controls-panel { position: static; }
      .container { padding: 12px; }
    }
    @media (max-width: 768px) {
      h1 { font-size: 20px; }
      .tabs { padding: 8px 12px 0; }
      .tab { padding: 10px 14px; font-size: 13px; }
      .card { padding: 16px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Matrix Analysis Lab</h1>
    <p>Desktop webview UI (HTML/CSS/JS) powered by pure Python computations.</p>
  </header>

  <div class="tabs">
    <div class="tab active" data-tab="tab-im">Im(s) vs z</div>
    <div class="tab" data-tab="tab-beta">Roots vs beta</div>
    <div class="tab" data-tab="tab-eig">Eigenvalue distribution</div>
    <div class="tab" data-tab="tab-folder-management">📁 Folder Management</div>
    <div class="tab" data-tab="tab-process">🖼️ Image Processing</div>
    <div class="tab" data-tab="tab-temp-files">🗂️ Output Files</div>
  </div>

  <div id="tab-im" class="tab-content active">
    <div class="container">
      <div class="grid">
        <!-- Controls Panel -->
        <div class="card controls-panel">
          <h3>⚙️ Parameters</h3>
          <label>Beta (β)</label>
          <input id="beta_z" type="number" value="0.5" step="0.05" min="0" max="1">

          <label>Y ratio</label>
          <input id="y_z" type="number" value="1.0" step="0.1" min="0.01">

          <label>Spike value (a)</label>
          <input id="a_z" type="number" value="2.0" step="0.1" min="0.01">

          <div class="section-title">Z Range</div>
          <label>Minimum</label>
          <input id="zmin_z" type="number" value="0.01" step="0.01">

          <label>Maximum</label>
          <input id="zmax_z" type="number" value="10" step="0.5">

          <div class="section-title">Computation</div>
          <label>Grid points</label>
          <input id="points_z" type="number" value="400" step="50" min="50" max="2000">

          <button onclick="runIm()">🚀 Compute Im(s) vs z</button>
          <div class="small">Solves cubic equation on logarithmic z grid.</div>
        </div>

        <!-- Plot Area -->
        <div class="plot-area">
          <div class="card">
            <details class="equation-panel">
              <summary>📐 Cubic Equation & Method</summary>
              <div class="equation-content">
                <h4>Cubic Polynomial Equation:</h4>
                <code>(z·a)·s³ + [z·(a+1) + a·(1-y)]·s² + [z + (a+1) - y - y·β·(a-1)]·s + 1 = 0</code>

                <h4>Stieltjes Transform:</h4>
                <p>The roots <code>s</code> represent the Stieltjes transform of the eigenvalue distribution:</p>
                <code>s(z) = ∫ (1/(λ - z)) dμ(λ)</code>

                <h4>Im(s) Interpretation:</h4>
                <p>• <code>Im(s)</code> relates to the density of eigenvalues</p>
                <p>• <code>Re(s)</code> relates to the principal value</p>
                <p>• Complex roots indicate overlapping eigenvalue support</p>
                <p>• Real roots indicate separated eigenvalue bands</p>
              </div>
            </details>
            <div class="plot-box empty" id="slot1">Click "Compute" to generate plot</div>
            <div class="status" id="status1"></div>
            <div class="bar-wrap"><div class="bar" id="bar1"></div></div>
            <div class="explain" id="explain1"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-beta" class="tab-content">
    <div class="container">
      <div class="grid">
        <!-- Controls Panel -->
        <div class="card controls-panel">
          <h3>⚙️ Parameters</h3>
          <label>Z position</label>
          <input id="z_beta" type="number" value="1.0" step="0.1">

          <label>Y ratio</label>
          <input id="y_beta" type="number" value="1.0" step="0.1">

          <label>Spike value (a)</label>
          <input id="a_beta" type="number" value="2.0" step="0.1">

          <div class="section-title">Beta Range</div>
          <label>Minimum</label>
          <input id="beta_min" type="number" value="0.0" step="0.05" min="0" max="1">

          <label>Maximum</label>
          <input id="beta_max" type="number" value="1.0" step="0.05" min="0" max="1">

          <div class="section-title">Computation</div>
          <label>Grid points</label>
          <input id="points_beta" type="number" value="400" step="50" min="50" max="2000">

          <button onclick="runBeta()">🚀 Generate Root Plots</button>
          <div class="small">Tracks root evolution across beta values.</div>

          <div class="status" id="status2"></div>
          <div class="bar-wrap"><div class="bar" id="bar2"></div></div>
        </div>

        <!-- Plot Area -->
        <div class="plot-area">
          <div class="card">
            <details class="equation-panel">
              <summary>📐 Cubic Equation & Discriminant</summary>
              <div class="equation-content">
                <h4>Same Cubic Equation:</h4>
                <code>(z·a)·s³ + [z·(a+1) + a·(1-y)]·s² + [z + (a+1) - y - y·β·(a-1)]·s + 1 = 0</code>

                <h4>Variable Parameter: β (beta)</h4>
                <p>This tab shows how the roots change as β varies (while z, a, y are fixed)</p>

                <h4>Discriminant:</h4>
                <code>Δ = 18abc·d - 27a²d² + b²c² - 2b³d - 9ac³</code>
                <p>• Discriminant crosses zero at transitions between 3 real roots and 1 real + 2 complex roots</p>
                <p>• Red dashed lines mark discriminant zero crossings</p>
              </div>
            </details>
          </div>
          <div class="grid-2col">
            <div class="card">
              <h3>📊 Imaginary Parts</h3>
              <div class="plot-box empty" id="slot2">Click "Generate" to create plots</div>
            </div>
            <div class="card">
              <h3>📈 Real Parts</h3>
              <div class="plot-box empty" id="slot3">Waiting for computation...</div>
            </div>
          </div>
          <div class="card">
            <h3>🔍 Cubic Discriminant</h3>
            <div class="plot-box empty" id="slot4">Waiting for computation...</div>
            <div class="explain" id="explain2"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-eig" class="tab-content">
    <div class="container">
      <div class="grid">
        <!-- Controls Panel -->
        <div class="card controls-panel">
          <h3>⚙️ Parameters</h3>
          <label>Beta (β)</label>
          <input id="beta_eig" type="number" value="0.5" step="0.05" min="0" max="1">

          <label>Spike value (a)</label>
          <input id="a_eig" type="number" value="2.0" step="0.1">

          <div class="section-title">Matrix Dimensions</div>
          <label>Samples (n)</label>
          <input id="n_eig" type="number" value="400" step="50" min="50" max="4000">

          <label>Rows (p)</label>
          <input id="p_eig" type="number" value="200" step="20" min="50" max="4000">

          <div class="section-title">Random Seed</div>
          <label>Seed</label>
          <input id="seed_eig" type="number" value="42" step="1" min="1">

          <button onclick="runEigen()">🚀 Generate Distribution</button>
          <div class="small">Simulates random matrix eigenvalues with mixed diagonal entries.</div>

          <div class="section-title">Statistics</div>
          <div id="eig_stats" class="metrics"></div>

          <div class="status" id="status4"></div>
          <div class="bar-wrap"><div class="bar" id="bar4"></div></div>
        </div>

        <!-- Plot Area -->
        <div class="plot-area">
          <div class="card">
            <details class="equation-panel">
              <summary>📐 Matrix Construction & Theory</summary>
              <div class="equation-content">
                <h4>Matrix Model:</h4>
                <code>B_n = S_n × T_n</code>

                <h4>Sample Covariance Matrix:</h4>
                <code>S_n = (1/n) × X × X^T</code>
                <p>Where X is a (p × n) random matrix with Gaussian entries</p>

                <h4>Spike Matrix:</h4>
                <code>T_n = diag([a, a, ..., a, 1, 1, ..., 1])</code>
                <p>• First β×p entries are <code>a</code> (spike values)</p>
                <p>• Remaining (1-β)×p entries are <code>1</code></p>

                <h4>Parameters:</h4>
                <p>• <code>y = p/n</code>: Aspect ratio (calculated from image dimensions)</p>
                <p>• <code>a</code>: Spike value (default: 2.0)</p>
                <p>• <code>β</code>: Fraction of spiked eigenvalues</p>
                <p>• <code>n</code>: Number of samples (images)</p>
                <p>• <code>p</code>: Dimension (image pixels)</p>
              </div>
            </details>
            <h3>📊 Eigenvalue Distribution</h3>
            <div class="plot-box empty" id="slot6">Click "Generate" to create distribution</div>
            <div class="explain" id="explain4"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-folder-management" class="tab-content">
    <div class="container">
      <div class="grid">
        <!-- Controls Panel -->
        <div class="card controls-panel" style="max-height: 90vh; overflow-y: auto;">
          <h3>📁 Folder Management</h3>

          <!-- Create Folder Section -->
          <div class="section-title">➕ Create New Folder</div>
          <label>Folder Name</label>
          <input id="new_folder_name" type="text" placeholder="Enter folder name" onkeypress="if(event.key==='Enter')createNewFolder()" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
          <button onclick="createNewFolder()" style="margin-top: 12px; width: 100%;">➕ Create Folder</button>
          <div class="small" style="margin-top: 8px; color: #64748b;">Create a new empty folder in .random_matrix</div>
          <div class="status" id="status_create"></div>
          <div class="bar-wrap"><div class="bar" id="bar_create"></div></div>

          <!-- Upload Folder Section -->
          <div class="section-title" style="margin-top: 24px; padding-top: 20px; border-top: 2px solid #e2e8f0;">📂 Upload Entire Folder</div>
          <label>Select Folder from Your Computer</label>
          <input id="local_folder" type="file" webkitdirectory directory multiple style="display:none;">
          <button onclick="document.getElementById('local_folder').click()" style="margin-top: 8px; width: 100%;">📂 Browse Folder</button>
          <div class="small" id="folder_count" style="margin-top: 8px; padding: 8px; background: #f1f5f9; border-radius: 4px; min-height: 40px;">No folder selected</div>
          <button onclick="importLocalFolder()" style="margin-top: 12px; width: 100%;">📥 Upload Folder</button>
          <div class="small" style="margin-top: 8px; color: #64748b;">Copies entire folder structure to .random_matrix</div>
          <div class="status" id="status_upload_folder"></div>
          <div class="bar-wrap"><div class="bar" id="bar_upload_folder"></div></div>

          <!-- Upload Files Section -->
          <div class="section-title" style="margin-top: 24px; padding-top: 20px; border-top: 2px solid #e2e8f0;">🖼️ Upload Files to Folder</div>
          <label>Select Destination Folder</label>
          <select id="upload_dest_folder" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; margin-top: 8px;">
            <option>Loading folders...</option>
          </select>
          <label style="margin-top: 12px;">Select Image Files</label>
          <input id="local_files" type="file" accept="image/*" multiple style="display:none;">
          <button onclick="document.getElementById('local_files').click()" style="margin-top: 8px; width: 100%;">🖼️ Browse Files</button>
          <div class="small" id="files_count" style="margin-top: 8px; padding: 8px; background: #f1f5f9; border-radius: 4px; min-height: 40px;">No files selected</div>
          <button onclick="uploadFilesToFolder()" style="margin-top: 12px; width: 100%;">📤 Upload Files</button>
          <div class="small" style="margin-top: 8px; color: #64748b;">Upload multiple images to selected folder</div>
          <div class="status" id="status_upload_files"></div>
          <div class="bar-wrap"><div class="bar" id="bar_upload_files"></div></div>
        </div>

        <!-- Main Area -->
        <div class="plot-area">
          <div class="card">
            <h3>📂 Browse & Manage Folders</h3>
            <div style="display: grid; grid-template-columns: 1fr 3fr; gap: 20px; margin-top: 16px;">
              <!-- Folder List Sidebar -->
              <div style="display: flex; flex-direction: column;">
                <label style="font-weight: 600; margin-bottom: 8px;">Available Folders</label>
                <select id="folder_list" size="10" ondblclick="viewFolder()" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; flex-grow: 1;">
                  <option>Loading folders...</option>
                </select>
                <div style="margin-top: 12px; display: grid; gap: 8px;">
                  <button onclick="refreshFolders()" style="width: 100%; padding: 10px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;">🔄 Refresh</button>
                  <button onclick="viewFolder()" style="width: 100%; padding: 10px; background: #10b981; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;">👁️ View Contents</button>
                  <button onclick="deleteFolder()" style="width: 100%; padding: 10px; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;">🗑️ Delete Folder</button>
                </div>
              </div>

              <!-- Folder Contents Area -->
              <div style="display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                  <h4 style="margin: 0; font-size: 14px; color: #475569;">Folder Contents</h4>
                  <div id="folder_info" style="font-size: 12px; color: #64748b;"></div>
                </div>
                <div id="folder_contents" style="min-height: 350px; max-height: 450px; overflow-y: auto; padding: 16px; background: #f8fafc; border-radius: 8px; border: 1px solid #cbd5e1;">
                  <p style="text-align: center; color: #94a3b8; padding: 60px 20px;">
                    <span style="font-size: 48px; display: block; margin-bottom: 12px;">📁</span>
                    Select a folder to view its contents<br>
                    <span style="font-size: 11px;">(Double-click or use View Contents button)</span>
                  </p>
                </div>
              </div>
            </div>
            <div class="status" id="status_browse" style="margin-top: 16px;"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-process" class="tab-content">
    <div class="container">
      <div class="grid">
        <!-- Controls Panel -->
        <div class="card controls-panel">
          <h3>🖼️ Image Processing</h3>

          <!-- Source Folder Selection -->
          <div style="background: #f8fafc; padding: 14px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0;">
            <div class="section-title" style="margin: 0 0 10px 0;">📁 Source Folder</div>
            <select id="process_folder" style="width: 100%; padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; background: white;">
              <option>Loading folders...</option>
            </select>
          </div>

          <!-- Denoising Methods -->
          <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 16px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #bae6fd;">
            <div style="font-weight: 600; color: #0c4a6e; margin-bottom: 14px; font-size: 14px;">🔬 Denoising Methods</div>

            <!-- Method 1 -->
            <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
              <label style="font-weight: 600; color: #334155; font-size: 12px; display: block; margin-bottom: 6px;">Method 1</label>
              <select id="method1" onchange="toggleMethodParams()" style="width: 100%; padding: 9px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
                <option value="mp_lower_bulk">M-P Law (Lower Bulk)</option>
                <option value="gen_lower_bulk">Gen. Cov (Lower Bulk)</option>
                <option value="gen_manual">Gen. Cov (Manual)</option>
                <option value="gen_sympy_optimal">Gen. Cov (Sympy Optimal)</option>
              </select>

              <div id="method1_params" style="display: none; background: #fffbeb; padding: 12px; border-radius: 6px; margin-top: 10px; border: 1px solid #fde68a;">
                <div style="font-weight: 600; color: #78350f; margin-bottom: 8px; font-size: 11px;">Manual Parameters</div>
                <div id="method1_centralized_note" style="display: none; background: #dbeafe; padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: 10px; color: #1e40af;">
                  ℹ️ For Centralized Manual: σ² is auto-computed from centered data. Only a and β are manual inputs.
                </div>
                <div id="method1_sympy_note" style="display: none; background: #dcfce7; padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: 10px; color: #166534;">
                  ℹ️ For Sympy Optimal: σ² is auto-optimized using symbolic math to match bulk_min. Input a and β manually.
                </div>
                <div id="method1_sigma2_container">
                  <label style="font-size: 11px;">σ² (sigma squared)</label>
                  <input id="method1_sigma2" type="number" value="0.01" step="0.001" min="0.001">
                </div>
                <label style="font-size: 11px;">a (spike value, > 1)</label>
                <input id="method1_a" type="number" value="2.0" step="0.1" min="1.01">
                <label style="font-size: 11px;">β (beta, 0 to 1)</label>
                <input id="method1_beta" type="number" value="0.5" step="0.05" min="0.01" max="0.99">
              </div>
            </div>

            <!-- Method 2 -->
            <div style="background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
              <label style="font-weight: 600; color: #334155; font-size: 12px; display: block; margin-bottom: 6px;">Method 2</label>
              <select id="method2" onchange="toggleMethodParams()" style="width: 100%; padding: 9px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
                <option value="mp_lower_bulk">M-P Law (Lower Bulk)</option>
                <option value="gen_lower_bulk" selected>Gen. Cov (Lower Bulk)</option>
                <option value="gen_manual">Gen. Cov (Manual)</option>
                <option value="gen_sympy_optimal">Gen. Cov (Sympy Optimal)</option>
              </select>

              <div id="method2_params" style="display: none; background: #fffbeb; padding: 12px; border-radius: 6px; margin-top: 10px; border: 1px solid #fde68a;">
                <div style="font-weight: 600; color: #78350f; margin-bottom: 8px; font-size: 11px;">Manual Parameters</div>
                <div id="method2_centralized_note" style="display: none; background: #dbeafe; padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: 10px; color: #1e40af;">
                  ℹ️ For Centralized Manual: σ² is auto-computed from centered data. Only a and β are manual inputs.
                </div>
                <div id="method2_sympy_note" style="display: none; background: #dcfce7; padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: 10px; color: #166534;">
                  ℹ️ For Sympy Optimal: σ² is auto-optimized using symbolic math to match bulk_min. Input a and β manually.
                </div>
                <div id="method2_sigma2_container">
                  <label style="font-size: 11px;">σ² (sigma squared)</label>
                  <input id="method2_sigma2" type="number" value="0.01" step="0.001" min="0.001">
                </div>
                <label style="font-size: 11px;">a (spike value, > 1)</label>
                <input id="method2_a" type="number" value="2.0" step="0.1" min="1.01">
                <label style="font-size: 11px;">β (beta, 0 to 1)</label>
                <input id="method2_beta" type="number" value="0.5" step="0.05" min="0.01" max="0.99">
              </div>
            </div>
          </div>

          <!-- Noise Configuration -->
          <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 16px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #d8b4fe;">
            <div style="font-weight: 600; color: #581c87; margin-bottom: 12px; font-size: 14px;">🎲 Noise Configuration</div>
            <label style="font-size: 12px; font-weight: 500; color: #4c1d95;">Noise Type</label>
            <select id="noise_type" onchange="toggleNoiseParams()" style="width: 100%; padding: 9px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; background: white; margin-bottom: 12px;">
              <option value="laplacian">Laplacian (heavy-tailed)</option>
              <option value="mixture_gaussian">Mixture of Gaussians</option>
            </select>

            <!-- Laplacian Noise Parameters -->
            <div id="laplacian_params">
              <label style="font-size: 12px; font-weight: 500; color: #4c1d95;">Scale (σ)</label>
              <input id="laplacian_scale" type="number" value="0.1" step="0.01" min="0.01" max="1.0" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; margin-bottom: 6px;">
              <div class="small" style="color: #6b21a8; font-size: 11px;">Heavy-tailed distribution (default: 0.1)</div>
            </div>

            <!-- Mixture of Gaussians Parameters -->
            <div id="mog_params" style="display: none;">
              <div class="small" style="color: #6b21a8; margin-bottom: 10px; font-size: 11px;">Configure 3 Gaussian components (weights auto-normalized)</div>

              <div style="background: white; padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #e9d5ff;">
                <div style="font-weight: 600; color: #4c1d95; margin-bottom: 6px; font-size: 11px;">Component 1</div>
                <label style="font-size: 11px;">Weight</label>
                <input id="mog_weight1" type="number" value="0.6" step="0.1" min="0.01" max="1.0" style="margin-bottom: 4px;">
                <label style="font-size: 11px;">Mean (μ)</label>
                <input id="mog_mean1" type="number" value="0.0" step="0.01" min="-0.5" max="0.5" style="margin-bottom: 4px;">
                <label style="font-size: 11px;">Sigma (σ)</label>
                <input id="mog_sigma1" type="number" value="0.1" step="0.01" min="0.001" max="1.0">
              </div>

              <div style="background: white; padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #e9d5ff;">
                <div style="font-weight: 600; color: #4c1d95; margin-bottom: 6px; font-size: 11px;">Component 2</div>
                <label style="font-size: 11px;">Weight</label>
                <input id="mog_weight2" type="number" value="0.3" step="0.1" min="0.01" max="1.0" style="margin-bottom: 4px;">
                <label style="font-size: 11px;">Mean (μ)</label>
                <input id="mog_mean2" type="number" value="0.0" step="0.01" min="-0.5" max="0.5" style="margin-bottom: 4px;">
                <label style="font-size: 11px;">Sigma (σ)</label>
                <input id="mog_sigma2" type="number" value="0.05" step="0.01" min="0.001" max="1.0">
              </div>

              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #e9d5ff;">
                <div style="font-weight: 600; color: #4c1d95; margin-bottom: 6px; font-size: 11px;">Component 3 (Heavy tail)</div>
                <label style="font-size: 11px;">Weight</label>
                <input id="mog_weight3" type="number" value="0.1" step="0.1" min="0.01" max="1.0" style="margin-bottom: 4px;">
                <label style="font-size: 11px;">Mean (μ)</label>
                <input id="mog_mean3" type="number" value="0.0" step="0.01" min="-0.5" max="0.5" style="margin-bottom: 4px;">
                <label style="font-size: 11px;">Sigma (σ)</label>
                <input id="mog_sigma3" type="number" value="0.3" step="0.01" min="0.001" max="1.0">
              </div>
            </div>

            <label style="font-size: 12px; font-weight: 500; color: #4c1d95; margin-top: 12px; display: block;">Random Seed</label>
            <input id="random_seed" type="number" value="" placeholder="Leave empty for random" step="1" min="0" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px;">
            <div class="small" style="color: #6b21a8; margin-top: 4px; font-size: 11px;">Set seed for reproducibility (e.g., 42) or leave empty</div>
          </div>

          <!-- Process Button -->
          <button onclick="processImages()" style="width: 100%; padding: 14px; font-size: 15px; font-weight: 600; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; border-radius: 8px; cursor: pointer; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2); transition: all 0.2s; margin-top: 20px;">🚀 Process Images</button>
          <div class="small" style="text-align: center; color: #64748b; margin-top: 8px;">y parameter auto-calculated from image dimensions</div>

          <div class="status" id="status_process" style="margin-top: 12px;"></div>
          <div class="bar-wrap"><div class="bar" id="bar_process"></div></div>

          <!-- Processing Details -->
          <div style="margin-top: 24px; background: #f8fafc; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="font-weight: 600; color: #334155; margin-bottom: 10px; font-size: 13px;">📋 Processing Details</div>
            <div id="processing_info" style="background: white; padding: 10px; border-radius: 6px; min-height: 40px; font-size: 12px; color: #475569; margin-bottom: 8px; border: 1px solid #e2e8f0;">
              Processing info will appear here...
            </div>
            <div id="method1_info" style="background: white; padding: 9px; border-radius: 6px; min-height: 30px; font-size: 11px; color: #64748b; margin-bottom: 6px; border: 1px solid #e2e8f0;">
              Method 1 details...
            </div>
            <div id="method2_info" style="background: white; padding: 9px; border-radius: 6px; min-height: 30px; font-size: 11px; color: #64748b; border: 1px solid #e2e8f0;">
              Method 2 details...
            </div>
          </div>
        </div>

        <!-- Display Area with 4-panel grid -->
        <div class="plot-area">
          <div class="card">
            <h3>📊 Comparison Results</h3>
            <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #93c5fd; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">💡</span>
                <span style="color: #1e40af; font-weight: 500; font-size: 13px;">Click on any image to view its eigenvalue distribution below</span>
              </div>
            </div>
            <div class="grid-2col" style="margin-top: 0; gap: 16px;">
              <div style="text-align: center;">
                <h4 id="result1_label" style="margin: 0 0 10px 0; font-size: 13px; color: #475569; font-weight: 600;">Method 1 Result</h4>
                <div id="result1" onclick="showImageEigenvalues('result1')" style="width: 100%; height: 300px; background: #f8fafc; border: 2px solid #cbd5e1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #94a3b8; cursor: pointer; transition: all 0.2s; position: relative; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05);" onmouseenter="this.style.borderColor='#3b82f6'; this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.3)'; this.style.transform='translateY(-2px)';" onmouseleave="this.style.borderColor='#cbd5e1'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)';">
                  Click "Process Images" to start
                </div>
              </div>
              <div style="text-align: center;">
                <h4 id="result2_label" style="margin: 0 0 10px 0; font-size: 13px; color: #475569; font-weight: 600;">Method 2 Result</h4>
                <div id="result2" onclick="showImageEigenvalues('result2')" style="width: 100%; height: 300px; background: #f8fafc; border: 2px solid #cbd5e1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #94a3b8; cursor: pointer; transition: all 0.2s; position: relative; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05);" onmouseenter="this.style.borderColor='#3b82f6'; this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.3)'; this.style.transform='translateY(-2px)';" onmouseleave="this.style.borderColor='#cbd5e1'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)';">
                  Click "Process Images" to start
                </div>
              </div>
              <div style="text-align: center;">
                <h4 style="margin: 0 0 10px 0; font-size: 13px; color: #475569; font-weight: 600;">Original Image</h4>
                <div id="result_original" onclick="showImageEigenvalues('result_original')" style="width: 100%; height: 300px; background: #f8fafc; border: 2px solid #cbd5e1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #94a3b8; cursor: pointer; transition: all 0.2s; position: relative; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05);" onmouseenter="this.style.borderColor='#3b82f6'; this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.3)'; this.style.transform='translateY(-2px)';" onmouseleave="this.style.borderColor='#cbd5e1'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)';">
                  Waiting...
                </div>
              </div>
              <div style="text-align: center;">
                <h4 id="result_blur_label" style="margin: 0 0 10px 0; font-size: 13px; color: #475569; font-weight: 600;">With Noise</h4>
                <div id="result_blur" onclick="showImageEigenvalues('result_blur')" style="width: 100%; height: 300px; background: #f8fafc; border: 2px solid #cbd5e1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #94a3b8; cursor: pointer; transition: all 0.2s; position: relative; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05);" onmouseenter="this.style.borderColor='#3b82f6'; this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.3)'; this.style.transform='translateY(-2px)';" onmouseleave="this.style.borderColor='#cbd5e1'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)';">
                  Waiting...
                </div>
              </div>
            </div>

            <div style="text-align: center; margin-top: 20px; display: flex; align-items: center; justify-content: center; gap: 16px;">
              <button onclick="prevImage()" style="width: 60px; padding: 10px; font-size: 18px;">&lt;</button>
              <div style="font-size: 20px; font-weight: 600; color: #0f172a; font-family: 'Courier New', monospace;" id="image_counter">000</div>
              <button onclick="nextImage()" style="width: 60px; padding: 10px; font-size: 18px;">&gt;</button>
            </div>

            <!-- Eigenvalue Distribution Visualization -->
            <div id="eigenvalue_section" style="margin-top: 30px; padding-top: 24px; border-top: 2px solid #e2e8f0; display: none;">
              <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 2px solid #86efac; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(34, 197, 94, 0.1);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                  <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: #22c55e; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 2px 4px rgba(34, 197, 94, 0.3);">📈</div>
                    <h3 style="margin: 0; color: #166534; font-size: 18px;">Eigenvalue Distribution Analysis</h3>
                  </div>
                  <button onclick="toggleEigenvalueList()" style="padding: 8px 16px; background: #10b981; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);" onmouseenter="this.style.background='#059669'" onmouseleave="this.style.background='#10b981'">📋 Show All Values</button>
                </div>
                <div id="eigenvalue_title" style="font-size: 14px; color: #15803d; font-weight: 500; padding-left: 52px;">
                  Click an image above to view its eigenvalue distribution
                </div>
              </div>

              <div style="background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <div id="eigenvalue_plot" style="width: 100%; height: 450px; border-radius: 8px;"></div>
              </div>

              <div id="eigenvalue_stats" style="margin-bottom: 20px; padding: 16px; background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); border-radius: 10px; font-size: 13px; color: #713f12; border: 1px solid #fde047; box-shadow: 0 2px 4px rgba(250, 204, 21, 0.1);"></div>

              <!-- Eigenvalue List Container -->
              <div id="eigenvalue_list_container" style="display: none; background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                  <h4 style="margin: 0; color: #0f172a; font-size: 16px;">📊 All Eigenvalues (Sorted by Magnitude)</h4>
                  <button onclick="downloadEigenvalues()" style="padding: 6px 12px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 12px;" onmouseenter="this.style.background='#2563eb'" onmouseleave="this.style.background='#3b82f6'">💾 Download CSV</button>
                </div>
                <div id="eigenvalue_list" style="max-height: 400px; overflow-y: auto; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.8;"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-temp-files" class="tab-content">
    <div class="container">
      <div class="card">
        <h3>🗂️ Output Files Management</h3>
        <p class="small">Manage timestamped output folders in the temp directory. Each denoising operation creates timestamped folders that are preserved for your review.</p>

        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
          <div style="font-size: 13px; color: #0c4a6e;">
            <strong>💡 Tip:</strong> Output folders are automatically timestamped (YYYYMMDD_HHMMSS) and never auto-deleted.
            Use this tab to manage them manually.
          </div>
        </div>

        <div style="display: flex; gap: 10px; margin-bottom: 20px;">
          <button onclick="refreshTempFiles()" style="flex: 1;">🔄 Refresh List</button>
          <button onclick="deleteAllTempFiles()" style="flex: 1; background: #dc2626;">🗑️ Delete All</button>
        </div>

        <div id="temp_files_status" style="margin-bottom: 10px; min-height: 24px; font-weight: 500;"></div>

        <div id="temp_files_list" style="max-height: 600px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; background: #f8fafc;">
          <div style="text-align: center; color: #94a3b8; padding: 40px;">
            Click "🔄 Refresh List" to load temp folders
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        contents.forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
      });
    });

    function setHTML(slot, html) { document.getElementById(slot).innerHTML = html || ''; }
    // Track resize handlers to avoid duplicates
    const resizeHandlers = new Map();

    function plotFromFigure(target, fig) {
      const el = document.getElementById(target);
      if (!el) return;

      // Remove empty class and add has-plot class
      el.classList.remove('empty');
      el.classList.add('has-plot');
      el.innerHTML = '';

      const layout = fig.layout || {};
      layout.autosize = true;
      layout.height = 450;
      layout.margin = layout.margin || {};
      Object.assign(layout.margin, { l: 60, r: 40, t: 60, b: 50 });

      const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
      };

      Plotly.newPlot(target, fig.data, layout, config);

      // Remove old resize handler if exists
      if (resizeHandlers.has(target)) {
        window.removeEventListener('resize', resizeHandlers.get(target));
      }

      // Create new resize handler
      const resizeHandler = () => {
        const plotDiv = document.getElementById(target);
        if (plotDiv && plotDiv.data) {
          Plotly.Plots.resize(target);
        }
      };

      resizeHandlers.set(target, resizeHandler);
      window.addEventListener('resize', resizeHandler);
    }
    function setProgress(barId, textId, progressList) {
      const bar = document.getElementById(barId);
      const textEl = document.getElementById(textId);
      if (!bar || !textEl) return;
      const vals = (progressList || []).map(p => {
        if (typeof p === 'object' && p !== null && ('step' in p) && ('total' in p)) {
          return { pct: Math.round((p.step / p.total) * 100), label: `${p.step}/${p.total} (${Math.round((p.step/p.total)*100)}%)` };
        }
        return { pct: 0, label: '' };
      });
      const last = vals.length ? vals[vals.length - 1] : { pct: 0, label: '' };
      bar.style.width = Math.min(100, last.pct || 0) + '%';
      textEl.innerHTML = last.label || '';
    }
    function apiReady() {
      return window.pywebview && window.pywebview.api;
    }
    function ensureApi(slotStatus) {
      if (!apiReady()) {
        setHTML(slotStatus, '<span style="color:red;">API not ready yet. Please wait a second.</span>');
        return false;
      }
      return true;
    }
    async function runIm() {
      setHTML('slot1', 'Running...');
      setHTML('status1', '');
      setHTML('explain1', '');
      setProgress('bar1', 'status1', []);
      if (!ensureApi('status1')) return;
      try {
        const payload = {
          beta: Number(document.getElementById('beta_z').value),
          y: Number(document.getElementById('y_z').value),
          a: Number(document.getElementById('a_z').value),
          z_min: Number(document.getElementById('zmin_z').value),
          z_max: Number(document.getElementById('zmax_z').value),
          points: Number(document.getElementById('points_z').value)
        };
        const res = await window.pywebview.api.im_vs_z(payload);

        // Plot using raw data
        const el = document.getElementById('slot1');
        el.classList.remove('empty');
        el.classList.add('has-plot');
        el.innerHTML = '';

        const {z_values, ims_values, real_values} = res.data;
        const {a, y, beta} = res.params;

        // Create traces for Im(s)
        const colors = ["#ef553b", "#2b8aef", "#1fbf68"];
        const traces = [];
        for (let i = 0; i < 3; i++) {
          traces.push({
            x: z_values,
            y: ims_values.map(row => row[i]),
            mode: 'lines',
            name: `Im s${i+1}`,
            line: {color: colors[i], width: 2}
          });
        }

        const layout = {
          title: `Im(s) vs z (beta=${beta.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
          xaxis: {
            title: 'z',
            autorange: true,
            color: '#e2e8f0',
            gridcolor: '#1e293b'
          },
          yaxis: {
            title: 'Im(s)',
            autorange: true,
            color: '#e2e8f0',
            gridcolor: '#1e293b'
          },
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: {
            color: '#e2e8f0'
          }
        };

        const config = {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        Plotly.newPlot('slot1', traces, layout, config);

        setProgress('bar1', 'status1', res.progress || []);
        setHTML('explain1', res.explanation || '');
      } catch (err) {
        setHTML('slot1', '<span style="color:red;">'+err+'</span>');
      }
    }
    async function runBeta() {
      setHTML('slot2', 'Running...');
      setHTML('status2', '');
      setHTML('explain2', '');
      setProgress('bar2', 'status2', []);
      if (!ensureApi('status2')) return;
      try {
        const payload = {
          z: Number(document.getElementById('z_beta').value),
          y: Number(document.getElementById('y_beta').value),
          a: Number(document.getElementById('a_beta').value),
          beta_min: Number(document.getElementById('beta_min').value),
          beta_max: Number(document.getElementById('beta_max').value),
          points: Number(document.getElementById('points_beta').value)
        };
        const res = await window.pywebview.api.roots_vs_beta(payload);

        const {beta_values, ims_values, real_values, discriminants} = res.data;
        const {z, y, a} = res.params;
        const colors = ["#ef553b", "#2b8aef", "#1fbf68"];

        // Plot Im(s) vs beta
        const el2 = document.getElementById('slot2');
        el2.classList.remove('empty');
        el2.classList.add('has-plot');
        el2.innerHTML = '';

        const imTraces = [];
        for (let i = 0; i < 3; i++) {
          imTraces.push({
            x: beta_values,
            y: ims_values.map(row => row[i]),
            mode: 'lines',
            name: `Im s${i+1}`,
            line: {color: colors[i], width: 2}
          });
        }

        Plotly.newPlot('slot2', imTraces, {
          title: `Im(s) vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
          xaxis: {title: 'beta', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          yaxis: {title: 'Im(s)', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' }
        }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

        // Plot Re(s) vs beta
        const el3 = document.getElementById('slot3');
        el3.classList.remove('empty');
        el3.classList.add('has-plot');
        el3.innerHTML = '';

        const reTraces = [];
        for (let i = 0; i < 3; i++) {
          reTraces.push({
            x: beta_values,
            y: real_values.map(row => row[i]),
            mode: 'lines',
            name: `Re s${i+1}`,
            line: {color: colors[i], width: 2}
          });
        }

        Plotly.newPlot('slot3', reTraces, {
          title: `Re(s) vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
          xaxis: {title: 'beta', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          yaxis: {title: 'Re(s)', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' }
        }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

        // Plot discriminant
        const el4 = document.getElementById('slot4');
        el4.classList.remove('empty');
        el4.classList.add('has-plot');
        el4.innerHTML = '';

        Plotly.newPlot('slot4', [{
          x: beta_values,
          y: discriminants,
          mode: 'lines',
          name: 'Cubic Discriminant',
          line: {color: 'white', width: 2}
        }], {
          title: `Discriminant vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
          xaxis: {title: 'beta', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          yaxis: {title: 'Discriminant', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' }
        }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

        setProgress('bar2', 'status2', res.progress || []);
        setHTML('explain2', res.explanation || '');
      } catch (err) {
        setHTML('slot2', '<span style="color:red;">'+err+'</span>');
      }
    }
    async function runEigen() {
      setHTML('slot6', 'Running...');
      setHTML('status4', '');
      setHTML('explain4', '');
      setProgress('bar4', 'status4', []);
      if (!ensureApi('status4')) return;
      try {
        const payload = {
          beta: Number(document.getElementById('beta_eig').value),
          a: Number(document.getElementById('a_eig').value),
          n: Number(document.getElementById('n_eig').value),
          p: Number(document.getElementById('p_eig').value),
          seed: Number(document.getElementById('seed_eig').value)
        };
        const res = await window.pywebview.api.eigen_distribution(payload);

        const {eigenvalues, kde_x, kde_y} = res.data;

        // Plot eigenvalue distribution
        const el6 = document.getElementById('slot6');
        el6.classList.remove('empty');
        el6.classList.add('has-plot');
        el6.innerHTML = '';

        const traces = [
          {
            x: eigenvalues,
            type: 'histogram',
            histnorm: 'probability density',
            name: 'Histogram',
            marker: {
              color: 'rgba(59, 130, 246, 0.7)',
              line: {
                color: 'rgba(59, 130, 246, 1)',
                width: 1
              }
            }
          },
          {
            x: kde_x,
            y: kde_y,
            mode: 'lines',
            name: 'KDE',
            line: {color: '#60a5fa', width: 2}
          }
        ];

        Plotly.newPlot('slot6', traces, {
          title: `Eigenvalue distribution (y=${payload.p / payload.n}, beta=${payload.beta}, a=${payload.a})`,
          xaxis: {title: 'Eigenvalue', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          yaxis: {title: 'Density', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
          hovermode: 'closest',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          legend: {
            font: { color: '#e2e8f0' }
          }
        }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

        document.getElementById('eig_stats').innerHTML = `
          <div class="metric">min: ${res.stats.min.toFixed(4)}</div>
          <div class="metric">max: ${res.stats.max.toFixed(4)}</div>
          <div class="metric">mean: ${res.stats.mean.toFixed(4)}</div>
          <div class="metric">std: ${res.stats.std.toFixed(4)}</div>
        `;
        setProgress('bar4', 'status4', res.progress || []);
        setHTML('explain4', res.explanation || '');
      } catch (err) {
        setHTML('slot6', '<span style="color:red;">'+err+'</span>');
      }
    }

    // Image Download & Management functions
    let currentImageIndex = 0;
    let totalImages = 0;
    let processedResults = [];

    document.getElementById('local_folder').addEventListener('change', function(e) {
      const files = e.target.files;
      if (files.length > 0) {
        // Get folder name from first file's path
        const path = files[0].webkitRelativePath || files[0].name;
        const folderName = path.split('/')[0];
        const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));
        document.getElementById('folder_count').innerHTML = `Folder: ${folderName} (${imageFiles.length} images)`;
      } else {
        document.getElementById('folder_count').innerHTML = 'No folder selected';
      }
    });

    document.getElementById('local_files').addEventListener('change', function(e) {
      const files = e.target.files;
      if (files.length > 0) {
        const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));
        const totalSize = imageFiles.reduce((sum, f) => sum + f.size, 0);
        const sizeMB = (totalSize / (1024 * 1024)).toFixed(2);
        document.getElementById('files_count').innerHTML = `${imageFiles.length} image(s) selected (${sizeMB} MB)`;
      } else {
        document.getElementById('files_count').innerHTML = 'No files selected';
      }
    });

    function toggleMethodParams() {
      const method1 = document.getElementById('method1').value;
      const method2 = document.getElementById('method2').value;
      const method1Params = document.getElementById('method1_params');
      const method2Params = document.getElementById('method2_params');

      // Show manual parameters for gen_manual, gen_centralized_manual, or gen_sympy_optimal methods
      if (method1Params) {
        const showMethod1 = (method1 === 'gen_manual' || method1 === 'gen_centralized_manual' || method1 === 'gen_sympy_optimal');
        method1Params.style.display = showMethod1 ? 'block' : 'none';

        // Hide all notes first
        document.getElementById('method1_centralized_note').style.display = 'none';
        document.getElementById('method1_sympy_note').style.display = 'none';

        // For centralized manual, hide sigma2 input and show centralized note
        if (method1 === 'gen_centralized_manual') {
          document.getElementById('method1_sigma2_container').style.display = 'none';
          document.getElementById('method1_centralized_note').style.display = 'block';
        }
        // For sympy optimal, hide sigma2 input and show sympy note
        else if (method1 === 'gen_sympy_optimal') {
          document.getElementById('method1_sigma2_container').style.display = 'none';
          document.getElementById('method1_sympy_note').style.display = 'block';
        }
        // For gen_manual, show sigma2 input
        else {
          document.getElementById('method1_sigma2_container').style.display = 'block';
        }
      }

      if (method2Params) {
        const showMethod2 = (method2 === 'gen_manual' || method2 === 'gen_centralized_manual' || method2 === 'gen_sympy_optimal');
        method2Params.style.display = showMethod2 ? 'block' : 'none';

        // Hide all notes first
        document.getElementById('method2_centralized_note').style.display = 'none';
        document.getElementById('method2_sympy_note').style.display = 'none';

        // For centralized manual, hide sigma2 input and show centralized note
        if (method2 === 'gen_centralized_manual') {
          document.getElementById('method2_sigma2_container').style.display = 'none';
          document.getElementById('method2_centralized_note').style.display = 'block';
        }
        // For sympy optimal, hide sigma2 input and show sympy note
        else if (method2 === 'gen_sympy_optimal') {
          document.getElementById('method2_sigma2_container').style.display = 'none';
          document.getElementById('method2_sympy_note').style.display = 'block';
        }
        // For gen_manual, show sigma2 input
        else {
          document.getElementById('method2_sigma2_container').style.display = 'block';
        }
      }
    }

    function toggleNoiseParams() {
      const noiseType = document.getElementById('noise_type').value;
      document.getElementById('laplacian_params').style.display = noiseType === 'laplacian' ? 'block' : 'none';
      document.getElementById('mog_params').style.display = noiseType === 'mixture_gaussian' ? 'block' : 'none';
    }

    async function downloadImages() {
      if (!ensureApi('status_download')) return;
      try {
        const url = document.getElementById('download_url').value;
        if (!url) {
          setHTML('status_download', '<span style="color:orange;">Please enter a URL</span>');
          return;
        }

        const payload = {
          url: url,
          count: Number(document.getElementById('download_count').value),
          scale: Number(document.getElementById('download_scale').value),
          folder: 'downloaded'
        };

        setHTML('status_download', 'Downloading images...');
        document.getElementById('bar_download').style.width = '50%';

        const res = await window.pywebview.api.download_images_playwright(payload);

        if (res.success) {
          setHTML('status_download', `<span style="color:green;">✓ Downloaded ${res.count} images to "${res.folder}"</span>`);
          document.getElementById('bar_download').style.width = '100%';
          refreshFolders();
        } else {
          setHTML('status_download', `<span style="color:red;">Error: ${res.error}</span>`);
          document.getElementById('bar_download').style.width = '0%';
        }
      } catch (err) {
        setHTML('status_download', '<span style="color:red;">'+err+'</span>');
        document.getElementById('bar_download').style.width = '0%';
      }
    }

    async function importLocalFolder() {
      if (!ensureApi('status_upload_folder')) return;
      const files = document.getElementById('local_folder').files;
      if (files.length === 0) {
        setHTML('status_upload_folder', '<span style="color:orange;">No folder selected</span>');
        return;
      }

      try {
        // Get folder name from first file's path
        const path = files[0].webkitRelativePath || files[0].name;
        const sourceFolderName = path.split('/')[0];

        // Filter image files only
        const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));

        if (imageFiles.length === 0) {
          setHTML('status_upload_folder', '<span style="color:orange;">No image files found in folder</span>');
          return;
        }

        const fileData = [];
        for (let i = 0; i < imageFiles.length; i++) {
          const file = imageFiles[i];
          const reader = new FileReader();
          const base64 = await new Promise((resolve) => {
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(file);
          });

          // Get relative path within the selected folder
          const fullPath = file.webkitRelativePath || file.name;
          const pathParts = fullPath.split('/');
          // Remove the first part (folder name) to get relative path
          const relativePath = pathParts.slice(1).join('/');

          fileData.push({
            name: file.name,
            path: relativePath,  // Preserve relative path
            data: base64
          });

          // Update progress bar
          const progress = Math.round(((i + 1) / imageFiles.length) * 100);
          document.getElementById('bar_upload_folder').style.width = progress + '%';
          setHTML('status_upload_folder', `Reading files: ${i + 1}/${imageFiles.length} (${progress}%)`);
        }

        const payload = {
          files: fileData,
          folder: sourceFolderName  // Use original folder name
        };

        setHTML('status_upload_folder', `Uploading ${imageFiles.length} images...`);
        const res = await window.pywebview.api.import_local_folder(payload);

        if (res && res.success) {
          setHTML('status_upload_folder', `<span style="color:green;">✓ Imported ${res.count} images to folder "${res.folder}"</span>`);
          document.getElementById('folder_count').innerHTML = 'No folder selected';
          document.getElementById('local_folder').value = '';
          document.getElementById('bar_upload_folder').style.width = '100%';
          refreshFolders();
        } else {
          const errorMsg = (res && res.error) ? res.error : 'Unknown error';
          setHTML('status_upload_folder', `<span style="color:red;">Error: ${errorMsg}</span>`);
          document.getElementById('bar_upload_folder').style.width = '0%';
        }
      } catch (err) {
        console.error('Upload error:', err);
        setHTML('status_upload_folder', '<span style="color:red;">Error: '+err.message+'</span>');
        document.getElementById('bar_upload_folder').style.width = '0%';
      }
    }

    async function uploadFilesToFolder() {
      if (!ensureApi('status_upload_files')) return;
      const files = document.getElementById('local_files').files;
      const destFolder = document.getElementById('upload_dest_folder').value;

      if (!destFolder || destFolder === 'Loading folders...' || destFolder === 'No folders found') {
        setHTML('status_upload_files', '<span style="color:orange;">Please select a destination folder</span>');
        return;
      }

      if (files.length === 0) {
        setHTML('status_upload_files', '<span style="color:orange;">No files selected</span>');
        return;
      }

      try {
        // Filter image files only
        const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));

        if (imageFiles.length === 0) {
          setHTML('status_upload_files', '<span style="color:orange;">No image files selected</span>');
          return;
        }

        const fileData = [];
        for (let i = 0; i < imageFiles.length; i++) {
          const file = imageFiles[i];
          const reader = new FileReader();
          const base64 = await new Promise((resolve) => {
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(file);
          });

          fileData.push({
            name: file.name,
            data: base64
          });

          // Update progress bar
          const progress = Math.round(((i + 1) / imageFiles.length) * 100);
          document.getElementById('bar_upload_files').style.width = progress + '%';
          setHTML('status_upload_files', `Reading files: ${i + 1}/${imageFiles.length} (${progress}%)`);
        }

        const payload = {
          files: fileData,
          folder: destFolder
        };

        setHTML('status_upload_files', `Uploading ${imageFiles.length} images to "${destFolder}"...`);
        const res = await window.pywebview.api.upload_files_to_folder(payload);

        if (res && res.success) {
          setHTML('status_upload_files', `<span style="color:green;">✓ Uploaded ${res.count} files to "${res.folder}"</span>`);
          document.getElementById('files_count').innerHTML = 'No files selected';
          document.getElementById('local_files').value = '';
          document.getElementById('bar_upload_files').style.width = '100%';
          refreshFolders();

          // Clear status after 3 seconds
          setTimeout(() => {
            setHTML('status_upload_files', '');
            document.getElementById('bar_upload_files').style.width = '0%';
          }, 3000);
        } else {
          const errorMsg = (res && res.error) ? res.error : 'Unknown error';
          setHTML('status_upload_files', `<span style="color:red;">Error: ${errorMsg}</span>`);
          document.getElementById('bar_upload_files').style.width = '0%';
        }
      } catch (err) {
        console.error('Upload error:', err);
        setHTML('status_upload_files', '<span style="color:red;">Error: '+err.message+'</span>');
        document.getElementById('bar_upload_files').style.width = '0%';
      }
    }

    async function refreshFolders() {
      // Wait for API to be ready
      if (!apiReady()) {
        console.log('API not ready, retrying in 500ms...');
        setTimeout(refreshFolders, 500);
        return;
      }

      try {
        const res = await window.pywebview.api.list_random_matrix_folders();
        const folderList = document.getElementById('folder_list');
        const processFolderList = document.getElementById('process_folder');
        const uploadDestFolder = document.getElementById('upload_dest_folder');

        if (!folderList || !processFolderList) {
          console.log('DOM elements not ready, retrying...');
          setTimeout(refreshFolders, 500);
          return;
        }

        folderList.innerHTML = '';
        processFolderList.innerHTML = '';
        if (uploadDestFolder) uploadDestFolder.innerHTML = '';

        if (res.folders.length === 0) {
          folderList.innerHTML = '<option>No folders found</option>';
          processFolderList.innerHTML = '<option>No folders found</option>';
          if (uploadDestFolder) uploadDestFolder.innerHTML = '<option>No folders found</option>';
        } else {
          res.folders.forEach(folder => {
            const opt1 = document.createElement('option');
            opt1.value = folder;
            opt1.textContent = folder;
            folderList.appendChild(opt1);

            const opt2 = document.createElement('option');
            opt2.value = folder;
            opt2.textContent = folder;
            processFolderList.appendChild(opt2);

            if (uploadDestFolder) {
              const opt3 = document.createElement('option');
              opt3.value = folder;
              opt3.textContent = folder;
              uploadDestFolder.appendChild(opt3);
            }
          });
        }
        console.log(`Loaded ${res.folders.length} folders`);
      } catch (err) {
        console.error('Error refreshing folders:', err);
        // Retry on error
        setTimeout(refreshFolders, 1000);
      }
    }

    async function viewFolder() {
      if (!ensureApi('status_browse')) return;
      const folderList = document.getElementById('folder_list');
      const selectedFolder = folderList.value;

      if (!selectedFolder || selectedFolder === 'Loading folders...' || selectedFolder === 'No folders found') {
        return;
      }

      try {
        const res = await window.pywebview.api.get_folder_contents(selectedFolder);
        const contentsDiv = document.getElementById('folder_contents');
        const folderInfo = document.getElementById('folder_info');

        // Update folder info
        const totalFiles = res.files.length;
        const displayCount = Math.min(totalFiles, 50);
        if (totalFiles > 50) {
          folderInfo.innerHTML = `📁 ${selectedFolder} - Showing first ${displayCount} of ${totalFiles} images`;
        } else if (totalFiles > 0) {
          folderInfo.innerHTML = `📁 ${selectedFolder} - ${totalFiles} image(s)`;
        } else {
          folderInfo.innerHTML = `📁 ${selectedFolder}`;
        }

        if (res.files.length === 0) {
          contentsDiv.innerHTML = '<p style="text-align: center; color: #94a3b8; padding: 40px;"><span style="font-size: 32px; display: block; margin-bottom: 8px;">📂</span>Folder is empty</p>';
        } else {
          let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px;">';
          res.files.forEach((file, idx) => {
            html += `
              <div style="text-align: center; padding: 8px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">
                <img src="data:image/png;base64,${file.data}" style="width: 100%; height: 120px; object-fit: contain; border-radius: 4px; background: #f8fafc;" />
                <div style="font-size: 11px; color: #64748b; margin-top: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${file.name}">${file.name}</div>
              </div>
            `;
          });
          html += '</div>';
          contentsDiv.innerHTML = html;
        }
      } catch (err) {
        document.getElementById('folder_contents').innerHTML = '<p style="color: red; text-align: center; padding: 40px;">Error loading folder contents</p>';
        document.getElementById('folder_info').innerHTML = '';
      }
    }

    async function createNewFolder() {
      if (!ensureApi('status_create')) return;
      const folderName = document.getElementById('new_folder_name').value.trim();

      if (!folderName) {
        setHTML('status_create', '<span style="color:orange;">Please enter a folder name</span>');
        return;
      }

      try {
        setHTML('status_create', 'Creating folder...');
        document.getElementById('bar_create').style.width = '50%';

        const res = await window.pywebview.api.create_folder({ folder_name: folderName });

        if (res.success) {
          setHTML('status_create', `<span style="color:green;">✓ Folder "${res.folder}" created successfully</span>`);
          document.getElementById('bar_create').style.width = '100%';
          document.getElementById('new_folder_name').value = '';
          refreshFolders();

          // Clear status after 3 seconds
          setTimeout(() => {
            setHTML('status_create', '');
            document.getElementById('bar_create').style.width = '0%';
          }, 3000);
        } else {
          setHTML('status_create', `<span style="color:red;">Error: ${res.error}</span>`);
          document.getElementById('bar_create').style.width = '0%';
        }
      } catch (err) {
        setHTML('status_create', '<span style="color:red;">'+err+'</span>');
        document.getElementById('bar_create').style.width = '0%';
      }
    }

    async function deleteFolder() {
      if (!ensureApi('status_browse')) return;
      const folderList = document.getElementById('folder_list');
      const selectedFolder = folderList.value;

      if (!selectedFolder || selectedFolder === 'Loading folders...' || selectedFolder === 'No folders found') {
        setHTML('status_browse', '<span style="color:orange;">Please select a folder to delete</span>');
        return;
      }

      // Confirm deletion
      const confirmed = confirm(`Are you sure you want to delete the folder "${selectedFolder}" and all its contents? This action cannot be undone.`);
      if (!confirmed) {
        return;
      }

      try {
        setHTML('status_browse', 'Deleting folder...');

        const res = await window.pywebview.api.delete_folder({ folder_name: selectedFolder });

        if (res.success) {
          setHTML('status_browse', `<span style="color:green;">✓ Folder "${res.folder}" deleted successfully</span>`);
          document.getElementById('folder_contents').innerHTML = '<p style="text-align: center; color: #94a3b8; padding: 40px;">Select a folder to view its contents</p>';
          refreshFolders();

          // Clear status after 3 seconds
          setTimeout(() => {
            setHTML('status_browse', '');
          }, 3000);
        } else {
          setHTML('status_browse', `<span style="color:red;">Error: ${res.error}</span>`);
        }
      } catch (err) {
        setHTML('status_browse', '<span style="color:red;">'+err+'</span>');
      }
    }

    async function processImages() {
      if (!ensureApi('status_process')) return;
      const folder = document.getElementById('process_folder').value;

      if (!folder || folder === 'Loading folders...' || folder === 'No folders found') {
        setHTML('status_process', '<span style="color:orange;">Please select a folder</span>');
        return;
      }

      try {
        // Reset info panels
        setHTML('processing_info', 'Starting processing...');
        setHTML('method1_info', 'Method 1 processing...');
        setHTML('method2_info', 'Method 2 processing...');

        const method1 = document.getElementById('method1').value;
        const method2 = document.getElementById('method2').value;

        const noiseType = document.getElementById('noise_type').value;
        const randomSeedInput = document.getElementById('random_seed').value;
        const payload = {
          folder: folder,
          method1: method1,
          method2: method2,
          noise_type: noiseType,
          laplacian_scale: Number(document.getElementById('laplacian_scale').value),
          random_seed: randomSeedInput ? Number(randomSeedInput) : null
        };

        // Add Mixture of Gaussians parameters if selected
        if (noiseType === 'mixture_gaussian') {
          payload.mog_weights = [
            Number(document.getElementById('mog_weight1').value),
            Number(document.getElementById('mog_weight2').value),
            Number(document.getElementById('mog_weight3').value)
          ];
          payload.mog_means = [
            Number(document.getElementById('mog_mean1').value),
            Number(document.getElementById('mog_mean2').value),
            Number(document.getElementById('mog_mean3').value)
          ];
          payload.mog_sigmas = [
            Number(document.getElementById('mog_sigma1').value),
            Number(document.getElementById('mog_sigma2').value),
            Number(document.getElementById('mog_sigma3').value)
          ];
        }

        // Add manual params for Method 1 if needed
        if (method1 === 'gen_manual') {
          payload.method1_sigma2 = Number(document.getElementById('method1_sigma2').value);
          payload.method1_a = Number(document.getElementById('method1_a').value);
          payload.method1_beta = Number(document.getElementById('method1_beta').value);
        }

        // Add manual params for Method 2 if needed
        if (method2 === 'gen_manual') {
          payload.method2_sigma2 = Number(document.getElementById('method2_sigma2').value);
          payload.method2_a = Number(document.getElementById('method2_a').value);
          payload.method2_beta = Number(document.getElementById('method2_beta').value);
        }

        setHTML('status_process', 'Processing images...');
        const res = await window.pywebview.api.process_images(payload);

        if (res.success) {
          totalImages = res.total_images;
          currentImageIndex = 0;
          processedResults = res.results;  // Store results
          displayCurrentImage(processedResults[0]);

          // Display processing details (already updated in real-time by Python)
          setHTML('status_process', `<span style="color:green;">✓ Completed! Processed ${res.total_images} images</span>`);
          updateImageCounter();
        } else {
          setHTML('status_process', `<span style="color:red;">Error: ${res.error}</span>`);
        }
      } catch (err) {
        setHTML('status_process', '<span style="color:red;">'+err+'</span>');
      }
    }

    function displayCurrentImage(data) {
      // Top Left: Method 1
      document.getElementById('result1').innerHTML = `<img src="data:image/png;base64,${data.method1}" style="width: 100%; height: 100%; object-fit: contain;" />`;

      // Top Right: Method 2
      document.getElementById('result2').innerHTML = `<img src="data:image/png;base64,${data.method2}" style="width: 100%; height: 100%; object-fit: contain;" />`;

      // Bottom Left: Original
      document.getElementById('result_original').innerHTML = `<img src="data:image/png;base64,${data.original}" style="width: 100%; height: 100%; object-fit: contain;" />`;

      // Bottom Right: With Noise
      document.getElementById('result_blur').innerHTML = `<img src="data:image/png;base64,${data.blurred}" style="width: 100%; height: 100%; object-fit: contain;" />`;

      // Update labels with method names and eigenvector count if available
      if (data.method1_name) {
        const eigvec1 = data.method1_eigvec !== undefined ? ` (${data.method1_eigvec} eigvec)` : '';
        document.getElementById('result1_label').textContent = data.method1_name + eigvec1;
      }
      if (data.method2_name) {
        const eigvec2 = data.method2_eigvec !== undefined ? ` (${data.method2_eigvec} eigvec)` : '';
        document.getElementById('result2_label').textContent = data.method2_name + eigvec2;
      }
      if (data.noise_type_label) {
        document.getElementById('result_blur_label').textContent = data.noise_type_label;
      }

      // Update method comparison info
      const comparisonDiv = document.getElementById('method_comparison');
      if (comparisonDiv && data.method1_name && data.method2_name) {
        comparisonDiv.innerHTML = `
          <div style="text-align: left; font-size: 13px; color: #475569;">
            <strong>Method 1:</strong> ${data.method1_name}
            <strong style="color: #2563eb;">(${data.method1_eigvec || 0} eigenvectors)</strong><br>
            <strong>Method 2:</strong> ${data.method2_name}
            <strong style="color: #2563eb;">(${data.method2_eigvec || 0} eigenvectors)</strong><br>
            <strong>Noise:</strong> ${data.noise_type_label}<br><br>
            <div style="margin-top: 12px; padding: 12px; background: #f0f9ff; border-radius: 6px;">
              Click on any image to view its eigenvalue distribution below
            </div>
          </div>
        `;
      }
    }

    async function prevImage() {
      if (currentImageIndex > 0) {
        currentImageIndex--;
        await loadImage(currentImageIndex);
        updateImageCounter();
      }
    }

    async function nextImage() {
      if (currentImageIndex < totalImages - 1) {
        currentImageIndex++;
        await loadImage(currentImageIndex);
        updateImageCounter();
      }
    }

    async function loadImage(index) {
      // Use cached results instead of calling backend
      if (processedResults && processedResults[index]) {
        displayCurrentImage(processedResults[index]);
      } else {
        console.error('No processed results available');
      }
    }

    function updateImageCounter() {
      const counter = String(currentImageIndex).padStart(3, '0');
      document.getElementById('image_counter').textContent = counter;
    }

    async function showImageEigenvalues(imageId) {
      if (!ensureApi('eigenvalue_title')) return;

      const imageDiv = document.getElementById(imageId);
      if (!imageDiv) return;

      // Get the image element inside the div
      const imgElement = imageDiv.querySelector('img');
      if (!imgElement) {
        setHTML('eigenvalue_title', '<span style="color: #dc2626;">No image available. Please process images first.</span>');
        document.getElementById('eigenvalue_section').style.display = 'none';
        return;
      }

      try {
        // Show the eigenvalue section
        document.getElementById('eigenvalue_section').style.display = 'block';

        // Highlight the clicked image temporarily
        imageDiv.style.boxShadow = '0 0 0 3px #3b82f6';
        setTimeout(() => { imageDiv.style.boxShadow = ''; }, 500);

        // Get image label
        const labelMap = {
          'result_original': 'Original Image',
          'result1': document.getElementById('result1_label')?.textContent || 'Method 1 Result',
          'result2': document.getElementById('result2_label')?.textContent || 'Method 2 Result',
          'result_blur': document.getElementById('result_blur_label')?.textContent || 'With Noise'
        };
        const label = labelMap[imageId] || 'Selected Image';

        setHTML('eigenvalue_title', `Computing eigenvalue distribution for: <strong>${label}</strong>...`);
        setHTML('eigenvalue_plot', '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #64748b;">Computing eigenvalues...</div>');
        setHTML('eigenvalue_stats', '');

        // Extract base64 image data from src
        const imgSrc = imgElement.src;
        if (!imgSrc || !imgSrc.includes('base64,')) {
          setHTML('eigenvalue_title', '<span style="color: #dc2626;">Invalid image data</span>');
          return;
        }

        const base64Data = imgSrc.split('base64,')[1];

        // Call backend to compute eigenvalues
        const res = await window.pywebview.api.compute_image_eigenvalues({ image_data: base64Data });

        if (res.success) {
          setHTML('eigenvalue_title', `Eigenvalue Distribution for: <strong>${label}</strong>`);

          // Store eigenvalues for later use
          window.currentEigenvalues = res.eigenvalues_all || res.eigenvalues;

          // Plot eigenvalue distribution with improved styling
          const trace1 = {
            x: res.eigenvalues,
            type: 'histogram',
            name: 'Histogram',
            marker: {
              color: '#60a5fa',
              opacity: 0.7,
              line: {
                color: '#3b82f6',
                width: 1.5
              }
            },
            nbinsx: 50,
            autobinx: true,
            hovertemplate: 'Range: %{x:.6f}<br>Count: %{y}<extra></extra>'
          };

          const trace2 = {
            x: res.kde_x,
            y: res.kde_y,
            type: 'scatter',
            mode: 'lines',
            name: 'KDE (Smoothed Density)',
            line: {
              color: '#f43f5e',
              width: 3,
              shape: 'spline',
              smoothing: 1.3
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(244, 63, 94, 0.15)',
            yaxis: 'y2',
            hovertemplate: 'Eigenvalue: %{x:.6f}<br>Density: %{y:.4f}<extra></extra>'
          };

          const layout = {
            title: {
              text: `<b>Distribution of ${res.stats.count} Eigenvalues</b>`,
              font: { size: 16, color: '#0f172a' },
              y: 0.98
            },
            xaxis: {
              title: {
                text: 'Eigenvalue (λ)',
                font: { size: 14, weight: 600, color: '#1e293b' }
              },
              gridcolor: '#e2e8f0',
              zeroline: false,
              showline: true,
              linewidth: 2,
              linecolor: '#cbd5e1'
            },
            yaxis: {
              title: {
                text: 'Frequency (Count)',
                font: { size: 14, weight: 600, color: '#3b82f6' }
              },
              gridcolor: '#e2e8f0',
              zeroline: false,
              showline: true,
              linewidth: 2,
              linecolor: '#cbd5e1'
            },
            yaxis2: {
              title: {
                text: 'Density',
                font: { size: 14, weight: 600, color: '#f43f5e' }
              },
              overlaying: 'y',
              side: 'right',
              showgrid: false,
              zeroline: false
            },
            showlegend: true,
            legend: {
              x: 0.02,
              y: 0.98,
              bgcolor: 'rgba(255, 255, 255, 0.95)',
              bordercolor: '#cbd5e1',
              borderwidth: 2,
              font: { size: 12, weight: 600 }
            },
            autosize: true,
            height: 450,
            margin: { l: 70, r: 70, t: 50, b: 70 },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            hovermode: 'closest',
            bargap: 0.05
          };

          const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
              format: 'png',
              filename: 'eigenvalue_distribution',
              height: 800,
              width: 1200,
              scale: 2
            }
          };

          Plotly.newPlot('eigenvalue_plot', [trace1, trace2], layout, config);

          // Display comprehensive statistics
          const stats = res.stats;
          const statsHTML = `
            <div style="margin-bottom: 14px; font-weight: 600; color: #92400e; font-size: 15px; border-bottom: 2px solid #fde68a; padding-bottom: 8px;">📊 Statistical Summary</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px;">
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Count (Non-zero)</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.count}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Total Count</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.total_count}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Minimum (λ_min)</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.min.toFixed(6)}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Maximum (λ_max)</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.max.toFixed(6)}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Mean (μ)</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.mean.toFixed(6)}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Median</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.median.toFixed(6)}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Std Dev (σ)</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.std.toFixed(6)}</div>
              </div>
              <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Image Size</div>
                <div style="font-weight: 600; font-size: 15px; color: #78350f;">${res.image_size.height}×${res.image_size.width}</div>
              </div>
            </div>
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #fde68a;">
              <div style="font-weight: 600; color: #92400e; margin-bottom: 8px; font-size: 13px;">Percentiles:</div>
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px;">
                <div style="background: #fffbeb; padding: 6px; border-radius: 4px; text-align: center;">
                  <div style="font-size: 10px; color: #92400e;">P25</div>
                  <div style="font-weight: 600; font-size: 13px; color: #78350f;">${stats.p25.toFixed(4)}</div>
                </div>
                <div style="background: #fffbeb; padding: 6px; border-radius: 4px; text-align: center;">
                  <div style="font-size: 10px; color: #92400e;">P50 (Median)</div>
                  <div style="font-weight: 600; font-size: 13px; color: #78350f;">${stats.p50.toFixed(4)}</div>
                </div>
                <div style="background: #fffbeb; padding: 6px; border-radius: 4px; text-align: center;">
                  <div style="font-size: 10px; color: #92400e;">P75</div>
                  <div style="font-weight: 600; font-size: 13px; color: #78350f;">${stats.p75.toFixed(4)}</div>
                </div>
                <div style="background: #fffbeb; padding: 6px; border-radius: 4px; text-align: center;">
                  <div style="font-size: 10px; color: #92400e;">P90</div>
                  <div style="font-weight: 600; font-size: 13px; color: #78350f;">${stats.p90.toFixed(4)}</div>
                </div>
                <div style="background: #fffbeb; padding: 6px; border-radius: 4px; text-align: center;">
                  <div style="font-size: 10px; color: #92400e;">P95</div>
                  <div style="font-weight: 600; font-size: 13px; color: #78350f;">${stats.p95.toFixed(4)}</div>
                </div>
                <div style="background: #fffbeb; padding: 6px; border-radius: 4px; text-align: center;">
                  <div style="font-size: 10px; color: #92400e;">P99</div>
                  <div style="font-weight: 600; font-size: 13px; color: #78350f;">${stats.p99.toFixed(4)}</div>
                </div>
              </div>
            </div>
          `;
          setHTML('eigenvalue_stats', statsHTML);

          // Scroll to eigenvalue section
          document.getElementById('eigenvalue_section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          setHTML('eigenvalue_title', `<span style="color: #dc2626;">Error: ${res.error}</span>`);
          setHTML('eigenvalue_plot', '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #dc2626;">Failed to compute eigenvalues</div>');
        }
      } catch (err) {
        console.error('Error computing eigenvalues:', err);
        setHTML('eigenvalue_title', `<span style="color: #dc2626;">Error: ${err.message}</span>`);
        setHTML('eigenvalue_plot', '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #dc2626;">Error occurred</div>');
      }
    }

    function toggleEigenvalueList() {
      const container = document.getElementById('eigenvalue_list_container');
      const button = event.target;

      if (container.style.display === 'none') {
        // Show the list
        if (!window.currentEigenvalues || window.currentEigenvalues.length === 0) {
          alert('No eigenvalues available. Please click on an image first to compute eigenvalues.');
          return;
        }

        // Generate eigenvalue list with indices
        const eigenvalues = window.currentEigenvalues;
        let listHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 6px;">';

        eigenvalues.forEach((val, idx) => {
          const color = val > 1e-6 ? '#0f172a' : '#94a3b8';
          listHTML += `
            <div style="padding: 4px 8px; background: white; border-radius: 4px; border: 1px solid #e2e8f0; display: flex; justify-content: space-between;">
              <span style="color: #64748b; font-size: 11px;">λ[${idx}]:</span>
              <span style="color: ${color}; font-weight: 600;">${val.toExponential(6)}</span>
            </div>
          `;
        });

        listHTML += '</div>';
        setHTML('eigenvalue_list', listHTML);

        container.style.display = 'block';
        button.textContent = '🔼 Hide All Values';

        // Scroll to the list
        container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        // Hide the list
        container.style.display = 'none';
        button.textContent = '📋 Show All Values';
      }
    }

    function downloadEigenvalues() {
      if (!window.currentEigenvalues || window.currentEigenvalues.length === 0) {
        alert('No eigenvalues available. Please click on an image first to compute eigenvalues.');
        return;
      }

      const eigenvalues = window.currentEigenvalues;

      // Create CSV content
      let csvContent = 'Index,Eigenvalue,Eigenvalue_Scientific\n';
      eigenvalues.forEach((val, idx) => {
        csvContent += `${idx},${val},${val.toExponential(10)}\n`;
      });

      // Create blob and download
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);

      link.setAttribute('href', url);
      link.setAttribute('download', `eigenvalues_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`);
      link.style.visibility = 'hidden';

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    // ========== Temp File Management Functions ==========

    async function refreshTempFiles() {
      if (!ensureApi('temp_files_status')) return;

      try {
        setHTML('temp_files_status', '<span style="color:#3b82f6;">Loading temp folders...</span>');
        const res = await window.pywebview.api.list_temp_folders();

        if (res.error) {
          setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
          return;
        }

        if (res.folders.length === 0) {
          setHTML('temp_files_status', '<span style="color:#94a3b8;">No output folders found</span>');
          setHTML('temp_files_list', '<div style="text-align: center; color: #94a3b8; padding: 40px;">No folders in temp directory</div>');
        } else {
          setHTML('temp_files_status', `<span style="color:green;">✓ Found ${res.folders.length} folder(s)</span>`);
          renderTempFoldersList(res.folders);
        }
      } catch (err) {
        setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
      }
    }

    function renderTempFoldersList(folders) {
      let html = '<div style="display: flex; flex-direction: column; gap: 12px;">';

      for (const folder of folders) {
        const sizeDisplay = folder.size_mb > 0 ? `${folder.size_mb} MB` : `${(folder.size_bytes / 1024).toFixed(2)} KB`;
        // Properly escape folder name for use in onclick attributes
        const escapedName = folder.name.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

        html += `
          <div style="border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
              <div style="flex: 1; min-width: 0;">
                <div style="font-weight: 600; font-size: 14px; color: #0f172a; margin-bottom: 4px; word-break: break-all;">
                  📁 ${folder.name}
                </div>
                <div style="font-size: 12px; color: #64748b; display: flex; flex-wrap: wrap; gap: 12px;">
                  <span>📅 ${folder.modified_str}</span>
                  <span>📊 ${sizeDisplay}</span>
                  <span>🖼️ ${folder.file_count} file(s)</span>
                </div>
              </div>
              <div style="display: flex; gap: 8px;">
                <button
                  onclick="openTempFolder('${escapedName}')"
                  style="background: #3b82f6; color: white; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; white-space: nowrap;"
                  title="Open folder in file explorer"
                >
                  📂 Open
                </button>
                <button
                  onclick="deleteTempFolder('${escapedName}')"
                  style="background: #dc2626; color: white; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; white-space: nowrap;"
                  title="Delete this folder"
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          </div>
        `;
      }

      html += '</div>';
      setHTML('temp_files_list', html);
    }

    async function deleteTempFolder(folderName) {
      if (!ensureApi('temp_files_status')) return;

      if (!confirm(`Delete folder "${folderName}"?\n\nThis will permanently delete all files in this folder.`)) {
        return;
      }

      try {
        setHTML('temp_files_status', `<span style="color:#3b82f6;">Deleting ${folderName}...</span>`);
        const res = await window.pywebview.api.delete_temp_folder(folderName);

        if (res.success) {
          setHTML('temp_files_status', `<span style="color:green;">✓ ${res.message}</span>`);
          // Refresh the list
          await refreshTempFiles();
        } else {
          setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
        }
      } catch (err) {
        setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
      }
    }

    async function deleteAllTempFiles() {
      if (!ensureApi('temp_files_status')) return;

      if (!confirm('Delete ALL temp folders?\n\nThis will permanently delete all output folders and files in the temp directory.\n\nThis action cannot be undone!')) {
        return;
      }

      try {
        setHTML('temp_files_status', '<span style="color:#3b82f6;">Deleting all folders...</span>');
        const res = await window.pywebview.api.delete_all_temp_folders();

        if (res.success) {
          setHTML('temp_files_status', `<span style="color:green;">✓ ${res.message}</span>`);
          // Refresh the list
          await refreshTempFiles();
        } else {
          setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
        }
      } catch (err) {
        setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
      }
    }

    async function openTempFolder(folderName) {
      if (!ensureApi('temp_files_status')) return;

      try {
        setHTML('temp_files_status', `<span style="color:#3b82f6;">Opening folder...</span>`);
        const res = await window.pywebview.api.open_temp_folder(folderName);

        if (res.success) {
          setHTML('temp_files_status', `<span style="color:green;">✓ Opened folder: ${folderName}</span>`);
        } else {
          setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
        }
      } catch (err) {
        setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
      }
    }

    // Initialize on load
    window.addEventListener('load', () => {
      refreshFolders();
    });
  </script>
</body>
</html>
"""


def main():
    # Suppress pywebview accessibility warnings
    import logging
    import warnings
    logging.getLogger('pywebview').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')

    # Redirect stderr to suppress Windows accessibility errors
    import sys
    if sys.platform == 'win32':
        sys.stderr = open(os.devnull, 'w')

    # Create .random_matrix folder on startup
    get_random_matrix_folder()

    # Register cleanup handler
    atexit.register(cleanup_temp_folder)

    # Create Bridge first (without window)
    api = Bridge(None)

    # Create window with the API
    window = webview.create_window(
        "Matrix Analysis Lab",
        html=HTML,
        js_api=api,
        width=1300,
        height=900,
    )

    # Now set the window reference in the Bridge so it can update the UI
    api.window = window

    webview.start()


if __name__ == "__main__":
    main()
