import json
import os
import io
import base64
import atexit
import shutil
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, minimize_scalar
from PIL import Image
import rmt_denoise
import webview
from tqdm import tqdm
import threading
import time


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

    pbar = tqdm(total=num_points, desc="Computing Im(s) vs z")
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
        pbar.update(1)
        if progress_cb:
            progress_cb(i + 1, num_points)
    pbar.close()

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


def generalized_mp_density(z_values, y, beta, a):
    """Generalized Marchenko-Pastur density f_{y,H}(z) for H = beta*delta_a + (1-beta)*delta_1.

    Returns the density on the same grid as z_values (zero where the discriminant is
    non-positive, since outside the support the cubic has three real roots).
    """
    z = np.asarray(z_values, dtype=float)
    f = np.zeros_like(z)
    mask = z > 0
    if not np.any(mask):
        return f
    zp = z[mask]
    A = a * (zp - y + 1.0) + zp
    B = a + zp - y + 1.0 - y * beta * (a - 1.0)
    az = a * zp
    a2z2 = az * az
    a3z3 = a2z2 * az
    P = A * B / (6.0 * a2z2) - A**3 / (27.0 * a3z3) - 1.0 / (2.0 * az)
    Q = B / (3.0 * az) - A * A / (9.0 * a2z2)
    D = P * P + Q**3
    pos = D > 0
    sqrtD = np.zeros_like(D)
    sqrtD[pos] = np.sqrt(D[pos])
    term = np.cbrt(P + sqrtD) - np.cbrt(P - sqrtD)
    density = np.where(pos, (np.sqrt(3.0) / (2.0 * np.pi * y)) * term, 0.0)
    density = np.clip(density, 0.0, None)
    f[mask] = density
    return f


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
    """Clean up temp folder on exit"""
    temp = get_temp_folder()
    if temp.exists():
        shutil.rmtree(temp, ignore_errors=True)
        temp.mkdir(exist_ok=True)

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

def mp_denoise(X):
    """M-P law denoising (from random_matrix.py lines 673-775)"""
    p, n = X.shape

    # Center data
    mean_vec = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean_vec

    # Sample covariance
    S = (Xc @ Xc.T) / n

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Aspect ratio
    y = p / float(n)
    sqrt_y = np.sqrt(y)

    # Positive eigenvalues
    pos_mask = eigvals > 1e-10
    pos_eigs = eigvals[pos_mask]

    if len(pos_eigs) < 3:
        return X, {"y": y, "sigma2_hat": None}

    # Use bottom half as noise bulk
    half = len(pos_eigs) // 2
    bulk_eigs = pos_eigs[half:]
    bulk_min = np.min(bulk_eigs)

    # Estimate sigma^2
    lambda_minus_factor = (1.0 - sqrt_y) ** 2
    sigma2_hat = bulk_min / lambda_minus_factor
    lambda_plus = sigma2_hat * (1.0 + sqrt_y) ** 2

    # Signal eigenvalues
    signal_mask = eigvals > (lambda_plus + 1e-9)
    num_signal = int(np.sum(signal_mask))

    if num_signal == 0:
        X_denoised = np.zeros_like(X) + mean_vec
    else:
        Gamma = eigvecs[:, signal_mask]
        X_denoised = Gamma @ (Gamma.T @ X)
        X_denoised = X_denoised + mean_vec

    info = {"y": y, "sigma2_hat": sigma2_hat, "lambda_plus": lambda_plus, "num_signal": num_signal}
    return X_denoised, info

def optimize_generalized_params(X):
    """Optimize sigma2, a, beta to match lower bound (from random_matrix.py lines 778-883)"""
    p, n = X.shape

    # Center and compute eigenvalues
    mean_vec = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean_vec
    S = (Xc @ Xc.T) / n
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

def generalized_denoise(X, sigma2=None, a=None, beta=None):
    """Generalized covariance denoising (from random_matrix.py lines 1003-1117)"""
    p, n = X.shape

    # Auto-optimize if not provided
    if sigma2 is None or a is None or beta is None:
        sigma2, a, beta = optimize_generalized_params(X)

    # Center data
    mean_vec = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean_vec

    # Sample covariance
    S = (Xc @ Xc.T) / n

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Aspect ratio
    y = p / float(n)

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
    num_signal = int(np.sum(signal_mask))

    if num_signal == 0:
        X_denoised = np.zeros_like(X) + mean_vec
    else:
        Gamma = eigvecs[:, signal_mask]
        X_denoised = Gamma @ (Gamma.T @ X)
        X_denoised = X_denoised + mean_vec

    info = {"y": y, "a": a, "beta": beta, "sigma2": sigma2, "num_signal": num_signal}
    return X_denoised, info


# --------- Desktop (PyWebView) bridge --------- #
class Bridge:
    def __init__(self, window):
        self.window = window
        self.processed_results = []

    def _log(self, event, payload=None):
        msg = {"event": event}
        if payload is not None:
            msg["payload"] = payload
        print(json.dumps(msg))
        try:
            import sys
            sys.stdout.flush()
        except Exception:
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
                print(f"Progress update error: {e}")

    def _update_status_ui(self, element_id, text):
        """Update any UI element text in real-time using evaluate_js"""
        if self.window:
            try:
                # Escape single quotes in text
                text_escaped = text.replace("'", "\\'")
                self.window.evaluate_js(f"document.getElementById('{element_id}').innerHTML = '{text_escaped}';")
            except Exception as e:
                print(f"Status update error: {e}")

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

        pbar = tqdm(total=points, desc="Computing roots vs beta")
        for idx, beta in enumerate(beta_points):
            roots = compute_cubic_roots(z, beta, a, y)
            roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
            all_roots.append(roots)
            discriminants.append(generate_cubic_discriminant(z, beta, a, y))
            pbar.update(1)
            if cb:
                cb(idx + 1, points)
        pbar.close()

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

        print("Generating eigenvalue distribution...")
        total_steps = 5
        with tqdm(total=total_steps, desc="Eigenvalue computation") as pbar:
            p_dim = int(y_eff * n)
            k = int(np.floor(beta * p_dim))
            diag_entries = np.concatenate([np.full(k, a), np.full(p_dim - k, 1.0)])
            np.random.shuffle(diag_entries)
            T_n = np.diag(diag_entries)
            pbar.update(1)
            if cb:
                cb(1, total_steps)

            X = np.random.randn(p_dim, n)
            pbar.update(1)
            if cb:
                cb(2, total_steps)

            S_n = (1 / n) * (X @ X.T)
            pbar.update(1)
            if cb:
                cb(3, total_steps)

            # B_n = S_n T_n is NOT symmetric, so eigvalsh on it is wrong (drops the
            # rank-deficient zero eigenvalues). B_n shares eigenvalues with the
            # symmetric PSD matrix T_n^{1/2} S_n T_n^{1/2} = (1/n)(T^{1/2}X)(T^{1/2}X)^T,
            # so compute them there to recover the full spectrum (including zeros).
            sqrt_diag = np.sqrt(diag_entries)
            X_sym = sqrt_diag[:, None] * X
            M = (1.0 / n) * (X_sym @ X_sym.T)
            eigenvalues = np.linalg.eigvalsh(M)
            pbar.update(1)
            if cb:
                cb(4, total_steps)

            # Separate the zero/near-zero atom from the bulk for plotting.
            zero_tol = 1e-8 * max(1.0, float(eigenvalues.max()))
            zero_mask = eigenvalues <= zero_tol
            zero_count = int(zero_mask.sum())
            zero_mass = zero_count / len(eigenvalues)
            bulk = eigenvalues[~zero_mask]
            if bulk.size >= 2 and np.ptp(bulk) > 1e-12:
                kde = gaussian_kde(bulk)
                bulk_lo = float(bulk.min())
                bulk_hi = float(bulk.max())
                x_vals = np.linspace(bulk_lo, bulk_hi, 400)
                # Plotting only the bulk: histogram (probability density), KDE, and
                # theoretical density all integrate to 1 over the bulk range.
                kde_vals = kde(x_vals)
            else:
                x_vals = np.array([])
                kde_vals = np.array([])
            pbar.update(1)
            if cb:
                cb(5, total_steps)

        ev_min = float(np.min(eigenvalues))
        ev_max = float(np.max(eigenvalues))
        # Theoretical Generalized-MP density. The simulation uses an effective aspect
        # ratio y_eff = max(y, 1/y); the formula's y matches that. Since we plot
        # only the bulk (zeros excluded), rescale the continuous part up by y_eff so
        # it integrates to 1 over the bulk and lines up with the empirical histogram.
        y_theory = y_eff
        bulk_min = float(bulk.min()) if bulk.size else max(ev_min, 1e-6)
        bulk_max = float(bulk.max()) if bulk.size else max(ev_max, bulk_min + 1e-6)
        theory_x = np.linspace(bulk_min, bulk_max, 800)
        theory_y = generalized_mp_density(theory_x, y_theory, beta, a)
        if y_theory > 1.0:
            theory_y = theory_y * y_theory

        stats = {
            "min": ev_min,
            "max": ev_max,
            "mean": float(np.mean(eigenvalues)),
            "std": float(np.std(eigenvalues)),
            "zero_count": zero_count,
            "zero_mass": zero_mass,
            "y_eff": y_theory,
        }
        explanation = (
            "Histogram of eigenvalues for B_n = S_n T_n with mixed diagonal entries, "
            "overlaid with the theoretical Generalized Marchenko-Pastur density "
            "f_{y,H}(z) for H = β·δ_a + (1−β)·δ_1 (continuous part). When y_eff > 1, "
            "the spectrum has a point mass at zero of size 1 − 1/y_eff."
        )
        self._log("eigen:done", {"stats": stats})
        # Plot only the nonzero (bulk) eigenvalues; the point mass at zero is
        # tracked in stats but excluded from the visualization.
        return {
            "data": {
                "eigenvalues": bulk.tolist(),
                "kde_x": x_vals.tolist(),
                "kde_y": kde_vals.tolist(),
                "theory_x": theory_x.tolist(),
                "theory_y": theory_y.tolist(),
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
            print(f"Error listing folders: {e}")
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
            print(f"Error getting folder contents: {e}")
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
            print(f"Error importing folder: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"Error downloading images: {e}")
            return {"success": False, "error": str(e)}

    def process_images(self, params):
        """Process images with denoising methods.

        params keys:
          folder, method1, method2, noise_type, laplacian_scale, random_seed,
          (optional) mog_weights/means/sigmas,
          train_files: list[str]   -- filenames in folder used as the training matrix
          test_files:  list[str]   -- filenames to be denoised (each becomes a test column)
          batch_size:  int         -- how many test images denoise together per call
        """
        try:
            folder_name = params.get("folder")
            method1 = params.get("method1")
            method2 = params.get("method2")
            noise_type = params.get("noise_type", "laplacian")
            laplacian_scale = params.get("laplacian_scale", 0.1)
            random_seed = params.get("random_seed", None)
            train_count = params.get("train_count")
            test_count = params.get("test_count")
            batch_size = max(1, int(params.get("batch_size", 1)))

            # Step 1: Load images
            self._update_status_ui('status_process', 'Step 1/6: Loading images...')
            self._update_progress_ui('bar_process', 'status_process', 1, 6)

            folder_path = get_random_matrix_folder() / folder_name
            all_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not all_files:
                return {"success": False, "error": "No images found in folder"}

            total = len(all_files)
            n_train_req = int(train_count) if train_count is not None else total
            n_test_req = int(test_count) if test_count is not None else max(1, total - n_train_req)
            n_train_req = max(2, min(n_train_req, total))
            # Test images come from the files AFTER the training slice; if not enough remain,
            # fall back to drawing from the training slice (overlap allowed).
            remaining = total - n_train_req
            if remaining >= 1:
                n_test_eff = max(1, min(n_test_req, remaining))
                train_paths = all_files[:n_train_req]
                test_paths = all_files[n_train_req:n_train_req + n_test_eff]
            else:
                n_test_eff = max(1, min(n_test_req, n_train_req))
                train_paths = all_files[:n_train_req]
                test_paths = all_files[:n_test_eff]

            def _load(paths):
                arrs = []
                for pth in paths:
                    img = Image.open(pth).convert('L')
                    arrs.append(np.array(img, dtype=np.float32) / 255.0)
                return np.stack(arrs, axis=0) if arrs else np.zeros((0, 0, 0), dtype=np.float32)

            train_imgs = _load(train_paths)
            test_imgs = _load(test_paths)
            n_train = train_imgs.shape[0]
            n_test = test_imgs.shape[0]
            if n_train < 2:
                return {"success": False, "error": "Need at least 2 training images."}
            if n_test < 1:
                return {"success": False, "error": "Need at least 1 test image."}

            H, W = train_imgs.shape[1], train_imgs.shape[2]
            p = H * W
            y = p / float(n_train + 1)
            self._update_status_ui(
                'processing_info',
                f'<strong>Train n={n_train}</strong>, test={n_test}, batch={batch_size} '
                f'(Size: {H}×{W}, p={p}, y=p/(n+1)={y:.4f})')

            # Step 2: Add noise to a stack — done per-batch to stay deterministic with seed
            def _add_noise(stack, seed_offset=0):
                seed = (random_seed + seed_offset) if random_seed is not None else None
                if noise_type == "laplacian":
                    return add_laplacian_noise(stack, scale=laplacian_scale, seed=seed)
                elif noise_type == "mixture_gaussian":
                    weights = params.get("mog_weights", [0.6, 0.3, 0.1])
                    means = params.get("mog_means", [0.0, 0.0, 0.0])
                    sigmas = params.get("mog_sigmas", [0.1, 0.05, 0.3])
                    return add_mixture_gaussian_noise(stack, weights, means, sigmas, seed=seed)
                else:
                    return add_laplacian_noise(stack, scale=laplacian_scale, seed=seed)

            self._update_status_ui('status_process', f'Step 2/6: Noise = {noise_type} ...')
            self._update_progress_ui('bar_process', 'status_process', 2, 6)

            # Storage for outputs (test images only)
            method1_images = np.zeros_like(test_imgs)
            method2_images = np.zeros_like(test_imgs)
            noisy_test_images = np.zeros_like(test_imgs)
            method1_details_last = ""
            method2_details_last = ""
            method1_name = "M-P Law (rmt-denoise)" if method1 == "mp" else "Generalized Covariance (rmt-denoise)"
            method2_name = "M-P Law (rmt-denoise)" if method2 == "mp" else "Generalized Covariance (rmt-denoise)"

            def _run(method_code, noisy_stack, train_clean, batch_clean):
                """Returns denoised stack of shape (n_train+b, H, W) and details string.
                For gen_cov, runs once per test image (lib oracle uses single test_index)."""
                b = batch_clean.shape[0]
                if method_code == "mp":
                    d = rmt_denoise.MPLawDenoiser()
                    out = d.denoise(noisy_stack)
                    info = d.info
                    details = (f"M-P Law: σ²={info.get('sigma2', 0):.6f}, "
                               f"λ₊={info.get('threshold', 0):.4f}, "
                               f"rank={info.get('rank', 0)}, y={info.get('y', 0):.4f}")
                    return out, details
                else:
                    out = np.zeros_like(noisy_stack)
                    last_a = last_beta = None
                    for k in range(b):
                        # Replace the test-column slot with kth batch image so
                        # GeneralizedCovDenoiser sees a single test image at index -1.
                        single_stack = np.concatenate(
                            [noisy_stack[:n_train], noisy_stack[n_train + k:n_train + k + 1]],
                            axis=0)
                        d = rmt_denoise.GeneralizedCovDenoiser(
                            apply_t=True, color_resize=True, center=True)
                        single_out = d.denoise(single_stack, clean=batch_clean[k], test_index=-1)
                        out[:n_train] = single_out[:n_train]
                        out[n_train + k] = single_out[-1]
                        last_a = getattr(d, "a", None)
                        last_beta = getattr(d, "beta", None)
                    a_str = f"{last_a:.4f}" if isinstance(last_a, (int, float)) else "n/a"
                    b_str = f"{last_beta:.4f}" if isinstance(last_beta, (int, float)) else "n/a"
                    details = (f"Gen. Cov (oracle): a≈{a_str}, β≈{b_str} (last batch), "
                               f"T applied, color-resize applied")
                    return out, details

            # Iterate test images in batches
            self._update_status_ui('status_process', 'Step 3/6: Denoising in batches ...')
            self._update_progress_ui('bar_process', 'status_process', 3, 6)
            num_batches = (n_test + batch_size - 1) // batch_size
            for bi in range(num_batches):
                lo = bi * batch_size
                hi = min(lo + batch_size, n_test)
                batch_clean = test_imgs[lo:hi]
                stack_clean = np.concatenate([train_imgs, batch_clean], axis=0)
                noisy_stack = _add_noise(stack_clean, seed_offset=bi)
                noisy_test_images[lo:hi] = noisy_stack[n_train:]

                m1_out, method1_details_last = _run(method1, noisy_stack, train_imgs, batch_clean)
                m2_out, method2_details_last = _run(method2, noisy_stack, train_imgs, batch_clean)

                method1_images[lo:hi] = m1_out[n_train:]
                method2_images[lo:hi] = m2_out[n_train:]

                self._update_status_ui(
                    'status_process',
                    f'Step 3/6: Batch {bi + 1}/{num_batches} (test {lo + 1}-{hi}/{n_test})')

            self._update_status_ui('method1_info', method1_details_last)
            self._update_status_ui('method2_info', method2_details_last)
            self._update_progress_ui('bar_process', 'status_process', 4, 6)

            # Step 5: results in image form. Re-bind for downstream save block.
            images = test_imgs            # what the save block calls "images"
            noisy_images = noisy_test_images
            n = n_test
            self._update_status_ui('status_process', 'Step 5/6: Preparing images...')
            self._update_progress_ui('bar_process', 'status_process', 5, 6)

            diff = float(np.mean(np.abs(method1_images - method2_images)))
            print(f"[DEBUG] Mean abs diff Method1 vs Method2: {diff:.6f}")

            # Step 6: Save to folders and encode images
            self._update_status_ui('status_process', 'Step 6/6: Saving to folders and encoding...')
            self._update_progress_ui('bar_process', 'status_process', 6, 6)

            # Create output folders in temp with descriptive names
            temp_folder = get_temp_folder()
            noise_folder_name = "noisy_laplacian" if noise_type == "laplacian" else "noisy_mixture_gaussian"
            noisy_folder = temp_folder / noise_folder_name

            # Map method codes to folder names
            method_names = {
                "mp": "mp_law",
                "gen_cov": "gen_cov",
            }
            method1_folder = temp_folder / method_names.get(method1, method1)
            method2_folder = temp_folder / method_names.get(method2, method2)

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
                m1_img = Image.fromarray((np.clip(method1_images[i], 0, 1) * 255).astype(np.uint8), mode='L')
                m2_img = Image.fromarray((np.clip(method2_images[i], 0, 1) * 255).astype(np.uint8), mode='L')

                # Save to disk first
                noisy_path = noisy_folder / f"{i:03d}.png"
                m1_path = method1_folder / f"{i:03d}.png"
                m2_path = method2_folder / f"{i:03d}.png"

                noisy_img.save(noisy_path)
                m1_img.save(m1_path)
                m2_img.save(m2_path)

                print(f"[DEBUG] Image {i:03d} saved:")
                print(f"  Noisy: {noisy_path}")
                print(f"  Method1 ({method1}): {m1_path}")
                print(f"  Method2 ({method2}): {m2_path}")

                # Now load FROM disk and encode to base64 to ensure they're correct
                def img_to_base64_from_file(filepath):
                    with open(filepath, 'rb') as f:
                        return base64.b64encode(f.read()).decode('utf-8')

                # Also encode original for display
                def img_to_base64(img):
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')

                results.append({
                    "original": img_to_base64(orig_img),
                    "blurred": img_to_base64_from_file(noisy_path),
                    "method1": img_to_base64_from_file(m1_path),
                    "method2": img_to_base64_from_file(m2_path),
                    "method1_name": method1_name,
                    "method2_name": method2_name,
                    "noise_type_label": noise_label
                })

                if i == 0:
                    # Verify first image is different between methods
                    import hashlib
                    m1_hash = hashlib.md5(open(m1_path, 'rb').read()).hexdigest()
                    m2_hash = hashlib.md5(open(m2_path, 'rb').read()).hexdigest()
                    print(f"[DEBUG] First image verification:")
                    print(f"  Method1 hash: {m1_hash}")
                    print(f"  Method2 hash: {m2_hash}")
                    print(f"  Images are DIFFERENT: {m1_hash != m2_hash}")

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
            print(f"Error processing images: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def get_processed_image(self, index):
        """Get a specific processed image by index"""
        try:
            if hasattr(self, 'processed_results') and index < len(self.processed_results):
                return self.processed_results[index]
            return {"error": "Image not found"}
        except Exception as e:
            return {"error": str(e)}

    def list_folder_images(self, folder_name):
        """List image filenames in a folder (sorted)."""
        try:
            folder_path = get_random_matrix_folder() / folder_name
            files = sorted(
                f.name for f in folder_path.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            )
            return {"files": files}
        except Exception as e:
            print(f"Error listing folder images: {e}")
            return {"files": []}

    def list_random_matrix_folders(self):
        """List all folders in .random_matrix directory"""
        try:
            base_folder = get_random_matrix_folder()
            folders = [f.name for f in base_folder.iterdir() if f.is_dir() and f.name != "temp"]
            return {"folders": sorted(folders)}
        except Exception as e:
            print(f"Error listing folders: {e}")
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
            print(f"Error getting folder contents: {e}")
            return {"files": []}


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
      --accent: #ff6a3d;
      --accent2: #ff9f55;
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
      box-shadow: 0 4px 12px rgba(255,106,61,0.3);
      letter-spacing: 0.3px;
      font-size: 14px;
      transition: all 0.2s ease;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(255,106,61,0.4);
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
    <div class="tab" data-tab="tab-download">📁 Folder Management</div>
    <div class="tab" data-tab="tab-process">🖼️ Image Processing</div>
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

                <h4>Generalized Marchenko–Pastur density:</h4>
                <p>For <code>H = β·δ_a + (1−β)·δ_1</code> and <code>z &gt; 0</code>:</p>
                <code>A = a(z − y + 1) + z</code><br>
                <code>B = a + z − y + 1 − y·β·(a − 1)</code><br>
                <code>P = AB/(6a²z²) − A³/(27a³z³) − 1/(2az)</code><br>
                <code>Q = B/(3az) − A²/(9a²z²)</code><br>
                <code>D = P² + Q³</code>
                <p>If <code>D &gt; 0</code>: <code>f<sub>y,H</sub>(z) = (√3 / (2π y)) · (∛(P+√D) − ∛(P−√D))</code>, else 0.</p>
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

  <div id="tab-download" class="tab-content">
    <div class="container">
      <!-- Upload Local Folder -->
      <div class="card controls-panel" style="max-width: 600px; margin: 0 auto;">
        <h3>📁 Upload Local Folder</h3>

        <div class="section-title">Select Folder to Upload</div>
        <label>Choose Folder</label>
        <input id="local_folder" type="file" webkitdirectory directory multiple style="display:none;">
        <button onclick="document.getElementById('local_folder').click()" style="margin-top: 8px; width: 100%;">📂 Browse Folder</button>
        <div class="small" id="folder_count" style="margin-top: 8px; padding: 8px; background: #f1f5f9; border-radius: 4px; min-height: 40px;">No folder selected</div>

        <button onclick="importLocalFolder()" style="margin-top: 12px;">📥 Upload Folder</button>
        <div class="small" style="margin-top: 8px; color: #64748b;">Folder will be saved with its original name in .random_matrix</div>

        <div class="status" id="status_upload"></div>
        <div class="bar-wrap"><div class="bar" id="bar_upload"></div></div>
      </div>

      <!-- Browse Existing Folders Section -->
      <div class="card" style="margin-top: 20px;">
        <h3>📂 Browse Existing Folders</h3>
        <div style="display: grid; grid-template-columns: 1fr 3fr; gap: 20px; margin-top: 16px;">
          <div>
            <label>Available Folders</label>
            <select id="folder_list" size="8" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
              <option>Loading folders...</option>
            </select>
            <button onclick="refreshFolders()" style="margin-top: 8px; width: 100%;">🔄 Refresh</button>
            <button onclick="viewFolder()" style="margin-top: 8px; width: 100%;">👁️ View Contents</button>
          </div>
          <div>
            <h4 style="margin-top: 0; margin-bottom: 12px; font-size: 14px; color: #475569;">Folder Contents</h4>
            <div id="folder_contents" style="min-height: 300px; max-height: 400px; overflow-y: auto; padding: 16px; background: #f8fafc; border-radius: 8px; border: 1px solid #cbd5e1;">
              <p style="text-align: center; color: #94a3b8; padding: 40px;">Select a folder to view its contents</p>
            </div>
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

          <div class="section-title">Source Folder</div>
          <label>Select Folder</label>
          <select id="process_folder" onchange="refreshImageList()" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
            <option>Loading folders...</option>
          </select>

          <div class="section-title">Image Counts</div>
          <div id="folder_image_count" class="small" style="color:#64748b; margin-bottom:6px;">Folder: – images</div>
          <label>Training images (n) — build covariance matrix</label>
          <input id="train_count" type="number" value="20" min="2" step="1">
          <label style="margin-top:8px;">Test images (to denoise)</label>
          <input id="test_count" type="number" value="5" min="1" step="1">
          <label style="margin-top:8px;">Batch size (test images per denoise call)</label>
          <input id="batch_size" type="number" value="1" min="1" step="1">
          <div class="small" style="color:#64748b; margin-top:4px;">
            Train = first <em>n</em> files in folder; test = next files after that.
          </div>

          <div class="section-title">Method 1</div>
          <label>Denoising Method</label>
          <select id="method1" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
            <option value="mp">M-P Law (rmt-denoise)</option>
            <option value="gen_cov">Gen. Covariance (rmt-denoise)</option>
          </select>

          <div class="section-title">Method 2</div>
          <label>Denoising Method</label>
          <select id="method2" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
            <option value="mp">M-P Law (rmt-denoise)</option>
            <option value="gen_cov" selected>Gen. Covariance (rmt-denoise)</option>
          </select>

          <div class="small" style="color: #64748b; margin-top: 6px;">
            Both methods use rmt-denoise with centring (X̃ = X − X̄), T(a,β) re-scaling, and color-resize applied.
          </div>

          <div class="section-title">Noise Settings</div>
          <label>Noise Type</label>
          <select id="noise_type" onchange="toggleNoiseParams()" style="width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;">
            <option value="laplacian">Laplacian (heavy-tailed)</option>
            <option value="mixture_gaussian">Mixture of Gaussians</option>
          </select>

          <!-- Laplacian Noise Parameters -->
          <div id="laplacian_params" style="margin-top: 12px;">
            <label>Laplacian Scale (σ)</label>
            <input id="laplacian_scale" type="number" value="0.1" step="0.01" min="0.01" max="1.0">
            <div class="small" style="color: #64748b; margin-top: 4px;">Heavy-tailed noise (default: 0.1)</div>
          </div>

          <!-- Mixture of Gaussians Parameters -->
          <div id="mog_params" style="margin-top: 12px; display: none;">
            <div class="small" style="color: #64748b; margin-bottom: 8px;">Configure 3 Gaussian components (weights auto-normalized)</div>

            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
              <div style="font-weight: 600; color: #475569; margin-bottom: 8px;">Component 1</div>
              <label style="font-size: 12px;">Weight</label>
              <input id="mog_weight1" type="number" value="0.6" step="0.1" min="0.01" max="1.0" style="margin-bottom: 6px;">
              <label style="font-size: 12px;">Mean (μ)</label>
              <input id="mog_mean1" type="number" value="0.0" step="0.01" min="-0.5" max="0.5" style="margin-bottom: 6px;">
              <label style="font-size: 12px;">Sigma (σ)</label>
              <input id="mog_sigma1" type="number" value="0.1" step="0.01" min="0.001" max="1.0">
            </div>

            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
              <div style="font-weight: 600; color: #475569; margin-bottom: 8px;">Component 2</div>
              <label style="font-size: 12px;">Weight</label>
              <input id="mog_weight2" type="number" value="0.3" step="0.1" min="0.01" max="1.0" style="margin-bottom: 6px;">
              <label style="font-size: 12px;">Mean (μ)</label>
              <input id="mog_mean2" type="number" value="0.0" step="0.01" min="-0.5" max="0.5" style="margin-bottom: 6px;">
              <label style="font-size: 12px;">Sigma (σ)</label>
              <input id="mog_sigma2" type="number" value="0.05" step="0.01" min="0.001" max="1.0">
            </div>

            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
              <div style="font-weight: 600; color: #475569; margin-bottom: 8px;">Component 3 (Heavy tail)</div>
              <label style="font-size: 12px;">Weight</label>
              <input id="mog_weight3" type="number" value="0.1" step="0.1" min="0.01" max="1.0" style="margin-bottom: 6px;">
              <label style="font-size: 12px;">Mean (μ)</label>
              <input id="mog_mean3" type="number" value="0.0" step="0.01" min="-0.5" max="0.5" style="margin-bottom: 6px;">
              <label style="font-size: 12px;">Sigma (σ)</label>
              <input id="mog_sigma3" type="number" value="0.3" step="0.01" min="0.001" max="1.0">
            </div>
          </div>

          <label>Random Seed (for reproducibility)</label>
          <input id="random_seed" type="number" value="" placeholder="Leave empty for random" step="1" min="0">
          <div class="small" style="color: #64748b; margin-top: 4px;">Set a seed number (e.g., 42) to get identical noise each time. Leave empty for random noise.</div>

          <button onclick="processImages()">🚀 Process Images</button>
          <div class="small">y parameter auto-calculated from image dimensions</div>

          <div class="status" id="status_process"></div>
          <div class="bar-wrap"><div class="bar" id="bar_process"></div></div>

          <div class="section-title" style="margin-top: 20px;">Processing Details</div>
          <div class="small" id="processing_info" style="background: #f1f5f9; padding: 10px; min-height: 40px;">
            Processing info will appear here...
          </div>
          <div class="small" id="method1_info" style="background: #f1f5f9; padding: 8px; margin-top: 6px; min-height: 30px; font-size: 11px;">
            Method 1 details...
          </div>
          <div class="small" id="method2_info" style="background: #f1f5f9; padding: 8px; margin-top: 6px; min-height: 30px; font-size: 11px;">
            Method 2 details...
          </div>
        </div>

        <!-- Display Area with 4-panel grid -->
        <div class="plot-area">
          <div class="card">
            <h3>📊 Comparison Results</h3>
            <div class="grid-2col" style="margin-top: 0;">
              <div style="text-align: center;">
                <h4 id="result1_label" style="margin: 8px 0; font-size: 13px; color: #475569;">Method 1 Result</h4>
                <div id="result1" style="width: 100%; height: 300px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #94a3b8;">
                  Click "Process Images" to start
                </div>
              </div>
              <div style="text-align: center;">
                <h4 id="result2_label" style="margin: 8px 0; font-size: 13px; color: #475569;">Method 2 Result</h4>
                <div id="result2" style="width: 100%; height: 300px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #94a3b8;">
                  Click "Process Images" to start
                </div>
              </div>
              <div style="text-align: center;">
                <h4 style="margin: 8px 0; font-size: 13px; color: #475569;">Original Image</h4>
                <div id="result_original" style="width: 100%; height: 300px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #94a3b8;">
                  Waiting...
                </div>
              </div>
              <div style="text-align: center;">
                <h4 id="result_blur_label" style="margin: 8px 0; font-size: 13px; color: #475569;">With Noise</h4>
                <div id="result_blur" style="width: 100%; height: 300px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #94a3b8;">
                  Waiting...
                </div>
              </div>
            </div>

            <div style="text-align: center; margin-top: 20px; display: flex; align-items: center; justify-content: center; gap: 16px;">
              <button onclick="prevImage()" style="width: 60px; padding: 10px; font-size: 18px;">&lt;</button>
              <div style="font-size: 20px; font-weight: 600; color: #0f172a; font-family: 'Courier New', monospace;" id="image_counter">000</div>
              <button onclick="nextImage()" style="width: 60px; padding: 10px; font-size: 18px;">&gt;</button>
            </div>
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
            autorange: true
          },
          yaxis: {
            title: 'Im(s)',
            autorange: true
          },
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: 'white',
          paper_bgcolor: 'white'
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
          xaxis: {title: 'beta', autorange: true},
          yaxis: {title: 'Im(s)', autorange: true},
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: 'white',
          paper_bgcolor: 'white'
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
          xaxis: {title: 'beta', autorange: true},
          yaxis: {title: 'Re(s)', autorange: true},
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: 'white',
          paper_bgcolor: 'white'
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
          line: {color: 'black', width: 2}
        }], {
          title: `Discriminant vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
          xaxis: {title: 'beta', autorange: true},
          yaxis: {title: 'Discriminant', autorange: true},
          hovermode: 'x unified',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: 'white',
          paper_bgcolor: 'white'
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

        const {eigenvalues, kde_x, kde_y, theory_x, theory_y} = res.data;

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
            name: 'Histogram (bulk)',
            marker: {color: '#4C7EF3', opacity: 0.65}
          }
        ];
        if (kde_x && kde_x.length) {
          traces.push({
            x: kde_x,
            y: kde_y,
            mode: 'lines',
            name: 'KDE',
            line: {color: '#FF8042', width: 3}
          });
        }
        if (theory_x && theory_y) {
          traces.push({
            x: theory_x,
            y: theory_y,
            mode: 'lines',
            name: 'Generalized MP density',
            line: {color: '#2CA02C', width: 3, dash: 'dash'}
          });
        }

        Plotly.newPlot('slot6', traces, {
          title: `Eigenvalue distribution (y=${payload.p / payload.n}, beta=${payload.beta}, a=${payload.a})`,
          xaxis: {title: 'Eigenvalue', autorange: true},
          yaxis: {title: 'Density', autorange: true},
          hovermode: 'closest',
          height: 450,
          margin: {l: 60, r: 40, t: 60, b: 50},
          plot_bgcolor: 'white',
          paper_bgcolor: 'white'
        }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

        document.getElementById('eig_stats').innerHTML = `
          <div class="metric">min: ${res.stats.min.toFixed(4)}</div>
          <div class="metric">max: ${res.stats.max.toFixed(4)}</div>
          <div class="metric">mean: ${res.stats.mean.toFixed(4)}</div>
          <div class="metric">std: ${res.stats.std.toFixed(4)}</div>
          <div class="metric">zeros: ${res.stats.zero_count} (${(res.stats.zero_mass*100).toFixed(1)}%)</div>
          <div class="metric">y_eff: ${res.stats.y_eff.toFixed(3)}</div>
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

    async function refreshImageList() {
      if (!apiReady()) { setTimeout(refreshImageList, 500); return; }
      const folder = document.getElementById('process_folder').value;
      const info = document.getElementById('folder_image_count');
      if (!folder || folder === 'Loading folders...' || folder === 'No folders found') {
        if (info) info.textContent = 'Folder: – images';
        return;
      }
      try {
        const res = await window.pywebview.api.list_folder_images(folder);
        const n = (res.files || []).length;
        if (info) info.textContent = `Folder "${folder}": ${n} images available`;
      } catch (err) { console.error('list_folder_images failed', err); }
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
      if (!ensureApi('status_upload')) return;
      const files = document.getElementById('local_folder').files;
      if (files.length === 0) {
        setHTML('status_upload', '<span style="color:orange;">No folder selected</span>');
        return;
      }

      try {
        // Get folder name from first file's path
        const path = files[0].webkitRelativePath || files[0].name;
        const sourceFolderName = path.split('/')[0];

        // Filter image files only
        const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));

        if (imageFiles.length === 0) {
          setHTML('status_upload', '<span style="color:orange;">No image files found in folder</span>');
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
          document.getElementById('bar_upload').style.width = progress + '%';
          setHTML('status_upload', `Reading files: ${i + 1}/${imageFiles.length} (${progress}%)`);
        }

        const payload = {
          files: fileData,
          folder: sourceFolderName  // Use original folder name
        };

        setHTML('status_upload', `Uploading ${imageFiles.length} images...`);
        const res = await window.pywebview.api.import_local_folder(payload);

        if (res && res.success) {
          setHTML('status_upload', `<span style="color:green;">✓ Imported ${res.count} images to folder "${res.folder}"</span>`);
          document.getElementById('folder_count').innerHTML = 'No folder selected';
          document.getElementById('local_folder').value = '';
          document.getElementById('bar_upload').style.width = '100%';
          refreshFolders();
        } else {
          const errorMsg = (res && res.error) ? res.error : 'Unknown error';
          setHTML('status_upload', `<span style="color:red;">Error: ${errorMsg}</span>`);
          document.getElementById('bar_upload').style.width = '0%';
        }
      } catch (err) {
        console.error('Upload error:', err);
        setHTML('status_upload', '<span style="color:red;">Error: '+err.message+'</span>');
        document.getElementById('bar_upload').style.width = '0%';
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

        if (!folderList || !processFolderList) {
          console.log('DOM elements not ready, retrying...');
          setTimeout(refreshFolders, 500);
          return;
        }

        folderList.innerHTML = '';
        processFolderList.innerHTML = '';

        if (res.folders.length === 0) {
          folderList.innerHTML = '<option>No folders found</option>';
          processFolderList.innerHTML = '<option>No folders found</option>';
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
          });
        }
        console.log(`Loaded ${res.folders.length} folders`);
        // Auto-load image list for the currently selected processing folder
        refreshImageList();
      } catch (err) {
        console.error('Error refreshing folders:', err);
        // Retry on error
        setTimeout(refreshFolders, 1000);
      }
    }

    async function viewFolder() {
      if (!ensureApi('status_download')) return;
      const folderList = document.getElementById('folder_list');
      const selectedFolder = folderList.value;

      if (!selectedFolder || selectedFolder === 'Loading folders...' || selectedFolder === 'No folders found') {
        return;
      }

      try {
        const res = await window.pywebview.api.get_folder_contents(selectedFolder);
        const contentsDiv = document.getElementById('folder_contents');

        if (res.files.length === 0) {
          contentsDiv.innerHTML = '<p style="text-align: center; color: #94a3b8; padding: 40px;">Folder is empty</p>';
        } else {
          let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px;">';
          res.files.forEach((file, idx) => {
            html += `
              <div style="text-align: center;">
                <img src="data:image/png;base64,${file.data}" style="width: 100%; height: 120px; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 6px; background: white;" />
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">${file.name}</div>
              </div>
            `;
          });
          html += '</div>';
          contentsDiv.innerHTML = html;
        }
      } catch (err) {
        document.getElementById('folder_contents').innerHTML = '<p style="color: red;">Error loading folder contents</p>';
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
          random_seed: randomSeedInput ? Number(randomSeedInput) : null,
          train_count: Math.max(2, Number(document.getElementById('train_count').value) || 2),
          test_count: Math.max(1, Number(document.getElementById('test_count').value) || 1),
          batch_size: Math.max(1, Number(document.getElementById('batch_size').value) || 1)
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
      document.getElementById('result1').innerHTML = `<img src="data:image/png;base64,${data.method1}" style="width: 100%; height: 100%; object-fit: contain;" />`;
      document.getElementById('result2').innerHTML = `<img src="data:image/png;base64,${data.method2}" style="width: 100%; height: 100%; object-fit: contain;" />`;
      document.getElementById('result_original').innerHTML = `<img src="data:image/png;base64,${data.original}" style="width: 100%; height: 100%; object-fit: contain;" />`;
      document.getElementById('result_blur').innerHTML = `<img src="data:image/png;base64,${data.blurred}" style="width: 100%; height: 100%; object-fit: contain;" />`;

      // Update labels with method names if available
      if (data.method1_name) {
        document.getElementById('result1_label').textContent = data.method1_name;
      }
      if (data.method2_name) {
        document.getElementById('result2_label').textContent = data.method2_name;
      }
      if (data.noise_type_label) {
        document.getElementById('result_blur_label').textContent = data.noise_type_label;
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
    import sys, os as _os
    _log = _os.path.join(_os.environ.get("TEMP", "."), "gcm_crash.log")
    def _ck(msg):
        with open(_log, "a", encoding="utf-8") as _f:
            _f.write(f"checkpoint: {msg}\n")
            _f.flush()

    # Bump default stack size for Python-created threads. Without this, pythonnet's
    # CLR initialization on background threads overflows the default 1 MB stack
    # (the main-thread reserve from /STACK doesn't propagate to threading.Thread).
    import threading as _threading
    try:
        _threading.stack_size(16 * 1024 * 1024)
    except (RuntimeError, ValueError):
        pass

    if sys.platform == 'win32':
        # Redirect to a log file (not /dev/null) so pywebview's logger.exception
        # output is captured — otherwise JS-bridge setup failures are invisible.
        try:
            sys.stderr = open(
                _os.path.join(_os.environ.get("TEMP", "."), "gcm_stderr.log"),
                "w", encoding="utf-8", buffering=1)
        except Exception:
            sys.stderr = open(os.devnull, 'w')
    # Mirror webview's own logger to a file so logger.exception calls inside
    # generate_js_object actually surface.
    import logging as _logging
    _wlh = _logging.FileHandler(
        _os.path.join(_os.environ.get("TEMP", "."), "gcm_webview.log"),
        mode="w", encoding="utf-8")
    _wlh.setLevel(_logging.DEBUG)
    _wlh.setFormatter(_logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _logging.getLogger("pywebview").addHandler(_wlh)
    _logging.getLogger("pywebview").setLevel(_logging.DEBUG)
    _ck("after stderr redirect")

    get_random_matrix_folder()
    _ck("after get_random_matrix_folder")

    try:
        import pythonnet as _pn
        _ck("imported pythonnet")
        _pn.load()
        _ck("pythonnet.load done")
        import clr  # noqa: F401
        _ck("imported clr")
    except Exception:
        import traceback as _tb
        with open(_log, "a", encoding="utf-8") as _f:
            _f.write("\n--- pythonnet load failure ---\n")
            _tb.print_exc(file=_f)
        raise

    atexit.register(cleanup_temp_folder)
    _ck("registered cleanup")

    api = Bridge(None)
    _ck("created Bridge")

    # Note: we do NOT pass js_api=api. pywebview's get_functions walks
    # `api.window = window` and recurses into Window's internals (events,
    # locks, threads), which under Nuitka-compiled methods leads to
    # `generate_func` hanging forever and JS never receiving the API. We
    # register every public method via window.expose() below instead.
    window = webview.create_window(
        "Matrix Analysis Lab",
        html=HTML,
        width=1300,
        height=900,
    )
    _ck("created window")

    api.window = window

    # Nuitka compiles methods to a private `compiled_method` type that
    # pywebview's util.py rejects via inspect.ismethod/isfunction, so the JS
    # side ends up with `window.pywebview.api.<name> is not a function`.
    # Build plain Python wrappers (signature `(payload=None)`) for every
    # public Bridge method and expose those — wrapped functions are real
    # `function` objects whose introspection works in both source and
    # Nuitka-compiled mode.
    _api_methods = [
        "im_vs_z",
        "roots_vs_beta",
        "eigen_distribution",
        "list_random_matrix_folders",
        "list_folder_images",
        "get_folder_contents",
        "import_local_folder",
        "download_images_playwright",
        "process_images",
        "get_processed_image",
    ]
    def _make_wrapper(_method, _name):
        # *args swallows whatever JS sends. pywebview reports `params=[]` to
        # JS based on the wrapper's argspec, but the JS shim forwards the
        # actual call args via Array.prototype.slice.call(arguments) anyway,
        # so 0-arg and 1-arg JS calls both reach the real method correctly.
        def _wrapper(*args):
            return _method(*args)
        _wrapper.__name__ = _name
        return _wrapper

    _wrappers = []
    for _n in _api_methods:
        _m = getattr(api, _n, None)
        if _m is None:
            _ck(f"WARNING: api.{_n} missing")
            continue
        _wrappers.append(_make_wrapper(_m, _n))
    window.expose(*_wrappers)
    _ck(f"exposed {len(_wrappers)} api methods: "
        + ",".join(getattr(w, "__name__", "?") for w in _wrappers))
    _ck(f"window._functions registry size: {len(window._functions)}")

    _ck("about to call webview.start")

    webview.start()
    _ck("webview.start returned")


if __name__ == "__main__":
    import os, sys, traceback
    _log = os.path.join(os.environ.get("TEMP", "."), "gcm_crash.log")
    try:
        with open(_log, "w", encoding="utf-8") as _f:
            _f.write("startup\n")
        main()
    except SystemExit:
        raise
    except BaseException:
        with open(_log, "a", encoding="utf-8") as _f:
            _f.write("\n--- exception ---\n")
            traceback.print_exc(file=_f)
        sys.exit(1)
