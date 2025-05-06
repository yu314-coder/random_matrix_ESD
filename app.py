import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
import sys
import os
import importlib.util

# Configure Streamlit for Hugging Face Spaces - THIS MUST COME FIRST
st.set_page_config(
    page_title="Cubic Root Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Try to import C++ module
try:
    import cubic_cpp
    cpp_available = True
except ImportError:
    cpp_available = False
    st.warning("⚠️ C++ acceleration unavailable. Using slower Python implementation.")

def add_sqrt_support(expr_str):
    """Replace 'sqrt(' with 'sp.sqrt(' for sympy compatibility"""
    return expr_str.replace('sqrt(', 'sp.sqrt(')

#############################
# 1) Define the discriminant
#############################

# Symbolic variables for the cubic discriminant
z_sym, beta_sym, z_a_sym, y_sym = sp.symbols("z beta z_a y", real=True, positive=True)

# Define coefficients a, b, c, d in terms of z_sym, beta_sym, z_a_sym, y_sym
a_sym = z_sym * z_a_sym
b_sym = z_sym * z_a_sym + z_sym + z_a_sym - z_a_sym*y_sym
c_sym = z_sym + z_a_sym + 1 - y_sym*(beta_sym*z_a_sym + 1 - beta_sym)
d_sym = 1

# Symbolic expression for the cubic discriminant
Delta_expr = (
    ((b_sym*c_sym)/(6*a_sym**2) - (b_sym**3)/(27*a_sym**3) - d_sym/(2*a_sym))**2
    + (c_sym/(3*a_sym) - (b_sym**2)/(9*a_sym**2))**3
)

# Define fallback Python implementations for all functions
# These will be used if C++ module is unavailable

def discriminant_func_py(z, beta, z_a, y):
    """Fast numeric function for the discriminant"""
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Coefficients
    a = z * z_a
    b = z * z_a + z + z_a - z_a*y_effective
    c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta)
    d = 1
    
    # Calculate the discriminant
    return ((b*c)/(6*a**2) - (b**3)/(27*a**3) - d/(2*a))**2 + (c/(3*a) - (b**2)/(9*a**2))**3

@st.cache_data
def find_z_at_discriminant_zero_py(z_a, y, beta, z_min, z_max, steps):
    """
    Scan z in [z_min, z_max] for sign changes in the discriminant,
    and return approximated roots (where the discriminant is zero).
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    z_grid = np.linspace(z_min, z_max, steps)
    disc_vals = np.array([discriminant_func_py(z, beta, z_a, y_effective) for z in z_grid])
    roots_found = []
    for i in range(len(z_grid) - 1):
        f1, f2 = disc_vals[i], disc_vals[i+1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 == 0.0:
            roots_found.append(z_grid[i])
        elif f2 == 0.0:
            roots_found.append(z_grid[i+1])
        elif f1 * f2 < 0:
            zl, zr = z_grid[i], z_grid[i+1]
            for _ in range(50):
                mid = 0.5 * (zl + zr)
                fm = discriminant_func_py(mid, beta, z_a, y_effective)
                if fm == 0:
                    zl = zr = mid
                    break
                if np.sign(fm) == np.sign(f1):
                    zl, f1 = mid, fm
                else:
                    zr, f2 = mid, fm
            roots_found.append(0.5 * (zl + zr))
    return np.array(roots_found)

@st.cache_data
def sweep_beta_and_find_z_bounds_py(z_a, y, z_min, z_max, beta_steps, z_steps):
    """
    For each beta in [0,1] (with beta_steps points), find the minimum and maximum z 
    for which the discriminant is zero.
    Returns: betas, lower z*(β) values, and upper z*(β) values.
    """
    betas = np.linspace(0, 1, beta_steps)
    z_min_values = []
    z_max_values = []
    for b in betas:
        roots = find_z_at_discriminant_zero_py(z_a, y, b, z_min, z_max, z_steps)
        if len(roots) == 0:
            z_min_values.append(np.nan)
            z_max_values.append(np.nan)
        else:
            z_min_values.append(np.min(roots))
            z_max_values.append(np.max(roots))
    return betas, np.array(z_min_values), np.array(z_max_values)

@st.cache_data
def compute_eigenvalue_support_boundaries_py(z_a, y, beta_values, n_samples=100, seeds=5):
    """
    Compute the support boundaries of the eigenvalue distribution by directly
    finding the minimum and maximum eigenvalues of B_n = S_n T_n for different beta values.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    min_eigenvalues = np.zeros_like(beta_values)
    max_eigenvalues = np.zeros_like(beta_values)
    
    # Use a progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, beta in enumerate(beta_values):
        # Update progress
        progress_bar.progress((i + 1) / len(beta_values))
        status_text.text(f"Processing β = {beta:.2f} ({i+1}/{len(beta_values)})")
            
        min_vals = []
        max_vals = []
        
        # Run multiple trials with different seeds for more stable results
        for seed in range(seeds):
            # Set random seed
            np.random.seed(seed * 100 + i)
            
            # Compute dimension p based on aspect ratio y
            n = n_samples
            p = int(y_effective * n)
            
            # Constructing T_n (Population / Shape Matrix)
            k = int(np.floor(beta * p))
            diag_entries = np.concatenate([
                np.full(k, z_a),
                np.full(p - k, 1.0)
            ])
            np.random.shuffle(diag_entries)
            T_n = np.diag(diag_entries)
            
            # Generate the data matrix X with i.i.d. standard normal entries
            X = np.random.randn(p, n)
            
            # Compute the sample covariance matrix S_n = (1/n) * XX^T
            S_n = (1 / n) * (X @ X.T)
            
            # Compute B_n = S_n T_n
            B_n = S_n @ T_n
            
            # Compute eigenvalues of B_n
            eigenvalues = np.linalg.eigvalsh(B_n)
            
            # Find minimum and maximum eigenvalues
            min_vals.append(np.min(eigenvalues))
            max_vals.append(np.max(eigenvalues))
        
        # Average over seeds for stability
        min_eigenvalues[i] = np.mean(min_vals)
        max_eigenvalues[i] = np.mean(max_vals)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return min_eigenvalues, max_eigenvalues

@st.cache_data
def compute_cubic_roots_py(z, beta, z_a, y):
    """
    Compute the roots of the cubic equation for given parameters.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Coefficients in the form as^3 + bs^2 + cs + d = 0
    a = z * z_a
    b = z * z_a + z + z_a - z_a*y_effective
    c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta)
    d = 1
    
    # Handle special cases
    if abs(a) < 1e-10:
        if abs(b) < 1e-10:  # Linear case
            roots = np.array([-d/c, 0, 0], dtype=complex)
        else:  # Quadratic case
            quad_roots = np.roots([b, c, d])
            roots = np.append(quad_roots, 0).astype(complex)
        return roots
    
    # Standard cubic case
    coeffs = [a, b, c, d]
    return np.roots(coeffs)

@st.cache_data
def compute_high_y_curve_py(betas, z_a, y):
    """
    Compute the "High y Expression" curve.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z_a
    betas = np.array(betas)
    denominator = 1 - 2*a
    if denominator == 0:
        return np.full_like(betas, np.nan)
    numerator = -4*a*(a-1)*y_effective*betas - 2*a*y_effective - 2*a*(2*a-1)
    return numerator/denominator

@st.cache_data
def compute_alternate_low_expr_py(betas, z_a, y):
    """
    Compute the alternate low expression.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    betas = np.array(betas)
    return (z_a * y_effective * betas * (z_a - 1) - 2*z_a*(1 - y_effective) - 2*z_a**2) / (2 + 2*z_a)

@st.cache_data
def compute_max_k_expression_py(betas, z_a, y, k_samples=1000):
    """
    Compute max_{k ∈ (0,∞)} (y*beta*(a-1)*k + (a*k+1)*((y-1)*k-1)) / ((a*k+1)*(k^2+k))
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z_a
    # Sample k values on a logarithmic scale
    k_values = np.logspace(-3, 3, k_samples)
    
    max_vals = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        values = np.zeros_like(k_values)
        for j, k in enumerate(k_values):
            numerator = y_effective*beta*(a-1)*k + (a*k+1)*((y_effective-1)*k-1)
            denominator = (a*k+1)*(k**2+k)
            if abs(denominator) < 1e-10:
                values[j] = np.nan
            else:
                values[j] = numerator/denominator
        
        valid_indices = ~np.isnan(values)
        if np.any(valid_indices):
            max_vals[i] = np.max(values[valid_indices])
        else:
            max_vals[i] = np.nan
            
    return max_vals

@st.cache_data
def compute_min_t_expression_py(betas, z_a, y, t_samples=1000):
    """
    Compute min_{t ∈ (-1/a, 0)} (y*beta*(a-1)*t + (a*t+1)*((y-1)*t-1)) / ((a*t+1)*(t^2+t))
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z_a
    if a <= 0:
        return np.full_like(betas, np.nan)
        
    lower_bound = -1/a + 1e-10  # Avoid division by zero
    t_values = np.linspace(lower_bound, -1e-10, t_samples)
    
    min_vals = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        values = np.zeros_like(t_values)
        for j, t in enumerate(t_values):
            numerator = y_effective*beta*(a-1)*t + (a*t+1)*((y_effective-1)*t-1)
            denominator = (a*t+1)*(t**2+t)
            if abs(denominator) < 1e-10:
                values[j] = np.nan
            else:
                values[j] = numerator/denominator
        
        valid_indices = ~np.isnan(values)
        if np.any(valid_indices):
            min_vals[i] = np.min(values[valid_indices])
        else:
            min_vals[i] = np.nan
            
    return min_vals

@st.cache_data
def compute_derivatives_py(curve, betas):
    """Compute first and second derivatives of a curve"""
    d1 = np.gradient(curve, betas)
    d2 = np.gradient(d1, betas)
    return d1, d2

@st.cache_data
def generate_eigenvalue_distribution_py(beta, y, z_a, n=1000, seed=42):
    """
    Generate the eigenvalue distribution of B_n = S_n T_n as n→∞
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Set random seed
    np.random.seed(seed)
    
    # Compute dimension p based on aspect ratio y
    p = int(y_effective * n)
    
    # Constructing T_n (Population / Shape Matrix)
    k = int(np.floor(beta * p))
    diag_entries = np.concatenate([
        np.full(k, z_a),
        np.full(p - k, 1.0)
    ])
    np.random.shuffle(diag_entries)
    T_n = np.diag(diag_entries)
    
    # Generate the data matrix X with i.i.d. standard normal entries
    X = np.random.randn(p, n)
    
    # Compute the sample covariance matrix S_n = (1/n) * XX^T
    S_n = (1 / n) * (X @ X.T)
    
    # Compute B_n = S_n T_n
    B_n = S_n @ T_n
    
    # Compute eigenvalues of B_n
    eigenvalues = np.linalg.eigvalsh(B_n)
    return eigenvalues

# Use C++ implementations if available, otherwise use Python implementations
if cpp_available:
    discriminant_func = cubic_cpp.discriminant_func
    find_z_at_discriminant_zero = cubic_cpp.find_z_at_discriminant_zero
    sweep_beta_and_find_z_bounds = cubic_cpp.sweep_beta_and_find_z_bounds
    compute_eigenvalue_support_boundaries = cubic_cpp.compute_eigenvalue_support_boundaries
    compute_cubic_roots = cubic_cpp.compute_cubic_roots
    compute_high_y_curve = cubic_cpp.compute_high_y_curve
    compute_alternate_low_expr = cubic_cpp.compute_alternate_low_expr
    compute_max_k_expression = cubic_cpp.compute_max_k_expression
    compute_min_t_expression = cubic_cpp.compute_min_t_expression
    compute_derivatives = cubic_cpp.compute_derivatives
    generate_eigenvalue_distribution = lambda beta, y, z_a, n=1000, seed=42: cubic_cpp.generate_eigenvalue_distribution(beta, y, z_a, n, seed)
else:
    discriminant_func = discriminant_func_py
    find_z_at_discriminant_zero = find_z_at_discriminant_zero_py
    sweep_beta_and_find_z_bounds = sweep_beta_and_find_z_bounds_py
    compute_eigenvalue_support_boundaries = compute_eigenvalue_support_boundaries_py
    compute_cubic_roots = compute_cubic_roots_py
    compute_high_y_curve = compute_high_y_curve_py
    compute_alternate_low_expr = compute_alternate_low_expr_py
    compute_max_k_expression = compute_max_k_expression_py
    compute_min_t_expression = compute_min_t_expression_py
    compute_derivatives = compute_derivatives_py
    generate_eigenvalue_distribution = generate_eigenvalue_distribution_py

def compute_all_derivatives(betas, z_mins, z_maxs, low_y_curve, high_y_curve, alt_low_expr, custom_curve1=None, custom_curve2=None):
    """Compute derivatives for all curves"""
    derivatives = {}
    
    # Upper z*(β)
    derivatives['upper'] = compute_derivatives(z_maxs, betas)
    
    # Lower z*(β)
    derivatives['lower'] = compute_derivatives(z_mins, betas)
    
    # Low y Expression (only if provided)
    if low_y_curve is not None:
        derivatives['low_y'] = compute_derivatives(low_y_curve, betas)
    
    # High y Expression
    if high_y_curve is not None:
        derivatives['high_y'] = compute_derivatives(high_y_curve, betas)
    
    # Alternate Low Expression
    if alt_low_expr is not None:
        derivatives['alt_low'] = compute_derivatives(alt_low_expr, betas)
    
    # Custom Expression 1 (if provided)
    if custom_curve1 is not None:
        derivatives['custom1'] = compute_derivatives(custom_curve1, betas)

    # Custom Expression 2 (if provided)
    if custom_curve2 is not None:
        derivatives['custom2'] = compute_derivatives(custom_curve2, betas)
        
    return derivatives

def compute_custom_expression(betas, z_a, y, s_num_expr, s_denom_expr, is_s_based=True):
    """
    Compute custom curve. If is_s_based=True, compute using s substitution.
    Otherwise, compute direct z(β) expression.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    beta_sym, z_a_sym, y_sym = sp.symbols("beta z_a y", positive=True)
    local_dict = {"beta": beta_sym, "z_a": z_a_sym, "y": y_sym, "sp": sp}
    
    try:
        # Add sqrt support
        s_num_expr = add_sqrt_support(s_num_expr)
        s_denom_expr = add_sqrt_support(s_denom_expr)
        
        num_expr = sp.sympify(s_num_expr, locals=local_dict)
        denom_expr = sp.sympify(s_denom_expr, locals=local_dict)
        
        if is_s_based:
            # Compute s and substitute into main expression
            s_expr = num_expr / denom_expr
            a = z_a_sym
            numerator = y_sym*beta_sym*(z_a_sym-1)*s_expr + (a*s_expr+1)*((y_sym-1)*s_expr-1)
            denominator = (a*s_expr+1)*(s_expr**2 + s_expr)
            final_expr = numerator/denominator
        else:
            # Direct z(β) expression
            final_expr = num_expr / denom_expr
            
    except sp.SympifyError as e:
        st.error(f"Error parsing expressions: {e}")
        return np.full_like(betas, np.nan)
    
    final_func = sp.lambdify((beta_sym, z_a_sym, y_sym), final_expr, modules=["numpy"])
    with np.errstate(divide='ignore', invalid='ignore'):
        result = final_func(betas, z_a, y_effective)
        if np.isscalar(result):
            result = np.full_like(betas, result)
    return result

def generate_z_vs_beta_plot(z_a, y, z_min, z_max, beta_steps, z_steps,
                          s_num_expr=None, s_denom_expr=None, 
                          z_num_expr=None, z_denom_expr=None,
                          show_derivatives=False,
                          show_high_y=False,
                          show_low_y=False,
                          show_max_k=True,
                          show_min_t=True,
                          use_eigenvalue_method=True,
                          n_samples=1000,
                          seeds=5):
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None

    betas = np.linspace(0, 1, beta_steps)
    
    if use_eigenvalue_method:
        # Use the eigenvalue method to compute boundaries
        st.info("Computing eigenvalue support boundaries. This may take a moment...")
        min_eigs, max_eigs = compute_eigenvalue_support_boundaries(z_a, y, betas, n_samples, seeds)
        z_mins, z_maxs = min_eigs, max_eigs
    else:
        # Use the original discriminant method
        betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
        
    high_y_curve = compute_high_y_curve(betas, z_a, y) if show_high_y else None
    alt_low_expr = compute_alternate_low_expr(betas, z_a, y) if show_low_y else None
    
    # Compute the max/min expressions
    max_k_curve = compute_max_k_expression(betas, z_a, y) if show_max_k else None
    min_t_curve = compute_min_t_expression(betas, z_a, y) if show_min_t else None
    
    # Compute both custom curves
    custom_curve1 = None
    custom_curve2 = None
    if s_num_expr and s_denom_expr:
        custom_curve1 = compute_custom_expression(betas, z_a, y, s_num_expr, s_denom_expr, True)
    if z_num_expr and z_denom_expr:
        custom_curve2 = compute_custom_expression(betas, z_a, y, z_num_expr, z_denom_expr, False)

    # Compute derivatives if needed
    if show_derivatives:
        derivatives = compute_all_derivatives(betas, z_mins, z_maxs, None, high_y_curve, 
                                           alt_low_expr, custom_curve1, custom_curve2)
        # Calculate derivatives for max_k and min_t curves if they exist
        if show_max_k:
            max_k_derivatives = compute_derivatives(max_k_curve, betas)
        if show_min_t:
            min_t_derivatives = compute_derivatives(min_t_curve, betas)

    fig = go.Figure()
    
    # Original curves
    if use_eigenvalue_method:
        fig.add_trace(go.Scatter(x=betas, y=z_maxs, mode="markers+lines", 
                                name="Upper Bound (Max Eigenvalue)", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=betas, y=z_mins, mode="markers+lines", 
                                name="Lower Bound (Min Eigenvalue)", line=dict(color='blue')))
        # Add shaded region between curves
        fig.add_trace(go.Scatter(
            x=np.concatenate([betas, betas[::-1]]),
            y=np.concatenate([z_maxs, z_mins[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        fig.add_trace(go.Scatter(x=betas, y=z_maxs, mode="markers+lines", 
                                name="Upper z*(β)", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=betas, y=z_mins, mode="markers+lines", 
                                name="Lower z*(β)", line=dict(color='blue')))

    # Add High y Expression only if selected
    if show_high_y and high_y_curve is not None:
        fig.add_trace(go.Scatter(x=betas, y=high_y_curve, mode="markers+lines", 
                                name="High y Expression", line=dict(color='green')))
    
    # Add Low Expression only if selected
    if show_low_y and alt_low_expr is not None:
        fig.add_trace(go.Scatter(x=betas, y=alt_low_expr, mode="markers+lines", 
                                name="Low Expression", line=dict(color='orange')))
    
    # Add the max/min curves if selected
    if show_max_k and max_k_curve is not None:
        fig.add_trace(go.Scatter(x=betas, y=max_k_curve, mode="lines", 
                                name="Max k Expression", line=dict(color='red', width=2)))
    
    if show_min_t and min_t_curve is not None:
        fig.add_trace(go.Scatter(x=betas, y=min_t_curve, mode="lines", 
                                name="Min t Expression", line=dict(color='purple', width=2)))
    
    if custom_curve1 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve1, mode="markers+lines", 
                                name="Custom 1 (s-based)", line=dict(color='magenta')))
    if custom_curve2 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve2, mode="markers+lines", 
                                name="Custom 2 (direct)", line=dict(color='brown')))

    if show_derivatives:
        # First derivatives
        curve_info = [
            ('upper', 'Upper Bound' if use_eigenvalue_method else 'Upper z*(β)', 'blue'),
            ('lower', 'Lower Bound' if use_eigenvalue_method else 'Lower z*(β)', 'lightblue'),
        ]
        
        if show_high_y and high_y_curve is not None:
            curve_info.append(('high_y', 'High y', 'green'))
        if show_low_y and alt_low_expr is not None:
            curve_info.append(('alt_low', 'Alt Low', 'orange'))
        
        if custom_curve1 is not None:
            curve_info.append(('custom1', 'Custom 1', 'magenta'))
        if custom_curve2 is not None:
            curve_info.append(('custom2', 'Custom 2', 'brown'))

        for key, name, color in curve_info:
            if key in derivatives:
                fig.add_trace(go.Scatter(x=betas, y=derivatives[key][0], mode="lines", 
                                        name=f"{name} d/dβ", line=dict(color=color, dash='dash')))
                fig.add_trace(go.Scatter(x=betas, y=derivatives[key][1], mode="lines", 
                                        name=f"{name} d²/dβ²", line=dict(color=color, dash='dot')))
        
        # Add derivatives for max_k and min_t curves if they exist
        if show_max_k and max_k_curve is not None:
            fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[0], mode="lines", 
                                    name="Max k d/dβ", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[1], mode="lines", 
                                    name="Max k d²/dβ²", line=dict(color='red', dash='dot')))
        
        if show_min_t and min_t_curve is not None:
            fig.add_trace(go.Scatter(x=betas, y=min_t_derivatives[0], mode="lines", 
                                    name="Min t d/dβ", line=dict(color='purple', dash='dash')))
            fig.add_trace(go.Scatter(x=betas, y=min_t_derivatives[1], mode="lines", 
                                    name="Min t d²/dβ²", line=dict(color='purple', dash='dot')))

    fig.update_layout(
        title="Curves vs β: Eigenvalue Support Boundaries and Asymptotic Expressions" if use_eigenvalue_method 
              else "Curves vs β: z*(β) Boundaries and Asymptotic Expressions",
        xaxis_title="β",
        yaxis_title="Value",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

def track_roots_consistently(z_values, all_roots):
    """
    Ensure consistent tracking of roots across z values by minimizing discontinuity.
    """
    n_points = len(z_values)
    n_roots = len(all_roots[0])
    tracked_roots = np.zeros((n_points, n_roots), dtype=complex)
    tracked_roots[0] = all_roots[0]
    
    for i in range(1, n_points):
        prev_roots = tracked_roots[i-1]
        current_roots = all_roots[i]
        
        # For each previous root, find the closest current root
        assigned = np.zeros(n_roots, dtype=bool)
        assignments = np.zeros(n_roots, dtype=int)
        
        for j in range(n_roots):
            distances = np.abs(current_roots - prev_roots[j])
            
            # Find the closest unassigned root
            while True:
                best_idx = np.argmin(distances)
                if not assigned[best_idx]:
                    assignments[j] = best_idx
                    assigned[best_idx] = True
                    break
                else:
                    # Mark as infinite distance and try again
                    distances[best_idx] = np.inf
                    
                # Safety check if all are assigned (shouldn't happen)
                if np.all(distances == np.inf):
                    assignments[j] = j  # Default to same index
                    break
        
        # Reorder current roots based on assignments
        tracked_roots[i] = current_roots[assignments]
    
    return tracked_roots

def generate_cubic_discriminant(z, beta, z_a, y_effective):
    """
    Calculate the cubic discriminant using the standard formula.
    For a cubic ax^3 + bx^2 + cx + d:
    Δ = 18abcd - 27a^2d^2 + b^2c^2 - 2b^3d - 9ac^3
    """
    a = z * z_a
    b = z * z_a + z + z_a - z_a*y_effective
    c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta)
    d = 1
    
    # Standard formula for cubic discriminant
    discriminant = (18*a*b*c*d - 27*a**2*d**2 + b**2*c**2 - 2*b**3*d - 9*a*c**3)
    return discriminant

def generate_root_plots(beta, y, z_a, z_min, z_max, n_points):
    """
    Generate Im(s) and Re(s) vs. z plots with improved accuracy using SymPy.
    """
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None, None, None

    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    z_points = np.linspace(z_min, z_max, n_points)
    
    # Collect all roots first
    all_roots = []
    discriminants = []
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, z in enumerate(z_points):
        # Update progress
        progress_bar.progress((i + 1) / n_points)
        status_text.text(f"Computing roots for z = {z:.3f} ({i+1}/{n_points})")
        
        # Calculate roots
        roots = compute_cubic_roots(z, beta, z_a, y)
        
        # Initial sorting to help with tracking
        roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
        all_roots.append(roots)
        
        # Calculate discriminant
        disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
        discriminants.append(disc)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    all_roots = np.array(all_roots)
    discriminants = np.array(discriminants)
    
    # Track roots consistently across z values
    tracked_roots = track_roots_consistently(z_points, all_roots)
    
    # Extract imaginary and real parts
    ims = np.imag(tracked_roots)
    res = np.real(tracked_roots)
    
    # Create figure for imaginary parts
    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=z_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    disc_zeros = []
    for i in range(len(discriminants)-1):
        if discriminants[i] * discriminants[i+1] <= 0:  # Sign change
            zero_pos = z_points[i] + (z_points[i+1] - z_points[i]) * (0 - discriminants[i]) / (discriminants[i+1] - discriminants[i])
            disc_zeros.append(zero_pos)
            fig_im.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_im.update_layout(title=f"Im{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Im{s}", hovermode="x unified")

    # Create figure for real parts
    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=z_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    for zero_pos in disc_zeros:
        fig_re.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_re.update_layout(title=f"Re{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Re{s}", hovermode="x unified")
    
    # Create discriminant plot
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=z_points, y=discriminants, mode="lines", 
                                 name="Cubic Discriminant", line=dict(color="black", width=2)))
    fig_disc.add_hline(y=0, line=dict(color="red", width=1, dash="dash"))
    
    fig_disc.update_layout(title=f"Cubic Discriminant vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                          xaxis_title="z", yaxis_title="Discriminant", hovermode="x unified")
    
    return fig_im, fig_re, fig_disc

def generate_roots_vs_beta_plots(z, y, z_a, beta_min, beta_max, n_points):
    """
    Generate Im(s) and Re(s) vs. β plots with improved accuracy.
    """
    if z_a <= 0 or y <= 0 or beta_min >= beta_max:
        st.error("Invalid input parameters.")
        return None, None, None

    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    beta_points = np.linspace(beta_min, beta_max, n_points)
    
    # Collect all roots first
    all_roots = []
    discriminants = []
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, beta in enumerate(beta_points):
        # Update progress
        progress_bar.progress((i + 1) / n_points)
        status_text.text(f"Computing roots for β = {beta:.3f} ({i+1}/{n_points})")
        
        # Calculate roots
        roots = compute_cubic_roots(z, beta, z_a, y)
        
        # Initial sorting to help with tracking
        roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
        all_roots.append(roots)
        
        # Calculate discriminant
        disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
        discriminants.append(disc)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    all_roots = np.array(all_roots)
    discriminants = np.array(discriminants)
    
    # Track roots consistently across beta values
    tracked_roots = track_roots_consistently(beta_points, all_roots)
    
    # Extract imaginary and real parts
    ims = np.imag(tracked_roots)
    res = np.real(tracked_roots)
    
    # Create figure for imaginary parts
    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=beta_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    disc_zeros = []
    for i in range(len(discriminants)-1):
        if discriminants[i] * discriminants[i+1] <= 0:  # Sign change
            zero_pos = beta_points[i] + (beta_points[i+1] - beta_points[i]) * (0 - discriminants[i]) / (discriminants[i+1] - discriminants[i])
            disc_zeros.append(zero_pos)
            fig_im.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_im.update_layout(title=f"Im{{s}} vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="β", yaxis_title="Im{s}", hovermode="x unified")

    # Create figure for real parts
    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=beta_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    for zero_pos in disc_zeros:
        fig_re.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_re.update_layout(title=f"Re{{s}} vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="β", yaxis_title="Re{s}", hovermode="x unified")
    
    # Create discriminant plot
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=beta_points, y=discriminants, mode="lines", 
                                 name="Cubic Discriminant", line=dict(color="black", width=2)))
    fig_disc.add_hline(y=0, line=dict(color="red", width=1, dash="dash"))
    
    fig_disc.update_layout(title=f"Cubic Discriminant vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                          xaxis_title="β", yaxis_title="Discriminant", hovermode="x unified")
    
    return fig_im, fig_re, fig_disc

def analyze_complex_root_structure(beta_values, z, z_a, y):
    """
    Analyze when the cubic equation switches between having all real roots
    and having a complex conjugate pair plus one real root.
    
    Returns:
    - transition_points: beta values where the root structure changes
    - structure_types: list indicating whether each interval has all real roots or complex roots
    """
    transition_points = []
    structure_types = []
    
    previous_type = None
    
    for beta in beta_values:
        roots = compute_cubic_roots(z, beta, z_a, y)
        
        # Check if all roots are real (imaginary parts close to zero)
        is_all_real = all(abs(root.imag) < 1e-10 for root in roots)
        
        current_type = "real" if is_all_real else "complex"
        
        if previous_type is not None and current_type != previous_type:
            # Found a transition point
            transition_points.append(beta)
            structure_types.append(previous_type)
        
        previous_type = current_type
    
    # Add the final interval type
    if previous_type is not None:
        structure_types.append(previous_type)
    
    return transition_points, structure_types

def generate_phase_diagram(z_a, y, beta_min=0.0, beta_max=1.0, z_min=-10.0, z_max=10.0, 
                          beta_steps=100, z_steps=100):
    """
    Generate a phase diagram showing regions of complex and real roots.
    
    Returns a heatmap where:
    - Value 1 (red): Region with all real roots
    - Value -1 (blue): Region with complex roots
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    beta_values = np.linspace(beta_min, beta_max, beta_steps)
    z_values = np.linspace(z_min, z_max, z_steps)
    
    # Initialize phase map
    phase_map = np.zeros((z_steps, beta_steps))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, z in enumerate(z_values):
        # Update progress
        progress_bar.progress((i + 1) / len(z_values))
        status_text.text(f"Analyzing phase at z = {z:.2f} ({i+1}/{len(z_values)})")
        
        for j, beta in enumerate(beta_values):
            roots = compute_cubic_roots(z, beta, z_a, y)
            
            # Check if all roots are real (imaginary parts close to zero)
            is_all_real = all(abs(root.imag) < 1e-10 for root in roots)
            
            phase_map[i, j] = 1 if is_all_real else -1
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=phase_map,
        x=beta_values,
        y=z_values,
        colorscale=[[0, 'blue'], [0.5, 'white'], [1.0, 'red']],
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(
            title="Root Type",
            tickvals=[-1, 1],
            ticktext=["Complex Roots", "All Real Roots"]
        )
    ))
    
    fig.update_layout(
        title=f"Phase Diagram: Root Structure (y={y:.3f}, z_a={z_a:.3f})",
        xaxis_title="β",
        yaxis_title="z",
        hovermode="closest"
    )
    
    return fig

@st.cache_data
def generate_eigenvalue_distribution_plot(beta, y, z_a, n=1000, seed=42):
    """
    Generate the eigenvalue distribution of B_n = S_n T_n as n→∞
    """
    # Generate eigenvalues
    eigenvalues = generate_eigenvalue_distribution(beta, y, z_a, n, seed)
    
    # Use KDE to compute a smooth density estimate
    kde = gaussian_kde(eigenvalues)
    x_vals = np.linspace(min(eigenvalues), max(eigenvalues), 500)
    kde_vals = kde(x_vals)
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(x=eigenvalues, histnorm='probability density', 
                              name="Histogram", marker=dict(color='blue', opacity=0.6)))
    
    # Add KDE trace
    fig.add_trace(go.Scatter(x=x_vals, y=kde_vals, mode="lines", 
                            name="KDE", line=dict(color='red', width=2)))
    
    fig.update_layout(
        title=f"Eigenvalue Distribution for B_n = S_n T_n (y={y:.1f}, β={beta:.2f}, a={z_a:.1f})",
        xaxis_title="Eigenvalue",
        yaxis_title="Density",
        hovermode="closest",
        showlegend=True
    )
    
    return fig, eigenvalues

# ----------------- Streamlit UI -----------------
def main():
    st.title("Cubic Root Analysis")
    
    # Define three tabs
    tab1, tab2, tab3 = st.tabs(["z*(β) Curves", "Complex Root Analysis", "Differential Analysis"])
    
    # ----- Tab 1: z*(β) Curves -----
    with tab1:
        st.header("Eigenvalue Support Boundaries")
        
        # Cleaner layout with better column organization
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            z_a_1 = st.number_input("z_a", value=1.0, key="z_a_1")
            y_1 = st.number_input("y", value=1.0, key="y_1")
            
        with col2:
            z_min_1 = st.number_input("z_min", value=-10.0, key="z_min_1")
            z_max_1 = st.number_input("z_max", value=10.0, key="z_max_1")
        
        with col1:
            method_type = st.radio(
                "Calculation Method",
                ["Eigenvalue Method", "Discriminant Method"],
                index=0  # Default to eigenvalue method
            )
        
        # Advanced settings in collapsed expanders
        with st.expander("Method Settings", expanded=False):
            if method_type == "Eigenvalue Method":
                beta_steps = st.slider("β steps", min_value=21, max_value=101, value=51, step=10, 
                                      key="beta_steps_eigen")
                n_samples = st.slider("Matrix size (n)", min_value=100, max_value=2000, value=1000, 
                                    step=100)
                seeds = st.slider("Number of seeds", min_value=1, max_value=10, value=5, step=1)
            else:
                beta_steps = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, 
                                      key="beta_steps")
                z_steps = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, 
                                  step=1000, key="z_steps")
        
        # Curve visibility options
        with st.expander("Curve Visibility", expanded=False):
            col_vis1, col_vis2 = st.columns(2)
            with col_vis1:
                show_high_y = st.checkbox("Show High y Expression", value=False, key="show_high_y")
                show_max_k = st.checkbox("Show Max k Expression", value=True, key="show_max_k")
            with col_vis2:
                show_low_y = st.checkbox("Show Low y Expression", value=False, key="show_low_y")
                show_min_t = st.checkbox("Show Min t Expression", value=True, key="show_min_t")
        
        # Custom expressions collapsed by default
        with st.expander("Custom Expression 1 (s-based)", expanded=False):
            st.markdown("""Enter expressions for s = numerator/denominator 
                        (using variables `y`, `beta`, `z_a`, and `sqrt()`)""")
            st.latex(r"\text{This s will be inserted into:}")
            st.latex(r"\frac{y\beta(z_a-1)\underline{s}+(a\underline{s}+1)((y-1)\underline{s}-1)}{(a\underline{s}+1)(\underline{s}^2 + \underline{s})}")
            s_num = st.text_input("s numerator", value="", key="s_num")
            s_denom = st.text_input("s denominator", value="", key="s_denom")

        with st.expander("Custom Expression 2 (direct z(β))", expanded=False):
            st.markdown("""Enter direct expression for z(β) = numerator/denominator 
                        (using variables `y`, `beta`, `z_a`, and `sqrt()`)""")
            z_num = st.text_input("z(β) numerator", value="", key="z_num")
            z_denom = st.text_input("z(β) denominator", value="", key="z_denom")

        # Move show_derivatives to main UI level for better visibility
        with col2:
            show_derivatives = st.checkbox("Show derivatives", value=False)

        # Compute button
        if st.button("Compute Curves", key="tab1_button"):
            with col3:
                use_eigenvalue_method = (method_type == "Eigenvalue Method")
                if use_eigenvalue_method:
                    fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, None,
                                                s_num, s_denom, z_num, z_denom, show_derivatives, 
                                                show_high_y, show_low_y, show_max_k, show_min_t,
                                                use_eigenvalue_method=True, n_samples=n_samples, 
                                                seeds=seeds)
                else:
                    fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, z_steps,
                                                s_num, s_denom, z_num, z_denom, show_derivatives,
                                                show_high_y, show_low_y, show_max_k, show_min_t,
                                                use_eigenvalue_method=False)
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Curve explanations in collapsed expander
                    with st.expander("Curve Explanations", expanded=False):
                        if use_eigenvalue_method:
                            st.markdown("""
                            - **Upper/Lower Bounds** (Blue): Maximum/minimum eigenvalues of B_n = S_n T_n
                            - **Shaded Region**: Eigenvalue support region
                            - **High y Expression** (Green): Asymptotic approximation for high y values
                            - **Low Expression** (Orange): Alternative asymptotic expression
                            - **Max k Expression** (Red): $\\max_{k \\in (0,\\infty)} \\frac{y\\beta (a-1)k + \\bigl(ak+1\\bigr)\\bigl((y-1)k-1\\bigr)}{(ak+1)(k^2+k)}$
                            - **Min t Expression** (Purple): $\\min_{t \\in \\left(-\\frac{1}{a},\\, 0\\right)} \\frac{y\\beta (a-1)t + \\bigl(at+1\\bigr)\\bigl((y-1)t-1\\bigr)}{(at+1)(t^2+t)}$
                            - **Custom Expression 1** (Magenta): Result from user-defined s substituted into the main formula
                            - **Custom Expression 2** (Brown): Direct z(β) expression
                            """)
                        else:
                            st.markdown("""
                            - **Upper z*(β)** (Blue): Maximum z value where discriminant is zero
                            - **Lower z*(β)** (Blue): Minimum z value where discriminant is zero
                            - **High y Expression** (Green): Asymptotic approximation for high y values
                            - **Low Expression** (Orange): Alternative asymptotic expression
                            - **Max k Expression** (Red): $\\max_{k \\in (0,\\infty)} \\frac{y\\beta (a-1)k + \\bigl(ak+1\\bigr)\\bigl((y-1)k-1\\bigr)}{(ak+1)(k^2+k)}$
                            - **Min t Expression** (Purple): $\\min_{t \\in \\left(-\\frac{1}{a},\\, 0\\right)} \\frac{y\\beta (a-1)t + \\bigl(at+1\\bigr)\\bigl((y-1)t-1\\bigr)}{(at+1)(t^2+t)}$
                            - **Custom Expression 1** (Magenta): Result from user-defined s substituted into the main formula
                            - **Custom Expression 2** (Brown): Direct z(β) expression
                            """)
                        if show_derivatives:
                            st.markdown("""
                            Derivatives are shown as:
                            - Dashed lines: First derivatives (d/dβ)
                            - Dotted lines: Second derivatives (d²/dβ²)
                            """)

    # ----- Tab 2: Complex Root Analysis -----
    with tab2:
        st.header("Complex Root Analysis")
        
        # Create tabs within the page for different plots
        plot_tabs = st.tabs(["Im{s} vs. z", "Im{s} vs. β", "Phase Diagram", "Eigenvalue Distribution"])
        
        # Tab for Im{s} vs. z plot
        with plot_tabs[0]:
            col1, col2 = st.columns([1, 2])
            with col1:
                beta_z = st.number_input("β", value=0.5, min_value=0.0, max_value=1.0, key="beta_tab2_z")
                y_z = st.number_input("y", value=1.0, key="y_tab2_z")
                z_a_z = st.number_input("z_a", value=1.0, key="z_a_tab2_z")
                z_min_z = st.number_input("z_min", value=-10.0, key="z_min_tab2_z")
                z_max_z = st.number_input("z_max", value=10.0, key="z_max_tab2_z")
                with st.expander("Resolution Settings", expanded=False):
                    z_points = st.slider("z grid points", min_value=100, max_value=2000, value=500, step=100, key="z_points_z")
            if st.button("Compute Complex Roots vs. z", key="tab2_button_z"):
                with col2:
                    fig_im, fig_re, fig_disc = generate_root_plots(beta_z, y_z, z_a_z, z_min_z, z_max_z, z_points)
                    if fig_im is not None and fig_re is not None and fig_disc is not None:
                        st.plotly_chart(fig_im, use_container_width=True)
                        st.plotly_chart(fig_re, use_container_width=True)
                        st.plotly_chart(fig_disc, use_container_width=True)
                        
                        with st.expander("Root Structure Analysis", expanded=False):
                            st.markdown("""
                            ### Root Structure Explanation
                            
                            The red dashed vertical lines mark the points where the cubic discriminant equals zero.
                            At these points, the cubic equation's root structure changes:
                            
                            - When the discriminant is positive, the cubic has three distinct real roots.
                            - When the discriminant is negative, the cubic has one real root and two complex conjugate roots.
                            - When the discriminant is exactly zero, the cubic has at least two equal roots.
                            
                            These transition points align perfectly with the z*(β) boundary curves from the first tab,
                            which represent exactly these transitions in the (β,z) plane.
                            """)

        # New tab for Im{s} vs. β plot
        with plot_tabs[1]:
            col1, col2 = st.columns([1, 2])
            with col1:
                z_beta = st.number_input("z", value=1.0, key="z_tab2_beta")
                y_beta = st.number_input("y", value=1.0, key="y_tab2_beta")
                z_a_beta = st.number_input("z_a", value=1.0, key="z_a_tab2_beta")
                beta_min = st.number_input("β_min", value=0.0, min_value=0.0, max_value=1.0, key="beta_min_tab2")
                beta_max = st.number_input("β_max", value=1.0, min_value=0.0, max_value=1.0, key="beta_max_tab2")
                with st.expander("Resolution Settings", expanded=False):
                    beta_points = st.slider("β grid points", min_value=100, max_value=1000, value=500, step=100, key="beta_points")
            if st.button("Compute Complex Roots vs. β", key="tab2_button_beta"):
                with col2:
                    fig_im_beta, fig_re_beta, fig_disc = generate_roots_vs_beta_plots(
                        z_beta, y_beta, z_a_beta, beta_min, beta_max, beta_points)
                    
                    if fig_im_beta is not None and fig_re_beta is not None and fig_disc is not None:
                        st.plotly_chart(fig_im_beta, use_container_width=True)
                        st.plotly_chart(fig_re_beta, use_container_width=True)
                        st.plotly_chart(fig_disc, use_container_width=True)
                        
                        # Add analysis of transition points
                        transition_points, structure_types = analyze_complex_root_structure(
                            np.linspace(beta_min, beta_max, beta_points), z_beta, z_a_beta, y_beta)
                        
                        if transition_points:
                            st.subheader("Root Structure Transition Points")
                            for i, beta in enumerate(transition_points):
                                prev_type = structure_types[i]
                                next_type = structure_types[i+1] if i+1 < len(structure_types) else "unknown"
                                st.markdown(f"- At β = {beta:.6f}: Transition from {prev_type} roots to {next_type} roots")
                        else:
                            st.info("No transitions detected in root structure across this β range.")
                        
                        # Explanation
                        with st.expander("Analysis Explanation", expanded=False):
                            st.markdown("""
                            ### Interpreting the Plots
                            
                            - **Im{s} vs. β**: Shows how the imaginary parts of the roots change with β. When all curves are at Im{s}=0, all roots are real.
                            - **Re{s} vs. β**: Shows how the real parts of the roots change with β.
                            - **Discriminant Plot**: The cubic discriminant changes sign at points where the root structure changes.
                              - When discriminant < 0: The cubic has one real root and two complex conjugate roots.
                              - When discriminant > 0: The cubic has three distinct real roots.
                              - When discriminant = 0: The cubic has multiple roots (at least two roots are equal).
                            
                            The vertical red dashed lines mark the transition points where the root structure changes.
                            """)
        
        # Tab for Phase Diagram
        with plot_tabs[2]:
            col1, col2 = st.columns([1, 2])
            with col1:
                z_a_phase = st.number_input("z_a", value=1.0, key="z_a_phase")
                y_phase = st.number_input("y", value=1.0, key="y_phase")
                beta_min_phase = st.number_input("β_min", value=0.0, min_value=0.0, max_value=1.0, key="beta_min_phase")
                beta_max_phase = st.number_input("β_max", value=1.0, min_value=0.0, max_value=1.0, key="beta_max_phase")
                z_min_phase = st.number_input("z_min", value=-10.0, key="z_min_phase")
                z_max_phase = st.number_input("z_max", value=10.0, key="z_max_phase")
                
                with st.expander("Resolution Settings", expanded=False):
                    beta_steps_phase = st.slider("β grid points", min_value=20, max_value=200, value=100, step=20, key="beta_steps_phase")
                    z_steps_phase = st.slider("z grid points", min_value=20, max_value=200, value=100, step=20, key="z_steps_phase")
            
            if st.button("Generate Phase Diagram", key="tab2_button_phase"):
                with col2:
                    st.info("Generating phase diagram. This may take a while depending on resolution...")
                    fig_phase = generate_phase_diagram(
                        z_a_phase, y_phase, beta_min_phase, beta_max_phase, z_min_phase, z_max_phase, 
                        beta_steps_phase, z_steps_phase)
                    
                    if fig_phase is not None:
                        st.plotly_chart(fig_phase, use_container_width=True)
                        
                        with st.expander("Phase Diagram Explanation", expanded=False):
                            st.markdown("""
                            ### Understanding the Phase Diagram
                            
                            This heatmap shows the regions in the (β, z) plane where:
                            
                            - **Red Regions**: The cubic equation has all real roots
                            - **Blue Regions**: The cubic equation has one real root and two complex conjugate roots
                            
                            The boundaries between these regions represent values where the discriminant is zero,
                            which are the exact same curves as the z*(β) boundaries in the first tab. This phase
                            diagram provides a comprehensive view of the eigenvalue support structure.
                            """)

        # Eigenvalue distribution tab
        with plot_tabs[3]:
            st.subheader("Eigenvalue Distribution for B_n = S_n T_n")
            with st.expander("Simulation Information", expanded=False):
                st.markdown("""
                This simulation generates the eigenvalue distribution of B_n as n→∞, where:
                - B_n = (1/n)XX^T with X being a p×n matrix
                - p/n → y as n→∞
                - The diagonal entries of T_n follow distribution β·δ(z_a) + (1-β)·δ(1)
                """)
            
            col_eigen1, col_eigen2 = st.columns([1, 2])
            with col_eigen1:
                beta_eigen = st.number_input("β", value=0.5, min_value=0.0, max_value=1.0, key="beta_eigen")
                y_eigen = st.number_input("y", value=1.0, key="y_eigen")
                z_a_eigen = st.number_input("z_a", value=1.0, key="z_a_eigen")
                n_samples = st.slider("Number of samples (n)", min_value=100, max_value=2000, value=1000, step=100)
                sim_seed = st.number_input("Random seed", min_value=1, max_value=1000, value=42, step=1)
                
                # Add comparison option
                show_theoretical = st.checkbox("Show theoretical boundaries", value=True)
                show_empirical_stats = st.checkbox("Show empirical statistics", value=True)

            if st.button("Generate Eigenvalue Distribution", key="tab2_eigen_button"):
                with col_eigen2:
                    # Generate the eigenvalue distribution
                    fig_eigen, eigenvalues = generate_eigenvalue_distribution_plot(beta_eigen, y_eigen, z_a_eigen, n=n_samples, seed=sim_seed)
                    
                    # If requested, compute and add theoretical boundaries
                    if show_theoretical:
                        # Calculate min and max eigenvalues using the support boundary functions
                        betas = np.array([beta_eigen])
                        min_eig, max_eig = compute_eigenvalue_support_boundaries(z_a_eigen, y_eigen, betas, n_samples=n_samples, seeds=5)
                        
                        # Add vertical lines for boundaries
                        fig_eigen.add_vline(
                            x=min_eig[0], 
                            line=dict(color="red", width=2, dash="dash"),
                            annotation_text="Min theoretical",
                            annotation_position="top right"
                        )
                        fig_eigen.add_vline(
                            x=max_eig[0], 
                            line=dict(color="red", width=2, dash="dash"),
                            annotation_text="Max theoretical",
                            annotation_position="top left"
                        )
                    
                    # Display the plot
                    st.plotly_chart(fig_eigen, use_container_width=True)
                    
                    # Add comparison of empirical vs theoretical bounds
                    if show_theoretical and show_empirical_stats:
                        empirical_min = eigenvalues.min()
                        empirical_max = eigenvalues.max()
                        
                        st.markdown("### Comparison of Empirical vs Theoretical Bounds")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Theoretical Min", f"{min_eig[0]:.4f}")
                            st.metric("Theoretical Max", f"{max_eig[0]:.4f}")
                            st.metric("Theoretical Width", f"{max_eig[0] - min_eig[0]:.4f}")
                        with col2:
                            st.metric("Empirical Min", f"{empirical_min:.4f}")
                            st.metric("Empirical Max", f"{empirical_max:.4f}")
                            st.metric("Empirical Width", f"{empirical_max - empirical_min:.4f}")
                        with col3:
                            st.metric("Min Difference", f"{empirical_min - min_eig[0]:.4f}")
                            st.metric("Max Difference", f"{empirical_max - max_eig[0]:.4f}")
                            st.metric("Width Difference", f"{(empirical_max - empirical_min) - (max_eig[0] - min_eig[0]):.4f}")
                    
                    # Display additional statistics
                    if show_empirical_stats:
                        st.markdown("### Eigenvalue Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean", f"{np.mean(eigenvalues):.4f}")
                            st.metric("Median", f"{np.median(eigenvalues):.4f}")
                        with col2:
                            st.metric("Standard Deviation", f"{np.std(eigenvalues):.4f}")
                            st.metric("Interquartile Range", f"{np.percentile(eigenvalues, 75) - np.percentile(eigenvalues, 25):.4f}")

    # ----- Tab 3: Differential Analysis -----
    with tab3:
        st.header("Differential Analysis vs. β")
        with st.expander("Description", expanded=False):
            st.markdown("This page shows the difference between the Upper (blue) and Lower (lightblue) z*(β) curves, along with their first and second derivatives with respect to β.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            z_a_diff = st.number_input("z_a", value=1.0, key="z_a_diff")
            y_diff = st.number_input("y", value=1.0, key="y_diff")
            z_min_diff = st.number_input("z_min", value=-10.0, key="z_min_diff")
            z_max_diff = st.number_input("z_max", value=10.0, key="z_max_diff")
            
            diff_method_type = st.radio(
                "Boundary Calculation Method",
                ["Eigenvalue Method", "Discriminant Method"],
                index=0,
                key="diff_method_type"
            )
            
            with st.expander("Resolution Settings", expanded=False):
                if diff_method_type == "Eigenvalue Method":
                    beta_steps_diff = st.slider("β steps", min_value=21, max_value=101, value=51, step=10, 
                                         key="beta_steps_diff_eigen")
                    diff_n_samples = st.slider("Matrix size (n)", min_value=100, max_value=2000, value=1000, 
                                        step=100, key="diff_n_samples")
                    diff_seeds = st.slider("Number of seeds", min_value=1, max_value=10, value=5, step=1,
                                     key="diff_seeds")
                else:
                    beta_steps_diff = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, 
                                         key="beta_steps_diff")
                    z_steps_diff = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, 
                                      step=1000, key="z_steps_diff")
            
            # Add options for curve selection
            st.subheader("Curves to Analyze")
            analyze_upper_lower = st.checkbox("Upper-Lower Difference", value=True)
            analyze_high_y = st.checkbox("High y Expression", value=False)
            analyze_alt_low = st.checkbox("Low y Expression", value=False)

        if st.button("Compute Differentials", key="tab3_button"):
            with col2:
                use_eigenvalue_method_diff = (diff_method_type == "Eigenvalue Method")
                
                if use_eigenvalue_method_diff:
                    betas_diff = np.linspace(0, 1, beta_steps_diff)
                    st.info("Computing eigenvalue support boundaries. This may take a moment...")
                    lower_vals, upper_vals = compute_eigenvalue_support_boundaries(
                        z_a_diff, y_diff, betas_diff, diff_n_samples, diff_seeds)
                else:
                    betas_diff, lower_vals, upper_vals = sweep_beta_and_find_z_bounds(
                        z_a_diff, y_diff, z_min_diff, z_max_diff, beta_steps_diff, z_steps_diff)
                
                # Create figure
                fig_diff = go.Figure()
                
                if analyze_upper_lower:
                    diff_curve = upper_vals - lower_vals
                    d1, d2 = compute_derivatives(diff_curve, betas_diff)
                    
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=diff_curve, mode="lines", 
                                    name="Upper-Lower Difference", line=dict(color="magenta", width=2)))
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                    name="Upper-Lower d/dβ", line=dict(color="magenta", dash='dash')))
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                    name="Upper-Lower d²/dβ²", line=dict(color="magenta", dash='dot')))

                if analyze_high_y:
                    high_y_curve = compute_high_y_curve(betas_diff, z_a_diff, y_diff)
                    d1, d2 = compute_derivatives(high_y_curve, betas_diff)
                    
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=high_y_curve, mode="lines", 
                                    name="High y", line=dict(color="green", width=2)))
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                    name="High y d/dβ", line=dict(color="green", dash='dash')))
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                    name="High y d²/dβ²", line=dict(color="green", dash='dot')))

                if analyze_alt_low:
                    alt_low_curve = compute_alternate_low_expr(betas_diff, z_a_diff, y_diff)
                    d1, d2 = compute_derivatives(alt_low_curve, betas_diff)
                    
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=alt_low_curve, mode="lines", 
                                    name="Low y", line=dict(color="orange", width=2)))
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                    name="Low y d/dβ", line=dict(color="orange", dash='dash')))
                    fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                    name="Low y d²/dβ²", line=dict(color="orange", dash='dot')))

                fig_diff.update_layout(
                    title="Differential Analysis vs. β" + 
                          (" (Eigenvalue Method)" if use_eigenvalue_method_diff else " (Discriminant Method)"),
                    xaxis_title="β",
                    yaxis_title="Value",
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                st.plotly_chart(fig_diff, use_container_width=True)
                
                with st.expander("Curve Types", expanded=False):
                    st.markdown("""
                    - Solid lines: Original curves
                    - Dashed lines: First derivatives (d/dβ)
                    - Dotted lines: Second derivatives (d²/dβ²)
                    """)

if __name__ == "__main__":
    main()