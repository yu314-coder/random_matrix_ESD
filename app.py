import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde

# Configure Streamlit for Hugging Face Spaces
st.set_page_config(
    page_title="Cubic Root Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# Fast numeric function for the discriminant
discriminant_func = sp.lambdify((z_sym, beta_sym, z_a_sym, y_sym), Delta_expr, "numpy")

@st.cache_data
def find_z_at_discriminant_zero(z_a, y, beta, z_min, z_max, steps):
    """
    Scan z in [z_min, z_max] for sign changes in the discriminant,
    and return approximated roots (where the discriminant is zero).
    """
    z_grid = np.linspace(z_min, z_max, steps)
    disc_vals = discriminant_func(z_grid, beta, z_a, y)
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
                fm = discriminant_func(mid, beta, z_a, y)
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
def sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps):
    """
    For each beta in [0,1] (with beta_steps points), find the minimum and maximum z 
    for which the discriminant is zero.
    Returns: betas, lower z*(β) values, and upper z*(β) values.
    """
    betas = np.linspace(0, 1, beta_steps)
    z_min_values = []
    z_max_values = []
    for b in betas:
        roots = find_z_at_discriminant_zero(z_a, y, b, z_min, z_max, z_steps)
        if len(roots) == 0:
            z_min_values.append(np.nan)
            z_max_values.append(np.nan)
        else:
            z_min_values.append(np.min(roots))
            z_max_values.append(np.max(roots))
    return betas, np.array(z_min_values), np.array(z_max_values)

@st.cache_data
def compute_eigenvalue_support_boundaries(z_a, y, beta_values, n_samples=100, seeds=5):
    """
    Compute the support boundaries of the eigenvalue distribution by directly
    finding the minimum and maximum eigenvalues of B_n = S_n T_n for different beta values.
    """
    y_effective =y
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
            B_n = S_n @ T_n /y_effective
            
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
def compute_high_y_curve(betas, z_a, y):
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

def compute_alternate_low_expr(betas, z_a, y):
    """
    Compute the alternate low expression:
    (z_a*y*beta*(z_a-1) - 2*z_a*(1-y) - 2*z_a**2) / (2+2*z_a)
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    betas = np.array(betas)
    return (z_a * y_effective * betas * (z_a - 1) - 2*z_a*(1 - y_effective) - 2*z_a**2) / (2 + 2*z_a)

@st.cache_data
def compute_max_k_expression(betas, z_a, y, k_samples=1000):
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
def compute_min_t_expression(betas, z_a, y, t_samples=1000):
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
def compute_derivatives(curve, betas):
    """Compute first and second derivatives of a curve"""
    d1 = np.gradient(curve, betas)
    d2 = np.gradient(d1, betas)
    return d1, d2

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
    derivatives['high_y'] = compute_derivatives(high_y_curve, betas)
    
    # Alternate Low Expression
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
        
    high_y_curve = compute_high_y_curve(betas, z_a, y)
    alt_low_expr = compute_alternate_low_expr(betas, z_a, y)
    
    # Compute the max/min expressions
    max_k_curve = compute_max_k_expression(betas, z_a, y)
    min_t_curve = compute_min_t_expression(betas, z_a, y)
    
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
        # Calculate derivatives for max_k and min_t curves
        max_k_derivatives = compute_derivatives(max_k_curve, betas)
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

    # Removed the Low y Expression trace
    fig.add_trace(go.Scatter(x=betas, y=high_y_curve, mode="markers+lines", 
                            name="High y Expression", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=betas, y=alt_low_expr, mode="markers+lines", 
                            name="Low Expression", line=dict(color='green')))
    
    # Add the new max/min curves
    fig.add_trace(go.Scatter(x=betas, y=max_k_curve, mode="lines", 
                            name="Max k Expression", line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=betas, y=min_t_curve, mode="lines", 
                            name="Min t Expression", line=dict(color='orange', width=2)))
    
    if custom_curve1 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve1, mode="markers+lines", 
                                name="Custom 1 (s-based)", line=dict(color='purple')))
    if custom_curve2 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve2, mode="markers+lines", 
                                name="Custom 2 (direct)", line=dict(color='magenta')))

    if show_derivatives:
        # First derivatives
        curve_info = [
            ('upper', 'Upper Bound' if use_eigenvalue_method else 'Upper z*(β)', 'blue'),
            ('lower', 'Lower Bound' if use_eigenvalue_method else 'Lower z*(β)', 'lightblue'),
            # Removed low_y curve
            ('high_y', 'High y', 'green'),
            ('alt_low', 'Alt Low', 'orange')
        ]
        
        if custom_curve1 is not None:
            curve_info.append(('custom1', 'Custom 1', 'purple'))
        if custom_curve2 is not None:
            curve_info.append(('custom2', 'Custom 2', 'magenta'))

        for key, name, color in curve_info:
            fig.add_trace(go.Scatter(x=betas, y=derivatives[key][0], mode="lines", 
                                    name=f"{name} d/dβ", line=dict(color=color, dash='dash')))
            fig.add_trace(go.Scatter(x=betas, y=derivatives[key][1], mode="lines", 
                                    name=f"{name} d²/dβ²", line=dict(color=color, dash='dot')))
        
        # Add derivatives for max_k and min_t curves
        fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[0], mode="lines", 
                                name="Max k d/dβ", line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[1], mode="lines", 
                                name="Max k d²/dβ²", line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=betas, y=min_t_derivatives[0], mode="lines", 
                                name="Min t d/dβ", line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=betas, y=min_t_derivatives[1], mode="lines", 
                                name="Min t d²/dβ²", line=dict(color='orange', dash='dot')))

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

def compute_cubic_roots(z, beta, z_a, y):
    """
    Compute the roots of the cubic equation for given parameters.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z * z_a
    b = z * z_a + z + z_a - z_a*y_effective
    c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta)
    d = 1
    coeffs = [a, b, c, d]
    roots = np.roots(coeffs)
    return roots

def generate_root_plots(beta, y, z_a, z_min, z_max, n_points):
    """
    Generate Im(s) and Re(s) vs. z plots.
    """
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None, None

    z_points = np.linspace(z_min, z_max, n_points)
    ims, res = [], []
    for z in z_points:
        roots = compute_cubic_roots(z, beta, z_a, y)
        roots = sorted(roots, key=lambda x: abs(x.imag))
        ims.append([root.imag for root in roots])
        res.append([root.real for root in roots])
    ims = np.array(ims)
    res = np.array(res)

    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=z_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}",
                                    line=dict(width=2)))
    fig_im.update_layout(title=f"Im{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Im{s}", hovermode="x unified")

    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=z_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}",
                                    line=dict(width=2)))
    fig_re.update_layout(title=f"Re{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Re{s}", hovermode="x unified")
    return fig_im, fig_re

@st.cache_data
def generate_eigenvalue_distribution(beta, y, z_a, n=1000, seed=42):
    """
    Generate the eigenvalue distribution of B_n = S_n T_n as n→∞
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Set random seed
    np.random.seed(seed)
    
    # Compute dimension p based on aspect ratio y
    p = int(y_effective * n)
    
    # Constructing T_n (Population / Shape Matrix) - using the approach from the second script
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
    
    return fig

# ----------------- Streamlit UI -----------------
st.title("Cubic Root Analysis")

# Define three tabs (removed "Curve Intersections")
tab1, tab2, tab3 = st.tabs(["z*(β) Curves", "Im{s} vs. z", "Differential Analysis"])

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
                                            use_eigenvalue_method=True, n_samples=n_samples, 
                                            seeds=seeds)
            else:
                fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, z_steps,
                                            s_num, s_denom, z_num, z_denom, show_derivatives,
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
                        - **Min t Expression** (Orange): $\\min_{t \\in \\left(-\\frac{1}{a},\\, 0\\right)} \\frac{y\\beta (a-1)t + \\bigl(at+1\\bigr)\\bigl((y-1)t-1\\bigr)}{(at+1)(t^2+t)}$
                        - **Custom Expression 1** (Purple): Result from user-defined s substituted into the main formula
                        - **Custom Expression 2** (Magenta): Direct z(β) expression
                        """)
                    else:
                        st.markdown("""
                        - **Upper z*(β)** (Blue): Maximum z value where discriminant is zero
                        - **Lower z*(β)** (Light Blue): Minimum z value where discriminant is zero
                        - **High y Expression** (Green): Asymptotic approximation for high y values
                        - **Low Expression** (Orange): Alternative asymptotic expression
                        - **Max k Expression** (Red): $\\max_{k \\in (0,\\infty)} \\frac{y\\beta (a-1)k + \\bigl(ak+1\\bigr)\\bigl((y-1)k-1\\bigr)}{(ak+1)(k^2+k)}$
                        - **Min t Expression** (Orange): $\\min_{t \\in \\left(-\\frac{1}{a},\\, 0\\right)} \\frac{y\\beta (a-1)t + \\bigl(at+1\\bigr)\\bigl((y-1)t-1\\bigr)}{(at+1)(t^2+t)}$
                        - **Custom Expression 1** (Purple): Result from user-defined s substituted into the main formula
                        - **Custom Expression 2** (Magenta): Direct z(β) expression
                        """)
                    if show_derivatives:
                        st.markdown("""
                        Derivatives are shown as:
                        - Dashed lines: First derivatives (d/dβ)
                        - Dotted lines: Second derivatives (d²/dβ²)
                        """)

# ----- Tab 2: Im{s} vs. z -----
with tab2:
    st.header("Plot Complex Roots vs. z")
    col1, col2 = st.columns([1, 2])
    with col1:
        beta = st.number_input("β", value=0.5, min_value=0.0, max_value=1.0, key="beta_tab2")
        y_2 = st.number_input("y", value=1.0, key="y_tab2")
        z_a_2 = st.number_input("z_a", value=1.0, key="z_a_tab2")
        z_min_2 = st.number_input("z_min", value=-10.0, key="z_min_tab2")
        z_max_2 = st.number_input("z_max", value=10.0, key="z_max_tab2")
        with st.expander("Resolution Settings", expanded=False):
            z_points = st.slider("z grid points", min_value=1000, max_value=10000, value=5000, step=500, key="z_points")
    if st.button("Compute Complex Roots vs. z", key="tab2_button"):
        with col2:
            fig_im, fig_re = generate_root_plots(beta, y_2, z_a_2, z_min_2, z_max_2, z_points)
            if fig_im is not None and fig_re is not None:
                st.plotly_chart(fig_im, use_container_width=True)
                st.plotly_chart(fig_re, use_container_width=True)
    
    # Add a separator
    st.markdown("---")
    
    # Add eigenvalue distribution section
    st.header("Eigenvalue Distribution for B_n = S_n T_n")
    with st.expander("Simulation Information", expanded=False):
        st.markdown("""
        This simulation generates the eigenvalue distribution of B_n as n→∞, where:
        - B_n = (1/n)XX* with X being a p×n matrix
        - p/n → y as n→∞
        - All elements of X are i.i.d with distribution β·δ(z_a) + (1-β)·δ(1)
        """)
    
    col_eigen1, col_eigen2 = st.columns([1, 2])
    with col_eigen1:
        n_samples = st.slider("Number of samples (n)", min_value=100, max_value=2000, value=1000, step=100)
        sim_seed = st.number_input("Random seed", min_value=1, max_value=1000, value=42, step=1)

    if st.button("Generate Eigenvalue Distribution", key="tab2_eigen_button"):
        with col_eigen2:
            fig_eigen = generate_eigenvalue_distribution(beta, y_2, z_a_2, n=n_samples, seed=sim_seed)
            if fig_eigen is not None:
                st.plotly_chart(fig_eigen, use_container_width=True)

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
        analyze_alt_low = st.checkbox("Alternate Low Expression", value=False)

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
                d1 = np.gradient(diff_curve, betas_diff)
                d2 = np.gradient(d1, betas_diff)
                
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=diff_curve, mode="lines", 
                                name="Upper-Lower Difference", line=dict(color="magenta", width=2)))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                name="Upper-Lower d/dβ", line=dict(color="magenta", dash='dash')))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                name="Upper-Lower d²/dβ²", line=dict(color="magenta", dash='dot')))

            if analyze_high_y:
                high_y_curve = compute_high_y_curve(betas_diff, z_a_diff, y_diff)
                d1 = np.gradient(high_y_curve, betas_diff)
                d2 = np.gradient(d1, betas_diff)
                
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=high_y_curve, mode="lines", 
                                name="High y", line=dict(color="green", width=2)))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                name="High y d/dβ", line=dict(color="green", dash='dash')))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                name="High y d²/dβ²", line=dict(color="green", dash='dot')))

            if analyze_alt_low:
                alt_low_curve = compute_alternate_low_expr(betas_diff, z_a_diff, y_diff)
                d1 = np.gradient(alt_low_curve, betas_diff)
                d2 = np.gradient(d1, betas_diff)
                
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=alt_low_curve, mode="lines", 
                                name="Alt Low", line=dict(color="orange", width=2)))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                name="Alt Low d/dβ", line=dict(color="orange", dash='dash')))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                name="Alt Low d²/dβ²", line=dict(color="orange", dash='dot')))

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