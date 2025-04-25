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

# Removed the compute_low_y_curve function

@st.cache_data
def compute_high_y_curve(betas, z_a, y):
    """
    Compute the "High y Expression" curve.
    """
    a = z_a
    betas = np.array(betas)
    denominator = 1 - 2*a
    if denominator == 0:
        return np.full_like(betas, np.nan)
    numerator = -4*a*(a-1)*y*betas - 2*a*y - 2*a*(2*a-1)
    return numerator/denominator

def compute_alternate_low_expr(betas, z_a, y):
    """
    Compute the alternate low expression:
    (z_a*y*beta*(z_a-1) - 2*z_a*(1-y) - 2*z_a**2) / (2+2*z_a)
    """
    betas = np.array(betas)
    return (z_a * y * betas * (z_a - 1) - 2*z_a*(1 - y) - 2*z_a**2) / (2 + 2*z_a)

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
        result = final_func(betas, z_a, y)
        if np.isscalar(result):
            result = np.full_like(betas, result)
    return result

def generate_z_vs_beta_plot(z_a, y, z_min, z_max, beta_steps, z_steps,
                          s_num_expr=None, s_denom_expr=None, 
                          z_num_expr=None, z_denom_expr=None,
                          show_derivatives=False):
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None

    betas = np.linspace(0, 1, beta_steps)
    betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
    # Removed low_y_curve computation
    high_y_curve = compute_high_y_curve(betas, z_a, y)
    alt_low_expr = compute_alternate_low_expr(betas, z_a, y)
    
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

    fig = go.Figure()
    
    # Original curves
    fig.add_trace(go.Scatter(x=betas, y=z_maxs, mode="markers+lines", 
                            name="Upper z*(β)", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=betas, y=z_mins, mode="markers+lines", 
                            name="Lower z*(β)", line=dict(color='lightblue')))
    # Removed the Low y Expression trace
    fig.add_trace(go.Scatter(x=betas, y=high_y_curve, mode="markers+lines", 
                            name="High y Expression", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=betas, y=alt_low_expr, mode="markers+lines", 
                            name="Alternate Low Expression", line=dict(color='orange')))
    
    if custom_curve1 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve1, mode="markers+lines", 
                                name="Custom 1 (s-based)", line=dict(color='purple')))
    if custom_curve2 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve2, mode="markers+lines", 
                                name="Custom 2 (direct)", line=dict(color='magenta')))

    if show_derivatives:
        # First derivatives
        curve_info = [
            ('upper', 'Upper z*(β)', 'blue'),
            ('lower', 'Lower z*(β)', 'lightblue'),
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

    fig.update_layout(
        title="Curves vs β: z*(β) Boundaries and Asymptotic Expressions",
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
    a = z * z_a
    b = z * z_a + z + z_a - z_a*y
    c = z + z_a + 1 - y*(beta*z_a + 1 - beta)
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
    
    Parameters:
    -----------
    beta : float
        Fraction of components equal to z_a
    y : float
        Aspect ratio p/n
    z_a : float
        Value for the delta mass at z_a
    n : int
        Number of samples
    seed : int
        Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Compute dimension p based on aspect ratio y
    p = int(y * n)
    
    # Constructing T_n (Population / Shape Matrix)
    T_diag = np.where(np.random.rand(p) < beta, z_a, 1.0)
    T_n = np.diag(T_diag)
    
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
    st.header("Find z Values where Cubic Roots Transition Between Real and Complex")
    col1, col2 = st.columns([1, 2])
    with col1:
        z_a_1 = st.number_input("z_a", value=1.0, key="z_a_1")
        y_1 = st.number_input("y", value=1.0, key="y_1")
        z_min_1 = st.number_input("z_min", value=-10.0, key="z_min_1")
        z_max_1 = st.number_input("z_max", value=10.0, key="z_max_1")
        with st.expander("Resolution Settings"):
            beta_steps = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, key="beta_steps")
            z_steps = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, step=1000, key="z_steps")
        
        st.subheader("Custom Expression 1 (s-based)")
        st.markdown("""Enter expressions for s = numerator/denominator 
                    (using variables `y`, `beta`, `z_a`, and `sqrt()`)""")
        st.latex(r"\text{This s will be inserted into:}")
        st.latex(r"\frac{y\beta(z_a-1)\underline{s}+(a\underline{s}+1)((y-1)\underline{s}-1)}{(a\underline{s}+1)(\underline{s}^2 + \underline{s})}")
        s_num = st.text_input("s numerator", value="", key="s_num")
        s_denom = st.text_input("s denominator", value="", key="s_denom")

        st.subheader("Custom Expression 2 (direct z(β))")
        st.markdown("""Enter direct expression for z(β) = numerator/denominator 
                    (using variables `y`, `beta`, `z_a`, and `sqrt()`)""")
        z_num = st.text_input("z(β) numerator", value="", key="z_num")
        z_denom = st.text_input("z(β) denominator", value="", key="z_denom")

        show_derivatives = st.checkbox("Show derivatives", value=False)

    if st.button("Compute z vs. β Curves", key="tab1_button"):
        with col2:
            fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, z_steps,
                                        s_num, s_denom, z_num, z_denom, show_derivatives)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Curve Explanations")
                st.markdown("""
                - **Upper z*(β)** (Blue): Maximum z value where discriminant is zero
                - **Lower z*(β)** (Light Blue): Minimum z value where discriminant is zero
                - **High y Expression** (Green): Asymptotic approximation for high y values
                - **Low Expression** (Orange): Alternative asymptotic expression
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
        with st.expander("Resolution Settings"):
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
    st.markdown("This page shows the difference between the Upper (blue) and Lower (lightblue) z*(β) curves, along with their first and second derivatives with respect to β.")
    col1, col2 = st.columns([1, 2])
    with col1:
        z_a_diff = st.number_input("z_a", value=1.0, key="z_a_diff")
        y_diff = st.number_input("y", value=1.0, key="y_diff")
        z_min_diff = st.number_input("z_min", value=-10.0, key="z_min_diff")
        z_max_diff = st.number_input("z_max", value=10.0, key="z_max_diff")
        with st.expander("Resolution Settings"):
            beta_steps_diff = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, key="beta_steps_diff")
            z_steps_diff = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, step=1000, key="z_steps_diff")
        
        # Add options for curve selection
        st.subheader("Curves to Analyze")
        analyze_upper_lower = st.checkbox("Upper-Lower Difference", value=True)
        analyze_high_y = st.checkbox("High y Expression", value=False)
        analyze_alt_low = st.checkbox("Alternate Low Expression", value=False)

    if st.button("Compute Differentials", key="tab3_button"):
        with col2:
            betas_diff, lower_vals, upper_vals = sweep_beta_and_find_z_bounds(z_a_diff, y_diff, z_min_diff, z_max_diff, beta_steps_diff, z_steps_diff)
            
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
                title="Differential Analysis vs. β",
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
            
            st.markdown("""
            ### Curve Types
            - Solid lines: Original curves
            - Dashed lines: First derivatives (d/dβ)
            - Dotted lines: Second derivatives (d²/dβ²)
            """)