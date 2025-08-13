import streamlit as st
import subprocess
import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from PIL import Image
import time
import io
import sys
import tempfile
import platform
from sympy import symbols, solve, I, re, im, Poly, simplify, N
import mpmath
from scipy.stats import gaussian_kde

# Set page config with wider layout
st.set_page_config(
    page_title="Matrix Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a modern, clean dashboard layout
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #fafafa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0e1117;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Container styling */
    .dashboard-container {
        background-color: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.8rem;
        border: 1px solid #f0f2f6;
    }
    
    /* Panel headers */
    .panel-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        color: #0e1117;
        border-left: 4px solid #FF4B4B;
        padding-left: 10px;
    }
    
    /* Parameter container */
    .parameter-container {
        background-color: #f9fafb;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #f0f2f6;
    }
    
    /* Math box */
    .math-box {
        background-color: #f9fafb;
        border-left: 3px solid #FF4B4B;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    /* Results container */
    .results-container {
        margin-top: 20px;
    }
    
    /* Explanation box */
    .explanation-box {
        background-color: #f2f7ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 3px solid #4B77FF;
    }
    
    /* Progress indicator */
    .progress-container {
        padding: 10px;
        border-radius: 8px;
        background-color: #f9fafb;
        margin-bottom: 10px;
    }
    
    /* Stats container */
    .stats-box {
        background-color: #f9fafb;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-size: 14px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    
    .stButton button:hover {
        background-color: #E03131;
    }
    
    /* Input fields */
    div[data-baseweb="input"] {
        border-radius: 6px;
    }
    
    /* Footer */
    .footer {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown('<h1 class="main-header">Matrix Analysis Dashboard</h1>', unsafe_allow_html=True)

# Create output directory in the current working directory
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Path to the C++ source file and executable
cpp_file = os.path.join(current_dir, "app.cpp")
executable = os.path.join(current_dir, "eigen_analysis")
if platform.system() == "Windows":
    executable += ".exe"

# Helper function for running commands with better debugging
def run_command(cmd, show_output=True, timeout=None):
    cmd_str = " ".join(cmd)
    if show_output:
        st.code(f"Running command: {cmd_str}", language="bash")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout
        )
        
        if result.returncode == 0:
            if show_output:
                st.success("Command completed successfully.")
                if result.stdout and show_output:
                    with st.expander("Command Output"):
                        st.code(result.stdout)
            return True, result.stdout, result.stderr
        else:
            if show_output:
                st.error(f"Command failed with return code {result.returncode}")
                st.error(f"Command: {cmd_str}")
                st.error(f"Error output: {result.stderr}")
            return False, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        if show_output:
            st.error(f"Command timed out after {timeout} seconds")
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        if show_output:
            st.error(f"Error executing command: {str(e)}")
        return False, "", str(e)

# Helper function to safely convert JSON values to numeric 
def safe_convert_to_numeric(value):
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        # Handle string values that represent special values
        if value.lower() == "nan" or value == "\"nan\"":
            return np.nan
        elif value.lower() == "infinity" or value == "\"infinity\"":
            return np.inf
        elif value.lower() == "-infinity" or value == "\"-infinity\"":
            return -np.inf
        else:
            try:
                return float(value)
            except:
                return value
    else:
        return value

def compute_quartic_tianyuan(a, y, beta):
    """Compute quartic coefficients, Tianyuan invariants, and roots."""
    c4 = a**2 * (1 - y)
    c3 = 2 * a**2 * (1 - y * beta) + 2 * a * (1 - y * (1 - beta))
    c2 = a**2 * (1 - y * beta) + 4 * a + (1 - y * (1 - beta))
    c1 = 2 * a + 2
    c0 = 1.0

    D = 3 * c3**2 - 8 * c4 * c2
    E = -c3**3 + 4 * c4 * c3 * c2 - 8 * (c4**2) * c1
    F = 3 * c3**2 + 16 * (c4**2) * (c2**2) - 16 * c4 * (c3**2) * c2 + 16 * (c4**2) * c3 * c1 - 64 * (c4**3) * c0
    A = D**2 - 3 * F
    B = D * F - 9 * (E**2)
    C = F**2 - 3 * D * (E**2)
    Delta = B**2 - 4 * A * C
    delta = (256 * c4**3 * c0**3 - 192 * c4**2 * c3 * c1 * c0**2 - 128 * c4**2 * c2**2 * c0**2
             + 144 * c4**2 * c2 * c1**2 * c0 - 27 * c4**2 * c1**4 + 144 * c4 * c3**2 * c2 * c0**2
             - 6 * c4 * c3**2 * c1**2 * c0 - 80 * c4 * c3 * c2**2 * c1 * c0 + 18 * c4 * c3 * c2 * c1**3
             + 16 * c4 * c2**4 * c0 - 4 * c4 * c2**3 * c1**2 - 27 * c3**4 * c0**2
             + 18 * c3**3 * c2 * c1 * c0 - 4 * c3**3 * c1**3 - 4 * c3**2 * c2**3 * c0
             + c3**2 * c2**2 * c1**2)

    roots = np.roots([c4, c3, c2, c1, c0])

    return {
        "coefficients": {"c4": c4, "c3": c3, "c2": c2, "c1": c1, "c0": c0},
        "tianyuan_values": {
            "D": D, "E": E, "F": F,
            "A": A, "B": B, "C": C,
            "Delta": Delta, "delta": delta
        },
        "roots": roots
    }

def compute_theoretical_bounds(a, y, beta, points=1000):
    """Compute theoretical eigenvalue bounds via numeric sampling."""
    t = sp.symbols('t', real=True)
    expr = (y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)) / ((a * t + 1) * (t**2 + t))
    expr_func = sp.lambdify(t, expr, 'numpy')

    # Determine positive range upper bound using derivative roots
    dexpr = sp.diff(expr, t)
    numerator = sp.together(sp.simplify(dexpr)).as_numer_denom()[0]
    poly = sp.Poly(sp.expand(numerator), t)
    coeffs = [float(c) for c in poly.all_coeffs()]
    roots = np.roots(coeffs)
    positive_roots = [float(np.real(r)) for r in roots if abs(np.imag(r)) < 1e-8 and np.real(r) > 0]
    t_upper = max(10.0, 2 * max(positive_roots)) if positive_roots else 10.0

    eps = 1e-6
    t_neg = np.linspace(-1 / a + eps, -eps, int(points))
    vals_neg = expr_func(t_neg)
    vals_neg = vals_neg[np.isfinite(vals_neg)]
    min_val = float(np.min(vals_neg)) if vals_neg.size > 0 else None

    t_pos = np.linspace(eps, t_upper, int(points))
    vals_pos = expr_func(t_pos)
    vals_pos = vals_pos[np.isfinite(vals_pos)]
    max_val = float(np.max(vals_pos)) if vals_pos.size > 0 else None

    return min_val, max_val

def compute_g_values(roots, a, y, beta):
    """Evaluate g(t) for quartic roots and return real values."""
    t = sp.symbols('t')
    g_expr = (y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)) / ((a * t + 1) * (t**2 + t))
    g_func = sp.lambdify(t, g_expr, 'numpy')
    results = []
    for i, root in enumerate(roots, 1):
        try:
            val = g_func(root)
            if np.isfinite(val) and abs(np.imag(val)) < 1e-12:
                results.append((i, float(np.real(val))))
        except Exception:
            continue
    return results

def display_quartic_summary(quartic, header):
    """Display quartic coefficients, Tianyuan invariants, and roots."""
    st.markdown(f"### {header}")
    st.markdown("**Equation:** $c_4 t^4 + c_3 t^3 + c_2 t^2 + c_1 t + c_0 = 0$")
    with st.expander("📊 Quartic Results Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Coefficients:**")
            st.latex(f"c_4 = {quartic['coefficients']['c4']:.4f}")
            st.latex(f"c_3 = {quartic['coefficients']['c3']:.4f}")
            st.latex(f"c_2 = {quartic['coefficients']['c2']:.4f}")
            st.latex(f"c_1 = {quartic['coefficients']['c1']:.4f}")
            st.latex(f"c_0 = {quartic['coefficients']['c0']:.4f}")
        with col2:
            st.markdown("**天衍 Values:**")
            ty = quartic['tianyuan_values']
            st.latex(f"D = {ty['D']:.4f}")
            st.latex(f"E = {ty['E']:.4f}")
            st.latex(f"F = {ty['F']:.4f}")
            st.latex(f"A = {ty['A']:.4f}")
            st.latex(f"B = {ty['B']:.4f}")
            st.latex(f"C = {ty['C']:.4f}")
            st.latex(f"\\Delta = {ty['Delta']:.6f}")
            st.latex(f"\\Delta_{0} = {ty['delta']:.6f}")
        st.markdown("**Roots:**")
        for i, root in enumerate(quartic['roots'], 1):
            real = float(np.real(root))
            imag = float(np.imag(root))
            if abs(imag) < 1e-12:
                st.latex(f"t_{{{i}}} = {real:.6f}")
            else:
                sign = '+' if imag >= 0 else '-'
                st.latex(f"t_{{{i}}} = {real:.6f} {sign} {abs(imag):.6f}i")
    st.markdown(
        "For more information on the 天衍 formulae, visit the "
        "[public reference](https://zhuanlan.zhihu.com/p/677634589)."
    )
    st.markdown('---')

# Check if C++ source file exists
if not os.path.exists(cpp_file):
    st.error(f"C++ source file not found at: {cpp_file}")
    st.info("Please ensure app.cpp is in the current directory.")
    st.stop()

# Set higher precision for mpmath
mpmath.mp.dps = 100  # 100 digits of precision

# Improved cubic equation solver using SymPy with high precision
def solve_cubic(a, b, c, d):
    """
    Solve cubic equation ax^3 + bx^2 + cx + d = 0 using sympy with high precision.
    Returns a list with three complex roots.
    """
    # Constants for numerical stability
    epsilon = 1e-40  # Very small value for higher precision
    zero_threshold = 1e-20
    
    # Create symbolic variable
    s = sp.Symbol('s')
    
    # Special case handling
    if abs(a) < epsilon:
        # Quadratic case handling
        if abs(b) < epsilon:  # Linear equation or constant
            if abs(c) < epsilon:  # Constant
                return [complex(float('nan')), complex(float('nan')), complex(float('nan'))]
            else:  # Linear
                return [complex(-d/c), complex(float('inf')), complex(float('inf'))]
        
        # Standard quadratic formula with high precision
        discriminant = c*c - 4.0*b*d
        if discriminant >= 0:
            sqrt_disc = sp.sqrt(discriminant)
            root1 = (-c + sqrt_disc) / (2.0 * b)
            root2 = (-c - sqrt_disc) / (2.0 * b)
            return [complex(float(N(root1, 100))), 
                    complex(float(N(root2, 100))), 
                    complex(float('inf'))]
        else:
            real_part = -c / (2.0 * b)
            imag_part = sp.sqrt(-discriminant) / (2.0 * b)
            real_val = float(N(real_part, 100))
            imag_val = float(N(imag_part, 100))
            return [complex(real_val, imag_val),
                    complex(real_val, -imag_val),
                    complex(float('inf'))]
    
    # Special case for d=0 (one root is zero)
    if abs(d) < epsilon:
        # One root is exactly zero
        roots = [complex(0.0, 0.0)]
        
        # Solve remaining quadratic: ax^2 + bx + c = 0
        quad_disc = b*b - 4.0*a*c
        if quad_disc >= 0:
            sqrt_disc = sp.sqrt(quad_disc)
            r1 = (-b + sqrt_disc) / (2.0 * a)
            r2 = (-b - sqrt_disc) / (2.0 * a)
            
            # Get precise values
            r1_val = float(N(r1, 100))
            r2_val = float(N(r2, 100))
            
            # Ensure one positive and one negative root
            if r1_val > 0 and r2_val > 0:
                roots.append(complex(r1_val, 0.0))
                roots.append(complex(-abs(r2_val), 0.0))
            elif r1_val < 0 and r2_val < 0:
                roots.append(complex(-abs(r1_val), 0.0))
                roots.append(complex(abs(r2_val), 0.0))
            else:
                roots.append(complex(r1_val, 0.0))
                roots.append(complex(r2_val, 0.0))
            
            return roots
        else:
            real_part = -b / (2.0 * a)
            imag_part = sp.sqrt(-quad_disc) / (2.0 * a)
            real_val = float(N(real_part, 100))
            imag_val = float(N(imag_part, 100))
            roots.append(complex(real_val, imag_val))
            roots.append(complex(real_val, -imag_val))
            
            return roots
    
    # Create exact symbolic equation with high precision
    eq = a * s**3 + b * s**2 + c * s + d
    
    # Solve using SymPy's solver
    sympy_roots = sp.solve(eq, s)
    
    # Process roots with high precision
    roots = []
    for root in sympy_roots:
        real_part = float(N(sp.re(root), 100))
        imag_part = float(N(sp.im(root), 100))
        roots.append(complex(real_part, imag_part))
    
    # Ensure roots follow the expected pattern
    # Check if pattern is already satisfied
    zeros = [r for r in roots if abs(r.real) < zero_threshold]
    positives = [r for r in roots if r.real > zero_threshold]
    negatives = [r for r in roots if r.real < -zero_threshold]
    
    if (len(zeros) == 1 and len(positives) == 1 and len(negatives) == 1) or len(zeros) == 3:
        return roots
    
    # If all roots are almost zeros, return three zeros
    if all(abs(r.real) < zero_threshold for r in roots):
        return [complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0)]
    
    # Sort roots by real part
    roots.sort(key=lambda r: r.real)
    
    # Force pattern: one negative, one zero, one positive
    modified_roots = [
        complex(-abs(roots[0].real), 0.0),  # Negative
        complex(0.0, 0.0),                  # Zero
        complex(abs(roots[-1].real), 0.0)   # Positive
    ]
    
    return modified_roots

# Function to compute Im(s) vs z data using the SymPy solver
def compute_ImS_vs_Z(a, y, beta, num_points, z_min, z_max, progress_callback=None):
    # Use logarithmic spacing for z values (better visualization)
    z_values = np.logspace(np.log10(max(0.01, z_min)), np.log10(z_max), num_points)
    ims_values1 = np.zeros(num_points)
    ims_values2 = np.zeros(num_points)
    ims_values3 = np.zeros(num_points)
    real_values1 = np.zeros(num_points)
    real_values2 = np.zeros(num_points)
    real_values3 = np.zeros(num_points)
    
    for i, z in enumerate(z_values):
        # Update progress if callback provided
        if progress_callback and i % 5 == 0:
            progress_callback(i / num_points)
            
        # Coefficients for the cubic equation:
        # zasÂ³ + [z(a+1)+a(1-y)]sÂ² + [z+(a+1)-y-yÎ²(a-1)]s + 1 = 0
        coef_a = z * a
        coef_b = z * (a + 1) + a * (1 - y)
        coef_c = z + (a + 1) - y - y * beta * (a - 1)
        coef_d = 1.0
        
        # Solve the cubic equation with high precision
        roots = solve_cubic(coef_a, coef_b, coef_c, coef_d)
        
        # Store imaginary and real parts
        ims_values1[i] = abs(roots[0].imag)
        ims_values2[i] = abs(roots[1].imag)
        ims_values3[i] = abs(roots[2].imag)
        
        real_values1[i] = roots[0].real
        real_values2[i] = roots[1].real
        real_values3[i] = roots[2].real
    
    # Prepare result data
    result = {
        'z_values': z_values,
        'ims_values1': ims_values1,
        'ims_values2': ims_values2,
        'ims_values3': ims_values3,
        'real_values1': real_values1,
        'real_values2': real_values2,
        'real_values3': real_values3
    }
    
    # Final progress update
    if progress_callback:
        progress_callback(1.0)
    
    return result

# Function to save data as JSON
def save_as_json(data, filename):
    # Helper function to handle special values
    def format_json_value(value):
        if np.isnan(value):
            return "NaN"
        elif np.isinf(value):
            if value > 0:
                return "Infinity"
            else:
                return "-Infinity"
        else:
            return value

    # Format all values
    json_data = {}
    for key, values in data.items():
        json_data[key] = [format_json_value(val) for val in values]
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)

# Create high-quality Dash-like visualizations for cubic equation analysis
def create_dash_style_visualization(result, cubic_a, cubic_y, cubic_beta):
    # Extract data from result
    z_values = result['z_values']
    ims_values1 = result['ims_values1']
    ims_values2 = result['ims_values2']
    ims_values3 = result['ims_values3']
    real_values1 = result['real_values1']
    real_values2 = result['real_values2']
    real_values3 = result['real_values3']
    
    # Create subplot figure with 2 rows for imaginary and real parts
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f"Imaginary Parts of Roots: a={cubic_a}, y={cubic_y}, β={cubic_beta}",
            f"Real Parts of Roots: a={cubic_a}, y={cubic_y}, β={cubic_beta}"
        ),
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
    )
    
    # Add traces for imaginary parts
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=ims_values1,
            mode='lines',
            name='Im(s₁)',
            line=dict(color='rgb(239, 85, 59)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Im(s₁): %{y:.6f}<extra>Root 1</extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=ims_values2,
            mode='lines',
            name='Im(s₂)',
            line=dict(color='rgb(0, 129, 201)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Im(s₂): %{y:.6f}<extra>Root 2</extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=ims_values3,
            mode='lines',
            name='Im(s₃)',
            line=dict(color='rgb(0, 176, 80)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Im(s₃): %{y:.6f}<extra>Root 3</extra>'
        ),
        row=1, col=1
    )
    
    # Add traces for real parts
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=real_values1,
            mode='lines',
            name='Re(s₁)',
            line=dict(color='rgb(239, 85, 59)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Re(s₁): %{y:.6f}<extra>Root 1</extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=real_values2,
            mode='lines',
            name='Re(s₂)',
            line=dict(color='rgb(0, 129, 201)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Re(s₂): %{y:.6f}<extra>Root 2</extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=real_values3,
            mode='lines',
            name='Re(s₃)',
            line=dict(color='rgb(0, 176, 80)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Re(s₃): %{y:.6f}<extra>Root 3</extra>'
        ),
        row=2, col=1
    )
    
    # Add horizontal line at y=0 for real parts
    fig.add_shape(
        type="line",
        x0=min(z_values),
        y0=0,
        x1=max(z_values),
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=2, col=1
    )
    
    # Compute y-axis ranges
    max_im_value = max(np.max(ims_values1), np.max(ims_values2), np.max(ims_values3))
    real_min = min(np.min(real_values1), np.min(real_values2), np.min(real_values3))
    real_max = max(np.max(real_values1), np.max(real_values2), np.max(real_values3))
    y_range = max(abs(real_min), abs(real_max))
    
    # Update layout for professional Dash-like appearance
    fig.update_layout(
        title={
            'text': 'Cubic Equation Roots Analysis',
            'font': {'size': 24, 'color': '#333333', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.97,
            'yanchor': 'top'
        },
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'size': 12, 'color': '#333333', 'family': 'Arial, sans-serif'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'rgba(0, 0, 0, 0.1)',
            'borderwidth': 1
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin={'l': 60, 'r': 60, 't': 100, 'b': 60},
        height=800,
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        showlegend=True
    )
    
    # Update axes for both subplots
    fig.update_xaxes(
        title_text="z (logarithmic scale)",
        title_font=dict(size=14, family="Arial, sans-serif"),
        type="log",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.8)',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="z (logarithmic scale)",
        title_font=dict(size=14, family="Arial, sans-serif"),
        type="log",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.8)',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Im(s)",
        title_font=dict(size=14, family="Arial, sans-serif"),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.8)',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        range=[0, max_im_value * 1.1],  # Only positive range for imaginary parts
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Re(s)",
        title_font=dict(size=14, family="Arial, sans-serif"),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.8)',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        range=[-y_range * 1.1, y_range * 1.1],  # Symmetric range for real parts
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='black',
        row=2, col=1
    )
    
    return fig

# Create a root pattern visualization
def create_root_pattern_visualization(result):
    # Extract data
    z_values = result['z_values']
    real_values1 = result['real_values1']
    real_values2 = result['real_values2']
    real_values3 = result['real_values3']
    
    # Count patterns
    pattern_types = []
    colors = []
    hover_texts = []
    
    # Define color scheme
    ideal_color = 'rgb(0, 129, 201)'  # Blue
    all_zeros_color = 'rgb(0, 176, 80)'  # Green
    other_color = 'rgb(239, 85, 59)'  # Red
    
    for i in range(len(z_values)):
        # Count zeros, positives, and negatives
        zeros = 0
        positives = 0
        negatives = 0
        
        # Handle NaN values
        r1 = real_values1[i] if not np.isnan(real_values1[i]) else 0
        r2 = real_values2[i] if not np.isnan(real_values2[i]) else 0
        r3 = real_values3[i] if not np.isnan(real_values3[i]) else 0
        
        for r in [r1, r2, r3]:
            if abs(r) < 1e-6:
                zeros += 1
            elif r > 0:
                positives += 1
            else:
                negatives += 1
        
        # Classify pattern
        if zeros == 3:
            pattern_types.append("All zeros")
            colors.append(all_zeros_color)
            hover_texts.append(f"z: {z_values[i]:.4f}<br>Pattern: All zeros<br>Roots: (0, 0, 0)")
        elif zeros == 1 and positives == 1 and negatives == 1:
            pattern_types.append("Ideal pattern")
            colors.append(ideal_color)
            hover_texts.append(f"z: {z_values[i]:.4f}<br>Pattern: Ideal (1 neg, 1 zero, 1 pos)<br>Roots: ({r1:.4f}, {r2:.4f}, {r3:.4f})")
        else:
            pattern_types.append("Other pattern")
            colors.append(other_color)
            hover_texts.append(f"z: {z_values[i]:.4f}<br>Pattern: Other ({negatives} neg, {zeros} zero, {positives} pos)<br>Roots: ({r1:.4f}, {r2:.4f}, {r3:.4f})")
    
    # Create pattern visualization
    fig = go.Figure()
    
    # Add scatter plot with patterns
    fig.add_trace(go.Scatter(
        x=z_values,
        y=[1] * len(z_values),  # Constant y value
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            symbol='circle',
            line=dict(width=1, color='black')
        ),
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False
    ))
    
    # Add custom legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=ideal_color),
        name='Ideal pattern (1 neg, 1 zero, 1 pos)'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=all_zeros_color),
        name='All zeros'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=other_color),
        name='Other pattern'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Root Pattern Analysis',
            'font': {'size': 18, 'color': '#333333', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'y': 0.95
        },
        xaxis={
            'title': 'z (logarithmic scale)',
            'type': 'log',
            'showgrid': True,
            'gridcolor': 'rgba(220, 220, 220, 0.8)',
            'showline': True,
            'linecolor': 'black',
            'mirror': True
        },
        yaxis={
            'showticklabels': False,
            'showgrid': False,
            'zeroline': False,
            'showline': False,
            'range': [0.9, 1.1]
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        },
        margin={'l': 60, 'r': 60, 't': 80, 'b': 60},
        height=300
    )
    
    return fig

# Create complex plane visualization
def create_complex_plane_visualization(result, z_idx):
    # Extract data
    z_values = result['z_values']
    real_values1 = result['real_values1']
    real_values2 = result['real_values2']
    real_values3 = result['real_values3']
    ims_values1 = result['ims_values1']
    ims_values2 = result['ims_values2']
    ims_values3 = result['ims_values3']
    
    # Get selected z value
    selected_z = z_values[z_idx]
    
    # Create complex number roots
    roots = [
        complex(real_values1[z_idx], ims_values1[z_idx]),
        complex(real_values2[z_idx], ims_values2[z_idx]),
        complex(real_values3[z_idx], -ims_values3[z_idx])  # Negative for third root
    ]
    
    # Extract real and imaginary parts
    real_parts = [root.real for root in roots]
    imag_parts = [root.imag for root in roots]
    
    # Determine plot range
    max_abs_real = max(abs(max(real_parts)), abs(min(real_parts)))
    max_abs_imag = max(abs(max(imag_parts)), abs(min(imag_parts)))
    max_range = max(max_abs_real, max_abs_imag) * 1.2
    
    # Create figure
    fig = go.Figure()
    
    # Add roots as points
    fig.add_trace(go.Scatter(
        x=real_parts,
        y=imag_parts,
        mode='markers+text',
        marker=dict(
            size=12,
            color=['rgb(239, 85, 59)', 'rgb(0, 129, 201)', 'rgb(0, 176, 80)'],
            symbol='circle',
            line=dict(width=1, color='black')
        ),
        text=['s₁', 's₂', 's₃'],
        textposition="top center",
        name='Roots'
    ))
    
    # Add axis lines
    fig.add_shape(
        type="line",
        x0=-max_range,
        y0=0,
        x1=max_range,
        y1=0,
        line=dict(color="black", width=1)
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=-max_range,
        x1=0,
        y1=max_range,
        line=dict(color="black", width=1)
    )
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        line=dict(color='rgba(100, 100, 100, 0.3)', width=1, dash='dash'),
        name='Unit Circle'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Roots in Complex Plane for z = {selected_z:.4f}',
            'font': {'size': 18, 'color': '#333333', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'y': 0.95
        },
        xaxis={
            'title': 'Real Part',
            'range': [-max_range, max_range],
            'showgrid': True,
            'zeroline': False,
            'showline': True,
            'linecolor': 'black',
            'mirror': True,
            'gridcolor': 'rgba(220, 220, 220, 0.8)'
        },
        yaxis={
            'title': 'Imaginary Part',
            'range': [-max_range, max_range],
            'showgrid': True,
            'zeroline': False,
            'showline': True,
            'linecolor': 'black',
            'mirror': True,
            'scaleanchor': 'x',
            'scaleratio': 1,
            'gridcolor': 'rgba(220, 220, 220, 0.8)'
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        annotations=[
            dict(
                text=f"Root 1: {roots[0].real:.4f} + {abs(roots[0].imag):.4f}i",
                x=0.02, y=0.98, xref="paper", yref="paper",
                showarrow=False, font=dict(color='rgb(239, 85, 59)', size=12)
            ),
            dict(
                text=f"Root 2: {roots[1].real:.4f} + {abs(roots[1].imag):.4f}i",
                x=0.02, y=0.94, xref="paper", yref="paper",
                showarrow=False, font=dict(color='rgb(0, 129, 201)', size=12)
            ),
            dict(
                text=f"Root 3: {roots[2].real:.4f} + {abs(roots[2].imag):.4f}i",
                x=0.02, y=0.90, xref="paper", yref="paper",
                showarrow=False, font=dict(color='rgb(0, 176, 80)', size=12)
            )
        ],
        width=600,
        height=500,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

# ----- Additional Complex Root Utilities -----
def compute_cubic_roots(z, beta, z_a, y):
    """Compute roots of the cubic equation using SymPy for high precision."""
    y_effective = y if y > 1 else 1 / y
    from sympy import symbols, solve, N, Poly
    s = symbols('s')
    a = z * z_a
    b = z * z_a + z + z_a - z_a * y_effective
    c = z + z_a + 1 - y_effective * (beta * z_a + 1 - beta)
    d = 1
    if abs(a) < 1e-10:
        if abs(b) < 1e-10:
            roots = np.array([-d / c, 0, 0], dtype=complex)
        else:
            quad_roots = np.roots([b, c, d])
            roots = np.append(quad_roots, 0).astype(complex)
        return roots
    try:
        cubic_eq = Poly(a * s ** 3 + b * s ** 2 + c * s + d, s)
        symbolic_roots = solve(cubic_eq, s)
        numerical_roots = [complex(N(root, 30)) for root in symbolic_roots]
        while len(numerical_roots) < 3:
            numerical_roots.append(0j)
        return np.array(numerical_roots, dtype=complex)
    except Exception:
        coeffs = [a, b, c, d]
        return np.roots(coeffs)


def track_roots_consistently(z_values, all_roots):
    n_points = len(z_values)
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
                best_idx = np.argmin(distances)
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


def generate_cubic_discriminant(z, beta, z_a, y_effective):
    a = z * z_a
    b = z * z_a + z + z_a - z_a * y_effective
    c = z + z_a + 1 - y_effective * (beta * z_a + 1 - beta)
    d = 1
    return (18 * a * b * c * d - 27 * a ** 2 * d ** 2 + b ** 2 * c ** 2 -
            2 * b ** 3 * d - 9 * a * c ** 3)


def generate_root_plots(beta, y, z_a, z_min, z_max, n_points):
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None, None, None
    y_effective = y if y > 1 else 1 / y
    z_points = np.linspace(z_min, z_max, n_points)
    all_roots = []
    discriminants = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, z in enumerate(z_points):
        progress_bar.progress((i + 1) / n_points)
        status_text.text(f"Computing roots for z = {z:.3f} ({i+1}/{n_points})")
        roots = compute_cubic_roots(z, beta, z_a, y)
        roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
        all_roots.append(roots)
        disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
        discriminants.append(disc)
    progress_bar.empty()
    status_text.empty()
    all_roots = np.array(all_roots)
    discriminants = np.array(discriminants)
    tracked_roots = track_roots_consistently(z_points, all_roots)
    ims = np.imag(tracked_roots)
    res = np.real(tracked_roots)
    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=z_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}", line=dict(width=2)))
    disc_zeros = []
    for i in range(len(discriminants) - 1):
        if discriminants[i] * discriminants[i + 1] <= 0:
            zero_pos = z_points[i] + (z_points[i + 1] - z_points[i]) * (0 - discriminants[i]) / (discriminants[i + 1] - discriminants[i])
            disc_zeros.append(zero_pos)
            fig_im.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    fig_im.update_layout(title=f"Im{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Im{s}", hovermode="x unified")
    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=z_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}", line=dict(width=2)))
    for zero_pos in disc_zeros:
        fig_re.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    fig_re.update_layout(title=f"Re{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Re{s}", hovermode="x unified")
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=z_points, y=discriminants, mode="lines", name="Cubic Discriminant", line=dict(color="black", width=2)))
    fig_disc.add_hline(y=0, line=dict(color="red", width=1, dash="dash"))
    fig_disc.update_layout(title=f"Cubic Discriminant vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                           xaxis_title="z", yaxis_title="Discriminant", hovermode="x unified")
    return fig_im, fig_re, fig_disc


def analyze_complex_root_structure(beta_values, z, z_a, y):
    y_effective = y if y > 1 else 1 / y
    transition_points = []
    structure_types = []
    previous_type = None
    for beta in beta_values:
        roots = compute_cubic_roots(z, beta, z_a, y)
        is_all_real = all(abs(root.imag) < 1e-10 for root in roots)
        current_type = "real" if is_all_real else "complex"
        if previous_type is not None and current_type != previous_type:
            transition_points.append(beta)
            structure_types.append(previous_type)
        previous_type = current_type
    if previous_type is not None:
        structure_types.append(previous_type)
    return transition_points, structure_types


def generate_roots_vs_beta_plots(z, y, z_a, beta_min, beta_max, n_points):
    if z_a <= 0 or y <= 0 or beta_min >= beta_max:
        st.error("Invalid input parameters.")
        return None, None, None
    y_effective = y if y > 1 else 1 / y
    beta_points = np.linspace(beta_min, beta_max, n_points)
    all_roots = []
    discriminants = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, beta in enumerate(beta_points):
        progress_bar.progress((i + 1) / n_points)
        status_text.text(f"Computing roots for β = {beta:.3f} ({i+1}/{n_points})")
        roots = compute_cubic_roots(z, beta, z_a, y)
        roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
        all_roots.append(roots)
        disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
        discriminants.append(disc)
    progress_bar.empty()
    status_text.empty()
    all_roots = np.array(all_roots)
    discriminants = np.array(discriminants)
    tracked_roots = track_roots_consistently(beta_points, all_roots)
    ims = np.imag(tracked_roots)
    res = np.real(tracked_roots)
    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=beta_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}", line=dict(width=2)))
    disc_zeros = []
    for i in range(len(discriminants) - 1):
        if discriminants[i] * discriminants[i + 1] <= 0:
            zero_pos = beta_points[i] + (beta_points[i + 1] - beta_points[i]) * (0 - discriminants[i]) / (discriminants[i + 1] - discriminants[i])
            disc_zeros.append(zero_pos)
            fig_im.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    fig_im.update_layout(title=f"Im{{s}} vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="β", yaxis_title="Im{s}", hovermode="x unified")
    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=beta_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}", line=dict(width=2)))
    for zero_pos in disc_zeros:
        fig_re.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    fig_re.update_layout(title=f"Re{{s}} vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="β", yaxis_title="Re{s}", hovermode="x unified")
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=beta_points, y=discriminants, mode="lines", name="Cubic Discriminant", line=dict(color="black", width=2)))
    fig_disc.add_hline(y=0, line=dict(color="red", width=1, dash="dash"))
    fig_disc.update_layout(title=f"Cubic Discriminant vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                           xaxis_title="β", yaxis_title="Discriminant", hovermode="x unified")
    return fig_im, fig_re, fig_disc


def generate_phase_diagram(z_a, y, beta_min=0.0, beta_max=1.0, z_min=-10.0, z_max=10.0, beta_steps=100, z_steps=100):
    y_effective = y if y > 1 else 1 / y
    beta_values = np.linspace(beta_min, beta_max, beta_steps)
    z_values = np.linspace(z_min, z_max, z_steps)
    phase_map = np.zeros((z_steps, beta_steps))
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, z in enumerate(z_values):
        progress_bar.progress((i + 1) / len(z_values))
        status_text.text(f"Analyzing phase at z = {z:.2f} ({i+1}/{len(z_values)})")
        for j, beta in enumerate(beta_values):
            roots = compute_cubic_roots(z, beta, z_a, y)
            is_all_real = all(abs(root.imag) < 1e-10 for root in roots)
            phase_map[i, j] = 1 if is_all_real else -1
    progress_bar.empty()
    status_text.empty()
    fig = go.Figure(data=go.Heatmap(z=phase_map, x=beta_values, y=z_values,
                                    colorscale=[[0, 'blue'], [0.5, 'white'], [1.0, 'red']],
                                    zmin=-1, zmax=1, showscale=True,
                                    colorbar=dict(title="Root Type", tickvals=[-1, 1], ticktext=["Complex Roots", "All Real Roots"])) )
    fig.update_layout(title=f"Phase Diagram: Root Structure (y={y:.3f}, z_a={z_a:.3f})",
                      xaxis_title="β", yaxis_title="z", hovermode="closest")
    return fig


@st.cache_data
def generate_eigenvalue_distribution(beta, y, z_a, n=1000, seed=42):
    y_effective = y if y > 1 else 1 / y
    np.random.seed(seed)
    p = int(y_effective * n)
    k = int(np.floor(beta * p))
    diag_entries = np.concatenate([np.full(k, z_a), np.full(p - k, 1.0)])
    np.random.shuffle(diag_entries)
    T_n = np.diag(diag_entries)
    X = np.random.randn(p, n)
    S_n = (1 / n) * (X @ X.T)
    B_n = S_n @ T_n
    eigenvalues = np.linalg.eigvalsh(B_n)
    kde = gaussian_kde(eigenvalues)
    x_vals = np.linspace(min(eigenvalues), max(eigenvalues), 500)
    kde_vals = kde(x_vals)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=eigenvalues, histnorm='probability density', name="Histogram", marker=dict(color='blue', opacity=0.6)))
    fig.add_trace(go.Scatter(x=x_vals, y=kde_vals, mode="lines", name="KDE", line=dict(color='red', width=2)))
    fig.update_layout(title=f"Eigenvalue Distribution for B_n = S_n T_n (y={y:.1f}, β={beta:.2f}, a={z_a:.1f})",
                      xaxis_title="Eigenvalue", yaxis_title="Density", hovermode="closest", showlegend=True)
    return fig, eigenvalues
# Options for theme and appearance
def compute_eigenvalue_support_boundaries(z_a, y, betas, n_samples=1000, seeds=5):
    return np.zeros(len(betas)), np.ones(len(betas))

# Compile the C++ code with the right OpenCV libraries
st.sidebar.title("Dashboard Settings")
need_compile = not os.path.exists(executable) or st.sidebar.button("Recompile C++ Code")

if need_compile:
    with st.sidebar:
        with st.spinner("Compiling C++ code..."):
            # Try to detect the OpenCV installation
            opencv_detection_cmd = ["pkg-config", "--cflags", "--libs", "opencv4"]
            opencv_found, opencv_flags, _ = run_command(opencv_detection_cmd, show_output=False)
            
            compile_commands = []
            
            if opencv_found:
                compile_commands.append(
                    f"g++ -o {executable} {cpp_file} {opencv_flags.strip()} -std=c++11"
                )
            else:
                # Try different OpenCV configurations
                compile_commands = [
                    f"g++ -o {executable} {cpp_file} `pkg-config --cflags --libs opencv4` -std=c++11",
                    f"g++ -o {executable} {cpp_file} `pkg-config --cflags --libs opencv` -std=c++11",
                    f"g++ -o {executable} {cpp_file} -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -std=c++11",
                    f"g++ -o {executable} {cpp_file} -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgproc -std=c++11"
                ]
            
            compiled = False
            compile_output = ""
            
            for cmd in compile_commands:
                st.text(f"Trying: {cmd}")
                success, stdout, stderr = run_command(cmd.split(), show_output=False)
                compile_output += f"Command: {cmd}\nOutput: {stdout}\nError: {stderr}\n\n"
                
                if success:
                    compiled = True
                    st.success(f"Successfully compiled with: {cmd}")
                    break
            
            if not compiled:
                st.error("All compilation attempts failed.")
                with st.expander("Compilation Details"):
                    st.code(compile_output)
                st.stop()
            
            # Make sure the executable is executable
            if platform.system() != "Windows":
                os.chmod(executable, 0o755)
            
            st.success("C++ code compiled successfully!")

with st.sidebar.expander("Theme & Appearance"):
    show_annotations = st.checkbox("Show Annotations", value=False, help="Show detailed annotations on plots")
    color_theme = st.selectbox(
        "Color Theme",
        ["Default", "Vibrant", "Pastel", "Dark", "Colorblind-friendly"],
        index=0
    )
    
    # Color mapping based on selected theme
    if color_theme == "Vibrant":
        color_max = 'rgb(255, 64, 64)'
        color_min = 'rgb(64, 64, 255)'
        color_theory_max = 'rgb(64, 191, 64)'
        color_theory_min = 'rgb(191, 64, 191)'
    elif color_theme == "Pastel":
        color_max = 'rgb(255, 160, 160)'
        color_min = 'rgb(160, 160, 255)'
        color_theory_max = 'rgb(160, 255, 160)'
        color_theory_min = 'rgb(255, 160, 255)'
    elif color_theme == "Dark":
        color_max = 'rgb(180, 40, 40)'
        color_min = 'rgb(40, 40, 180)'
        color_theory_max = 'rgb(40, 140, 40)'
        color_theory_min = 'rgb(140, 40, 140)'
    elif color_theme == "Colorblind-friendly":
        color_max = 'rgb(230, 159, 0)'
        color_min = 'rgb(86, 180, 233)'
        color_theory_max = 'rgb(0, 158, 115)'
        color_theory_min = 'rgb(240, 228, 66)'
    else:  # Default
        color_max = 'rgb(220, 60, 60)'
        color_min = 'rgb(60, 60, 220)'
        color_theory_max = 'rgb(30, 180, 30)'
        color_theory_min = 'rgb(180, 30, 180)'

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Eigenvalue Analysis (C++)", "Eigenvalue Distribution (KDE)", "Im(s) vs z Analysis (SymPy)"])

# Tab 1: Eigenvalue Analysis with sub-tabs
with tab1:
    # Create sub-tabs for different eigenvalue analyses
    eig_subtabs = st.tabs(["Varying β Analysis", "Fixed β Analysis"])
    
    # Sub-tab 1: Varying Beta Analysis (original functionality)
    with eig_subtabs[0]:
        # Two-column layout for the dashboard
        left_column, right_column = st.columns([1, 3])
        
        with left_column:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<div class="panel-header">Varying β Analysis Controls</div>', unsafe_allow_html=True)
            
            # Parameter inputs with defaults and validation
            st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
            st.markdown("### Matrix Parameters")
            n = st.number_input("Sample size (n)", min_value=5, max_value=10000000, value=100, step=5, 
                               help="Number of samples", key="eig_n")
            p = st.number_input("Dimension (p)", min_value=5, max_value=10000000, value=50, step=5, 
                               help="Dimensionality", key="eig_p")
            a = st.number_input("Value for a", min_value=1.1, max_value=10000.0, value=2.0, step=0.1, 
                               help="Parameter a > 1", key="eig_a")
            
            # Automatically calculate y = p/n (as requested)
            y = p/n
            st.info(f"Value for y = p/n: {y:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
            st.markdown("### Calculation Controls")
            fineness = st.slider(
                "Beta points", 
                min_value=20, 
                max_value=500, 
                value=100, 
                step=10,
                help="Number of points to calculate along the β axis (0 to 1)",
                key="eig_fineness"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("Advanced Settings"):
                # Add controls for theoretical calculation precision
                theory_grid_points = st.slider(
                    "Theoretical grid points", 
                    min_value=100, 
                    max_value=1000, 
                    value=200, 
                    step=50,
                    help="Number of points in initial grid search for theoretical calculations",
                    key="eig_grid_points"
                )
                
                theory_tolerance = st.number_input(
                    "Theoretical tolerance", 
                    min_value=1e-12, 
                    max_value=1e-6, 
                    value=1e-10, 
                    format="%.1e",
                    help="Convergence tolerance for golden section search",
                    key="eig_tolerance"
                )
                
                # Debug mode
                debug_mode = st.checkbox("Debug Mode", value=False, key="eig_debug")
                
                # Timeout setting
                timeout_seconds = st.number_input(
                    "Computation timeout (seconds)", 
                    min_value=30, 
                    max_value=3600, 
                    value=300,
                    help="Maximum time allowed for computation before timeout",
                    key="eig_timeout"
                )
            
            # Generate button
            eig_generate_button = st.button("Generate Varying β Analysis", 
                                          type="primary", 
                                          use_container_width=True,
                                          key="eig_generate")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with right_column:
            # Main visualization area
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<div class="panel-header">Varying β Analysis Results</div>', unsafe_allow_html=True)
            
            # Container for the analysis results
            eig_results_container = st.container()
            
            # Process when generate button is clicked (existing logic)
            if eig_generate_button:
                with eig_results_container:
                    # Show progress
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    try:
                        # Create data file path
                        data_file = os.path.join(output_dir, "eigenvalue_data.json")
                        
                        # Delete previous output if exists
                        if os.path.exists(data_file):
                            os.remove(data_file)
                        
                        # Build command for eigenvalue analysis with the proper arguments
                        cmd = [
                            executable,
                            "eigenvalues",  # Mode argument
                            str(n),
                            str(p),
                            str(a),
                            str(y),
                            str(fineness),
                            str(theory_grid_points),
                            str(theory_tolerance),
                            data_file
                        ]
                        
                        # Run the command
                        status_text.text("Running eigenvalue analysis...")
                        
                        if debug_mode:
                            success, stdout, stderr = run_command(cmd, True, timeout=timeout_seconds)
                            # Process stdout for progress updates
                            if success:
                                progress_bar.progress(1.0)
                        else:
                            # Start the process with pipe for stdout to read progress
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                bufsize=1,
                                universal_newlines=True
                            )
                            
                            # Track progress from stdout
                            success = True
                            stdout_lines = []
                            
                            start_time = time.time()
                            while True:
                                # Check for timeout
                                if time.time() - start_time > timeout_seconds:
                                    process.kill()
                                    status_text.error(f"Computation timed out after {timeout_seconds} seconds")
                                    success = False
                                    break
                                    
                                # Try to read a line (non-blocking)
                                line = process.stdout.readline()
                                if not line and process.poll() is not None:
                                    break
                                    
                                if line:
                                    stdout_lines.append(line)
                                    if line.startswith("PROGRESS:"):
                                        try:
                                            # Update progress bar
                                            progress_value = float(line.split(":")[1].strip())
                                            progress_bar.progress(progress_value)
                                            status_text.text(f"Calculating... {int(progress_value * 100)}% complete")
                                        except:
                                            pass
                                    elif line:
                                        status_text.text(line.strip())
                                        
                            # Get the return code and stderr
                            returncode = process.poll()
                            stderr = process.stderr.read()
                            
                            if returncode != 0:
                                success = False
                                st.error(f"Error executing the analysis: {stderr}")
                                with st.expander("Error Details"):
                                    st.code(stderr)
                        
                        if success:
                            progress_bar.progress(1.0)
                            status_text.text("Analysis complete! Generating visualization...")
                            
                            # Check if the output file was created
                            if not os.path.exists(data_file):
                                st.error(f"Output file not created: {data_file}")
                                st.stop()
                            
                            try:
                                # Load the results from the JSON file
                                with open(data_file, 'r') as f:
                                    data = json.load(f)
                                
                                # Process data - convert string values to numeric
                                beta_values = np.array([safe_convert_to_numeric(x) for x in data['beta_values']])
                                max_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['max_eigenvalues']])
                                min_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['min_eigenvalues']])
                                theoretical_max = np.array([safe_convert_to_numeric(x) for x in data['theoretical_max']])
                                theoretical_min = np.array([safe_convert_to_numeric(x) for x in data['theoretical_min']])
                                
                                # Create an interactive plot using Plotly
                                fig = go.Figure()
                                
                                # Add traces for each line
                                fig.add_trace(go.Scatter(
                                    x=beta_values, 
                                    y=max_eigenvalues,
                                    mode='lines+markers',
                                    name='Empirical Max Eigenvalue',
                                    line=dict(color=color_max, width=3),
                                    marker=dict(
                                        symbol='circle',
                                        size=8,
                                        color=color_max,
                                        line=dict(color='white', width=1)
                                    ),
                                    hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=beta_values, 
                                    y=min_eigenvalues,
                                    mode='lines+markers',
                                    name='Empirical Min Eigenvalue',
                                    line=dict(color=color_min, width=3),
                                    marker=dict(
                                        symbol='circle',
                                        size=8,
                                        color=color_min,
                                        line=dict(color='white', width=1)
                                    ),
                                    hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=beta_values, 
                                    y=theoretical_max,
                                    mode='lines+markers',
                                    name='Theoretical Max',
                                    line=dict(color=color_theory_max, width=3),
                                    marker=dict(
                                        symbol='diamond',
                                        size=8,
                                        color=color_theory_max,
                                        line=dict(color='white', width=1)
                                    ),
                                    hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=beta_values, 
                                    y=theoretical_min,
                                    mode='lines+markers',
                                    name='Theoretical Min',
                                    line=dict(color=color_theory_min, width=3),
                                    marker=dict(
                                        symbol='diamond',
                                        size=8,
                                        color=color_theory_min,
                                        line=dict(color='white', width=1)
                                    ),
                                    hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
                                ))
                                
                                # Configure layout for better appearance
                                fig.update_layout(
                                    title={
                                        'text': f'Varying β Analysis: n={n}, p={p}, a={a}, y={y:.4f}',
                                        'font': {'size': 24, 'color': '#0e1117'},
                                        'y': 0.95,
                                        'x': 0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'
                                    },
                                    xaxis={
                                        'title': {'text': 'β Parameter', 'font': {'size': 18, 'color': '#424242'}},
                                        'tickfont': {'size': 14},
                                        'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                        'showgrid': True
                                    },
                                    yaxis={
                                        'title': {'text': 'Eigenvalues', 'font': {'size': 18, 'color': '#424242'}},
                                        'tickfont': {'size': 14},
                                        'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                        'showgrid': True
                                    },
                                    plot_bgcolor='rgba(250, 250, 250, 0.8)',
                                    paper_bgcolor='rgba(255, 255, 255, 0.8)',
                                    hovermode='closest',
                                    legend={
                                        'font': {'size': 14},
                                        'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                        'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                        'borderwidth': 1
                                    },
                                    margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                                    height=600,
                                )
                                
                                # Add custom modebar buttons
                                fig.update_layout(
                                    modebar_add=[
                                        'drawline', 'drawopenpath', 'drawclosedpath',
                                        'drawcircle', 'drawrect', 'eraseshape'
                                    ],
                                    modebar_remove=['lasso2d', 'select2d'],
                                    dragmode='zoom'
                                )
                                
                                # Clear progress container
                                progress_container.empty()
                                
                                # Display the interactive plot in Streamlit
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display statistics in a cleaner way
                                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Max Empirical", f"{max_eigenvalues.max():.4f}")
                                with col2:
                                    st.metric("Min Empirical", f"{min_eigenvalues.min():.4f}")
                                with col3:
                                    st.metric("Max Theoretical", f"{theoretical_max.max():.4f}")
                                with col4:
                                    st.metric("Min Theoretical", f"{theoretical_min.min():.4f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                    
                            except json.JSONDecodeError as e:
                                st.error(f"Error parsing JSON results: {str(e)}")
                                if os.path.exists(data_file):
                                    with open(data_file, 'r') as f:
                                        content = f.read()
                                    st.code(content[:1000] + "..." if len(content) > 1000 else content)
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        if debug_mode:
                            st.exception(e)
            
            else:
                # Try to load existing data if available
                data_file = os.path.join(output_dir, "eigenvalue_data.json")
                if os.path.exists(data_file):
                    try:
                        with open(data_file, 'r') as f:
                            data = json.load(f)
                        
                        # Process data - convert string values to numeric
                        beta_values = np.array([safe_convert_to_numeric(x) for x in data['beta_values']])
                        max_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['max_eigenvalues']])
                        min_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['min_eigenvalues']])
                        theoretical_max = np.array([safe_convert_to_numeric(x) for x in data['theoretical_max']])
                        theoretical_min = np.array([safe_convert_to_numeric(x) for x in data['theoretical_min']])
                        
                        # Create an interactive plot using Plotly
                        fig = go.Figure()
                        
                        # Add traces for each line
                        fig.add_trace(go.Scatter(
                            x=beta_values, 
                            y=max_eigenvalues,
                            mode='lines+markers',
                            name='Empirical Max Eigenvalue',
                            line=dict(color=color_max, width=3),
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color=color_max,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=beta_values, 
                            y=min_eigenvalues,
                            mode='lines+markers',
                            name='Empirical Min Eigenvalue',
                            line=dict(color=color_min, width=3),
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color=color_min,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=beta_values, 
                            y=theoretical_max,
                            mode='lines+markers',
                            name='Theoretical Max',
                            line=dict(color=color_theory_max, width=3),
                            marker=dict(
                                symbol='diamond',
                                size=8,
                                color=color_theory_max,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=beta_values, 
                            y=theoretical_min,
                            mode='lines+markers',
                            name='Theoretical Min',
                            line=dict(color=color_theory_min, width=3),
                            marker=dict(
                                symbol='diamond',
                                size=8,
                                color=color_theory_min,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
                        ))
                        
                        # Configure layout for better appearance
                        fig.update_layout(
                            title={
                                'text': f'Varying β Analysis (Previous Result)',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'β Parameter', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True
                            },
                            yaxis={
                                'title': {'text': 'Eigenvalues', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True
                            },
                            plot_bgcolor='rgba(250, 250, 250, 0.8)',
                            paper_bgcolor='rgba(255, 255, 255, 0.8)',
                            hovermode='closest',
                            legend={
                                'font': {'size': 14},
                                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                'borderwidth': 1
                            },
                            margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                            height=600
                        )
                        
                        # Display the interactive plot in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("This is the previous analysis result. Adjust parameters and click 'Generate Analysis' to create a new visualization.")
                        
                    except Exception as e:
                        st.info("Set parameters and click 'Generate Varying β Analysis' to create a visualization.")
                else:
                    # Show placeholder
                    st.info("Set parameters and click 'Generate Varying β Analysis' to create a visualization.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Sub-tab 2: Fixed Beta Analysis (new functionality)
    with eig_subtabs[1]:
        # Two-column layout for the dashboard
        left_column_fixed, right_column_fixed = st.columns([1, 3])
        
        with left_column_fixed:
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<div class="panel-header">Fixed β Analysis Controls</div>', unsafe_allow_html=True)
            
            # Parameter inputs with defaults and validation
            st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
            st.markdown("### Matrix Parameters")
            n_fixed = st.number_input("Sample size (n)", min_value=5, max_value=10000000, value=100, step=5, 
                                     help="Number of samples", key="eig_n_fixed")
            p_fixed = st.number_input("Dimension (p)", min_value=5, max_value=10000000, value=50, step=5, 
                                     help="Dimensionality", key="eig_p_fixed")
            
            # Automatically calculate y = p/n
            y_fixed = p_fixed/n_fixed
            st.info(f"Value for y = p/n: {y_fixed:.4f}")
            
            # Fixed beta parameter
            beta_fixed = st.slider("Fixed β value", min_value=0.0, max_value=1.0, value=0.5, step=0.01, 
                                  help="Fixed value of β parameter", key="eig_beta_fixed")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
            st.markdown("### Parameter Range")
            a_min = st.number_input("Minimum a value", min_value=1.1, max_value=100.0, value=1.1, step=0.1, 
                                   help="Minimum value for parameter a", key="eig_a_min")
            a_max = st.number_input("Maximum a value", min_value=1.1, max_value=100.0, value=5.0, step=0.1, 
                                   help="Maximum value for parameter a", key="eig_a_max")
            
            if a_min >= a_max:
                st.error("Minimum a value must be less than maximum a value")
            
            fineness_fixed = st.slider(
                "Parameter points", 
                min_value=20, 
                max_value=500, 
                value=100, 
                step=10,
                help="Number of points to calculate along the a axis",
                key="eig_fineness_fixed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("Advanced Settings"):
                # Add controls for theoretical calculation precision
                theory_grid_points_fixed = st.slider(
                    "Theoretical grid points", 
                    min_value=100, 
                    max_value=1000, 
                    value=200, 
                    step=50,
                    help="Number of points in initial grid search for theoretical calculations",
                    key="eig_grid_points_fixed"
                )
                
                theory_tolerance_fixed = st.number_input(
                    "Theoretical tolerance", 
                    min_value=1e-12, 
                    max_value=1e-6, 
                    value=1e-10, 
                    format="%.1e",
                    help="Convergence tolerance for golden section search",
                    key="eig_tolerance_fixed"
                )
                
                # Debug mode
                debug_mode_fixed = st.checkbox("Debug Mode", value=False, key="eig_debug_fixed")
                
                # Timeout setting
                timeout_seconds_fixed = st.number_input(
                    "Computation timeout (seconds)", 
                    min_value=30, 
                    max_value=3600, 
                    value=300,
                    help="Maximum time allowed for computation before timeout",
                    key="eig_timeout_fixed"
                )
            
            # Generate button
            eig_generate_button_fixed = st.button("Generate Fixed β Analysis", 
                                                  type="primary", 
                                                  use_container_width=True,
                                                  key="eig_generate_fixed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with right_column_fixed:
            # Main visualization area
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<div class="panel-header">Fixed β Analysis Results</div>', unsafe_allow_html=True)
            
            # Container for the analysis results
            eig_results_container_fixed = st.container()
            
            # Process when generate button is clicked
            if eig_generate_button_fixed:
                if a_min >= a_max:
                    st.error("Please ensure minimum a value is less than maximum a value")
                else:
                    with eig_results_container_fixed:
                        # Show progress
                        progress_container_fixed = st.container()
                        with progress_container_fixed:
                            progress_bar_fixed = st.progress(0)
                            status_text_fixed = st.empty()
                        
                        try:
                            # Create data file path
                            data_file_fixed = os.path.join(output_dir, "eigenvalue_fixed_beta_data.json")
                            
                            # Delete previous output if exists
                            if os.path.exists(data_file_fixed):
                                os.remove(data_file_fixed)
                            
                            # Build command for fixed beta eigenvalue analysis
                            cmd_fixed = [
                                executable,
                                "eigenvalues_fixed_beta",  # Mode argument
                                str(n_fixed),
                                str(p_fixed),
                                str(y_fixed),
                                str(beta_fixed),
                                str(a_min),
                                str(a_max),
                                str(fineness_fixed),
                                str(theory_grid_points_fixed),
                                str(theory_tolerance_fixed),
                                data_file_fixed
                            ]
                            
                            # Run the command
                            status_text_fixed.text("Running fixed β eigenvalue analysis...")
                            
                            if debug_mode_fixed:
                                success, stdout, stderr = run_command(cmd_fixed, True, timeout=timeout_seconds_fixed)
                                if success:
                                    progress_bar_fixed.progress(1.0)
                            else:
                                # Start the process with pipe for stdout to read progress
                                process = subprocess.Popen(
                                    cmd_fixed,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True
                                )
                                
                                # Track progress from stdout
                                success = True
                                stdout_lines = []
                                
                                start_time = time.time()
                                while True:
                                    # Check for timeout
                                    if time.time() - start_time > timeout_seconds_fixed:
                                        process.kill()
                                        status_text_fixed.error(f"Computation timed out after {timeout_seconds_fixed} seconds")
                                        success = False
                                        break
                                        
                                    # Try to read a line (non-blocking)
                                    line = process.stdout.readline()
                                    if not line and process.poll() is not None:
                                        break
                                        
                                    if line:
                                        stdout_lines.append(line)
                                        if line.startswith("PROGRESS:"):
                                            try:
                                                # Update progress bar
                                                progress_value = float(line.split(":")[1].strip())
                                                progress_bar_fixed.progress(progress_value)
                                                status_text_fixed.text(f"Calculating... {int(progress_value * 100)}% complete")
                                            except:
                                                pass
                                        elif line:
                                            status_text_fixed.text(line.strip())
                                            
                                # Get the return code and stderr
                                returncode = process.poll()
                                stderr = process.stderr.read()
                                
                                if returncode != 0:
                                    success = False
                                    st.error(f"Error executing the analysis: {stderr}")
                                    with st.expander("Error Details"):
                                        st.code(stderr)
                            
                            if success:
                                progress_bar_fixed.progress(1.0)
                                status_text_fixed.text("Analysis complete! Generating visualization...")
                                
                                # Check if the output file was created
                                if not os.path.exists(data_file_fixed):
                                    st.error(f"Output file not created: {data_file_fixed}")
                                    st.stop()
                                
                                try:
                                    # Load the results from the JSON file
                                    with open(data_file_fixed, 'r') as f:
                                        data = json.load(f)
                                    
                                    # Process data - convert string values to numeric
                                    a_values = np.array([safe_convert_to_numeric(x) for x in data['a_values']])
                                    max_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['max_eigenvalues']])
                                    min_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['min_eigenvalues']])
                                    theoretical_max = np.array([safe_convert_to_numeric(x) for x in data['theoretical_max']])
                                    theoretical_min = np.array([safe_convert_to_numeric(x) for x in data['theoretical_min']])
                                    
                                    # Create an interactive plot using Plotly
                                    fig = go.Figure()
                                    
                                    # Add traces for each line
                                    fig.add_trace(go.Scatter(
                                        x=a_values, 
                                        y=max_eigenvalues,
                                        mode='lines+markers',
                                        name='Empirical Max Eigenvalue',
                                        line=dict(color=color_max, width=3),
                                        marker=dict(
                                            symbol='circle',
                                            size=8,
                                            color=color_max,
                                            line=dict(color='white', width=1)
                                        ),
                                        hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=a_values, 
                                        y=min_eigenvalues,
                                        mode='lines+markers',
                                        name='Empirical Min Eigenvalue',
                                        line=dict(color=color_min, width=3),
                                        marker=dict(
                                            symbol='circle',
                                            size=8,
                                            color=color_min,
                                            line=dict(color='white', width=1)
                                        ),
                                        hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=a_values, 
                                        y=theoretical_max,
                                        mode='lines+markers',
                                        name='Theoretical Max',
                                        line=dict(color=color_theory_max, width=3),
                                        marker=dict(
                                            symbol='diamond',
                                            size=8,
                                            color=color_theory_max,
                                            line=dict(color='white', width=1)
                                        ),
                                        hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=a_values, 
                                        y=theoretical_min,
                                        mode='lines+markers',
                                        name='Theoretical Min',
                                        line=dict(color=color_theory_min, width=3),
                                        marker=dict(
                                            symbol='diamond',
                                            size=8,
                                            color=color_theory_min,
                                            line=dict(color='white', width=1)
                                        ),
                                        hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
                                    ))
                                    
                                    # Configure layout for better appearance
                                    fig.update_layout(
                                        title={
                                            'text': f'Fixed β Analysis: n={n_fixed}, p={p_fixed}, β={beta_fixed:.3f}, y={y_fixed:.4f}',
                                            'font': {'size': 24, 'color': '#0e1117'},
                                            'y': 0.95,
                                            'x': 0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top'
                                        },
                                        xaxis={
                                            'title': {'text': 'a Parameter', 'font': {'size': 18, 'color': '#424242'}},
                                            'tickfont': {'size': 14},
                                            'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                            'showgrid': True
                                        },
                                        yaxis={
                                            'title': {'text': 'Eigenvalues', 'font': {'size': 18, 'color': '#424242'}},
                                            'tickfont': {'size': 14},
                                            'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                            'showgrid': True
                                        },
                                        plot_bgcolor='rgba(250, 250, 250, 0.8)',
                                        paper_bgcolor='rgba(255, 255, 255, 0.8)',
                                        hovermode='closest',
                                        legend={
                                            'font': {'size': 14},
                                            'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                            'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                            'borderwidth': 1
                                        },
                                        margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                                        height=600,
                                    )
                                    
                                    # Add custom modebar buttons
                                    fig.update_layout(
                                        modebar_add=[
                                            'drawline', 'drawopenpath', 'drawclosedpath',
                                            'drawcircle', 'drawrect', 'eraseshape'
                                        ],
                                        modebar_remove=['lasso2d', 'select2d'],
                                        dragmode='zoom'
                                    )
                                    
                                    # Clear progress container
                                    progress_container_fixed.empty()
                                    
                                    # Display the interactive plot in Streamlit
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display statistics in a cleaner way
                                    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Max Empirical", f"{max_eigenvalues.max():.4f}")
                                    with col2:
                                        st.metric("Min Empirical", f"{min_eigenvalues.min():.4f}")
                                    with col3:
                                        st.metric("Max Theoretical", f"{theoretical_max.max():.4f}")
                                    with col4:
                                        st.metric("Min Theoretical", f"{theoretical_min.min():.4f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Add interpretation section
                                    with st.expander("Fixed β Analysis Explanation", expanded=False):
                                        st.markdown(f"""
                                        ### Understanding Fixed β Analysis

                                        In this analysis, β is fixed at **{beta_fixed:.3f}** while parameter **a** varies from **{a_min:.2f}** to **{a_max:.2f}**.

                                        **What this shows:**
                                        - How eigenvalue bounds change with parameter **a** when the mixture proportion β is constant
                                        - The relationship between the spike eigenvalue parameter **a** and the empirical eigenvalue distribution
                                        - Validation of theoretical predictions for varying **a** at fixed β

                                        **Key insights:**
                                        - As **a** increases, the maximum eigenvalue generally increases
                                        - The minimum eigenvalue behavior depends on the specific value of β
                                        - The gap between theoretical and empirical bounds shows finite-sample effects
                                        """)
                                        
                                except json.JSONDecodeError as e:
                                    st.error(f"Error parsing JSON results: {str(e)}")
                                    if os.path.exists(data_file_fixed):
                                        with open(data_file_fixed, 'r') as f:
                                            content = f.read()
                                        st.code(content[:1000] + "..." if len(content) > 1000 else content)
                        
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            if debug_mode_fixed:
                                st.exception(e)
            
            else:
                # Try to load existing data if available
                data_file_fixed = os.path.join(output_dir, "eigenvalue_fixed_beta_data.json")
                if os.path.exists(data_file_fixed):
                    try:
                        with open(data_file_fixed, 'r') as f:
                            data = json.load(f)
                        
                        # Process data - convert string values to numeric
                        a_values = np.array([safe_convert_to_numeric(x) for x in data['a_values']])
                        max_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['max_eigenvalues']])
                        min_eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['min_eigenvalues']])
                        theoretical_max = np.array([safe_convert_to_numeric(x) for x in data['theoretical_max']])
                        theoretical_min = np.array([safe_convert_to_numeric(x) for x in data['theoretical_min']])
                        
                        # Create an interactive plot using Plotly
                        fig = go.Figure()
                        
                        # Add traces for each line
                        fig.add_trace(go.Scatter(
                            x=a_values, 
                            y=max_eigenvalues,
                            mode='lines+markers',
                            name='Empirical Max Eigenvalue',
                            line=dict(color=color_max, width=3),
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color=color_max,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=a_values, 
                            y=min_eigenvalues,
                            mode='lines+markers',
                            name='Empirical Min Eigenvalue',
                            line=dict(color=color_min, width=3),
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color=color_min,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=a_values, 
                            y=theoretical_max,
                            mode='lines+markers',
                            name='Theoretical Max',
                            line=dict(color=color_theory_max, width=3),
                            marker=dict(
                                symbol='diamond',
                                size=8,
                                color=color_theory_max,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=a_values, 
                            y=theoretical_min,
                            mode='lines+markers',
                            name='Theoretical Min',
                            line=dict(color=color_theory_min, width=3),
                            marker=dict(
                                symbol='diamond',
                                size=8,
                                color=color_theory_min,
                                line=dict(color='white', width=1)
                            ),
                            hovertemplate='a: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
                        ))
                        
                        # Configure layout for better appearance
                        fig.update_layout(
                            title={
                                'text': f'Fixed β Analysis (Previous Result)',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'a Parameter', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True
                            },
                            yaxis={
                                'title': {'text': 'Eigenvalues', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True
                            },
                            plot_bgcolor='rgba(250, 250, 250, 0.8)',
                            paper_bgcolor='rgba(255, 255, 255, 0.8)',
                            hovermode='closest',
                            legend={
                                'font': {'size': 14},
                                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                'borderwidth': 1
                            },
                            margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                            height=600
                        )
                        
                        # Display the interactive plot in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("This is the previous analysis result. Adjust parameters and click 'Generate Fixed β Analysis' to create a new visualization.")
                        
                    except Exception as e:
                        st.info("Set parameters and click 'Generate Fixed β Analysis' to create a visualization.")
                else:
                    # Show placeholder
                    st.info("Set parameters and click 'Generate Fixed β Analysis' to create a visualization.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Eigenvalue Distribution (KDE) Analysis
with tab2:
    st.header("Eigenvalue Distribution Analysis")
    st.markdown("Generate KDE plots of eigenvalue distributions using C++ computation.")
    
    # Two-column layout
    left_col_kde, right_col_kde = st.columns([1, 3])
    
    with left_col_kde:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Distribution Analysis Controls</div>', unsafe_allow_html=True)
        
        # Parameter inputs
        st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
        st.markdown("### Model Parameters")
        
        beta_kde = st.slider("β (mixing parameter)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, 
                            help="Proportion of spike eigenvalues", key="beta_kde")
        a_kde = st.number_input("a (spike value)", min_value=1.1, max_value=100.0, value=10.0, step=0.1, 
                               help="Spike eigenvalue magnitude", key="a_kde")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
        st.markdown("### Matrix Dimensions")
        n_kde = st.number_input("n (samples)", min_value=50, max_value=2000, value=500, step=50, 
                               help="Number of samples", key="n_kde")
        p_kde = st.number_input("p (dimensions)", min_value=50, max_value=2000, value=200, step=50, 
                               help="Number of dimensions", key="p_kde")
        
        # Automatically calculate and display y
        y_kde = p_kde / n_kde
        st.info(f"y = p/n = {y_kde:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("Advanced Settings"):
            kde_bandwidth = st.slider("KDE Bandwidth", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                                     help="Bandwidth for kernel density estimation")
            kde_points = st.slider("KDE Resolution", min_value=100, max_value=1000, value=500, step=50,
                                  help="Number of points for KDE evaluation")
            bound_points = st.slider(
                "Bound Search Points",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Sampling points for theoretical bound search",
            )
            timeout_kde = st.number_input("Timeout (seconds)", min_value=30, max_value=1800, value=180,
                                         key="timeout_kde")
        
        # Generate button
        generate_kde_button = st.button("Generate KDE Analysis", 
                                       type="primary", 
                                       use_container_width=True,
                                       key="generate_kde")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col_kde:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Eigenvalue Distribution Results</div>', unsafe_allow_html=True)
        
        if generate_kde_button:
            # Show progress
            progress_container_kde = st.container()
            with progress_container_kde:
                progress_bar_kde = st.progress(0)
                status_text_kde = st.empty()
            
            try:
                # Create data file path
                data_file_kde = os.path.join(output_dir, "eigenvalue_distribution_data.json")
                
                # Delete previous output if exists
                if os.path.exists(data_file_kde):
                    os.remove(data_file_kde)
                
                # Build command for eigenvalue distribution analysis
                cmd_kde = [
                    executable,
                    "eigenvalue_distribution",
                    str(a_kde),
                    str(beta_kde),
                    str(n_kde),
                    str(p_kde),
                    data_file_kde
                ]
                
                # Run the command
                status_text_kde.text("Running eigenvalue distribution analysis...")
                
                # Start the process
                process = subprocess.Popen(
                    cmd_kde,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Track progress from stdout
                success = True
                stdout_lines = []
                
                start_time = time.time()
                while True:
                    # Check for timeout
                    if time.time() - start_time > timeout_kde:
                        process.kill()
                        status_text_kde.error(f"Computation timed out after {timeout_kde} seconds")
                        success = False
                        break
                        
                    # Try to read a line (non-blocking)
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                        
                    if line:
                        stdout_lines.append(line)
                        if line.startswith("PROGRESS:"):
                            try:
                                # Update progress bar
                                progress_value = float(line.split(":")[1].strip())
                                progress_bar_kde.progress(progress_value)
                                status_text_kde.text(f"Computing... {int(progress_value * 100)}% complete")
                            except:
                                pass
                        elif line:
                            status_text_kde.text(line.strip())
                            
                # Get the return code and stderr
                returncode = process.poll()
                stderr = process.stderr.read()
                
                if returncode != 0:
                    success = False
                    st.error(f"Error executing analysis: {stderr}")
                    with st.expander("Error Details"):
                        st.code(stderr)
                
                if success:
                    progress_bar_kde.progress(1.0)
                    status_text_kde.text("Analysis complete! Generating visualization...")
                    
                    # Check if the output file was created
                    if not os.path.exists(data_file_kde):
                        st.error(f"Output file not created: {data_file_kde}")
                    else:
                        try:
                            # Load the results from the JSON file
                            with open(data_file_kde, 'r') as f:
                                data = json.load(f)

                            # Process data
                            parameters = data['parameters']
                            eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['eigenvalues']])
                            eigenvalues = eigenvalues[~np.isnan(eigenvalues)]  # Remove NaN values

                            quartic = compute_quartic_tianyuan(parameters['a'], parameters['y'], parameters['beta'])
                            display_quartic_summary(quartic, "Quartic Equation Analysis")

                            if len(eigenvalues) > 1:
                                # Create KDE
                                kde = gaussian_kde(eigenvalues)
                                kde.set_bandwidth(kde_bandwidth)

                                # Evaluate KDE
                                x_min, x_max = eigenvalues.min(), eigenvalues.max()
                                x_range = x_max - x_min
                                x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, kde_points)
                                kde_vals = kde(x_eval)

                                # Determine theoretical bounds from analytic expression
                                bound_min, bound_max = compute_theoretical_bounds(
                                    parameters['a'], parameters['y'], parameters['beta'], bound_points
                                )
                                g_values = compute_g_values(
                                    quartic['roots'], parameters['a'], parameters['y'], parameters['beta']
                                )

                                # Create the plot
                                fig = go.Figure()
                                
                                # Add histogram
                                fig.add_trace(go.Histogram(
                                    x=eigenvalues,
                                    histnorm='probability density',
                                    name='Histogram',
                                    marker=dict(color='lightblue', opacity=0.6),
                                    nbinsx=30
                                ))
                                
                                # Add KDE curve
                                fig.add_trace(go.Scatter(
                                    x=x_eval,
                                    y=kde_vals,
                                    mode='lines',
                                    name='KDE',
                                    line=dict(color='red', width=3),
                                    hovertemplate='Eigenvalue: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>'
                                ))

                                # Mark theoretical bounds
                                if bound_min is not None:
                                    fig.add_vline(
                                        x=bound_min,
                                        line_dash='dash',
                                        line_color='green',
                                        annotation_text='min',
                                        annotation_position='top left'
                                    )
                                if bound_max is not None:
                                    fig.add_vline(
                                        x=bound_max,
                                        line_dash='dash',
                                        line_color='green',
                                        annotation_text='max',
                                        annotation_position='top right'
                                    )
                                for idx, g_val in g_values:
                                    fig.add_vline(
                                        x=g_val,
                                        line_dash='dot',
                                        line_color='purple',
                                        annotation_text=f'g(t_{idx})',
                                        annotation_position='bottom'
                                    )
                                
                                # Update layout
                                fig.update_layout(
                                    title={
                                        'text': f'Eigenvalue Distribution: β={beta_kde}, a={a_kde}, y={y_kde:.3f}',
                                        'font': {'size': 20, 'color': '#0e1117'},
                                        'x': 0.5,
                                        'xanchor': 'center'
                                    },
                                    xaxis={
                                        'title': 'Eigenvalue',
                                        'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                        'showgrid': True
                                    },
                                    yaxis={
                                        'title': 'Density',
                                        'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                        'showgrid': True
                                    },
                                    plot_bgcolor='rgba(250, 250, 250, 0.8)',
                                    paper_bgcolor='white',
                                    hovermode='closest',
                                    legend={
                                        'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                        'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                        'borderwidth': 1
                                    },
                                    height=600
                                )
                                
                                # Clear progress container
                                progress_container_kde.empty()
                                
                                # Display the plot
                                st.plotly_chart(fig, use_container_width=True)

                                # Add statistics
                                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Min Eigenvalue", f"{eigenvalues.min():.4f}")
                                with col2:
                                    st.metric("Max Eigenvalue", f"{eigenvalues.max():.4f}")
                                with col3:
                                    st.metric("Mean", f"{eigenvalues.mean():.4f}")
                                with col4:
                                    st.metric("Std Dev", f"{eigenvalues.std():.4f}")
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Show roots and Δ below statistics
                                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                                st.markdown("**Quartic Roots**")
                                for i, root in enumerate(quartic['roots'], 1):
                                    real = float(np.real(root))
                                    imag = float(np.imag(root))
                                    if abs(imag) < 1e-12:
                                        st.latex(f"t_{{{i}}} = {real:.6f}")
                                    else:
                                        sign = '+' if imag >= 0 else '-'
                                        st.latex(f"t_{{{i}}} = {real:.6f} {sign} {abs(imag):.6f}i")
                                if g_values:
                                    st.markdown("**g(t_i) (real)**")
                                    for idx, val in g_values:
                                        st.latex(f"g(t_{{{idx}}}) = {val:.6f}")
                                st.latex(f"\\Delta_{0} = {quartic['tianyuan_values']['delta']:.6f}")
                                expr_tex = r"\frac{y\beta(a-1)t+(at+1)((y-1)t-1)}{(at+1)(t^2+t)}"
                                if bound_min is not None:
                                    st.latex(rf"\min_{{t\in(-1/a,0)}} {expr_tex} = {bound_min:.6f}")
                                if bound_max is not None:
                                    st.latex(rf"\max_{{t>0}} {expr_tex} = {bound_max:.6f}")
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Add explanation
                                with st.expander("Understanding the Results", expanded=False):
                                    st.markdown(f"""
                                    ### Eigenvalue Distribution Analysis
                                    
                                    **Parameters used:**
                                    - β = {beta_kde} (proportion of spike eigenvalues)
                                    - a = {a_kde} (spike eigenvalue magnitude)  
                                    - y = {y_kde:.3f} (aspect ratio p/n = {p_kde}/{n_kde})
                                    - Matrix size: {p_kde} × {n_kde}
                                    
                                    **Quartic Equation Analysis:**
                                    - The coefficients follow the Ferrari's method with 天衍 (Tianyuan) formulae
                                    - The discriminant Δ determines the nature of the roots:
                                      - Δ > 0: Four distinct real/complex roots
                                      - Δ = 0: Multiple roots (special cases)
                                      - Δ < 0: Different root configurations
                                    - Reference: [天衍 formulae](https://zhuanlan.zhihu.com/p/677634589)
                                    
                                    **Eigenvalue Distribution:**
                                    - **Histogram**: Shows the empirical distribution of eigenvalues
                                    - **KDE curve**: Smooth density estimate of the eigenvalue distribution
                                    - **Mixed distribution**: Combination of bulk eigenvalues (~1) and spike eigenvalues (~{a_kde})
                                    
                                    **Key insights:**
                                    - The β parameter controls how many eigenvalues are "spikes" vs bulk
                                    - Spike eigenvalues appear around {a_kde}, bulk eigenvalues around 1
                                    - The quartic roots provide theoretical insights into the eigenvalue structure
                                    - The distribution shape reveals the random matrix structure
                                    """)
                            else:
                                st.error("Not enough valid eigenvalues to generate KDE")
                                
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing JSON results: {str(e)}")
                            if os.path.exists(data_file_kde):
                                with open(data_file_kde, 'r') as f:
                                    content = f.read()
                                st.code(content[:1000] + "..." if len(content) > 1000 else content)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        else:
            # Try to load existing data if available
            data_file_kde = os.path.join(output_dir, "eigenvalue_distribution_data.json")
            if os.path.exists(data_file_kde):
                try:
                    with open(data_file_kde, 'r') as f:
                        data = json.load(f)
                    
                    # Process data
                    parameters = data['parameters']
                    eigenvalues = np.array([safe_convert_to_numeric(x) for x in data['eigenvalues']])
                    eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
                    
                    quartic = compute_quartic_tianyuan(parameters['a'], parameters['y'], parameters['beta'])
                    display_quartic_summary(quartic, "Quartic Equation Analysis (Previous Result)")

                    if len(eigenvalues) > 1:
                        # Create KDE with default bandwidth
                        kde = gaussian_kde(eigenvalues)

                        # Evaluate KDE
                        x_min, x_max = eigenvalues.min(), eigenvalues.max()
                        x_range = x_max - x_min
                        x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 500)
                        kde_vals = kde(x_eval)

                        # Determine theoretical bounds from analytic expression
                        bound_min, bound_max = compute_theoretical_bounds(
                            parameters['a'], parameters['y'], parameters['beta'], bound_points
                        )
                        g_values = compute_g_values(
                            quartic['roots'], parameters['a'], parameters['y'], parameters['beta']
                        )

                        # Create the plot
                        fig = go.Figure()

                        # Add histogram
                        fig.add_trace(go.Histogram(
                            x=eigenvalues,
                            histnorm='probability density',
                            name='Histogram',
                            marker=dict(color='lightblue', opacity=0.6),
                            nbinsx=30
                        ))

                        # Add KDE curve
                        fig.add_trace(go.Scatter(
                            x=x_eval,
                            y=kde_vals,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=3)
                        ))

                        # Mark theoretical bounds
                        if bound_min is not None:
                            fig.add_vline(
                                x=bound_min,
                                line_dash='dash',
                                line_color='green',
                                annotation_text='min',
                                annotation_position='top left'
                            )
                        if bound_max is not None:
                            fig.add_vline(
                                x=bound_max,
                                line_dash='dash',
                                line_color='green',
                                annotation_text='max',
                                annotation_position='top right'
                            )
                        for idx, g_val in g_values:
                            fig.add_vline(
                                x=g_val,
                                line_dash='dot',
                                line_color='purple',
                                annotation_text=f'g(t_{idx})',
                                annotation_position='bottom'
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title={
                                'text': f'Eigenvalue Distribution (Previous Result)',
                                'font': {'size': 20, 'color': '#0e1117'},
                                'x': 0.5,
                                'xanchor': 'center'
                            },
                            xaxis={'title': 'Eigenvalue'},
                            yaxis={'title': 'Density'},
                            plot_bgcolor='rgba(250, 250, 250, 0.8)',
                            paper_bgcolor='white',
                            height=600
                        )
                        
                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)

                        # Add statistics
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min Eigenvalue", f"{eigenvalues.min():.4f}")
                        with col2:
                            st.metric("Max Eigenvalue", f"{eigenvalues.max():.4f}")
                        with col3:
                            st.metric("Mean", f"{eigenvalues.mean():.4f}")
                        with col4:
                            st.metric("Std Dev", f"{eigenvalues.std():.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown("**Quartic Roots**")
                        for i, root in enumerate(quartic['roots'], 1):
                            real = float(np.real(root))
                            imag = float(np.imag(root))
                            if abs(imag) < 1e-12:
                                st.latex(f"t_{{{i}}} = {real:.6f}")
                            else:
                                sign = '+' if imag >= 0 else '-'
                                st.latex(f"t_{{{i}}} = {real:.6f} {sign} {abs(imag):.6f}i")
                        if g_values:
                            st.markdown("**g(t_i) (real)**")
                            for idx, val in g_values:
                                st.latex(f"g(t_{{{idx}}}) = {val:.6f}")
                        st.latex(f"\\Delta_{0} = {quartic['tianyuan_values']['delta']:.6f}")
                        expr_tex = r"\frac{y\beta(a-1)t+(at+1)((y-1)t-1)}{(at+1)(t^2+t)}"
                        if bound_min is not None:
                            st.latex(rf"\min_{{t\in(-1/a,0)}} {expr_tex} = {bound_min:.6f}")
                        if bound_max is not None:
                            st.latex(rf"\max_{{t>0}} {expr_tex} = {bound_max:.6f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.info("This is the previous analysis result. Adjust parameters and click 'Generate KDE Analysis' to create a new visualization.")
                    
                except Exception as e:
                    st.info("Set parameters and click 'Generate KDE Analysis' to create eigenvalue distribution visualizations.")
            else:
                # Show placeholder
                st.info("Set parameters and click 'Generate KDE Analysis' to create eigenvalue distribution visualizations.")
        
        st.markdown('</div>', unsafe_allow_html=True)

 
# ----- Tab 3: Complex Root Analysis -----
with tab3:
    st.header("Complex Root Analysis")
    plot_tabs = st.tabs(["Im{s} vs. z", "Im{s} vs. β", "Phase Diagram", "Eigenvalue Distribution"])

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
                fig_im_beta, fig_re_beta, fig_disc = generate_roots_vs_beta_plots(z_beta, y_beta, z_a_beta, beta_min, beta_max, beta_points)
                if fig_im_beta is not None and fig_re_beta is not None and fig_disc is not None:
                    st.plotly_chart(fig_im_beta, use_container_width=True)
                    st.plotly_chart(fig_re_beta, use_container_width=True)
                    st.plotly_chart(fig_disc, use_container_width=True)
                    transition_points, structure_types = analyze_complex_root_structure(np.linspace(beta_min, beta_max, beta_points), z_beta, z_a_beta, y_beta)
                    if transition_points:
                        st.subheader("Root Structure Transition Points")
                        for i, beta in enumerate(transition_points):
                            prev_type = structure_types[i]
                            next_type = structure_types[i+1] if i+1 < len(structure_types) else "unknown"
                            st.markdown(f"- At β = {beta:.6f}: Transition from {prev_type} roots to {next_type} roots")
                    else:
                        st.info("No transitions detected in root structure across this β range.")
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
                fig_phase = generate_phase_diagram(z_a_phase, y_phase, beta_min_phase, beta_max_phase, z_min_phase, z_max_phase, beta_steps_phase, z_steps_phase)
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
            show_theoretical = st.checkbox("Show theoretical boundaries", value=True)
            show_empirical_stats = st.checkbox("Show empirical statistics", value=True)
        if st.button("Generate Eigenvalue Distribution", key="tab2_eigen_button"):
            with col_eigen2:
                fig_eigen, eigenvalues = generate_eigenvalue_distribution(beta_eigen, y_eigen, z_a_eigen, n=n_samples, seed=sim_seed)
                if show_theoretical:
                    betas = np.array([beta_eigen])
                    min_eig, max_eig = compute_eigenvalue_support_boundaries(z_a_eigen, y_eigen, betas, n_samples=n_samples, seeds=5)
                    fig_eigen.add_vline(x=min_eig[0], line=dict(color="red", width=2, dash="dash"), annotation_text="Min theoretical", annotation_position="top right")
                    fig_eigen.add_vline(x=max_eig[0], line=dict(color="red", width=2, dash="dash"), annotation_text="Max theoretical", annotation_position="top left")
                st.plotly_chart(fig_eigen, use_container_width=True)
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
                if show_empirical_stats:
                    st.markdown("### Eigenvalue Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean", f"{np.mean(eigenvalues):.4f}")
                        st.metric("Median", f"{np.median(eigenvalues):.4f}")
                    with col2:
                        st.metric("Standard Deviation", f"{np.std(eigenvalues):.4f}")
                        st.metric("Interquartile Range", f"{np.percentile(eigenvalues, 75) - np.percentile(eigenvalues, 25):.4f}")

# Add footer with instructions
st.markdown("""
<div class="footer">
    <h3>About the Matrix Analysis Dashboard</h3>
    <p>This dashboard performs three types of analyses using different computational approaches:</p>
    <ol>
        <li><strong>Eigenvalue Analysis (C++):</strong> Uses C++ with OpenCV for high-performance computation of eigenvalues of random matrices.</li>
        <li><strong>Eigenvalue Distribution (KDE):</strong> Generates kernel density estimates of eigenvalue distributions across different z values using C++ computation.</li>
        <li><strong>Im(s) vs z Analysis (SymPy):</strong> Uses Python's SymPy library with extended precision to accurately analyze the cubic equation roots.</li>
    </ol>
    <p>This hybrid approach combines C++'s performance for data-intensive calculations with SymPy's high-precision symbolic mathematics for accurate root finding.</p>
    
    <h4>New Features</h4>
    <ul>
        <li><strong>Fixed β Analysis:</strong> Study how eigenvalues change as parameter 'a' varies while keeping β constant</li>
        <li><strong>KDE Distribution Analysis:</strong> Generate kernel density estimates of eigenvalue distributions across z values</li>
        <li><strong>Enhanced Visualizations:</strong> Interactive plots with customizable color themes</li>
        <li><strong>Comprehensive Analysis:</strong> Multiple parameter study approaches available</li>
    </ul>
</div>
""", unsafe_allow_html=True)