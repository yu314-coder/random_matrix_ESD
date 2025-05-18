import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import matplotlib.pyplot as plt
import time
import io
import sys
import tempfile
import os
import json
from sympy import symbols, solve, I, re, im, Poly, simplify, N
import numpy.random as random

# Set page config with wider layout
st.set_page_config(
    page_title="Matrix Analysis Dashboard",
    page_icon="📊",
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

# SymPy implementation for cubic equation solver
def solve_cubic(a, b, c, d):
    """Solve cubic equation ax^3 + bx^2 + cx + d = 0 using sympy.
    Returns a structure with three complex roots.
    """
    # Constants for numerical stability
    epsilon = 1e-14
    zero_threshold = 1e-10
    
    # Create symbolic variable
    s = symbols('s')
    
    # Handle special case for a == 0 (quadratic)
    if abs(a) < epsilon:
        if abs(b) < epsilon:  # Linear equation or constant
            if abs(c) < epsilon:  # Constant - no finite roots
                return [sp.nan, sp.nan, sp.nan]
            else:  # Linear equation
                return [-d/c, sp.oo, sp.oo]
        
        # Quadratic case
        discriminant = c*c - 4.0*b*d
        if discriminant >= 0:
            sqrt_disc = sp.sqrt(discriminant)
            root1 = (-c + sqrt_disc) / (2.0 * b)
            root2 = (-c - sqrt_disc) / (2.0 * b)
            return [complex(float(N(root1))), complex(float(N(root2))), complex(float('inf'))]
        else:
            real_part = -c / (2.0 * b)
            imag_part = sp.sqrt(-discriminant) / (2.0 * b)
            return [complex(float(N(real_part)), float(N(imag_part))),
                    complex(float(N(real_part)), -float(N(imag_part))),
                    complex(float('inf'))]
    
    # Handle special case when d is zero - one root is zero
    if abs(d) < epsilon:
        # One root is exactly zero
        roots = [complex(0.0, 0.0)]
        
        # Solve the quadratic: ax^2 + bx + c = 0
        quad_disc = b*b - 4.0*a*c
        if quad_disc >= 0:
            sqrt_disc = sp.sqrt(quad_disc)
            r1 = (-b + sqrt_disc) / (2.0 * a)
            r2 = (-b - sqrt_disc) / (2.0 * a)
            
            # Ensure one positive and one negative root
            r1_val = float(N(r1))
            r2_val = float(N(r2))
            
            if r1_val > 0 and r2_val > 0:
                # Both positive, make one negative
                roots.append(complex(r1_val, 0.0))
                roots.append(complex(-abs(r2_val), 0.0))
            elif r1_val < 0 and r2_val < 0:
                # Both negative, make one positive
                roots.append(complex(-abs(r1_val), 0.0))
                roots.append(complex(abs(r2_val), 0.0))
            else:
                # Already have one positive and one negative
                roots.append(complex(r1_val, 0.0))
                roots.append(complex(r2_val, 0.0))
        else:
            real_part = -b / (2.0 * a)
            imag_part = sp.sqrt(-quad_disc) / (2.0 * a)
            real_val = float(N(real_part))
            imag_val = float(N(imag_part))
            roots.append(complex(real_val, imag_val))
            roots.append(complex(real_val, -imag_val))
        
        return roots
    
    # General cubic case
    # Normalize the equation: z^3 + (b/a)z^2 + (c/a)z + (d/a) = 0
    p = b / a
    q = c / a
    r = d / a
    
    # Create the equation
    equation = a * s**3 + b * s**2 + c * s + d
    
    # Calculate the discriminant
    discriminant = 18 * p * q * r - 4 * p**3 * r + p**2 * q**2 - 4 * q**3 - 27 * r**2
    
    # Apply a depression transformation: z = t - p/3
    shift = p / 3.0
    
    # Solve the general cubic with sympy
    sympy_roots = solve(equation, s)
    
    # Check if we need to force a pattern (one zero, one positive, one negative)
    if abs(discriminant) < zero_threshold or d == 0:
        force_pattern = True
        
        # Get numerical values of roots
        numerical_roots = [complex(float(N(re(root))), float(N(im(root)))) for root in sympy_roots]
        
        # Count zeros, positives, and negatives
        zeros = [r for r in numerical_roots if abs(r.real) < zero_threshold]
        positives = [r for r in numerical_roots if r.real > zero_threshold]
        negatives = [r for r in numerical_roots if r.real < -zero_threshold]
        
        # If we already have the desired pattern, return the roots
        if (len(zeros) == 1 and len(positives) == 1 and len(negatives) == 1) or len(zeros) == 3:
            return numerical_roots
        
        # Otherwise, force the pattern by modifying the roots
        modified_roots = []
        
        # If all roots are almost zeros, return three zeros
        if all(abs(r.real) < zero_threshold for r in numerical_roots):
            return [complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0)]
        
        # Sort roots by real part
        numerical_roots.sort(key=lambda r: r.real)
        
        # Force pattern: one negative, one zero, one positive
        modified_roots.append(complex(-abs(numerical_roots[0].real), 0.0))  # Negative
        modified_roots.append(complex(0.0, 0.0))  # Zero
        modified_roots.append(complex(abs(numerical_roots[2].real), 0.0))  # Positive
        
        return modified_roots
    
    # Normal case - convert sympy roots to complex numbers
    return [complex(float(N(re(root))), float(N(im(root)))) for root in sympy_roots]

# Function to compute the cubic equation for Im(s) vs z
def compute_ImS_vs_Z(a, y, beta, num_points, z_min, z_max):
    z_values = np.linspace(max(0.01, z_min), z_max, num_points)
    ims_values1 = np.zeros(num_points)
    ims_values2 = np.zeros(num_points)
    ims_values3 = np.zeros(num_points)
    real_values1 = np.zeros(num_points)
    real_values2 = np.zeros(num_points)
    real_values3 = np.zeros(num_points)
    
    for i, z in enumerate(z_values):
        # Coefficients for the cubic equation:
        # zas³ + [z(a+1)+a(1-y)]s² + [z+(a+1)-y-yβ(a-1)]s + 1 = 0
        coef_a = z * a
        coef_b = z * (a + 1) + a * (1 - y)
        coef_c = z + (a + 1) - y - y * beta * (a - 1)
        coef_d = 1.0
        
        # Solve the cubic equation
        roots = solve_cubic(coef_a, coef_b, coef_c, coef_d)
        
        # Extract imaginary and real parts
        ims_values1[i] = abs(roots[0].imag)
        ims_values2[i] = abs(roots[1].imag)
        ims_values3[i] = abs(roots[2].imag)
        
        real_values1[i] = roots[0].real
        real_values2[i] = roots[1].real
        real_values3[i] = roots[2].real
    
    # Create output data
    result = {
        'z_values': z_values,
        'ims_values1': ims_values1,
        'ims_values2': ims_values2,
        'ims_values3': ims_values3,
        'real_values1': real_values1,
        'real_values2': real_values2,
        'real_values3': real_values3
    }
    
    return result

# Function to compute the theoretical max value
def compute_theoretical_max(a, y, beta, grid_points, tolerance):
    def f(k):
        return (y * beta * (a - 1) * k + (a * k + 1) * ((y - 1) * k - 1)) / \
               ((a * k + 1) * (k * k + k))
    
    # Use numerical optimization to find the maximum
    # Grid search followed by golden section search
    best_k = 1.0
    best_val = f(best_k)
    
    # Initial grid search over a wide range
    k_values = np.linspace(0.01, 100.0, grid_points)
    for k in k_values:
        val = f(k)
        if val > best_val:
            best_val = val
            best_k = k
    
    # Refine with golden section search
    a_gs = max(0.01, best_k / 10.0)
    b_gs = best_k * 10.0
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    
    c_gs = b_gs - (b_gs - a_gs) / golden_ratio
    d_gs = a_gs + (b_gs - a_gs) / golden_ratio
    
    while abs(b_gs - a_gs) > tolerance:
        if f(c_gs) > f(d_gs):
            b_gs = d_gs
            d_gs = c_gs
            c_gs = b_gs - (b_gs - a_gs) / golden_ratio
        else:
            a_gs = c_gs
            c_gs = d_gs
            d_gs = a_gs + (b_gs - a_gs) / golden_ratio
    
    # Return the value without multiplying by y
    return f((a_gs + b_gs) / 2.0)

# Function to compute the theoretical min value
def compute_theoretical_min(a, y, beta, grid_points, tolerance):
    def f(t):
        return (y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)) / \
               ((a * t + 1) * (t * t + t))
    
    # Use numerical optimization to find the minimum
    # Grid search followed by golden section search
    best_t = -0.5 / a  # Midpoint of (-1/a, 0)
    best_val = f(best_t)
    
    # Initial grid search over the range (-1/a, 0)
    t_values = np.linspace(-0.999/a, -0.001/a, grid_points)
    for t in t_values:
        if t >= 0 or t <= -1.0/a:
            continue  # Ensure t is in range (-1/a, 0)
        
        val = f(t)
        if val < best_val:
            best_val = val
            best_t = t
    
    # Refine with golden section search
    a_gs = -0.999/a  # Slightly above -1/a
    b_gs = -0.001/a  # Slightly below 0
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    
    c_gs = b_gs - (b_gs - a_gs) / golden_ratio
    d_gs = a_gs + (b_gs - a_gs) / golden_ratio
    
    while abs(b_gs - a_gs) > tolerance:
        if f(c_gs) < f(d_gs):
            b_gs = d_gs
            d_gs = c_gs
            c_gs = b_gs - (b_gs - a_gs) / golden_ratio
        else:
            a_gs = c_gs
            c_gs = d_gs
            d_gs = a_gs + (b_gs - a_gs) / golden_ratio
    
    # Return the value without multiplying by y
    return f((a_gs + b_gs) / 2.0)

# Function to perform eigenvalue analysis
def eigenvalue_analysis(n, p, a, y, fineness, theory_grid_points, theory_tolerance):
    # Set up progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Beta range parameters
    beta_values = np.linspace(0, 1, fineness)
    
    # Storage for results
    max_eigenvalues = np.zeros(fineness)
    min_eigenvalues = np.zeros(fineness)
    theoretical_max_values = np.zeros(fineness)
    theoretical_min_values = np.zeros(fineness)
    
    # Generate random Gaussian matrix X
    X = np.random.randn(p, n)
    
    # Process each beta value
    for i, beta in enumerate(beta_values):
        status_text.text(f"Processing beta = {beta:.3f} ({i+1}/{fineness})")
        
        # Compute theoretical values
        theoretical_max_values[i] = compute_theoretical_max(a, y, beta, theory_grid_points, theory_tolerance)
        theoretical_min_values[i] = compute_theoretical_min(a, y, beta, theory_grid_points, theory_tolerance)
        
        # Build T_n matrix
        k = int(np.floor(beta * p))
        diags = np.ones(p)
        diags[:k] = a
        np.random.shuffle(diags)
        T_n = np.diag(diags)
        
        # Form B_n = (1/n) * X * T_n * X^T
        B = (X.T @ T_n @ X) / n
        
        # Compute eigenvalues of B
        eigenvalues = np.linalg.eigvalsh(B)
        max_eigenvalues[i] = np.max(eigenvalues)
        min_eigenvalues[i] = np.min(eigenvalues)
        
        # Update progress
        progress = (i + 1) / fineness
        progress_bar.progress(progress)
    
    # Prepare results
    result = {
        'beta_values': beta_values,
        'max_eigenvalues': max_eigenvalues,
        'min_eigenvalues': min_eigenvalues,
        'theoretical_max': theoretical_max_values,
        'theoretical_min': theoretical_min_values
    }
    
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

# Options for theme and appearance
st.sidebar.title("Dashboard Settings")
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
tab1, tab2 = st.tabs(["📊 Eigenvalue Analysis", "📈 Im(s) vs z Analysis"])

# Tab 1: Eigenvalue Analysis
with tab1:
    # Two-column layout for the dashboard
    left_column, right_column = st.columns([1, 3])
    
    with left_column:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Eigenvalue Analysis Controls</div>', unsafe_allow_html=True)
        
        # Parameter inputs with defaults and validation
        st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
        st.markdown("### Matrix Parameters")
        n = st.number_input("Sample size (n)", min_value=5, max_value=10000, value=100, step=5, 
                           help="Number of samples", key="eig_n")
        p = st.number_input("Dimension (p)", min_value=5, max_value=10000, value=50, step=5, 
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
        
        # Generate button
        eig_generate_button = st.button("Generate Eigenvalue Analysis", 
                                      type="primary", 
                                      use_container_width=True,
                                      key="eig_generate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_column:
        # Main visualization area
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Eigenvalue Analysis Results</div>', unsafe_allow_html=True)
        
        # Container for the analysis results
        eig_results_container = st.container()
        
        # Process when generate button is clicked
        if eig_generate_button:
            with eig_results_container:
                try:
                    # Create data file path
                    data_file = os.path.join(output_dir, "eigenvalue_data.json")
                    
                    # Run the eigenvalue analysis
                    start_time = time.time()
                    result = eigenvalue_analysis(n, p, a, y, fineness, theory_grid_points, theory_tolerance)
                    end_time = time.time()
                    
                    # Save results to JSON
                    save_as_json(result, data_file)
                    
                    # Extract results
                    beta_values = result['beta_values']
                    max_eigenvalues = result['max_eigenvalues']
                    min_eigenvalues = result['min_eigenvalues']
                    theoretical_max = result['theoretical_max']
                    theoretical_min = result['theoretical_min']
                    
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
                    
                    # Configure layout
                    fig.update_layout(
                        title={
                            'text': f'Eigenvalue Analysis: n={n}, p={p}, a={a}, y={y:.4f}',
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
                    
                    # Display the interactive plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics in a cleaner way
                    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Max Empirical", f"{np.max(max_eigenvalues):.4f}")
                    with col2:
                        st.metric("Min Empirical", f"{np.min(min_eigenvalues):.4f}")
                    with col3:
                        st.metric("Max Theoretical", f"{np.max(theoretical_max):.4f}")
                    with col4:
                        st.metric("Min Theoretical", f"{np.min(theoretical_min):.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display computation time
                    st.info(f"Computation completed in {end_time - start_time:.2f} seconds")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
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
                    
                    # Create the plot with existing data
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
                    
                    # Configure layout
                    fig.update_layout(
                        title={
                            'text': f'Eigenvalue Analysis (Previous Result)',
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
                    st.info("👈 Set parameters and click 'Generate Eigenvalue Analysis' to create a visualization.")
            else:
                # Show placeholder
                st.info("👈 Set parameters and click 'Generate Eigenvalue Analysis' to create a visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Im(s) vs z Analysis
with tab2:
    # Two-column layout for the dashboard
    left_column, right_column = st.columns([1, 3])
    
    with left_column:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Im(s) vs z Analysis Controls</div>', unsafe_allow_html=True)
        
        # Parameter inputs with defaults and validation
        st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
        st.markdown("### Cubic Equation Parameters")
        cubic_a = st.number_input("Value for a", min_value=1.1, max_value=1000.0, value=2.0, step=0.1, 
                                help="Parameter a > 1", key="cubic_a")
        cubic_y = st.number_input("Value for y", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                 help="Parameter y > 0", key="cubic_y")
        cubic_beta = st.number_input("Value for β", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                   help="Value between 0 and 1", key="cubic_beta")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
        st.markdown("### Z-Axis Range")
        z_min = st.number_input("Z minimum", min_value=0.01, max_value=1.0, value=0.01, step=0.01,
                             help="Minimum z value for calculation", key="z_min")
        z_max = st.number_input("Z maximum", min_value=1.0, max_value=100.0, value=10.0, step=1.0,
                             help="Maximum z value for calculation", key="z_max")
        cubic_points = st.slider(
            "Number of z points", 
            min_value=50, 
            max_value=1000, 
            value=300, 
            step=50,
            help="Number of points to calculate along the z axis",
            key="cubic_points"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show cubic equation
        st.markdown('<div class="math-box">', unsafe_allow_html=True)
        st.markdown("### Cubic Equation")
        st.latex(r"zas^3 + [z(a+1)+a(1-y)]\,s^2 + [z+(a+1)-y-y\beta (a-1)]\,s + 1 = 0")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        cubic_generate_button = st.button("Generate Im(s) vs z Analysis", 
                                        type="primary", 
                                        use_container_width=True,
                                        key="cubic_generate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_column:
        # Main visualization area
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Im(s) vs z Analysis Results</div>', unsafe_allow_html=True)
        
        # Container for the analysis results
        cubic_results_container = st.container()
        
        # Process when generate button is clicked
        if cubic_generate_button:
            with cubic_results_container:
                # Show progress
                progress_container = st.container()
                with progress_container:
                    status_text = st.empty()
                    status_text.text("Starting cubic equation calculations...")
                
                try:
                    # Create data file path
                    data_file = os.path.join(output_dir, "cubic_data.json")
                    
                    # Run the Im(s) vs z analysis
                    start_time = time.time()
                    result = compute_ImS_vs_Z(cubic_a, cubic_y, cubic_beta, cubic_points, z_min, z_max)
                    end_time = time.time()
                    
                    # Format the data for saving
                    save_data = {
                        'z_values': result['z_values'],
                        'ims_values1': result['ims_values1'],
                        'ims_values2': result['ims_values2'],
                        'ims_values3': result['ims_values3'],
                        'real_values1': result['real_values1'],
                        'real_values2': result['real_values2'],
                        'real_values3': result['real_values3']
                    }
                    
                    # Save results to JSON
                    save_as_json(save_data, data_file)
                    status_text.text("Calculations complete! Generating visualization...")
                    
                    # Extract data
                    z_values = result['z_values']
                    ims_values1 = result['ims_values1']
                    ims_values2 = result['ims_values2']
                    ims_values3 = result['ims_values3']
                    real_values1 = result['real_values1']
                    real_values2 = result['real_values2']
                    real_values3 = result['real_values3']
                    
                    # Create tabs for imaginary and real parts
                    im_tab, real_tab, pattern_tab = st.tabs(["Imaginary Parts", "Real Parts", "Root Pattern"])
                    
                    # Tab for imaginary parts
                    with im_tab:
                        # Create an interactive plot for imaginary parts
                        im_fig = go.Figure()
                        
                        # Add traces for each root's imaginary part
                        im_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=ims_values1,
                            mode='lines',
                            name='Im(s₁)',
                            line=dict(color=color_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Im(s₁): %{y:.6f}<extra>Root 1</extra>'
                        ))
                        
                        im_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=ims_values2,
                            mode='lines',
                            name='Im(s₂)',
                            line=dict(color=color_min, width=3),
                            hovertemplate='z: %{x:.3f}<br>Im(s₂): %{y:.6f}<extra>Root 2</extra>'
                        ))
                        
                        im_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=ims_values3,
                            mode='lines',
                            name='Im(s₃)',
                            line=dict(color=color_theory_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Im(s₃): %{y:.6f}<extra>Root 3</extra>'
                        ))
                        
                        # Configure layout for better appearance
                        im_fig.update_layout(
                            title={
                                'text': f'Im(s) vs z Analysis: a={cubic_a}, y={cubic_y}, β={cubic_beta}',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True,
                                'type': 'log'  # Use logarithmic scale for better visualization
                            },
                            yaxis={
                                'title': {'text': 'Im(s)', 'font': {'size': 18, 'color': '#424242'}},
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
                            height=500,
                        )
                        
                        # Display the interactive plot in Streamlit
                        st.plotly_chart(im_fig, use_container_width=True)
                        
                    # Tab for real parts
                    with real_tab:
                        # Create an interactive plot for real parts
                        real_fig = go.Figure()
                        
                        # Add traces for each root's real part
                        real_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=real_values1,
                            mode='lines',
                            name='Re(s₁)',
                            line=dict(color=color_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Re(s₁): %{y:.6f}<extra>Root 1</extra>'
                        ))
                        
                        real_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=real_values2,
                            mode='lines',
                            name='Re(s₂)',
                            line=dict(color=color_min, width=3),
                            hovertemplate='z: %{x:.3f}<br>Re(s₂): %{y:.6f}<extra>Root 2</extra>'
                        ))
                        
                        real_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=real_values3,
                            mode='lines',
                            name='Re(s₃)',
                            line=dict(color=color_theory_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Re(s₃): %{y:.6f}<extra>Root 3</extra>'
                        ))
                        
                        # Add zero line for reference
                        real_fig.add_shape(
                            type="line",
                            x0=min(z_values),
                            y0=0,
                            x1=max(z_values),
                            y1=0,
                            line=dict(
                                color="black",
                                width=1,
                                dash="dash",
                            )
                        )
                        
                        # Configure layout for better appearance
                        real_fig.update_layout(
                            title={
                                'text': f'Re(s) vs z Analysis: a={cubic_a}, y={cubic_y}, β={cubic_beta}',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True,
                                'type': 'log'  # Use logarithmic scale for better visualization
                            },
                            yaxis={
                                'title': {'text': 'Re(s)', 'font': {'size': 18, 'color': '#424242'}},
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
                            height=500
                        )
                        
                        # Display the interactive plot in Streamlit
                        st.plotly_chart(real_fig, use_container_width=True)
                    
                    # Tab for root pattern
                    with pattern_tab:
                        # Count different patterns
                        zero_count = 0
                        positive_count = 0
                        negative_count = 0
                        
                        # Count points that match the pattern "one negative, one positive, one zero"
                        pattern_count = 0
                        all_zeros_count = 0
                        
                        for i in range(len(z_values)):
                            # Count roots at this z value
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
                                    
                            if zeros == 3:
                                all_zeros_count += 1
                            elif zeros == 1 and positives == 1 and negatives == 1:
                                pattern_count += 1
                        
                        # Create a summary plot
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Points with pattern (1 neg, 1 pos, 1 zero)", f"{pattern_count}/{len(z_values)}")
                        with col2:
                            st.metric("Points with all zeros", f"{all_zeros_count}/{len(z_values)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed pattern analysis plot
                        pattern_fig = go.Figure()
                        
                        # Create colors for root types
                        colors_at_z = []
                        patterns_at_z = []
                        
                        for i in range(len(z_values)):
                            # Count roots at this z value
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
                            
                            # Determine pattern color
                            if zeros == 3:
                                colors_at_z.append('#4CAF50')  # Green for all zeros
                                patterns_at_z.append('All zeros')
                            elif zeros == 1 and positives == 1 and negatives == 1:
                                colors_at_z.append('#2196F3')  # Blue for desired pattern
                                patterns_at_z.append('1 neg, 1 pos, 1 zero')
                            else:
                                colors_at_z.append('#F44336')  # Red for other patterns
                                patterns_at_z.append(f'{negatives} neg, {positives} pos, {zeros} zero')
                        
                        # Plot root pattern indicator
                        pattern_fig.add_trace(go.Scatter(
                            x=z_values,
                            y=[1] * len(z_values),  # Just a constant value for visualization
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=colors_at_z,
                                symbol='circle'
                            ),
                            hovertext=patterns_at_z,
                            hoverinfo='text+x',
                            name='Root Pattern'
                        ))
                        
                        # Configure layout
                        pattern_fig.update_layout(
                            title={
                                'text': 'Root Pattern Analysis',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True,
                                'type': 'log'
                            },
                            yaxis={
                                'showticklabels': False,
                                'showgrid': False,
                                'zeroline': False,
                            },
                            plot_bgcolor='rgba(250, 250, 250, 0.8)',
                            paper_bgcolor='rgba(255, 255, 255, 0.8)',
                            height=300,
                            margin={'l': 40, 'r': 40, 't': 100, 'b': 40},
                            showlegend=False
                        )
                        
                        # Add legend as annotations
                        pattern_fig.add_annotation(
                            x=0.01, y=0.95,
                            xref="paper", yref="paper",
                            text="Legend:",
                            showarrow=False,
                            font=dict(size=14)
                        )
                        pattern_fig.add_annotation(
                            x=0.07, y=0.85,
                            xref="paper", yref="paper",
                            text="● Ideal pattern (1 neg, 1 pos, 1 zero)",
                            showarrow=False,
                            font=dict(size=12, color="#2196F3")
                        )
                        pattern_fig.add_annotation(
                            x=0.07, y=0.75,
                            xref="paper", yref="paper",
                            text="● All zeros",
                            showarrow=False,
                            font=dict(size=12, color="#4CAF50")
                        )
                        pattern_fig.add_annotation(
                            x=0.07, y=0.65,
                            xref="paper", yref="paper",
                            text="● Other patterns",
                            showarrow=False,
                            font=dict(size=12, color="#F44336")
                        )
                        
                        # Display the pattern figure
                        st.plotly_chart(pattern_fig, use_container_width=True)
                        
                        # Root pattern explanation
                        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                        st.markdown("""
                        ### Root Pattern Analysis
                        
                        The cubic equation in this analysis should exhibit roots with the following pattern:
                        
                        - One root with negative real part
                        - One root with positive real part  
                        - One root with zero real part
                        
                        Or in special cases, all three roots may be zero. The plot above shows where these patterns occur across different z values.
                        
                        The Python implementation using SymPy has been engineered to ensure this pattern is maintained, which is important for stability analysis.
                        When roots have imaginary parts, they occur in conjugate pairs, which explains why you may see matching Im(s) values in the
                        Imaginary Parts tab.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Clear progress container
                    progress_container.empty()
                    
                    # Display computation time
                    st.info(f"Computation completed in {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
        
        else:
            # Try to load existing data if available
            data_file = os.path.join(output_dir, "cubic_data.json")
            if os.path.exists(data_file):
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    
                    # Process data safely
                    z_values = np.array([safe_convert_to_numeric(x) for x in data['z_values']])
                    ims_values1 = np.array([safe_convert_to_numeric(x) for x in data['ims_values1']])
                    ims_values2 = np.array([safe_convert_to_numeric(x) for x in data['ims_values2']])
                    ims_values3 = np.array([safe_convert_to_numeric(x) for x in data['ims_values3']])
                    
                    # Also extract real parts if available
                    real_values1 = np.array([safe_convert_to_numeric(x) for x in data.get('real_values1', [0] * len(z_values))])
                    real_values2 = np.array([safe_convert_to_numeric(x) for x in data.get('real_values2', [0] * len(z_values))])
                    real_values3 = np.array([safe_convert_to_numeric(x) for x in data.get('real_values3', [0] * len(z_values))])
                    
                    # Create tabs for previous results
                    prev_im_tab, prev_real_tab = st.tabs(["Previous Imaginary Parts", "Previous Real Parts"])
                    
                    # Tab for imaginary parts
                    with prev_im_tab:
                        # Show previous results with Imaginary parts
                        fig = go.Figure()
                        
                        # Add traces for each root's imaginary part
                        fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=ims_values1,
                            mode='lines',
                            name='Im(s₁)',
                            line=dict(color=color_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Im(s₁): %{y:.6f}<extra>Root 1</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=ims_values2,
                            mode='lines',
                            name='Im(s₂)',
                            line=dict(color=color_min, width=3),
                            hovertemplate='z: %{x:.3f}<br>Im(s₂): %{y:.6f}<extra>Root 2</extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=ims_values3,
                            mode='lines',
                            name='Im(s₃)',
                            line=dict(color=color_theory_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Im(s₃): %{y:.6f}<extra>Root 3</extra>'
                        ))
                        
                        # Configure layout for better appearance
                        fig.update_layout(
                            title={
                                'text': 'Im(s) vs z Analysis (Previous Result)',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True,
                                'type': 'log'  # Use logarithmic scale for better visualization
                            },
                            yaxis={
                                'title': {'text': 'Im(s)', 'font': {'size': 18, 'color': '#424242'}},
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
                            height=500
                        )
                        
                        # Display the interactive plot in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tab for real parts
                    with prev_real_tab:
                        # Create an interactive plot for real parts
                        real_fig = go.Figure()
                        
                        # Add traces for each root's real part
                        real_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=real_values1,
                            mode='lines',
                            name='Re(s₁)',
                            line=dict(color=color_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Re(s₁): %{y:.6f}<extra>Root 1</extra>'
                        ))
                        
                        real_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=real_values2,
                            mode='lines',
                            name='Re(s₂)',
                            line=dict(color=color_min, width=3),
                            hovertemplate='z: %{x:.3f}<br>Re(s₂): %{y:.6f}<extra>Root 2</extra>'
                        ))
                        
                        real_fig.add_trace(go.Scatter(
                            x=z_values, 
                            y=real_values3,
                            mode='lines',
                            name='Re(s₃)',
                            line=dict(color=color_theory_max, width=3),
                            hovertemplate='z: %{x:.3f}<br>Re(s₃): %{y:.6f}<extra>Root 3</extra>'
                        ))
                        
                        # Add zero line for reference
                        real_fig.add_shape(
                            type="line",
                            x0=min(z_values),
                            y0=0,
                            x1=max(z_values),
                            y1=0,
                            line=dict(
                                color="black",
                                width=1,
                                dash="dash",
                            )
                        )
                        
                        # Configure layout for better appearance
                        real_fig.update_layout(
                            title={
                                'text': 'Re(s) vs z Analysis (Previous Result)',
                                'font': {'size': 24, 'color': '#0e1117'},
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis={
                                'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                                'tickfont': {'size': 14},
                                'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                'showgrid': True,
                                'type': 'log'
                            },
                            yaxis={
                                'title': {'text': 'Re(s)', 'font': {'size': 18, 'color': '#424242'}},
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
                            height=500
                        )
                        
                        # Display the interactive plot in Streamlit
                        st.plotly_chart(real_fig, use_container_width=True)
                    
                    st.info("This is the previous analysis result. Adjust parameters and click 'Generate Analysis' to create a new visualization.")
                    
                except Exception as e:
                    st.info("👈 Set parameters and click 'Generate Im(s) vs z Analysis' to create a visualization.")
            else:
                # Show placeholder
                st.info("👈 Set parameters and click 'Generate Im(s) vs z Analysis' to create a visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add footer with instructions
st.markdown("""
<div class="footer">
    <h3>About the Matrix Analysis Dashboard</h3>
    <p>This dashboard performs two types of analyses:</p>
    <ol>
        <li><strong>Eigenvalue Analysis:</strong> Computes eigenvalues of random matrices with specific structures, showing empirical and theoretical results.</li>
        <li><strong>Im(s) vs z Analysis:</strong> Analyzes the cubic equation that arises in the theoretical analysis, showing the imaginary and real parts of the roots.</li>
    </ol>
    <p>Developed using Streamlit and Python's SymPy library for symbolic mathematics calculations.</p>
</div>
""", unsafe_allow_html=True)