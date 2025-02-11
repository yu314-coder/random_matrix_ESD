import sympy as sp
import numpy as np
import gradio as gr
import plotly.graph_objects as go

#############################
# 1) Define the discriminant
#############################

# Symbolic variables to build a symbolic expression of discriminant
z_sym, beta_sym, z_a_sym, y_sym = sp.symbols("z beta z_a y", real=True, positive=True)

# Define a, b, c, d in terms of z_sym, beta_sym, z_a_sym, y_sym
a_sym = z_sym * z_a_sym
b_sym = z_sym * z_a_sym + z_sym + z_a_sym
c_sym = z_sym + z_a_sym + 1 - y_sym*(beta_sym*z_a_sym + 1 - beta_sym)
d_sym = 1

# Symbolic expression for the standard cubic discriminant:
# Δ = ( (b c)/(6 a^2) - (b^3)/(27 a^3) - d/(2a) )^2 + ( c/(3a) - (b^2)/(9 a^2) )^3
Delta_expr = (
    ( (b_sym*c_sym)/(6*a_sym**2) - (b_sym**3)/(27*a_sym**3) - d_sym/(2*a_sym) )**2
    + ( c_sym/(3*a_sym) - (b_sym**2)/(9*a_sym**2) )**3
)

# Turn that into a fast numeric function:
discriminant_func = sp.lambdify((z_sym, beta_sym, z_a_sym, y_sym), Delta_expr, "numpy")

def find_z_at_discriminant_zero(z_a, y, beta, z_min, z_max, steps=20000):
    """
    Numerically scan z in [z_min, z_max] looking for sign changes of
    Delta(z) = 0. Returns all roots found via bisection.
    """
    z_grid = np.linspace(z_min, z_max, steps)
    disc_vals = discriminant_func(z_grid, beta, z_a, y)
    
    roots_found = []
    
    # Scan for sign changes
    for i in range(len(z_grid) - 1):
        f1, f2 = disc_vals[i], disc_vals[i+1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        
        if f1 == 0.0:
            roots_found.append(z_grid[i])
        elif f2 == 0.0:
            roots_found.append(z_grid[i+1])
        elif f1*f2 < 0:
            zl = z_grid[i]
            zr = z_grid[i+1]
            for _ in range(50):
                mid = 0.5*(zl + zr)
                fm = discriminant_func(mid, beta, z_a, y)
                if fm == 0:
                    zl = zr = mid
                    break
                if np.sign(fm) == np.sign(f1):
                    zl = mid
                    f1 = fm
                else:
                    zr = mid
                    f2 = fm
            root_approx = 0.5*(zl + zr)
            roots_found.append(root_approx)
    
    return np.array(roots_found)

def sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps=51):
    """
    For each beta, find both the largest and smallest z where discriminant=0.
    Returns (betas, z_min_values, z_max_values).
    """
    betas = np.linspace(0, 1, beta_steps)
    z_min_values = []
    z_max_values = []
    
    for b in betas:
        roots = find_z_at_discriminant_zero(z_a, y, b, z_min, z_max)
        if len(roots) == 0:
            z_min_values.append(np.nan)
            z_max_values.append(np.nan)
        else:
            z_min_values.append(np.min(roots))
            z_max_values.append(np.max(roots))
    
    return betas, np.array(z_min_values), np.array(z_max_values)

def compute_additional_curve(betas, z_a, y):
    """
    Compute the additional curve with proper handling of divide by zero cases
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        # Handle the sqrt first to avoid complex numbers
        sqrt_term = y * betas * (z_a - 1)
        sqrt_term = np.where(sqrt_term < 0, np.nan, np.sqrt(sqrt_term))
        
        term = (-1 + sqrt_term)/z_a
        numerator = (y - 2)*term + y * betas * ((z_a - 1)/z_a) - 1/z_a - 1
        denominator = term**2 + term
        
        # Handle division by zero and invalid values
        mask = (denominator == 0) | np.isnan(denominator) | np.isnan(numerator)
        result = np.zeros_like(denominator)
        result[~mask] = numerator[~mask] / denominator[~mask]
        result[mask] = np.nan
        
        return result

def generate_z_vs_beta_plot(z_a, y, z_min, z_max):
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        return "Invalid input parameters."
    
    beta_steps = 101
    
    # Sweep beta for both min and max discriminant curves:
    betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps=beta_steps)
    
    # Compute the additional curve with improved error handling
    new_curve = compute_additional_curve(betas, z_a, y)
    
    fig = go.Figure()
    
    # Blue trace: upper discriminant=0 boundary
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=z_maxs,
            mode="markers+lines",
            name="Upper z*(β)",
            marker=dict(size=5, color='blue'),
            line=dict(color='blue'),
        )
    )
    
    # Light blue trace: lower discriminant=0 boundary
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=z_mins,
            mode="markers+lines",
            name="Lower z*(β)",
            marker=dict(size=5, color='lightblue'),
            line=dict(color='lightblue'),
        )
    )
    
    # Red trace: the additional expression
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=new_curve,
            mode="markers+lines",
            name="Additional Expression",
            marker=dict(size=5, color='red'),
            line=dict(color='red'),
        )
    )
    
    fig.update_layout(
        title="Curves vs β: z*(β) boundaries (blue) and Additional Expression (red)",
        xaxis_title="β",
        yaxis_title="Value",
        hovermode="x unified",
    )
    return fig

def compute_cubic_roots(z, beta, z_a, y):
    """
    Compute the roots of the cubic equation for given parameters.
    Returns array of complex roots.
    """
    # Coefficients of the cubic equation
    a = z * z_a
    b = z * z_a + z + z_a
    c = z + z_a + 1 - y*(beta*z_a + 1 - beta)
    d = 1
    
    # Use numpy's roots function to find complex roots
    coeffs = [a, b, c, d]
    roots = np.roots(coeffs)
    return roots

def generate_ims_vs_z_plot(beta, y, z_a, z_min, z_max):
    """
    Generate plot of Im{s} vs. z for given parameters
    """
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        return "Invalid input parameters."
    
    # Create z grid
    z_points = np.linspace(z_min, z_max, 1000)
    
    # Arrays to store imaginary parts
    ims = []
    
    # Compute roots for each z
    for z in z_points:
        roots = compute_cubic_roots(z, beta, z_a, y)
        # Sort roots by imaginary part
        roots = sorted(roots, key=lambda x: abs(x.imag))
        # Store imaginary parts
        ims.append([root.imag for root in roots])
    
    ims = np.array(ims)
    
    # Create plot
    fig = go.Figure()
    
    # Plot imaginary parts of each root
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=z_points,
                y=ims[:,i],
                mode="lines",
                name=f"Im{{s{i+1}}}",
                line=dict(width=2),
            )
        )
    
    fig.update_layout(
        title=f"Im{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
        xaxis_title="z",
        yaxis_title="Im{s}",
        hovermode="x unified",
    )
    return fig

# Modified Gradio interface with tabs
with gr.Blocks() as app:
    with gr.Tabs():
        with gr.Tab("z*(β) Curves"):
            gr.Markdown("## Find z Values where Cubic Roots Transition Between Real and Complex")
            with gr.Row():
                with gr.Column():
                    z_a_input1 = gr.Number(label="z_a", value=1.0)
                    y_input1 = gr.Number(label="y", value=1.0)
                    z_min_input1 = gr.Number(label="z_min", value=-10)
                    z_max_input1 = gr.Number(label="z_max", value=10)
                    generate_button1 = gr.Button("Compute z vs. β Curves")
                with gr.Column():
                    plot_output1 = gr.Plot(label="z vs. β (Discriminant=0 boundaries & extra curve)")
            
            def on_generate_click(z_a, y, z_min, z_max):
                fig = generate_z_vs_beta_plot(z_a, y, z_min, z_max)
                return fig
            
            generate_button1.click(
                fn=on_generate_click,
                inputs=[z_a_input1, y_input1, z_min_input1, z_max_input1],
                outputs=plot_output1
            )
        
        with gr.Tab("Im{s} vs. z"):
            gr.Markdown("## Plot Imaginary Parts of Roots vs. z")
            with gr.Row():
                with gr.Column():
                    beta_input = gr.Number(label="β", value=0.5)
                    y_input2 = gr.Number(label="y", value=1.0)
                    z_a_input2 = gr.Number(label="z_a", value=1.0)
                    z_min_input2 = gr.Number(label="z_min", value=-10)
                    z_max_input2 = gr.Number(label="z_max", value=10)
                    generate_button2 = gr.Button("Compute Im{s} vs. z")
                with gr.Column():
                    plot_output2 = gr.Plot(label="Im{s} vs. z")
            
            def on_generate_ims_click(beta, y, z_a, z_min, z_max):
                fig = generate_ims_vs_z_plot(beta, y, z_a, z_min, z_max)
                return fig
            
            generate_button2.click(
                fn=on_generate_ims_click,
                inputs=[beta_input, y_input2, z_a_input2, z_min_input2, z_max_input2],
                outputs=plot_output2
            )

app.launch()
