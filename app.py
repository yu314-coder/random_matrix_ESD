import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# Configure Streamlit for Hugging Face Spaces
st.set_page_config(
   page_title="Cubic Root Analysis",
   layout="wide",
   initial_sidebar_state="expanded"
)

# Move custom expression inputs to sidebar
with st.sidebar:
    st.header("Custom Expression Settings")
    expression_type = st.radio(
        "Select Expression Type",
        ["Original Low y", "Alternative Low y"]
    )
    
    if expression_type == "Original Low y":
        default_num = "(y - 2)*((-1 + sqrt(y*beta*(z_a - 1)))/z_a) + y*beta*((z_a-1)/z_a) - 1/z_a - 1"
        default_denom = "((-1 + sqrt(y*beta*(z_a - 1)))/z_a)**2 + ((-1 + sqrt(y*beta*(z_a - 1)))/z_a)"
    else:
        default_num = "1*z_a*y*beta*(z_a-1) - 2*z_a*(1 - y) - 2*z_a**2"
        default_denom = "2+2*z_a"
    
    custom_num_expr = st.text_input("Numerator Expression", value=default_num)
    custom_denom_expr = st.text_input("Denominator Expression", value=default_denom)

#############################
# 1) Define the discriminant
#############################

# Symbolic variables to build a symbolic expression of discriminant
z_sym, beta_sym, z_a_sym, y_sym = sp.symbols("z beta z_a y", real=True, positive=True)

# Define a, b, c, d in terms of z_sym, beta_sym, z_a_sym, y_sym
a_sym = z_sym * z_a_sym
b_sym = z_sym * z_a_sym + z_sym + z_a_sym - z_a_sym*y_sym
c_sym = z_sym + z_a_sym + 1 - y_sym*(beta_sym*z_a_sym + 1 - beta_sym)
d_sym = 1

# Symbolic expression for the standard cubic discriminant
Delta_expr = (
   ((b_sym*c_sym)/(6*a_sym**2) - (b_sym**3)/(27*a_sym**3) - d_sym/(2*a_sym))**2
   + (c_sym/(3*a_sym) - (b_sym**2)/(9*a_sym**2))**3
)

# Turn that into a fast numeric function:
discriminant_func = sp.lambdify((z_sym, beta_sym, z_a_sym, y_sym), Delta_expr, "numpy")

@st.cache_data
def find_z_at_discriminant_zero(z_a, y, beta, z_min, z_max, steps):
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

@st.cache_data
def sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps):
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
def compute_low_y_curve(betas, z_a, y):
   betas = np.array(betas)
   with np.errstate(invalid='ignore', divide='ignore'):
       sqrt_term = y * betas * (z_a - 1)
       sqrt_term = np.where(sqrt_term < 0, np.nan, np.sqrt(sqrt_term))
       
       term = (-1 + sqrt_term)/z_a
       numerator = (y - 2)*term + y * betas * ((z_a - 1)/z_a) - 1/z_a - 1
       denominator = term**2 + term
       mask = (denominator != 0) & ~np.isnan(denominator) & ~np.isnan(numerator)
       return np.where(mask, numerator/denominator, np.nan)

@st.cache_data
def compute_high_y_curve(betas, z_a, y):
    a = z_a
    betas = np.array(betas)
    denominator = 1 - 2*a
    
    if denominator == 0:
        return np.full_like(betas, np.nan)
        
    numerator = -4*a*(a-1)*y*betas - 2*a*y - 2*a*(2*a-1)
    return numerator/denominator

@st.cache_data
def compute_z_difference_and_derivatives(z_a, y, z_min, z_max, beta_steps, z_steps):
    betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
    
    z_difference = z_maxs - z_mins
    dz_diff_dbeta = np.gradient(z_difference, betas)
    d2z_diff_dbeta2 = np.gradient(dz_diff_dbeta, betas)
    
    return betas, z_difference, dz_diff_dbeta, d2z_diff_dbeta2

def compute_custom_expression(betas, z_a, y, num_expr_str, denom_expr_str):
    beta_sym, z_a_sym, y_sym, a_sym = sp.symbols("beta z_a y a", positive=True)
    local_dict = {"beta": beta_sym, "z_a": z_a_sym, "y": y_sym, "a": z_a_sym}
    
    try:
        num_expr = sp.sympify(num_expr_str, locals=local_dict)
        denom_expr = sp.sympify(denom_expr_str, locals=local_dict)
    except sp.SympifyError as e:
        st.error(f"Error parsing expressions: {e}")
        return np.full_like(betas, np.nan)
    
    num_func = sp.lambdify((beta_sym, z_a_sym, y_sym), num_expr, modules=["numpy"])
    denom_func = sp.lambdify((beta_sym, z_a_sym, y_sym), denom_expr, modules=["numpy"])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num_func(betas, z_a, y) / denom_func(betas, z_a, y)
    return result

def generate_z_vs_beta_plot(z_a, y, z_min, z_max, beta_steps, z_steps,
                            custom_num_expr=None, custom_denom_expr=None):
   if z_a <= 0 or y <= 0 or z_min >= z_max:
       st.error("Invalid input parameters.")
       return None, None
   
   betas = np.linspace(0, 1, beta_steps)
   betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
   low_y_curve = compute_low_y_curve(betas, z_a, y)
   high_y_curve = compute_high_y_curve(betas, z_a, y)
   
   fig = go.Figure()
   
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
   
   fig.add_trace(
       go.Scatter(
           x=betas,
           y=low_y_curve,
           mode="markers+lines",
           name="Low y Expression",
           marker=dict(size=5, color='red'),
           line=dict(color='red'),
       )
   )
   
   fig.add_trace(
       go.Scatter(
           x=betas,
           y=high_y_curve,
           mode="markers+lines",
           name="High y Expression",
           marker=dict(size=5, color='green'),
           line=dict(color='green'),
       )
   )
   
   custom_curve = None
   if custom_num_expr and custom_denom_expr:
       custom_curve = compute_custom_expression(betas, z_a, y, custom_num_expr, custom_denom_expr)
       fig.add_trace(
           go.Scatter(
               x=betas,
               y=custom_curve,
               mode="markers+lines",
               name="Custom Expression",
               marker=dict(size=5, color='purple'),
               line=dict(color='purple'),
           )
       )
   
   fig.update_layout(
       title="Curves vs β: z*(β) Boundaries and Asymptotic Expressions",
       xaxis_title="β",
       yaxis_title="Value",
       hovermode="x unified",
   )
   
   dzmax_dbeta = np.gradient(z_maxs, betas)
   dzmin_dbeta = np.gradient(z_mins, betas)
   dlowy_dbeta = np.gradient(low_y_curve, betas)
   dhighy_dbeta = np.gradient(high_y_curve, betas)
   dcustom_dbeta = np.gradient(custom_curve, betas) if custom_curve is not None else None
   
   fig_deriv = go.Figure()
   
   fig_deriv.add_trace(
       go.Scatter(
           x=betas,
           y=dzmax_dbeta,
           mode="markers+lines",
           name="d/dβ Upper z*(β)",
           marker=dict(size=5, color='blue'),
           line=dict(color='blue'),
       )
   )
   
   fig_deriv.add_trace(
       go.Scatter(
           x=betas,
           y=dzmin_dbeta,
           mode="markers+lines",
           name="d/dβ Lower z*(β)",
           marker=dict(size=5, color='lightblue'),
           line=dict(color='lightblue'),
       )
   )
   
   fig_deriv.add_trace(
       go.Scatter(
           x=betas,
           y=dlowy_dbeta,
           mode="markers+lines",
           name="d/dβ Low y Expression",
           marker=dict(size=5, color='red'),
           line=dict(color='red'),
       )
   )
   
   fig_deriv.add_trace(
       go.Scatter(
           x=betas,
           y=dhighy_dbeta,
           mode="markers+lines",
           name="d/dβ High y Expression",
           marker=dict(size=5, color='green'),
           line=dict(color='green'),
       )
   )
   
   if dcustom_dbeta is not None:
       fig_deriv.add_trace(
           go.Scatter(
               x=betas,
               y=dcustom_dbeta,
               mode="markers+lines",
               name="d/dβ Custom Expression",
               marker=dict(size=5, color='purple'),
               line=dict(color='purple'),
           )
       )
       
   fig_deriv.update_layout(
       title="Derivatives vs β of Each Curve",
       xaxis_title="β",
       yaxis_title="d(Value)/dβ",
       hovermode="x unified",
   )
   
   return fig, fig_deriv

def compute_cubic_roots(z, beta, z_a, y):
   a = z * z_a
   b = z * z_a + z + z_a - z_a*y
   c = z + z_a + 1 - y*(beta*z_a + 1 - beta)
   d = 1
   
   coeffs = [a, b, c, d]
   roots = np.roots(coeffs)
   return roots

def generate_root_plots(beta, y, z_a, z_min, z_max, n_points):
   if z_a <= 0 or y <= 0 or z_min >= z_max:
       st.error("Invalid input parameters.")
       return None, None
   
   z_points = np.linspace(z_min, z_max, n_points)
   ims = []
   res = []
   
   for z in z_points:
       roots = compute_cubic_roots(z, beta, z_a, y)
       roots = sorted(roots, key=lambda x: abs(x.imag))
       ims.append([root.imag for root in roots])
       res.append([root.real for root in roots])
   
   ims = np.array(ims)
   res = np.array(res)
   
   fig_im = go.Figure()
   for i in range(3):
       fig_im.add_trace(
           go.Scatter(
               x=z_points,
               y=ims[:,i],
               mode="lines",
               name=f"Im{{s{i+1}}}",
               line=dict(width=2),
           )
       )
   fig_im.update_layout(
       title=f"Im{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
       xaxis_title="z",
       yaxis_title="Im{s}",
       hovermode="x unified",
   )
   
   fig_re = go.Figure()
   for i in range(3):
       fig_re.add_trace(
           go.Scatter(
               x=z_points,
               y=res[:,i],
               mode="lines",
               name=f"Re{{s{i+1}}}",
               line=dict(width=2),
           )
       )
   fig_re.update_layout(
       title=f"Re{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
       xaxis_title="z",
       yaxis_title="Re{s}",
       hovermode="x unified",
   )
   
   return fig_im, fig_re

# ------------------- Streamlit UI -------------------

st.title("Cubic Root Analysis")

tab1, tab2, tab3 = st.tabs(["z*(β) Curves", "Im{s} vs. z", "z*(β) Difference Analysis"])

with tab1:
    st.header("Find z Values where Cubic Roots Transition Between Real and Complex")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        z_a_1 = st.number_input("z_a", value=1.0, key="z_a_1")
        y_1 = st.number_input("y", value=1.0, key="y_1")
        z_min_1 = st.number_input("z_min", value=-10.0, key="z_min_1")
        z_max_1 = st.number_input("z_max", value=10.0, key="z_max_1")
        
        with st.expander("Resolution Settings"):
            beta_steps = st.slider("β steps", min_value=51, max_value=501, value=201, step=50)
            z_steps = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, step=1000)
        
    if st.button("Compute z vs. β Curves"):
        with col2:
            fig_main, fig_deriv = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1,
                                                         beta_steps, z_steps,
                                                         custom_num_expr, custom_denom_expr)
            if fig_main is not None and fig_deriv is not None:
                st.plotly_chart(fig_main, use_container_width=True)
                st.plotly_chart(fig_deriv, use_container_width=True)

with tab2:
    st.header("Plot Complex Roots vs. z")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        beta = st.number_input("β", value=0.5, min_value=0.0, max_value=1.0)
        y_2 = st.number_input("y", value=1.0, key="y_2")
        z_a_2 = st.number_input("z_a", value=1.0, key="z_a_2")
        z_min_2 = st.number_input("z_min", value=-10.0, key="z_min_2")
        z_max_2 = st.number_input("z_max", value=10.0, key="z_max_2")
        
        with st.expander("Resolution Settings"):
            z_points = st.slider("z grid points", min_value=1000, max_value=10000, value=5000, step=500)
        
    if st.button("Compute Complex Roots vs. z"):
        with col2:
            fig_im, fig_re = generate_root_plots(beta, y_2, z_a_2, z_min_2, z_max_2, z_points)
            if fig_im is not None and fig_re is not None:
                st.plotly_chart(fig_im, use_container_width=True)
                st.plotly_chart(fig_re, use_container_width=True)

with tab3:
    st.header("z*(β) Difference Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        z_a_4 = st.number_input("z_a", value=1.0, key="z_a_4")
        y_4 = st.number_input("y", value=1.0, key="y_4")
        z_min_4 = st.number_input("z_min", value=-10.0, key="z_min_4")
        z_max_4 = st.number_input("z_max", value=10.0, key="z_max_4")
        
        with st.expander("Resolution Settings"):
            beta_steps_4 = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, key="beta_steps_4")
            z_steps_4 = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, step=1000, key="z_steps_4")
    
    if st.button("Compute Difference Analysis"):
        with col2:
            betas, z_diff, dz_diff, d2z_diff = compute_z_difference_and_derivatives(
                z_a_4, y_4, z_min_4, z_max_4, beta_steps_4, z_steps_4
            )
            
            # Plot difference
            fig_diff = go.Figure()
            fig_diff.add_trace(
                go.Scatter(
                    x=betas,
                    y=z_diff,
                    mode="lines",
                    name="z*(β) Difference",
                    line=dict(color='purple', width=2)
                )
            )
            fig_diff.update_layout(
                title="Difference between Upper and Lower z*(β)",
                xaxis_title="β",
                yaxis_title="z_max - z_min",
                hovermode="x unified"
            )
            st.plotly_chart(fig_diff, use_container_width=True)
            
            # Plot first derivative
            fig_first_deriv = go.Figure()
            fig_first_deriv.add_trace(
                go.Scatter(
                    x=betas,
                    y=dz_diff,
                    mode="lines",
                    name="First Derivative",
                    line=dict(color='blue', width=2)
                )
            )
            fig_first_deriv.update_layout(
                title="First Derivative of z*(β) Difference",
                xaxis_title="β",
                yaxis_title="d(z_max - z_min)/dβ",
                hovermode="x unified"
            )
            st.plotly_chart(fig_first_deriv, use_container_width=True)
            
            # Plot second derivative
            fig_second_deriv = go.Figure()
            fig_second_deriv.add_trace(
                go.Scatter(
                    x=betas,
                    y=d2z_diff,
                    mode="lines",
                    name="Second Derivative",
                    line=dict(color='green', width=2)
                )
            )
            fig_second_deriv.update_layout(
                title="Second Derivative of z*(β) Difference",
                xaxis_title="β",
                yaxis_title="d²(z_max - z_min)/dβ²",
                hovermode="x unified"
            )
            st.plotly_chart(fig_second_deriv, use_container_width=True)