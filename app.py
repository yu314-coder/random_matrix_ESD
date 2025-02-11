import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# Configure Streamlit for Hugging Face Spaces
st.set_page_config(
   page_title="Cubic Root Analysis",
   layout="wide",
   initial_sidebar_state="collapsed"
)

#############################
# 1) Define the discriminant
#############################

# Symbolic variables to build a symbolic expression of discriminant
z_sym, beta_sym, z_a_sym, y_sym = sp.symbols("z beta z_a y", real=True, positive=True)

# Define a, b, c, d in terms of z_sym, beta_sym, z_a_sym, y_sym
a_sym = z_sym * z_a_sym
b_sym = z_sym * z_a_sym + z_sym + z_a_sym - z_a_sym*y_sym  # Fixed coefficient b
c_sym = z_sym + z_a_sym + 1 - y_sym*(beta_sym*z_a_sym + 1 - beta_sym)
d_sym = 1

# Symbolic expression for the standard cubic discriminant
Delta_expr = (
   ( (b_sym*c_sym)/(6*a_sym**2) - (b_sym**3)/(27*a_sym**3) - d_sym/(2*a_sym) )**2
   + ( c_sym/(3*a_sym) - (b_sym**2)/(9*a_sym**2) )**3
)

# Turn that into a fast numeric function:
discriminant_func = sp.lambdify((z_sym, beta_sym, z_a_sym, y_sym), Delta_expr, "numpy")

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def compute_low_y_curve(betas, z_a, y):
   """
   Compute the additional curve with proper handling of divide by zero cases
   """
   betas = np.array(betas)
   with np.errstate(invalid='ignore', divide='ignore'):
       sqrt_term = y * betas * (z_a - 1)
       sqrt_term = np.where(sqrt_term < 0, np.nan, np.sqrt(sqrt_term))
       
       term = (-1 + sqrt_term)/z_a
       numerator = (y - 2)*term + y * betas * ((z_a - 1)/z_a) - 1/z_a - 1
       denominator = term**2 + term
       
       # Handle division by zero and invalid values
       mask = (denominator != 0) & ~np.isnan(denominator) & ~np.isnan(numerator)
       return np.where(mask, numerator/denominator, np.nan)

@st.cache_data
def compute_high_y_curve(betas, z_a, y):
   """
   Compute the expression: ((4y + 12)(4 - a) + 16y*β*(a - 1))/(3(4 - a))
   """
   betas = np.array(betas)
   denominator = 3*(4 - z_a)
   
   if denominator == 0:
       return np.full_like(betas, np.nan)
       
   numerator = (4*y + 12)*(4 - z_a) + 16*y*betas*(z_a - 1)
   return numerator/denominator

def generate_z_vs_beta_plot(z_a, y, z_min, z_max):
   if z_a <= 0 or y <= 0 or z_min >= z_max:
       st.error("Invalid input parameters.")
       return None
   
   beta_steps = 101
   betas = np.linspace(0, 1, beta_steps)
   
   betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps=beta_steps)
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
   
   fig.update_layout(
       title="Curves vs β: z*(β) boundaries and Asymptotic Expressions",
       xaxis_title="β",
       yaxis_title="Value",
       hovermode="x unified",
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

def generate_ims_vs_z_plot(beta, y, z_a, z_min, z_max):
   if z_a <= 0 or y <= 0 or z_min >= z_max:
       st.error("Invalid input parameters.")
       return None
   
   z_points = np.linspace(z_min, z_max, 1000)
   ims = []
   
   for z in z_points:
       roots = compute_cubic_roots(z, beta, z_a, y)
       roots = sorted(roots, key=lambda x: abs(x.imag))
       ims.append([root.imag for root in roots])
   
   ims = np.array(ims)
   
   fig = go.Figure()
   
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

def curve1(s, z, y):
   """First curve: z*s^2 + (z-y+1)*s + 1"""
   return z*s**2 + (z-y+1)*s + 1

def curve2(s, y, beta, a):
   """Second curve: y*β*((a-1)*s)/(a*s+1)"""
   return y*beta*((a-1)*s)/(a*s+1)

def find_intersections(z, y, beta, a, s_range):
   """Find intersections between the two curves with improved accuracy"""
   def equation(s):
       return curve1(s, z, y) - curve2(s, y, beta, a)
   
   # Create a finer grid of initial guesses
   s_guesses = np.linspace(s_range[0], s_range[1], 200)
   intersections = []
   
   # Parameters for accuracy
   tolerance = 1e-10
   
   # First pass: find all potential intersections
   for s_guess in s_guesses:
       try:
           s_sol = fsolve(equation, s_guess, full_output=True, xtol=tolerance)
           if s_sol[2] == 1:  # Check if convergence was achieved
               s_val = s_sol[0][0]
               if (s_range[0] <= s_val <= s_range[1] and 
                   not any(abs(s_val - s_prev) < tolerance for s_prev in intersections)):
                   if abs(equation(s_val)) < tolerance:
                       intersections.append(s_val)
       except:
           continue
   
   # Sort intersections
   intersections = np.sort(np.array(intersections))
   
   # Ensure even number of intersections by checking for missed ones
   if len(intersections) % 2 != 0:
       refined_intersections = []
       for i in range(len(intersections)-1):
           mid_point = (intersections[i] + intersections[i+1])/2
           try:
               s_sol = fsolve(equation, mid_point, full_output=True, xtol=tolerance)
               if s_sol[2] == 1:
                   s_val = s_sol[0][0]
                   if (intersections[i] < s_val < intersections[i+1] and 
                       abs(equation(s_val)) < tolerance):
                       refined_intersections.append(s_val)
           except:
               continue
       
       intersections = np.sort(np.append(intersections, refined_intersections))
   
   return intersections

def generate_curves_plot(z, y, beta, a, s_range):
   s = np.linspace(s_range[0], s_range[1], 2000)
   
   # Compute curves
   y1 = curve1(s, z, y)
   y2 = curve2(s, y, beta, a)
   
   # Find intersections with improved accuracy
   intersections = find_intersections(z, y, beta, a, s_range)
   
   fig = go.Figure()
   
   fig.add_trace(
       go.Scatter(
           x=s, y=y1,
           mode='lines',
           name='z*s² + (z-y+1)*s + 1',
           line=dict(color='blue', width=2)
       )
   )
   
   fig.add_trace(
       go.Scatter(
           x=s, y=y2,
           mode='lines',
           name='y*β*((a-1)*s)/(a*s+1)',
           line=dict(color='red', width=2)
       )
   )
   
   if len(intersections) > 0:
       fig.add_trace(
           go.Scatter(
               x=intersections,
               y=curve1(intersections, z, y),
               mode='markers',
               name='Intersections',
               marker=dict(
                   size=12,
                   color='green',
                   symbol='x',
                   line=dict(width=2)
               )
           )
       )
   
   fig.update_layout(
       title=f"Curve Intersection Analysis (y={y:.4f}, β={beta:.4f}, a={a:.4f})",
       xaxis_title="s",
       yaxis_title="Value",
       hovermode="closest",
       showlegend=True,
       legend=dict(
           yanchor="top",
           y=0.99,
           xanchor="left",
           x=0.01
       )
   )
   
   return fig, intersections

# Streamlit UI
st.title("Cubic Root Analysis")

tab1, tab2, tab3 = st.tabs(["z*(β) Curves", "Im{s} vs. z", "Curve Intersections"])

with tab1:
   st.header("Find z Values where Cubic Roots Transition Between Real and Complex")
   
   col1, col2 = st.columns([1, 2])
   
   with col1:
       z_a_1 = st.number_input("z_a", value=1.0, key="z_a_1")
       y_1 = st.number_input("y", value=1.0, key="y_1")
       z_min_1 = st.number_input("z_min", value=-10.0, key="z_min_1")
       z_max_1 = st.number_input("z_max", value=10.0, key="z_max_1")
        
        if st.button("Compute z vs. β Curves"):
            with col2:
                fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Additional Expressions")
                    st.markdown("""
                    **Low y Expression (Red):**
                    ```
                    ((y - 2)*(-1 + sqrt(y*β*(a-1)))/a + y*β*((a-1)/a) - 1/a - 1) / 
                    ((-1 + sqrt(y*β*(a-1)))/a)^2 + (-1 + sqrt(y*β*(a-1)))/a)
                    ```
                    
                    **High y Expression (Green):**
                    ```
                    ((4y + 12)(4 - a) + 16y*β*(a - 1))/(3(4 - a))
                    ```
                    where a = z_a
                    """)

with tab2:
    st.header("Plot Imaginary Parts of Roots vs. z")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        beta = st.number_input("β", value=0.5, min_value=0.0, max_value=1.0)
        y_2 = st.number_input("y", value=1.0, key="y_2")
        z_a_2 = st.number_input("z_a", value=1.0, key="z_a_2")
        z_min_2 = st.number_input("z_min", value=-10.0, key="z_min_2")
        z_max_2 = st.number_input("z_max", value=10.0, key="z_max_2")
        
        if st.button("Compute Im{s} vs. z"):
            with col2:
                fig = generate_ims_vs_z_plot(beta, y_2, z_a_2, z_min_2, z_max_2)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Curve Intersection Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        z = st.slider("z", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
        y_3 = st.slider("y", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="y_3")
        beta_3 = st.slider("β", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="beta_3")
        a = st.slider("a", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        # Add range inputs for s
        st.subheader("s Range")
        s_min = st.number_input("s_min", value=-5.0)
        s_max = st.number_input("s_max", value=5.0)
        
        if st.button("Compute Intersections"):
            with col2:
                s_range = (s_min, s_max)
                fig, intersections = generate_curves_plot(z, y_3, beta_3, a, s_range)
                st.plotly_chart(fig, use_container_width=True)
                
                if len(intersections) > 0:
                    st.subheader("Intersection Points")
                    for i, s_val in enumerate(intersections):
                        y_val = curve1(s_val, z, y_3)
                        st.write(f"Point {i+1}: s = {s_val:.6f}, y = {y_val:.6f}")
                else:
                    st.write("No intersections found in the given range.")