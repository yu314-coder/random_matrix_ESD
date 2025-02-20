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
def compute_low_y_curve(betas, z_a, y):
    """
    Compute the "Low y Expression" curve.
    """
    betas = np.array(betas)
    with np.errstate(invalid='ignore', divide='ignore'):
        sqrt_term = y * betas * (z_a - 1)
        sqrt_term = np.where(sqrt_term < 0, np.nan, np.sqrt(sqrt_term))
        term = (-1 + sqrt_term) / z_a
        numerator = (y - 2)*term + y * betas * ((z_a - 1)/z_a) - 1/z_a - 1
        denominator = term**2 + term
        mask = (denominator != 0) & ~np.isnan(denominator) & ~np.isnan(numerator)
        result = np.where(mask, numerator/denominator, np.nan)
    return result

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

def compute_custom_expression(betas, z_a, y, s, num_expr_str, denom_expr_str):
    """
    Compute a custom curve from numerator and denominator expressions given as strings.
    This version allows the expressions to depend on s (an extra parameter).
    Allowed variables: z_a, beta, y, s.
    Also, 'a' is accepted as an alias for z_a.
    """
    beta_sym, z_a_sym, y_sym, s_sym, a_sym = sp.symbols("beta z_a y s a", positive=True)
    local_dict = {"beta": beta_sym, "z_a": z_a_sym, "y": y_sym, "s": s_sym, "a": z_a_sym}
    try:
        num_expr = sp.sympify(num_expr_str, locals=local_dict)
        denom_expr = sp.sympify(denom_expr_str, locals=local_dict)
    except sp.SympifyError as e:
        st.error(f"Error parsing expressions: {e}")
        return np.full_like(betas, np.nan)
    
    num_func = sp.lambdify((beta_sym, z_a_sym, y_sym, s_sym), num_expr, modules=["numpy"])
    denom_func = sp.lambdify((beta_sym, z_a_sym, y_sym, s_sym), denom_expr, modules=["numpy"])
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num_func(betas, z_a, y, s) / denom_func(betas, z_a, y, s)
    return result

def generate_z_vs_beta_plot(z_a, y, z_min, z_max, beta_steps, z_steps,
                            custom_num_expr=None, custom_denom_expr=None, s_custom=None):
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None

    betas = np.linspace(0, 1, beta_steps)
    betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
    low_y_curve = compute_low_y_curve(betas, z_a, y)
    high_y_curve = compute_high_y_curve(betas, z_a, y)
    alt_low_expr = compute_alternate_low_expr(betas, z_a, y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=z_maxs, mode="markers+lines", name="Upper z*(β)",
                             marker=dict(size=5, color='blue'), line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=betas, y=z_mins, mode="markers+lines", name="Lower z*(β)",
                             marker=dict(size=5, color='lightblue'), line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=betas, y=low_y_curve, mode="markers+lines", name="Low y Expression",
                             marker=dict(size=5, color='red'), line=dict(color='red')))
    fig.add_trace(go.Scatter(x=betas, y=high_y_curve, mode="markers+lines", name="High y Expression",
                             marker=dict(size=5, color='green'), line=dict(color='green')))
    fig.add_trace(go.Scatter(x=betas, y=alt_low_expr, mode="markers+lines", name="Alternate Low Expression",
                             marker=dict(size=5, color='orange'), line=dict(color='orange')))
    
    if custom_num_expr and custom_denom_expr and s_custom is not None:
        custom_curve = compute_custom_expression(betas, z_a, y, s_custom, custom_num_expr, custom_denom_expr)
        fig.add_trace(go.Scatter(x=betas, y=custom_curve, mode="markers+lines", name="Custom Expression",
                                 marker=dict(size=5, color='purple'), line=dict(color='purple')))
    
    fig.update_layout(title="Curves vs β: z*(β) Boundaries and Asymptotic Expressions",
                      xaxis_title="β", yaxis_title="Value", hovermode="x unified")
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

def curve1(s, z, y):
    """First curve: z*s^2 + (z-y+1)*s + 1"""
    return z*s**2 + (z-y+1)*s + 1

def curve2(s, y, beta, a):
    """Second curve: y*β*((a-1)*s)/(a*s+1)"""
    return y*beta*((a-1)*s)/(a*s+1)

def find_intersections(z, y, beta, a, s_range, n_guesses, tolerance):
    """Find intersections between curve1 and curve2."""
    def equation(s):
        return curve1(s, z, y) - curve2(s, y, beta, a)
    s_guesses = np.linspace(s_range[0], s_range[1], n_guesses)
    intersections = []
    for s_guess in s_guesses:
        try:
            s_sol = fsolve(equation, s_guess, full_output=True, xtol=tolerance)
            if s_sol[2] == 1:
                s_val = s_sol[0][0]
                if (s_range[0] <= s_val <= s_range[1] and 
                    not any(abs(s_val - s_prev) < tolerance for s_prev in intersections)):
                    if abs(equation(s_val)) < tolerance:
                        intersections.append(s_val)
        except:
            continue
    intersections = np.sort(np.array(intersections))
    if len(intersections) % 2 != 0:
        refined_intersections = []
        for i in range(len(intersections)-1):
            mid_point = (intersections[i] + intersections[i+1]) / 2
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

def generate_curves_plot(z, y, beta, a, s_range, n_points, n_guesses, tolerance):
    s = np.linspace(s_range[0], s_range[1], n_points)
    y1 = curve1(s, z, y)
    y2 = curve2(s, y, beta, a)
    intersections = find_intersections(z, y, beta, a, s_range, n_guesses, tolerance)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=y1, mode='lines', name='z*s² + (z-y+1)*s + 1', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=s, y=y2, mode='lines', name='y*β*((a-1)*s)/(a*s+1)', line=dict(color='red', width=2)))
    if len(intersections) > 0:
        fig.add_trace(go.Scatter(x=intersections, y=curve1(intersections, z, y),
                                 mode='markers', name='Intersections',
                                 marker=dict(size=12, color='green', symbol='x', line=dict(width=2))))
    fig.update_layout(title=f"Curve Intersection Analysis (y={y:.4f}, β={beta:.4f}, a={a:.4f})",
                      xaxis_title="s", yaxis_title="Value", hovermode="closest",
                      showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig, intersections

# ----------------- Streamlit UI -----------------
st.title("Cubic Root Analysis")

# Define four tabs
tab1, tab2, tab3, tab4 = st.tabs(["z*(β) Curves", "Im{s} vs. z", "Curve Intersections", "Differential Analysis"])

# ----- Tab 1: z*(β) Curves -----
with tab1:
    st.header("Find z Values where Cubic Roots Transition Between Real and Complex")
    col1, col2 = st.columns([1, 2])
    with col1:
        z_a_1 = st.number_input("z_a", value=1.0, key="z_a_1")
        y_1 = st.number_input("y", value=1.0, key="y_1")
        z_min_1 = st.number_input("z_min", value=-10.0, key="z_min_1")
        z_max_1 = st.number_input("z_max", value=10.0, key="z_max_1")
        with st.expander("Resolution Settings", key="res1"):
            beta_steps = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, key="beta_steps")
            z_steps = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, step=1000, key="z_steps")
        st.subheader("Custom Expression")
        st.markdown("Enter the **numerator** and **denominator** expressions (as functions of `z_a`, `beta`, `y`, and `s`) for the custom curve. Here, 'a' is an alias for `z_a`.")
        st.markdown("The default expressions yield:")
        st.latex(r"\frac{y\beta(z_a-1)s+(a\,s+1)((y-1)s-1)}{(a\,s+1)(s^2+s)}")
        default_num = "y*beta*(z_a-1)*s + (a*s+1)*((y-1)*s-1)"
        default_denom = "(a*s+1)*(s**2+s)"
        custom_num_expr = st.text_input("Numerator Expression", value=default_num, key="custom_num")
        custom_denom_expr = st.text_input("Denominator Expression", value=default_denom, key="custom_denom")
        s_custom = st.number_input("Custom s value", value=1.0, step=0.1, key="s_custom")
    if st.button("Compute z vs. β Curves", key="tab1_button"):
        with col2:
            fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, z_steps,
                                          custom_num_expr, custom_denom_expr, s_custom)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Additional Expressions")
            st.markdown("""
**Low y Expression (Red):**
```
((y - 2)*((-1 + sqrt(y*beta*(z_a - 1)))/z_a) + y*beta*((z_a-1)/z_a) - 1/z_a - 1) / 
(((-1 + sqrt(y*beta*(z_a - 1)))/z_a)**2 + ((-1 + sqrt(y*beta*(z_a - 1)))/z_a))
```

**High y Expression (Green):**
```
(-4*z_a*(z_a-1)*y*beta - 2*z_a*y + 2*z_a*(2*z_a-1))/(1-2*z_a)
```

**Alternate Low Expression (Orange):**
```
(z_a*y*beta*(z_a-1) - 2*z_a*(1-y) - 2*z_a**2)/(2+2*z_a)
```
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
        with st.expander("Resolution Settings", key="res2"):
            z_points = st.slider("z grid points", min_value=1000, max_value=10000, value=5000, step=500, key="z_points")
    if st.button("Compute Complex Roots vs. z", key="tab2_button"):
        with col2:
            fig_im, fig_re = generate_root_plots(beta, y_2, z_a_2, z_min_2, z_max_2, z_points)
            if fig_im is not None and fig_re is not None:
                st.plotly_chart(fig_im, use_container_width=True)
                st.plotly_chart(fig_re, use_container_width=True)

# ----- Tab 3: Curve Intersections -----
with tab3:
    st.header("Curve Intersection Analysis")
    col1, col2 = st.columns([1, 2])
    with col1:
        z = st.slider("z", min_value=-10.0, max_value=10000.0, value=1.0, step=0.1, key="z_tab3")
        y_3 = st.slider("y", min_value=0.1, max_value=1000.0, value=1.0, step=0.1, key="y_tab3")
        beta_3 = st.slider("β", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="beta_tab3")
        a = st.slider("a", min_value=0.1, max_value=1000.0, value=1.0, step=0.1, key="a_tab3")
        st.subheader("s Range")
        s_min = st.number_input("s_min", value=-5.0, key="s_min_tab3")
        s_max = st.number_input("s_max", value=5.0, key="s_max_tab3")
        with st.expander("Resolution Settings", key="res3"):
            s_points = st.slider("s grid points", min_value=1000, max_value=10000, value=5000, step=500, key="s_points_tab3")
            intersection_guesses = st.slider("Intersection search points", min_value=200, max_value=2000, value=1000, step=100, key="intersect_guesses")
            intersection_tolerance = st.select_slider(
                "Intersection tolerance",
                options=[1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20],
                value=1e-10,
                key="intersect_tol"
            )
    if st.button("Compute Intersections", key="tab3_button"):
        with col2:
            s_range = (s_min, s_max)
            fig, intersections = generate_curves_plot(z, y_3, beta_3, a, s_range, s_points, intersection_guesses, intersection_tolerance)
            st.plotly_chart(fig, use_container_width=True)
            if len(intersections) > 0:
                st.subheader("Intersection Points")
                for i, s_val in enumerate(intersections):
                    y_val = curve1(s_val, z, y_3)
                    st.write(f"Point {i+1}: s = {s_val:.6f}, y = {y_val:.6f}")
            else:
                st.write("No intersections found in the given range.")

# ----- Tab 4: Differential Analysis -----
with tab4:
    st.header("Differential Analysis vs. β")
    st.markdown("This page shows the difference between the Upper (blue) and Lower (lightblue) z*(β) curves, along with their first and second derivatives with respect to β.")
    col1, col2 = st.columns([1, 2])
    with col1:
        z_a_diff = st.number_input("z_a", value=1.0, key="z_a_diff")
        y_diff = st.number_input("y", value=1.0, key="y_diff")
        z_min_diff = st.number_input("z_min", value=-10.0, key="z_min_diff")
        z_max_diff = st.number_input("z_max", value=10.0, key="z_max_diff")
        with st.expander("Resolution Settings", key="res4"):
            beta_steps_diff = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, key="beta_steps_diff")
            z_steps_diff = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, step=1000, key="z_steps_diff")
    if st.button("Compute Differentials", key="tab4_button"):
        with col2:
            betas_diff, lower_vals, upper_vals = sweep_beta_and_find_z_bounds(z_a_diff, y_diff, z_min_diff, z_max_diff, beta_steps_diff, z_steps_diff)
            diff_curve = upper_vals - lower_vals
            d1 = np.gradient(diff_curve, betas_diff)
            d2 = np.gradient(d1, betas_diff)
            
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(x=betas_diff, y=diff_curve, mode="lines", name="Difference (Upper - Lower)", line=dict(color="magenta", width=2)))
            fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", name="First Derivative", line=dict(color="brown", width=2)))
            fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", name="Second Derivative", line=dict(color="black", width=2)))
            fig_diff.update_layout(title="Differential Analysis vs. β", xaxis_title="β", yaxis_title="Value", hovermode="x unified")
            st.plotly_chart(fig_diff, use_container_width=True)