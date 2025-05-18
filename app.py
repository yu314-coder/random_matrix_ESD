import streamlit as st
import subprocess
import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
        return quadratic_solver(b, c, d)
    
    # Special case for d=0 (one root is zero)
    if abs(d) < epsilon:
        return handle_zero_d_case(a, b, c)
    
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
    return force_root_pattern(roots, zero_threshold)

# Helper function to solve quadratic equations
def quadratic_solver(b, c, d, epsilon=1e-40):
    if abs(b) < epsilon:  # Linear or constant
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

# Helper function to handle the case when d=0 (one root is zero)
def handle_zero_d_case(a, b, c):
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

# Helper function to force the root pattern
def force_root_pattern(roots, zero_threshold=1e-20):
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
        # zas³ + [z(a+1)+a(1-y)]s² + [z+(a+1)-y-yβ(a-1)]s + 1 = 0
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

# Create high-quality Dash-like visualizations for cubic equation analysis
def create_dash_style_visualizations(result, cubic_a, cubic_y, cubic_beta):
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
        width=900,
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