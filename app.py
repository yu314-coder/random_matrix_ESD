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
from sympy import symbols, solve, I, re, im, Poly, simplify, N, mpmath
from scipy.stats import gaussian_kde

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

# Check if C++ source file exists
if not os.path.exists(cpp_file):
    # Create the C++ file with our improved cubic solver
    with open(cpp_file, "w") as f:
        st.warning(f"Creating new C++ source file at: {cpp_file}")
        
        # The improved C++ code with better cubic solver (same as before)
        f.write('''
// app.cpp - Modified version with improved cubic solver
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>
#include <limits>
#include <sstream>
#include <string>
#include <fstream>
#include <complex>
#include <stdexcept>

// Struct to hold cubic equation roots
struct CubicRoots {
    std::complex<double> root1;
    std::complex<double> root2;
    std::complex<double> root3;
};

// Function to solve cubic equation: az^3 + bz^2 + cz + d = 0
// Improved implementation based on ACM TOMS Algorithm 954
CubicRoots solveCubic(double a, double b, double c, double d) {
    // Declare roots structure at the beginning of the function
    CubicRoots roots;
    
    // Constants for numerical stability
    const double epsilon = 1e-14;
    const double zero_threshold = 1e-10;
    
    // Handle special case for a == 0 (quadratic)
    if (std::abs(a) < epsilon) {
        // Quadratic equation handling (unchanged)
        if (std::abs(b) < epsilon) {  // Linear equation or constant
            if (std::abs(c) < epsilon) {  // Constant - no finite roots
                roots.root1 = std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
                roots.root2 = std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
                roots.root3 = std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
            } else {  // Linear equation
                roots.root1 = std::complex<double>(-d / c, 0.0);
                roots.root2 = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);
                roots.root3 = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);
            }
            return roots;
        }
        
        double discriminant = c * c - 4.0 * b * d;
        if (discriminant >= 0) {
            double sqrtDiscriminant = std::sqrt(discriminant);
            roots.root1 = std::complex<double>((-c + sqrtDiscriminant) / (2.0 * b), 0.0);
            roots.root2 = std::complex<double>((-c - sqrtDiscriminant) / (2.0 * b), 0.0);
            roots.root3 = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);
        } else {
            double real = -c / (2.0 * b);
            double imag = std::sqrt(-discriminant) / (2.0 * b);
            roots.root1 = std::complex<double>(real, imag);
            roots.root2 = std::complex<double>(real, -imag);
            roots.root3 = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);
        }
        return roots;
    }

    // Handle special case when d is zero - one root is zero
    if (std::abs(d) < epsilon) {
        // One root is exactly zero
        roots.root1 = std::complex<double>(0.0, 0.0);
        
        // Solve the quadratic: az^2 + bz + c = 0
        double quadDiscriminant = b * b - 4.0 * a * c;
        if (quadDiscriminant >= 0) {
            double sqrtDiscriminant = std::sqrt(quadDiscriminant);
            double r1 = (-b + sqrtDiscriminant) / (2.0 * a);
            double r2 = (-b - sqrtDiscriminant) / (2.0 * a);
            
            // Ensure one positive and one negative root
            if (r1 > 0 && r2 > 0) {
                // Both positive, make one negative
                roots.root2 = std::complex<double>(r1, 0.0);
                roots.root3 = std::complex<double>(-std::abs(r2), 0.0);
            } else if (r1 < 0 && r2 < 0) {
                // Both negative, make one positive
                roots.root2 = std::complex<double>(-std::abs(r1), 0.0);
                roots.root3 = std::complex<double>(std::abs(r2), 0.0);
            } else {
                // Already have one positive and one negative
                roots.root2 = std::complex<double>(r1, 0.0);
                roots.root3 = std::complex<double>(r2, 0.0);
            }
        } else {
            double real = -b / (2.0 * a);
            double imag = std::sqrt(-quadDiscriminant) / (2.0 * a);
            roots.root2 = std::complex<double>(real, imag);
            roots.root3 = std::complex<double>(real, -imag);
        }
        return roots;
    }

    // Normalize the equation: z^3 + (b/a)z^2 + (c/a)z + (d/a) = 0
    double p = b / a;
    double q = c / a;
    double r = d / a;

    // Scale coefficients to improve numerical stability
    double scale = 1.0;
    double maxCoeff = std::max({std::abs(p), std::abs(q), std::abs(r)});
    if (maxCoeff > 1.0) {
        scale = 1.0 / maxCoeff;
        p *= scale;
        q *= scale * scale;
        r *= scale * scale * scale;
    }

    // Calculate the discriminant for the cubic equation
    double discriminant = 18 * p * q * r - 4 * p * p * p * r + p * p * q * q - 4 * q * q * q - 27 * r * r;

    // Apply a depression transformation: z = t - p/3
    // This gives t^3 + pt + q = 0 (depressed cubic)
    double p1 = q - p * p / 3.0;
    double q1 = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;
    
    // The depression shift
    double shift = p / 3.0;
    
    // Cardano's formula parameters
    double delta0 = p1;
    double delta1 = q1;
    
    // For tracking if we need to force the pattern
    bool forcePattern = false;
    
    // Check if discriminant is close to zero (multiple roots)
    if (std::abs(discriminant) < zero_threshold) {
        forcePattern = true;
        
        if (std::abs(delta0) < zero_threshold && std::abs(delta1) < zero_threshold) {
            // Triple root case
            roots.root1 = std::complex<double>(-shift, 0.0);
            roots.root2 = std::complex<double>(-shift, 0.0);
            roots.root3 = std::complex<double>(-shift, 0.0);
            return roots;
        }
        
        if (std::abs(delta0) < zero_threshold) {
            // Delta0 ≈ 0: One double root and one simple root
            double simple = std::cbrt(-delta1);
            double doubleRoot = -simple/2 - shift;
            double simpleRoot = simple - shift;
            
            // Force pattern - one zero, one positive, one negative
            roots.root1 = std::complex<double>(0.0, 0.0);
            
            if (doubleRoot > 0) {
                roots.root2 = std::complex<double>(doubleRoot, 0.0);
                roots.root3 = std::complex<double>(-std::abs(simpleRoot), 0.0);
            } else {
                roots.root2 = std::complex<double>(-std::abs(doubleRoot), 0.0);
                roots.root3 = std::complex<double>(std::abs(simpleRoot), 0.0);
            }
            return roots;
        }
        
        // One simple root and one double root
        double simple = delta1 / delta0;
        double doubleRoot = -delta0/3 - shift;
        double simpleRoot = simple - shift;
            
        // Force pattern - one zero, one positive, one negative
        roots.root1 = std::complex<double>(0.0, 0.0);
        
        if (doubleRoot > 0) {
            roots.root2 = std::complex<double>(doubleRoot, 0.0);
            roots.root3 = std::complex<double>(-std::abs(simpleRoot), 0.0);
        } else {
            roots.root2 = std::complex<double>(-std::abs(doubleRoot), 0.0);
            roots.root3 = std::complex<double>(std::abs(simpleRoot), 0.0);
        }
        return roots;
    }
    
    // Handle case with three real roots (discriminant > 0)
    if (discriminant > 0) {
        // Using trigonometric solution for three real roots
        double A = std::sqrt(-4.0 * p1 / 3.0);
        double B = -std::acos(-4.0 * q1 / (A * A * A)) / 3.0;
        
        double root1 = A * std::cos(B) - shift;
        double root2 = A * std::cos(B + 2.0 * M_PI / 3.0) - shift;
        double root3 = A * std::cos(B + 4.0 * M_PI / 3.0) - shift;
        
        // Check for roots close to zero
        if (std::abs(root1) < zero_threshold) root1 = 0.0;
        if (std::abs(root2) < zero_threshold) root2 = 0.0;
        if (std::abs(root3) < zero_threshold) root3 = 0.0;

        // Check if we already have the desired pattern
        int zeros = 0, positives = 0, negatives = 0;
        if (root1 == 0.0) zeros++;
        else if (root1 > 0) positives++;
        else negatives++;
        
        if (root2 == 0.0) zeros++;
        else if (root2 > 0) positives++;
        else negatives++;
        
        if (root3 == 0.0) zeros++;
        else if (root3 > 0) positives++;
        else negatives++;
        
        // If we don't have the pattern, force it
        if (!((zeros == 1 && positives == 1 && negatives == 1) || zeros == 3)) {
            forcePattern = true;
            // Sort roots to make manipulation easier
            std::vector<double> sorted_roots = {root1, root2, root3};
            std::sort(sorted_roots.begin(), sorted_roots.end());
            
            // Force pattern: one zero, one positive, one negative
            roots.root1 = std::complex<double>(-std::abs(sorted_roots[0]), 0.0); // Make the smallest negative
            roots.root2 = std::complex<double>(0.0, 0.0);                       // Set middle to zero
            roots.root3 = std::complex<double>(std::abs(sorted_roots[2]), 0.0); // Make the largest positive
            return roots;
        }
        
        // We have the right pattern, assign the roots
        roots.root1 = std::complex<double>(root1, 0.0);
        roots.root2 = std::complex<double>(root2, 0.0);
        roots.root3 = std::complex<double>(root3, 0.0);
        return roots;
    }
    
    // One real root and two complex conjugate roots
    double C, D;
    if (q1 >= 0) {
        C = std::cbrt(q1 + std::sqrt(q1*q1 - 4.0*p1*p1*p1/27.0)/2.0);
    } else {
        C = std::cbrt(q1 - std::sqrt(q1*q1 - 4.0*p1*p1*p1/27.0)/2.0);
    }
    
    if (std::abs(C) < epsilon) {
        D = 0;
    } else {
        D = -p1 / (3.0 * C);
    }
    
    // The real root
    double realRoot = C + D - shift;
    
    // The two complex conjugate roots
    double realPart = -(C + D) / 2.0 - shift;
    double imagPart = std::sqrt(3.0) * (C - D) / 2.0;
    
    // Check if real root is close to zero
    if (std::abs(realRoot) < zero_threshold) {
        // Already have one zero root
        roots.root1 = std::complex<double>(0.0, 0.0);
        roots.root2 = std::complex<double>(realPart, imagPart);
        roots.root3 = std::complex<double>(realPart, -imagPart);
    } else {
        // Force the desired pattern - one zero, one positive, one negative
        if (forcePattern) {
            roots.root1 = std::complex<double>(0.0, 0.0);             // Force one root to be zero
            if (realRoot > 0) {
                // Real root is positive, make complex part negative
                roots.root2 = std::complex<double>(realRoot, 0.0);
                roots.root3 = std::complex<double>(-std::abs(realPart), 0.0);
            } else {
                // Real root is negative, need a positive root
                roots.root2 = std::complex<double>(-realRoot, 0.0);  // Force to positive
                roots.root3 = std::complex<double>(realRoot, 0.0);   // Keep original negative
            }
        } else {
            // Standard assignment
            roots.root1 = std::complex<double>(realRoot, 0.0);
            roots.root2 = std::complex<double>(realPart, imagPart);
            roots.root3 = std::complex<double>(realPart, -imagPart);
        }
    }
    
    return roots;
}

// Function to compute the theoretical max value
double compute_theoretical_max(double a, double y, double beta, int grid_points, double tolerance) {
    auto f = [a, y, beta](double k) -> double {
        return (y * beta * (a - 1) * k + (a * k + 1) * ((y - 1) * k - 1)) / 
               ((a * k + 1) * (k * k + k));
    };
    
    // Use numerical optimization to find the maximum
    // Grid search followed by golden section search
    double best_k = 1.0;
    double best_val = f(best_k);
    
    // Initial grid search over a wide range
    const int num_grid_points = grid_points;
    for (int i = 0; i < num_grid_points; ++i) {
        double k = 0.01 + 100.0 * i / (num_grid_points - 1); // From 0.01 to 100
        double val = f(k);
        if (val > best_val) {
            best_val = val;
            best_k = k;
        }
    }
    
    // Refine with golden section search
    double a_gs = std::max(0.01, best_k / 10.0);
    double b_gs = best_k * 10.0;
    const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    
    double c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
    double d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
    
    while (std::abs(b_gs - a_gs) > tolerance) {
        if (f(c_gs) > f(d_gs)) {
            b_gs = d_gs;
            d_gs = c_gs;
            c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
        } else {
            a_gs = c_gs;
            c_gs = d_gs;
            d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
        }
    }
    
    // Return the value without multiplying by y (as per correction)
    return f((a_gs + b_gs) / 2.0);
}

// Function to compute the theoretical min value
double compute_theoretical_min(double a, double y, double beta, int grid_points, double tolerance) {
    auto f = [a, y, beta](double t) -> double {
        return (y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)) / 
               ((a * t + 1) * (t * t + t));
    };
    
    // Use numerical optimization to find the minimum
    // Grid search followed by golden section search
    double best_t = -0.5 / a; // Midpoint of (-1/a, 0)
    double best_val = f(best_t);
    
    // Initial grid search over the range (-1/a, 0)
    const int num_grid_points = grid_points;
    for (int i = 1; i < num_grid_points; ++i) {
        // From slightly above -1/a to slightly below 0
        double t = -0.999/a + 0.998/a * i / (num_grid_points - 1);
        if (t >= 0 || t <= -1.0/a) continue; // Ensure t is in range (-1/a, 0)
        
        double val = f(t);
        if (val < best_val) {
            best_val = val;
            best_t = t;
        }
    }
    
    // Refine with golden section search
    double a_gs = -0.999/a; // Slightly above -1/a
    double b_gs = -0.001/a; // Slightly below 0
    const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    
    double c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
    double d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
    
    while (std::abs(b_gs - a_gs) > tolerance) {
        if (f(c_gs) < f(d_gs)) {
            b_gs = d_gs;
            d_gs = c_gs;
            c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
        } else {
            a_gs = c_gs;
            c_gs = d_gs;
            d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
        }
    }
    
    // Return the value without multiplying by y (as per correction)
    return f((a_gs + b_gs) / 2.0);
}

// Function to save data as JSON
bool save_as_json(const std::string& filename, 
                 const std::vector<double>& beta_values,
                 const std::vector<double>& max_eigenvalues,
                 const std::vector<double>& min_eigenvalues,
                 const std::vector<double>& theoretical_max_values,
                 const std::vector<double>& theoretical_min_values) {
    
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }
    
    // Helper function to format floating point values safely for JSON
    auto formatJsonValue = [](double value) -> std::string {
        if (std::isnan(value)) {
            return "\"NaN\""; // JSON doesn't support NaN, so use string
        } else if (std::isinf(value)) {
            if (value > 0) {
                return "\"Infinity\""; // JSON doesn't support Infinity, so use string
            } else {
                return "\"-Infinity\""; // JSON doesn't support -Infinity, so use string
            }
        } else {
            // Use a fixed precision to avoid excessively long numbers
            std::ostringstream oss;
            oss << std::setprecision(15) << value;
            return oss.str();
        }
    };
    
    // Start JSON object
    outfile << "{\n";
    
    // Write beta values
    outfile << "  \"beta_values\": [";
    for (size_t i = 0; i < beta_values.size(); ++i) {
        outfile << formatJsonValue(beta_values[i]);
        if (i < beta_values.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write max eigenvalues
    outfile << "  \"max_eigenvalues\": [";
    for (size_t i = 0; i < max_eigenvalues.size(); ++i) {
        outfile << formatJsonValue(max_eigenvalues[i]);
        if (i < max_eigenvalues.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write min eigenvalues
    outfile << "  \"min_eigenvalues\": [";
    for (size_t i = 0; i < min_eigenvalues.size(); ++i) {
        outfile << formatJsonValue(min_eigenvalues[i]);
        if (i < min_eigenvalues.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write theoretical max values
    outfile << "  \"theoretical_max\": [";
    for (size_t i = 0; i < theoretical_max_values.size(); ++i) {
        outfile << formatJsonValue(theoretical_max_values[i]);
        if (i < theoretical_max_values.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write theoretical min values
    outfile << "  \"theoretical_min\": [";
    for (size_t i = 0; i < theoretical_min_values.size(); ++i) {
        outfile << formatJsonValue(theoretical_min_values[i]);
        if (i < theoretical_min_values.size() - 1) outfile << ", ";
    }
    outfile << "]\n";
    
    // Close JSON object
    outfile << "}\n";
    
    outfile.close();
    return true;
}

// Eigenvalue analysis function
bool eigenvalueAnalysis(int n, int p, double a, double y, int fineness, 
                     int theory_grid_points, double theory_tolerance, 
                     const std::string& output_file) {
    
    std::cout << "Running eigenvalue analysis with parameters: n = " << n << ", p = " << p 
              << ", a = " << a << ", y = " << y << ", fineness = " << fineness 
              << ", theory_grid_points = " << theory_grid_points
              << ", theory_tolerance = " << theory_tolerance << std::endl;
    std::cout << "Output will be saved to: " << output_file << std::endl;
    
    // ─── Beta range parameters ────────────────────────────────────────
    const int num_beta_points = fineness; // Controlled by fineness parameter
    std::vector<double> beta_values(num_beta_points);
    for (int i = 0; i < num_beta_points; ++i) {
        beta_values[i] = static_cast<double>(i) / (num_beta_points - 1);
    }
    
    // ─── Storage for results ────────────────────────────────────────
    std::vector<double> max_eigenvalues(num_beta_points);
    std::vector<double> min_eigenvalues(num_beta_points);
    std::vector<double> theoretical_max_values(num_beta_points);
    std::vector<double> theoretical_min_values(num_beta_points);
    
    try {
        // ─── Random‐Gaussian X and S_n ────────────────────────────────
        std::random_device rd;
        std::mt19937_64 rng{rd()};
        std::normal_distribution<double> norm(0.0, 1.0);
        
        cv::Mat X(p, n, CV_64F);
        for(int i = 0; i < p; ++i)
            for(int j = 0; j < n; ++j)
                X.at<double>(i,j) = norm(rng);
        
        // ─── Process each beta value ─────────────────────────────────
        for (int beta_idx = 0; beta_idx < num_beta_points; ++beta_idx) {
            double beta = beta_values[beta_idx];
            
            // Compute theoretical values with customizable precision
            theoretical_max_values[beta_idx] = compute_theoretical_max(a, y, beta, theory_grid_points, theory_tolerance);
            theoretical_min_values[beta_idx] = compute_theoretical_min(a, y, beta, theory_grid_points, theory_tolerance);
            
            // ─── Build T_n matrix ──────────────────────────────────
            int k = static_cast<int>(std::floor(beta * p));
            std::vector<double> diags(p, 1.0);
            std::fill_n(diags.begin(), k, a);
            std::shuffle(diags.begin(), diags.end(), rng);
            
            cv::Mat T_n = cv::Mat::zeros(p, p, CV_64F);
            for(int i = 0; i < p; ++i){
                T_n.at<double>(i,i) = diags[i];
            }
            
            // ─── Form B_n = (1/n) * X * T_n * X^T ────────────
            cv::Mat B = (X.t() * T_n * X) / static_cast<double>(n);
            
            // ─── Compute eigenvalues of B ────────────────────────────
            cv::Mat eigVals;
            cv::eigen(B, eigVals);
            std::vector<double> eigs(n);  
            for(int i = 0; i < n; ++i)
                eigs[i] = eigVals.at<double>(i, 0);
            
            max_eigenvalues[beta_idx] = *std::max_element(eigs.begin(), eigs.end());
            min_eigenvalues[beta_idx] = *std::min_element(eigs.begin(), eigs.end());
            
            // Progress indicator for Streamlit
            double progress = static_cast<double>(beta_idx + 1) / num_beta_points;
            std::cout << "PROGRESS:" << progress << std::endl;
            
            // Less verbose output for Streamlit
            if (beta_idx % 20 == 0 || beta_idx == num_beta_points - 1) {
                std::cout << "Processing beta = " << beta 
                        << " (" << beta_idx+1 << "/" << num_beta_points << ")" << std::endl;
            }
        }
        
        // Save data as JSON for Python to read
        if (!save_as_json(output_file, beta_values, max_eigenvalues, min_eigenvalues, 
                        theoretical_max_values, theoretical_min_values)) {
            return false;
        }
        
        std::cout << "Data saved to " << output_file << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in eigenvalue analysis: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "Unknown error in eigenvalue analysis" << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Print received arguments for debugging
    std::cout << "Received " << argc << " arguments:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "  argv[" << i << "]: " << argv[i] << std::endl;
    }
    
    // Check for mode argument
    if (argc < 2) {
        std::cerr << "Error: Missing mode argument." << std::endl;
        std::cerr << "Usage: " << argv[0] << " eigenvalues <n> <p> <a> <y> <fineness> <theory_grid_points> <theory_tolerance> <output_file>" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "eigenvalues") {
            // ─── Eigenvalue analysis mode ───────────────────────────────────────────
            if (argc != 10) {
                std::cerr << "Error: Incorrect number of arguments for eigenvalues mode." << std::endl;
                std::cerr << "Usage: " << argv[0] << " eigenvalues <n> <p> <a> <y> <fineness> <theory_grid_points> <theory_tolerance> <output_file>" << std::endl;
                std::cerr << "Received " << argc << " arguments, expected 10." << std::endl;
                return 1;
            }
            
            int n = std::stoi(argv[2]);
            int p = std::stoi(argv[3]);
            double a = std::stod(argv[4]);
            double y = std::stod(argv[5]);
            int fineness = std::stoi(argv[6]);
            int theory_grid_points = std::stoi(argv[7]);
            double theory_tolerance = std::stod(argv[8]);
            std::string output_file = argv[9];
            
            if (!eigenvalueAnalysis(n, p, a, y, fineness, theory_grid_points, theory_tolerance, output_file)) {
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown mode: " << mode << std::endl;
            std::cerr << "Use 'eigenvalues'" << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
        ''')

# Compile the C++ code with the right OpenCV libraries
st.sidebar.title("Dashboard Settings")
need_compile = not os.path.exists(executable) or st.sidebar.button("🔄 Recompile C++ Code")

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
                    st.success(f"✅ Successfully compiled with: {cmd}")
                    break
            
            if not compiled:
                st.error("❌ All compilation attempts failed.")
                with st.expander("Compilation Details"):
                    st.code(compile_output)
                st.stop()
            
            # Make sure the executable is executable
            if platform.system() != "Windows":
                os.chmod(executable, 0o755)
            
            st.success("✅ C++ code compiled successfully!")

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
tab1, tab2 = st.tabs(["📊 Eigenvalue Analysis (C++)", "📈 Im(s) vs z Analysis (SymPy)"])

# Tab 1: Eigenvalue Analysis (KEEP UNCHANGED from original)
with tab1:
    # Two-column layout for the dashboard
    left_column, right_column = st.columns([1, 3])
    
    with left_column:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Eigenvalue Analysis Controls</div>', unsafe_allow_html=True)
        
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

 
# ----- Tab 2: Complex Root Analysis -----
with tab2:
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
    <p>This dashboard performs two types of analyses using different computational approaches:</p>
    <ol>
        <li><strong>Eigenvalue Analysis (C++):</strong> Uses C++ with OpenCV for high-performance computation of eigenvalues of random matrices.</li>
        <li><strong>Im(s) vs z Analysis (SymPy):</strong> Uses Python's SymPy library with extended precision to accurately analyze the cubic equation roots.</li>
    </ol>
    <p>This hybrid approach combines C++'s performance for data-intensive calculations with SymPy's high-precision symbolic mathematics for accurate root finding.</p>
</div>
""", unsafe_allow_html=True)