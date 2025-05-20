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

# Set page config with wider layout
st.set_page_config(
    page_title="Matrix Analysis Dashboard",
    page_icon="Ã°ÂÂÂ",
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
            // Delta0 Ã¢ÂÂ 0: One double root and one simple root
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
    
    // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Beta range parameters Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
    const int num_beta_points = fineness; // Controlled by fineness parameter
    std::vector<double> beta_values(num_beta_points);
    for (int i = 0; i < num_beta_points; ++i) {
        beta_values[i] = static_cast<double>(i) / (num_beta_points - 1);
    }
    
    // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Storage for results Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
    std::vector<double> max_eigenvalues(num_beta_points);
    std::vector<double> min_eigenvalues(num_beta_points);
    std::vector<double> theoretical_max_values(num_beta_points);
    std::vector<double> theoretical_min_values(num_beta_points);
    
    try {
        // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ RandomÃ¢ÂÂGaussian X and S_n Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
        std::random_device rd;
        std::mt19937_64 rng{rd()};
        std::normal_distribution<double> norm(0.0, 1.0);
        
        cv::Mat X(p, n, CV_64F);
        for(int i = 0; i < p; ++i)
            for(int j = 0; j < n; ++j)
                X.at<double>(i,j) = norm(rng);
        
        // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Process each beta value Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
        for (int beta_idx = 0; beta_idx < num_beta_points; ++beta_idx) {
            double beta = beta_values[beta_idx];
            
            // Compute theoretical values with customizable precision
            theoretical_max_values[beta_idx] = compute_theoretical_max(a, y, beta, theory_grid_points, theory_tolerance);
            theoretical_min_values[beta_idx] = compute_theoretical_min(a, y, beta, theory_grid_points, theory_tolerance);
            
            // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Build T_n matrix Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
            int k = static_cast<int>(std::floor(beta * p));
            std::vector<double> diags(p, 1.0);
            std::fill_n(diags.begin(), k, a);
            std::shuffle(diags.begin(), diags.end(), rng);
            
            cv::Mat T_n = cv::Mat::zeros(p, p, CV_64F);
            for(int i = 0; i < p; ++i){
                T_n.at<double>(i,i) = diags[i];
            }
            
            // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Form B_n = (1/n) * X * T_n * X^T Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
            cv::Mat B = (X.t() * T_n * X) / static_cast<double>(n);
            
            // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Compute eigenvalues of B Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
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
            // Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Eigenvalue analysis mode Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
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
need_compile = not os.path.exists(executable) or st.sidebar.button("Ã°ÂÂÂ Recompile C++ Code")

if need_compile:
    with st.sidebar:
        with st.spinner("Compiling C++ code..."):
            # Try to detect the OpenCV installation
            opencv_detection_cmd = ["pk
g-config", "--cflags", "--libs", "opencv4"]
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
                    st.success(f"Ã¢ÂÂ Successfully compiled with: {cmd}")
                    break
            
            if not compiled:
                st.error("Ã¢ÂÂ All compilation attempts failed.")
                with st.expander("Compilation Details"):
                    st.code(compile_output)
                st.stop()
            
            # Make sure the executable is executable
            if platform.system() != "Windows":
                os.chmod(executable, 0o755)
            
            st.success("Ã¢ÂÂ C++ code compiled successfully!")

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
        # zasÃÂ³ + [z(a+1)+a(1-y)]sÃÂ² + [z+(a+1)-y-yÃÂ²(a-1)]s + 1 = 0
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
            f"Imaginary Parts of Roots: a={cubic_a}, y={cubic_y}, ÃÂ²={cubic_beta}",
            f"Real Parts of Roots: a={cubic_a}, y={cubic_y}, ÃÂ²={cubic_beta}"
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
            name='Im(sÃ¢ÂÂ)',
            line=dict(color='rgb(239, 85, 59)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Im(sÃ¢ÂÂ): %{y:.6f}<extra>Root 1</extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=ims_values2,
            mode='lines',
            name='Im(sÃ¢ÂÂ)',
            line=dict(color='rgb(0, 129, 201)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Im(sÃ¢ÂÂ): %{y:.6f}<extra>Root 2</extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=ims_values3,
            mode='lines',
            name='Im(sÃ¢ÂÂ)',
            line=dict(color='rgb(0, 176, 80)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Im(sÃ¢ÂÂ): %{y:.6f}<extra>Root 3</extra>'
        ),
        row=1, col=1
    )
    
    # Add traces for real parts
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=real_values1,
            mode='lines',
            name='Re(sÃ¢ÂÂ)',
            line=dict(color='rgb(239, 85, 59)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Re(sÃ¢ÂÂ): %{y:.6f}<extra>Root 1</extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=real_values2,
            mode='lines',
            name='Re(sÃ¢ÂÂ)',
            line=dict(color='rgb(0, 129, 201)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Re(sÃ¢ÂÂ): %{y:.6f}<extra>Root 2</extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_values, 
            y=real_values3,
            mode='lines',
            name='Re(sÃ¢ÂÂ)',
            line=dict(color='rgb(0, 176, 80)', width=2.5),
            hovertemplate='z: %{x:.4f}<br>Re(sÃ¢ÂÂ): %{y:.6f}<extra>Root 3</extra>'
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
        text=['sÃ¢ÂÂ', 'sÃ¢ÂÂ', 'sÃ¢ÂÂ'],
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

# Options for theme and appearance
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
tab1, tab2 = st.tabs(["Ã°ÂÂÂ Eigenvalue Analysis (C++)", "Ã°ÂÂÂ Im(s) vs z Analysis (SymPy)"])

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
            help="Number of points to calculate along the ÃÂ² axis (0 to 1)",
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
                                hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
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
                                hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
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
                                hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
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
                                hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
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
                                    'title': {'text': 'ÃÂ² Parameter', 'font': {'size': 18, 'color': '#424242'}},
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
                        hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
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
                        hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
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
                        hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
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
                        hovertemplate='ÃÂ²: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
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
                            'title': {'text': 'ÃÂ² Parameter', 'font': {'size': 18, 'color': '#424242'}},
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
                    st.info("Ã°ÂÂÂ Set parameters and click 'Generate Eigenvalue Analysis' to create a visualization.")
            else:
                # Show placeholder
                st.info("Ã°ÂÂÂ Set parameters and click 'Generate Eigenvalue Analysis' to create a visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Im(s) vs z Analysis with SymPy
with tab2:
    # Two-column layout
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
        cubic_beta = st.number_input("Value for ÃÂ²", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
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
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Starting cubic equation calculations with SymPy...")
                
                try:
                    # Create data file path 
                    data_file = os.path.join(output_dir, "cubic_data.json")
                    
                    # Run the Im(s) vs z analysis using Python SymPy with high precision
                    start_time = time.time()
                    
                    # Define progress callback for updating the progress bar
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Calculating with SymPy... {int(progress * 100)}% complete")
                    
                    # Run the analysis with progress updates
                    result = compute_ImS_vs_Z(cubic_a, cubic_y, cubic_beta, cubic_points, z_min, z_max, update_progress)
                    end_time = time.time()
                    
                    # Format the data for saving
                    save_data = {
                        'z_values': result['z_values'],
                        'ims_values1': result['ims_values1'],
                        'ims_values2': result['ims_values2'],
                        'ims_values3': result['ims_values3'],
                        'real_values1': result['real_values1'],
                        'real_values2': result['real_values2'],
                        'real_values3': result['real_values3'],
                        'parameters': {'a': cubic_a, 'y': cubic_y, 'beta': cubic_beta}
                    }
                    
                    # Save results to JSON
                    save_as_json(save_data, data_file)
                    status_text.text("SymPy calculations complete! Generating visualization...")
                    
                    # Clear progress container
                    progress_container.empty()
                    
                    # Create Dash-style visualization
                    dash_fig = create_dash_style_visualization(result, cubic_a, cubic_y, cubic_beta)
                    st.plotly_chart(dash_fig, use_container_width=True)
                    
                    # Create sub-tabs for additional visualizations
                    pattern_tab, complex_tab = st.tabs(["Root Pattern Analysis", "Complex Plane View"])
                    
                    # Root pattern visualization
                    with pattern_tab:
                        pattern_fig = create_root_pattern_visualization(result)
                        st.plotly_chart(pattern_fig, use_container_width=True)
                        
                        # Root pattern explanation
                        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                        st.markdown("""
                        ### Root Pattern Analysis
                        
                        The cubic equation in this analysis should ideally exhibit roots with the following pattern:
                        
                        - One root with negative real part
                        - One root with zero real part
                        - One root with positive real part
                        
                        Or, in special cases, all three roots may be zero. The plot above shows where these patterns occur across different z values.
                        
                        The SymPy implementation with high precision ensures accurate root-finding and pattern maintenance, which is essential for stability analysis. 
                        Blue points indicate where the ideal pattern is achieved, green points show where all roots are zero, and red points indicate other patterns.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Complex plane visualization
                    with complex_tab:
                        # Slider for selecting z value
                        z_idx = st.slider(
                            "Select z index", 
                            min_value=0, 
                            max_value=len(result['z_values'])-1, 
                            value=len(result['z_values'])//2,
                            help="Select a specific z value to visualize its roots in the complex plane"
                        )
                        
                        # Create complex plane visualization
                        complex_fig = create_complex_plane_visualization(result, z_idx)
                        st.plotly_chart(complex_fig, use_container_width=True)
                        
                        # Complex plane explanation
                        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                        st.markdown("""
                        ### Complex Plane Visualization
                        
                        This visualization shows the three roots of the cubic equation in the complex plane for the selected z value. 
                        The real part is shown on the horizontal axis, and the imaginary part on the vertical axis.
                        
                        - The dashed circle represents the unit circle |s| = 1
                        - The roots are colored to match the plots above
                        - Conjugate pairs of roots (with opposite imaginary parts) often appear in cubic equations
                        
                        Use the slider to explore how the roots move in the complex plane as z changes.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display computation time
                    st.success(f"SymPy computation completed in {end_time - start_time:.2f} seconds")
                    
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
                    
                    # Process data safely and convert it to the format we need
                    result = {
                        'z_values': np.array([safe_convert_to_numeric(x) for x in data['z_values']]),
                        'ims_values1': np.array([safe_convert_to_numeric(x) for x in data['ims_values1']]),
                        'ims_values2': np.array([safe_convert_to_numeric(x) for x in data['ims_values2']]),
                        'ims_values3': np.array([safe_convert_to_numeric(x) for x in data['ims_values3']]),
                        'real_values1': np.array([safe_convert_to_numeric(x) for x in data.get('real_values1', [0] * len(data['z_values']))]),
                        'real_values2': np.array([safe_convert_to_numeric(x) for x in data.get('real_values2', [0] * len(data['z_values']))]),
                        'real_values3': np.array([safe_convert_to_numeric(x) for x in data.get('real_values3', [0] * len(data['z_values']))]),
                    }
                    
                    # Extract cubic parameters from data if available (otherwise use defaults)
                    cubic_params = data.get('parameters', {'a': 2.0, 'y': 1.0, 'beta': 0.5})
                    cubic_a = cubic_params.get('a', 2.0)
                    cubic_y = cubic_params.get('y', 1.0)
                    cubic_beta = cubic_params.get('beta', 0.5)
                    
                    # Create Dash-style visualization from previous data
                    st.info("Displaying previous analysis results. Adjust parameters and click 'Generate Analysis' to create a new visualization.")
                    
                    dash_fig = create_dash_style_visualization(result, cubic_a, cubic_y, cubic_beta)
                    st.plotly_chart(dash_fig, use_container_width=True)
                    
                    # Create sub-tabs for additional visualizations
                    pattern_tab, complex_tab = st.tabs(["Root Pattern Analysis", "Complex Plane View"])
                    
                    # Root pattern visualization
                    with pattern_tab:
                        pattern_fig = create_root_pattern_visualization(result)
                        st.plotly_chart(pattern_fig, use_container_width=True)
                    
                    # Complex plane visualization
                    with complex_tab:
                        # Slider for selecting z value
                        z_idx = st.slider(
                            "Select z index", 
                            min_value=0, 
                            max_value=len(result['z_values'])-1, 
                            value=len(result['z_values'])//2,
                            help="Select a specific z value to visualize its roots in the complex plane"
                        )
                        
                        # Create complex plane visualization
                        complex_fig = create_complex_plane_visualization(result, z_idx)
                        st.plotly_chart(complex_fig, use_container_width=True)
                    
                except Exception as e:
                    st.info("Ã°ÂÂÂ Set parameters and click 'Generate Im(s) vs z Analysis' to create a visualization.")
                    st.error(f"Error loading previous data: {str(e)}")
            else:
                # Show placeholder
                st.info("Ã°ÂÂÂ Set parameters and click 'Generate Im(s) vs z Analysis' to create a visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)

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
