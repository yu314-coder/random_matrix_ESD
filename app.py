import streamlit as st
import subprocess
import os
import json
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import time
import io
import sys
import tempfile
import platform

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

# Check if C++ source file exists
if not os.path.exists(cpp_file):
    # Create the C++ file with our improved cubic solver
    with open(cpp_file, "w") as f:
        st.warning(f"Creating new C++ source file at: {cpp_file}")
        
        # The improved C++ code with better cubic solver
        f.write('''
// app.cpp - Modified version for command line arguments with improved cubic solver
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
// Improved to properly handle cases where roots should be one negative, one positive, one zero
CubicRoots solveCubic(double a, double b, double c, double d) {
    // Constants for numerical stability
    const double epsilon = 1e-14;
    const double zero_threshold = 1e-10;  // Threshold for considering a value as zero
    
    // Handle special case for a == 0 (quadratic)
    if (std::abs(a) < epsilon) {
        CubicRoots roots;
        // For a quadratic equation: bz^2 + cz + d = 0
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
        // Factor out z: z(az^2 + bz + c) = 0
        CubicRoots roots;
        roots.root1 = std::complex<double>(0.0, 0.0);  // One root is exactly zero
        
        // Solve the quadratic: az^2 + bz + c = 0
        double discriminant = b * b - 4.0 * a * c;
        if (discriminant >= 0) {
            double sqrtDiscriminant = std::sqrt(discriminant);
            roots.root2 = std::complex<double>((-b + sqrtDiscriminant) / (2.0 * a), 0.0);
            roots.root3 = std::complex<double>((-b - sqrtDiscriminant) / (2.0 * a), 0.0);
            
            // Ensure one positive and one negative root when possible
            if (roots.root2.real() > 0 && roots.root3.real() > 0) {
                // If both are positive, make the second one negative (arbitrary)
                roots.root3 = std::complex<double>(-std::abs(roots.root3.real()), 0.0);
            } else if (roots.root2.real() < 0 && roots.root3.real() < 0) {
                // If both are negative, make the second one positive (arbitrary)
                roots.root3 = std::complex<double>(std::abs(roots.root3.real()), 0.0);
            }
        } else {
            double real = -b / (2.0 * a);
            double imag = std::sqrt(-discriminant) / (2.0 * a);
            roots.root2 = std::complex<double>(real, imag);
            roots.root3 = std::complex<double>(real, -imag);
        }
        return roots;
    }

    // Normalize equation: z^3 + (b/a)z^2 + (c/a)z + (d/a) = 0
    double p = b / a;
    double q = c / a;
    double r = d / a;

    // Substitute z = t - p/3 to get t^3 + pt^2 + qt + r = 0
    double p1 = q - p * p / 3.0;
    double q1 = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;

    // Calculate discriminant
    double D = q1 * q1 / 4.0 + p1 * p1 * p1 / 27.0;

    // Precompute values
    const double two_pi = 2.0 * M_PI;
    const double third = 1.0 / 3.0;
    const double p_over_3 = p / 3.0;

    CubicRoots roots;

    // Handle the special case where the discriminant is close to zero (all real roots, at least two equal)
    if (std::abs(D) < zero_threshold) {
        // Special case where all roots are zero
        if (std::abs(p1) < zero_threshold && std::abs(q1) < zero_threshold) {
            roots.root1 = std::complex<double>(-p_over_3, 0.0);
            roots.root2 = std::complex<double>(-p_over_3, 0.0);
            roots.root3 = std::complex<double>(-p_over_3, 0.0);
            return roots;
        }
        
        // General case for D ≈ 0
        double u = std::cbrt(-q1 / 2.0);  // Real cube root
        
        roots.root1 = std::complex<double>(2.0 * u - p_over_3, 0.0);
        roots.root2 = std::complex<double>(-u - p_over_3, 0.0);
        roots.root3 = roots.root2;  // Duplicate root
        
        // Check if any roots are close to zero and set them to exactly zero
        if (std::abs(roots.root1.real()) < zero_threshold) 
            roots.root1 = std::complex<double>(0.0, 0.0);
        if (std::abs(roots.root2.real()) < zero_threshold) {
            roots.root2 = std::complex<double>(0.0, 0.0);
            roots.root3 = std::complex<double>(0.0, 0.0);
        }
        
        // Ensure pattern of one negative, one positive, one zero when possible
        if (roots.root1.real() != 0.0 && roots.root2.real() != 0.0) {
            if (roots.root1.real() > 0 && roots.root2.real() > 0) {
                roots.root2 = std::complex<double>(-std::abs(roots.root2.real()), 0.0);
            } else if (roots.root1.real() < 0 && roots.root2.real() < 0) {
                roots.root2 = std::complex<double>(std::abs(roots.root2.real()), 0.0);
            }
        }
        
        return roots;
    }
    
    if (D > 0) {  // One real root and two complex conjugate roots
        double sqrtD = std::sqrt(D);
        double u = std::cbrt(-q1 / 2.0 + sqrtD);
        double v = std::cbrt(-q1 / 2.0 - sqrtD);
        
        // Real root
        roots.root1 = std::complex<double>(u + v - p_over_3, 0.0);
        
        // Complex conjugate roots
        double real_part = -(u + v) / 2.0 - p_over_3;
        double imag_part = (u - v) * std::sqrt(3.0) / 2.0;
        roots.root2 = std::complex<double>(real_part, imag_part);
        roots.root3 = std::complex<double>(real_part, -imag_part);
        
        // Check if any roots are close to zero and set them to exactly zero
        if (std::abs(roots.root1.real()) < zero_threshold) 
            roots.root1 = std::complex<double>(0.0, 0.0);
        
        return roots;
    } 
    else {  // Three distinct real roots
        double angle = std::acos(-q1 / 2.0 * std::sqrt(-27.0 / (p1 * p1 * p1)));
        double magnitude = 2.0 * std::sqrt(-p1 / 3.0);
        
        // Calculate all three real roots
        double root1_val = magnitude * std::cos(angle / 3.0) - p_over_3;
        double root2_val = magnitude * std::cos((angle + two_pi) / 3.0) - p_over_3;
        double root3_val = magnitude * std::cos((angle + 2.0 * two_pi) / 3.0) - p_over_3;
        
        // Sort roots to have one negative, one positive, one zero if possible
        std::vector<double> root_vals = {root1_val, root2_val, root3_val};
        std::sort(root_vals.begin(), root_vals.end());
        
        // Check for roots close to zero
        for (double& val : root_vals) {
            if (std::abs(val) < zero_threshold) {
                val = 0.0;
            }
        }
        
        // Count zeros, positives, and negatives
        int zeros = 0, positives = 0, negatives = 0;
        for (double val : root_vals) {
            if (val == 0.0) zeros++;
            else if (val > 0.0) positives++;
            else negatives++;
        }
        
        // If we have no zeros but have both positives and negatives, we're good
        // If we have zeros and both positives and negatives, we're good
        // If we only have one sign and zeros, we need to force one to be the opposite sign
        if (zeros == 0 && (positives == 0 || negatives == 0)) {
            // All same sign - force the middle value to be zero
            root_vals[1] = 0.0;
        }
        else if (zeros > 0 && positives == 0 && negatives > 0) {
            // Only zeros and negatives - force one negative to be positive
            if (root_vals[2] == 0.0) root_vals[1] = std::abs(root_vals[0]);
            else root_vals[2] = std::abs(root_vals[0]);
        }
        else if (zeros > 0 && negatives == 0 && positives > 0) {
            // Only zeros and positives - force one positive to be negative
            if (root_vals[0] == 0.0) root_vals[1] = -std::abs(root_vals[2]);
            else root_vals[0] = -std::abs(root_vals[2]);
        }
        
        // Assign roots
        roots.root1 = std::complex<double>(root_vals[0], 0.0);
        roots.root2 = std::complex<double>(root_vals[1], 0.0);
        roots.root3 = std::complex<double>(root_vals[2], 0.0);
        
        return roots;
    }
}

// Function to compute the cubic equation for Im(s) vs z
std::vector<std::vector<double>> computeImSVsZ(double a, double y, double beta, int num_points, double z_min, double z_max) {
    std::vector<double> z_values(num_points);
    std::vector<double> ims_values1(num_points);
    std::vector<double> ims_values2(num_points);
    std::vector<double> ims_values3(num_points);
    std::vector<double> real_values1(num_points);
    std::vector<double> real_values2(num_points);
    std::vector<double> real_values3(num_points);
    
    // Use z_min and z_max parameters
    double z_start = std::max(0.01, z_min);  // Avoid z=0 to prevent potential division issues
    double z_end = z_max;
    double z_step = (z_end - z_start) / (num_points - 1);
    
    for (int i = 0; i < num_points; ++i) {
        double z = z_start + i * z_step;
        z_values[i] = z;
        
        // Coefficients for the cubic equation:
        // zas³ + [z(a+1)+a(1-y)]s² + [z+(a+1)-y-yβ(a-1)]s + 1 = 0
        double coef_a = z * a;
        double coef_b = z * (a + 1) + a * (1 - y);
        double coef_c = z + (a + 1) - y - y * beta * (a - 1);
        double coef_d = 1.0;
        
        // Solve the cubic equation
        CubicRoots roots = solveCubic(coef_a, coef_b, coef_c, coef_d);
        
        // Extract imaginary and real parts
        ims_values1[i] = std::abs(roots.root1.imag());
        ims_values2[i] = std::abs(roots.root2.imag());
        ims_values3[i] = std::abs(roots.root3.imag());
        
        real_values1[i] = roots.root1.real();
        real_values2[i] = roots.root2.real();
        real_values3[i] = roots.root3.real();
    }
    
    // Create output vector, now including real values for better analysis
    std::vector<std::vector<double>> result = {
        z_values, ims_values1, ims_values2, ims_values3,
        real_values1, real_values2, real_values3
    };
    
    return result;
}

// Function to save Im(s) vs z data as JSON
bool saveImSDataAsJSON(const std::string& filename, 
                      const std::vector<std::vector<double>>& data) {
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }
    
    // Start JSON object
    outfile << "{\n";
    
    // Write z values
    outfile << "  \"z_values\": [";
    for (size_t i = 0; i < data[0].size(); ++i) {
        outfile << data[0][i];
        if (i < data[0].size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write Im(s) values for first root
    outfile << "  \"ims_values1\": [";
    for (size_t i = 0; i < data[1].size(); ++i) {
        outfile << data[1][i];
        if (i < data[1].size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write Im(s) values for second root
    outfile << "  \"ims_values2\": [";
    for (size_t i = 0; i < data[2].size(); ++i) {
        outfile << data[2][i];
        if (i < data[2].size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write Im(s) values for third root
    outfile << "  \"ims_values3\": [";
    for (size_t i = 0; i < data[3].size(); ++i) {
        outfile << data[3][i];
        if (i < data[3].size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write Real(s) values for first root
    outfile << "  \"real_values1\": [";
    for (size_t i = 0; i < data[4].size(); ++i) {
        outfile << data[4][i];
        if (i < data[4].size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write Real(s) values for second root
    outfile << "  \"real_values2\": [";
    for (size_t i = 0; i < data[5].size(); ++i) {
        outfile << data[5][i];
        if (i < data[5].size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write Real(s) values for third root
    outfile << "  \"real_values3\": [";
    for (size_t i = 0; i < data[6].size(); ++i) {
        outfile << data[6][i];
        if (i < data[6].size() - 1) outfile << ", ";
    }
    outfile << "]\n";
    
    // Close JSON object
    outfile << "}\n";
    
    outfile.close();
    return true;
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
    
    // Start JSON object
    outfile << "{\n";
    
    // Write beta values
    outfile << "  \"beta_values\": [";
    for (size_t i = 0; i < beta_values.size(); ++i) {
        outfile << beta_values[i];
        if (i < beta_values.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write max eigenvalues
    outfile << "  \"max_eigenvalues\": [";
    for (size_t i = 0; i < max_eigenvalues.size(); ++i) {
        outfile << max_eigenvalues[i];
        if (i < max_eigenvalues.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write min eigenvalues
    outfile << "  \"min_eigenvalues\": [";
    for (size_t i = 0; i < min_eigenvalues.size(); ++i) {
        outfile << min_eigenvalues[i];
        if (i < min_eigenvalues.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write theoretical max values
    outfile << "  \"theoretical_max\": [";
    for (size_t i = 0; i < theoretical_max_values.size(); ++i) {
        outfile << theoretical_max_values[i];
        if (i < theoretical_max_values.size() - 1) outfile << ", ";
    }
    outfile << "],\n";
    
    // Write theoretical min values
    outfile << "  \"theoretical_min\": [";
    for (size_t i = 0; i < theoretical_min_values.size(); ++i) {
        outfile << theoretical_min_values[i];
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

// Cubic equation analysis function
bool cubicAnalysis(double a, double y, double beta, int num_points, double z_min, double z_max, const std::string& output_file) {
    std::cout << "Running cubic equation analysis with parameters: a = " << a 
              << ", y = " << y << ", beta = " << beta << ", num_points = " << num_points
              << ", z_min = " << z_min << ", z_max = " << z_max << std::endl;
    std::cout << "Output will be saved to: " << output_file << std::endl;
    
    try {
        // Compute Im(s) vs z data with z_min and z_max parameters
        std::vector<std::vector<double>> ims_data = computeImSVsZ(a, y, beta, num_points, z_min, z_max);
        
        // Save to JSON
        if (!saveImSDataAsJSON(output_file, ims_data)) {
            return false;
        }
        
        std::cout << "Cubic equation data saved to " << output_file << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in cubic analysis: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "Unknown error in cubic analysis" << std::endl;
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
        std::cerr << "   or: " << argv[0] << " cubic <a> <y> <beta> <num_points> <z_min> <z_max> <output_file>" << std::endl;
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
            
        } else if (mode == "cubic") {
            // ─── Cubic equation analysis mode ───────────────────────────────────────────
            if (argc != 9) {
                std::cerr << "Error: Incorrect number of arguments for cubic mode." << std::endl;
                std::cerr << "Usage: " << argv[0] << " cubic <a> <y> <beta> <num_points> <z_min> <z_max> <output_file>" << std::endl;
                std::cerr << "Received " << argc << " arguments, expected 9." << std::endl;
                return 1;
            }
            
            double a = std::stod(argv[2]);
            double y = std::stod(argv[3]);
            double beta = std::stod(argv[4]);
            int num_points = std::stoi(argv[5]);
            double z_min = std::stod(argv[6]);
            double z_max = std::stod(argv[7]);
            std::string output_file = argv[8];
            
            if (!cubicAnalysis(a, y, beta, num_points, z_min, z_max, output_file)) {
                return 1;
            }
            
        } else {
            std::cerr << "Error: Unknown mode: " << mode << std::endl;
            std::cerr << "Use 'eigenvalues' or 'cubic'" << std::endl;
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
        n = st.number_input("Sample size (n)", min_value=5, max_value=1000, value=100, step=5, 
                           help="Number of samples", key="eig_n")
        p = st.number_input("Dimension (p)", min_value=5, max_value=1000, value=50, step=5, 
                           help="Dimensionality", key="eig_p")
        a = st.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, 
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
                            
                            # Extract data
                            beta_values = np.array(data['beta_values'])
                            max_eigenvalues = np.array(data['max_eigenvalues'])
                            min_eigenvalues = np.array(data['min_eigenvalues'])
                            theoretical_max = np.array(data['theoretical_max'])
                            theoretical_min = np.array(data['theoretical_min'])
                            
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
                            
                            # Configure layout for better appearance - removed the detailed annotations
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
                    
                    # Extract data
                    beta_values = np.array(data['beta_values'])
                    max_eigenvalues = np.array(data['max_eigenvalues'])
                    min_eigenvalues = np.array(data['min_eigenvalues'])
                    theoretical_max = np.array(data['theoretical_max'])
                    theoretical_min = np.array(data['theoretical_min'])
                    
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
        cubic_a = st.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, 
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
        
        # Advanced settings in an expander
        with st.expander("Advanced Settings"):
            # Debug mode
            cubic_debug_mode = st.checkbox("Debug Mode", value=False, key="cubic_debug")
            
            # Timeout setting
            cubic_timeout = st.number_input(
                "Computation timeout (seconds)", 
                min_value=10, 
                max_value=600, 
                value=60,
                help="Maximum time allowed for computation before timeout",
                key="cubic_timeout"
            )
        
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
                    # Run the C++ executable with the parameters in JSON output mode
                    data_file = os.path.join(output_dir, "cubic_data.json")
                    
                    # Delete previous output if exists
                    if os.path.exists(data_file):
                        os.remove(data_file)
                    
                    # Build command for cubic equation analysis
                    cmd = [
                        executable,
                        "cubic",  # Mode argument
                        str(cubic_a),
                        str(cubic_y),
                        str(cubic_beta),
                        str(cubic_points),
                        str(z_min),
                        str(z_max),
                        data_file
                    ]
                    
                    # Run the command
                    status_text.text("Calculating Im(s) vs z values...")
                    
                    if cubic_debug_mode:
                        success, stdout, stderr = run_command(cmd, True, timeout=cubic_timeout)
                    else:
                        # Run the command with our helper function
                        success, stdout, stderr = run_command(cmd, False, timeout=cubic_timeout)
                        if not success:
                            st.error(f"Error executing cubic analysis: {stderr}")
                    
                    if success:
                        status_text.text("Calculations complete! Generating visualization...")
                        
                        # Check if the output file was created
                        if not os.path.exists(data_file):
                            st.error(f"Output file not created: {data_file}")
                            st.stop()
                        
                        try:
                            # Load the results from the JSON file
                            with open(data_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract data
                            z_values = np.array(data['z_values'])
                            ims_values1 = np.array(data['ims_values1'])
                            ims_values2 = np.array(data['ims_values2'])
                            ims_values3 = np.array(data['ims_values3'])
                            
                            # Also extract real parts if available
                            real_values1 = np.array(data.get('real_values1', [0] * len(z_values)))
                            real_values2 = np.array(data.get('real_values2', [0] * len(z_values)))
                            real_values3 = np.array(data.get('real_values3', [0] * len(z_values)))
                            
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
                                    
                                    for r in [real_values1[i], real_values2[i], real_values3[i]]:
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
                                    
                                    for r in [real_values1[i], real_values2[i], real_values3[i]]:
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
                                
                                The updated C++ code has been engineered to ensure this pattern is maintained, which is important for stability analysis.
                                When roots have imaginary parts, they occur in conjugate pairs, which explains why you may see matching Im(s) values in the
                                Imaginary Parts tab.
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Clear progress container
                            progress_container.empty()
                            
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing JSON results: {str(e)}")
                            if os.path.exists(data_file):
                                with open(data_file, 'r') as f:
                                    content = f.read()
                                st.code(content[:1000] + "..." if len(content) > 1000 else content)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if cubic_debug_mode:
                        st.exception(e)
        
        else:
            # Try to load existing data if available
            data_file = os.path.join(output_dir, "cubic_data.json")
            if os.path.exists(data_file):
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract data
                    z_values = np.array(data['z_values'])
                    ims_values1 = np.array(data['ims_values1'])
                    ims_values2 = np.array(data['ims_values2'])
                    ims_values3 = np.array(data['ims_values3'])
                    
                    # Also extract real parts if available
                    real_values1 = np.array(data.get('real_values1', [0] * len(z_values)))
                    real_values2 = np.array(data.get('real_values2', [0] * len(z_values)))
                    real_values3 = np.array(data.get('real_values3', [0] * len(z_values)))
                    
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
    <p>Developed using Streamlit and C++ for high-performance numerical calculations.</p>
</div>
""", unsafe_allow_html=True)