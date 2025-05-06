import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
import os
import sys
import tempfile
import subprocess
import importlib.util
import shutil

# Configure Streamlit for Hugging Face Spaces
st.set_page_config(
    page_title="Cubic Root Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define C++ extension code as a string
CPP_CODE = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

// Compute the cubic discriminant
double compute_discriminant(double z, double beta, double z_a, double y_effective) {
    double a = z * z_a;
    double b = z * z_a + z + z_a - z_a * y_effective;
    double c = z + z_a + 1 - y_effective * (beta * z_a + 1 - beta);
    double d = 1;
    
    // Symbolic expression for the cubic discriminant
    return std::pow((b*c)/(6*a*a) - std::pow(b, 3)/(27*std::pow(a, 3)) - d/(2*a), 2) + 
           std::pow(c/(3*a) - std::pow(b, 2)/(9*std::pow(a, 2)), 3);
}

// Find z values where the discriminant equals zero
std::vector<double> find_z_at_discriminant_zero(double z_a, double y, double beta, 
                                                double z_min, double z_max, int steps) {
    // Apply the condition for y
    double y_effective = y > 1 ? y : 1/y;
    
    // Create z grid
    std::vector<double> z_grid(steps);
    double step_size = (z_max - z_min) / (steps - 1);
    for (int i = 0; i < steps; i++) {
        z_grid[i] = z_min + i * step_size;
    }
    
    // Calculate discriminant values
    std::vector<double> disc_vals(steps);
    for (int i = 0; i < steps; i++) {
        disc_vals[i] = compute_discriminant(z_grid[i], beta, z_a, y_effective);
    }
    
    // Find roots
    std::vector<double> roots_found;
    
    for (int i = 0; i < steps - 1; i++) {
        double f1 = disc_vals[i];
        double f2 = disc_vals[i+1];
        
        // Skip if NaN
        if (std::isnan(f1) || std::isnan(f2)) {
            continue;
        }
        
        // Check for exact zero
        if (f1 == 0.0) {
            roots_found.push_back(z_grid[i]);
        } 
        else if (f2 == 0.0) {
            roots_found.push_back(z_grid[i+1]);
        }
        // Check for sign change
        else if (f1 * f2 < 0) {
            double zl = z_grid[i];
            double zr = z_grid[i+1];
            double f1_local = f1;
            double f2_local = f2;
            
            // Use binary search to refine the root
            for (int j = 0; j < 50; j++) {
                double mid = 0.5 * (zl + zr);
                double fm = compute_discriminant(mid, beta, z_a, y_effective);
                
                if (fm == 0) {
                    zl = zr = mid;
                    break;
                }
                
                if ((fm > 0 && f1_local > 0) || (fm < 0 && f1_local < 0)) {
                    zl = mid;
                    f1_local = fm;
                } else {
                    zr = mid;
                    f2_local = fm;
                }
            }
            
            roots_found.push_back(0.5 * (zl + zr));
        }
    }
    
    return roots_found;
}

// Sweep beta values and find z boundary values
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> 
sweep_beta_and_find_z_bounds(double z_a, double y, double z_min, double z_max, int beta_steps, int z_steps) {
    // Create beta values
    py::array_t<double> betas(beta_steps);
    auto betas_ptr = betas.mutable_data();
    double beta_step = 1.0 / (beta_steps - 1);
    for (int i = 0; i < beta_steps; i++) {
        betas_ptr[i] = i * beta_step;
    }
    
    // Initialize arrays for min and max z values
    py::array_t<double> z_min_values(beta_steps);
    py::array_t<double> z_max_values(beta_steps);
    auto z_min_ptr = z_min_values.mutable_data();
    auto z_max_ptr = z_max_values.mutable_data();
    
    for (int i = 0; i < beta_steps; i++) {
        double beta = betas_ptr[i];
        std::vector<double> roots = find_z_at_discriminant_zero(z_a, y, beta, z_min, z_max, z_steps);
        
        if (roots.size() == 0) {
            z_min_ptr[i] = std::numeric_limits<double>::quiet_NaN();
            z_max_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Find min and max roots
            double min_root = roots[0];
            double max_root = roots[0];
            for (size_t j = 1; j < roots.size(); j++) {
                if (roots[j] < min_root) min_root = roots[j];
                if (roots[j] > max_root) max_root = roots[j];
            }
            z_min_ptr[i] = min_root;
            z_max_ptr[i] = max_root;
        }
    }
    
    return std::make_tuple(betas, z_min_values, z_max_values);
}

// Compute High y Expression curve
py::array_t<double> compute_high_y_curve(py::array_t<double> betas, double z_a, double y) {
    // Apply the condition for y
    double y_effective = y > 1 ? y : 1/y;
    
    auto betas_ptr = betas.data();
    size_t n = betas.size();
    py::array_t<double> result(n);
    auto result_ptr = result.mutable_data();
    
    double a = z_a;
    double denominator = 1 - 2*a;
    
    if (std::abs(denominator) < 1e-10) {
        for (size_t i = 0; i < n; i++) {
            result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            double beta = betas_ptr[i];
            double numerator = -4*a*(a-1)*y_effective*beta - 2*a*y_effective - 2*a*(2*a-1);
            result_ptr[i] = numerator/denominator;
        }
    }
    
    return result;
}

// Compute alternative low expression
py::array_t<double> compute_alternate_low_expr(py::array_t<double> betas, double z_a, double y) {
    // Apply the condition for y
    double y_effective = y > 1 ? y : 1/y;
    
    auto betas_ptr = betas.data();
    size_t n = betas.size();
    py::array_t<double> result(n);
    auto result_ptr = result.mutable_data();
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas_ptr[i];
        result_ptr[i] = (z_a * y_effective * beta * (z_a - 1) - 2*z_a*(1 - y_effective) - 2*z_a*z_a) / (2 + 2*z_a);
    }
    
    return result;
}

// Compute max k expression
py::array_t<double> compute_max_k_expression(py::array_t<double> betas, double z_a, double y, int k_samples=1000) {
    // Apply the condition for y
    double y_effective = y > 1 ? y : 1/y;
    
    auto betas_ptr = betas.data();
    size_t n = betas.size();
    py::array_t<double> result(n);
    auto result_ptr = result.mutable_data();
    
    double a = z_a;
    
    // Sample k values on a logarithmic scale
    std::vector<double> k_values(k_samples);
    double log_min = -3;
    double log_max = 3;
    double log_step = (log_max - log_min) / (k_samples - 1);
    
    for (int j = 0; j < k_samples; j++) {
        k_values[j] = std::pow(10, log_min + j * log_step);
    }
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas_ptr[i];
        std::vector<double> values(k_samples);
        
        for (int j = 0; j < k_samples; j++) {
            double k = k_values[j];
            double numerator = y_effective*beta*(a-1)*k + (a*k+1)*((y_effective-1)*k-1);
            double denominator = (a*k+1)*(k*k+k);
            
            if (std::abs(denominator) < 1e-10) {
                values[j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                values[j] = numerator/denominator;
            }
        }
        
        // Find max value, ignoring NaNs
        double max_val = -std::numeric_limits<double>::infinity();
        bool found_valid = false;
        
        for (double val : values) {
            if (!std::isnan(val) && val > max_val) {
                max_val = val;
                found_valid = true;
            }
        }
        
        result_ptr[i] = found_valid ? max_val : std::numeric_limits<double>::quiet_NaN();
    }
    
    return result;
}

// Compute min t expression
py::array_t<double> compute_min_t_expression(py::array_t<double> betas, double z_a, double y, int t_samples=1000) {
    // Apply the condition for y
    double y_effective = y > 1 ? y : 1/y;
    
    auto betas_ptr = betas.data();
    size_t n = betas.size();
    py::array_t<double> result(n);
    auto result_ptr = result.mutable_data();
    
    double a = z_a;
    
    if (a <= 0) {
        for (size_t i = 0; i < n; i++) {
            result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return result;
    }
    
    // Create t values from -1/a to 0
    double lower_bound = -1/a + 1e-10;  // Avoid division by zero
    double step_size = (-1e-10 - lower_bound) / (t_samples - 1);
    std::vector<double> t_values(t_samples);
    
    for (int j = 0; j < t_samples; j++) {
        t_values[j] = lower_bound + j * step_size;
    }
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas_ptr[i];
        std::vector<double> values(t_samples);
        
        for (int j = 0; j < t_samples; j++) {
            double t = t_values[j];
            double numerator = y_effective*beta*(a-1)*t + (a*t+1)*((y_effective-1)*t-1);
            double denominator = (a*t+1)*(t*t+t);
            
            if (std::abs(denominator) < 1e-10) {
                values[j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                values[j] = numerator/denominator;
            }
        }
        
        // Find min value, ignoring NaNs
        double min_val = std::numeric_limits<double>::infinity();
        bool found_valid = false;
        
        for (double val : values) {
            if (!std::isnan(val) && val < min_val) {
                min_val = val;
                found_valid = true;
            }
        }
        
        result_ptr[i] = found_valid ? min_val : std::numeric_limits<double>::quiet_NaN();
    }
    
    return result;
}

// Compute eigenvalue support boundaries
std::tuple<py::array_t<double>, py::array_t<double>> 
compute_eigenvalue_support_boundaries(double z_a, double y, py::array_t<double> beta_values, 
                                      int n_samples = 100, int seeds = 5) {
    // Apply the condition for y
    double y_effective = y > 1 ? y : 1/y;
    
    auto beta_ptr = beta_values.data();
    size_t num_betas = beta_values.size();
    
    py::array_t<double> min_eigenvalues(num_betas);
    py::array_t<double> max_eigenvalues(num_betas);
    auto min_eig_ptr = min_eigenvalues.mutable_data();
    auto max_eig_ptr = max_eigenvalues.mutable_data();
    
    for (size_t i = 0; i < num_betas; i++) {
        double beta = beta_ptr[i];
        
        std::vector<double> min_vals;
        std::vector<double> max_vals;
        
        // Run multiple trials with different seeds
        for (int seed = 0; seed < seeds; seed++) {
            // Set random seed
            srand(seed * 100 + i);
            
            // Compute dimension p based on aspect ratio y
            int n = n_samples;
            int p = int(y_effective * n);
            
            // Constructing T_n (Population / Shape Matrix)
            int k = int(std::floor(beta * p));
            
            // Create diagonal entries
            std::vector<double> diag_entries(p);
            for (int j = 0; j < k; j++) {
                diag_entries[j] = z_a;
            }
            for (int j = k; j < p; j++) {
                diag_entries[j] = 1.0;
            }
            
            // Shuffle the diagonal entries (simple Fisher-Yates shuffle)
            for (int j = p-1; j > 0; j--) {
                int idx = rand() % (j+1);
                std::swap(diag_entries[j], diag_entries[idx]);
            }
            
            // Generate the data matrix X with i.i.d. standard normal entries
            std::vector<std::vector<double>> X(p, std::vector<double>(n));
            for (int row = 0; row < p; row++) {
                for (int col = 0; col < n; col++) {
                    // Box-Muller transform to generate normal distribution
                    double u1 = rand() / (RAND_MAX + 1.0);
                    double u2 = rand() / (RAND_MAX + 1.0);
                    if (u1 < 1e-10) u1 = 1e-10;  // Avoid log(0)
                    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                    X[row][col] = z;
                }
            }
            
            // Compute the sample covariance matrix S_n = (1/n) * XX^T
            std::vector<std::vector<double>> S_n(p, std::vector<double>(p, 0.0));
            for (int row = 0; row < p; row++) {
                for (int col = 0; col < p; col++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += X[row][k] * X[col][k];
                    }
                    S_n[row][col] = sum / n;
                }
            }
            
            // Compute B_n = S_n T_n
            std::vector<std::vector<double>> B_n(p, std::vector<double>(p, 0.0));
            for (int row = 0; row < p; row++) {
                for (int col = 0; col < p; col++) {
                    B_n[row][col] = S_n[row][col] * diag_entries[col];
                }
            }
            
            // Use Eigen library to compute eigenvalues
            Eigen::MatrixXd B_n_eigen(p, p);
            for (int row = 0; row < p; row++) {
                for (int col = 0; col < p; col++) {
                    B_n_eigen(row, col) = B_n[row][col];
                }
            }
            
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(B_n_eigen);
            Eigen::VectorXd eigenvalues = solver.eigenvalues();
            
            // Find min and max eigenvalues
            if (p > 0) {
                min_vals.push_back(eigenvalues(0));
                max_vals.push_back(eigenvalues(p-1));
            }
        }
        
        // Average over seeds for stability
        if (!min_vals.empty() && !max_vals.empty()) {
            double min_sum = 0.0, max_sum = 0.0;
            for (double val : min_vals) min_sum += val;
            for (double val : max_vals) max_sum += val;
            
            min_eig_ptr[i] = min_sum / min_vals.size();
            max_eig_ptr[i] = max_sum / max_vals.size();
        } else {
            min_eig_ptr[i] = std::numeric_limits<double>::quiet_NaN();
            max_eig_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
    
    return std::make_tuple(min_eigenvalues, max_eigenvalues);
}

PYBIND11_MODULE(cubic_cpp, m) {
    m.doc() = "C++ implementation of cubic root analysis functions";
    
    m.def("sweep_beta_and_find_z_bounds", &sweep_beta_and_find_z_bounds, 
          "Sweep beta values and find z boundary values",
          py::arg("z_a"), py::arg("y"), py::arg("z_min"), py::arg("z_max"), 
          py::arg("beta_steps"), py::arg("z_steps"));
    
    m.def("compute_high_y_curve", &compute_high_y_curve,
          "Compute High y Expression curve",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_alternate_low_expr", &compute_alternate_low_expr,
          "Compute alternative low expression",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_max_k_expression", &compute_max_k_expression,
          "Compute max k expression",
          py::arg("betas"), py::arg("z_a"), py::arg("y"), py::arg("k_samples")=1000);
    
    m.def("compute_min_t_expression", &compute_min_t_expression,
          "Compute min t expression",
          py::arg("betas"), py::arg("z_a"), py::arg("y"), py::arg("t_samples")=1000);
    
    m.def("compute_eigenvalue_support_boundaries", &compute_eigenvalue_support_boundaries,
          "Compute eigenvalue support boundaries",
          py::arg("z_a"), py::arg("y"), py::arg("beta_values"), 
          py::arg("n_samples")=100, py::arg("seeds")=5);
}
'''

# Function to build and load the C++ extension
@st.cache_resource
def build_cpp_extension():
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Write C++ code to a file
        cpp_file = os.path.join(temp_dir, "cubic_cpp.cpp")
        with open(cpp_file, "w") as f:
            f.write(CPP_CODE)
        
        # Check if pybind11 and Eigen are installed
        try:
            import pybind11
            pybind11_include = pybind11.get_include()
        except ImportError:
            # Install pybind11 if not available
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
            import pybind11
            pybind11_include = pybind11.get_include()
        
        # Try to find Eigen or download it
        eigen_include = os.path.join(temp_dir, "eigen")
        if not os.path.exists(eigen_include):
            os.makedirs(eigen_include)
            # Download Eigen headers (just the minimal required parts)
            subprocess.check_call(["wget", "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz", "-O", os.path.join(temp_dir, "eigen.tar.gz")])
            subprocess.check_call(["tar", "-xzf", os.path.join(temp_dir, "eigen.tar.gz"), "-C", temp_dir])
            # Move Eigen headers to the include directory
            eigen_src = os.path.join(temp_dir, "eigen-3.4.0")
            for folder in ["Eigen", "unsupported"]:
                if os.path.exists(os.path.join(eigen_src, folder)):
                    shutil.copytree(os.path.join(eigen_src, folder), os.path.join(eigen_include, folder))
        
        # Build the extension module
        setup_py = os.path.join(temp_dir, "setup.py")
        with open(setup_py, "w") as f:
            f.write(f'''
from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        'cubic_cpp',
        ['cubic_cpp.cpp'],
        include_dirs=[
            pybind11.get_include(),
            os.path.dirname(os.path.abspath(__file__))
        ],
        language='c++'
    )
]

setup(
    name='cubic_cpp',
    ext_modules=ext_modules,
    py_modules=[],
)
''')
        
        # Build the extension in place
        subprocess.check_call([sys.executable, setup_py, "build_ext", "--inplace"], cwd=temp_dir)
        
        # Find the compiled module
        extension_path = None
        for file in os.listdir(temp_dir):
            if file.startswith("cubic_cpp") and file.endswith(".so"):
                extension_path = os.path.join(temp_dir, file)
                break
        
        if extension_path is None:
            st.warning("Failed to find the compiled C++ extension")
            return None
        
        # Load the module
        spec = importlib.util.spec_from_file_location("cubic_cpp", extension_path)
        cubic_cpp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cubic_cpp)
        
        return cubic_cpp
    except Exception as e:
        st.warning(f"Failed to build C++ extension: {str(e)}")
        return None

# Try to build and load the C++ extension
cubic_cpp = build_cpp_extension()

def add_sqrt_support(expr_str):
    """Replace 'sqrt(' with 'sp.sqrt(' for sympy compatibility"""
    return expr_str.replace('sqrt(', 'sp.sqrt(')

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
    # Use C++ implementation if available
    if cubic_cpp is not None:
        roots = np.array(cubic_cpp.find_z_at_discriminant_zero(z_a, y, beta, z_min, z_max, steps))
        return roots
    
    # Python fallback implementation
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    z_grid = np.linspace(z_min, z_max, steps)
    disc_vals = discriminant_func(z_grid, beta, z_a, y_effective)
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
                fm = discriminant_func(mid, beta, z_a, y_effective)
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
    # Use C++ implementation if available
    if cubic_cpp is not None:
        betas, z_min_values, z_max_values = cubic_cpp.sweep_beta_and_find_z_bounds(
            z_a, y, z_min, z_max, beta_steps, z_steps)
        return np.array(betas), np.array(z_min_values), np.array(z_max_values)
    
    # Python fallback implementation
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
def compute_eigenvalue_support_boundaries(z_a, y, beta_values, n_samples=100, seeds=5):
    """
    Compute the support boundaries of the eigenvalue distribution by directly
    finding the minimum and maximum eigenvalues of B_n = S_n T_n for different beta values.
    """
    # Use C++ implementation if available
    if cubic_cpp is not None:
        min_eigenvalues, max_eigenvalues = cubic_cpp.compute_eigenvalue_support_boundaries(
            z_a, y, beta_values, n_samples, seeds)
        return np.array(min_eigenvalues), np.array(max_eigenvalues)
    
    # Python fallback implementation
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    min_eigenvalues = np.zeros_like(beta_values)
    max_eigenvalues = np.zeros_like(beta_values)
    
    # Use a progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, beta in enumerate(beta_values):
        # Update progress
        progress_bar.progress((i + 1) / len(beta_values))
        status_text.text(f"Processing β = {beta:.2f} ({i+1}/{len(beta_values)})")
            
        min_vals = []
        max_vals = []
        
        # Run multiple trials with different seeds for more stable results
        for seed in range(seeds):
            # Set random seed
            np.random.seed(seed * 100 + i)
            
            # Compute dimension p based on aspect ratio y
            n = n_samples
            p = int(y_effective * n)
            
            # Constructing T_n (Population / Shape Matrix)
            k = int(np.floor(beta * p))
            diag_entries = np.concatenate([
                np.full(k, z_a),
                np.full(p - k, 1.0)
            ])
            np.random.shuffle(diag_entries)
            T_n = np.diag(diag_entries)
            
            # Generate the data matrix X with i.i.d. standard normal entries
            X = np.random.randn(p, n)
            
            # Compute the sample covariance matrix S_n = (1/n) * XX^T
            S_n = (1 / n) * (X @ X.T)
            
            # Compute B_n = S_n T_n
            B_n = S_n @ T_n
            
            # Compute eigenvalues of B_n
            eigenvalues = np.linalg.eigvalsh(B_n)
            
            # Find minimum and maximum eigenvalues
            min_vals.append(np.min(eigenvalues))
            max_vals.append(np.max(eigenvalues))
        
        # Average over seeds for stability
        min_eigenvalues[i] = np.mean(min_vals)
        max_eigenvalues[i] = np.mean(max_vals)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return min_eigenvalues, max_eigenvalues

@st.cache_data
def compute_high_y_curve(betas, z_a, y):
    """
    Compute the "High y Expression" curve.
    """
    # Use C++ implementation if available
    if cubic_cpp is not None:
        curve = cubic_cpp.compute_high_y_curve(betas, z_a, y)
        return np.array(curve)
    
    # Python fallback implementation
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z_a
    betas = np.array(betas)
    denominator = 1 - 2*a
    if denominator == 0:
        return np.full_like(betas, np.nan)
    numerator = -4*a*(a-1)*y_effective*betas - 2*a*y_effective - 2*a*(2*a-1)
    return numerator/denominator

@st.cache_data
def compute_alternate_low_expr(betas, z_a, y):
    """
    Compute the alternate low expression:
    (z_a*y*beta*(z_a-1) - 2*z_a*(1-y) - 2*z_a**2) / (2+2*z_a)
    """
    # Use C++ implementation if available
    if cubic_cpp is not None:
        curve = cubic_cpp.compute_alternate_low_expr(betas, z_a, y)
        return np.array(curve)
    
    # Python fallback implementation
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    betas = np.array(betas)
    return (z_a * y_effective * betas * (z_a - 1) - 2*z_a*(1 - y_effective) - 2*z_a**2) / (2 + 2*z_a)

@st.cache_data
def compute_max_k_expression(betas, z_a, y, k_samples=1000):
    """
    Compute max_{k ∈ (0,∞)} (y*beta*(a-1)*k + (a*k+1)*((y-1)*k-1)) / ((a*k+1)*(k^2+k))
    """
    # Use C++ implementation if available
    if cubic_cpp is not None:
        curve = cubic_cpp.compute_max_k_expression(betas, z_a, y, k_samples)
        return np.array(curve)
    
    # Python fallback implementation
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z_a
    # Sample k values on a logarithmic scale
    k_values = np.logspace(-3, 3, k_samples)
    
    max_vals = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        values = np.zeros_like(k_values)
        for j, k in enumerate(k_values):
            numerator = y_effective*beta*(a-1)*k + (a*k+1)*((y_effective-1)*k-1)
            denominator = (a*k+1)*(k**2+k)
            if abs(denominator) < 1e-10:
                values[j] = np.nan
            else:
                values[j] = numerator/denominator
        
        valid_indices = ~np.isnan(values)
        if np.any(valid_indices):
            max_vals[i] = np.max(values[valid_indices])
        else:
            max_vals[i] = np.nan
            
    return max_vals

@st.cache_data
def compute_min_t_expression(betas, z_a, y, t_samples=1000):
    """
    Compute min_{t ∈ (-1/a, 0)} (y*beta*(a-1)*t + (a*t+1)*((y-1)*t-1)) / ((a*t+1)*(t^2+t))
    """
    # Use C++ implementation if available
    if cubic_cpp is not None:
        curve = cubic_cpp.compute_min_t_expression(betas, z_a, y, t_samples)
        return np.array(curve)
    
    # Python fallback implementation
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    a = z_a
    if a <= 0:
        return np.full_like(betas, np.nan)
        
    lower_bound = -1/a + 1e-10  # Avoid division by zero
    t_values = np.linspace(lower_bound, -1e-10, t_samples)
    
    min_vals = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        values = np.zeros_like(t_values)
        for j, t in enumerate(t_values):
            numerator = y_effective*beta*(a-1)*t + (a*t+1)*((y_effective-1)*t-1)
            denominator = (a*t+1)*(t**2+t)
            if abs(denominator) < 1e-10:
                values[j] = np.nan
            else:
                values[j] = numerator/denominator
        
        valid_indices = ~np.isnan(values)
        if np.any(valid_indices):
            min_vals[i] = np.min(values[valid_indices])
        else:
            min_vals[i] = np.nan
            
    return min_vals

@st.cache_data
def compute_derivatives(curve, betas):
    """Compute first and second derivatives of a curve"""
    d1 = np.gradient(curve, betas)
    d2 = np.gradient(d1, betas)
    return d1, d2

def compute_all_derivatives(betas, z_mins, z_maxs, low_y_curve, high_y_curve, alt_low_expr, custom_curve1=None, custom_curve2=None):
    """Compute derivatives for all curves"""
    derivatives = {}
    
    # Upper z*(β)
    derivatives['upper'] = compute_derivatives(z_maxs, betas)
    
    # Lower z*(β)
    derivatives['lower'] = compute_derivatives(z_mins, betas)
    
    # Low y Expression (only if provided)
    if low_y_curve is not None:
        derivatives['low_y'] = compute_derivatives(low_y_curve, betas)
    
    # High y Expression
    if high_y_curve is not None:
        derivatives['high_y'] = compute_derivatives(high_y_curve, betas)
    
    # Alternate Low Expression
    if alt_low_expr is not None:
        derivatives['alt_low'] = compute_derivatives(alt_low_expr, betas)
    
    # Custom Expression 1 (if provided)
    if custom_curve1 is not None:
        derivatives['custom1'] = compute_derivatives(custom_curve1, betas)

    # Custom Expression 2 (if provided)
    if custom_curve2 is not None:
        derivatives['custom2'] = compute_derivatives(custom_curve2, betas)
        
    return derivatives

def compute_custom_expression(betas, z_a, y, s_num_expr, s_denom_expr, is_s_based=True):
    """
    Compute custom curve. If is_s_based=True, compute using s substitution.
    Otherwise, compute direct z(β) expression.
    """
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    beta_sym, z_a_sym, y_sym = sp.symbols("beta z_a y", positive=True)
    local_dict = {"beta": beta_sym, "z_a": z_a_sym, "y": y_sym, "sp": sp}
    
    try:
        # Add sqrt support
        s_num_expr = add_sqrt_support(s_num_expr)
        s_denom_expr = add_sqrt_support(s_denom_expr)
        
        num_expr = sp.sympify(s_num_expr, locals=local_dict)
        denom_expr = sp.sympify(s_denom_expr, locals=local_dict)
        
        if is_s_based:
            # Compute s and substitute into main expression
            s_expr = num_expr / denom_expr
            a = z_a_sym
            numerator = y_sym*beta_sym*(z_a_sym-1)*s_expr + (a*s_expr+1)*((y_sym-1)*s_expr-1)
            denominator = (a*s_expr+1)*(s_expr**2 + s_expr)
            final_expr = numerator/denominator
        else:
            # Direct z(β) expression
            final_expr = num_expr / denom_expr
            
    except sp.SympifyError as e:
        st.error(f"Error parsing expressions: {e}")
        return np.full_like(betas, np.nan)
    
    final_func = sp.lambdify((beta_sym, z_a_sym, y_sym), final_expr, modules=["numpy"])
    with np.errstate(divide='ignore', invalid='ignore'):
        result = final_func(betas, z_a, y_effective)
        if np.isscalar(result):
            result = np.full_like(betas, result)
    return result

def generate_z_vs_beta_plot(z_a, y, z_min, z_max, beta_steps, z_steps,
                          s_num_expr=None, s_denom_expr=None, 
                          z_num_expr=None, z_denom_expr=None,
                          show_derivatives=False,
                          show_high_y=False,
                          show_low_y=False,
                          show_max_k=True,
                          show_min_t=True,
                          use_eigenvalue_method=True,
                          n_samples=1000,
                          seeds=5):
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None

    betas = np.linspace(0, 1, beta_steps)
    
    if use_eigenvalue_method:
        # Use the eigenvalue method to compute boundaries
        st.info("Computing eigenvalue support boundaries. This may take a moment...")
        min_eigs, max_eigs = compute_eigenvalue_support_boundaries(z_a, y, betas, n_samples, seeds)
        z_mins, z_maxs = min_eigs, max_eigs
    else:
        # Use the original discriminant method
        betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
        
    high_y_curve = compute_high_y_curve(betas, z_a, y) if show_high_y else None
    alt_low_expr = compute_alternate_low_expr(betas, z_a, y) if show_low_y else None
    
    # Compute the max/min expressions
    max_k_curve = compute_max_k_expression(betas, z_a, y) if show_max_k else None
    min_t_curve = compute_min_t_expression(betas, z_a, y) if show_min_t else None
    
    # Compute both custom curves
    custom_curve1 = None
    custom_curve2 = None
    if s_num_expr and s_denom_expr:
        custom_curve1 = compute_custom_expression(betas, z_a, y, s_num_expr, s_denom_expr, True)
    if z_num_expr and z_denom_expr:
        custom_curve2 = compute_custom_expression(betas, z_a, y, z_num_expr, z_denom_expr, False)

    # Compute derivatives if needed
    if show_derivatives:
        derivatives = compute_all_derivatives(betas, z_mins, z_maxs, None, high_y_curve, 
                                           alt_low_expr, custom_curve1, custom_curve2)
        # Calculate derivatives for max_k and min_t curves if they exist
        if show_max_k:
            max_k_derivatives = compute_derivatives(max_k_curve, betas)
        if show_min_t:
            min_t_derivatives = compute_derivatives(min_t_curve, betas)

    fig = go.Figure()
    
    # Original curves
    if use_eigenvalue_method:
        fig.add_trace(go.Scatter(x=betas, y=z_maxs, mode="markers+lines", 
                                name="Upper Bound (Max Eigenvalue)", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=betas, y=z_mins, mode="markers+lines", 
                                name="Lower Bound (Min Eigenvalue)", line=dict(color='blue')))
        # Add shaded region between curves
        fig.add_trace(go.Scatter(
            x=np.concatenate([betas, betas[::-1]]),
            y=np.concatenate([z_maxs, z_mins[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        fig.add_trace(go.Scatter(x=betas, y=z_maxs, mode="markers+lines", 
                                name="Upper z*(β)", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=betas, y=z_mins, mode="markers+lines", 
                                name="Lower z*(β)", line=dict(color='blue')))

    # Add High y Expression only if selected
    if show_high_y and high_y_curve is not None:
        fig.add_trace(go.Scatter(x=betas, y=high_y_curve, mode="markers+lines", 
                                name="High y Expression", line=dict(color='green')))
    
    # Add Low Expression only if selected
    if show_low_y and alt_low_expr is not None:
        fig.add_trace(go.Scatter(x=betas, y=alt_low_expr, mode="markers+lines", 
                                name="Low Expression", line=dict(color='orange')))
    
    # Add the max/min curves if selected
    if show_max_k and max_k_curve is not None:
        fig.add_trace(go.Scatter(x=betas, y=max_k_curve, mode="lines", 
                                name="Max k Expression", line=dict(color='red', width=2)))
    
    if show_min_t and min_t_curve is not None:
        fig.add_trace(go.Scatter(x=betas, y=min_t_curve, mode="lines", 
                                name="Min t Expression", line=dict(color='purple', width=2)))
    
    if custom_curve1 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve1, mode="markers+lines", 
                                name="Custom 1 (s-based)", line=dict(color='magenta')))
    if custom_curve2 is not None:
        fig.add_trace(go.Scatter(x=betas, y=custom_curve2, mode="markers+lines", 
                                name="Custom 2 (direct)", line=dict(color='brown')))

    if show_derivatives:
        # First derivatives
        curve_info = [
            ('upper', 'Upper Bound' if use_eigenvalue_method else 'Upper z*(β)', 'blue'),
            ('lower', 'Lower Bound' if use_eigenvalue_method else 'Lower z*(β)', 'lightblue'),
        ]
        
        if show_high_y and high_y_curve is not None:
            curve_info.append(('high_y', 'High y', 'green'))
        if show_low_y and alt_low_expr is not None:
            curve_info.append(('alt_low', 'Alt Low', 'orange'))
        
        if custom_curve1 is not None:
            curve_info.append(('custom1', 'Custom 1', 'magenta'))
        if custom_curve2 is not None:
            curve_info.append(('custom2', 'Custom 2', 'brown'))

        for key, name, color in curve_info:
            if key in derivatives:
                fig.add_trace(go.Scatter(x=betas, y=derivatives[key][0], mode="lines", 
                                        name=f"{name} d/dβ", line=dict(color=color, dash='dash')))
                fig.add_trace(go.Scatter(x=betas, y=derivatives[key][1], mode="lines", 
                                        name=f"{name} d²/dβ²", line=dict(color=color, dash='dot')))
        
        # Add derivatives for max_k and min_t curves if they exist
        if show_max_k and max_k_curve is not None:
            fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[0], mode="lines", 
                                    name="Max k d/dβ", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[1], mode="lines", 
                                    name="Max k d²/dβ²", line=dict(color='red', dash='dot')))
        
        if show_min_t and min_t_curve is not None:
            fig.add_trace(go.Scatter(x=betas, y=min_t_derivatives[0], mode="lines", 
                                    name="Min t d/dβ", line=dict(color='purple', dash='dash')))
            fig.add_trace(go.Scatter(x=betas, y=min_t_derivatives[1], mode="lines", 
                                    name="Min t d²/dβ²", line=dict(color='purple', dash='dot')))

    fig.update_layout(
        title="Curves vs β: Eigenvalue Support Boundaries and Asymptotic Expressions" if use_eigenvalue_method 
              else "Curves vs β: z*(β) Boundaries and Asymptotic Expressions",
        xaxis_title="β",
        yaxis_title="Value",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

# Rest of your code for other functions and tabs...
# [...]

# ----- Tab 1: z*(β) Curves -----
st.title("Cubic Root Analysis")
st.header("Eigenvalue Support Boundaries")

# Cleaner layout with better column organization
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    z_a_1 = st.number_input("z_a", value=1.0, key="z_a_1")
    y_1 = st.number_input("y", value=1.0, key="y_1")
    
with col2:
    z_min_1 = st.number_input("z_min", value=-10.0, key="z_min_1")
    z_max_1 = st.number_input("z_max", value=10.0, key="z_max_1")

with col1:
    method_type = st.radio(
        "Calculation Method",
        ["Eigenvalue Method", "Discriminant Method"],
        index=0  # Default to eigenvalue method
    )

# Advanced settings in collapsed expanders
with st.expander("Method Settings", expanded=False):
    if method_type == "Eigenvalue Method":
        beta_steps = st.slider("β steps", min_value=21, max_value=101, value=51, step=10, 
                              key="beta_steps_eigen")
        n_samples = st.slider("Matrix size (n)", min_value=100, max_value=2000, value=1000, 
                            step=100)
        seeds = st.slider("Number of seeds", min_value=1, max_value=10, value=5, step=1)
    else:
        beta_steps = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, 
                              key="beta_steps")
        z_steps = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, 
                          step=1000, key="z_steps")

# Curve visibility options
with st.expander("Curve Visibility", expanded=False):
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        show_high_y = st.checkbox("Show High y Expression", value=False, key="show_high_y")
        show_max_k = st.checkbox("Show Max k Expression", value=True, key="show_max_k")
    with col_vis2:
        show_low_y = st.checkbox("Show Low y Expression", value=False, key="show_low_y")
        show_min_t = st.checkbox("Show Min t Expression", value=True, key="show_min_t")

# Custom expressions collapsed by default
with st.expander("Custom Expression 1 (s-based)", expanded=False):
    st.markdown("""Enter expressions for s = numerator/denominator 
                (using variables `y`, `beta`, `z_a`, and `sqrt()`)""")
    st.latex(r"\text{This s will be inserted into:}")
    st.latex(r"\frac{y\beta(z_a-1)\underline{s}+(a\underline{s}+1)((y-1)\underline{s}-1)}{(a\underline{s}+1)(\underline{s}^2 + \underline{s})}")
    s_num = st.text_input("s numerator", value="", key="s_num")
    s_denom = st.text_input("s denominator", value="", key="s_denom")

with st.expander("Custom Expression 2 (direct z(β))", expanded=False):
    st.markdown("""Enter direct expression for z(β) = numerator/denominator 
                (using variables `y`, `beta`, `z_a`, and `sqrt()`)""")
    z_num = st.text_input("z(β) numerator", value="", key="z_num")
    z_denom = st.text_input("z(β) denominator", value="", key="z_denom")

# Move show_derivatives to main UI level for better visibility
with col2:
    show_derivatives = st.checkbox("Show derivatives", value=False)

# Compute button
if st.button("Compute Curves", key="tab1_button"):
    with col3:
        use_eigenvalue_method = (method_type == "Eigenvalue Method")
        if use_eigenvalue_method:
            fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, None,
                                        s_num, s_denom, z_num, z_denom, show_derivatives, 
                                        show_high_y, show_low_y, show_max_k, show_min_t,
                                        use_eigenvalue_method=True, n_samples=n_samples, 
                                        seeds=seeds)
        else:
            fig = generate_z_vs_beta_plot(z_a_1, y_1, z_min_1, z_max_1, beta_steps, z_steps,
                                        s_num, s_denom, z_num, z_denom, show_derivatives,
                                        show_high_y, show_low_y, show_max_k, show_min_t,
                                        use_eigenvalue_method=False)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Curve explanations in collapsed expander
            with st.expander("Curve Explanations", expanded=False):
                if use_eigenvalue_method:
                    st.markdown("""
                    - **Upper/Lower Bounds** (Blue): Maximum/minimum eigenvalues of B_n = S_n T_n
                    - **Shaded Region**: Eigenvalue support region
                    - **High y Expression** (Green): Asymptotic approximation for high y values
                    - **Low Expression** (Orange): Alternative asymptotic expression
                    - **Max k Expression** (Red): $\\max_{k \\in (0,\\infty)} \\frac{y\\beta (a-1)k + \\bigl(ak+1\\bigr)\\bigl((y-1)k-1\\bigr)}{(ak+1)(k^2+k)}$
                    - **Min t Expression** (Purple): $\\min_{t \\in \\left(-\\frac{1}{a},\\, 0\\right)} \\frac{y\\beta (a-1)t + \\bigl(at+1\\bigr)\\bigl((y-1)t-1\\bigr)}{(at+1)(t^2+t)}$
                    - **Custom Expression 1** (Magenta): Result from user-defined s substituted into the main formula
                    - **Custom Expression 2** (Brown): Direct z(β) expression
                    """)
                else:
                    st.markdown("""
                    - **Upper z*(β)** (Blue): Maximum z value where discriminant is zero
                    - **Lower z*(β)** (Blue): Minimum z value where discriminant is zero
                    - **High y Expression** (Green): Asymptotic approximation for high y values
                    - **Low Expression** (Orange): Alternative asymptotic expression
                    - **Max k Expression** (Red): $\\max_{k \\in (0,\\infty)} \\frac{y\\beta (a-1)k + \\bigl(ak+1\\bigr)\\bigl((y-1)k-1\\bigr)}{(ak+1)(k^2+k)}$
                    - **Min t Expression** (Purple): $\\min_{t \\in \\left(-\\frac{1}{a},\\, 0\\right)} \\frac{y\\beta (a-1)t + \\bigl(at+1\\bigr)\\bigl((y-1)t-1\\bigr)}{(at+1)(t^2+t)}$
                    - **Custom Expression 1** (Magenta): Result from user-defined s substituted into the main formula
                    - **Custom Expression 2** (Brown): Direct z(β) expression
                    """)
                if show_derivatives:
                    st.markdown("""
                    Derivatives are shown as:
                    - Dashed lines: First derivatives (d/dβ)
                    - Dotted lines: Second derivatives (d²/dβ²)
                    """)

# Display C++ build status
if cubic_cpp is None:
    st.warning("C++ extension could not be built. Using Python implementation.")
else:
    st.success("C++ extension successfully built and loaded.")