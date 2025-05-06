import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
import os
import tempfile
import subprocess
import sys
import importlib.util
import time
from datetime import timedelta
import mpmath  # Direct import instead of through sympy

# Set high precision for mpmath calculations
mpmath.mp.dps = 50  # 50 digits of precision

# Initialize session state for persistent UI elements
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.progress_containers = {}
    st.session_state.status_texts = {}
    st.session_state.progress_values = {}
    st.session_state.start_times = {}

# Configure Streamlit for Hugging Face Spaces
st.set_page_config(
    page_title="Cubic Root Analysis (C++ Accelerated)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Create containers for main UI elements that should persist
header_container = st.empty()
subtitle_container = st.empty()
cpp_status_container = st.empty()

# Display header - won't be refreshed later
with header_container:
    st.title("Cubic Root Analysis (C++ Accelerated)")

with subtitle_container:
    st.markdown("Analyze cubic equations with high-precision calculations and visualizations")


# Advanced Progress Bar Class that persists across refreshes
class AdvancedProgressBar:
    def __init__(self, key, total_steps, description="Processing"):
        self.key = key
        self.total_steps = total_steps
        self.description = description
        
        # Initialize in session state if not present
        if key not in st.session_state.progress_containers:
            st.session_state.progress_containers[key] = st.empty()
            st.session_state.status_texts[key] = st.empty()
            st.session_state.progress_values[key] = 0
            st.session_state.start_times[key] = time.time()
    
    def update(self, step=None, description=None):
        if step is not None:
            st.session_state.progress_values[self.key] = step
        else:
            st.session_state.progress_values[self.key] += 1
            
        if description:
            self.description = description
            
        # Calculate progress percentage
        progress = min(st.session_state.progress_values[self.key] / self.total_steps, 1.0)
        
        # Update progress bar
        st.session_state.progress_containers[self.key].progress(progress)
        
        # Calculate time metrics
        elapsed = time.time() - st.session_state.start_times[self.key]
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        if progress > 0:
            eta = elapsed * (1 - progress) / progress
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "Calculating..."
        
        # Update status text
        status_text = (f"{self.description} - Step {st.session_state.progress_values[self.key]} of {self.total_steps} "
                      f"({progress*100:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")
        st.session_state.status_texts[self.key].info(status_text)
    
    def complete(self):
        # Mark as completed
        st.session_state.progress_containers[self.key].progress(1.0)
        elapsed = time.time() - st.session_state.start_times[self.key]
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        st.session_state.status_texts[self.key].success(f"✅ {self.description} completed in {elapsed_str}")
    
    def clear(self):
        # Remove from session state
        if self.key in st.session_state.progress_containers:
            st.session_state.progress_containers[self.key].empty()
            st.session_state.status_texts[self.key].empty()
            del st.session_state.progress_containers[self.key]
            del st.session_state.status_texts[self.key]
            del st.session_state.progress_values[self.key]
            del st.session_state.start_times[self.key]


def add_sqrt_support(expr_str):
    """Replace 'sqrt(' with 'sp.sqrt(' for sympy compatibility"""
    return expr_str.replace('sqrt(', 'sp.sqrt(')


# C++ Module Compilation Function
@st.cache_resource
def compile_cpp_module():
    """Compile C++ acceleration module once and cache it"""
    # Create a unique progress tracker for compilation
    progress = AdvancedProgressBar("cpp_compilation", 5, "Compiling C++ acceleration module")
    
    # Define C++ code as a string
    cpp_code = """
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h>
    #include <Eigen/Dense>
    #include <Eigen/Eigenvalues>
    #include <complex>
    #include <vector>
    #include <random>
    #include <cmath>
    #include <algorithm>
    #include <omp.h>

    namespace py = pybind11;
    using namespace Eigen;

    // Fast discriminant computation function using standard cubic form
    double compute_discriminant_fast(double z, double beta, double z_a, double y) {
        double a = z * z_a;
        double b = z * z_a + z + z_a - z_a*y;
        double c = z + z_a + 1 - y*(beta*z_a + 1 - beta);
        double d = 1.0;
        
        // Standard formula for cubic discriminant
        return 18*a*b*c*d - 27*a*a*d*d + b*b*c*c - 2*b*b*b*d - 9*a*c*c*c;
    }

    // Batch computation of discriminant for array of z values
    py::array_t<double> discriminant_array(double beta, double z_a, double y, py::array_t<double> z_values) {
        auto z_buf = z_values.request();
        auto result = py::array_t<double>(z_buf.size);
        auto result_buf = result.request();
        
        double* z_ptr = static_cast<double*>(z_buf.ptr);
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        #pragma omp parallel for
        for (size_t i = 0; i < z_buf.size; i++) {
            result_ptr[i] = compute_discriminant_fast(z_ptr[i], beta, z_a, y);
        }
        
        return result;
    }

    // Find zeros of discriminant function
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> 
    find_discriminant_zeros(double z_a, double y, double z_min, double z_max, int beta_steps, int z_steps) {
        // Create beta grid
        auto betas = py::array_t<double>(beta_steps);
        auto betas_buf = betas.request();
        double* betas_ptr = static_cast<double*>(betas_buf.ptr);
        
        for (int i = 0; i < beta_steps; i++) {
            betas_ptr[i] = static_cast<double>(i) / (beta_steps - 1);
        }
        
        // Arrays for results
        auto z_mins = py::array_t<double>(beta_steps);
        auto z_maxs = py::array_t<double>(beta_steps);
        auto z_mins_buf = z_mins.request();
        auto z_maxs_buf = z_maxs.request();
        double* z_mins_ptr = static_cast<double*>(z_mins_buf.ptr);
        double* z_maxs_ptr = static_cast<double*>(z_maxs_buf.ptr);
        
        // Apply condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        // Create z grid
        std::vector<double> z_grid(z_steps);
        for (int i = 0; i < z_steps; i++) {
            z_grid[i] = z_min + (z_max - z_min) * static_cast<double>(i) / (z_steps - 1);
        }
        
        // For each beta value, find min and max z where discriminant is zero
        #pragma omp parallel for
        for (int b_idx = 0; b_idx < beta_steps; b_idx++) {
            double beta = betas_ptr[b_idx];
            std::vector<double> roots_found;
            
            // Calculate discriminant for all z values
            std::vector<double> disc_vals(z_steps);
            for (int i = 0; i < z_steps; i++) {
                disc_vals[i] = compute_discriminant_fast(z_grid[i], beta, z_a, y_effective);
            }
            
            // Find sign changes (zeros of discriminant)
            for (int i = 0; i < z_steps - 1; i++) {
                double f1 = disc_vals[i];
                double f2 = disc_vals[i+1];
                
                if (std::isnan(f1) || std::isnan(f2)) {
                    continue;
                }
                
                if (f1 == 0.0) {
                    roots_found.push_back(z_grid[i]);
                } else if (f2 == 0.0) {
                    roots_found.push_back(z_grid[i+1]);
                } else if (f1 * f2 < 0) {
                    // Binary search for more accurate root
                    double zl = z_grid[i], zr = z_grid[i+1];
                    double fl = f1, fr = f2;
                    for (int j = 0; j < 50; j++) {
                        double mid = 0.5 * (zl + zr);
                        double fm = compute_discriminant_fast(mid, beta, z_a, y_effective);
                        
                        if (fm == 0) {
                            zl = zr = mid;
                            break;
                        }
                        
                        if ((fm > 0 && fl > 0) || (fm < 0 && fl < 0)) {
                            zl = mid;
                            fl = fm;
                        } else {
                            zr = mid;
                            fr = fm;
                        }
                    }
                    roots_found.push_back(0.5 * (zl + zr));
                }
            }
            
            // Store min and max roots if any found
            if (roots_found.empty()) {
                z_mins_ptr[b_idx] = std::numeric_limits<double>::quiet_NaN();
                z_maxs_ptr[b_idx] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double min_root = *std::min_element(roots_found.begin(), roots_found.end());
                double max_root = *std::max_element(roots_found.begin(), roots_found.end());
                
                z_mins_ptr[b_idx] = min_root;
                z_maxs_ptr[b_idx] = max_root;
            }
        }
        
        return std::make_tuple(betas, z_mins, z_maxs);
    }

    // Compute eigenvalue support boundaries
    std::tuple<py::array_t<double>, py::array_t<double>> 
    compute_eigenvalue_boundaries(double z_a, double y, py::array_t<double> beta_values, int n_samples, int seeds) {
        auto beta_buf = beta_values.request();
        int beta_steps = beta_buf.size;
        
        // Results arrays
        auto min_eigenvalues = py::array_t<double>(beta_steps);
        auto max_eigenvalues = py::array_t<double>(beta_steps);
        auto min_buf = min_eigenvalues.request();
        auto max_buf = max_eigenvalues.request();
        double* min_ptr = static_cast<double*>(min_buf.ptr);
        double* max_ptr = static_cast<double*>(max_buf.ptr);
        double* beta_ptr = static_cast<double*>(beta_buf.ptr);
        
        // Apply condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        // Compute eigenvalues for each beta value
        #pragma omp parallel for
        for (int i = 0; i < beta_steps; i++) {
            double beta = beta_ptr[i];
            
            std::vector<double> min_vals;
            std::vector<double> max_vals;
            
            // Run multiple trials with different seeds for more stable results
            for (int seed = 0; seed < seeds; seed++) {
                // Set random seed
                std::mt19937 gen(seed * 100 + i);
                std::normal_distribution<double> normal_dist(0.0, 1.0);
                
                // Compute dimension p based on aspect ratio y
                int p = static_cast<int>(y_effective * n_samples);
                
                // Constructing T_n (Population / Shape Matrix)
                int k = static_cast<int>(std::floor(beta * p));
                
                // Create diagonal entries
                std::vector<double> diag_entries(p);
                std::fill_n(diag_entries.begin(), k, z_a);
                std::fill_n(diag_entries.begin() + k, p - k, 1.0);
                
                // Shuffle the diagonal entries
                std::shuffle(diag_entries.begin(), diag_entries.end(), gen);
                
                // Create T_n matrix
                MatrixXd T_n = MatrixXd::Zero(p, p);
                for (int j = 0; j < p; j++) {
                    T_n(j, j) = diag_entries[j];
                }
                
                // Generate random data matrix X with standard normal entries
                MatrixXd X(p, n_samples);
                for (int r = 0; r < p; r++) {
                    for (int c = 0; c < n_samples; c++) {
                        X(r, c) = normal_dist(gen);
                    }
                }
                
                // Compute sample covariance matrix S_n = (1/n) * XX^T
                MatrixXd S_n = (1.0 / n_samples) * (X * X.transpose());
                
                // Compute B_n = S_n T_n
                MatrixXd B_n = S_n * T_n;
                
                // Compute eigenvalues
                SelfAdjointEigenSolver<MatrixXd> solver(B_n);
                VectorXd eigenvalues = solver.eigenvalues();
                
                // Store min and max eigenvalues
                min_vals.push_back(eigenvalues.minCoeff());
                max_vals.push_back(eigenvalues.maxCoeff());
            }
            
            // Compute averages
            double min_avg = 0.0, max_avg = 0.0;
            for (double val : min_vals) min_avg += val;
            for (double val : max_vals) max_avg += val;
            
            min_ptr[i] = min_avg / seeds;
            max_ptr[i] = max_avg / seeds;
        }
        
        return std::make_tuple(min_eigenvalues, max_eigenvalues);
    }

    // Compute cubic roots using fast C++ implementation
    py::array_t<std::complex<double>> compute_cubic_roots_cpp(double z, double beta, double z_a, double y) {
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        // Coefficients in the form as³ + bs² + cs + d = 0
        double a = z * z_a;
        double b = z * z_a + z + z_a - z_a*y_effective;
        double c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta);
        double d = 1.0;
        
        // Handle special cases
        if (std::abs(a) < 1e-10) {
            // Create result array
            auto result = py::array_t<std::complex<double>>(3);
            auto buf = result.request();
            std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);
            
            if (std::abs(b) < 1e-10) {  // Linear case
                ptr[0] = std::complex<double>(-d/c, 0.0);
                ptr[1] = std::complex<double>(0.0, 0.0);
                ptr[2] = std::complex<double>(0.0, 0.0);
            } else {  // Quadratic case
                double discriminant = c*c - 4*b*d;
                if (discriminant >= 0) {
                    double sqrt_disc = std::sqrt(discriminant);
                    ptr[0] = std::complex<double>((-c + sqrt_disc) / (2*b), 0.0);
                    ptr[1] = std::complex<double>((-c - sqrt_disc) / (2*b), 0.0);
                } else {
                    double sqrt_disc = std::sqrt(-discriminant);
                    ptr[0] = std::complex<double>(-c/(2*b), sqrt_disc/(2*b));
                    ptr[1] = std::complex<double>(-c/(2*b), -sqrt_disc/(2*b));
                }
                ptr[2] = std::complex<double>(0.0, 0.0);
            }
            return result;
        }
        
        // For better numerical stability, normalize the equation: x³ + px² + qx + r = 0
        double p = b / a;
        double q = c / a;
        double r = d / a;
        
        // Depressed cubic: t³ + pt + q = 0 where x = t - p/3
        double p_prime = q - p*p/3.0;
        double q_prime = r - p*q/3.0 + 2.0*p*p*p/27.0;
        
        // Compute discriminant
        double discriminant = 4.0*p_prime*p_prime*p_prime/27.0 + q_prime*q_prime;
        
        // Create result array
        auto result = py::array_t<std::complex<double>>(3);
        auto buf = result.request();
        std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);
        
        // Calculate roots based on discriminant
        if (std::abs(discriminant) < 1e-10) {  // Discriminant ≈ 0
            if (std::abs(q_prime) < 1e-10) {  // Triple root
                ptr[0] = ptr[1] = ptr[2] = std::complex<double>(-p/3.0, 0.0);
            } else {  // One simple, one double root
                double u = std::cbrt(-q_prime/2.0);
                ptr[0] = std::complex<double>(2*u - p/3.0, 0.0);
                ptr[1] = ptr[2] = std::complex<double>(-u - p/3.0, 0.0);
            }
        } else if (discriminant > 0) {  // One real, two complex conjugate roots
            double sqrt_disc = std::sqrt(discriminant);
            std::complex<double> u = std::pow(std::complex<double>(-q_prime/2.0 + sqrt_disc/2.0, 0.0), 1.0/3.0);
            std::complex<double> v = std::pow(std::complex<double>(-q_prime/2.0 - sqrt_disc/2.0, 0.0), 1.0/3.0);
            
            ptr[0] = std::complex<double>(std::real(u + v) - p/3.0, 0.0);
            
            std::complex<double> omega(-0.5, 0.866025403784439);  // -1/2 + i*√3/2
            std::complex<double> omega2(-0.5, -0.866025403784439); // -1/2 - i*√3/2
            
            ptr[1] = omega * u + omega2 * v - std::complex<double>(p/3.0, 0.0);
            ptr[2] = omega2 * u + omega * v - std::complex<double>(p/3.0, 0.0);
        } else {  // Three distinct real roots
            double sqrt_disc = std::sqrt(-discriminant);
            double theta = std::atan2(sqrt_disc, -2.0*q_prime);
            double r_prime = std::pow(q_prime*q_prime + discriminant/4.0, 1.0/6.0);
            
            ptr[0] = std::complex<double>(2.0*r_prime*std::cos(theta/3.0) - p/3.0, 0.0);
            ptr[1] = std::complex<double>(2.0*r_prime*std::cos((theta + 2.0*M_PI)/3.0) - p/3.0, 0.0);
            ptr[2] = std::complex<double>(2.0*r_prime*std::cos((theta + 4.0*M_PI)/3.0) - p/3.0, 0.0);
        }
        
        return result;
    }

    // Compute high y curve
    py::array_t<double> compute_high_y_curve(py::array_t<double> betas, double z_a, double y) {
        auto beta_buf = betas.request();
        auto result = py::array_t<double>(beta_buf.size);
        auto result_buf = result.request();
        
        double* beta_ptr = static_cast<double*>(beta_buf.ptr);
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        double a = z_a;
        double denominator = 1.0 - 2.0*a;
        
        #pragma omp parallel for
        for (size_t i = 0; i < beta_buf.size; i++) {
            if (std::abs(denominator) < 1e-10) {
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            
            double beta = beta_ptr[i];
            double numerator = -4.0*a*(a-1.0)*y_effective*beta - 2.0*a*y_effective - 2.0*a*(2.0*a-1.0);
            result_ptr[i] = numerator/denominator;
        }
        
        return result;
    }

    // Compute alternate low expression
    py::array_t<double> compute_alternate_low_expr(py::array_t<double> betas, double z_a, double y) {
        auto beta_buf = betas.request();
        auto result = py::array_t<double>(beta_buf.size);
        auto result_buf = result.request();
        
        double* beta_ptr = static_cast<double*>(beta_buf.ptr);
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        #pragma omp parallel for
        for (size_t i = 0; i < beta_buf.size; i++) {
            double beta = beta_ptr[i];
            result_ptr[i] = (z_a * y_effective * beta * (z_a - 1.0) - 2.0*z_a*(1.0 - y_effective) - 2.0*z_a*z_a) / (2.0 + 2.0*z_a);
        }
        
        return result;
    }

    // Compute max k expression
    py::array_t<double> compute_max_k_expression(py::array_t<double> betas, double z_a, double y, int k_samples=1000) {
        auto beta_buf = betas.request();
        auto result = py::array_t<double>(beta_buf.size);
        auto result_buf = result.request();
        
        double* beta_ptr = static_cast<double*>(beta_buf.ptr);
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        double a = z_a;
        // Sample k values on a logarithmic scale
        std::vector<double> k_values(k_samples);
        for (int j = 0; j < k_samples; j++) {
            k_values[j] = std::pow(10.0, -3.0 + 6.0 * static_cast<double>(j) / (k_samples - 1));
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < beta_buf.size; i++) {
            double beta = beta_ptr[i];
            std::vector<double> values(k_samples);
            
            // Compute expression value for each k
            for (int j = 0; j < k_samples; j++) {
                double k = k_values[j];
                double numerator = y_effective*beta*(a-1.0)*k + (a*k+1.0)*((y_effective-1.0)*k-1.0);
                double denominator = (a*k+1.0)*(k*k+k);
                
                if (std::abs(denominator) < 1e-10) {
                    values[j] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    values[j] = numerator/denominator;
                }
            }
            
            // Find maximum value (excluding NaNs)
            double max_val = -std::numeric_limits<double>::infinity();
            bool has_valid = false;
            
            for (double val : values) {
                if (!std::isnan(val)) {
                    max_val = std::max(max_val, val);
                    has_valid = true;
                }
            }
            
            result_ptr[i] = has_valid ? max_val : std::numeric_limits<double>::quiet_NaN();
        }
        
        return result;
    }

    // Compute min t expression
    py::array_t<double> compute_min_t_expression(py::array_t<double> betas, double z_a, double y, int t_samples=1000) {
        auto beta_buf = betas.request();
        auto result = py::array_t<double>(beta_buf.size);
        auto result_buf = result.request();
        
        double* beta_ptr = static_cast<double*>(beta_buf.ptr);
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        double a = z_a;
        if (a <= 0) {
            for (size_t i = 0; i < beta_buf.size; i++) {
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
            }
            return result;
        }
        
        double lower_bound = -1.0/a + 1e-10;  // Avoid division by zero
        
        #pragma omp parallel for
        for (size_t i = 0; i < beta_buf.size; i++) {
            double beta = beta_ptr[i];
            
            // Sample t values
            std::vector<double> t_values(t_samples);
            for (int j = 0; j < t_samples; j++) {
                t_values[j] = lower_bound + (-1e-10 - lower_bound) * static_cast<double>(j) / (t_samples - 1);
            }
            
            std::vector<double> values(t_samples);
            
            // Compute expression value for each t
            for (int j = 0; j < t_samples; j++) {
                double t = t_values[j];
                double numerator = y_effective*beta*(a-1.0)*t + (a*t+1.0)*((y_effective-1.0)*t-1.0);
                double denominator = (a*t+1.0)*(t*t+t);
                
                if (std::abs(denominator) < 1e-10) {
                    values[j] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    values[j] = numerator/denominator;
                }
            }
            
            // Find minimum value (excluding NaNs)
            double min_val = std::numeric_limits<double>::infinity();
            bool has_valid = false;
            
            for (double val : values) {
                if (!std::isnan(val)) {
                    min_val = std::min(min_val, val);
                    has_valid = true;
                }
            }
            
            result_ptr[i] = has_valid ? min_val : std::numeric_limits<double>::quiet_NaN();
        }
        
        return result;
    }

    // Generate phase diagram
    py::array_t<int> generate_phase_diagram_cpp(
        double z_a, double y, double beta_min, double beta_max, 
        double z_min, double z_max, int beta_steps, int z_steps) {
        
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        // Create result array
        auto result = py::array_t<int>({z_steps, beta_steps});
        auto result_buf = result.request();
        int* result_ptr = static_cast<int*>(result_buf.ptr);
        
        // Create beta and z grids
        std::vector<double> beta_values(beta_steps);
        std::vector<double> z_values(z_steps);
        
        for (int i = 0; i < beta_steps; i++) {
            beta_values[i] = beta_min + (beta_max - beta_min) * static_cast<double>(i) / (beta_steps - 1);
        }
        
        for (int i = 0; i < z_steps; i++) {
            z_values[i] = z_min + (z_max - z_min) * static_cast<double>(i) / (z_steps - 1);
        }
        
        // Analyze roots for each (z, beta) point
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < z_steps; i++) {
            for (int j = 0; j < beta_steps; j++) {
                double z = z_values[i];
                double beta = beta_values[j];
                
                // Coefficients for cubic equation
                double a = z * z_a;
                double b = z * z_a + z + z_a - z_a*y_effective;
                double c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta);
                double d = 1.0;
                
                // Calculate discriminant
                double discriminant = 18*a*b*c*d - 27*a*a*d*d + b*b*c*c - 2*b*b*b*d - 9*a*c*c*c;
                
                // Set result based on sign of discriminant
                // 1 for all real roots (discriminant > 0), -1 for complex roots (discriminant < 0)
                result_ptr[i * beta_steps + j] = (discriminant > 0) ? 1 : -1;
            }
        }
        
        return result;
    }

    // Generate eigenvalue distribution
    std::tuple<py::array_t<double>, py::array_t<double>> 
    generate_eigenvalue_distribution_cpp(double beta, double y, double z_a, int n, int seed) {
        // Apply the condition for y
        double y_effective = y > 1.0 ? y : 1.0/y;
        
        // Set random seed
        std::mt19937 gen(seed);
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        
        // Compute dimension p based on aspect ratio y
        int p = static_cast<int>(y_effective * n);
        
        // Constructing T_n (Population / Shape Matrix)
        int k = static_cast<int>(std::floor(beta * p));
        
        // Create diagonal entries
        std::vector<double> diag_entries(p);
        std::fill_n(diag_entries.begin(), k, z_a);
        std::fill_n(diag_entries.begin() + k, p - k, 1.0);
        
        // Shuffle the diagonal entries
        std::shuffle(diag_entries.begin(), diag_entries.end(), gen);
        
        // Create T_n matrix
        MatrixXd T_n = MatrixXd::Zero(p, p);
        for (int i = 0; i < p; i++) {
            T_n(i, i) = diag_entries[i];
        }
        
        // Generate random data matrix X with standard normal entries
        MatrixXd X(p, n);
        for (int r = 0; r < p; r++) {
            for (int c = 0; c < n; c++) {
                X(r, c) = normal_dist(gen);
            }
        }
        
        // Compute sample covariance matrix S_n = (1/n) * XX^T
        MatrixXd S_n = (1.0 / n) * (X * X.transpose());
        
        // Compute B_n = S_n T_n
        MatrixXd B_n = S_n * T_n;
        
        // Compute eigenvalues
        SelfAdjointEigenSolver<MatrixXd> solver(B_n);
        VectorXd eigenvalues = solver.eigenvalues();
        
        // Return eigenvalues as numpy array
        auto result = py::array_t<double>(p);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        for (int i = 0; i < p; i++) {
            result_ptr[i] = eigenvalues(i);
        }
        
        // Create x grid for KDE estimation (done in Python)
        auto x_grid = py::array_t<double>(500);
        auto x_grid_buf = x_grid.request();
        double* x_grid_ptr = static_cast<double*>(x_grid_buf.ptr);
        
        double min_eig = eigenvalues.minCoeff();
        double max_eig = eigenvalues.maxCoeff();
        
        for (int i = 0; i < 500; i++) {
            x_grid_ptr[i] = min_eig + (max_eig - min_eig) * static_cast<double>(i) / 499.0;
        }
        
        return std::make_tuple(result, x_grid);
    }

    PYBIND11_MODULE(cubic_cpp, m) {
        m.doc() = "C++ accelerated cubic root analysis";
        
        // Expose all the C++ functions to Python
        m.def("discriminant_array", &discriminant_array, "Compute cubic discriminant for array of z values");
        m.def("find_discriminant_zeros", &find_discriminant_zeros, "Find zeros of the discriminant function");
        m.def("compute_eigenvalue_boundaries", &compute_eigenvalue_boundaries, "Compute eigenvalue boundaries");
        m.def("compute_cubic_roots_cpp", &compute_cubic_roots_cpp, "Compute cubic roots");
        m.def("compute_high_y_curve", &compute_high_y_curve, "Compute high y curve");
        m.def("compute_alternate_low_expr", &compute_alternate_low_expr, "Compute alternate low expression");
        m.def("compute_max_k_expression", &compute_max_k_expression, "Compute max k expression");
        m.def("compute_min_t_expression", &compute_min_t_expression, "Compute min t expression");
        m.def("generate_eigenvalue_distribution_cpp", &generate_eigenvalue_distribution_cpp, "Generate eigenvalue distribution");
        m.def("generate_phase_diagram_cpp", &generate_phase_diagram_cpp, "Generate phase diagram");
    }
    """
    
    progress.update(1, "Creating temporary directory for compilation")
    
    # Create a temporary directory to compile the C++ code
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Write C++ code to file
        progress.update(2, "Writing C++ code to temporary file")
        with open(os.path.join(tmpdirname, "cubic_cpp.cpp"), "w") as f:
            f.write(cpp_code)
        
        # Write setup.py for compiling with pybind11
        progress.update(3, "Creating setup.py file")
        setup_py = """
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cubic_cpp",
        ["cubic_cpp.cpp"],
        include_dirs=["/usr/include/eigen3"],
        extra_compile_args=["-fopenmp", "-O3", "-march=native", "-ffast-math"],
        extra_link_args=["-fopenmp"],
        cxx_std=17,
    ),
]

setup(
    name="cubic_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
        """
        
        with open(os.path.join(tmpdirname, "setup.py"), "w") as f:
            f.write(setup_py)
        
        # Compile the module
        progress.update(4, "Running compilation process")
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdirname,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get the compiled module path
            module_path = None
            for file in os.listdir(tmpdirname):
                if file.startswith("cubic_cpp") and file.endswith(".so"):
                    module_path = os.path.join(tmpdirname, file)
                    break
            
            if not module_path:
                progress.complete()
                return None, f"Failed to find compiled module. Output: {result.stdout}\n{result.stderr}"
            
            # Import the module
            progress.update(5, "Importing compiled module")
            spec = importlib.util.spec_from_file_location("cubic_cpp", module_path)
            cubic_cpp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cubic_cpp)
            
            progress.complete()
            return cubic_cpp, None
            
        except subprocess.CalledProcessError as e:
            progress.complete()
            return None, f"Compilation failed: {e.stdout}\n{e.stderr}"
        except Exception as e:
            progress.complete()
            return None, f"Error: {str(e)}"

# Try to compile the C++ module
cpp_module, cpp_error = compile_cpp_module()

# Display C++ status
with cpp_status_container:
    if cpp_module:
        st.success("✅ C++ acceleration module loaded successfully! Calculations will run faster.")
    else:
        st.warning("⚠️ C++ acceleration not available. Using Python fallback implementation.")
        if cpp_error:
            with st.expander("Error details"):
                st.code(cpp_error)

# ----- SymPy-based high-precision implementations -----

# Symbolic variables for the cubic discriminant
z_sym, beta_sym, z_a_sym, y_sym = sp.symbols("z beta z_a y", real=True, positive=True)

# Define coefficients a, b, c, d in terms of z_sym, beta_sym, z_a_sym, y_sym
a_sym = z_sym * z_a_sym
b_sym = z_sym * z_a_sym + z_sym + z_a_sym - z_a_sym*y_sym
c_sym = z_sym + z_a_sym + 1 - y_sym*(beta_sym*z_a_sym + 1 - beta_sym)
d_sym = 1

# Standard discriminant formula for cubic equations
# Δ = 18abcd - 27a²d² + b²c² - 2b³d - 9ac³
Delta_expr = (
    18*a_sym*b_sym*c_sym*d_sym - 
    27*a_sym**2*d_sym**2 + 
    b_sym**2*c_sym**2 - 
    2*b_sym**3*d_sym - 
    9*a_sym*c_sym**3
)

# Fast numeric function for the discriminant
discriminant_func = sp.lambdify((z_sym, beta_sym, z_a_sym, y_sym), Delta_expr, modules=[{"sqrt": np.sqrt}, "numpy"])


# Function to find zeros of the discriminant with SymPy precision
@st.cache_data(show_spinner=False)
def find_z_at_discriminant_zero(z_a, y, beta, z_min, z_max, steps):
    """
    Scan z in [z_min, z_max] for sign changes in the discriminant,
    and return approximated roots (where the discriminant is zero).
    Uses high-precision SymPy calculations.
    """
    # Create an advanced progress bar
    progress_key = f"find_zeros_{z_a}_{y}_{beta}_{z_min}_{z_max}_{steps}"
    progress = AdvancedProgressBar(progress_key, steps, "Finding discriminant zeros")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Create SymPy lambdified function with high precision
    mpmath_discriminant_func = sp.lambdify(
        (z_sym, beta_sym, z_a_sym, y_sym), 
        Delta_expr, 
        modules="mpmath"
    )
    
    # Create grid with high precision
    z_grid = np.linspace(z_min, z_max, steps)
    disc_vals = []
    
    # Calculate discriminant values with progress tracking
    for i, z in enumerate(z_grid):
        progress.update(i+1, f"Computing discriminant at z = {z:.4f}")
        
        # Use high precision mpmath for calculation
        disc_vals.append(float(mpmath_discriminant_func(z, beta, z_a, y_effective)))
    
    disc_vals = np.array(disc_vals)
    roots_found = []
    
    # Loop through and find sign changes
    for i in range(len(z_grid) - 1):
        f1, f2 = disc_vals[i], disc_vals[i+1]
        if np.isnan(f1) or np.isnan(f2):
            continue
            
        if abs(f1) < 1e-12:
            roots_found.append(z_grid[i])
        elif abs(f2) < 1e-12:
            roots_found.append(z_grid[i+1])
        elif f1 * f2 < 0:
            # Use high-precision binary search to refine root location
            zl, zr = mpmath.mpf(str(z_grid[i])), mpmath.mpf(str(z_grid[i+1]))
            fl = mpmath_discriminant_func(zl, beta, z_a, y_effective)
            fr = mpmath_discriminant_func(zr, beta, z_a, y_effective)
            
            # Binary search with high precision
            for _ in range(50):
                mid = (zl + zr) / 2
                fm = mpmath_discriminant_func(mid, beta, z_a, y_effective)
                
                if abs(fm) < 1e-20:
                    zl = zr = mid
                    break
                    
                if fm * fl > 0:
                    zl, fl = mid, fm
                else:
                    zr, fr = mid, fm
            
            # Convert back to float for NumPy compatibility
            roots_found.append(float(zl))
    
    progress.complete()
    return np.array(roots_found)


@st.cache_data(show_spinner=False)
def sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps):
    """
    For each beta in [0,1] (with beta_steps points), find the minimum and maximum z 
    for which the discriminant is zero. Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.find_discriminant_zeros(z_a, y, z_min, z_max, beta_steps, z_steps)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Create progress tracking
    progress_key = f"sweep_beta_{z_a}_{y}_{z_min}_{z_max}_{beta_steps}_{z_steps}"
    progress = AdvancedProgressBar(progress_key, beta_steps, "Computing discriminant zeros across β values")
    
    betas = np.linspace(0, 1, beta_steps)
    z_min_values = []
    z_max_values = []
    
    for i, b in enumerate(betas):
        progress.update(i+1, f"Processing β = {b:.4f}")
        roots = find_z_at_discriminant_zero(z_a, y, b, z_min, z_max, max(100, z_steps // beta_steps))
        if len(roots) == 0:
            z_min_values.append(np.nan)
            z_max_values.append(np.nan)
        else:
            z_min_values.append(np.min(roots))
            z_max_values.append(np.max(roots))
    
    progress.complete()
    return betas, np.array(z_min_values), np.array(z_max_values)


@st.cache_data(show_spinner=False)
def compute_eigenvalue_support_boundaries(z_a, y, beta_values, n_samples=100, seeds=5):
    """
    Compute the support boundaries of the eigenvalue distribution. 
    Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.compute_eigenvalue_boundaries(z_a, y, beta_values, n_samples, seeds)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Create progress tracking
    progress_key = f"eigenval_bound_{z_a}_{y}_{len(beta_values)}_{n_samples}_{seeds}"
    progress = AdvancedProgressBar(progress_key, len(beta_values), "Computing eigenvalue support boundaries")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    min_eigenvalues = np.zeros_like(beta_values)
    max_eigenvalues = np.zeros_like(beta_values)
    
    for i, beta in enumerate(beta_values):
        # Update progress
        progress.update(i+1, f"Processing β = {beta:.4f} (matrix size: {n_samples})")
            
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
    
    progress.complete()
    return min_eigenvalues, max_eigenvalues


@st.cache_data(show_spinner=False)
def compute_high_y_curve(betas, z_a, y):
    """
    Compute the "High y Expression" curve with high precision.
    Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.compute_high_y_curve(betas, z_a, y)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Create progress bar
    progress_key = f"high_y_{z_a}_{y}_{len(betas)}"
    progress = AdvancedProgressBar(progress_key, 3, "Computing high y expression")
    progress.update(1, "Setting up symbolic expression")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Use SymPy for higher precision
    a = z_a_sym
    denominator = 1 - 2*a
    numerator = -4*a*(a-1)*y_sym*beta_sym - 2*a*y_sym - 2*a*(2*a-1)
    expr = numerator / denominator
    
    progress.update(2, "Creating high-precision lambda function")
    
    # Convert to a high-precision numeric function
    func = sp.lambdify((beta_sym, z_a_sym, y_sym), expr, modules="mpmath")
    
    progress.update(3, "Computing values")
    
    # Check if denominator is near zero (division by zero case)
    denom_val = float(sp.N(denominator.subs(z_a_sym, z_a)))
    if abs(denom_val) < 1e-12:
        progress.complete()
        return np.full_like(betas, np.nan)
    
    # Compute values with high precision for each beta
    result = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        result[i] = float(func(beta, z_a, y_effective))
    
    progress.complete()
    return result


@st.cache_data(show_spinner=False)
def compute_alternate_low_expr(betas, z_a, y):
    """
    Compute the alternate low expression with high precision.
    Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.compute_alternate_low_expr(betas, z_a, y)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Create progress bar
    progress_key = f"low_expr_{z_a}_{y}_{len(betas)}"
    progress = AdvancedProgressBar(progress_key, 3, "Computing low y expression")
    progress.update(1, "Setting up symbolic expression")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Use SymPy for higher precision
    expr = (z_a_sym * y_sym * beta_sym * (z_a_sym - 1) - 
            2*z_a_sym*(1 - y_sym) - 2*z_a_sym**2) / (2 + 2*z_a_sym)
    
    progress.update(2, "Creating high-precision lambda function")
    
    # Convert to a high-precision numeric function
    func = sp.lambdify((beta_sym, z_a_sym, y_sym), expr, modules="mpmath")
    
    progress.update(3, "Computing values")
    
    # Compute values with high precision for each beta
    result = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        result[i] = float(func(beta, z_a, y_effective))
    
    progress.complete()
    return result


@st.cache_data(show_spinner=False)
def compute_max_k_expression(betas, z_a, y, k_samples=1000):
    """
    Compute max_{k ∈ (0,∞)} (y*beta*(a-1)*k + (a*k+1)*((y-1)*k-1)) / ((a*k+1)*(k^2+k))
    with high precision. Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.compute_max_k_expression(betas, z_a, y, k_samples)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Create progress bar
    progress_key = f"max_k_{z_a}_{y}_{len(betas)}_{k_samples}"
    progress = AdvancedProgressBar(progress_key, len(betas), "Computing max k expression")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Use SymPy for symbolic expression
    k_sym = sp.Symbol("k", positive=True)
    a = z_a_sym
    numerator = y_sym*beta_sym*(a-1)*k_sym + (a*k_sym+1)*((y_sym-1)*k_sym-1)
    denominator = (a*k_sym+1)*(k_sym**2+k_sym)
    expr = numerator / denominator
    
    # Convert to high-precision function
    func = sp.lambdify((k_sym, beta_sym, z_a_sym, y_sym), expr, modules="mpmath")
    
    # Sample k values on a logarithmic scale
    k_values = np.logspace(-3, 3, k_samples)
    
    max_vals = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        progress.update(i+1, f"Processing β = {beta:.4f}")
        
        values = np.zeros_like(k_values)
        for j, k in enumerate(k_values):
            try:
                val = float(func(k, beta, z_a, y_effective))
                if np.isfinite(val):
                    values[j] = val
                else:
                    values[j] = np.nan
            except (ZeroDivisionError, OverflowError):
                values[j] = np.nan
        
        valid_indices = ~np.isnan(values)
        if np.any(valid_indices):
            max_vals[i] = np.max(values[valid_indices])
        else:
            max_vals[i] = np.nan
    
    progress.complete()
    return max_vals


@st.cache_data(show_spinner=False)
def compute_min_t_expression(betas, z_a, y, t_samples=1000):
    """
    Compute min_{t ∈ (-1/a, 0)} (y*beta*(a-1)*t + (a*t+1)*((y-1)*t-1)) / ((a*t+1)*(t^2+t))
    with high precision. Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.compute_min_t_expression(betas, z_a, y, t_samples)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Create progress bar
    progress_key = f"min_t_{z_a}_{y}_{len(betas)}_{t_samples}"
    progress = AdvancedProgressBar(progress_key, len(betas), "Computing min t expression")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Use SymPy for symbolic expression
    t_sym = sp.Symbol("t")
    a = z_a_sym
    numerator = y_sym*beta_sym*(a-1)*t_sym + (a*t_sym+1)*((y_sym-1)*t_sym-1)
    denominator = (a*t_sym+1)*(t_sym**2+t_sym)
    expr = numerator / denominator
    
    # Convert to high-precision function
    func = sp.lambdify((t_sym, beta_sym, z_a_sym, y_sym), expr, modules="mpmath")
    
    a = z_a
    if a <= 0:
        progress.complete()
        return np.full_like(betas, np.nan)
        
    lower_bound = -1/a + 1e-10  # Avoid division by zero
    t_values = np.linspace(lower_bound, -1e-10, t_samples)
    
    min_vals = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        progress.update(i+1, f"Processing β = {beta:.4f}")
        
        values = np.zeros_like(t_values)
        for j, t in enumerate(t_values):
            try:
                val = float(func(t, beta, z_a, y_effective))
                if np.isfinite(val):
                    values[j] = val
                else:
                    values[j] = np.nan
            except (ZeroDivisionError, OverflowError):
                values[j] = np.nan
        
        valid_indices = ~np.isnan(values)
        if np.any(valid_indices):
            min_vals[i] = np.min(values[valid_indices])
        else:
            min_vals[i] = np.nan
    
    progress.complete()
    return min_vals


@st.cache_data(show_spinner=False)
def compute_derivatives(curve, betas):
    """Compute first and second derivatives of a curve using smooth spline interpolation."""
    from scipy.interpolate import CubicSpline
    
    # Filter out NaN values
    valid_idx = ~np.isnan(curve)
    if not np.any(valid_idx):
        return np.full_like(betas, np.nan), np.full_like(betas, np.nan)
    
    valid_betas = betas[valid_idx]
    valid_curve = curve[valid_idx]
    
    if len(valid_betas) < 4:  # Need at least 4 points for cubic spline
        # Fall back to numpy gradient
        d1 = np.gradient(curve, betas)
        d2 = np.gradient(d1, betas)
        return d1, d2
    
    # Create cubic spline for smoother derivatives
    cs = CubicSpline(valid_betas, valid_curve)
    
    # Evaluate first and second derivatives
    d1 = np.zeros_like(betas)
    d2 = np.zeros_like(betas)
    
    for i, beta in enumerate(betas):
        if np.isnan(curve[i]):
            d1[i] = np.nan
            d2[i] = np.nan
        else:
            d1[i] = cs(beta, 1)  # First derivative
            d2[i] = cs(beta, 2)  # Second derivative
    
    return d1, d2


def compute_all_derivatives(betas, z_mins, z_maxs, low_y_curve, high_y_curve, alt_low_expr, custom_curve1=None, custom_curve2=None):
    """Compute derivatives for all curves"""
    # Progress tracking
    progress_key = f"derivatives_{len(betas)}"
    progress = AdvancedProgressBar(progress_key, 7, "Computing derivatives")
    
    derivatives = {}
    
    # Upper z*(β)
    progress.update(1, "Computing upper z*(β) derivatives")
    derivatives['upper'] = compute_derivatives(z_maxs, betas)
    
    # Lower z*(β)
    progress.update(2, "Computing lower z*(β) derivatives")
    derivatives['lower'] = compute_derivatives(z_mins, betas)
    
    # Low y Expression (only if provided)
    if low_y_curve is not None:
        progress.update(3, "Computing low y expression derivatives")
        derivatives['low_y'] = compute_derivatives(low_y_curve, betas)
    else:
        progress.update(3, "Skipping low y expression (not provided)")
    
    # High y Expression
    if high_y_curve is not None:
        progress.update(4, "Computing high y expression derivatives")
        derivatives['high_y'] = compute_derivatives(high_y_curve, betas)
    else:
        progress.update(4, "Skipping high y expression (not provided)")
    
    # Alternate Low Expression
    if alt_low_expr is not None:
        progress.update(5, "Computing alternate low expression derivatives")
        derivatives['alt_low'] = compute_derivatives(alt_low_expr, betas)
    else:
        progress.update(5, "Skipping alternate low expression (not provided)")
    
    # Custom Expression 1 (if provided)
    if custom_curve1 is not None:
        progress.update(6, "Computing custom expression 1 derivatives")
        derivatives['custom1'] = compute_derivatives(custom_curve1, betas)
    else:
        progress.update(6, "Skipping custom expression 1 (not provided)")

    # Custom Expression 2 (if provided)
    if custom_curve2 is not None:
        progress.update(7, "Computing custom expression 2 derivatives")
        derivatives['custom2'] = compute_derivatives(custom_curve2, betas)
    else:
        progress.update(7, "Skipping custom expression 2 (not provided)")
    
    progress.complete()
    return derivatives


def compute_custom_expression(betas, z_a, y, s_num_expr, s_denom_expr, is_s_based=True):
    """
    Compute custom curve with high precision using SymPy.
    If is_s_based=True, compute using s substitution. Otherwise, compute direct z(β) expression.
    """
    # Progress tracking
    progress_key = f"custom_expr_{z_a}_{y}_{len(betas)}"
    progress = AdvancedProgressBar(progress_key, 4, "Computing custom expression")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Create SymPy symbols
    beta_sym, z_a_sym, y_sym = sp.symbols("beta z_a y", positive=True)
    local_dict = {"beta": beta_sym, "z_a": z_a_sym, "y": y_sym, "sp": sp}
    
    try:
        # Add sqrt support
        progress.update(1, "Parsing expression")
        s_num_expr = add_sqrt_support(s_num_expr)
        s_denom_expr = add_sqrt_support(s_denom_expr)
        
        num_expr = sp.sympify(s_num_expr, locals=local_dict)
        denom_expr = sp.sympify(s_denom_expr, locals=local_dict)
        
        if is_s_based:
            # Compute s and substitute into main expression
            progress.update(2, "Computing s-based expression")
            s_expr = num_expr / denom_expr
            a = z_a_sym
            numerator = y_sym*beta_sym*(z_a_sym-1)*s_expr + (a*s_expr+1)*((y_sym-1)*s_expr-1)
            denominator = (a*s_expr+1)*(s_expr**2 + s_expr)
            final_expr = numerator/denominator
        else:
            # Direct z(β) expression
            progress.update(2, "Computing direct z(β) expression")
            final_expr = num_expr / denom_expr
            
    except sp.SympifyError as e:
        progress.complete()
        st.error(f"Error parsing expressions: {e}")
        return np.full_like(betas, np.nan)
    
    progress.update(3, "Creating lambda function with mpmath precision")
    
    # Convert to high-precision numeric function
    final_func = sp.lambdify((beta_sym, z_a_sym, y_sym), final_expr, modules="mpmath")
    
    progress.update(4, "Evaluating expression for all β values")
    
    # Compute values for each beta
    result = np.zeros_like(betas)
    
    for i, beta in enumerate(betas):
        try:
            # Calculate with high precision
            val = final_func(beta, z_a, y_effective)
            # Convert to float for compatibility
            result[i] = float(val)
        except Exception as e:
            result[i] = np.nan
    
    progress.complete()
    return result


def compute_cubic_roots(z, beta, z_a, y):
    """
    Compute the roots of the cubic equation for given parameters with high precision.
    Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            return cpp_module.compute_cubic_roots_cpp(z, beta, z_a, y)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
    
    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Create a symbolic variable for the equation
    s = sp.Symbol('s')
    
    # Coefficients in the form as^3 + bs^2 + cs + d = 0
    a = z * z_a
    b = z * z_a + z + z_a - z_a*y_effective
    c = z + z_a + 1 - y_effective*(beta*z_a + 1 - beta)
    d = 1
    
    # Handle special cases
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:  # Linear case
            roots = np.array([-d/c, 0, 0], dtype=complex)
        else:  # Quadratic case
            # Use SymPy for higher precision
            quad_eq = b*s**2 + c*s + d
            symbolic_roots = sp.solve(quad_eq, s)
            
            # Convert to complex numbers with proper precision
            numerical_roots = []
            for root in symbolic_roots:
                high_prec_root = complex(float(sp.re(root).evalf(50)), float(sp.im(root).evalf(50)))
                numerical_roots.append(high_prec_root)
                
            # Pad to 3 roots
            while len(numerical_roots) < 3:
                numerical_roots.append(0j)
                
            roots = np.array(numerical_roots, dtype=complex)
        return roots
    
    try:
        # Create the cubic equation with high precision
        cubic_eq = a*s**3 + b*s**2 + c*s + d
        
        # Solve using SymPy's solver with high precision
        symbolic_roots = sp.solve(cubic_eq, s)
        
        # Convert to high-precision complex numbers
        numerical_roots = []
        for root in symbolic_roots:
            # Use SymPy's evalf with high precision (50 digits)
            high_prec_root = root.evalf(50)
            numerical_root = complex(float(sp.re(high_prec_root)), float(sp.im(high_prec_root)))
            numerical_roots.append(numerical_root)
        
        # If we got fewer than 3 roots (due to multiplicity), pad with zeros
        while len(numerical_roots) < 3:
            numerical_roots.append(0j)
            
        return np.array(numerical_roots, dtype=complex)
        
    except Exception as e:
        # Fallback to numpy if SymPy has issues
        coeffs = [a, b, c, d]
        return np.roots(coeffs)


def track_roots_consistently(z_values, all_roots):
    """
    Ensure consistent tracking of roots across z values by minimizing discontinuity.
    """
    n_points = len(z_values)
    n_roots = all_roots[0].shape[0]
    tracked_roots = np.zeros((n_points, n_roots), dtype=complex)
    tracked_roots[0] = all_roots[0]
    
    for i in range(1, n_points):
        prev_roots = tracked_roots[i-1]
        current_roots = all_roots[i]
        
        # For each previous root, find the closest current root
        assigned = np.zeros(n_roots, dtype=bool)
        assignments = np.zeros(n_roots, dtype=int)
        
        for j in range(n_roots):
            distances = np.abs(current_roots - prev_roots[j])
            
            # Find the closest unassigned root
            while True:
                best_idx = np.argmin(distances)
                if not assigned[best_idx]:
                    assignments[j] = best_idx
                    assigned[best_idx] = True
                    break
                else:
                    # Mark as infinite distance and try again
                    distances[best_idx] = np.inf
                    
                # Safety check if all are assigned (shouldn't happen)
                if np.all(distances == np.inf):
                    assignments[j] = j  # Default to same index
                    break
        
        # Reorder current roots based on assignments
        tracked_roots[i] = current_roots[assignments]
    
    return tracked_roots


def generate_cubic_discriminant(z, beta, z_a, y_effective):
    """
    Calculate the cubic discriminant with high precision using the standard formula.
    For a cubic ax^3 + bx^2 + cx + d: Δ = 18abcd - 27a^2d^2 + b^2c^2 - 2b^3d - 9ac^3
    """
    # Use SymPy for high precision calculation
    a = mpmath.mpf(str(z * z_a))
    b = mpmath.mpf(str(z * z_a + z + z_a - z_a*y_effective))
    c = mpmath.mpf(str(z + z_a + 1 - y_effective*(beta*z_a + 1 - beta)))
    d = mpmath.mpf("1.0")
    
    # Standard formula for cubic discriminant with high precision
    discriminant = (18*a*b*c*d - 27*a**2*d**2 + b**2*c**2 - 2*b**3*d - 9*a*c**3)
    return float(discriminant)


@st.cache_data(show_spinner=False)
def generate_root_plots(beta, y, z_a, z_min, z_max, n_points):
    """
    Generate Im(s) and Re(s) vs. z plots with improved accuracy using SymPy.
    """
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        st.error("Invalid input parameters.")
        return None, None, None

    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Create progress bar
    progress_key = f"root_plots_{beta}_{y}_{z_a}_{z_min}_{z_max}_{n_points}"
    progress = AdvancedProgressBar(progress_key, n_points + 1, "Computing cubic roots vs. z")
    
    z_points = np.linspace(z_min, z_max, n_points)
    
    # Collect all roots first
    all_roots = []
    discriminants = []
    
    for i, z in enumerate(z_points):
        # Update progress
        progress.update(i+1, f"Computing roots for z = {z:.3f}")
        
        # Calculate roots using high precision
        roots = compute_cubic_roots(z, beta, z_a, y)
        
        # Initial sorting to help with tracking
        roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
        all_roots.append(roots)
        
        # Calculate discriminant with high precision
        disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
        discriminants.append(disc)
    
    # Update for final tracking step
    progress.update(n_points+1, "Tracking roots across z values")
    
    all_roots = np.array(all_roots)
    discriminants = np.array(discriminants)
    
    # Track roots consistently across z values
    tracked_roots = track_roots_consistently(z_points, all_roots)
    
    progress.complete()
    
    # Extract imaginary and real parts
    ims = np.imag(tracked_roots)
    res = np.real(tracked_roots)
    
    # Create figure for imaginary parts
    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=z_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    disc_zeros = []
    for i in range(len(discriminants)-1):
        if discriminants[i] * discriminants[i+1] <= 0:  # Sign change
            zero_pos = z_points[i] + (z_points[i+1] - z_points[i]) * (0 - discriminants[i]) / (discriminants[i+1] - discriminants[i])
            disc_zeros.append(zero_pos)
            fig_im.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_im.update_layout(title=f"Im{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Im{s}", hovermode="x unified")

    # Create figure for real parts
    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=z_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    for zero_pos in disc_zeros:
        fig_re.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_re.update_layout(title=f"Re{{s}} vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="z", yaxis_title="Re{s}", hovermode="x unified")
    
    # Create discriminant plot
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=z_points, y=discriminants, mode="lines", 
                                 name="Cubic Discriminant", line=dict(color="black", width=2)))
    fig_disc.add_hline(y=0, line=dict(color="red", width=1, dash="dash"))
    
    fig_disc.update_layout(title=f"Cubic Discriminant vs. z (β={beta:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                          xaxis_title="z", yaxis_title="Discriminant", hovermode="x unified")
    
    return fig_im, fig_re, fig_disc


@st.cache_data(show_spinner=False)
def generate_roots_vs_beta_plots(z, y, z_a, beta_min, beta_max, n_points):
    """
    Generate Im(s) and Re(s) vs. β plots with improved accuracy using SymPy.
    """
    if z_a <= 0 or y <= 0 or beta_min >= beta_max:
        st.error("Invalid input parameters.")
        return None, None, None

    # Apply the condition for y
    y_effective = y if y > 1 else 1/y
    
    # Create progress bar
    progress_key = f"roots_beta_{z}_{y}_{z_a}_{beta_min}_{beta_max}_{n_points}"
    progress = AdvancedProgressBar(progress_key, n_points + 1, "Computing cubic roots vs. β")
    
    beta_points = np.linspace(beta_min, beta_max, n_points)
    
    # Collect all roots first
    all_roots = []
    discriminants = []
    
    for i, beta in enumerate(beta_points):
        # Update progress
        progress.update(i+1, f"Computing roots for β = {beta:.3f}")
        
        # Calculate roots using high precision
        roots = compute_cubic_roots(z, beta, z_a, y)
        
        # Initial sorting to help with tracking
        roots = sorted(roots, key=lambda x: (abs(x.imag), x.real))
        all_roots.append(roots)
        
        # Calculate discriminant with high precision
        disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
        discriminants.append(disc)
    
    # Update for final tracking step
    progress.update(n_points+1, "Tracking roots across β values")
    
    all_roots = np.array(all_roots)
    discriminants = np.array(discriminants)
    
    # Track roots consistently across beta values
    tracked_roots = track_roots_consistently(beta_points, all_roots)
    
    progress.complete()
    
    # Extract imaginary and real parts
    ims = np.imag(tracked_roots)
    res = np.real(tracked_roots)
    
    # Create figure for imaginary parts
    fig_im = go.Figure()
    for i in range(3):
        fig_im.add_trace(go.Scatter(x=beta_points, y=ims[:, i], mode="lines", name=f"Im{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    disc_zeros = []
    for i in range(len(discriminants)-1):
        if discriminants[i] * discriminants[i+1] <= 0:  # Sign change
            zero_pos = beta_points[i] + (beta_points[i+1] - beta_points[i]) * (0 - discriminants[i]) / (discriminants[i+1] - discriminants[i])
            disc_zeros.append(zero_pos)
            fig_im.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_im.update_layout(title=f"Im{{s}} vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="β", yaxis_title="Im{s}", hovermode="x unified")

    # Create figure for real parts
    fig_re = go.Figure()
    for i in range(3):
        fig_re.add_trace(go.Scatter(x=beta_points, y=res[:, i], mode="lines", name=f"Re{{s{i+1}}}",
                                    line=dict(width=2)))
    
    # Add vertical lines at discriminant zero crossings
    for zero_pos in disc_zeros:
        fig_re.add_vline(x=zero_pos, line=dict(color="red", width=1, dash="dash"))
    
    fig_re.update_layout(title=f"Re{{s}} vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                         xaxis_title="β", yaxis_title="Re{s}", hovermode="x unified")
    
    # Create discriminant plot
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=beta_points, y=discriminants, mode="lines", 
                                 name="Cubic Discriminant", line=dict(color="black", width=2)))
    fig_disc.add_hline(y=0, line=dict(color="red", width=1, dash="dash"))
    
    fig_disc.update_layout(title=f"Cubic Discriminant vs. β (z={z:.3f}, y={y:.3f}, z_a={z_a:.3f})",
                          xaxis_title="β", yaxis_title="Discriminant", hovermode="x unified")
    
    return fig_im, fig_re, fig_disc


@st.cache_data(show_spinner=False)
def generate_phase_diagram(z_a, y, beta_min=0.0, beta_max=1.0, z_min=-10.0, z_max=10.0, 
                          beta_steps=100, z_steps=100):
    """
    Generate a phase diagram showing regions of complex and real roots.
    Uses C++ acceleration if available.
    """
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            phase_map = cpp_module.generate_phase_diagram_cpp(z_a, y, beta_min, beta_max, z_min, z_max, beta_steps, z_steps)
            beta_values = np.linspace(beta_min, beta_max, beta_steps)
            z_values = np.linspace(z_min, z_max, z_steps)
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
            # Fall back to Python implementation below
            phase_map = None
    else:
        phase_map = None
    
    # If C++ failed or is not available, use Python implementation
    if phase_map is None:
        # Create progress tracking
        progress_key = f"phase_diagram_{z_a}_{y}_{beta_steps}_{z_steps}"
        progress = AdvancedProgressBar(progress_key, z_steps, "Generating phase diagram")
        
        # Apply the condition for y
        y_effective = y if y > 1 else 1/y
        
        beta_values = np.linspace(beta_min, beta_max, beta_steps)
        z_values = np.linspace(z_min, z_max, z_steps)
        
        # Initialize phase map
        phase_map = np.zeros((z_steps, beta_steps))
        
        for i, z in enumerate(z_values):
            # Update progress
            progress.update(i+1, f"Analyzing phase at z = {z:.2f} ({i+1}/{len(z_values)})")
            
            for j, beta in enumerate(beta_values):
                # Use high-precision discriminant calculation
                disc = generate_cubic_discriminant(z, beta, z_a, y_effective)
                
                # Set result based on sign of discriminant
                # 1 for all real roots (discriminant > 0), -1 for complex roots (discriminant < 0)
                phase_map[i, j] = 1 if disc > 0 else -1
        
        progress.complete()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=phase_map,
        x=beta_values,
        y=z_values,
        colorscale=[[0, 'blue'], [0.5, 'white'], [1.0, 'red']],
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(
            title="Root Type",
            tickvals=[-1, 1],
            ticktext=["Complex Roots", "All Real Roots"]
        )
    ))
    
    fig.update_layout(
        title=f"Phase Diagram: Root Structure (y={y:.3f}, z_a={z_a:.3f})",
        xaxis_title="β",
        yaxis_title="z",
        hovermode="closest"
    )
    
    return fig


@st.cache_data(show_spinner=False)
def generate_eigenvalue_distribution(beta, y, z_a, n=1000, seed=42):
    """
    Generate the eigenvalue distribution of B_n = S_n T_n as n→∞
    Uses C++ acceleration if available.
    """
    # Create progress tracking
    progress_key = f"eigenval_dist_{beta}_{y}_{z_a}_{n}_{seed}"
    progress = AdvancedProgressBar(progress_key, 7, "Generating eigenvalue distribution")
    
    # Try to use C++ acceleration if available
    if cpp_module is not None:
        try:
            progress.update(1, "Using C++ accelerated eigenvalue calculation")
            eigenvalues, x_vals = cpp_module.generate_eigenvalue_distribution_cpp(beta, y, z_a, n, seed)
            progress.update(6, "Eigenvalues computed successfully")
        except Exception as e:
            st.warning(f"C++ acceleration failed, falling back to Python: {str(e)}")
            # Fall back to Python implementation below
            eigenvalues = None
    else:
        eigenvalues = None
    
    # If C++ failed or is not available, use Python implementation
    if eigenvalues is None:
        # Apply the condition for y
        y_effective = y if y > 1 else 1/y
        
        # Set random seed
        progress.update(1, "Setting up random seed")
        np.random.seed(seed)
        
        # Compute dimension p based on aspect ratio y
        progress.update(2, "Initializing matrices")
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
        progress.update(3, "Generating random data matrix")
        X = np.random.randn(p, n)
        
        # Compute the sample covariance matrix S_n = (1/n) * XX^T
        progress.update(4, "Computing sample covariance matrix")
        S_n = (1 / n) * (X @ X.T)
        
        # Compute B_n = S_n T_n
        progress.update(5, "Computing B_n matrix")
        B_n = S_n @ T_n
        
        # Compute eigenvalues of B_n
        progress.update(6, "Computing eigenvalues")
        eigenvalues = np.linalg.eigvalsh(B_n)
        
        # Create x grid for KDE
        x_vals = np.linspace(min(eigenvalues), max(eigenvalues), 500)
    
    # Use KDE to compute a smooth density estimate
    progress.update(7, "Computing kernel density estimate")
    kde = gaussian_kde(eigenvalues)
    kde_vals = kde(x_vals)
    
    progress.complete()
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(x=eigenvalues, histnorm='probability density', 
                              name="Histogram", marker=dict(color='blue', opacity=0.6)))
    
    # Add KDE trace
    fig.add_trace(go.Scatter(x=x_vals, y=kde_vals, mode="lines", 
                            name="KDE", line=dict(color='red', width=2)))
    
    fig.update_layout(
        title=f"Eigenvalue Distribution for B_n = S_n T_n (y={y:.1f}, β={beta:.2f}, a={z_a:.1f})",
        xaxis_title="Eigenvalue",
        yaxis_title="Density",
        hovermode="closest",
        showlegend=True
    )
    
    return fig, eigenvalues


@st.cache_data(show_spinner=False)
def analyze_complex_root_structure(beta_values, z, z_a, y):
    """
    Analyze when the cubic equation switches between having all real roots
    and having a complex conjugate pair plus one real root.
    
    Returns:
    - transition_points: beta values where the root structure changes
    - structure_types: list indicating whether each interval has all real roots or complex roots
    """
    # Create progress tracking
    progress_key = f"root_struct_{z}_{z_a}_{y}_{len(beta_values)}"
    progress = AdvancedProgressBar(progress_key, len(beta_values), "Analyzing root structure")
    
    transition_points = []
    structure_types = []
    
    previous_type = None
    
    for i, beta in enumerate(beta_values):
        progress.update(i+1, f"Analyzing root structure at β = {beta:.4f}")
        
        # Calculate roots with high precision
        roots = compute_cubic_roots(z, beta, z_a, y)
        
        # Check if all roots are real (imaginary parts close to zero)
        is_all_real = all(abs(root.imag) < 1e-10 for root in roots)
        
        current_type = "real" if is_all_real else "complex"
        
        if previous_type is not None and current_type != previous_type:
            # Found a transition point
            transition_points.append(beta)
            structure_types.append(previous_type)
        
        previous_type = current_type
    
    # Add the final interval type
    if previous_type is not None:
        structure_types.append(previous_type)
    
    progress.complete()
    return transition_points, structure_types


@st.cache_data(show_spinner=False)
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
    """
    Generate z vs beta plot with high precision calculations.
    """
    # Create main progress tracking
    progress_key = f"z_vs_beta_{z_a}_{y}_{beta_steps}"
    progress = AdvancedProgressBar(progress_key, 5, "Computing z*(β) curves")
    
    if z_a <= 0 or y <= 0 or z_min >= z_max:
        progress.complete()
        st.error("Invalid input parameters.")
        return None

    progress.update(1, "Creating β grid")
    betas = np.linspace(0, 1, beta_steps)
    
    if use_eigenvalue_method:
        # Use the eigenvalue method to compute boundaries
        progress.update(2, "Computing eigenvalue support boundaries")
        min_eigs, max_eigs = compute_eigenvalue_support_boundaries(z_a, y, betas, n_samples, seeds)
        z_mins, z_maxs = min_eigs, max_eigs
    else:
        # Use the original discriminant method
        progress.update(2, "Computing discriminant zeros")
        betas, z_mins, z_maxs = sweep_beta_and_find_z_bounds(z_a, y, z_min, z_max, beta_steps, z_steps)
    
    progress.update(3, "Computing additional curves")
    
    # Compute additional curves
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
        progress.update(4, "Computing derivatives")
        derivatives = compute_all_derivatives(betas, z_mins, z_maxs, None, high_y_curve, 
                                           alt_low_expr, custom_curve1, custom_curve2)
        # Calculate derivatives for max_k and min_t curves if they exist
        if show_max_k and max_k_curve is not None:
            max_k_derivatives = compute_derivatives(max_k_curve, betas)
        if show_min_t and min_t_curve is not None:
            min_t_derivatives = compute_derivatives(min_t_curve, betas)
    
    progress.update(5, "Creating plot")

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
        if show_max_k and max_k_curve is not None and 'max_k_derivatives' in locals():
            fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[0], mode="lines", 
                                    name="Max k d/dβ", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=betas, y=max_k_derivatives[1], mode="lines", 
                                    name="Max k d²/dβ²", line=dict(color='red', dash='dot')))
        
        if show_min_t and min_t_curve is not None and 'min_t_derivatives' in locals():
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
    
    progress.complete()
    return fig


# ----------------- Streamlit UI -----------------

# Create tab containers that won't be refreshed
if 'tab_placeholders' not in st.session_state:
    st.session_state.tab_placeholders = st.tabs(["z*(β) Curves", "Complex Root Analysis", "Differential Analysis"])

# ----- Tab 1: z*(β) Curves -----
with st.session_state.tab_placeholders[0]:
    # Create persistent containers for this tab
    if 'tab1_header' not in st.session_state:
        st.session_state.tab1_header = st.empty()
        st.session_state.tab1_col1 = st.empty()
        st.session_state.tab1_col2 = st.empty()
        st.session_state.tab1_col3 = st.empty()
    
    # Fill header
    with st.session_state.tab1_header:
        st.header("Eigenvalue Support Boundaries")
    
    # Create column layout
    with st.session_state.tab1_col1:
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

    # Compute button container
    with st.session_state.tab1_col2:
        compute_button = st.button("Compute Curves", key="tab1_button")

    # Results container
    with st.session_state.tab1_col3:
        result_container = st.empty()

    # Compute if button pressed
    if compute_button:
        with result_container:
            plot_container = st.empty()
            explanation_container = st.empty()
            
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
                plot_container.plotly_chart(fig, use_container_width=True)
                
                # Curve explanations in collapsed expander
                with explanation_container.expander("Curve Explanations", expanded=False):
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


# ----- Tab 2: Complex Root Analysis -----
with st.session_state.tab_placeholders[1]:
    # Create persistent containers for this tab
    if 'tab2_header' not in st.session_state:
        st.session_state.tab2_header = st.empty()
        st.session_state.tab2_subtabs = st.empty()
    
    # Fill header
    with st.session_state.tab2_header:
        st.header("Complex Root Analysis")
    
    # Create tabs within the page for different plots
    with st.session_state.tab2_subtabs:
        plot_tabs = st.tabs(["Im{s} vs. z", "Im{s} vs. β", "Phase Diagram", "Eigenvalue Distribution"])
        
        # Tab for Im{s} vs. z plot
        with plot_tabs[0]:
            if 'tab2_z_col1' not in st.session_state:
                st.session_state.tab2_z_col1 = st.empty()
                st.session_state.tab2_z_col2 = st.empty()
            
            with st.session_state.tab2_z_col1:
                col1, col2 = st.columns([1, 2])
                with col1:
                    beta_z = st.number_input("β", value=0.5, min_value=0.0, max_value=1.0, key="beta_tab2_z")
                    y_z = st.number_input("y", value=1.0, key="y_tab2_z")
                    z_a_z = st.number_input("z_a", value=1.0, key="z_a_tab2_z")
                    z_min_z = st.number_input("z_min", value=-10.0, key="z_min_tab2_z")
                    z_max_z = st.number_input("z_max", value=10.0, key="z_max_tab2_z")
                    with st.expander("Resolution Settings", expanded=False):
                        z_points = st.slider("z grid points", min_value=100, max_value=2000, value=500, step=100, key="z_points_z")
                    compute_button_z = st.button("Compute Complex Roots vs. z", key="tab2_button_z")
            
            with st.session_state.tab2_z_col2:
                result_container_z = st.empty()
            
            if compute_button_z:
                with result_container_z:
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
    
        # Tab for Im{s} vs. β plot
        with plot_tabs[1]:
            if 'tab2_beta_col1' not in st.session_state:
                st.session_state.tab2_beta_col1 = st.empty()
                st.session_state.tab2_beta_col2 = st.empty()
            
            with st.session_state.tab2_beta_col1:
                col1, col2 = st.columns([1, 2])
                with col1:
                    z_beta = st.number_input("z", value=1.0, key="z_tab2_beta")
                    y_beta = st.number_input("y", value=1.0, key="y_tab2_beta")
                    z_a_beta = st.number_input("z_a", value=1.0, key="z_a_tab2_beta")
                    beta_min = st.number_input("β_min", value=0.0, min_value=0.0, max_value=1.0, key="beta_min_tab2")
                    beta_max = st.number_input("β_max", value=1.0, min_value=0.0, max_value=1.0, key="beta_max_tab2")
                    with st.expander("Resolution Settings", expanded=False):
                        beta_points = st.slider("β grid points", min_value=100, max_value=1000, value=500, step=100, key="beta_points")
                    compute_button_beta = st.button("Compute Complex Roots vs. β", key="tab2_button_beta")
            
            with st.session_state.tab2_beta_col2:
                result_container_beta = st.empty()
            
            if compute_button_beta:
                with result_container_beta:
                    fig_im_beta, fig_re_beta, fig_disc = generate_roots_vs_beta_plots(
                        z_beta, y_beta, z_a_beta, beta_min, beta_max, beta_points)
                    
                    if fig_im_beta is not None and fig_re_beta is not None and fig_disc is not None:
                        st.plotly_chart(fig_im_beta, use_container_width=True)
                        st.plotly_chart(fig_re_beta, use_container_width=True)
                        st.plotly_chart(fig_disc, use_container_width=True)
                        
                        # Add analysis of transition points
                        transition_points, structure_types = analyze_complex_root_structure(
                            np.linspace(beta_min, beta_max, beta_points), z_beta, z_a_beta, y_beta)
                        
                        if transition_points:
                            st.subheader("Root Structure Transition Points")
                            for i, beta in enumerate(transition_points):
                                prev_type = structure_types[i]
                                next_type = structure_types[i+1] if i+1 < len(structure_types) else "unknown"
                                st.markdown(f"- At β = {beta:.6f}: Transition from {prev_type} roots to {next_type} roots")
                        else:
                            st.info("No transitions detected in root structure across this β range.")
                        
                        # Explanation
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
        
        # Tab for Phase Diagram
        with plot_tabs[2]:
            if 'tab2_phase_col1' not in st.session_state:
                st.session_state.tab2_phase_col1 = st.empty()
                st.session_state.tab2_phase_col2 = st.empty()
            
            with st.session_state.tab2_phase_col1:
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
                    
                    compute_button_phase = st.button("Generate Phase Diagram", key="tab2_button_phase")
            
            with st.session_state.tab2_phase_col2:
                result_container_phase = st.empty()
            
            if compute_button_phase:
                with result_container_phase:
                    st.info("Generating phase diagram. This may take a while depending on resolution...")
                    fig_phase = generate_phase_diagram(
                        z_a_phase, y_phase, beta_min_phase, beta_max_phase, z_min_phase, z_max_phase, 
                        beta_steps_phase, z_steps_phase)
                    
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
    
        # Eigenvalue distribution tab
        with plot_tabs[3]:
            if 'tab2_eigen_col1' not in st.session_state:
                st.session_state.tab2_eigen_col1 = st.empty()
                st.session_state.tab2_eigen_col2 = st.empty()
            
            with st.session_state.tab2_eigen_col1:
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
                    
                    # Add comparison option
                    show_theoretical = st.checkbox("Show theoretical boundaries", value=True)
                    show_empirical_stats = st.checkbox("Show empirical statistics", value=True)
                    
                    compute_button_eigen = st.button("Generate Eigenvalue Distribution", key="tab2_eigen_button")
            
            with st.session_state.tab2_eigen_col2:
                result_container_eigen = st.empty()
    
            if compute_button_eigen:
                with result_container_eigen:
                    # Generate the eigenvalue distribution
                    fig_eigen, eigenvalues = generate_eigenvalue_distribution(beta_eigen, y_eigen, z_a_eigen, n=n_samples, seed=sim_seed)
                    
                    # If requested, compute and add theoretical boundaries
                    if show_theoretical:
                        # Calculate min and max eigenvalues using the support boundary functions
                        betas = np.array([beta_eigen])
                        min_eig, max_eig = compute_eigenvalue_support_boundaries(z_a_eigen, y_eigen, betas, n_samples=n_samples, seeds=5)
                        
                        # Add vertical lines for boundaries
                        fig_eigen.add_vline(
                            x=min_eig[0], 
                            line=dict(color="red", width=2, dash="dash"),
                            annotation_text="Min theoretical",
                            annotation_position="top right"
                        )
                        fig_eigen.add_vline(
                            x=max_eig[0], 
                            line=dict(color="red", width=2, dash="dash"),
                            annotation_text="Max theoretical",
                            annotation_position="top left"
                        )
                    
                    # Display the plot
                    st.plotly_chart(fig_eigen, use_container_width=True)
                    
                    # Add comparison of empirical vs theoretical bounds
                    if show_theoretical and show_empirical_stats:
                        empirical_min = eigenvalues.min()
                        empirical_max = eigenvalues.max()
                        
                        st.markdown("### Comparison of Empirical vs Theoretical Bounds")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Theoretical Min", f"{min_eig[0]:.6f}")
                            st.metric("Theoretical Max", f"{max_eig[0]:.6f}")
                            st.metric("Theoretical Width", f"{max_eig[0] - min_eig[0]:.6f}")
                        with col2:
                            st.metric("Empirical Min", f"{empirical_min:.6f}")
                            st.metric("Empirical Max", f"{empirical_max:.6f}")
                            st.metric("Empirical Width", f"{empirical_max - empirical_min:.6f}")
                        with col3:
                            st.metric("Min Difference", f"{empirical_min - min_eig[0]:.6f}")
                            st.metric("Max Difference", f"{empirical_max - max_eig[0]:.6f}")
                            st.metric("Width Difference", f"{(empirical_max - empirical_min) - (max_eig[0] - min_eig[0]):.6f}")
                    
                    # Display additional statistics
                    if show_empirical_stats:
                        st.markdown("### Eigenvalue Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean", f"{np.mean(eigenvalues):.6f}")
                            st.metric("Median", f"{np.median(eigenvalues):.6f}")
                        with col2:
                            st.metric("Standard Deviation", f"{np.std(eigenvalues):.6f}")
                            st.metric("Interquartile Range", f"{np.percentile(eigenvalues, 75) - np.percentile(eigenvalues, 25):.6f}")


# ----- Tab 3: Differential Analysis -----
with st.session_state.tab_placeholders[2]:
    # Create persistent containers for this tab
    if 'tab3_header' not in st.session_state:
        st.session_state.tab3_header = st.empty()
        st.session_state.tab3_col1 = st.empty()
        st.session_state.tab3_col2 = st.empty()
    
    # Fill header
    with st.session_state.tab3_header:
        st.header("Differential Analysis vs. β")
        with st.expander("Description", expanded=False):
            st.markdown("This page shows the difference between the Upper (blue) and Lower (lightblue) z*(β) curves, along with their first and second derivatives with respect to β.")
    
    # Create column layout
    with st.session_state.tab3_col1:
        col1, col2 = st.columns([1, 2])
        with col1:
            z_a_diff = st.number_input("z_a", value=1.0, key="z_a_diff")
            y_diff = st.number_input("y", value=1.0, key="y_diff")
            z_min_diff = st.number_input("z_min", value=-10.0, key="z_min_diff")
            z_max_diff = st.number_input("z_max", value=10.0, key="z_max_diff")
            
            diff_method_type = st.radio(
                "Boundary Calculation Method",
                ["Eigenvalue Method", "Discriminant Method"],
                index=0,
                key="diff_method_type"
            )
            
            with st.expander("Resolution Settings", expanded=False):
                if diff_method_type == "Eigenvalue Method":
                    beta_steps_diff = st.slider("β steps", min_value=21, max_value=101, value=51, step=10, 
                                             key="beta_steps_diff_eigen")
                    diff_n_samples = st.slider("Matrix size (n)", min_value=100, max_value=2000, value=1000, 
                                            step=100, key="diff_n_samples")
                    diff_seeds = st.slider("Number of seeds", min_value=1, max_value=10, value=5, step=1,
                                         key="diff_seeds")
                else:
                    beta_steps_diff = st.slider("β steps", min_value=51, max_value=501, value=201, step=50, 
                                             key="beta_steps_diff")
                    z_steps_diff = st.slider("z grid steps", min_value=1000, max_value=100000, value=50000, 
                                          step=1000, key="z_steps_diff")
            
            # Add options for curve selection
            st.subheader("Curves to Analyze")
            analyze_upper_lower = st.checkbox("Upper-Lower Difference", value=True)
            analyze_high_y = st.checkbox("High y Expression", value=False)
            analyze_alt_low = st.checkbox("Low y Expression", value=False)
            
            compute_button_diff = st.button("Compute Differentials", key="tab3_button")
    
    # Results container
    with st.session_state.tab3_col2:
        result_container_diff = st.empty()

    # Compute if button pressed
    if compute_button_diff:
        with result_container_diff:
            use_eigenvalue_method_diff = (diff_method_type == "Eigenvalue Method")
            
            # Create a progress tracker
            progress_key = f"diff_analysis_{z_a_diff}_{y_diff}"
            progress = AdvancedProgressBar(progress_key, 5, "Computing differential analysis")
            
            progress.update(1, "Setting up β grid")
            if use_eigenvalue_method_diff:
                betas_diff = np.linspace(0, 1, beta_steps_diff)
                progress.update(2, "Computing eigenvalue support boundaries")
                lower_vals, upper_vals = compute_eigenvalue_support_boundaries(
                    z_a_diff, y_diff, betas_diff, diff_n_samples, diff_seeds)
            else:
                progress.update(2, "Computing discriminant zeros")
                betas_diff, lower_vals, upper_vals = sweep_beta_and_find_z_bounds(
                    z_a_diff, y_diff, z_min_diff, z_max_diff, beta_steps_diff, z_steps_diff)
            
            # Create figure
            progress.update(3, "Creating plot")
            fig_diff = go.Figure()
            
            progress.update(4, "Computing derivatives")
            if analyze_upper_lower:
                diff_curve = upper_vals - lower_vals
                d1 = np.gradient(diff_curve, betas_diff)
                d2 = np.gradient(d1, betas_diff)
                
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=diff_curve, mode="lines", 
                                name="Upper-Lower Difference", line=dict(color="magenta", width=2)))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                name="Upper-Lower d/dβ", line=dict(color="magenta", dash='dash')))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                name="Upper-Lower d²/dβ²", line=dict(color="magenta", dash='dot')))

            if analyze_high_y:
                high_y_curve = compute_high_y_curve(betas_diff, z_a_diff, y_diff)
                d1 = np.gradient(high_y_curve, betas_diff)
                d2 = np.gradient(d1, betas_diff)
                
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=high_y_curve, mode="lines", 
                                name="High y", line=dict(color="green", width=2)))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                name="High y d/dβ", line=dict(color="green", dash='dash')))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                name="High y d²/dβ²", line=dict(color="green", dash='dot')))

            if analyze_alt_low:
                alt_low_curve = compute_alternate_low_expr(betas_diff, z_a_diff, y_diff)
                d1 = np.gradient(alt_low_curve, betas_diff)
                d2 = np.gradient(d1, betas_diff)
                
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=alt_low_curve, mode="lines", 
                                name="Low y", line=dict(color="orange", width=2)))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d1, mode="lines", 
                                name="Low y d/dβ", line=dict(color="orange", dash='dash')))
                fig_diff.add_trace(go.Scatter(x=betas_diff, y=d2, mode="lines", 
                                name="Low y d²/dβ²", line=dict(color="orange", dash='dot')))

            progress.update(5, "Finalizing plot")
            fig_diff.update_layout(
                title="Differential Analysis vs. β" + 
                      (" (Eigenvalue Method)" if use_eigenvalue_method_diff else " (Discriminant Method)"),
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
            
            progress.complete()
            st.plotly_chart(fig_diff, use_container_width=True)
            
            with st.expander("Curve Types", expanded=False):
                st.markdown("""
                - Solid lines: Original curves
                - Dashed lines: First derivatives (d/dβ)
                - Dotted lines: Second derivatives (d²/dβ²)
                """)


# Show high precision info
st.sidebar.markdown(f"#### Calculation Settings")
st.sidebar.info(f"Using high precision calculations with {mpmath.mp.dps} digits for accurate results.")

# Add OpenMP thread control if C++ acceleration is available
if cpp_module is not None:
    st.sidebar.markdown("#### C++ Acceleration Settings")
    omp_threads = st.sidebar.slider("OpenMP Threads", min_value=1, max_value=12, value=4, step=1)
    
    # Set OpenMP threads
    try:
        # Try to import OpenMP control module
        import ctypes
        omp_lib = ctypes.CDLL("libgomp.so.1")
        omp_lib.omp_set_num_threads(ctypes.c_int(omp_threads))
        st.sidebar.success(f"Set OpenMP to use {omp_threads} threads.")
    except Exception as e:
        st.sidebar.warning(f"Could not set OpenMP threads: {str(e)}")