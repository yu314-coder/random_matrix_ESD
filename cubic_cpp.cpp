#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>

namespace py = pybind11;

// Apply the condition for y
double apply_y_condition(double y) {
    return y > 1.0 ? y : 1.0 / y;
}

// Fast discriminant calculation
double discriminant_func(double z, double beta, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    
    // Coefficients
    double a = z * z_a;
    double b = z * z_a + z + z_a - z_a * y_effective;
    double c = z + z_a + 1.0 - y_effective * (beta * z_a + 1.0 - beta);
    double d = 1.0;
    
    // Standard formula for cubic discriminant - optimized calculation
    double p1 = b*c/(6.0*a*a);
    double p2 = b*b*b/(27.0*a*a*a);
    double p3 = d/(2.0*a);
    double term1 = p1 - p2 - p3;
    term1 *= term1;
    
    double q1 = c/(3.0*a);
    double q2 = b*b/(9.0*a*a);
    double term2 = q1 - q2;
    term2 = term2*term2*term2;
    
    return term1 + term2;
}

// Function to compute the theoretical max value - optimized with fewer function calls
double compute_theoretical_max(double a, double y, double beta) {
    // Exit early if parameters would cause division by zero or other issues
    if (a <= 0 || y <= 0 || beta < 0 || beta > 1) {
        return 0.0;
    }
    
    // Precompute constants for the formula
    double y_effective = apply_y_condition(y);
    double beta_term = y_effective * beta * (a - 1);
    double y_term = y_effective - 1.0;
    
    auto f = [a, beta_term, y_term, y_effective](double k) -> double {
        // Fast evaluation of the function
        double ak_plus_1 = a * k + 1.0;
        double numerator = beta_term * k + ak_plus_1 * (y_term * k - 1.0);
        double denominator = ak_plus_1 * (k * k + k) * y_effective;
        return numerator / denominator;
    };
    
    // Use numerical optimization to find the maximum
    // Grid search followed by golden section search
    double best_k = 1.0;
    double best_val = f(best_k);
    
    // Initial fast grid search with fewer points
    const int num_grid_points = 50; // Reduced from 200
    for (int i = 0; i < num_grid_points; ++i) {
        double k = 0.01 + 100.0 * i / (num_grid_points - 1);
        double val = f(k);
        if (val > best_val) {
            best_val = val;
            best_k = k;
        }
    }
    
    // Refine with golden section search
    double a_gs = std::max(0.01, best_k / 10.0);
    double b_gs = best_k * 10.0;
    const double golden_ratio = 1.618033988749895;
    const double tolerance = 1e-6; // Increased from 1e-10 for speed
    
    double c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
    double d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
    double fc = f(c_gs);
    double fd = f(d_gs);
    
    // Limited iterations for faster convergence
    for (int iter = 0; iter < 20 && std::abs(b_gs - a_gs) > tolerance; ++iter) {
        if (fc > fd) {
            b_gs = d_gs;
            d_gs = c_gs;
            c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
            fd = fc;
            fc = f(c_gs);
        } else {
            a_gs = c_gs;
            c_gs = d_gs;
            d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
            fc = fd;
            fd = f(d_gs);
        }
    }
    
    return f((a_gs + b_gs) / 2.0);
}

// Function to compute the theoretical min value - optimized similarly
double compute_theoretical_min(double a, double y, double beta) {
    // Exit early if parameters would cause division by zero or other issues
    if (a <= 0 || y <= 0 || beta < 0 || beta > 1) {
        return 0.0;
    }
    
    // Precompute constants
    double y_effective = apply_y_condition(y);
    double beta_term = y_effective * beta * (a - 1);
    double y_term = y_effective - 1.0;
    
    auto f = [a, beta_term, y_term, y_effective](double t) -> double {
        double at_plus_1 = a * t + 1.0;
        double numerator = beta_term * t + at_plus_1 * (y_term * t - 1.0);
        double denominator = at_plus_1 * (t * t + t) * y_effective;
        return numerator / denominator;
    };
    
    // Initial bound check
    if (a <= 0) return 0.0;
    
    // Find midpoint of range as starting guess
    double best_t = -0.5 / a;
    double best_val = f(best_t);
    
    // Initial grid search over the range (-1/a, 0)
    const int num_grid_points = 50; // Reduced from 200
    double range = 0.998/a;
    double start = -0.999/a;
    
    for (int i = 1; i < num_grid_points; ++i) {
        double t = start + range * i / (num_grid_points - 1);
        if (t >= 0 || t <= -1.0/a) continue;
        
        double val = f(t);
        if (val < best_val) {
            best_val = val;
            best_t = t;
        }
    }
    
    // Refine with golden section search
    double a_gs = start; 
    double b_gs = -0.001/a;
    const double golden_ratio = 1.618033988749895;
    const double tolerance = 1e-6; // Increased from 1e-10
    
    double c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
    double d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
    double fc = f(c_gs);
    double fd = f(d_gs);
    
    // Limited iterations
    for (int iter = 0; iter < 20 && std::abs(b_gs - a_gs) > tolerance; ++iter) {
        if (fc < fd) {
            b_gs = d_gs;
            d_gs = c_gs;
            c_gs = b_gs - (b_gs - a_gs) / golden_ratio;
            fd = fc;
            fc = f(c_gs);
        } else {
            a_gs = c_gs;
            c_gs = d_gs;
            d_gs = a_gs + (b_gs - a_gs) / golden_ratio;
            fc = fd;
            fd = f(d_gs);
        }
    }
    
    return f((a_gs + b_gs) / 2.0);
}

// Fast eigendecomposition of a symmetric matrix using Jacobi method
void eigen_decomposition(const std::vector<std::vector<double>>& matrix, 
                       std::vector<double>& eigenvalues) {
    int n = matrix.size();
    eigenvalues.resize(n);
    
    // Copy matrix for computation
    std::vector<std::vector<double>> a = matrix;
    
    // Allocate temp arrays
    std::vector<double> d(n);
    std::vector<double> z(n, 0.0);
    
    // Initialize eigenvalues with diagonal elements
    for (int i = 0; i < n; i++) {
        d[i] = a[i][i];
    }
    
    // Main algorithm: Jacobi rotations
    const int MAX_ITER = 50;  // Limit iterations for speed
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Sum off-diagonal elements
        double sum = 0.0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                sum += std::abs(a[i][j]);
            }
        }
        
        // Check for convergence
        if (sum < 1e-8) break;
        
        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                double theta, t, c, s;
                
                // Skip very small elements
                if (std::abs(a[p][q]) < 1e-10) continue;
                
                // Compute rotation angle
                theta = 0.5 * std::atan2(2*a[p][q], a[p][p] - a[q][q]);
                c = std::cos(theta);
                s = std::sin(theta);
                t = std::tan(theta);
                
                // Update diagonal elements
                double h = t * a[p][q];
                z[p] -= h;
                z[q] += h;
                d[p] -= h;
                d[q] += h;
                
                // Set off-diagonal element to zero
                a[p][q] = 0.0;
                
                // Update other elements
                for (int i = 0; i < p; i++) {
                    double g = a[i][p], h = a[i][q];
                    a[i][p] = c*g - s*h;
                    a[i][q] = s*g + c*h;
                }
                
                for (int i = p+1; i < q; i++) {
                    double g = a[p][i], h = a[i][q];
                    a[p][i] = c*g - s*h;
                    a[i][q] = s*g + c*h;
                }
                
                for (int i = q+1; i < n; i++) {
                    double g = a[p][i], h = a[q][i];
                    a[p][i] = c*g - s*h;
                    a[q][i] = s*g + c*h;
                }
            }
        }
        
        // Update eigenvalues
        for (int i = 0; i < n; i++) {
            d[i] += z[i];
            z[i] = 0.0;
        }
    }
    
    // Return eigenvalues
    eigenvalues = d;
}

// Optimized matrix multiplication: C = A * B
void matrix_multiply(const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    std::vector<std::vector<double>>& C) {
    int m = A.size();
    int n = B[0].size();
    int k = A[0].size();
    
    C.resize(m, std::vector<double>(n, 0.0));
    
    // Transpose B for better cache locality
    std::vector<std::vector<double>> B_t(n, std::vector<double>(k, 0.0));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B_t[j][i] = B[i][j];
        }
    }
    
    // Multiply with transposed B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += A[i][l] * B_t[j][l];
            }
            C[i][j] = sum;
        }
    }
}

// Highly optimized eigenvalue computation for a given beta
std::tuple<double, double> compute_eigenvalues_for_beta(double z_a, double y, double beta, int n, int seed) {
    double y_effective = apply_y_condition(y);
    
    // Set random seed
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    // Compute dimension p based on aspect ratio y
    int p = static_cast<int>(y_effective * n);
    
    // Generate random matrix X (with pre-allocation)
    std::vector<std::vector<double>> X(p, std::vector<double>(n, 0.0));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            X[i][j] = norm(gen);
        }
    }
    
    // Compute X * X^T / n - optimized matrix multiplication
    std::vector<std::vector<double>> S_n(p, std::vector<double>(p, 0.0));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {  // Compute only lower triangle
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += X[i][k] * X[j][k];
            }
            sum /= n;
            S_n[i][j] = sum;
            if (i != j) S_n[j][i] = sum;  // Mirror to upper triangle
        }
    }
    
    // Build T_n diagonal matrix
    int k = static_cast<int>(std::floor(beta * p));
    std::vector<double> diags(p);
    std::fill_n(diags.begin(), k, z_a);
    std::fill_n(diags.begin() + k, p - k, 1.0);
    
    // Shuffle diagonal entries
    std::shuffle(diags.begin(), diags.end(), gen);
    
    // Create T_sqrt diagonal matrix
    std::vector<double> t_sqrt_diag(p);
    for (int i = 0; i < p; i++) {
        t_sqrt_diag[i] = std::sqrt(diags[i]);
    }
    
    // Compute B = T_sqrt * S_n * T_sqrt directly without full matrix multiplication
    // (optimize for diagonal T_sqrt)
    std::vector<std::vector<double>> B(p, std::vector<double>(p, 0.0));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            B[i][j] = S_n[i][j] * t_sqrt_diag[i] * t_sqrt_diag[j];
        }
    }
    
    // Compute eigenvalues efficiently
    std::vector<double> eigenvalues;
    eigen_decomposition(B, eigenvalues);
    
    // Sort eigenvalues
    std::sort(eigenvalues.begin(), eigenvalues.end());
    
    // Return min and max
    return std::make_tuple(eigenvalues.front(), eigenvalues.back());
}

// Fast computation of eigenvalue support boundaries
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
compute_eigenvalue_support_boundaries(double z_a, double y, const std::vector<double>& beta_values, 
                                    int n_samples, int seeds) {
    size_t num_betas = beta_values.size();
    std::vector<double> min_eigenvalues(num_betas, 0.0);
    std::vector<double> max_eigenvalues(num_betas, 0.0);
    std::vector<double> theoretical_min_values(num_betas, 0.0);
    std::vector<double> theoretical_max_values(num_betas, 0.0);
    
    // Pre-compute theoretical values for all betas (can be done in parallel)
    #pragma omp parallel for if(num_betas > 10)
    for (size_t i = 0; i < num_betas; i++) {
        double beta = beta_values[i];
        theoretical_max_values[i] = compute_theoretical_max(z_a, y, beta);
        theoretical_min_values[i] = compute_theoretical_min(z_a, y, beta);
    }
    
    // Compute eigenvalues for all betas (more expensive)
    for (size_t i = 0; i < num_betas; i++) {
        double beta = beta_values[i];
        
        std::vector<double> min_vals;
        std::vector<double> max_vals;
        
        // Use just one seed for speed if the seeds parameter is small
        int actual_seeds = (seeds <= 2) ? 1 : seeds;
        
        for (int seed = 0; seed < actual_seeds; seed++) {
            auto [min_eig, max_eig] = compute_eigenvalues_for_beta(z_a, y, beta, n_samples, seed);
            min_vals.push_back(min_eig);
            max_vals.push_back(max_eig);
        }
        
        // Average over seeds
        double min_sum = 0.0, max_sum = 0.0;
        for (double val : min_vals) min_sum += val;
        for (double val : max_vals) max_sum += val;
        
        min_eigenvalues[i] = min_sum / min_vals.size();
        max_eigenvalues[i] = max_sum / max_vals.size();
    }
    
    return std::make_tuple(min_eigenvalues, max_eigenvalues, theoretical_min_values, theoretical_max_values);
}

// Very optimized version to find zeros of discriminant
std::vector<double> find_z_at_discriminant_zero(double z_a, double y, double beta, 
                                              double z_min, double z_max, int steps) {
    std::vector<double> roots_found;
    double y_effective = apply_y_condition(y);
    
    // Adaptive step size for better accuracy in important regions
    double step = (z_max - z_min) / (steps - 1);
    
    // Evaluate discriminant at first point
    double z_prev = z_min;
    double f_prev = discriminant_func(z_prev, beta, z_a, y_effective);
    
    // Scan through the range looking for sign changes
    for (int i = 1; i < steps; ++i) {
        double z_curr = z_min + i * step;
        double f_curr = discriminant_func(z_curr, beta, z_a, y_effective);
        
        if (std::isnan(f_prev) || std::isnan(f_curr)) {
            z_prev = z_curr;
            f_prev = f_curr;
            continue;
        }
        
        // Check for exact zero
        if (f_prev == 0.0) {
            roots_found.push_back(z_prev);
        } 
        else if (f_curr == 0.0) {
            roots_found.push_back(z_curr);
        }
        // Check for sign change
        else if (f_prev * f_curr < 0) {
            // Binary search for more precise zero
            double zl = z_prev;
            double zr = z_curr;
            double fl = f_prev;
            double fr = f_curr;
            
            // Fewer iterations, still good precision
            for (int iter = 0; iter < 20; iter++) {
                double zm = (zl + zr) / 2;
                double fm = discriminant_func(zm, beta, z_a, y_effective);
                
                if (fm == 0.0 || std::abs(zr - zl) < 1e-8) {
                    roots_found.push_back(zm);
                    break;
                }
                
                if ((fm < 0 && fl < 0) || (fm > 0 && fl > 0)) {
                    zl = zm;
                    fl = fm;
                } else {
                    zr = zm;
                    fr = fm;
                }
            }
            
            if (std::abs(zr - zl) >= 1e-8) {
                // Add the midpoint if we didn't converge fully
                roots_found.push_back((zl + zr) / 2);
            }
        }
        
        z_prev = z_curr;
        f_prev = f_curr;
    }
    
    return roots_found;
}

// Compute z bounds but with fewer steps for speed
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
sweep_beta_and_find_z_bounds(double z_a, double y, double z_min, double z_max, 
                           int beta_steps, int z_steps) {
    std::vector<double> betas(beta_steps);
    std::vector<double> z_min_values(beta_steps);
    std::vector<double> z_max_values(beta_steps);
    
    // Use fewer z steps for faster computation
    int actual_z_steps = std::min(z_steps, 10000);
    
    double beta_step = 1.0 / (beta_steps - 1);
    for (int i = 0; i < beta_steps; i++) {
        betas[i] = i * beta_step;
        
        std::vector<double> roots = find_z_at_discriminant_zero(z_a, y, betas[i], z_min, z_max, actual_z_steps);
        
        if (roots.empty()) {
            z_min_values[i] = std::numeric_limits<double>::quiet_NaN();
            z_max_values[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Find min and max roots
            double min_root = *std::min_element(roots.begin(), roots.end());
            double max_root = *std::max_element(roots.begin(), roots.end());
            
            z_min_values[i] = min_root;
            z_max_values[i] = max_root;
        }
    }
    
    return std::make_tuple(betas, z_min_values, z_max_values);
}

// Fast implementations of curve computations
std::vector<double> compute_high_y_curve(const std::vector<double>& betas, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    size_t n = betas.size();
    std::vector<double> result(n);
    
    double a = z_a;
    double denominator = 1.0 - 2.0 * a;
    
    if (std::abs(denominator) < 1e-10) {
        std::fill(result.begin(), result.end(), std::numeric_limits<double>::quiet_NaN());
        return result;
    }
    
    // Precompute constants
    double term1 = -2.0 * a * y_effective;
    double term2 = -2.0 * a * (2.0 * a - 1.0);
    double term3 = -4.0 * a * (a - 1.0) * y_effective;
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas[i];
        double numerator = term3 * beta + term1 + term2;
        result[i] = numerator / denominator;
    }
    
    return result;
}

std::vector<double> compute_alternate_low_expr(const std::vector<double>& betas, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    size_t n = betas.size();
    std::vector<double> result(n);
    
    // Precompute constants
    double term1 = -2.0 * z_a * (1.0 - y_effective);
    double term2 = -2.0 * z_a * z_a;
    double term3 = z_a * y_effective * (z_a - 1.0);
    double denominator = 2.0 + 2.0 * z_a;
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas[i];
        result[i] = (term3 * beta + term1 + term2) / denominator;
    }
    
    return result;
}

std::vector<double> compute_max_k_expression(const std::vector<double>& betas, double z_a, double y) {
    size_t n = betas.size();
    std::vector<double> result(n);
    
    // Since we've optimized compute_theoretical_max, just call it in a loop
    #pragma omp parallel for if(n > 20)
    for (size_t i = 0; i < n; i++) {
        result[i] = compute_theoretical_max(z_a, y, betas[i]);
    }
    
    return result;
}

std::vector<double> compute_min_t_expression(const std::vector<double>& betas, double z_a, double y) {
    size_t n = betas.size();
    std::vector<double> result(n);
    
    // Similarly for min
    #pragma omp parallel for if(n > 20)
    for (size_t i = 0; i < n; i++) {
        result[i] = compute_theoretical_min(z_a, y, betas[i]);
    }
    
    return result;
}

// Generate eigenvalue distribution - faster implementation
std::vector<double> generate_eigenvalue_distribution(double beta, double y, double z_a, int n, int seed) {
    double y_effective = apply_y_condition(y);
    
    // Set random seed
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    // Compute dimension p based on aspect ratio y
    int p = static_cast<int>(y_effective * n);
    
    // Generate random matrix X
    std::vector<std::vector<double>> X(p, std::vector<double>(n, 0.0));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            X[i][j] = norm(gen);
        }
    }
    
    // Compute S_n = X * X^T / n efficiently
    std::vector<std::vector<double>> S_n(p, std::vector<double>(p, 0.0));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {  // Compute only lower triangle
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += X[i][k] * X[j][k];
            }
            sum /= n;
            S_n[i][j] = sum;
            if (i != j) S_n[j][i] = sum;  // Mirror to upper triangle
        }
    }
    
    // Build T_n diagonal matrix
    int k = static_cast<int>(std::floor(beta * p));
    std::vector<double> diags(p);
    std::fill_n(diags.begin(), k, z_a);
    std::fill_n(diags.begin() + k, p - k, 1.0);
    
    // Shuffle diagonal entries
    std::shuffle(diags.begin(), diags.end(), gen);
    
    // Compute B_n = S_n * diag(T_n) efficiently
    std::vector<std::vector<double>> B_n(p, std::vector<double>(p, 0.0));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            B_n[i][j] = S_n[i][j] * diags[j];
        }
    }
    
    // Compute eigenvalues efficiently
    std::vector<double> eigenvalues;
    eigen_decomposition(B_n, eigenvalues);
    
    // Sort eigenvalues
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

// ADD THE MISSING COMPUTE_DERIVATIVES FUNCTION
std::tuple<std::vector<double>, std::vector<double>> 
compute_derivatives(const std::vector<double>& curve, const std::vector<double>& betas) {
    size_t n = curve.size();
    std::vector<double> d1(n, 0.0);
    std::vector<double> d2(n, 0.0);
    
    if (n < 2 || n != betas.size()) {
        return std::make_tuple(d1, d2); // Return zeros if invalid input
    }
    
    // First derivative using central differences
    for (size_t i = 1; i < n-1; i++) {
        d1[i] = (curve[i+1] - curve[i-1]) / (betas[i+1] - betas[i-1]);
    }
    
    // Edge cases using forward/backward differences
    if (n > 1) {
        d1[0] = (curve[1] - curve[0]) / (betas[1] - betas[0]);
        d1[n-1] = (curve[n-1] - curve[n-2]) / (betas[n-1] - betas[n-2]);
    }
    
    // Second derivative using the same method applied to first derivative
    for (size_t i = 1; i < n-1; i++) {
        d2[i] = (d1[i+1] - d1[i-1]) / (betas[i+1] - betas[i-1]);
    }
    
    // Edge cases for second derivative
    if (n > 1) {
        d2[0] = (d1[1] - d1[0]) / (betas[1] - betas[0]);
        d2[n-1] = (d1[n-1] - d1[n-2]) / (betas[n-1] - betas[n-2]);
    }
    
    return std::make_tuple(d1, d2);
}
// Compute cubic equation roots
std::vector<std::complex<double>> compute_cubic_roots(double z, double beta, double z_a, double y) {
    // Apply the condition for y
    double y_effective = apply_y_condition(y);
    
    // Coefficients in the form ax^3 + bx^2 + cx + d = 0
    double a = z * z_a;
    double b = z * z_a + z + z_a - z_a * y_effective;
    double c = z + z_a + 1 - y_effective * (beta * z_a + 1 - beta);
    double d = 1.0;
    
    // Handle special cases
    if (std::abs(a) < 1e-10) {
        // Quadratic case or linear case
        std::vector<std::complex<double>> roots(3);
        if (std::abs(b) < 1e-10) {
            // Linear case
            roots[0] = std::complex<double>(-d / c, 0.0);
            roots[1] = std::complex<double>(0.0, 0.0);
            roots[2] = std::complex<double>(0.0, 0.0);
        } else {
            // Quadratic case: bx^2 + cx + d = 0
            double discriminant = c*c - 4*b*d;
            if (discriminant >= 0) {
                double sqrt_disc = std::sqrt(discriminant);
                roots[0] = std::complex<double>((-c + sqrt_disc) / (2 * b), 0.0);
                roots[1] = std::complex<double>((-c - sqrt_disc) / (2 * b), 0.0);
            } else {
                double sqrt_disc = std::sqrt(-discriminant);
                roots[0] = std::complex<double>(-c / (2 * b), sqrt_disc / (2 * b));
                roots[1] = std::complex<double>(-c / (2 * b), -sqrt_disc / (2 * b));
            }
            roots[2] = std::complex<double>(0.0, 0.0);
        }
        return roots;
    }
    
    // Standard cubic case
    // First, convert to depressed cubic t^3 + pt + q = 0
    b /= a;
    c /= a;
    d /= a;
    
    double p = c - b*b/3;
    double q = d - b*c/3 + 2*b*b*b/27;
    double disc = q*q/4 + p*p*p/27;
    
    std::vector<std::complex<double>> roots(3);
    
    // Handle different cases based on discriminant
    if (std::abs(disc) < 1e-10) {
        // Discriminant is zero, potential multiple roots
        if (std::abs(p) < 1e-10 && std::abs(q) < 1e-10) {
            // Triple root
            roots[0] = roots[1] = roots[2] = std::complex<double>(-b/3, 0.0);
        } else {
            // One double root and one single root
            double u;
            if (q > 0) u = -std::cbrt(q/2);
            else u = std::cbrt(-q/2);
            
            roots[0] = std::complex<double>(2*u - b/3, 0.0);
            roots[1] = roots[2] = std::complex<double>(-u - b/3, 0.0);
        }
    } else if (disc > 0) {
        // One real root and two complex conjugate roots
        double u = std::cbrt(-q/2 + std::sqrt(disc));
        double v = std::cbrt(-q/2 - std::sqrt(disc));
        
        roots[0] = std::complex<double>(u + v - b/3, 0.0);
        
        double real_part = -(u + v)/2 - b/3;
        double imag_part = std::sqrt(3) * (u - v) / 2;
        
        roots[1] = std::complex<double>(real_part, imag_part);
        roots[2] = std::complex<double>(real_part, -imag_part);
    } else {
        // Three distinct real roots
        double theta = std::acos(-q/2 * std::sqrt(-27/(p*p*p)));
        double coef = 2 * std::sqrt(-p/3);
        
        roots[0] = std::complex<double>(coef * std::cos(theta/3) - b/3, 0.0);
        roots[1] = std::complex<double>(coef * std::cos((theta + 2*M_PI)/3) - b/3, 0.0);
        roots[2] = std::complex<double>(coef * std::cos((theta + 4*M_PI)/3) - b/3, 0.0);
    }
    
    return roots;
}
// Python module definition
PYBIND11_MODULE(cubic_cpp, m) {
    m.doc() = "C++ accelerated functions for cubic root analysis";
    
    m.def("discriminant_func", &discriminant_func, 
          "Calculate cubic discriminant",
          py::arg("z"), py::arg("beta"), py::arg("z_a"), py::arg("y"));
    
    m.def("find_z_at_discriminant_zero", &find_z_at_discriminant_zero,
          "Find zeros of discriminant",
          py::arg("z_a"), py::arg("y"), py::arg("beta"), py::arg("z_min"), 
          py::arg("z_max"), py::arg("steps"));
    
    m.def("sweep_beta_and_find_z_bounds", &sweep_beta_and_find_z_bounds,
          "Compute support boundaries by sweeping beta",
          py::arg("z_a"), py::arg("y"), py::arg("z_min"), py::arg("z_max"), 
          py::arg("beta_steps"), py::arg("z_steps"));
    
    m.def("compute_theoretical_max", &compute_theoretical_max,
          "Compute theoretical maximum function value",
          py::arg("a"), py::arg("y"), py::arg("beta"));
          
    m.def("compute_theoretical_min", &compute_theoretical_min,
          "Compute theoretical minimum function value",
          py::arg("a"), py::arg("y"), py::arg("beta"));
    
    m.def("compute_eigenvalue_support_boundaries", &compute_eigenvalue_support_boundaries,
          "Compute empirical and theoretical eigenvalue support boundaries",
          py::arg("z_a"), py::arg("y"), py::arg("beta_values"), 
          py::arg("n_samples"), py::arg("seeds"));
    
    m.def("compute_high_y_curve", &compute_high_y_curve,
          "Compute high y expression curve",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_alternate_low_expr", &compute_alternate_low_expr,
          "Compute alternate low expression curve",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_max_k_expression", &compute_max_k_expression,
          "Compute max k expression for multiple beta values",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_min_t_expression", &compute_min_t_expression,
          "Compute min t expression for multiple beta values",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_derivatives", &compute_derivatives,
          "Compute first and second derivatives",
          py::arg("curve"), py::arg("betas"));
          
    m.def("generate_eigenvalue_distribution", &generate_eigenvalue_distribution,
          "Generate eigenvalue distribution for a specific beta",
          py::arg("beta"), py::arg("y"), py::arg("z_a"), py::arg("n"), py::arg("seed"));
    m.def("compute_cubic_roots", &compute_cubic_roots,
      "Compute the roots of the cubic equation",
      py::arg("z"), py::arg("beta"), py::arg("z_a"), py::arg("y"));
}