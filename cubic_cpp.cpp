#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <cmath>
#include <random>

namespace py = pybind11;

// Helper function to apply y condition
double apply_y_condition(double y) {
    return y > 1.0 ? y : 1.0 / y;
}

// Discriminant calculation
double discriminant_func(double z, double beta, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    
    // Coefficients
    double a = z * z_a;
    double b = z * z_a + z + z_a - z_a * y_effective;
    double c = z + z_a + 1.0 - y_effective * (beta * z_a + 1.0 - beta);
    double d = 1.0;
    
    // Discriminant formula
    return std::pow((b * c) / (6.0 * a * a) - std::pow(b, 3) / (27.0 * std::pow(a, 3)) - d / (2.0 * a), 2) +
           std::pow(c / (3.0 * a) - std::pow(b, 2) / (9.0 * std::pow(a, 2)), 3);
}

// Find zeros of discriminant
std::vector<double> find_z_at_discriminant_zero(double z_a, double y, double beta, 
                                              double z_min, double z_max, int steps) {
    std::vector<double> roots_found;
    double y_effective = apply_y_condition(y);
    
    // Create z grid
    std::vector<double> z_grid(steps);
    double step_size = (z_max - z_min) / (steps - 1);
    for (int i = 0; i < steps; i++) {
        z_grid[i] = z_min + i * step_size;
    }
    
    // Evaluate discriminant at each grid point
    std::vector<double> disc_vals(steps);
    for (int i = 0; i < steps; i++) {
        disc_vals[i] = discriminant_func(z_grid[i], beta, z_a, y_effective);
    }
    
    // Find sign changes (zeros)
    for (int i = 0; i < steps - 1; i++) {
        double f1 = disc_vals[i];
        double f2 = disc_vals[i+1];
        
        // Skip if NaN
        if (std::isnan(f1) || std::isnan(f2)) {
            continue;
        }
        
        // Check for exact zeros
        if (f1 == 0.0) {
            roots_found.push_back(z_grid[i]);
        } else if (f2 == 0.0) {
            roots_found.push_back(z_grid[i+1]);
        } else if (f1 * f2 < 0) {
            // Sign change - use binary search to refine
            double zl = z_grid[i];
            double zr = z_grid[i+1];
            
            for (int iter = 0; iter < 50; iter++) {
                double mid = 0.5 * (zl + zr);
                double fm = discriminant_func(mid, beta, z_a, y_effective);
                
                if (fm == 0.0) {
                    zl = zr = mid;
                    break;
                }
                
                if ((fm < 0 && f1 < 0) || (fm > 0 && f1 > 0)) {
                    zl = mid;
                    f1 = fm;
                } else {
                    zr = mid;
                    f2 = fm;
                }
            }
            
            roots_found.push_back(0.5 * (zl + zr));
        }
    }
    
    return roots_found;
}

// Sweep beta and find z bounds
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
sweep_beta_and_find_z_bounds(double z_a, double y, double z_min, double z_max, 
                           int beta_steps, int z_steps) {
    std::vector<double> betas(beta_steps);
    std::vector<double> z_min_values(beta_steps, 0.0);
    std::vector<double> z_max_values(beta_steps, 0.0);
    
    double beta_step = 1.0 / (beta_steps - 1);
    for (int i = 0; i < beta_steps; i++) {
        betas[i] = i * beta_step;
        
        std::vector<double> roots = find_z_at_discriminant_zero(z_a, y, betas[i], z_min, z_max, z_steps);
        
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

// Compute cubic roots
std::vector<std::complex<double>> compute_cubic_roots(double z, double beta, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    
    // Coefficients
    double a = z * z_a;
    double b = z * z_a + z + z_a - z_a * y_effective;
    double c = z + z_a + 1.0 - y_effective * (beta * z_a + 1.0 - beta);
    double d = 1.0;
    
    std::vector<std::complex<double>> roots(3);
    
    // Handle special cases
    if (std::abs(a) < 1e-10) {
        if (std::abs(b) < 1e-10) {
            // Linear case
            roots[0] = std::complex<double>(-d/c, 0);
            roots[1] = std::complex<double>(0, 0);
            roots[2] = std::complex<double>(0, 0);
        } else {
            // Quadratic case
            double discriminant = c*c - 4.0*b*d;
            if (discriminant >= 0) {
                double sqrt_disc = std::sqrt(discriminant);
                roots[0] = std::complex<double>((-c + sqrt_disc) / (2.0 * b), 0);
                roots[1] = std::complex<double>((-c - sqrt_disc) / (2.0 * b), 0);
            } else {
                double sqrt_disc = std::sqrt(-discriminant);
                roots[0] = std::complex<double>(-c / (2.0 * b), sqrt_disc / (2.0 * b));
                roots[1] = std::complex<double>(-c / (2.0 * b), -sqrt_disc / (2.0 * b));
            }
            roots[2] = std::complex<double>(0, 0);
        }
        return roots;
    }
    
    // Standard cubic formula implementation
    // Normalize to form: x^3 + px^2 + qx + r = 0
    double p = b / a;
    double q = c / a;
    double r = d / a;
    
    // Depress the cubic: substitute x = y - p/3 to get y^3 + py + q = 0
    double p_over_3 = p / 3.0;
    double new_p = q - p * p / 3.0;
    double new_q = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;
    
    // Calculate discriminant
    double discriminant = 4.0 * new_p * new_p * new_p / 27.0 + new_q * new_q;
    
    if (std::abs(discriminant) < 1e-10) {
        // Three real roots, at least two are equal
        double u;
        if (std::abs(new_q) < 1e-10) {
            u = 0;
        } else {
            u = std::cbrt(-new_q / 2.0);
        }
        roots[0] = std::complex<double>(2.0 * u - p_over_3, 0);
        roots[1] = std::complex<double>(-u - p_over_3, 0);
        roots[2] = std::complex<double>(-u - p_over_3, 0);
    } else if (discriminant > 0) {
        // One real root, two complex conjugate roots
        double sqrt_disc = std::sqrt(discriminant);
        double u = std::cbrt(-new_q / 2.0 + sqrt_disc / 2.0);
        double v = std::cbrt(-new_q / 2.0 - sqrt_disc / 2.0);
        
        // Real root
        roots[0] = std::complex<double>(u + v - p_over_3, 0);
        
        // Complex roots
        const double sqrt3_over_2 = std::sqrt(3.0) / 2.0;
        roots[1] = std::complex<double>(-0.5 * (u + v) - p_over_3, sqrt3_over_2 * (u - v));
        roots[2] = std::complex<double>(-0.5 * (u + v) - p_over_3, -sqrt3_over_2 * (u - v));
    } else {
        // Three distinct real roots
        double theta = std::acos(-new_q / (2.0 * std::sqrt(-std::pow(new_p, 3) / 27.0)));
        double sqrt_term = 2.0 * std::sqrt(-new_p / 3.0);
        
        roots[0] = std::complex<double>(sqrt_term * std::cos(theta / 3.0) - p_over_3, 0);
        roots[1] = std::complex<double>(sqrt_term * std::cos((theta + 2.0 * M_PI) / 3.0) - p_over_3, 0);
        roots[2] = std::complex<double>(sqrt_term * std::cos((theta + 4.0 * M_PI) / 3.0) - p_over_3, 0);
    }
    
    return roots;
}

// Compute eigenvalue support boundaries
std::tuple<std::vector<double>, std::vector<double>>
compute_eigenvalue_support_boundaries(double z_a, double y, const std::vector<double>& beta_values,
                                    int n_samples, int seeds) {
    double y_effective = apply_y_condition(y);
    size_t num_betas = beta_values.size();
    
    std::vector<double> min_eigenvalues(num_betas, 0.0);
    std::vector<double> max_eigenvalues(num_betas, 0.0);
    
    for (size_t i = 0; i < num_betas; i++) {
        double beta = beta_values[i];
        
        std::vector<double> min_vals;
        std::vector<double> max_vals;
        
        // Run multiple trials
        for (int seed = 0; seed < seeds; seed++) {
            // Set random seed
            std::mt19937 gen(seed * 100 + i);
            std::normal_distribution<double> normal_dist(0.0, 1.0);
            
            // Compute dimensions
            int n = n_samples;
            int p = static_cast<int>(y_effective * n);
            
            // Construct T_n (Population/Shape Matrix)
            int k = static_cast<int>(std::floor(beta * p));
            Eigen::VectorXd diag_entries(p);
            
            // Fill diagonal entries
            for (int j = 0; j < k; j++) {
                diag_entries(j) = z_a;
            }
            for (int j = k; j < p; j++) {
                diag_entries(j) = 1.0;
            }
            
            // Shuffle diagonal entries
            for (int j = p - 1; j > 0; j--) {
                std::uniform_int_distribution<int> uniform_dist(0, j);
                int idx = uniform_dist(gen);
                std::swap(diag_entries(j), diag_entries(idx));
            }
            
            Eigen::MatrixXd T_n = diag_entries.asDiagonal();
            
            // Generate the data matrix X with i.i.d. standard normal entries
            Eigen::MatrixXd X(p, n);
            for (int row = 0; row < p; row++) {
                for (int col = 0; col < n; col++) {
                    X(row, col) = normal_dist(gen);
                }
            }
            
            // Compute the sample covariance matrix S_n = (1/n) * XX^T
            Eigen::MatrixXd S_n = (1.0 / n) * (X * X.transpose());
            
            // Compute B_n = S_n T_n
            Eigen::MatrixXd B_n = S_n * T_n;
            
            // Compute eigenvalues of B_n
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(B_n);
            Eigen::VectorXd eigenvalues = solver.eigenvalues();
            
            // Find minimum and maximum eigenvalues
            min_vals.push_back(eigenvalues(0));
            max_vals.push_back(eigenvalues(p-1));
        }
        
        // Average over seeds for stability
        double min_sum = 0.0, max_sum = 0.0;
        for (double val : min_vals) min_sum += val;
        for (double val : max_vals) max_sum += val;
        
        min_eigenvalues[i] = min_sum / seeds;
        max_eigenvalues[i] = max_sum / seeds;
    }
    
    return std::make_tuple(min_eigenvalues, max_eigenvalues);
}

// Compute high y curve
std::vector<double> compute_high_y_curve(const std::vector<double>& betas, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    size_t n = betas.size();
    std::vector<double> result(n);
    
    double a = z_a;
    double denominator = 1.0 - 2.0 * a;
    
    if (std::abs(denominator) < 1e-10) {
        // Handle division by zero
        std::fill(result.begin(), result.end(), std::numeric_limits<double>::quiet_NaN());
        return result;
    }
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas[i];
        double numerator = -4.0 * a * (a - 1.0) * y_effective * beta - 2.0 * a * y_effective - 2.0 * a * (2.0 * a - 1.0);
        result[i] = numerator / denominator;
    }
    
    return result;
}

// Compute alternate low expression
std::vector<double> compute_alternate_low_expr(const std::vector<double>& betas, double z_a, double y) {
    double y_effective = apply_y_condition(y);
    size_t n = betas.size();
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas[i];
        result[i] = (z_a * y_effective * beta * (z_a - 1.0) - 2.0 * z_a * (1.0 - y_effective) - 2.0 * z_a * z_a) / (2.0 + 2.0 * z_a);
    }
    
    return result;
}

// Compute max k expression
std::vector<double> compute_max_k_expression(const std::vector<double>& betas, double z_a, double y, int k_samples=1000) {
    double y_effective = apply_y_condition(y);
    size_t n = betas.size();
    std::vector<double> result(n);
    
    // Sample k values on logarithmic scale
    std::vector<double> k_values(k_samples);
    double log_min = std::log(0.001);
    double log_max = std::log(1000.0);
    double log_step = (log_max - log_min) / (k_samples - 1);
    
    for (int i = 0; i < k_samples; i++) {
        k_values[i] = std::exp(log_min + i * log_step);
    }
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas[i];
        std::vector<double> values(k_samples);
        
        for (int j = 0; j < k_samples; j++) {
            double k = k_values[j];
            double numerator = y_effective * beta * (z_a - 1.0) * k + (z_a * k + 1.0) * ((y_effective - 1.0) * k - 1.0);
            double denominator = (z_a * k + 1.0) * (k * k + k);
            
            if (std::abs(denominator) < 1e-10) {
                values[j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                values[j] = numerator / denominator;
            }
        }
        
        // Find maximum value, ignoring NaNs
        double max_val = -std::numeric_limits<double>::infinity();
        bool found_valid = false;
        
        for (double val : values) {
            if (!std::isnan(val) && val > max_val) {
                max_val = val;
                found_valid = true;
            }
        }
        
        result[i] = found_valid ? max_val : std::numeric_limits<double>::quiet_NaN();
    }
    
    return result;
}

// Compute min t expression
std::vector<double> compute_min_t_expression(const std::vector<double>& betas, double z_a, double y, int t_samples=1000) {
    double y_effective = apply_y_condition(y);
    size_t n = betas.size();
    std::vector<double> result(n);
    
    if (z_a <= 0) {
        std::fill(result.begin(), result.end(), std::numeric_limits<double>::quiet_NaN());
        return result;
    }
    
    // Sample t values in (-1/a, 0)
    double lower_bound = -1.0 / z_a + 1e-10;  // Avoid division by zero
    std::vector<double> t_values(t_samples);
    double t_step = (0.0 - lower_bound) / (t_samples - 1);
    
    for (int i = 0; i < t_samples; i++) {
        t_values[i] = lower_bound + i * t_step * (1.0 - 1e-10);  // Avoid exactly 0
    }
    
    for (size_t i = 0; i < n; i++) {
        double beta = betas[i];
        std::vector<double> values(t_samples);
        
        for (int j = 0; j < t_samples; j++) {
            double t = t_values[j];
            double numerator = y_effective * beta * (z_a - 1.0) * t + (z_a * t + 1.0) * ((y_effective - 1.0) * t - 1.0);
            double denominator = (z_a * t + 1.0) * (t * t + t);
            
            if (std::abs(denominator) < 1e-10) {
                values[j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                values[j] = numerator / denominator;
            }
        }
        
        // Find minimum value, ignoring NaNs
        double min_val = std::numeric_limits<double>::infinity();
        bool found_valid = false;
        
        for (double val : values) {
            if (!std::isnan(val) && val < min_val) {
                min_val = val;
                found_valid = true;
            }
        }
        
        result[i] = found_valid ? min_val : std::numeric_limits<double>::quiet_NaN();
    }
    
    return result;
}

// Compute derivatives
std::tuple<std::vector<double>, std::vector<double>>
compute_derivatives(const std::vector<double>& curve, const std::vector<double>& betas) {
    size_t n = betas.size();
    std::vector<double> d1(n, 0.0);
    std::vector<double> d2(n, 0.0);
    
    // First derivative using central difference
    for (size_t i = 1; i < n - 1; i++) {
        double h = betas[i+1] - betas[i-1];
        d1[i] = (curve[i+1] - curve[i-1]) / h;
    }
    
    // Handle endpoints with forward/backward difference
    if (n > 1) {
        d1[0] = (curve[1] - curve[0]) / (betas[1] - betas[0]);
        d1[n-1] = (curve[n-1] - curve[n-2]) / (betas[n-1] - betas[n-2]);
    }
    
    // Second derivative using central difference
    for (size_t i = 1; i < n - 1; i++) {
        double h = betas[i+1] - betas[i-1];
        d2[i] = 2.0 * (curve[i+1] - 2.0 * curve[i] + curve[i-1]) / (h * h);
    }
    
    // Handle endpoints
    if (n > 2) {
        d2[0] = d2[1];
        d2[n-1] = d2[n-2];
    }
    
    return std::make_tuple(d1, d2);
}

// Generate eigenvalue distribution
std::vector<double> generate_eigenvalue_distribution(double beta, double y, double z_a, int n, int seed) {
    double y_effective = apply_y_condition(y);
    
    // Set random seed
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // Compute dimension p based on aspect ratio y
    int p = static_cast<int>(y_effective * n);
    
    // Constructing T_n (Population/Shape Matrix)
    int k = static_cast<int>(std::floor(beta * p));
    Eigen::VectorXd diag_entries(p);
    
    // Fill diagonal entries
    for (int j = 0; j < k; j++) {
        diag_entries(j) = z_a;
    }
    for (int j = k; j < p; j++) {
        diag_entries(j) = 1.0;
    }
    
    // Shuffle diagonal entries
    for (int j = p - 1; j > 0; j--) {
        std::uniform_int_distribution<int> uniform_dist(0, j);
        int idx = uniform_dist(gen);
        std::swap(diag_entries(j), diag_entries(idx));
    }
    
    Eigen::MatrixXd T_n = diag_entries.asDiagonal();
    
    // Generate the data matrix X with i.i.d. standard normal entries
    Eigen::MatrixXd X(p, n);
    for (int row = 0; row < p; row++) {
        for (int col = 0; col < n; col++) {
            X(row, col) = normal_dist(gen);
        }
    }
    
    // Compute the sample covariance matrix S_n = (1/n) * XX^T
    Eigen::MatrixXd S_n = (1.0 / n) * (X * X.transpose());
    
    // Compute B_n = S_n T_n
    Eigen::MatrixXd B_n = S_n * T_n;
    
    // Compute eigenvalues of B_n
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(B_n);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    
    // Convert to std::vector
    std::vector<double> result(p);
    for (int i = 0; i < p; i++) {
        result[i] = eigenvalues(i);
    }
    
    return result;
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
    
    m.def("compute_cubic_roots", &compute_cubic_roots,
          "Compute roots of cubic equation",
          py::arg("z"), py::arg("beta"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_eigenvalue_support_boundaries", &compute_eigenvalue_support_boundaries,
          "Compute eigenvalue support boundaries using random matrices",
          py::arg("z_a"), py::arg("y"), py::arg("beta_values"), 
          py::arg("n_samples"), py::arg("seeds"));
    
    m.def("compute_high_y_curve", &compute_high_y_curve,
          "Compute high y expression curve",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_alternate_low_expr", &compute_alternate_low_expr,
          "Compute alternate low expression curve",
          py::arg("betas"), py::arg("z_a"), py::arg("y"));
    
    m.def("compute_max_k_expression", &compute_max_k_expression,
          "Compute max k expression",
          py::arg("betas"), py::arg("z_a"), py::arg("y"), py::arg("k_samples") = 1000);
    
    m.def("compute_min_t_expression", &compute_min_t_expression,
          "Compute min t expression",
          py::arg("betas"), py::arg("z_a"), py::arg("y"), py::arg("t_samples") = 1000);
    
    m.def("compute_derivatives", &compute_derivatives,
          "Compute first and second derivatives",
          py::arg("curve"), py::arg("betas"));
    
    m.def("generate_eigenvalue_distribution", &generate_eigenvalue_distribution,
          "Generate eigenvalue distribution simulation",
          py::arg("beta"), py::arg("y"), py::arg("z_a"), py::arg("n"), py::arg("seed"));
}