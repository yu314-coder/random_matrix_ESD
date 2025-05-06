#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace py = pybind11;

// Apply the condition for y
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
    
    // Simple formula for cubic discriminant
    return std::pow((b*c)/(6.0*a*a) - std::pow(b, 3)/(27.0*std::pow(a, 3)) - d/(2.0*a), 2) +
           std::pow(c/(3.0*a) - std::pow(b, 2)/(9.0*std::pow(a, 2)), 3);
}

// Function to compute the theoretical max value
double compute_theoretical_max(double a, double y, double beta) {
    auto f = [a, y, beta](double k) -> double {
        return (y * beta * (a - 1) * k + (a * k + 1) * ((y - 1) * k - 1)) / 
               ((a * k + 1) * (k * k + k) * y);
    };
    
    // Use numerical optimization to find the maximum
    // Grid search followed by golden section search
    double best_k = 1.0;
    double best_val = f(best_k);
    
    // Initial grid search over a wide range
    const int num_grid_points = 200;
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
    const double tolerance = 1e-10;
    
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
    
    return f((a_gs + b_gs) / 2.0);
}

// Function to compute the theoretical min value
double compute_theoretical_min(double a, double y, double beta) {
    auto f = [a, y, beta](double t) -> double {
        return (y * beta * (a - 1) * t + (a * t + 1) * ((y - 1) * t - 1)) / 
               ((a * t + 1) * (t * t + t) * y);
    };
    
    // Use numerical optimization to find the minimum
    // Grid search followed by golden section search
    double best_t = -0.5 / a; // Midpoint of (-1/a, 0)
    double best_val = f(best_t);
    
    // Initial grid search over the range (-1/a, 0)
    const int num_grid_points = 200;
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
    const double tolerance = 1e-10;
    
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
    
    return f((a_gs + b_gs) / 2.0);
}

// Compute eigenvalues for a given beta value
std::tuple<double, double> compute_eigenvalues_for_beta(double z_a, double y, double beta, int n, int seed) {
    // Apply the condition for y
    double y_effective = apply_y_condition(y);
    
    // Set random seed
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    // Compute dimension p based on aspect ratio y
    int p = static_cast<int>(y_effective * n);
    
    // Generate random matrix X
    Eigen::MatrixXd X(p, n);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            X(i, j) = norm(gen);
        }
    }
    
    // Compute sample covariance matrix S_n = (1/n) * X * X^T
    Eigen::MatrixXd S_n = (X * X.transpose()) / static_cast<double>(n);
    
    // Build T_n diagonal matrix
    int k = static_cast<int>(std::floor(beta * p));
    std::vector<double> diags(p);
    std::fill_n(diags.begin(), k, z_a);
    std::fill_n(diags.begin() + k, p - k, 1.0);
    
    // Shuffle diagonal entries
    std::shuffle(diags.begin(), diags.end(), gen);
    
    // Create T_n and its square root
    Eigen::MatrixXd T_n = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd T_sqrt = Eigen::MatrixXd::Zero(p, p);
    
    for (int i = 0; i < p; i++) {
        double v = diags[i];
        T_n(i, i) = v;
        T_sqrt(i, i) = std::sqrt(v);
    }
    
    // Form B = T_sqrt * S_n * T_sqrt (symmetric)
    Eigen::MatrixXd B = T_sqrt * S_n * T_sqrt;
    
    // Compute eigenvalues of B
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(B);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    
    // Return min and max eigenvalues
    double min_eigenvalue = eigenvalues(0);
    double max_eigenvalue = eigenvalues(p-1);
    
    return std::make_tuple(min_eigenvalue, max_eigenvalue);
}

// Compute eigenvalue support boundaries
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
compute_eigenvalue_support_boundaries(double z_a, double y, const std::vector<double>& beta_values, 
                                    int n_samples, int seeds) {
    size_t num_betas = beta_values.size();
    std::vector<double> min_eigenvalues(num_betas, 0.0);
    std::vector<double> max_eigenvalues(num_betas, 0.0);
    std::vector<double> theoretical_min_values(num_betas, 0.0);
    std::vector<double> theoretical_max_values(num_betas, 0.0);
    
    for (size_t i = 0; i < num_betas; i++) {
        double beta = beta_values[i];
        
        // Calculate theoretical values
        theoretical_max_values[i] = compute_theoretical_max(z_a, y, beta);
        theoretical_min_values[i] = compute_theoretical_min(z_a, y, beta);
        
        std::vector<double> min_vals;
        std::vector<double> max_vals;
        
        // Run multiple trials with different seeds
        for (int seed = 0; seed < seeds; seed++) {
            auto [min_eig, max_eig] = compute_eigenvalues_for_beta(z_a, y, beta, n_samples, seed);
            min_vals.push_back(min_eig);
            max_vals.push_back(max_eig);
        }
        
        // Average over seeds
        double min_sum = 0.0, max_sum = 0.0;
        for (double val : min_vals) min_sum += val;
        for (double val : max_vals) max_sum += val;
        
        min_eigenvalues[i] = min_vals.empty() ? 0.0 : min_sum / min_vals.size();
        max_eigenvalues[i] = max_vals.empty() ? 0.0 : max_sum / max_vals.size();
    }
    
    return std::make_tuple(min_eigenvalues, max_eigenvalues, theoretical_min_values, theoretical_max_values);
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
        
        if (std::isnan(f1) || std::isnan(f2)) {
            continue;
        }
        
        if (f1 == 0.0) {
            roots_found.push_back(z_grid[i]);
        } else if (f2 == 0.0) {
            roots_found.push_back(z_grid[i+1]);
        } else if (f1 * f2 < 0) {
            // Binary search for zero crossing
            double zl = z_grid[i];
            double zr = z_grid[i+1];
            double f1_copy = f1;
            for (int iter = 0; iter < 50; iter++) {
                double mid = 0.5 * (zl + zr);
                double fm = discriminant_func(mid, beta, z_a, y_effective);
                if (fm == 0.0) {
                    zl = zr = mid;
                    break;
                }
                if ((fm < 0 && f1_copy < 0) || (fm > 0 && f1_copy > 0)) {
                    zl = mid;
                    f1_copy = fm;
                } else {
                    zr = mid;
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
    std::vector<double> z_min_values(beta_steps);
    std::vector<double> z_max_values(beta_steps);
    
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

// Compute max k expression over a range of betas
std::vector<double> compute_max_k_expression(const std::vector<double>& betas, double z_a, double y) {
    size_t n = betas.size();
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; i++) {
        result[i] = compute_theoretical_max(z_a, y, betas[i]);
    }
    
    return result;
}

// Compute min t expression over a range of betas
std::vector<double> compute_min_t_expression(const std::vector<double>& betas, double z_a, double y) {
    size_t n = betas.size();
    std::vector<double> result(n);
    
    for (size_t i = 0; i < n; i++) {
        result[i] = compute_theoretical_min(z_a, y, betas[i]);
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

// Generate eigenvalue distribution for a specific beta
std::vector<double> generate_eigenvalue_distribution(double beta, double y, double z_a, int n, int seed) {
    // Apply the condition for y
    double y_effective = apply_y_condition(y);
    
    // Set random seed
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    // Compute dimension p based on aspect ratio y
    int p = static_cast<int>(y_effective * n);
    
    // Generate random matrix X
    Eigen::MatrixXd X(p, n);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            X(i, j) = norm(gen);
        }
    }
    
    // Compute sample covariance matrix S_n = (1/n) * X * X^T
    Eigen::MatrixXd S_n = (X * X.transpose()) / static_cast<double>(n);
    
    // Build T_n diagonal matrix
    int k = static_cast<int>(std::floor(beta * p));
    std::vector<double> diags(p);
    std::fill_n(diags.begin(), k, z_a);
    std::fill_n(diags.begin() + k, p - k, 1.0);
    
    // Shuffle diagonal entries
    std::shuffle(diags.begin(), diags.end(), gen);
    
    // Create T_n
    Eigen::MatrixXd T_n = Eigen::MatrixXd::Zero(p, p);
    for (int i = 0; i < p; i++) {
        T_n(i, i) = diags[i];
    }
    
    // Compute B_n = S_n * T_n
    Eigen::MatrixXd B_n = S_n * T_n;
    
    // Compute eigenvalues
    Eigen::EigenSolver<Eigen::MatrixXd> solver(B_n);
    
    // Extract and return real parts of eigenvalues
    std::vector<double> eigenvalues(p);
    for (int i = 0; i < p; i++) {
        eigenvalues[i] = solver.eigenvalues()(i).real();
    }
    
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
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
}