// app.cpp - Modified version for Hugging Face Spaces (calculation only)
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
void save_as_json(const std::string& filename, 
                 const std::vector<double>& beta_values,
                 const std::vector<double>& max_eigenvalues,
                 const std::vector<double>& min_eigenvalues,
                 const std::vector<double>& theoretical_max_values,
                 const std::vector<double>& theoretical_min_values) {
    
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
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
}

int main(int argc, char* argv[]) {
    // ─── Inputs from command line ───────────────────────────────────────────
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <n> <p> <a> <y> <fineness> <theory_grid_points> <theory_tolerance> <output_file>" << std::endl;
        return 1;
    }
    
    int n = std::stoi(argv[1]);
    int p = std::stoi(argv[2]);
    double a = std::stod(argv[3]);
    double y = std::stod(argv[4]);
    int fineness = std::stoi(argv[5]);
    int theory_grid_points = std::stoi(argv[6]);
    double theory_tolerance = std::stod(argv[7]);
    std::string output_file = argv[8];
    const double b = 1.0;
    
    std::cout << "Running with parameters: n = " << n << ", p = " << p 
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
    
    // ─── Random‐Gaussian X and S_n ────────────────────────────────
    std::mt19937_64 rng{std::random_device{}()};
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
        std::vector<double> diags(p);
        std::fill_n(diags.begin(), k, a);
        std::fill_n(diags.begin()+k, p-k, b);
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
    save_as_json(output_file, beta_values, max_eigenvalues, min_eigenvalues, 
                theoretical_max_values, theoretical_min_values);
    
    std::cout << "Data saved to " << output_file << std::endl;
    
    return 0;
}