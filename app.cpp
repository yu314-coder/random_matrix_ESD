// app.cpp - Modified version for Hugging Face Spaces
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

// Function to compute the theoretical max value
double compute_theoretical_max(double a, double y, double beta) {
    auto f = [a, y, beta](double k) -> double {
        return (y * beta * (a - 1) * k + (a * k + 1) * ((y - 1) * k - 1)) / 
               ((a * k + 1) * (k * k + k));
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
    
    // Multiply the result by y before returning
    return f((a_gs + b_gs) / 2.0) *y ;
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
    
    // Multiply the result by y before returning
    return f((a_gs + b_gs) / 2.0) *y ;
}

int main(int argc, char* argv[]) {
    // ─── Inputs from command line ───────────────────────────────────────────
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <n> <p> <a> <y> <output_file>" << std::endl;
        return 1;
    }
    
    int n = std::stoi(argv[1]);
    int p = std::stoi(argv[2]);
    double a = std::stod(argv[3]);
    double y = std::stod(argv[4]);
    std::string output_file = argv[5];
    const double b = 1.0;
    
    std::cout << "Running with parameters: n = " << n << ", p = " << p 
              << ", a = " << a << ", y = " << y << std::endl;
    std::cout << "Output will be saved to: " << output_file << std::endl;
    
    // ─── Beta range parameters ────────────────────────────────────────
    const int num_beta_points = 100; // More points for smoother curves
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
        
        // Compute theoretical values
        theoretical_max_values[beta_idx] = compute_theoretical_max(a, y, beta);
        theoretical_min_values[beta_idx] = compute_theoretical_min(a, y, beta);
        
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
        
        // Progress indicator - modified to be less verbose for Streamlit
        if (beta_idx % 20 == 0) {
            std::cout << "Processing beta = " << beta 
                      << " (" << beta_idx+1 << "/" << num_beta_points << ")" << std::endl;
        }
    }
    
    // ─── Prepare canvas for plotting ────────────────────────────────
    const int H = 950, W = 1200; // Taller canvas to accommodate legend below
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(250, 250, 250)); // Slightly off-white background
    
    // ─── Find min/max for scaling ───────────────────────────────────
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    
    for (double v : max_eigenvalues) max_y = std::max(max_y, v);
    for (double v : min_eigenvalues) min_y = std::min(min_y, v);
    for (double v : theoretical_max_values) max_y = std::max(max_y, v);
    for (double v : theoretical_min_values) min_y = std::min(min_y, v);
    
    // Add some padding
    double y_padding = (max_y - min_y) * 0.15; // More padding for better spacing
    min_y -= y_padding;
    max_y += y_padding;
    
    // ─── Draw coordinate axes ───────────────────────────────────────
    const int margin = 100; // Larger margin for better spacing
    const int plot_width = W - 2 * margin;
    const int plot_height = H - 2 * margin - 150; // Reduced height to make room for legend below
    
    // Plot area background (light gray)
    cv::rectangle(canvas, 
                 cv::Point(margin, margin),
                 cv::Point(W - margin, margin + plot_height),
                 cv::Scalar(245, 245, 245), cv::FILLED);
    
    // X-axis (beta)
    cv::line(canvas, 
             cv::Point(margin, margin + plot_height),
             cv::Point(W - margin, margin + plot_height),
             cv::Scalar(40, 40, 40), 2);
    
    // Y-axis (eigenvalues)
    cv::line(canvas,
             cv::Point(margin, margin + plot_height),
             cv::Point(margin, margin),
             cv::Scalar(40, 40, 40), 2);
    
    // ─── Draw axes labels ────────────────────────────────────────────
    cv::putText(canvas, "β", 
                cv::Point(W - margin/2, margin + plot_height + 30),
                cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
                
    // Y-axis label (fixed - no rotation)
    cv::putText(canvas, "Eigenvalues", 
                cv::Point(margin/4, margin/2 - 10),
                cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
                
    // ─── Draw title ───────────────────────────────────────────────────
    std::stringstream title_ss;
    title_ss << std::fixed << std::setprecision(2);
    title_ss << "Eigenvalue Analysis: a = " << a << ", y = " << y;
    cv::putText(canvas, title_ss.str(),
                cv::Point(W/2 - 200, 45),
                cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 0), 2);
    
    // ─── Draw grid lines ────────────────────────────────────────────────
    const int num_grid_lines = 11; // 0.0, 0.1, 0.2, ..., 1.0 for beta
    for (int i = 0; i < num_grid_lines; ++i) {
        // Horizontal grid lines
        int y_pos = margin + i * (plot_height / (num_grid_lines - 1));
        cv::line(canvas,
                 cv::Point(margin, y_pos),
                 cv::Point(W - margin, y_pos),
                 cv::Scalar(220, 220, 220), 1);
                 
        // Vertical grid lines
        int x_pos = margin + i * (plot_width / (num_grid_lines - 1));
        cv::line(canvas,
                 cv::Point(x_pos, margin),
                 cv::Point(x_pos, margin + plot_height),
                 cv::Scalar(220, 220, 220), 1);
                 
        // X-axis labels (beta values)
        double beta_val = static_cast<double>(i) / (num_grid_lines - 1);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << beta_val;
        cv::putText(canvas, ss.str(),
                    cv::Point(x_pos - 10, margin + plot_height + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
                    
        // Y-axis labels (eigenvalue values)
        double eig_val = min_y + (max_y - min_y) * i / (num_grid_lines - 1);
        std::stringstream ss2;
        ss2 << std::fixed << std::setprecision(2) << eig_val;
        cv::putText(canvas, ss2.str(),
                    cv::Point(margin/2 - 40, margin + plot_height - i * (plot_height / (num_grid_lines - 1)) + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    }
    
    // ─── Draw the four curves ───────────────────────────────────────────
    // Convert data points to pixel coordinates
    auto to_point = [&](double beta, double val) -> cv::Point {
        int x = margin + static_cast<int>(beta * plot_width);
        int y = margin + plot_height - static_cast<int>((val - min_y) / (max_y - min_y) * plot_height);
        return cv::Point(x, y);
    };
    
    // Better colors for visibility
    cv::Scalar emp_max_color(60, 60, 220);    // Dark red
    cv::Scalar emp_min_color(220, 60, 60);    // Dark blue
    cv::Scalar theo_max_color(30, 180, 30);   // Dark green
    cv::Scalar theo_min_color(180, 30, 180);  // Dark purple
    
    // Empirical max eigenvalues (red)
    std::vector<cv::Point> max_eig_points;
    for (int i = 0; i < num_beta_points; ++i) {
        max_eig_points.push_back(to_point(beta_values[i], max_eigenvalues[i]));
    }
    cv::polylines(canvas, max_eig_points, false, emp_max_color, 3);
    
    // Empirical min eigenvalues (blue)
    std::vector<cv::Point> min_eig_points;
    for (int i = 0; i < num_beta_points; ++i) {
        min_eig_points.push_back(to_point(beta_values[i], min_eigenvalues[i]));
    }
    cv::polylines(canvas, min_eig_points, false, emp_min_color, 3);
    
    // Theoretical max values (green)
    std::vector<cv::Point> theo_max_points;
    for (int i = 0; i < num_beta_points; ++i) {
        theo_max_points.push_back(to_point(beta_values[i], theoretical_max_values[i]));
    }
    cv::polylines(canvas, theo_max_points, false, theo_max_color, 3);
    
    // Theoretical min values (purple)
    std::vector<cv::Point> theo_min_points;
    for (int i = 0; i < num_beta_points; ++i) {
        theo_min_points.push_back(to_point(beta_values[i], theoretical_min_values[i]));
    }
    cv::polylines(canvas, theo_min_points, false, theo_min_color, 3);
    
    // ─── Draw markers on the curves for better visibility ──────────────
    const int marker_interval = 10; // Show markers every 10 points
    for (int i = 0; i < num_beta_points; i += marker_interval) {
        // Max empirical eigenvalue markers
        cv::circle(canvas, max_eig_points[i], 5, emp_max_color, cv::FILLED);
        cv::circle(canvas, max_eig_points[i], 5, cv::Scalar(255, 255, 255), 1);
        
        // Min empirical eigenvalue markers
        cv::circle(canvas, min_eig_points[i], 5, emp_min_color, cv::FILLED);
        cv::circle(canvas, min_eig_points[i], 5, cv::Scalar(255, 255, 255), 1);
        
        // Theoretical max markers
        cv::drawMarker(canvas, theo_max_points[i], theo_max_color, cv::MARKER_DIAMOND, 10, 2);
        
        // Theoretical min markers
        cv::drawMarker(canvas, theo_min_points[i], theo_min_color, cv::MARKER_DIAMOND, 10, 2);
    }
    
    // ─── Draw legend BELOW the graph ────────────────────────────────────
    // Set up dimensions for the legend
    const int legend_width = 600;
    const int legend_height = 100;
    // Center the legend horizontally
    const int legend_x = W/2 - legend_width/2;
    // Position legend below the graph
    const int legend_y = margin + plot_height + 70;
    const int line_length = 40;
    const int line_spacing = 35;
    
    // Box around legend with shadow effect
    cv::rectangle(canvas,
                 cv::Point(legend_x + 3, legend_y + 3),
                 cv::Point(legend_x + legend_width + 3, legend_y + legend_height + 3),
                 cv::Scalar(180, 180, 180), cv::FILLED); // Shadow
    cv::rectangle(canvas,
                 cv::Point(legend_x, legend_y),
                 cv::Point(legend_x + legend_width, legend_y + legend_height),
                 cv::Scalar(240, 240, 240), cv::FILLED); // Main box
    cv::rectangle(canvas,
                 cv::Point(legend_x, legend_y),
                 cv::Point(legend_x + legend_width, legend_y + legend_height),
                 cv::Scalar(0, 0, 0), 1); // Border
    
    // Legend title
    cv::putText(canvas, "Legend",
                cv::Point(legend_x + legend_width/2 - 30, legend_y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
    cv::line(canvas,
             cv::Point(legend_x + 5, legend_y + 30),
             cv::Point(legend_x + legend_width - 5, legend_y + 30),
             cv::Scalar(150, 150, 150), 1);
    
    // Two legend entries per row, in two columns
    // First row
    // Empirical max (red)
    cv::line(canvas, 
             cv::Point(legend_x + 20, legend_y + 50),
             cv::Point(legend_x + 20 + line_length, legend_y + 50),
             emp_max_color, 3);
    cv::circle(canvas, cv::Point(legend_x + 20 + line_length/2, legend_y + 50), 5, emp_max_color, cv::FILLED);
    cv::putText(canvas, "Empirical Max Eigenvalue",
                cv::Point(legend_x + 20 + line_length + 10, legend_y + 50 + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
                
    // Empirical min (blue)
    cv::line(canvas, 
             cv::Point(legend_x + 20 + legend_width/2, legend_y + 50),
             cv::Point(legend_x + 20 + line_length + legend_width/2, legend_y + 50),
             emp_min_color, 3);
    cv::circle(canvas, cv::Point(legend_x + 20 + line_length/2 + legend_width/2, legend_y + 50), 5, emp_min_color, cv::FILLED);
    cv::putText(canvas, "Empirical Min Eigenvalue",
                cv::Point(legend_x + 20 + line_length + 10 + legend_width/2, legend_y + 50 + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    
    // Second row
    // Theoretical max (green)
    cv::line(canvas, 
             cv::Point(legend_x + 20, legend_y + 80),
             cv::Point(legend_x + 20 + line_length, legend_y + 80),
             theo_max_color, 3);
    cv::drawMarker(canvas, cv::Point(legend_x + 20 + line_length/2, legend_y + 80), 
                  theo_max_color, cv::MARKER_DIAMOND, 10, 2);
    cv::putText(canvas, "Theoretical Max Function",
                cv::Point(legend_x + 20 + line_length + 10, legend_y + 80 + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
                
    // Theoretical min (purple)
    cv::line(canvas, 
             cv::Point(legend_x + 20 + legend_width/2, legend_y + 80),
             cv::Point(legend_x + 20 + line_length + legend_width/2, legend_y + 80),
             theo_min_color, 3);
    cv::drawMarker(canvas, cv::Point(legend_x + 20 + line_length/2 + legend_width/2, legend_y + 80), 
                  theo_min_color, cv::MARKER_DIAMOND, 10, 2);
    cv::putText(canvas, "Theoretical Min Function",
                cv::Point(legend_x + 20 + line_length + 10 + legend_width/2, legend_y + 80 + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    
    // ─── Draw mathematical formulas in a box ──────────────────────────────
    cv::rectangle(canvas,
                 cv::Point(margin + 3, H - 80 + 3),
                 cv::Point(W - margin + 3, H - 20 + 3),
                 cv::Scalar(180, 180, 180), cv::FILLED); // Shadow
    cv::rectangle(canvas,
                 cv::Point(margin, H - 80),
                 cv::Point(W - margin, H - 20),
                 cv::Scalar(240, 240, 240), cv::FILLED); // Main box
    cv::rectangle(canvas,
                 cv::Point(margin, H - 80),
                 cv::Point(W - margin, H - 20),
                 cv::Scalar(0, 0, 0), 1); // Border
    
    std::string formula_text1 = "Max Function: max{k ∈ (0,∞)} [yβ(a-1)k + (ak+1)((y-1)k-1)]/[(ak+1)(k²+k)y]";
    std::string formula_text2 = "Min Function: min{t ∈ (-1/a,0)} [yβ(a-1)t + (at+1)((y-1)t-1)]/[(at+1)(t²+t)y]";
    
    cv::putText(canvas, formula_text1,
                cv::Point(margin + 20, H - 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, theo_max_color, 2);
                
    cv::putText(canvas, formula_text2,
                cv::Point(W/2 + 20, H - 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, theo_min_color, 2);
    
    // ─── Draw parameter info ────────────────────────────────────────────
    std::stringstream params_ss;
    params_ss << std::fixed << std::setprecision(2);
    params_ss << "Parameters: n = " << n << ", p = " << p << ", a = " << a << ", y = " << y;
    cv::putText(canvas, params_ss.str(),
                cv::Point(margin, 80),
                cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
    
    // ─── Save the image to the output directory ───────────────────────────
    cv::imwrite(output_file, canvas);
    std::cout << "Plot saved as " << output_file << std::endl;
    
    return 0;
}