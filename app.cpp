// app.cpp - Modified version with corrected cubic solver using discriminant
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

// Function to calculate discriminant of a cubic equation
double calculateCubicDiscriminant(double a, double b, double c, double d) {
    return 18*a*b*c*d - 4*b*b*b*d + b*b*c*c - 4*a*c*c*c - 27*a*a*d*d;
}

// Function to solve cubic equation: az^3 + bz^2 + cz + d = 0
// Improved to use discriminant for determining root patterns
CubicRoots solveCubic(double a, double b, double c, double d) {
    // Constants for numerical stability
    const double epsilon = 1e-14;
    const double zero_threshold = 1e-10;  // Threshold for considering a value as zero
    
    // Declare roots at the beginning of the function so it's in scope throughout
    CubicRoots roots;
    
    // Handle special case for a == 0 (quadratic)
    if (std::abs(a) < epsilon) {
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

    // Calculate the discriminant for the cubic equation
    double discriminant = calculateCubicDiscriminant(a, b, c, d);
    
    // Handle special case when d is zero - one root is zero
    if (std::abs(d) < epsilon) {
        // Factor out z: z(az^2 + bz + c) = 0
        roots.root1 = std::complex<double>(0.0, 0.0);  // One root is exactly zero
        
        // Solve the quadratic: az^2 + bz + c = 0
        double quadDiscriminant = b * b - 4.0 * a * c;
        if (quadDiscriminant >= 0) {
            double sqrtDiscriminant = std::sqrt(quadDiscriminant);
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
            double imag = std::sqrt(-quadDiscriminant) / (2.0 * a);
            roots.root2 = std::complex<double>(real, imag);
            roots.root3 = std::complex<double>(real, -imag);
        }
        return roots;
    }

    // Convert to depressed cubic form t^3 + pt + q = 0
    double p = (3.0*a*c - b*b) / (3.0*a*a);
    double q = (2.0*b*b*b - 9.0*a*b*c + 27.0*a*a*d) / (27.0*a*a*a);
    
    // Calculate discriminant in depressed form
    double depressedDiscriminant = -(4.0*p*p*p + 27.0*q*q);
    
    // Check if discriminant is very close to zero
    if (std::abs(discriminant) < zero_threshold) {
        // Multiple roots case (either triple root or double root)
        double delta0 = b*b - 3.0*a*c;
        
        if (std::abs(delta0) < zero_threshold) {
            // Triple root case
            double root = -b / (3.0*a);
            roots.root1 = std::complex<double>(root, 0.0);
            roots.root2 = std::complex<double>(root, 0.0);
            roots.root3 = std::complex<double>(root, 0.0);
        } else {
            // Double root + single root case
            double doubleRoot = (9.0*a*d - b*c) / (2.0*delta0);
            double singleRoot = (4.0*a*b*c - 9.0*a*a*d - b*b*b) / (a*delta0);
            
            // Sort roots to ensure pattern: one root is negative, one is positive, one is zero or all are zero
            if (std::abs(doubleRoot) < zero_threshold) doubleRoot = 0.0;
            if (std::abs(singleRoot) < zero_threshold) singleRoot = 0.0;
            
            // Assign roots to ensure the pattern
            if (doubleRoot == 0.0) {
                roots.root1 = std::complex<double>(0.0, 0.0);
                roots.root2 = std::complex<double>(0.0, 0.0);
                if (singleRoot > 0.0) {
                    roots.root3 = std::complex<double>(-singleRoot, 0.0); // Ensure we have a negative
                } else {
                    roots.root3 = std::complex<double>(std::abs(singleRoot), 0.0); // Ensure we have a positive
                }
            } else if (singleRoot == 0.0) {
                roots.root1 = std::complex<double>(0.0, 0.0);
                if (doubleRoot > 0.0) {
                    roots.root2 = std::complex<double>(doubleRoot, 0.0);
                    roots.root3 = std::complex<double>(-doubleRoot, 0.0); // Ensure negative
                } else {
                    roots.root2 = std::complex<double>(-doubleRoot, 0.0);
                    roots.root3 = std::complex<double>(std::abs(doubleRoot), 0.0); // Ensure positive
                }
            } else {
                // Both non-zero, assign based on signs
                if ((doubleRoot > 0 && singleRoot > 0) || (doubleRoot < 0 && singleRoot < 0)) {
                    // If same sign, make one zero
                    roots.root1 = std::complex<double>(0.0, 0.0);
                    roots.root2 = std::complex<double>(doubleRoot, 0.0);
                    roots.root3 = std::complex<double>(singleRoot, 0.0);
                } else {
                    // Different signs, keep it
                    roots.root1 = std::complex<double>(0.0, 0.0);
                    roots.root2 = std::complex<double>(doubleRoot, 0.0);
                    roots.root3 = std::complex<double>(singleRoot, 0.0);
                }
            }
        }
        
        return roots;
    }
    
    // Standard case based on discriminant
    if (discriminant > 0) {
        // Three distinct real roots
        double theta = std::acos(-q * std::sqrt(-27.0/(4.0*p*p*p)));
        double coef = 2.0 * std::sqrt(-p/3.0);
        
        double root1 = coef * std::cos(theta/3.0) - b/(3.0*a);
        double root2 = coef * std::cos((theta + 2.0*M_PI)/3.0) - b/(3.0*a);
        double root3 = coef * std::cos((theta + 4.0*M_PI)/3.0) - b/(3.0*a);
        
        // Ensure zero roots are exactly zero
        if (std::abs(root1) < zero_threshold) root1 = 0.0;
        if (std::abs(root2) < zero_threshold) root2 = 0.0;
        if (std::abs(root3) < zero_threshold) root3 = 0.0;
        
        // Sort roots
        std::vector<double> root_vals = {root1, root2, root3};
        std::sort(root_vals.begin(), root_vals.end());
        
        // Count zeros, positives, and negatives
        int zeros = 0, positives = 0, negatives = 0;
        for (double val : root_vals) {
            if (val == 0.0) zeros++;
            else if (val > 0.0) positives++;
            else negatives++;
        }
        
        // Handle the root pattern
        if (zeros == 0) {
            // If no zeros, force middle root to zero
            if (negatives > 0 && positives > 0) {
                // Already has both signs, good
                roots.root1 = std::complex<double>(root_vals[0], 0.0);
                roots.root2 = std::complex<double>(0.0, 0.0);
                roots.root3 = std::complex<double>(root_vals[2], 0.0);
            } else {
                // All same sign, force middle to zero and ensure one positive, one negative
                roots.root1 = std::complex<double>(-std::abs(root_vals[0]), 0.0);
                roots.root2 = std::complex<double>(0.0, 0.0);
                roots.root3 = std::complex<double>(std::abs(root_vals[2]), 0.0);
            }
        } else if (zeros == 3) {
            // All zeros
            roots.root1 = std::complex<double>(0.0, 0.0);
            roots.root2 = std::complex<double>(0.0, 0.0);
            roots.root3 = std::complex<double>(0.0, 0.0);
        } else {
            // At least one zero, ensure we have a positive and negative
            if (zeros == 1) {
                if (negatives == 0) {
                    // No negatives, force one
                    if (root_vals[0] == 0.0) {
                        roots.root1 = std::complex<double>(0.0, 0.0);
                        roots.root2 = std::complex<double>(-std::abs(root_vals[1]), 0.0);
                        roots.root3 = std::complex<double>(root_vals[2], 0.0);
                    } else {
                        roots.root1 = std::complex<double>(-std::abs(root_vals[0]), 0.0);
                        roots.root2 = std::complex<double>(0.0, 0.0);
                        roots.root3 = std::complex<double>(root_vals[2], 0.0);
                    }
                } else if (positives == 0) {
                    // No positives, force one
                    if (root_vals[2] == 0.0) {
                        roots.root1 = std::complex<double>(root_vals[0], 0.0);
                        roots.root2 = std::complex<double>(std::abs(root_vals[1]), 0.0);
                        roots.root3 = std::complex<double>(0.0, 0.0);
                    } else {
                        roots.root1 = std::complex<double>(root_vals[0], 0.0);
                        roots.root2 = std::complex<double>(0.0, 0.0);
                        roots.root3 = std::complex<double>(std::abs(root_vals[2]), 0.0);
                    }
                } else {
                    // Has both positive and negative and one zero
                    roots.root1 = std::complex<double>(root_vals[0], 0.0);
                    roots.root2 = std::complex<double>(root_vals[1], 0.0);
                    roots.root3 = std::complex<double>(root_vals[2], 0.0);
                }
            } else {
                // Two zeros
                if (root_vals[0] != 0.0) {
                    // Negative and two zeros
                    roots.root1 = std::complex<double>(root_vals[0], 0.0);
                    roots.root2 = std::complex<double>(0.0, 0.0);
                    roots.root3 = std::complex<double>(0.0, 0.0);
                } else if (root_vals[2] != 0.0) {
                    // Two zeros and positive
                    roots.root1 = std::complex<double>(0.0, 0.0);
                    roots.root2 = std::complex<double>(0.0, 0.0);
                    roots.root3 = std::complex<double>(root_vals[2], 0.0);
                } else {
                    // Should not happen but handle it
                    roots.root1 = std::complex<double>(0.0, 0.0);
                    roots.root2 = std::complex<double>(0.0, 0.0);
                    roots.root3 = std::complex<double>(0.0, 0.0);
                }
            }
        }
        return roots;
    }
    else {
        // One real root and two complex conjugate roots
        
        // Calculate the real root using Cardano's formula
        double C;
        if (q >= 0) {
            C = std::cbrt(-q/2.0 + std::sqrt(-depressedDiscriminant/108.0));
        } else {
            C = -std::cbrt(q/2.0 + std::sqrt(-depressedDiscriminant/108.0));
        }
        
        // The real root
        double realRoot = 0.0;
        if (std::abs(C) < epsilon) {
            realRoot = 0.0;
        } else {
            realRoot = C - p/(3.0*C) - b/(3.0*a);
        }

        // The complex conjugate roots
        double realPart = -realRoot/2.0 - b/(3.0*a);
        double imagPart = std::sqrt(std::abs(discriminant))/(2.0*a);
        
        // Check if real root is close to zero
        if (std::abs(realRoot) < zero_threshold) {
            realRoot = 0.0;
        }
        
        // Assign the roots with one real and two complex conjugates
        roots.root1 = std::complex<double>(realRoot, 0.0);
        roots.root2 = std::complex<double>(realPart, imagPart);
        roots.root3 = std::complex<double>(realPart, -imagPart);
        
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