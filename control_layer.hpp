#ifndef CONTROL_LAYER_HPP
#define CONTROL_LAYER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

namespace Apex {

    // Step 3: Trajectory Control - Conjugate Gradient Solver
    // Layman: Iteratively solving a massive system without direct matrix inversion.
    // Solves Ax = b for x, using the Krylov subspace.
    cv::Mat solve_conjugate_gradient(const cv::Mat& A, const cv::Mat& b, cv::Mat x0, int max_iterations = 50, float tolerance = 1e-4) {
        // We use OpenCV cv::Mat for matrix memory management, but write the math from scratch.
        cv::Mat x = x0.clone();
        
        // 1. Initial residual: r_0 = b - Ax_0
        cv::Mat r = b - A * x;
        // 2. Initial search direction: p_0 = r_0
        cv::Mat p = r.clone();
        
        // Initial squared residual magnitude
        double rs_old = r.dot(r);

        for (int i = 0; i < max_iterations; ++i) {
            // Matrix-vector multiplication: A * p
            cv::Mat Ap = A * p;
            
            // Step size alpha: (r^T r) / (p^T A p)
            double alpha = rs_old / p.dot(Ap);
            
            // Update the state: x_{k+1} = x_k + alpha * p_k
            x = x + alpha * p;
            
            // Update the residual: r_{k+1} = r_k - alpha * A p_k
            r = r - alpha * Ap;
            
            // New squared residual magnitude
            double rs_new = r.dot(r);
            
            // If the residual error is tiny, the solver has converged to the exact answer!
            if (std::sqrt(rs_new) < tolerance) {
                // Convergence achieved.
                break;
            }
            
            // Orthogonalization factor beta: (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            double beta = rs_new / rs_old;
            
            // Update the search direction: p_{k+1} = r_{k+1} + beta * p_k
            p = r + beta * p;
            
            rs_old = rs_new;
        }

        return x;
    }

    // Step 3: Trajectory Control - Quadratic Programming Formulation
    // Generates a smooth optimal trajectory by setting up an Ax = b system representing 
    // minimal steering jerk constraints. Features dynamic obstacle avoidance swerving.
    std::vector<cv::Point> calculate_optimal_trajectory(int start_x, int start_y, int height, const LaneBoundaries& bounds, bool proximity_warning, int obstacle_idx) {
        int horizon = bounds.left_curve.size(); // Typically 21 nodes matching the vision curve depth
        
        // Matrix A (Q in Quadratic Programming): Represents the penalty for steering jerk.
        cv::Mat A = cv::Mat::zeros(horizon, horizon, CV_32F);
        cv::Mat b = cv::Mat::zeros(horizon, 1, CV_32F);
        cv::Mat x0 = cv::Mat::zeros(horizon, 1, CV_32F); // Initial guess (straight line)

        for (int i = 0; i < horizon; i++) {
            A.at<float>(i, i) = 2.0f; // Diagonal penalty
            if (i > 0) A.at<float>(i, i - 1) = -1.0f; // Cross penalty
            if (i < horizon - 1) A.at<float>(i, i + 1) = -1.0f; // Cross penalty
            
            float left_x = bounds.left_curve[i].x;
            float right_x = bounds.right_curve[i].x;
            
            // The optimal safe target is exactly between the two boundaries
            float target_center_x = (left_x + right_x) / 2.0f;
            
            // Dynamic KKT Constraint Target: Obstacle Avoidance Swerving
            // Only swerve if the obstacle is actually close to us. 
            // The swerve starts smoothly a few nodes before the actual obstacle impact point.
            if (proximity_warning && i >= std::max(0, obstacle_idx - 6)) {
                target_center_x = right_x - (right_x - left_x) * 0.2f; // Hug the right edge
            }
            
            // Setup the constraint vector 'b' to pull the CG solver towards this optimal target
            b.at<float>(i, 0) = (target_center_x - start_x); 
        }

        // Run our custom Conjugate Gradient loop to solve the massive Quadratic Programming matrix
        cv::Mat optimal_x = solve_conjugate_gradient(A, b, x0);

        // Convert the solved 1D mathematical state vector 'x' into physical 2D screen coordinates
        std::vector<cv::Point> trajectory_points;
        
        for (int i = 0; i < horizon; i++) {
            float optimal_steering_offset = optimal_x.at<float>(i, 0);
            int p_x = start_x + static_cast<int>(optimal_steering_offset);
            int p_y = bounds.left_curve[i].y; // Sync physical Y axis perfectly with vision curve nodes
            trajectory_points.push_back(cv::Point(p_x, p_y));
        }

        return trajectory_points;
    }

} // namespace Apex

#endif // CONTROL_LAYER_HPP
