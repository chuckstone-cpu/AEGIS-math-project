#ifndef CONTROL_LAYER_HPP
#define CONTROL_LAYER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

namespace Apex {

    cv::Mat solve_conjugate_gradient(const cv::Mat& A, const cv::Mat& b, cv::Mat x0, int max_iterations = 50, float tolerance = 1e-4) {
        cv::Mat x = x0.clone();
        cv::Mat r = b - A * x;
        cv::Mat p = r.clone();
        double rs_old = r.dot(r);

        for (int i = 0; i < max_iterations; ++i) {
            cv::Mat Ap = A * p;
            double alpha = rs_old / p.dot(Ap);
            x = x + alpha * p;
            r = r - alpha * Ap;
            
            double rs_new = r.dot(r);
            if (std::sqrt(rs_new) < tolerance) break;
            
            double beta = rs_new / rs_old;
            p = r + beta * p;
            rs_old = rs_new;
        }
        return x;
    }

    std::vector<cv::Point> calculate_optimal_trajectory(int start_x, int start_y, int height, const LaneBoundaries& bounds, bool proximity_warning, int obstacle_idx) {
        int horizon = bounds.left_curve.size(); 
        
        cv::Mat A = cv::Mat::zeros(horizon, horizon, CV_32F);
        cv::Mat b = cv::Mat::zeros(horizon, 1, CV_32F);
        cv::Mat x0 = cv::Mat::zeros(horizon, 1, CV_32F); 

        for (int i = 0; i < horizon; i++) {
            A.at<float>(i, i) = 2.0f;
            if (i > 0) A.at<float>(i, i - 1) = -1.0f;
            if (i < horizon - 1) A.at<float>(i, i + 1) = -1.0f;
            
            float left_x = bounds.left_curve[i].x;
            float right_x = bounds.right_curve[i].x;
            float target_center_x = (left_x + right_x) / 2.0f;
            
            if (proximity_warning && i >= std::max(0, obstacle_idx - 6)) {
                target_center_x = right_x - (right_x - left_x) * 0.2f;
            }
            
            b.at<float>(i, 0) = (target_center_x - start_x); 
        }

        cv::Mat optimal_x = solve_conjugate_gradient(A, b, x0);
        std::vector<cv::Point> trajectory_points;
        
        for (int i = 0; i < horizon; i++) {
            int p_x = start_x + static_cast<int>(optimal_x.at<float>(i, 0));
            int p_y = bounds.left_curve[i].y;
            trajectory_points.push_back(cv::Point(p_x, p_y));
        }

        return trajectory_points;
    }
}

#endif
