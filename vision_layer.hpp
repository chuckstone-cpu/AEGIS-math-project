#ifndef VISION_LAYER_HPP
#define VISION_LAYER_HPP

#include <opencv2/opencv.hpp>

namespace Apex {

    // Step 1: Flatten 3D Color Matrix to 2D Grayscale Matrix
    // The Input:  F ∈ R^(W × H × 3)
    // The Output: I ∈ R^(W × H)
    cv::Mat apply_grayscale_transform(const cv::Mat& color_frame) {
        // Pre-allocate the 2D output matrix (single channel, 8-bit unsigned integer)
        cv::Mat grayscale_frame(color_frame.rows, color_frame.cols, CV_8UC1);

        // The Math: Loop through the matrix (Height and Width)
        for (int y = 0; y < color_frame.rows; ++y) {
            for (int x = 0; x < color_frame.cols; ++x) {
                // Extract B, G, R scalar values. (OpenCV uses BGR natively)
                cv::Vec3b pixel = color_frame.at<cv::Vec3b>(y, x);
                float B = pixel[0];
                float G = pixel[1];
                float R = pixel[2];

                // The Linear Transformation (Dot Product): 
                // Intensity = (0.2989 * R) + (0.5870 * G) + (0.1140 * B)
                float intensity = (0.2989f * R) + (0.5870f * G) + (0.1140f * B);

                // Ensure the intensity stays within the 0-255 bound
                if (intensity > 255.0f) intensity = 255.0f;
                if (intensity < 0.0f) intensity = 0.0f;

                // Assign to our new 2D grid
                grayscale_frame.at<uchar>(y, x) = static_cast<uchar>(intensity);
            }
        }

        return grayscale_frame;
    }
    // Step 2: PDE Smoothing (Heat Equation)
    // Layman: "Virtual heat" that melts away sharp, random noise.
    // The Math: Finite Difference Method stencil.
    cv::Mat apply_pde_smoothing(const cv::Mat& grayscale_frame, float alpha = 0.1f) {
        // Create a copy of the current frame matrix
        cv::Mat smoothed_frame = grayscale_frame.clone();

        // Loop through every pixel (excluding the very edges to avoid out-of-bounds error)
        for (int y = 1; y < grayscale_frame.rows - 1; ++y) {
            for (int x = 1; x < grayscale_frame.cols - 1; ++x) {
                // Look at its neighbors (we cast to float to prevent mathematical overflow during the calculation)
                float P_old = static_cast<float>(grayscale_frame.at<uchar>(y, x));
                float P_Top = static_cast<float>(grayscale_frame.at<uchar>(y - 1, x));
                float P_Bottom = static_cast<float>(grayscale_frame.at<uchar>(y + 1, x));
                float P_Left = static_cast<float>(grayscale_frame.at<uchar>(y, x - 1));
                float P_Right = static_cast<float>(grayscale_frame.at<uchar>(y, x + 1));

                // The Math: Pnew = Pold + alpha * (PTop + PBottom + PLeft + PRight - 4 * Pold)
                float P_new = P_old + alpha * (P_Top + P_Bottom + P_Left + P_Right - 4.0f * P_old);

                // Ensure the calculated intensity stays within the 0-255 bound
                if (P_new > 255.0f) P_new = 255.0f;
                if (P_new < 0.0f) P_new = 0.0f;

                // Update the pixel in the new matrix
                smoothed_frame.at<uchar>(y, x) = static_cast<uchar>(P_new);
            }
        }

        return smoothed_frame;
    }

    // Lane Boundaries Structure (Provides discrete nodes for the QP Solver)
    struct LaneBoundaries {
        std::vector<cv::Point> left_curve;
        std::vector<cv::Point> right_curve;
    };

    // Advanced Vision Feature: Detect physical lane lines (Robust Linear)
    LaneBoundaries detect_lane_boundaries(const cv::Mat& smoothed_frame) {
        LaneBoundaries bounds;

        // 1. Canny Edge Detection
        cv::Mat edges;
        cv::Canny(smoothed_frame, edges, 50, 150);

        int height = edges.rows;
        int width = edges.cols;

        // 2. Region of Interest (ROI) Mask - Extended to true horizon (53%)
        // Crucial Fix: Narrow the bottom width to strictly capture the EGO lane (ignore adjacent lanes)
        int bottom_y = static_cast<int>(height * 0.9);
        int horizon_y = static_cast<int>(height * 0.53);
        
        cv::Mat mask = cv::Mat::zeros(edges.size(), edges.type());
        cv::Point pts[4] = {
            cv::Point(static_cast<int>(width * 0.2), bottom_y),
            cv::Point(static_cast<int>(width * 0.45), horizon_y),
            cv::Point(static_cast<int>(width * 0.55), horizon_y),
            cv::Point(static_cast<int>(width * 0.8), bottom_y)
        };
        cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255));
        cv::Mat roi_edges;
        cv::bitwise_and(edges, mask, roi_edges);

        // 3. Robust Hough Line Transform (Replaces unstable polyfit)
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(roi_edges, lines, 1, CV_PI/180, 50, 50, 150);

        // 4. Sort lines into Left and Right lanes based on slope
        std::vector<cv::Vec4i> left_lines, right_lines;
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            double slope = (double)(l[3] - l[1]) / (l[2] - l[0] + 0.0001); 
            if (std::abs(slope) < 0.3) continue; // Ignore horizontal noise
            if (slope < 0) left_lines.push_back(l); 
            else right_lines.push_back(l);          
        }

        // 5. Average the slopes and intercepts
        auto average_line = [&](const std::vector<cv::Vec4i>& lns, double& out_slope, double& out_intercept) {
            if (lns.empty()) return false;
            double avg_slope = 0, avg_intercept = 0;
            for (const auto& l : lns) {
                double slope = (double)(l[3] - l[1]) / (l[2] - l[0] + 0.0001);
                double intercept = l[1] - slope * l[0];
                avg_slope += slope;
                avg_intercept += intercept;
            }
            out_slope = avg_slope / lns.size();
            out_intercept = avg_intercept / lns.size();
            return true;
        };

        double left_slope = 0, left_intercept = 0;
        double right_slope = 0, right_intercept = 0;
        bool left_valid = average_line(left_lines, left_slope, left_intercept);
        bool right_valid = average_line(right_lines, right_slope, right_intercept);

        // 6. Generate discrete node arrays from bottom of screen to horizon for the QP solver
        // We still generate vectors so the multi-zone polygon and QP swerving continue to work perfectly!
        int num_steps = 20;
        float step = (float)(horizon_y - height) / num_steps; 

        for (int i = 0; i <= num_steps; i++) {
            float y = height + i * step; // Start at physical bottom (height), step up to horizon
            
            // Left Boundary Point
            int x_left = 0;
            if (left_valid) {
                x_left = static_cast<int>((y - left_intercept) / left_slope);
            } else {
                // Fallback straight line
                float progress = (float)i / num_steps;
                x_left = static_cast<int>(width*0.2 + progress * (width*0.45 - width*0.2));
            }
            bounds.left_curve.push_back(cv::Point(x_left, static_cast<int>(y)));

            // Right Boundary Point
            int x_right = 0;
            if (right_valid) {
                x_right = static_cast<int>((y - right_intercept) / right_slope);
            } else {
                // Fallback straight line
                float progress = (float)i / num_steps;
                x_right = static_cast<int>(width*0.8 + progress * (width*0.55 - width*0.8));
            }
            bounds.right_curve.push_back(cv::Point(x_right, static_cast<int>(y)));
        }

        return bounds;
    }

} // namespace Apex

#endif // VISION_LAYER_HPP
