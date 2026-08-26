#ifndef VISION_LAYER_HPP
#define VISION_LAYER_HPP

#include <opencv2/opencv.hpp>

namespace Apex {

    cv::Mat apply_grayscale_transform(const cv::Mat& color_frame) {
        cv::Mat grayscale_frame(color_frame.rows, color_frame.cols, CV_8UC1);

        for (int y = 0; y < color_frame.rows; ++y) {
            for (int x = 0; x < color_frame.cols; ++x) {
                cv::Vec3b pixel = color_frame.at<cv::Vec3b>(y, x);
                float intensity = (0.2989f * pixel[2]) + (0.5870f * pixel[1]) + (0.1140f * pixel[0]);
                
                if (intensity > 255.0f) intensity = 255.0f;
                if (intensity < 0.0f) intensity = 0.0f;
                
                grayscale_frame.at<uchar>(y, x) = static_cast<uchar>(intensity);
            }
        }
        return grayscale_frame;
    }

    cv::Mat apply_pde_smoothing(const cv::Mat& grayscale_frame, float alpha = 0.1f) {
        cv::Mat smoothed_frame = grayscale_frame.clone();

        for (int y = 1; y < grayscale_frame.rows - 1; ++y) {
            for (int x = 1; x < grayscale_frame.cols - 1; ++x) {
                float P_old = static_cast<float>(grayscale_frame.at<uchar>(y, x));
                float P_Top = static_cast<float>(grayscale_frame.at<uchar>(y - 1, x));
                float P_Bottom = static_cast<float>(grayscale_frame.at<uchar>(y + 1, x));
                float P_Left = static_cast<float>(grayscale_frame.at<uchar>(y, x - 1));
                float P_Right = static_cast<float>(grayscale_frame.at<uchar>(y, x + 1));

                float P_new = P_old + alpha * (P_Top + P_Bottom + P_Left + P_Right - 4.0f * P_old);

                if (P_new > 255.0f) P_new = 255.0f;
                if (P_new < 0.0f) P_new = 0.0f;

                smoothed_frame.at<uchar>(y, x) = static_cast<uchar>(P_new);
            }
        }
        return smoothed_frame;
    }

    struct LaneBoundaries {
        std::vector<cv::Point> left_curve;
        std::vector<cv::Point> right_curve;
    };

    LaneBoundaries detect_lane_boundaries(const cv::Mat& smoothed_frame) {
        LaneBoundaries bounds;
        cv::Mat edges;
        cv::Canny(smoothed_frame, edges, 50, 150);

        int height = edges.rows;
        int width = edges.cols;
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

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(roi_edges, lines, 1, CV_PI/180, 50, 50, 150);

        std::vector<cv::Vec4i> left_lines, right_lines;
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            double slope = (double)(l[3] - l[1]) / (l[2] - l[0] + 0.0001); 
            if (std::abs(slope) < 0.3) continue;
            if (slope < 0) left_lines.push_back(l); 
            else right_lines.push_back(l);          
        }

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

        double left_slope = 0, left_intercept = 0, right_slope = 0, right_intercept = 0;
        bool left_valid = average_line(left_lines, left_slope, left_intercept);
        bool right_valid = average_line(right_lines, right_slope, right_intercept);

        int num_steps = 20;
        float step = (float)(horizon_y - height) / num_steps; 

        for (int i = 0; i <= num_steps; i++) {
            float y = height + i * step; 
            
            int x_left = 0, x_right = 0;
            if (left_valid) {
                x_left = static_cast<int>((y - left_intercept) / left_slope);
            } else {
                float progress = (float)i / num_steps;
                x_left = static_cast<int>(width*0.2 + progress * (width*0.45 - width*0.2));
            }
            bounds.left_curve.push_back(cv::Point(x_left, static_cast<int>(y)));

            if (right_valid) {
                x_right = static_cast<int>((y - right_intercept) / right_slope);
            } else {
                float progress = (float)i / num_steps;
                x_right = static_cast<int>(width*0.8 + progress * (width*0.55 - width*0.8));
            }
            bounds.right_curve.push_back(cv::Point(x_right, static_cast<int>(y)));
        }

        return bounds;
    }
}

#endif
