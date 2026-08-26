#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision_layer.hpp"
#include "control_layer.hpp"

int main(int argc, char** argv) {
    std::string video_path = "../Train_set/dataset_video175.mp4";
    if (argc > 1) video_path = argv[1];

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    std::cout << "--- Phase 3: Trajectory & Perception ---\n";
    int smooth_obstacle_idx = 20;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat grayscale_frame = Apex::apply_grayscale_transform(frame);
        cv::Mat smoothed_frame = Apex::apply_pde_smoothing(grayscale_frame, 0.1f);
        
        Apex::LaneBoundaries bounds = Apex::detect_lane_boundaries(smoothed_frame);
        int curve_nodes = bounds.left_curve.size();

        cv::Mat canny;
        cv::Canny(smoothed_frame, canny, 50, 150);
        cv::Rect hood_rect(0, static_cast<int>(frame.rows * 0.80), frame.cols, static_cast<int>(frame.rows * 0.20));
        canny(hood_rect).setTo(cv::Scalar(0));
        
        int detected_obstacle_idx = curve_nodes - 1; 
        int scan_start = curve_nodes / 3; 
        
        for (int i = scan_start; i < curve_nodes - 1; i++) {
            cv::Mat slice_mask = cv::Mat::zeros(canny.size(), canny.type());
            
            int l_x1 = bounds.left_curve[i].x + (bounds.right_curve[i].x - bounds.left_curve[i].x) * 0.15;
            int r_x1 = bounds.right_curve[i].x - (bounds.right_curve[i].x - bounds.left_curve[i].x) * 0.15;
            int l_x2 = bounds.left_curve[i+1].x + (bounds.right_curve[i+1].x - bounds.left_curve[i+1].x) * 0.15;
            int r_x2 = bounds.right_curve[i+1].x - (bounds.right_curve[i+1].x - bounds.left_curve[i+1].x) * 0.15;

            std::vector<cv::Point> slice_pts = {
                cv::Point(l_x1, bounds.left_curve[i].y),
                cv::Point(l_x2, bounds.left_curve[i+1].y),
                cv::Point(r_x2, bounds.right_curve[i+1].y),
                cv::Point(r_x1, bounds.right_curve[i].y)
            };
            
            std::vector<std::vector<cv::Point>> polys = {slice_pts};
            cv::fillPoly(slice_mask, polys, cv::Scalar(255));
            
            cv::Mat slice_edges;
            cv::bitwise_and(canny, slice_mask, slice_edges);
            
            int slice_area = std::max(1, (int)cv::contourArea(slice_pts));
            float density = (float)cv::countNonZero(slice_edges) / slice_area;
            
            if (density > 0.05f) { 
                cv::Scalar mean_edge = cv::mean(grayscale_frame, slice_edges);
                if (mean_edge[0] < 130.0) { 
                    detected_obstacle_idx = i;
                    break;
                }
            }
        }
        
        if (detected_obstacle_idx < smooth_obstacle_idx) {
            smooth_obstacle_idx = detected_obstacle_idx; 
        } else {
            smooth_obstacle_idx = std::min(curve_nodes - 1, smooth_obstacle_idx + 1); 
        }
        
        bool proximity_warning = (smooth_obstacle_idx < curve_nodes - 1); 

        int start_x = (bounds.left_curve[0].x + bounds.right_curve[0].x) / 2;
        int start_y = bounds.left_curve[0].y;
        
        std::vector<cv::Point> trajectory = Apex::calculate_optimal_trajectory(start_x, start_y, frame.rows / 2, bounds, proximity_warning, smooth_obstacle_idx);

        cv::Mat display_frame;
        cv::cvtColor(smoothed_frame, display_frame, cv::COLOR_GRAY2BGR);
        cv::Mat overlay = display_frame.clone();
        
        std::vector<cv::Point> lower_lane_pts;
        for(int i = 0; i <= smooth_obstacle_idx; i++) lower_lane_pts.push_back(bounds.left_curve[i]);
        for(int i = smooth_obstacle_idx; i >= 0; i--) lower_lane_pts.push_back(bounds.right_curve[i]);
        
        std::vector<std::vector<cv::Point>> lower_poly = {lower_lane_pts};
        cv::fillPoly(overlay, lower_poly, cv::Scalar(255, 0, 150)); 
        
        if (smooth_obstacle_idx < curve_nodes - 1) {
            int red_end = std::min(curve_nodes - 1, smooth_obstacle_idx + 2);
            std::vector<cv::Point> upper_lane_pts;
            for(int i = smooth_obstacle_idx; i <= red_end; i++) upper_lane_pts.push_back(bounds.left_curve[i]);
            for(int i = red_end; i >= smooth_obstacle_idx; i--) upper_lane_pts.push_back(bounds.right_curve[i]);
            
            std::vector<std::vector<cv::Point>> upper_poly = {upper_lane_pts};
            cv::fillPoly(overlay, upper_poly, cv::Scalar(0, 0, 255)); 
        }

        cv::addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame); 

        int draw_nodes = proximity_warning ? std::min(curve_nodes - 1, smooth_obstacle_idx + 2) : curve_nodes - 1;
        for (int i = 0; i < draw_nodes; i++) {
            cv::line(display_frame, bounds.left_curve[i], bounds.left_curve[i+1], cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
            cv::line(display_frame, bounds.right_curve[i], bounds.right_curve[i+1], cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
            cv::line(display_frame, trajectory[i], trajectory[i+1], cv::Scalar(0, 255, 0), 4, cv::LINE_AA);
            cv::circle(display_frame, trajectory[i+1], 6, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        cv::putText(display_frame, "APEX TELEMETRY", cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        cv::putText(display_frame, "RED: PROXIMITY WARNING", cv::Point(20, 90), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        
        cv::imshow("Phase 3", display_frame);

        if ((char)cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
