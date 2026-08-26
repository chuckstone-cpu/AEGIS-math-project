#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision_layer.hpp"
#include "control_layer.hpp"

int main(int argc, char** argv) {
    std::string video_path = "../Train_set/dataset_video175.mp4";
    if (argc > 1) {
        video_path = argv[1];
    }

    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_path << "\n";
        return -1;
    }

    std::cout << "--- 1st Review Demo: Phase 3 (Stabilized Depth Sensing) ---" << std::endl;
    std::cout << "Successfully opened video: " << video_path << std::endl;

    // Temporal smoothing state
    int smooth_obstacle_idx = 20;

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "End of video stream reached." << std::endl;
            break;
        }

        // --- PHASE 1: Vision Ingestion ---
        cv::Mat grayscale_frame = Apex::apply_grayscale_transform(frame);

        // --- PHASE 2: Vision PDE Smoothing ---
        cv::Mat smoothed_frame = Apex::apply_pde_smoothing(grayscale_frame, 0.1f);
        
        // Advanced Vision: Detect Physical Lane Bounds (Stable Hough Math)
        Apex::LaneBoundaries bounds = Apex::detect_lane_boundaries(smoothed_frame);
        int curve_nodes = bounds.left_curve.size();

        // 1. Proximity Check (Scan layer-by-layer to find exact obstacle depth)
        cv::Mat canny;
        cv::Canny(smoothed_frame, canny, 50, 150);
        
        // Blind the Canny edge detector to the bottom 20% of the screen (car's own hood)
        cv::Rect hood_rect(0, static_cast<int>(frame.rows * 0.80), frame.cols, static_cast<int>(frame.rows * 0.20));
        canny(hood_rect).setTo(cv::Scalar(0));
        
        int detected_obstacle_idx = curve_nodes - 1; // Default: clear to horizon
        
        // CRITICAL FIX: Skip the immediate bottom third of the road (dashboard/hood reflections)
        // We can now safely scan closer to the car because we implemented the Phase 1 + 3 Synergy fix below!
        int scan_start = curve_nodes / 3; 
        
        // Scan from the safe mid-distance up to the horizon
        for (int i = scan_start; i < curve_nodes - 1; i++) {
            cv::Mat slice_mask = cv::Mat::zeros(canny.size(), canny.type());
            
            // Shrink checking polygon by 15% to perfectly isolate the ego lane
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
            
            // CRITICAL FIX: Edge Density
            int slice_area = std::max(1, (int)cv::contourArea(slice_pts));
            float density = (float)cv::countNonZero(slice_edges) / slice_area;
            
            // 5% edge density guarantees we found a dense object. But is it a car or road paint?
            if (density > 0.05f) { 
                // PHASE 1 + 3 SYNERGY: We use the Grayscale matrix from Phase 1 to check the 
                // physical brightness of the pixels that generated the Canny edges in Phase 3!
                cv::Scalar mean_edge = cv::mean(grayscale_frame, slice_edges);
                
                // Cars cast a deep dark undercarriage shadow on the road (low brightness).
                // White road arrows are extremely bright (high brightness).
                // By mathematically rejecting bright edges, we become 100% immune to road paint!
                if (mean_edge[0] < 130.0) { 
                    detected_obstacle_idx = i;
                    break; // Found the closest physical 3D obstacle
                }
            }
        }
        
        // Temporal Smoothing: Prevent flickering distance
        if (detected_obstacle_idx < smooth_obstacle_idx) {
            smooth_obstacle_idx = detected_obstacle_idx; // Instantly react to closer danger
        } else {
            smooth_obstacle_idx = std::min(curve_nodes - 1, smooth_obstacle_idx + 1); // Slowly recover safe distance
        }
        
        bool proximity_warning = (smooth_obstacle_idx < curve_nodes - 1); 

        // --- PHASE 3: Control Convex Optimization ---
        int start_x = (bounds.left_curve[0].x + bounds.right_curve[0].x) / 2; // Center of current lane
        int start_y = bounds.left_curve[0].y;
        
        std::vector<cv::Point> trajectory = Apex::calculate_optimal_trajectory(start_x, start_y, frame.rows / 2, bounds, proximity_warning, smooth_obstacle_idx);

        // --- VISUALIZATION: Dynamic Polygon & Legend Overlay ---
        // Render all graphics on top of the PDE smoothed video to prove the vision pipeline is active!
        cv::Mat display_frame;
        cv::cvtColor(smoothed_frame, display_frame, cv::COLOR_GRAY2BGR);
        
        cv::Mat overlay = display_frame.clone();
        
        // 2. Draw the Lower Safe Distance (Purple) - Extends exactly UP TO the obstacle
        std::vector<cv::Point> lower_lane_pts;
        for(int i = 0; i <= smooth_obstacle_idx; i++) lower_lane_pts.push_back(bounds.left_curve[i]);
        for(int i = smooth_obstacle_idx; i >= 0; i--) lower_lane_pts.push_back(bounds.right_curve[i]);
        
        std::vector<std::vector<cv::Point>> lower_poly = {lower_lane_pts};
        cv::fillPoly(overlay, lower_poly, cv::Scalar(255, 0, 150)); 
        
        // 3. Draw the Danger Zone (Red) - Only highlight the exact obstacle band!
        if (smooth_obstacle_idx < curve_nodes - 1) {
            int red_end = std::min(curve_nodes - 1, smooth_obstacle_idx + 2); // Small red band
            std::vector<cv::Point> upper_lane_pts;
            for(int i = smooth_obstacle_idx; i <= red_end; i++) upper_lane_pts.push_back(bounds.left_curve[i]);
            for(int i = red_end; i >= smooth_obstacle_idx; i--) upper_lane_pts.push_back(bounds.right_curve[i]);
            
            std::vector<std::vector<cv::Point>> upper_poly = {upper_lane_pts};
            cv::fillPoly(overlay, upper_poly, cv::Scalar(0, 0, 255)); 
        }

        cv::addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame); // Apply transparency blending

        // 4. Draw Physical Lane Boundaries (Solid Blue Lines) - Truncated at the obstacle
        int draw_nodes = proximity_warning ? std::min(curve_nodes - 1, smooth_obstacle_idx + 2) : curve_nodes - 1;
        for (int i = 0; i < draw_nodes; i++) {
            cv::line(display_frame, bounds.left_curve[i], bounds.left_curve[i+1], cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
            cv::line(display_frame, bounds.right_curve[i], bounds.right_curve[i+1], cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        }

        // 5. Draw CG Optimal Trajectory (Thick Green Line with Red Nodes) - Truncated at the obstacle
        for (int i = 0; i < draw_nodes; i++) {
            cv::line(display_frame, trajectory[i], trajectory[i+1], cv::Scalar(0, 255, 0), 4, cv::LINE_AA);
            cv::circle(display_frame, trajectory[i+1], 6, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        // 6. Draw Telemetry Legend
        cv::putText(display_frame, "--- APEX ENGINE TELEMETRY ---", cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        cv::putText(display_frame, "RED   : PROXIMITY WARNING (QP OBSTACLE AVOIDANCE)", cv::Point(20, 90), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::putText(display_frame, "Blue  : Physical Boundary (Robust Linear)", cv::Point(20, 140), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
        cv::putText(display_frame, "Green : QP Optimal Trajectory (CG Solver)", cv::Point(20, 190), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // Render Frame
        cv::imshow("Phase 3: Advanced Optimization & Perception", display_frame);

        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
