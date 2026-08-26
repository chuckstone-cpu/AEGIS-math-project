#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision_layer.hpp"

int main(int argc, char** argv) {
    // 1. Initialize cv2.VideoCapture
    std::string video_path = "../Train_set/dataset_video175.mp4";
    if (argc > 1) {
        video_path = argv[1];
    }

    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_path << "\n";
        return -1;
    }

    std::cout << "--- 1st Review Demo: Step 2 (PDE Smoothing) ---" << std::endl;
    std::cout << "Successfully opened video: " << video_path << std::endl;
    std::cout << "Starting The Processing Loop..." << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "End of video stream reached." << std::endl;
            break;
        }

        // Step 1: Ingestion / Grayscale
        cv::Mat grayscale_frame = Apex::apply_grayscale_transform(frame);

        // Step 2: PDE Smoothing
        // We set alpha to 0.1 as recommended in the syllabus.
        cv::Mat smoothed_frame = Apex::apply_pde_smoothing(grayscale_frame, 0.1f);

        // Display the results side by side
        cv::imshow("Step 1: Grayscale Output", grayscale_frame);
        cv::imshow("Step 2: PDE Smoothing (Heat Equation)", smoothed_frame);

        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q') {
            std::cout << "Loop terminated by user." << std::endl;
            break;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
