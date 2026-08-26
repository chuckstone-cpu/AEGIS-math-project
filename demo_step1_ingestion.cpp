#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision_layer.hpp"

int main(int argc, char** argv) {
    std::string video_path = "../Train_set/dataset_video175.mp4";
    if (argc > 1) video_path = argv[1];

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    std::cout << "--- 1st Review Demo: Step 1 (Ingestion Loop) ---\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat grayscale_frame = Apex::apply_grayscale_transform(frame);

        cv::imshow("Original Color", frame);
        cv::imshow("Grayscale", grayscale_frame);

        if ((char)cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
