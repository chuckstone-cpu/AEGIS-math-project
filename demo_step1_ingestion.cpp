#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision_layer.hpp"

int main(int argc, char** argv) {
    // 1. Initialize cv2.VideoCapture
    // We will use the converted H.264 version of the video by default
    // We use ../ to account for running this from the /build directory
    std::string video_path = "../Train_set/dataset_video175.mp4";
    
    // Allow user to pass a custom video via command line
    if (argc > 1) {
        video_path = argv[1];
    }

    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_path << "\n";
        std::cerr << "Make sure OpenCV is built with FFmpeg/GStreamer support for .hevc files." << std::endl;
        return -1;
    }

    std::cout << "--- 1st Review Demo: Step 1 (Ingestion Loop) ---" << std::endl;
    std::cout << "Successfully opened video: " << video_path << std::endl;
    std::cout << "Starting The Ingestion Loop..." << std::endl;

    // 2. Start a while(True) loop to grab one frame at a time
    while (true) {
        cv::Mat frame;
        cap >> frame; // Grab one frame at a time

        // If the frame is empty, we reached the end of the video
        if (frame.empty()) {
            std::cout << "End of video stream reached." << std::endl;
            break;
        }

        // 3. & 4. & 5. The Math: Convert 3D Matrix to 2D Matrix
        cv::Mat grayscale_frame = Apex::apply_grayscale_transform(frame);

        // Display the results so we can visualize our mathematical output
        cv::imshow("Original Color (3D Matrix)", frame);
        cv::imshow("Mathematical Grayscale (2D Grid)", grayscale_frame);

        // Wait for 1 millisecond and check if the user pressed 'q' to quit
        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q') {
            std::cout << "Ingestion Loop terminated by user." << std::endl;
            break;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
