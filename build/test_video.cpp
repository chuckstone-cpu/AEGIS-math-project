#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main() {
    VideoCapture cap("C:/Users/SATHYANARAYANAN.T/My_Docs/SEM 3/MFC-3/Updated project/Train_set/dataset_video175.mp4");
    if (!cap.isOpened()) {
        cout << "FAILED" << endl;
        return 1;
    }
    cout << "SUCCESS" << endl;
    return 0;
}
