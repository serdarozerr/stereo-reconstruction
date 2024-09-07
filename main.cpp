#include <iostream>
#include <array>
#include "Rectification.h"
#include "VirtualSensor.h"
#include "stereo_reconstruction.h"

int main() {
    /// Make sure this path points to the data folder
    std::string filenameIn = "../../Data/ETH3/";


    /// Load Calibration Files
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }


    std::string processFrames[2] = {"DSC_0288.JPG", "DSC_0289.JPG"};
    ETH3DFrame* eth3DImage1;
    ETH3DFrame* eth3DImage2;



    eth3DImage1= sensor.readFrame(processFrames[0]);
    eth3DImage2= sensor.readFrame(processFrames[1]);

    Rectification r;
    r.Rectify(eth3DImage1,eth3DImage2);
    cv::imwrite(filenameIn + "rec1.png", eth3DImage1->frameRectified);
    cv::imwrite(filenameIn + "rec2.png", eth3DImage2->frameRectified);
    StereoReconstructor s = StereoReconstructor(*eth3DImage1, *eth3DImage2);
    cv::Mat disparityMap = s.getDisparity(new PatchMatch());
    cv::imwrite(filenameIn + "disparity.png", disparityMap);
    return 0;

}
