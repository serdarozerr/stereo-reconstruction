#pragma  once

#include <iostream>
#include <opencv2/core/types.hpp>
#include "Eigen.h"

struct KeyPoints2Dand3DCorrespondences {
    float point2DX = 0;
    float point2DY = 0;
    float point3DX = 0;
    float point3DY = 0;
    float point3DZ = 0;
    bool has3DPoint = false;

};

struct ETH3DFrame {
    cv::Mat frame;
    cv::Mat frameRectified;
    std::string frameName;
    std::string framePath;
    Eigen::Matrix4f cameraExtrinsic;
    Eigen::Matrix3f cameraIntrinsic;
    std::vector<KeyPoints2Dand3DCorrespondences> keyPoints2Dand3DCorrespondences;

};

const int FRAME_WIDTH = 6205;
const int FRAME_HEIGHT = 4135;
