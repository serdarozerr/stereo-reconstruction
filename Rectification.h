#pragma  once

#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Helper.h"

//#define DEBUG



class Rectification {

public:

    void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
    void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                          cv::Mat &descRef, std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

    void Rectify(ETH3DFrame* _eth3DFrame1 ,ETH3DFrame* _eth3DFrame2);


private:

    std::vector<cv::KeyPoint> kp1;
    std::vector<cv::KeyPoint> kp2;


    cv::Mat descriptors1;
    cv::Mat descriptors2;
    std::vector<cv::DMatch> matches;




    /// these intrinsic matrices should be
    /// the same to algorithm work properly
    cv::Mat Kmatrix1;
    cv::Mat Kmatrix2;


    double goodMatchesPercentage=0.1;
    std::vector<cv::Point2f> goodMatchesKeyPoints2D_1;
    std::vector<cv::Point2f> goodMatchesKeyPoints2D_2;


    /// R = Rotation , T =Translation matrices
    cv::Mat fundamentalMatrix;
    cv::Mat essentialMatrix;
    cv::Mat R, T;


    /// cv::Rectification parameters
    /// R1 and R2 are rotation  matrices for frame1 and frame2
    /// P1 and P2 ara projection matrices(intrinsic) for frame1 and frame2
    cv::Mat R1, R2, P1, P2, Q;




    /// map for frame1 before and after rectification
    /// include correspondence location.
    cv::Mat map2_1, map2_2;
    cv::Mat map1_1, map1_2;



};


