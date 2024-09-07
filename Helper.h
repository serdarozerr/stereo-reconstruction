#pragma  once

#include "Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FrameDataTypes.h"


void convert2DPointsToCvKeyPoints(const ETH3DFrame *eth3DFrame, std::vector<cv::KeyPoint> &kp);


float distancePointLine(const cv::Point2f point, const cv::Vec<float, 3> &line);


void drawLine(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> points1,
              std::vector<cv::Point2f> points2, cv::Mat F, float inlierDistance);


void checkCameraMatrices(const ETH3DFrame *eth3DFrame1, const ETH3DFrame *eth3DFrame2, std::vector<cv::DMatch> matches);

void checkRotationAndTranslationMatrices(const cv::Mat R, const cv::Mat T, const std::vector<cv::DMatch> matches,
                                         const ETH3DFrame *eth3DFrame1, const ETH3DFrame *eth3DFrame2);


