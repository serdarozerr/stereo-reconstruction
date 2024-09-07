#include "Helper.h"

void convert2DPointsToCvKeyPoints(const ETH3DFrame* eth3DFrame ,  std::vector<cv::KeyPoint>& kp ){

    for (int i = 0; i < eth3DFrame->keyPoints2Dand3DCorrespondences.size(); i++) {


        kp.push_back(cv::KeyPoint(eth3DFrame->keyPoints2Dand3DCorrespondences.at(i).point2DX,
                                  eth3DFrame->keyPoints2Dand3DCorrespondences.at(i).point2DY, 10, -1, 0, 0, i));


    }

}


float distancePointLine(const cv::Point2f point, const cv::Vec<float, 3> &line) {
    //Line is given as a*x + b*y + c = 0
    return std::fabs(line(0) * point.x + line(1) * point.y + line(2)) /
           std::sqrt(line(0) * line(0) + line(1) * line(1));
}



void drawLine(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, cv::Mat F, float inlierDistance = -1) {

    std::cout<<"Epiploar Lines are Drawning on Image and Saving ..."<<std::endl;

    cv::Rect rect1(0, 0, img1.cols, img1.rows);
    cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
    cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);

    std::vector<cv::Vec < float, 3 >> epilines1, epilines2;

    //// F * x1 = lines2  ,  x2^t * F = lines1
    cv::computeCorrespondEpilines(points1, 1, F, epilines2); //Index starts with 1
    cv::computeCorrespondEpilines(points2, 2, F, epilines1);

    cv::RNG rng(0);
    for (size_t i = 0; i < points1.size(); i++) {


        /////////////////// check fundamental matrix calculated corect
        /*
        cv::Mat p1Mat = (cv::Mat_<float>(3,1) <<points1[i].x, points1[i].y,  1);
        cv::Mat p2Mat = (cv::Mat_<float>(3,1) <<points2[i].x, points2[i].y,  1);

        Eigen::Vector<float, Eigen::Dynamic> eigenv1;
        Eigen::Vector<float, Eigen::Dynamic> eigenv2;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> fun;

        cv::cv2eigen(p1Mat, eigenv1);
        cv::cv2eigen(p2Mat, eigenv2);
        cv::cv2eigen(F, fun);

        cout <<  eigenv2.transpose() * fun * eigenv1 << endl;

        */


        if (inlierDistance > 0) {
            if (distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
                distancePointLine(points2[i], epilines1[i]) > inlierDistance) {
                //The point match is no inlier
                std::cout << " not matched" <<std::endl;
                continue;
            }
        }

        /// Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
        /// using ax+ by +cz = 0  ===>
        cv::Scalar color(rng(256), rng(256), rng(256));

        cv::line(img1,
                 cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
                 cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
                 color, 5);
        cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);

        cv::line(img2,
                 cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
                 cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
                 color, 5);
        cv::circle(outImg(rect2), points2[i], 3, color, -1, cv::LINE_AA);
    }

    cv::imwrite("epipole_frame1.png", img1);
    cv::imwrite("epipole_frame2.png", img2);

}




void checkCameraMatrices(const ETH3DFrame* eth3DFrame1 , const ETH3DFrame* eth3DFrame2 ,  std::vector<cv::DMatch> matches ){
/// check both intrinsic and extrinsic are correct

    for (int i = 0; i < matches.size(); i++) {
        int query = matches.at(i).queryIdx;
        int train = matches.at(i).trainIdx;

        KeyPoints2Dand3DCorrespondences c1 = eth3DFrame1->keyPoints2Dand3DCorrespondences.at(query);
        KeyPoints2Dand3DCorrespondences c2 = eth3DFrame2->keyPoints2Dand3DCorrespondences.at(train);


        if (c1.has3DPoint && c2.has3DPoint) {

            /// we check the 3d point location of matched 2d key points
//            cout << c1.point3DX << " " << c1.point3DY << " " << c1.point3DZ << " " << std::endl;
//            cout << c2.point3DX << " " << c2.point3DY << " " << c2.point3DZ << " " << std::endl;


            /// from 3d world coordinate --->    2d pixel location,
            /// not relevant just for checking the Extrinsic and Intrinsic matrices working okey


            Eigen::Vector4f pointWorld;
            Eigen::Vector4f pointCamera;
            Eigen::Vector3f pointCamera3d;
            Eigen::Vector3f pointFrame;


            pointWorld << c2.point3DX, c2.point3DY, c2.point3DZ, 1;
//            Eigen::Vector3f vec(c1.point3DX , c1.point3DY , c1.point3DZ);
            pointCamera = eth3DFrame1->cameraExtrinsic * pointWorld;
            pointCamera3d << pointCamera(0), pointCamera(1), pointCamera(2);

            pointFrame = (eth3DFrame2->cameraIntrinsic * pointCamera3d);
            pointFrame = pointFrame / pointFrame(2);

            std::cout<<"real 2d point "<< c2.point2DX << " " << c2.point2DY << " --- " << " calculated 2d point "<< pointFrame(0) << " "<< pointFrame(1) <<std::endl;



        }

    }
}

void checkRotationAndTranslationMatrices(const cv::Mat R, const cv::Mat T, const std::vector<cv::DMatch> matches , const ETH3DFrame* eth3DFrame1, const ETH3DFrame* eth3DFrame2 ){
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenRotation;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenTranslation;

    cv::cv2eigen(R, eigenRotation);
    cv::cv2eigen(T, eigenTranslation);

    float diff_x = 0;
    float diff_y = 0;


    int points_3d_count = 0;
    for (int i = 0; i < matches.size(); i++) {
        int query = matches.at(i).queryIdx;
        int train = matches.at(i).trainIdx;

        KeyPoints2Dand3DCorrespondences c1 = eth3DFrame1->keyPoints2Dand3DCorrespondences.at(query);
        KeyPoints2Dand3DCorrespondences c2 = eth3DFrame2->keyPoints2Dand3DCorrespondences.at(train);

        if (c1.has3DPoint && c2.has3DPoint) {
            points_3d_count++;

            Eigen::Vector4f pointWorld;
            Eigen::Vector4f pointCamera1_4d;
            Eigen::Vector3f pointCamera1_3d;
            Eigen::Vector3f pointCamera2_3d;
            Eigen::Vector2f pointFrame2_2d;


            pointWorld << c1.point3DX, c1.point3DY, c1.point3DZ, 1;
            pointCamera1_4d = eth3DFrame1->cameraExtrinsic * pointWorld;
            pointCamera1_3d << pointCamera1_4d(0), pointCamera1_4d(1), pointCamera1_4d(2);

            pointCamera2_3d = eth3DFrame2->cameraIntrinsic * (eigenRotation * pointCamera1_3d + eigenTranslation);
            pointFrame2_2d << pointCamera2_3d(0) / pointCamera2_3d(2), pointCamera2_3d(1) / pointCamera2_3d(2);

            diff_x += fabs((c2.point2DX - pointFrame2_2d(0)));
            diff_y += fabs((c2.point2DY - pointFrame2_2d(1)));

            std::cout << "2D point real: " << c2.point2DX << " " << c2.point2DY << "  2D point transformed: " << pointFrame2_2d(0) << " "
                      << pointFrame2_2d(1) << std::endl;
        }
    }

    float mean_error_on_x = diff_x / points_3d_count;
    float mean_error_on_y = diff_y / points_3d_count;

    std::cout << "mean pixel error  on x : " << mean_error_on_x << std::endl;
    std::cout << "mean pixel error  on y : " << mean_error_on_y << std::endl;

    std::cout << "percantage error  on x : " << mean_error_on_x * 100 / 6205 << std::endl;
    std::cout << "percantage error on y : " << mean_error_on_y * 100 / 4135 << std::endl;
}
