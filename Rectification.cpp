#include "Rectification.h"



/// Find best matches for keypoints in two camera images based on several matching methods
void Rectification::matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                                     std::vector<cv::KeyPoint> &kPtsRef,
                                     cv::Mat &descSource,
                                     cv::Mat &descRef,
                                     std::vector<cv::DMatch> &matches,
                                     std::string descriptorType,
                                     std::string matcherType,
                                     std::string selectorType) {

    /// configure matcher
    bool crossCheck = false;
    cv::Ptr <cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0) {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType.compare("MAT_FLANN") == 0) {
        if (descSource.type() != CV_32F) {
            /// OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        /// FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    /// perform matching task
    if (selectorType.compare("SEL_NN") == 0) {
        /// nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") == 0) {
        /// k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        /// filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }

# ifdef DEBUG
        std::cout << "Number of  Keypoints Removed After KNN Matche: " << knn_matches.size() - matches.size() << std::endl;
#endif

    }
}


/// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void Rectification::descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType) {
    // select appropriate descriptor
    cv::Ptr <cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    } else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();

    } else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::SIFT::create();
    } else {
        throw std::invalid_argument(descriptorType + "is not a valid type!!");
    }

    // perform feature description
    extractor->compute(img, keypoints, descriptors);

}



void Rectification::Rectify(ETH3DFrame *eth3DFrame1, ETH3DFrame *eth3DFrame2) {

    convert2DPointsToCvKeyPoints(eth3DFrame1, kp1);
    convert2DPointsToCvKeyPoints(eth3DFrame2, kp2);

#ifdef DEBUG
    std::cout << "Frame1 Keypoint Numbers: " << eth3DFrame1->keyPoints2Dand3DCorrespondences.size() << std::endl;
    std::cout << "Frame2 Keypoint Numbers:  " << eth3DFrame2->keyPoints2Dand3DCorrespondences.size() << std::endl;
#endif

    descKeypoints(kp1, eth3DFrame1->frame, descriptors1, "SIFT");
    descKeypoints(kp2, eth3DFrame2->frame, descriptors2, "SIFT");

    matchDescriptors(kp1, kp2, descriptors1, descriptors2, matches, "SIFT", "MAT_FLANN", "SEL_KNN");


    std::sort(matches.begin(), matches.end());
    int numGoodMatches = matches.size() * goodMatchesPercentage;
    matches.erase(matches.begin() + numGoodMatches, matches.end());


#ifdef DEBUG
    std::cout << "Number of Good Matches: "<< matches.size() << std::endl;
    std::cout << "KeyPoint Matches on Images are Saving..."<< std::endl;

    cv::Mat imMatches;
    drawMatches(eth3DFrame1->frame, kp1, eth3DFrame2->frame, kp2, matches, imMatches);
    imwrite("KeyPointMatches.jpg", imMatches);
#endif


    for (int i = 0; i < matches.size(); i++) {
        int query = matches.at(i).queryIdx;
        int train = matches.at(i).trainIdx;

        KeyPoints2Dand3DCorrespondences c1 = eth3DFrame1->keyPoints2Dand3DCorrespondences.at(query);
        KeyPoints2Dand3DCorrespondences c2 = eth3DFrame2->keyPoints2Dand3DCorrespondences.at(train);


        /// take points which have 3D correspondences
        /// checking the correction is easy when we have 3D correspondence
        if (c1.has3DPoint && c2.has3DPoint) {

            /// convert cv::KeyPoint to the cv::Point2f
            goodMatchesKeyPoints2D_1.push_back(cv::Point2f(kp1.at(query).pt.x, kp1.at(query).pt.y));
            goodMatchesKeyPoints2D_2.push_back(cv::Point2f(kp2.at(train).pt.x, kp2.at(train).pt.y));

        }

    }

#ifdef DEBUG
    std::cout<< "Number of Good Matches Which have 3D Point Correspondence: "<< goodMatchesKeyPoints2D_1.size()<<std::endl;
    std::cout<< "We will Use These Points For The Rest of The Calculations "<< std::endl;
#endif

    /// convert Eigen intrinsic to cv:Mat
    eigen2cv(eth3DFrame1->cameraIntrinsic, Kmatrix1);
    eigen2cv(eth3DFrame2->cameraIntrinsic, Kmatrix2);


    ////////////////////////////// Essential matrix  ////////////////////
    /// find  from corresponding points
    cv::Mat mask;
    essentialMatrix = cv::findEssentialMat(
                                            goodMatchesKeyPoints2D_1,
                                            goodMatchesKeyPoints2D_2,
                                            Kmatrix1,
                                            cv::Mat(),
                                            Kmatrix2,
                                            cv::Mat(),
                                            cv::RANSAC,
                                            0.999,
                                            1.0,
                                            mask
    );


#ifdef DEBUG
    /// SVD OF Essential matrix should be like;  singular1 = singual2, singula3 = 0
    cv::Mat w, u, vt;
    cv::SVD::compute(essentialMatrix, w, u, vt);
    std::cout << "Essential Matrix Singular: \n" << w << std::endl;
#endif


    ///////////////////////////////  recoverPose /////////////////////////////////
    /// find T vector and R matrix
    cv::recoverPose(essentialMatrix, goodMatchesKeyPoints2D_1, goodMatchesKeyPoints2D_2, Kmatrix1, R, T, mask);



    ////////////////////////////////////////////// find fundemental matrix ////////////////////////////
    /// calculate fundamental matrix from essential matrix F = K^-T . E . K^-1

    cv::Mat K_t_inv = Kmatrix1.inv().t();
    cv::Mat K_inv = Kmatrix1.inv();

    K_inv.convertTo(K_inv, CV_64F);
    K_t_inv.convertTo(K_t_inv, CV_64F);
    fundamentalMatrix = K_t_inv * essentialMatrix * K_inv;

#ifdef  DEBUG
    /// check fundemental matrix is correct,  singular1 and singular2 could be diff. and singular3 should be 0
    cv::Mat _w, _u, _vt;
    cv::SVD::compute(fundamentalMatrix.t(), _w, _u, _vt);
    std::cout <<"Fundamental Matrix Singular: \n" << _w << std::endl;
#endif


#ifdef  DEBUG
    ////////////////////////////////////////////// draw epipoler lines ////////////////////////////
    drawLine(eth3DFrame1->frame, eth3DFrame2->frame, goodMatchesKeyPoints2D_1, goodMatchesKeyPoints2D_2,
             fundamentalMatrix, -1);
#endif





///////////////////////////////////////////////// Rectification //////////////////////////

    cv::stereoRectify(Kmatrix1,
                      cv::Mat(),
                      Kmatrix2,
                      cv::Mat(),
                      eth3DFrame1->frame.size(),
                      R,
                      T,
                      R1,
                      R2,
                      P1,
                      P2,
                      Q
    );



    /// define 2 new frames for rectified frames
    cv::Mat frame1Remaped(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat frame2Remaped(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3, cv::Scalar(0, 0, 0));


    /// first image rectifiad by founded parameters R1, P1.
    /// find map correspondences for before rectf. and after rectf.
    cv::initUndistortRectifyMap(Kmatrix1, cv::Mat(), R1, P1, eth3DFrame1->frame.size(), CV_32FC1, map1_1, map1_2);
    cv::remap(eth3DFrame1->frame, frame1Remaped, map1_1, map1_2, cv::INTER_CUBIC);

#ifdef  DEBUG
    std::cout<< "Rectified Frame1 is Saving..."<<std::endl;
    cv::imwrite(eth3DFrame1->frameName + "_rectified1.png", frame1Remaped);
#endif



    /// second image rectifiad by founded parameters R2 ,P2
    /// find map correspondences for before rectf. and after rectf.
    cv::initUndistortRectifyMap(Kmatrix2, cv::Mat(), R2, P2, eth3DFrame2->frame.size(), CV_32FC1, map2_1, map2_2);
    cv::remap(eth3DFrame2->frame, frame2Remaped, map2_1, map2_2, cv::INTER_CUBIC);

#ifdef  DEBUG
    std::cout<< "Rectified Frame2 is Saving..."<<std::endl;
    cv::imwrite(eth3DFrame2->frameName + "_rectified2.png", frame2Remaped);
#endif



    eth3DFrame1->frameRectified = frame1Remaped.clone();
    eth3DFrame2->frameRectified = frame2Remaped.clone();
    frame1Remaped.release();
    frame2Remaped.release();

}
