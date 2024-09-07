#pragma once

#include <vector>
#include <map>
#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include "Eigen.h"
#include "FrameDataTypes.h"

#include <iostream>
#include <stdexcept>

typedef unsigned char BYTE;


// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats

//
//struct KeyPoints2Dand3DCorrespondences {
//    float point2DX = 0;
//    float point2DY = 0;
//    float point3DX = 0;
//    float point3DY = 0;
//    float point3DZ = 0;
//    bool has3DPoint = false;
//
//};
//
//struct ETH3DFrame {
//    cv::Mat frame;
//    std::string frameName;
//    std::string framePath;
//    Eigen::Matrix4f cameraExtrinsic;
//    Eigen::Matrix3f cameraIntrinsic;
//    std::vector<KeyPoints2Dand3DCorrespondences> keyPoints2Dand3DCorrespondences;
//
//};


class VirtualSensor {
public:

    VirtualSensor(){

    }

    ~VirtualSensor() {
        for (int i = 0; i < eth3DFrames.size(); i++) {

            ETH3DFrame *e = eth3DFrames.at(i);

            if (e->frame.empty() == false) {
                e->frame.release();
            }
            delete e;
        }
    }

    bool Init(const std::string &datasetDir) {
        baseDir = datasetDir;

        // read cameras instirinsics
        if (!ReadCamerasInstiricts(datasetDir + "dslr_calibration_undistorted/cameras.txt", camerasIntrinsics))
            return false;

        // read given 3D points
        if (!read3DPoints(datasetDir + "dslr_calibration_undistorted/points3D.txt", points3D))
            return false;

        // read 2D keypoints and put all things together into ETH3DFrame struct
        if (!ReadTrajectoryFile(datasetDir + "dslr_calibration_undistorted/images.txt", camerasIntrinsics, points3D, eth3DFrames))
            return false;


        return true;
    }

    int findFrameIndex(const std::string &frameName, std::vector<ETH3DFrame *> _eth3DFrames) {

        int j = 0;
        while (_eth3DFrames.at(j)->frameName != frameName) {
            j++;
        }

        return j >= _eth3DFrames.size() ? -1 : j;
    }


    ETH3DFrame *readFrame(const std::string& _frameName) {
        try {
            int frameIndex = findFrameIndex(_frameName, eth3DFrames);
            if(frameIndex == -1)
                throw  std::out_of_range("frame name not found in the vector");

            ETH3DFrame *eth3DFrame = eth3DFrames.at(frameIndex);
            eth3DFrame->frame = cv::imread("../../Data/ETH3/" + eth3DFrame->framePath, 1);
            return eth3DFrame;
        }
        catch (const std::out_of_range& e){

            std::cerr<< "You provide wrong frame name" << e.what() << std::endl;
            std::exit(1);
        }
        catch (...){
            std::cerr<< "image file not found" << std::endl;
            std::exit(1);
        }
    }


private:

    bool ReadCamerasInstiricts(const std::string &filename, std::map<int, Eigen::Matrix3f> &_camerasIntrinsics) {
        std::ifstream file(filename, std::ios::in);
        if (!file.is_open()) return false;


        // read comment line first
        std::string dump;
        std::getline(file, dump);
        std::getline(file, dump);
        std::getline(file, dump);


        int cameraId = 0;
        std::string cameraType;
        int frameWidth;
        int frameHeight;

        // get intrinsic values
        while (file.good()) {

            Eigen::Vector4f focal_length_and_principle_point;

            file >> cameraId >>
                 cameraType >>
                 frameWidth >>
                 frameHeight >>
                 focal_length_and_principle_point.x() >>
                 focal_length_and_principle_point.y() >>
                 focal_length_and_principle_point.z() >>
                 focal_length_and_principle_point.w();


            Eigen::Matrix3f intrinsic;
            intrinsic.setIdentity();

            intrinsic(0, 0) = focal_length_and_principle_point.x();
            intrinsic(1, 1) = focal_length_and_principle_point.y();
            intrinsic(0, 2) = focal_length_and_principle_point.z();
            intrinsic(1, 2) = focal_length_and_principle_point.w();


            _camerasIntrinsics[cameraId] = intrinsic;

        }

        file.close();
        return true;
    }


    bool read3DPoints(const std::string &filename, std::map<int, std::vector<float>> &_points3D) {
        std::ifstream file(filename, std::ios::in);
        if (!file.is_open()) return false;


        std::string dump;
        std::getline(file, dump);
        std::getline(file, dump);
        std::getline(file, dump);


        int point3d_id = 0;
        float pointx = 0;
        float pointy = 0;
        float pointz = 0;

        while (file.good()) {

            std::vector<float> point3d;

            file >> point3d_id >>
                 pointx >>
                 pointy >>
                 pointz;

            //ignore remaning part until the end of the line
            std::getline(file, dump);

            point3d.push_back(pointx);
            point3d.push_back(pointy);
            point3d.push_back(pointz);

            _points3D[point3d_id] = point3d;
        }

        file.close();
        return true;


    }


    void parseString(const std::string &_delimiter, std::string line, std::vector<std::string> &_result) {

        std::size_t pos_start = 0, pos_end, delim_len = _delimiter.length();
        std::string token;

        while ((pos_end = line.find(_delimiter, pos_start)) != std::string::npos) {
            token = line.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            _result.push_back(token);
        }
        _result.push_back(line.substr(pos_start));

    }


    void Add2DKeyPointsAndCorresponding3DPointToETH3DImageStruct(const std::map<int, std::vector<float>> points3D,
                                                                 ETH3DFrame *eth3DImage,
                                                                 std::string keyPointsLine) {


        std::string delimiter = " ";
        std::vector<std::string> result;

        parseString(delimiter, keyPointsLine, result);

        float point2dX;
        float point2dY;
        int point3dId;

        try {
            for (int i = 0; (i + 3) < result.size(); i += 3) {
                KeyPoints2Dand3DCorrespondences kc;

                kc.point2DX = std::stof(result.at(i));
                kc.point2DY = std::stof(result.at(i + 1));
                point3dId = std::stoi(result.at(i + 2));

                /// -1 means no corresponded 3d point for this key point
                if (point3dId == -1) {
                    kc.has3DPoint = false;
                } else {

                    try {

                        std::vector<float> corresponding3DPoints = points3D.at(point3dId);
                        kc.point3DX = corresponding3DPoints.at(0);
                        kc.point3DY = corresponding3DPoints.at(1);
                        kc.point3DZ = corresponding3DPoints.at(2);
                        kc.has3DPoint = true;
                    }
                    catch (const std::out_of_range &e) {
                        std::cerr << "Out of Range error: " << e.what() << '\n';
                        kc.has3DPoint = false;
                    }
                }

                eth3DImage->keyPoints2Dand3DCorrespondences.push_back(kc);

            }
        }
        catch (const std::out_of_range &e) {
            std::cerr << "Out of Range error: " << e.what() << '\n';

        }


    }


    bool ReadTrajectoryFile(const std::string &filename, const std::map<int, Eigen::Matrix3f> &_camerasIntrinsics,
                            const std::map<int, std::vector<float>> &_points3D,
                            std::vector<ETH3DFrame *> &_eth3DFrames) {

        std::ifstream file(filename, std::ios::in);
        if (!file.is_open()) return false;


        // comment lines,  skipp it
        std::string dump;
        std::getline(file, dump);
        std::getline(file, dump);
        std::getline(file, dump);
        std::getline(file, dump);


        int image_id;
        int camera_id;
        std::string frame_path;

        while (file.good()) {
            Eigen::Vector3f translation;
            Eigen::Quaternionf rot;

            ETH3DFrame *eth3DImage = new ETH3DFrame();


            file >> image_id >>
                 rot.w() >>
                 rot.x() >>
                 rot.y() >>
                 rot.z() >>
                 translation.x() >>
                 translation.y() >>
                 translation.z() >>
                 camera_id >>
                 frame_path;


            //transf matrix from the global coordinate system to the image's local coordinate system
            Eigen::Matrix4f transf;
            transf.setIdentity();
            transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
            transf.block<3, 1>(0, 3) = translation;


            // reading empty line
            std::getline(file, dump);

            // this line is the keypoint observations of this frame
            std::getline(file, dump);

            // read 2D key points from string line and  get 3D correspondence point for
            // the current frame and put it into eth3DImage struct
            Add2DKeyPointsAndCorresponding3DPointToETH3DImageStruct(_points3D, eth3DImage, dump);
            eth3DImage->cameraExtrinsic = transf;
            eth3DImage->cameraIntrinsic = _camerasIntrinsics.at(camera_id);
            eth3DImage->framePath = frame_path;


            // get frame name
            std::vector<std::string> result;
            parseString("/", frame_path, result);
            eth3DImage->frameName = result.at(1);

            //add newly created frame struct to the vector
            _eth3DFrames.push_back(eth3DImage);

        }

        file.close();
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW



    // vector of ETH3DFrame* struct
    std::vector<ETH3DFrame *> eth3DFrames;

    // image path list
    std::vector<std::string> imagesPath;

    //camera id list (diff. for every image)
    std::vector<int> imagesCameraIDs;

    //intrinsic matrix of 4 different cameras
    std::map<int, Eigen::Matrix3f> camerasIntrinsics;

    std::map<int, std::vector<float>> points3D;

    // base dir
    std::string baseDir;


};
