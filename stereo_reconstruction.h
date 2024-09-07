#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Eigen.h"
#include "FrameDataTypes.h"

/*interface for correspondence search functors
* search() is the search method itself
*/
class CorrSearchMethod {
public:
	virtual cv::Mat search(cv::Mat img1, cv::Mat img2, int dmax, int win_size = 1) = 0;
};

//naive search implements the correspondence search interface
class NaiveSearch : public CorrSearchMethod {
private:
	double SSD(cv::Mat subimg1, cv::Mat subimg2) {
		return cv::norm(subimg1, subimg2, cv::NORM_L2, cv::noArray());
	}

	double SAD(cv::Mat subimg1, cv::Mat subimg2) {
		return cv::norm(subimg1, subimg2, cv::NORM_L1, cv::noArray());
	}

	double NCC(cv::Mat subimg1, cv::Mat subimg2) {
		return subimg1.dot(subimg2)/(cv::norm(subimg1, cv::NORM_L2, cv::noArray()) * cv::norm(subimg2, cv::NORM_L2, cv::noArray()));
	}

public:
	cv::Mat search(cv::Mat img1, cv::Mat img2, int dmax, int win_size) override;
	cv::Mat test(cv::Mat img1, cv::Mat img2, int dmax, int win_size);
};

class PatchMatch : public CorrSearchMethod {
private:
	cv::Mat random_init(int dmax, int height, int width);
	void spatial_propagation(cv::Mat& img, cv::Mat& sur, int x, int y, int win_size);
	void view_propagation(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int x, int y, int win_size);
	void plane_refinement(cv::Mat& img, cv::Mat& sur, int x, int y, int win_size);
	void top_left_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int win_size);
	void bottom_right_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int win_size);
	cv::Mat to_disparity(cv::Mat& sur);
	double m(cv::Mat& main_img, cv::Mat& corr_img, int x, int y, cv::Mat abc, int win_size, int dmax);
public:
	cv::Mat search(cv::Mat img1, cv::Mat img2, int dmax, int win_size) override;
};

class StereoReconstructor {

public:
	//constructors
	StereoReconstructor(cv::Mat img1, cv::Mat img2) : img1(img1), img2(img2) {}
	StereoReconstructor(ETH3DFrame img1, ETH3DFrame img2) : img1(img1.frame), img2(img2.frame) {}
	//function to obtain the disparity map of the stored images; parameter is the correspondence search method
	cv::Mat getDisparity(CorrSearchMethod *c);

private:
	//the images
	cv::Mat img1, img2;
	
};