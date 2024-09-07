#include "stereo_reconstruction.h"
#include <limits>
#include <iostream>

double clr_thresh{10.0};
double grd_thresh{2.0};
double Gamma{10.0};
double alpha{0.9};

cv::Mat StereoReconstructor::getDisparity(CorrSearchMethod* c) {
	return (c->search(this->img1, this->img2, 100, 21));
}

cv::Mat NaiveSearch::search(cv::Mat img1, cv::Mat img2, int dmax, int w_size) {
	//setting internal variables
	int imageHeight = img1.rows; //4135
	int imageWidth = img1.cols; //6205
	//return variable
	cv::Mat disparity = cv::Mat(imageHeight, imageWidth, CV_64F);
	auto image_rect = cv::Rect({}, img1.size());
	for (int i{ 0 }; i < imageWidth * imageHeight; ++i) {
		int row{ i / imageWidth };
		int col{ i % imageWidth };
		double min{ 100000.0 };
		int min_pos{ col };
		auto roi1 = cv::Rect(col - w_size/2, row - w_size/2, w_size, w_size);
		auto intersection1 = image_rect & roi1; //target range for image 1
		
		for (int j{std::max(col - dmax, 0)}; j <= col; ++j) {
			double c{ 0.0 };
			auto roi2 = cv::Rect(j - w_size/2, row - w_size/2, w_size, w_size);
			auto intersection2 = image_rect & roi2;
			auto temp = intersection2 - roi2.tl() + roi1.tl();
			intersection1 = temp & intersection1;
			intersection2 = intersection1 - roi1.tl() + roi2.tl();
			cv::Mat tar_patch1 = img1(intersection1);
			cv::Mat tar_patch2 = img2(intersection2);
			c = SSD(tar_patch1, tar_patch2);
			min_pos = (c < min) ? j : min_pos;
			min = (c < min) ? c : min;
		}
		disparity.at<double>(row, col) = col - min_pos;//(min);
		std::cout << "Naive Search: disparity for pixel " << i << " of " << imageHeight * imageWidth << " is " << (col - min_pos) << std::endl;
	}
	return disparity;
}

cv::Mat PatchMatch::random_init(int dmax, int height, int width) {
	std::cout << "random initializing" << std::endl;
	cv::Mat surface(height, width, CV_64FC3);
	for(int i{0}; i < height; ++i) {
		for(int j{0}; j < width; ++j) {
			cv::Mat normal = cv::Mat(1, 3, CV_64F);
			cv::Mat d = cv::Mat(1, 1, CV_64F);
			cv::randu(normal, -10, 10);
			cv::randu(d, 0, dmax);
			cv::normalize(normal, normal, 1.0, 0.0, cv::NORM_L2);
			cv::Mat abc = cv::Mat(1, 3, CV_64F);
			abc.at<double>(0) = -normal.at<double>(0)/normal.at<double>(2);
			abc.at<double>(1) = -normal.at<double>(1)/normal.at<double>(2);
			abc.at<double>(2) = (j * normal.at<double>(0) + i * normal.at<double>(1) + d.at<double>(0) * normal.at<double>(2)) / normal.at<double>(2);
			surface.at<cv::Vec3d>(i, j) = abc;
		}
	}
	return surface;
}

void PatchMatch::spatial_propagation(cv::Mat& img, cv::Mat& sur, int x, int y, int win_size) {

}

void PatchMatch::view_propagation(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int x, int y, int win_size) {

}

void PatchMatch::plane_refinement(cv::Mat& img, cv::Mat& sur, int x, int y, int win_size) {

}

double PatchMatch::m(cv::Mat& main_img, cv::Mat& corr_img, int x, int y, cv::Mat abc, int win_size, int dmax) {
	cv::Mat pix_loc = cv::Mat(1, 3, CV_64F);
	pix_loc.at<double>(0) = x;
	pix_loc.at<double>(1) = y;
	pix_loc.at<double>(2) = 1.0;
	int disp1 = abc.dot(pix_loc);
	int disp2{disp1 + 1};
	int corr_x1 = x - disp1;
	int corr_x2 = x - disp2;

	if((disp1 > dmax) || (disp2 > dmax) || (corr_x1 < 0) || (corr_x2 < 0)) {
		return std::numeric_limits<double>::infinity();
	}

	//compute smallest intersection between patches
	auto main_roi = cv::Rect(x - win_size/2, y - win_size/2, win_size, win_size);
	auto corr_roi1 = cv::Rect(corr_x1 - win_size/2, y - win_size/2, win_size, win_size);
	auto corr_roi2 = cv::Rect(corr_x2 - win_size/2, y - win_size/2, win_size, win_size);
	auto image_rect = cv::Rect({}, main_img.size());
	auto intersection_main = image_rect & main_roi;
	auto intersection_corr1 = image_rect & corr_roi1;
	auto intersection_corr2 = image_rect & corr_roi2;
	auto corr1_temp = intersection_corr1 - corr_roi1.tl() + main_roi.tl();
	auto corr2_temp = intersection_corr2 - corr_roi2.tl() + main_roi.tl();
	intersection_main = intersection_main & corr1_temp;
	intersection_main = intersection_main & corr2_temp;
	intersection_corr1 = intersection_main - main_roi.tl() + corr_roi1.tl();
	intersection_corr2 = intersection_main - main_roi.tl() + corr_roi2.tl();

	//compute corresponding patch with linear interpolation
	cv::Mat main_patch = main_img(intersection_main);
	cv::Mat corr1_patch = corr_img(intersection_corr1);
	cv::Mat corr2_patch = corr_img(intersection_corr2);
	
	int ptch_height = main_patch.rows;
	int ptch_width = main_patch.cols;
	cv::Mat corr_patch;
	addWeighted(corr1_patch, 0.5, corr2_patch, 0.5, 0, corr_patch);

	//compute color diff: clr_diff
	cv::Mat abs_diff;
	cv::absdiff(main_patch, corr_patch, abs_diff);
	abs_diff.convertTo(abs_diff, CV_64FC3);
	cv::Mat channels[3];
	cv::split(abs_diff, channels);
	cv::Mat clr_diff = channels[0] + channels[1] + channels[2]; //double
	cv::min(clr_diff, clr_thresh, clr_diff);

	// compute gray-scale gradient: grad_diff
	cv::Mat main_gray;
	cv::Mat corr_gray;
	cv::cvtColor(main_patch, main_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(corr_patch, corr_gray, cv::COLOR_BGR2GRAY);
	cv::Mat main_grad_x, main_grad_y, corr_grad_x, corr_grad_y;
    cv::Mat main_abs_grad_x, main_abs_grad_y, corr_abs_grad_x, corr_abs_grad_y;
	cv::Mat grad_diff;
    cv::Sobel(main_gray, main_grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(main_gray, main_grad_y, CV_16S, 0, 1, 3);
	cv::Sobel(corr_gray, corr_grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(corr_gray, corr_grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(main_grad_x, main_abs_grad_x);
    convertScaleAbs(main_grad_y, main_abs_grad_y);
	convertScaleAbs(corr_grad_x, corr_abs_grad_x);
    convertScaleAbs(corr_grad_y, corr_abs_grad_y);
	addWeighted(main_abs_grad_x, 0.5, main_abs_grad_y, 0.5, 0, main_abs_grad_x);
	addWeighted(corr_abs_grad_x, 0.5, corr_abs_grad_y, 0.5, 0, corr_abs_grad_x);
	cv::absdiff(main_abs_grad_x, corr_abs_grad_x, grad_diff);
	cv::min(grad_diff, grd_thresh, grad_diff);
	grad_diff.convertTo(grad_diff, CV_64FC1);

	//compute the cost
	double m{0};
	cv::Mat w = cv::Mat(1, 1, CV_64F);
	double r{0};

	for(int i{0}; i < ptch_height; ++i) {
		for(int j{0}; j < ptch_width; ++j) {
			cv::exp(-clr_diff.at<double>(i, j) / Gamma, w);
			r = (1 - alpha) * clr_diff.at<double>(i, j) + alpha * grad_diff.at<double>(i, j);
			m = m + w.at<double>(0) * r;
		}
	}

	return m/(ptch_height * ptch_width);
}

void PatchMatch::top_left_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int win_size) {
	int height = main_img.rows;
	int width = main_img.cols;
	for (int i{ 0 }; i < width * height; ++i) {
		int row{ i / width };
		int col{ i % width };
		spatial_propagation(main_img, main_sur, col, row, win_size);
		view_propagation(main_img, corr_img, main_sur, corr_sur, col, row, win_size);
		plane_refinement(main_img, main_sur, col, row, win_size);
	}
}

void PatchMatch::bottom_right_iter(cv::Mat& main_img, cv::Mat& corr_img, cv::Mat& main_sur, cv::Mat& corr_sur, int win_size) {
	int height = main_img.rows;
	int width = main_img.cols;
	for (int i{ width * height - 1 }; i >= 0; --i) {
		int row{ i / width };
		int col{ i % width };
		spatial_propagation(main_img, main_sur, col, row, win_size);
		view_propagation(main_img, corr_img, main_sur, corr_sur, col, row, win_size);
		plane_refinement(main_img, main_sur, col, row, win_size);
	}
}

cv::Mat PatchMatch::to_disparity(cv::Mat& sur) {
	return sur;
}

// cv::Mat PatchMatch::search_off(cv::Mat img1, cv::Mat img2, int dmax, int win_size) {
// 	int height = img1.rows;
// 	int width = img1.cols;
// 	cv::Mat surface1 = random_init(dmax, height, width);
// 	cv::Mat surface2 = random_init(dmax, height, width);
	
// 	for (int i{ 0 }; i < 3; ++i) {
// 		if (i % 2) {
// 			bottom_right_iter(img1, img2, surface1, surface2, win_size);
// 			bottom_right_iter(img2, img1, surface2, surface1, win_size);
// 		} else {
// 			top_left_iter(img1, img2, surface1, surface2, win_size);
// 			top_left_iter(img2, img1, surface2, surface1, win_size);
// 		}
// 	}

// 	cv::Mat disparity = to_disparity(surface1);

// 	return disparity;
// }

cv::Mat PatchMatch::search(cv::Mat img1, cv::Mat img2, int dmax, int win_size) {
	int height = img1.rows;
	int width = img1.cols;
	cv::Mat surface1 = random_init(dmax, height, width);
	//cv::Mat surface2 = random_init(dmax, height, width);
	cv::Mat cost = cv::Mat(1, width, CV_64F);
	cv::Mat abc = cv::Mat(1, 3, CV_64F);
	cv::Vec3d tmp;

	for(int i{0}; i < width; ++i) {
		tmp = surface1.at<cv::Vec3d>(0, i);
		abc.at<double>(0) = tmp[0];
		abc.at<double>(1) = tmp[1];
		abc.at<double>(2) = tmp[2];
		cost.at<double>(i) = m(img1, img2, i, 0, abc, win_size, dmax);
	}

	cv::Mat disparity = cv::Mat(height, width, CV_64F);
	disparity = 0.0;
	return disparity;
}