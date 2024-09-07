#include <vector>
#include "VirtualSensor.h"
#include "Eigen.h"

int imageWidth = 942;
int imageHeight = 489;

double SSD(Vector4f pix1, Vector4f pix2) {
    return (pix1 - pix2).squaredNorm();
}

double SAD(Vector4f pix1, Vector4f pix2) {
    return (pix1 - pix2).lpNorm<1>();
}

double NCC();

std::vector<int> Naive(BYTE* img1, BYTE* img2) {
    std::vector<int> disparity;
    disparity.reserve(imageWidth * imageHeight);
    for(int i{0}; i < imageWidth * imageHeight; ++i) {
        Vector4f pix1;
        int row{i / imageWidth};
        int col{i % imageWidth};
        double min{0.0};
        int min_pos{0};

        pix1 << img1[i * 4], img1[i * 4 + 1], img1[i * 4 + 2], img1[i * 4 + 3];

        for(int j{0}; j < imageWidth; ++j) {
            Vector4f pix2;
            int pos{(row * imageWidth + j) * 4};
            pix2 << img2[pos], img2[pos + 1], img2[pos + 2], img2[pos + 3];
            double d{0.0};

            d = SSD(pix1, pix2);
            min = (d < min) ? d : min;
            min_pos = (d < min) ? pos/4 : min_pos;
        }
        disparity.push_back(i - min_pos);
    }
    return disparity;
}

std::vector<int> NaiveW(BYTE* img1, BYTE* img2, int wsize) {
    std::vector<int> disparity;
    disparity.reserve(imageWidth * imageHeight);
    for(int i{0}; i < imageWidth * imageHeight; ++i) {
        Vector4f pix1;
        int row{i / imageWidth};
        int col{i % imageWidth};
        double min{0.0};
        int min_pos{0};

        pix1 << img1[i * 4], img1[i * 4 + 1], img1[i * 4 + 2], img1[i * 4 + 3];

        for(int j{0}; j < imageWidth; ++j) {
            Vector4f pix2;
            int pos{(row * imageWidth + j) * 4};
            pix2 << img2[pos], img2[pos + 1], img2[pos + 2], img2[pos + 3];
            double d{0.0};

            d = SSD(pix1, pix2);
            min = (d < min) ? d : min;
            min_pos = (d < min) ? pos/4 : min_pos;
        }
        disparity.push_back(i - min_pos);
    }
    return disparity;
}