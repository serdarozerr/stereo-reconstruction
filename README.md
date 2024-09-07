## Stereo Reconstruction
### Overview
This project aims to construct a dense 3D model from two images of a static scene using stereo reconstruction techniques. The process involves using known intrinsic and extrinsic parameters of the images to rectify them, match common features, and triangulate image points using epipolar geometry. Various algorithms will be tested to find and match features effectively.

### Assumptions
The project is based on the following assumptions:

Multiple images of the same scene, taken from different but known vantage points, are available (extrinsic parameters are known).
The scene is static, meaning none of the observed 3D points moved during the camera motion.
The intrinsic camera (calibration) parameters are known.

### Project Steps
The stereo reconstruction process will follow these steps:

- Rectify Images: Adjust images so that the corresponding points lie on the same horizontal line.
- Find Correspondences: Identify matching points between the two images.
- Compute Disparity Map: Calculate the difference in the position of corresponding points between the two images.
Obtain Depth from Disparity Map: Use the disparity map to compute the depth information.
- Construct 3D Model: Generate a 3D model of the scene using the depth information.
### Requirements
The project uses the following datasets and libraries:

#### Dataset: 
Stereo benchmark dataset provided by ETH3D (https://www.eth3d.net/datasets).
#### Libraries:
- Eigen
- Ceres
- OpenCV
- FreeImage