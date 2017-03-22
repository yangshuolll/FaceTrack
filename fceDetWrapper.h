#pragma once
#include "LandmarkCoreIncludes.h"
typedef struct _stDetectResult
{
	int numMarks;
	float landMarks_2D[200]; //2D image points
	float landMarks_3D[300]; //3D image points	
	float headPose[6];
}stDetectResult;
class openFaceDetector
{
public:
	openFaceDetector();
	~openFaceDetector();
	LandmarkDetector::CLNF *mpCLNFModel;
	LandmarkDetector::FaceModelParameters *mpdet_parameters;

	bool initFaceDetector(int processWidth,int processHeight);

	void detectFace(unsigned char*pImg, int width, int height, stDetectResult &result);

	void beautifyImage(cv::Mat &img);

	void packageResult(stDetectResult &result);
	void drawResult(cv::Mat& captured_image);
	void releaseDetector();

	void extractContourImage(cv::Mat img, std::vector<cv::Point> &srcContours, cv::Mat &png);
	void extractFacePng(cv::Mat img,  float *pLandmarks_2d, cv::Mat &png);

	int mProcessWidth;
	int mProcessHeight;

	//initial camera intrinsic parameters
	float mfx;
	float mfy;
	int mcx;
	int mcy;

	float msx;
	float msy;

	int mIsInited;
};

