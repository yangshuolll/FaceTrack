#include "fceDetWrapper.h"
#include <string>

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
using namespace std;
using namespace cv;

const int draw_multiplier = 1 << 4;
std::ofstream gLogger;
openFaceDetector::openFaceDetector()
{
	mIsInited = 0;
}


openFaceDetector::~openFaceDetector()
{
}


bool openFaceDetector::initFaceDetector(int processWidth, int processHeight)
{
	printf("start init.............\n");
	//std::ofstream loggxer("dllInit.log");
	gLogger.open("detector.log");
	gLogger << "enter" << endl;
	mpdet_parameters = new LandmarkDetector::FaceModelParameters();
	//loggxer << "parameters" << endl;
	mpCLNFModel = new LandmarkDetector::CLNF(mpdet_parameters->model_location);
	gLogger << "inited" << endl;
	mProcessWidth = processWidth;
	mProcessHeight = processHeight;
	

	mcx = mProcessWidth / 2;
	mcy = mProcessHeight / 2;

	mfx = 500 * (mProcessWidth / 640.0);
	mfy = 500 * (mProcessHeight / 480.0);

	mfx = (mfx + mfy) / 2.0;
	mfy = mfx;

	mIsInited = 1;
	msx = msy = 1;
	return true;
}

void openFaceDetector::detectFace(unsigned char*pImg, int width, int height, stDetectResult &result)
{

	gLogger << "start to detect face" << endl;
	memset(&result, 0, sizeof(result));

	cv::Mat srcImg(Size(width, height), CV_8UC1, pImg);	
	cv::Mat grayImg;

	msx = width / (float)(mProcessWidth);
	msy = height/ (float)(mProcessHeight);

	resize(srcImg, grayImg, Size(mProcessWidth, mProcessHeight));

	LandmarkDetector::CLNF &clnf_model = *mpCLNFModel;
	LandmarkDetector::FaceModelParameters &det_parameters = *mpdet_parameters;
	bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayImg, Mat(), clnf_model, det_parameters);
	gLogger << "detect face done" << endl;
	if (!detection_success)
	{
		//printf("detect fail\n");
	}

	packageResult(result);

}

//package the landmarks to the orginal image size, not to the processed image size
void openFaceDetector::packageResult(stDetectResult &result)
{
	cv::Mat_<double> landmarks_2D = mpCLNFModel->detected_landmarks;
	landmarks_2D = landmarks_2D.reshape(1, 2).t();	

	result.numMarks = landmarks_2D.rows;

	cv::Mat_<double> landmarks_3D;
	mpCLNFModel->pdm.CalcShape3D(landmarks_3D, mpCLNFModel->params_local);
	landmarks_3D = landmarks_3D.reshape(1, 3).t();

	LandmarkDetector::CLNF &clnf_model = *mpCLNFModel;
	cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, mfx, mfy, mcx, mcy);

	int numPoints = landmarks_2D.rows;
	// Drawing feature points
	for (int i = 0; i < numPoints; ++i)
	{
		double *pt = landmarks_2D.ptr<double>(i);
		//float ptx = landmarks_2D.ptr<double>(i)[0];// *(double)draw_multiplier;
		//float pty = landmarks_2D.ptr<double>(i)[1];// *(double)draw_multiplier;
		result.landMarks_2D[2 * i] = pt[0]*msx;
		result.landMarks_2D[2 * i + 1] = pt[1]*msy;

		double *pXYZ = landmarks_3D.ptr<double>(i);		
		result.landMarks_3D[3 * i] = pXYZ[0];
		result.landMarks_3D[3 * i + 1] = pXYZ[1];
		result.landMarks_3D[3 * i + 2] = pXYZ[2];
	}
	double *pPose = &(pose_estimate_to_draw[0]);
	for (int k = 0; k < 6; k++)
	{
		result.headPose[k] = pPose[k];
	}
}
// Visualising the results
void  openFaceDetector::drawResult(cv::Mat& captured_image)
{
	const LandmarkDetector::CLNF& face_model = *mpCLNFModel;
	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;
	/*float fx = gFaceDetector.mfx;
	float fy = gFaceDetector.mfy;
	float cx = gFaceDetector.mcx;
	float cy = gFaceDetector.mcy;*/

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, mfx, mfy, mcx, mcy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, mfx, mfy, mcx, mcy);
	}

}

void openFaceDetector::extractContourImage(cv::Mat img, std::vector<cv::Point> &srcContours, cv::Mat &png)
{
	//get bounding box
	int xMin = 1000, yMin = 1000, xMax = 0, yMax = 0;
	for (int i = 0; i < srcContours.size(); i++)
	{
		int x = srcContours[i].x;
		int y = srcContours[i].y;
		if (x > xMax)
			xMax = x;
		if (x<xMin)
			xMin = x;
		if (y > yMax)
			yMax = y;
		if (y<yMin)
			yMin = y;

	}
	if (xMin < 0 || yMin < 0 || xMax>=img.cols || yMax>=img.rows)
		return;

	cv::Rect rct(xMin,yMin,xMax-xMin+1,yMax-yMin+1);
	if (rct.area() == 0 || rct.area() > img.size().area()||rct.x+rct.width>=img.cols || rct.y+rct.height>=img.rows)
		return;

	Mat subImg = img(rct).clone();
	//beautifyImage(subImg);
	
	std::vector<cv::Point> subContours;
	for (int i = 0; i < srcContours.size(); i++)
	{
		cv::Point pt = srcContours[i];
		pt.x -= rct.x;
		pt.y -= rct.y;
		subContours.push_back(pt);
	}
	Mat mask = Mat::zeros(Size(rct.width, rct.height), CV_8UC1);
	std::vector<std::vector<cv::Point>> subCCs;
	subCCs.push_back(subContours);
	drawContours(mask, subCCs, 0, Scalar(25, 255, 255), -1);

	png = Mat::zeros(subImg.size(), CV_8UC4);
	for (int r = 0; r < subImg.rows; r++)
	{
		uchar *ptrImgRow = subImg.ptr<uchar>(r);
		uchar *ptrAlphaRow = mask.ptr<uchar>(r);
		uchar *ptrPngRow = png.ptr<uchar>(r);
		for (int c = 0; c < subImg.cols; c++)
		{
			uchar flag = ptrAlphaRow[c];
			if (flag == 0)
				continue;
			ptrPngRow[4 * c + 0] = ptrImgRow[3 * c + 0];
			ptrPngRow[4 * c + 1] = ptrImgRow[3 * c + 1];
			ptrPngRow[4 * c + 2] = ptrImgRow[3 * c + 2];
			ptrPngRow[4 * c + 3] = 255;
		}
	}
	normalize(png, png, 0, 255, CV_MINMAX);
}
void openFaceDetector::extractFacePng(cv::Mat img, float *pLandmarks_2d, cv::Mat &png)
{
	gLogger << "start to extract face png ....." << endl;
	/*float processW = processSize.width;
	float processH = processSize.height;
	float sx = img.cols / processW;
	float sy = img.rows / processH;*/

	float sx = 1, sy = 1;
	vector<Point> facePts;
	for (int i = 0; i < 17; i++)
	{
		Point pt;
		pt.x = pLandmarks_2d[2 * i] * sx;
		pt.y = pLandmarks_2d[2 * i + 1] * sy;
		facePts.push_back(pt);
	}

	//offset eyebrow points
	Point2f noseVf;
	noseVf.x = (pLandmarks_2d[27 * 2] - pLandmarks_2d[30 * 2])*sx;
	noseVf.y = (pLandmarks_2d[27 * 2 + 1] - pLandmarks_2d[30 * 2 + 1])*sy;

	float offsetScale = 0.5;
	Point2f offsetV = noseVf*offsetScale;

	//right eyebrow landmarks
	for (int i = 26; i >= 24; i--)
	{
		Point pt;
		pt.x = pLandmarks_2d[2 * i] * sx + noseVf.x;
		pt.y = pLandmarks_2d[2 * i + 1] * sy + noseVf.y;
		facePts.push_back(pt);
	}
	for (int i = 19; i >= 17; i--)
	{
		Point pt;
		pt.x = pLandmarks_2d[2 * i] * sx + noseVf.x;
		pt.y = pLandmarks_2d[2 * i + 1] * sy + noseVf.y;
		facePts.push_back(pt);
	}
	//Mat png;
	extractContourImage(img, facePts, png);
	
	gLogger << "extract face png done......" << endl;
}

void openFaceDetector::beautifyImage(cv::Mat &img)
{
	return;
	Mat biFilMat,guassMat;
	int val1 = 5, val2 = 1;
	int dx = val1 * 5;
	float fc = val1*12.5;
	bilateralFilter(img, biFilMat, dx, fc, fc);
	Mat highPass = biFilMat - img + 180;

	GaussianBlur(highPass, guassMat, Size(3, 3), 0,0);

	
	Mat result = img + guassMat -125;

	//normalize(result, result, 0, 255, NORM_MINMAX);
	result.copyTo(img);
	//imshow("beautfiy", result);

}
void openFaceDetector::releaseDetector()
{
	delete mpCLNFModel;
	delete mpdet_parameters;
	gLogger.close();
	mIsInited = 0;
}