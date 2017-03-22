#include "faceDetDllWrapper.h"

#include "fceDetWrapper.h"
#include <process.h>
#include <opencv2\opencv.hpp>
#include <Windows.h>
#include <time.h>
openFaceDetector gFaceDetector;
int gWidth, gHeight,gFaceImgWidth,gFaceImgHeight;
static void saveFaceImgToFile(cv::Mat facePng, char *pImgFile);
void  initThread(void *para)
{
	gFaceDetector.initFaceDetector(gWidth, gHeight);	
	CreateDirectory("face_images", NULL);
}
int initFaceDetector(int processWidth, int processHeight, int faceImgWidth, int faceImgHeight)
{
	if (processWidth == gWidth && processHeight == gHeight && 
		faceImgWidth== gFaceImgWidth && faceImgHeight == gFaceImgHeight &&gFaceDetector.mIsInited)
		return 1;

	gWidth = processWidth;
	gHeight = processHeight;
	gFaceImgWidth = faceImgWidth;
	gFaceImgHeight = faceImgHeight;
	_beginthread(initThread, 0, NULL);

	return 1;
}


void detectFace(unsigned char*pImg, int width, int height, stFaceDetectResult *result)
{
	memset(result, 0, sizeof(result));
	if (gFaceDetector.mIsInited)
	{
		stDetectResult detResult;
		gFaceDetector.detectFace(pImg, width, height, detResult);
		memcpy(result, &detResult, sizeof(detResult));
		//result.numMarks = detResult.numMarks;
	}
}
//void detectFaceGray(unsigned char*pImg, int width, int height, int &numMarks, float *p2D, float *p3D, float *pPose)
void detectFaceGray(unsigned char*pImg, int width, int height, int &numMarks, float p2D[], float p3D[], float pPose[])
{
	numMarks = 0;
	/*memset(p2D, 0, 200 * sizeof(float));
	memset(p3D, 0, 300 * sizeof(float));
	memset(pPose, 0, 6 * sizeof(float));*/
	if (gFaceDetector.mIsInited)
	{
		stDetectResult detResult;
		gFaceDetector.detectFace(pImg, width, height, detResult);
		numMarks = detResult.numMarks;
		for (int k = 0; k < numMarks*2; k++)
		{
			p2D[k] = detResult.landMarks_2D[k];
		}
		for (int k = 0; k < numMarks * 3; k++)
		{
			p3D[k] = detResult.landMarks_3D[k];
		}
		for (int k = 0; k < 6; k++)
		{
			pPose[k] = detResult.headPose[k];
		}
		/*memcpy(p2D, detResult.landMarks_2D, sizeof(detResult.landMarks_2D));
		memcpy(p3D, detResult.landMarks_3D, sizeof(detResult.landMarks_3D));
		memcpy(pPose, detResult.headPose, sizeof(detResult.headPose));*/
	}
	//pPose[0] = 100;
}



void detectFaceRGBA(Color_32 *pImg, int width, int height, int &numMarks,
					float p2D[], float p3D[], float pPose[], char *pFaceImgPath,Color_32 *pFaceTexture)
{
	using namespace cv;
	uchar *pImgRaw = (uchar*)pImg;
	Mat img(Size(width, height), CV_8UC4, pImgRaw);
	Mat clrImg;
	cvtColor(img, clrImg, CV_RGBA2BGR);

	flip(clrImg, clrImg, 0);

	Mat gray;
	cvtColor(clrImg, gray, CV_RGB2GRAY);

	detectFaceGray(gray.data, width, height, numMarks, p2D, p3D, pPose);		
	if (numMarks >= 66)
	{
		cv::Mat facePng;
		gFaceDetector.extractFacePng(clrImg, p2D, facePng);
		if (facePng.data == 0)
			return;
		resize(facePng, facePng, cv::Size(gFaceImgWidth, gFaceImgHeight));
		if (pFaceImgPath)
		{
			saveFaceImgToFile(facePng, pFaceImgPath);
		}
		if (pFaceTexture)
		{
			uchar *pDst = (uchar *)pFaceTexture;
			memcpy(pDst, facePng.data, facePng.cols*facePng.rows*facePng.channels());
		}
	}
		

	if (gFaceDetector.mIsInited)
	{
		gFaceDetector.drawResult(clrImg);
	}
		
	imshow("result", clrImg);
	waitKey(1);
}
extern "C" __declspec(dllexport)
void detectFaceRGB(unsigned char *pImg, int width, int height, int &numMarks, float p2D[], float p3D[], float pPose[], char *pFaceImgPath)
{

	using namespace cv;
	uchar *pImgRaw = (uchar*)pImg;
	Mat img(Size(width, height), CV_8UC3, pImgRaw);
	
	Mat gray;
	cvtColor(img, gray, CV_BGRA2GRAY);
	
	detectFaceGray(gray.data, width, height, numMarks, p2D, p3D, pPose);

	if (numMarks >=66)
	{
		cv::Mat facePng;
		gFaceDetector.extractFacePng(img, p2D, facePng);
		if (facePng.data == nullptr)
		{
			printf("empty face\n");
			return;
		}
		resize(facePng, facePng, cv::Size(gFaceImgWidth, gFaceImgHeight));
		if (pFaceImgPath)
		{
			saveFaceImgToFile(facePng, pFaceImgPath);
		}

		//normalize(facePng, facePng, 0, 255, CV_MINMAX);

		imshow("face", facePng);
		
		
	}
	else if (numMarks > 0)
	{
		printf("found %d marks\n", numMarks);
	}
	
	//imshow("beati", beautifyImg);
	/*if (gFaceDetector.mIsInited)	
		gFaceDetector.drawResult(img);
	
		
	imshow("box", img);
	waitKey(1);*/

}

static void saveFaceImgToFile(cv::Mat facePng, char *pImgFile)
{
	time_t t1 = time(0);
	struct tm * now = localtime(&t1);
	now->tm_year += 1900;
	now->tm_mon += 1;

	sprintf_s(pImgFile,128, "face_images/%d-%02d-%02d_%02d%02d%02d.png\0", now->tm_year, now->tm_mon, now->tm_mday,
		now->tm_hour, now->tm_min, now->tm_sec);
	
	if (facePng.size().area() > 100)
	{
		//
		cv::imwrite(pImgFile, facePng);
		//printf("save image:%s\n", pImgFile);
	}
		
	//return png;
}
void uinitFaceDetector()
{
	gFaceDetector.releaseDetector();
}