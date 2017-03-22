#ifndef FILE_FACE_DET_DLL_H
#define FILE_FACE_DET_DLL_H


typedef struct _stFaceDetectResult
{
	int numMarks;
	float landMarks_2D[200]; //2D image points
	float landMarks_3D[300]; //3D image points	
	float headPose[6];
}stFaceDetectResult;

struct Color_32
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

extern "C" __declspec(dllexport)
int initFaceDetector(int processWidth, int processHeight,int faceImgWidth,int faceImgHeight);

extern "C" __declspec(dllexport)
void detectFace(unsigned char*pImg, int width, int height, stFaceDetectResult *result);

extern "C" __declspec(dllexport)
void detectFaceGray(unsigned char*pImg, int width, int height, int &numMarks, float p2D[], float p3D[],float pPose[]);

extern "C" __declspec(dllexport)
void detectFaceRGBA(Color_32 *pImg, int width, int height, int &numMarks, 
					float p2D[], float p3D[], float pPose[],char *pFaceImgPath,Color_32 *pFaceTexture);

extern "C" __declspec(dllexport)
void detectFaceRGB(unsigned char *pImg, int width, int height, int &numMarks, float p2D[], float p3D[], float pPose[],char *pFaceImgPath);


extern "C" __declspec(dllexport)
void uinitFaceDetector();




#endif


