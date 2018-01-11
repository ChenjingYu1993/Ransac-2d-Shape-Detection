#include<cv.h>
#include<highgui.h>
#include<iostream>
#include<io.h>
#include"ransac_line2d.h"
#include"ransac_circle2d.h"
#include"ransac_ellipse2d.h"

using namespace cv;
using namespace std;

bool testLine2d()
{
	cv::Mat src = cv::imread("linesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	sac::ransacModelLine2D line2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	line2D.setInputCloud(pCloud2D);
	line2D.setDistanceThreshold(5);
	line2D.setMaxIterations(2500);
	line2D.computeModel();
	line2D.getInliers(inliers);
	line2D.getModelCoefficients(parameter);

	Point sp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
	Point ep((int)parameter.modelParam[2], (int)parameter.modelParam[3]);
	int dx = sp.x - ep.x;
	int dy = sp.y - ep.y;
	sp.x = sp.x + 10 * dx;
	sp.y = sp.y + 10 * dy;
	ep.x = ep.x - 10 * dx;
	ep.y = ep.y - 10 * dy;

	line(showMat, sp, ep, Scalar(0, 0, 255), 2, 8);

	cout << "Parameter of 2D line: < " << parameter.modelParam[0] << ", " <<
		parameter.modelParam[1] << " >---< " << parameter.modelParam[2] << ", " <<
		parameter.modelParam[3] << " > " << endl;

	return true;
}

bool testLine2dMulti()
{
	cv::Mat src = cv::imread("linesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	sac::ransacModelLine2D line2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	line2D.setDistanceThreshold(5);
	line2D.setMaxIterations(1000);

	while (pCloud2D.size() > 500)
	{
		line2D.setInputCloud(pCloud2D);
		line2D.computeModel();
		line2D.getModelCoefficients(parameter);

		Point sp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
		Point ep((int)parameter.modelParam[2], (int)parameter.modelParam[3]);
		int dx = sp.x - ep.x;
		int dy = sp.y - ep.y;
		sp.x = sp.x + 10 * dx;
		sp.y = sp.y + 10 * dy;
		ep.x = ep.x - 10 * dx;
		ep.y = ep.y - 10 * dy;

		line(showMat, sp, ep, Scalar(0, 255, 0), 2, 8);

		imshow("multiLines", showMat);
		waitKey(1);

		cout << "Parameter of 2D line: < " << parameter.modelParam[0] << ", " <<
			parameter.modelParam[1] << " >---< " << parameter.modelParam[2] << ", " <<
			parameter.modelParam[3] << " > " << "  k: " << dy*1.0 / dx << endl;

		line2D.getInliers(inliers);
		line2D.removeInliders(pCloud2D, inliers);
	}

	waitKey(-1);

	cout << endl;
	
	return true;
}

bool testCircle2d()
{
	cv::Mat src = cv::imread("circlesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	sac::ransacModelCircle2D circle2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	circle2D.setInputCloud(pCloud2D);
	circle2D.setDistanceThreshold(5);
	circle2D.setMaxIterations(2500);
	circle2D.computeModel();
	circle2D.getInliers(inliers);
	circle2D.getModelCoefficients(parameter);

	Point cp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
	int radius = (int)parameter.modelParam[2];
	circle(showMat, cp, radius, Scalar(0, 0, 255), 2, 8);

	cout << "Parameter of 2D circle: center < " << parameter.modelParam[0] << ", " <<
		parameter.modelParam[1] << " > --- radius: " << parameter.modelParam[2] << endl;

	return true;
}

bool testCircle2dMulti()
{
	cv::Mat src = cv::imread("circlesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	double specRadius = 100;

	sac::ransacModelCircle2D circle2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	circle2D.setDistanceThreshold(5);
	circle2D.setMaxIterations(2500);
	//circle2D.setSpecificRadius(specRadius, 0.2);

	while (pCloud2D.size() > 500)
	{
		circle2D.setInputCloud(pCloud2D);
		circle2D.computeModel();
		circle2D.getInliers(inliers);
		circle2D.getModelCoefficients(parameter);

		if (inliers.size() < specRadius * 2 * CV_PI)
			break;	

		Point cp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
		int radius = (int)parameter.modelParam[2];
		circle(showMat, cp, radius, Scalar(0, 255, 0), 2, 8);

		imshow("circles", showMat);
		waitKey(500);

		cout << "Parameter of 2D line: < " << parameter.modelParam[0] << ", " <<
			parameter.modelParam[1] << " >---" << parameter.modelParam[2] << endl;

		circle2D.removeInliders(pCloud2D, inliers);
	}

	waitKey();

	cout << endl;

	return true;
}

bool testEllipse2dMulti()
{
	cv::Mat src = cv::imread("ellipsesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;
	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	sac::ransacModelEllipse2D ellipse2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	ellipse2D.setDistanceThreshold(5);
	ellipse2D.setMaxIterations(2500);
	ellipse2D.setSpecficAxisLength(120, 80, 0.3);
	//ellipse2D.setSpecficAxisLength(135, 85, 0.5);

	while (pCloud2D.size() > 500)
	{
		cout << pCloud2D.size() << endl;
		ellipse2D.setInputCloud(pCloud2D);
		ellipse2D.computeModel();
		ellipse2D.getInliers(inliers);
		ellipse2D.getModelCoefficients(parameter);

		if (inliers.size() < 500)
			break;

		cv::Point2f ellipseCenter;
		ellipseCenter.x = (float)parameter.modelParam[0];
		ellipseCenter.y = (float)parameter.modelParam[1];
		cv::Size2f ellipseSize;
		ellipseSize.width = (float)parameter.modelParam[2] * 2;
		ellipseSize.height = (float)parameter.modelParam[3] * 2;

		float ellipseAngle = (float)parameter.modelParam[4];
		cout << "Parameters of ellipse2D: < " << parameter.modelParam[0] << ", " <<
			parameter.modelParam[1] << " > --- ";
		cout << "Long/Short Axis: " << parameter.modelParam[2] << "/" << parameter.modelParam[3] << " --- ";
		cout << "Angle: " << parameter.modelParam[4] << endl;
		
		cv::ellipse(showMat, cv::RotatedRect(ellipseCenter, ellipseSize, ellipseAngle), cv::Scalar(0, 255, 0), 2, 8);
		
		imshow("ellipses", showMat);
		waitKey(12);

		ellipse2D.removeInliders(pCloud2D, inliers);
	}

	return true;
}

bool testCircle2dPlus()
{
	cv::Mat src = cv::imread("circlesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	vector<double> multiRaius; multiRaius.push_back(10); multiRaius.push_back(75);

	sac::ransacModelCircle2D circle2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	circle2D.setInputCloud(pCloud2D);
	circle2D.setDistanceThreshold(5);
	circle2D.setMaxIterations(2500);
	circle2D.setMultiRadius(multiRaius, 0.2);
	circle2D.computeModel();
	circle2D.getInliers(inliers);
	circle2D.getModelCoefficients(parameter);

	Point cp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
	int radius = (int)parameter.modelParam[2];
	circle(showMat, cp, radius, Scalar(0, 0, 255), 2, 8);

	cout << "Parameter of 2D circle: center < " << parameter.modelParam[0] << ", " <<
		parameter.modelParam[1] << " > --- radius: " << parameter.modelParam[2] << endl;

	return true;
}

bool testCircle2dMultiPlus()
{
	cv::Mat src = cv::imread("circlesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);

	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	double specRadius = 100;

	vector<double> multiRaius; multiRaius.push_back(100); multiRaius.push_back(75);
	sac::ransacModelCircle2D circle2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	circle2D.setDistanceThreshold(5);
	circle2D.setMaxIterations(2000);
	//circle2D.setSpecificRadius(specRadius, 0.2);
	circle2D.setMultiRadius(multiRaius, 0.2);

	while (pCloud2D.size() > 500)
	{
		circle2D.setInputCloud(pCloud2D);
		circle2D.computeModel();
		circle2D.getInliers(inliers);
		circle2D.getModelCoefficients(parameter);

		if (inliers.size() < parameter.modelParam[2] * 2 * CV_PI)
			break;

		//Point cp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
		//int radius = (int)parameter.modelParam[2];
		//circle(showMat, cp, radius, Scalar(0, 0, 255), 2, 8);

		//imshow("circles", showMat);
		//waitKey(5);

		//cout << "Parameter of 2D line: < " << parameter.modelParam[0] << ", " << parameter.modelParam[1] << " >---" << parameter.modelParam[2] << endl;

		circle2D.removeInliders(pCloud2D, inliers);
	}

	//cout << endl;

	return true;
}

int main()
{
	testEllipse2dMulti();
	
	testCircle2dMulti();
	
	testLine2dMulti();

	return 1;
}
