#ifndef RANSAC_ELLIPSE2D_IMPL_H_
#define RANSAC_ELLIPSE2D_IMPL_H_

#include "ransac_ellipse2D.h"
#include<cv.h>
#include<highgui.h>

namespace sac
{
	bool ransacModelEllipse2D::isGoodSample(const std::vector<int> &samples) const
	{
		if (samples.size() != 5)
			return false;

		double aa[30] = { 0 };
		for (size_t i = 0; i < 5; i++)
		{
			double x = input_[samples[i]].x;
			double y = input_[samples[i]].y;
			aa[6 * i] = x*x;
			aa[6 * i + 1] = x*y;
			aa[6 * i + 2] = y*y;
			aa[6 * i + 3] = x;
			aa[6 * i + 4] = y;
			aa[6 * i + 5] = 1;
		}

		CvMat A = cvMat(5, 6, CV_64F, aa);
		CvMat* D = cvCreateMat(5, 6, CV_64F);
		CvMat* U = cvCreateMat(5, 5, CV_64F);
		CvMat* V = cvCreateMat(6, 6, CV_64F);
		cvSVD(&A, D, U, V, CV_SVD_U_T);

		double V05 = cvmGet(V, 0, 5);
		if (abs(V05) < 0.000001)
			return false;

		double epA = 1;
		double epB = cvmGet(V, 1, 5) / V05;
		double epC = cvmGet(V, 2, 5) / V05;
		double epD = cvmGet(V, 3, 5) / V05;
		double epE = cvmGet(V, 4, 5) / V05;
		double epF = cvmGet(V, 5, 5) / V05;

		cvReleaseMat(&D);
		cvReleaseMat(&U);
		cvReleaseMat(&V);

		//ax^2 + bxy + cy^2 + dx + ey + f = 0;
		//     | 2a  b |                  |2a  b   d|
		//  if | b  2c |  >0  &&  (2a+2c) |b   2c  f|  < 0
		//   	                          |d   f  2g|
		// a,b,c,d,e,f is a ellipse
		double thres1 = 4 * epA * epC - epB * epB;
		double ellParam[9] = { 2 * epA, epB, epD, epB, 2 * epC, epE, epD, epE, 2 * epF };
		CvMat ellParamMat = cvMat(3, 3, CV_64F, ellParam);		
		double thres2 = cvDet(&ellParamMat)*(epA + epC);

		if (thres1 <= 0 || thres2 >= 0)
			return false;

		return true;
	}

	bool ransacModelEllipse2D::computeModelCoefficients(const std::vector<int> &samples, ModelCoefficient &model_coefficient)
	{
		if (samples.size() != 5)
			return false;

		//https://www.zhihu.com/question/40362085
		//http://yester-place.blogspot.jp/2008/08/opencv-cvsvd2.html
		//http://www.mathchina.net/dvbbs/dv_rss.asp?s=xhtml&boardid=3&id=408&page=118
		//http://m.blog.csdn.net/ningyaliuhebei/article/details/46327681
		//http://blog.csdn.net/ningyaliuhebei/article/details/46327681
		double aa[30] = { 0 };
		for (size_t i = 0; i < 5; i++)
		{
			double x = input_[samples[i]].x;
			double y = input_[samples[i]].y;
			aa[6 * i] = x*x;
			aa[6 * i + 1] = x*y;
			aa[6 * i + 2] = y*y;
			aa[6 * i + 3] = x;
			aa[6 * i + 4] = y;
			aa[6 * i + 5] = 1;
		}

		CvMat A = cvMat(5, 6, CV_64F, aa);
		CvMat* D = cvCreateMat(5, 6, CV_64F);
		CvMat* U = cvCreateMat(5, 5, CV_64F);
		CvMat* V = cvCreateMat(6, 6, CV_64F);
		cvSVD(&A, D, U, V, CV_SVD_U_T);

		double V05 = cvmGet(V, 0, 5);
		if (abs(V05) < 0.000001)
			return false;

		double epA = 1;
		double epB = cvmGet(V, 1, 5) / V05;
		double epC = cvmGet(V, 2, 5) / V05;
		double epD = cvmGet(V, 3, 5) / V05;
		double epE = cvmGet(V, 4, 5) / V05;
		double epF = cvmGet(V, 5, 5) / V05;

		cvReleaseMat(&D);
		cvReleaseMat(&U);
		cvReleaseMat(&V);

		//ax^2 + bxy + cy^2 + dx + ey + f = 0;
		//     | 2a  b |                  |2a  b   d|
		//  if | b  2c |  >0  &&  (2a+2c) |b   2c  f|  < 0
		//   	                          |d   f  2g|
		// a,b,c,d,e,f is a ellipse
		double thres1 = 4 * epA * epC - epB * epB;
		double ellParam[9] = { 2 * epA, epB, epD, epB, 2 * epC, epE, epD, epE, 2 * epF };
		CvMat ellParamMat = cvMat(3, 3, CV_64F, ellParam);		
		double thres2 = cvDet(&ellParamMat)*(epA + epC);

		if (thres1 <= 0 || thres2 >= 0)
			return false;

		double epX = (epB*epE - 2 * epC*epD) / (4 * epA*epC - epB*epB);
		double epY = (epB*epD - 2 * epA*epE) / (4 * epA*epC - epB*epB);

		double epAngle = 0;
		if (abs(epB) <= 0.0001&&epA < epC)
			epAngle = 0;
		else if (abs(epB) <= 0.0001&&epA > epC)
			epAngle = 90;
		else if (epA < epC)
			epAngle = 0.5*atan(epB / (epA - epC)) * 180 / 3.1415926;
		else epAngle = 90 + 0.5*atan(epB / (epA - epC)) * 180 / 3.1415926;

		double epTemp1 = epA*epX*epX + epC*epY*epY + epB*epX*epY - epF;
		double epTemp2 = epA + epC;
		double epTemp3 = sqrt((epA - epC)*(epA - epC) + epB*epB);
		double epSAxis = sqrt(2 * epTemp1 / (epTemp2 + epTemp3));
		double epLAxis = sqrt(2 * epTemp1 / (epTemp2 - epTemp3));

		if (spLAxis > 0.001&&abs(epLAxis - spLAxis) / spLAxis > spRatio)
			return false;

		if (spSAxis > 0.001&&abs(epSAxis - spSAxis) / spSAxis > spRatio)
			return false;

		if (spAngle > 0.001&&abs(epAngle - spAngle) / spAngle > spRatio)
			return false;

		model_coefficient.modelParam[0] = epX;
		model_coefficient.modelParam[1] = epY;
		model_coefficient.modelParam[2] = epLAxis;
		model_coefficient.modelParam[3] = epSAxis;
		model_coefficient.modelParam[4] = epAngle;

		return true;
	}

	int ransacModelEllipse2D::countWithinDistance(const ModelCoefficient model_coefficients, const double threshold)
	{
		double cx = model_coefficients.modelParam[0];
		double cy = model_coefficients.modelParam[1];
		double lA = model_coefficients.modelParam[2];
		double sA = model_coefficients.modelParam[3];
		double angle = model_coefficients.modelParam[4];

		double cA = sqrt(lA*lA - sA*sA);
		double f1x = cx - cA*cos(angle*3.141592653 / 180);
		double f1y = cy - cA*sin(angle*3.141592653 / 180);
		double f2x = cx + cA*cos(angle*3.141592653 / 180);
		double f2y = cy + cA*sin(angle*3.141592653 / 180);

		int count(0);
		Point2D cP(cx, cy), cf1(f1x, f1y), cf2(f2x, f2y);
		for (size_t i = 0; i < indices_.size(); i++)
		{
			Point2D iP = input_[indices_[i]];
			double fd1 = cf1.calDistance(iP);
			double fd2 = cf2.calDistance(iP);

			if (abs(fd1 + fd2 - 2 * lA) < threshold)
				count++;
		}
		return count;
	}

	void ransacModelEllipse2D::selectWithinDistance(const ModelCoefficient model_coefficients, const double threshold, std::vector<int> &inliers)
	{
		double cx = model_coefficients.modelParam[0];
		double cy = model_coefficients.modelParam[1];
		double lA = model_coefficients.modelParam[2];
		double sA = model_coefficients.modelParam[3];
		double angle = model_coefficients.modelParam[4];

		double cA = sqrt(lA*lA - sA*sA);

		double f1x = cx - cA*cos(angle*3.141592653 / 180);
		double f1y = cy - cA*sin(angle*3.141592653 / 180);
		double f2x = cx + cA*cos(angle*3.141592653 / 180);
		double f2y = cy + cA*sin(angle*3.141592653 / 180);


		inliers.resize(indices_.size());
		error_sqr_dists_.resize(indices_.size());
		int count(0);
		Point2D cP(cx, cy), cf1(f1x, f1y), cf2(f2x, f2y);
		for (size_t i = 0; i < indices_.size(); i++)
		{
			Point2D iP = input_[indices_[i]];

			double fd1 = cf1.calDistance(iP);
			double fd2 = cf2.calDistance(iP);

			if (abs(fd1 + fd2 - 2 * lA) < threshold)
			{
				inliers[count] = indices_[i];
				error_sqr_dists_[count] = abs(fd1 + fd2 - 2 * lA);
				count++;
			}
		}
		inliers.resize(count);
		inliers.resize(count);
	}

	bool ransacModelEllipse2D::computeModel()
	{
		//warn and exit if no threshold was set	
		assert(threshold_ != std::numeric_limits<double>::max());

		iterations_ = 0;
		int n_best_inliers_count = -INT_MAX;
		double log_probability = log(1.0 - probability_);
		double one_over_indices = 1 / static_cast<double>(getIndices().size());

		int n_inliers_count(0);
		int skipped_count = 0;
		const int max_skip = max_iterations_ * 10;

		ModelCoefficient model_coeff;
		std::vector<int> selection;

		while (iterations_ < max_iterations_ && skipped_count < max_skip)
		{
			getSamples(iterations_, selection);
			assert(selection.size() != 0);

			if (!computeModelCoefficients(selection, model_coeff))
			{
				++skipped_count;
				++iterations_;
				continue;
			}

			n_inliers_count = countWithinDistance(model_coeff, threshold_);
			if (n_inliers_count > n_best_inliers_count)
			{
				n_best_inliers_count = n_inliers_count;
				model_ = selection;
				model_coefficients_ = model_coeff;

				//compute the k parameter
				//TODO

			}

			iterations_++;
			if (iterations_ > max_iterations_)
				break;
		}

		if (model_.size() == 0)
		{
			inliers_.clear();
			return false;
		}

		selectWithinDistance(model_coefficients_, threshold_, inliers_);
		return true;
	}
	
	double matDet(int n, double* mat)
	{
		double* mPtr = new double[n*n];
		memcpy(mPtr, mat, n*n*sizeof(double));

		int ii, jj, k, u;
		int iter = 0;
		double det1 = 1, yin;

		for (ii = 0; ii < n; ii++)
		{
			if (mat[ii*n + ii] == 0)
			for (jj = ii; jj < n; jj++)
			{
				if (mat[jj*n + ii] != 0)
				{
					double* aa = mat + ii*n;
					double* bb = mat + jj*n;
					double temp1;
					for (int i = 0; i < n; i++)
					{
						temp1 = aa[i];
						aa[i] = bb[i];
						bb[i] = temp1;
					}

					iter++;
				}
			}

			for (k = ii + 1; k < n; k++)
			{
				yin = -1 * mat[k*n + ii] / mat[ii*n + ii];

				for (u = 0; u < n; u++)
				{
					mat[k*n + u] = mat[k*n + u] + mat[ii*n + u] * yin;
				}
			}
		}
		for (ii = 0; ii < n; ii++)
			det1 = det1 * mat[ii*n + ii];

		if (iter % 2 == 1)
			det1 = -det1;

		memcpy(mat, mPtr, n*n*sizeof(double));
		free(mPtr);

		return (det1);
	}

}

#endif
