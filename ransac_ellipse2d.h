#ifndef RANSAC_ELLIPSE2D_H_
#define RANSAC_ELLIPSE2D_H_

#include "ransac2d.h"

namespace sac
{
	class ransacModelEllipse2D : public RansacModel
	{
	public:

		using RansacModel::input_;
		using RansacModel::indices_;
		using RansacModel::error_sqr_dists_;

		typedef RansacModel::PointCloud PointCloud;

		ransacModelEllipse2D(const PointCloud &cloud, double threshold, int max_iterations)
			:RansacModel(cloud, threshold, max_iterations){ spLAxis = 0; spSAxis = 0; spAngle = 0; spRatio = 0.2; }

		ransacModelEllipse2D(const PointCloud &cloud, const std::vector<int> &indices, double threshold, int max_iterations)
			:RansacModel(cloud, indices, threshold, max_iterations){ spLAxis = 0; spSAxis = 0; spAngle = 0; spRatio = 0.2; }

		ransacModelEllipse2D(){};
		~ransacModelEllipse2D(){};

		modelType getModelType()const{ return MODEL_ELLIPSE2D; }

		bool isGoodSample(const std::vector<int> &samples) const;

		bool computeModelCoefficients(const std::vector<int> &samples, ModelCoefficient &model_coefficient);

		int countWithinDistance(const ModelCoefficient model_coefficients, const double threshold);

		void selectWithinDistance(const ModelCoefficient model_coefficients, const double threshold, std::vector<int> &inliers);

		/**
		* \Brief: set the long/short axis and different ratio between the calculated axis
		* \Details:
		*/
		inline void setSpecficAxisLength(const double lAxis, const double sAxis, const double ratioThld)
		{
			spLAxis = lAxis;
			spSAxis = sAxis;
			spRatio = ratioThld;
			return;
		}

		/**
		* \Brief: set specfic angle of the target ellipse
		* \Details:
		*/
		inline void setSpecficAngle(const double angle)
		{
			spAngle = angle;
			return;
		}

		/**
		* \Brief: calculate model of 2D lines with 2 points
		* \Details: 2 points represent a 2D lines consist of 4 parameters( first 2 is one point and last 2 is another)
		*/
		bool computeModel();

	private:
		double spLAxis;
		double spSAxis;
		double spAngle;
		double spRatio;

	};
}



#endif
