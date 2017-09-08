#ifndef RANSAC_LINE2D_H_
#define RANSAC_LINE2D_H_

#include "ransac2d.h"

namespace sac
{
	class ransacModelLine2D : public RansacModel
	{
	public:

		using RansacModel::input_;
		using RansacModel::indices_;
		using RansacModel::error_sqr_dists_;

		typedef RansacModel::PointCloud PointCloud;

		ransacModelLine2D(const PointCloud &cloud, double threshold, int max_iterations)
			:RansacModel(cloud, threshold, max_iterations){};

		ransacModelLine2D(const PointCloud &cloud, const std::vector<int> &indices, double threshold, int max_iterations)
			:RansacModel(cloud, indices, threshold, max_iterations){};

		ransacModelLine2D(){};
		~ransacModelLine2D(){};

		modelType getModelType()const{ return MODEL_LINE2D; }

		bool isGoodSample(const std::vector<int> &samples) const;

		bool computeModelCoefficients(const std::vector<int> &samples, ModelCoefficient &model_coefficient);

		int countWithinDistance(const ModelCoefficient model_coefficients, const double threshold);

		void selectWithinDistance(const ModelCoefficient model_coefficients, const double threshold, std::vector<int> &inliers);

		/**
		* \Brief: calculate model of 2D lines with 2 points
		* \Details: 2 points represent a 2D lines consist of 4 parameters( first 2 is one point and last 2 is another)
		*/
		bool computeModel();

	};

}

#endif