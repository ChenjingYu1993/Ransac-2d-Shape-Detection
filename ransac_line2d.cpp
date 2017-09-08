#ifndef RANSAC_LINE2D_CPP_
#define RANSAC_LINE2D_CPP_

#include "ransac_line2d.h"

namespace sac
{

	bool ransacModelLine2D::isGoodSample(const std::vector<int> &samples) const
	{
		if (samples.size() != 2)
			return false;

		Point2D p1 = input_[samples[0]];
		Point2D p2 = input_[samples[1]];

		if (p1.calDistance(p2) < 0.0001)
			return false;

		return true;
	}

	bool ransacModelLine2D::computeModelCoefficients(const std::vector<int> &samples, ModelCoefficient &model_coefficient)
	{
		if (samples.size() != 2)
			return false;

		model_coefficient.modelParam[0] = input_[samples[0]].x;
		model_coefficient.modelParam[1] = input_[samples[0]].y;
		model_coefficient.modelParam[2] = input_[samples[1]].x;
		model_coefficient.modelParam[3] = input_[samples[1]].y;

		return true;
	}

	int ransacModelLine2D::countWithinDistance(const ModelCoefficient model_coefficients, const double threshold)
	{
		double mc1x = model_coefficients.modelParam[0];
		double mc1y = model_coefficients.modelParam[1];
		double mc2x = model_coefficients.modelParam[2];
		double mc2y = model_coefficients.modelParam[3];

		int count(0);
		for (size_t i = 0; i < indices_.size(); i++)
		{
			double px = input_[indices_[i]].x;
			double py = input_[indices_[i]].y;

			//calculate the area "s" of point mc1,mc2 and p
			double a, b, c, p, s;
			a = sqrt((mc1x - px)*(mc1x - px) + (mc1y - py)*(mc1y - py));
			b = sqrt((mc2x - px)*(mc2x - px) + (mc2y - py)*(mc2y - py));
			c = sqrt((mc1x - mc2x)*(mc1x - mc2x) + (mc1y - mc2y)*(mc1y - mc2y));
			p = (a + b + c) / 2.0;
			s = sqrt(p*(p - a)*(p - b)*(p - c));

			double pDistance = s / c;
			count = pDistance > threshold ? count : count + 1;
		}

		return count;
	}

	void ransacModelLine2D::selectWithinDistance(const ModelCoefficient model_coefficients, const double threshold, std::vector<int> &inliers)
	{
		int count(0);
		inliers.resize(indices_.size());
		error_sqr_dists_.resize(indices_.size());

		double mc1x = model_coefficients.modelParam[0];
		double mc1y = model_coefficients.modelParam[1];
		double mc2x = model_coefficients.modelParam[2];
		double mc2y = model_coefficients.modelParam[3];

		for (size_t i = 0; i < indices_.size(); i++)
		{
			double px = input_[indices_[i]].x;
			double py = input_[indices_[i]].y;

			//calculate the area "s" of point mc1,mc2 and p
			double a, b, c, p, s;
			a = sqrt((mc1x - px)*(mc1x - px) + (mc1y - py)*(mc1y - py));
			b = sqrt((mc2x - px)*(mc2x - px) + (mc2y - py)*(mc2y - py));
			c = sqrt((mc1x - mc2x)*(mc1x - mc2x) + (mc1y - mc2y)*(mc1y - mc2y));
			p = (a + b + c) / 2.0;
			s = sqrt(p*(p - a)*(p - b)*(p - c));

			double pDistance = s / c;
			if (pDistance < threshold)
			{
				inliers[count] = indices_[i];
				error_sqr_dists_[count] = pDistance;
				count++;
			}
		}

		inliers.resize(count);
		error_sqr_dists_.resize(count);
	}

	bool ransacModelLine2D::computeModel()
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

}
#endif