#pragma once

#include <opencv2/opencv.hpp>
#include <vector_types.h>

class PlaneParamEstimator {
public:
	PlaneParamEstimator(double delta);

	bool estimate(std::vector<float3 *> &data,
								std::vector<double> &parameters);

	bool leastSquaresEstimate(std::vector<float3 *> &data,
														std::vector<double> &parameters);

	bool agree(std::vector<double> &parameters, float3 &data);

	int numForEstimate() { return 3; }

	double m_deltaSquared;
};

