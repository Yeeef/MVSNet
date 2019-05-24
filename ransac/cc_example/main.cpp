#include "ransac.h"
#include "plane_param_estimator.h"

int main() {
	PlaneParamEstimator plane_param_estimator(0.01f);
	Ransac ransac;
	std::vector<double> parameters_1;
	std::vector<int> inliers_1;
	ransac.compute(parameters_1, inliers_1, plane_param_estimator, patch_vertex_pos_vec, 0.5, 0.99, false);
}