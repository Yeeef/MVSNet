#include "ransac.h"
#include "plane_param_estimator.h"

/* Actually, I am given four points and depth value alongside
 * I don't get the exact form of the intermediate file, so I will assume some middle file format
 * or I could just try some fake data.
 * So, currently I am given several points, and I just need to parse this intermediate file
 */

int main() {
	PlaneParamEstimator plane_param_estimator(0.01f);
	Ransac ransac;
	std::vector<double> parameters_1;
	std::vector<int> inliers_1;
	ransac.compute(parameters_1, inliers_1, plane_param_estimator, patch_vertex_pos_vec, 0.5, 0.99, false);
}