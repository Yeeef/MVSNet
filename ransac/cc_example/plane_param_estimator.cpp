#include "plane_param_estimator.h"

#include <Eigen/Eigen>

#include "math_utils.h"

PlaneParamEstimator::PlaneParamEstimator(double delta) : m_deltaSquared(delta * delta) {}
/*****************************************************************************/
/*
* Compute the line parameters  [n_x,n_y,a_x,a_y]
* ͨ�������������ȷ������ֱ�ߣ����÷��������ķ�ʽ����ʾ���Լ���ƽ�л�ֱ�����
* ����n_x,n_yΪ��һ������ԭ�㹹�ɵķ���������a_x,a_yΪֱ��������һ��
*/
bool PlaneParamEstimator::estimate(std::vector<float3 *> &data,
																	 std::vector<double> &parameters)
{


	float3 normal = normalize(cross(*data[1] - *data[0], *data[2] - *data[0]));
	float3 mean_pos = (*data[0] + *data[1] + *data[2]) / 3.0f;
	parameters.clear();
	parameters.emplace_back(normal.x);
	parameters.emplace_back(normal.y);
	parameters.emplace_back(normal.z);
	parameters.emplace_back(mean_pos.x);
	parameters.emplace_back(mean_pos.y);
	parameters.emplace_back(mean_pos.z);

	return true;
}
/*****************************************************************************/
/*
* Compute the line parameters  [n_x,n_y,a_x,a_y]
* ʹ����С���˷��������������ϳ�ȷ��ֱ��ģ�͵��������
*/
bool PlaneParamEstimator::leastSquaresEstimate(std::vector<float3 *> &data,
											   std::vector<double> &parameters)
{
	// TODO: float3 means a 3-element tuple?
	float3 mean_pos = make_float3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < data.size(); i++) {
		mean_pos += (*data[i]);
	}
	mean_pos /= data.size();

	Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
	for (int i = 0; i < data.size(); i++) {
		float3 diff = (*data[i] - mean_pos);
		covariance += Eigen::Vector3f(diff.x, diff.y, diff.z) * Eigen::Vector3f(diff.x, diff.y, diff.z).transpose();
	}
	Eigen::EigenSolver<Eigen::Matrix3f> es(covariance);
	Eigen::Matrix3f V = es.pseudoEigenvectors();
	Eigen::Matrix3f D = es.pseudoEigenvalueMatrix();

	float3 normal;
	if (D(2, 2) < D(0, 0) && D(2, 2) < D(1, 1))
	{
		normal.x = V(0, 2);
		normal.y = V(1, 2);
		normal.z = V(2, 2);
	}
	if (D(1, 1) < D(0, 0) && D(1, 1) < D(2, 2))
	{
		normal.x = V(0, 1);
		normal.y = V(1, 1);
		normal.z = V(2, 1);
	}
	if (D(0, 0) < D(1, 1) && D(0, 0) < D(2, 2))
	{
		normal.x = V(0, 0);
		normal.y = V(1, 0);
		normal.z = V(2, 0);
	}
	normal = normalize(normal);

	parameters.clear();
	parameters.emplace_back(normal.x);
	parameters.emplace_back(normal.y);
	parameters.emplace_back(normal.z);
	parameters.emplace_back(mean_pos.x);
	parameters.emplace_back(mean_pos.y);
	parameters.emplace_back(mean_pos.z);

	return true;
}

/*****************************************************************************/
/*
* Given the line parameters  [n_x,n_y,a_x,a_y] check if
* [n_x, n_y] dot [data.x-a_x, data.y-a_y] < m_delta
* ͨ������֪���ߵĵ�˽����ȷ�����������ֱ֪�ߵ�ƥ��̶ȣ����ԽС��Խ���ϣ�Ϊ
* �����������ֱ����
*/

bool PlaneParamEstimator::agree(std::vector<double> &parameters, float3 &data)
{
	float d = -(parameters[0] * parameters[3] + parameters[1] * parameters[4] + parameters[2] * parameters[5]);

	//float3 diff = make_float3(parameters[3] - data.x, parameters[4] - data.y, parameters[5] - data.z);
	//double dist2 = dot(diff, diff);

	double signedDistance = parameters[0] * data.x + parameters[1] * data.y + parameters[2] * data.z + d;
	//return ((signedDistance*signedDistance) < m_deltaSquared && (dist2 < m_deltaSquared * 10));
	return ((signedDistance*signedDistance) < m_deltaSquared);
}
