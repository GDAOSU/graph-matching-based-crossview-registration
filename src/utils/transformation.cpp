/**************************************************************************************/
/*                                                                                    */
/*  Transformation Library                                                            */
/*  https://github.com/keepdash/Transformation                                        */
/*                                                                                    */
/*  Copyright (c) 2017-2017, Wei Ye                                                   */
/*  All rights reserved.                                                              */
/*                                                                                    */
/**************************************************************************************/

#include "Transformation.h"
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;

int RigidTransformation(
	const vector<Eigen::Vector3d> &src_pts,
	const vector<Eigen::Vector3d> &dst_pts,
	const vector<double>& weight,
	Eigen::Matrix4d &tfMat,
	double& err)
{
	Eigen::Matrix3d R;
	Eigen::Vector3d T;
	RigidTransformation(src_pts, dst_pts, weight, R, T, err);

	// Generate 4 by 4 matrix
	tfMat.data()[0] = R.data()[0];
	tfMat.data()[1] = R.data()[1];
	tfMat.data()[2] = R.data()[2];
	tfMat.data()[3] = 0;
	tfMat.data()[4] = R.data()[3];
	tfMat.data()[5] = R.data()[4];
	tfMat.data()[6] = R.data()[5];
	tfMat.data()[7] = 0;
	tfMat.data()[8] = R.data()[6];
	tfMat.data()[9] = R.data()[7];
	tfMat.data()[10] = R.data()[8];
	tfMat.data()[11] = 0;
	tfMat.data()[12] = T.data()[0];
	tfMat.data()[13] = T.data()[1];
	tfMat.data()[14] = T.data()[2];
	tfMat.data()[15] = 1;

	return 1;
}

extern int RigidTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix3d &rotation,
	Eigen::Vector3d &translation,
	double& err)
{
	int expl_size = dst_pts.size();
	if (expl_size < 3 || src_pts.size() != expl_size || weight.size() != expl_size)
		return 0;

	// compute center
	Eigen::Vector3d d_center, s_center;
	s_center.setZero();
	d_center.setZero();
	for (int i = 0; i < expl_size; i++){
		s_center += src_pts[i];
		d_center += dst_pts[i];
	}
	s_center /= expl_size;
	d_center /= expl_size;

	// Generate the covariance matrix
	Eigen::Matrix3d cov_m;
	for (int i = 0; i < 9; i++)
		cov_m.data()[i] = 0.0f;
	for (int i = 0; i < expl_size; i++){
		Eigen::Vector3d ds = src_pts[i] - s_center;
		Eigen::Vector3d dd = dst_pts[i] - d_center;

		cov_m += (ds * dd.transpose() * weight[i]);
	}

	// use SVD to compute rotation and then the translation.
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_m, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	rotation = V*(U.transpose());
	if (rotation.determinant() < 0.0f){
		// reflection in R, negate rightest column (3rd column in 3 by 3 matrix)
		V.data()[6] = -V.data()[6];
		V.data()[7] = -V.data()[7];
		V.data()[8] = -V.data()[8];
		rotation = V*(U.transpose());
	}
	translation = -rotation * s_center + d_center;

	// compute error:
	double sum_err = 0.0f;
	for (int i = 0; i < expl_size; i++){
		Eigen::Vector3d diff = rotation * src_pts[i] + translation - dst_pts[i];
		sum_err += (diff.data()[0] * diff.data()[0] + diff.data()[1] * diff.data()[1] + diff.data()[2] * diff.data()[2]);
	}
	err = sqrt((sum_err / expl_size / 3));

	return 1;
}

int SimilarityTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix4d &tfMat,
	double& err)
{
	Eigen::Matrix3d R;
	Eigen::Vector3d T;
	double S;
	SimilarityTransformation(src_pts, dst_pts, weight, R, T, S, err);

	// Generate 4 by 4 matrix
	tfMat.data()[0] = R.data()[0] * S;
	tfMat.data()[1] = R.data()[1] * S;
	tfMat.data()[2] = R.data()[2] * S;
	tfMat.data()[3] = 0;
	tfMat.data()[4] = R.data()[3] * S;
	tfMat.data()[5] = R.data()[4] * S;
	tfMat.data()[6] = R.data()[5] * S;
	tfMat.data()[7] = 0;
	tfMat.data()[8] = R.data()[6] * S;
	tfMat.data()[9] = R.data()[7] * S;
	tfMat.data()[10] = R.data()[8] * S;
	tfMat.data()[11] = 0;
	tfMat.data()[12] = T.data()[0];
	tfMat.data()[13] = T.data()[1];
	tfMat.data()[14] = T.data()[2];
	tfMat.data()[15] = 1;

	return 1;
}

extern int SimilarityTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix3d &rotation,
	Eigen::Vector3d &translation,
	double& scale,
	double& err)
{
	int expl_size = dst_pts.size();
	if (expl_size < 3 || src_pts.size() != expl_size || weight.size() != expl_size)
		return 0;

	// compute center
	Eigen::Vector3d s_center, d_center;
	s_center.setZero();
	d_center.setZero();
	for (int i = 0; i < expl_size; i++){
		s_center += src_pts[i];
		d_center += dst_pts[i];
	}
	s_center /= expl_size;
	d_center /= expl_size;

	// Generate the covariance matrix
	Eigen::Matrix3d cov_m;
	cov_m.setZero();
	double s_mv = 0.0;
	double d_mv = 0.0;

	for (int i = 0; i < expl_size; i++){
		Eigen::Vector3d ds = src_pts[i] - s_center;
		Eigen::Vector3d dd = dst_pts[i] - d_center;

		Eigen::MatrixXd dsTds = ds.transpose() * ds;
		Eigen::MatrixXd ddTdd = dd.transpose() * dd;

		s_mv += dsTds.data()[0] * weight[i];
		d_mv += ddTdd.data()[0] * weight[i];
		cov_m += (ds * dd.transpose() * weight[i]);
	}

	d_mv /= expl_size;
	s_mv /= expl_size;
	cov_m /= expl_size;

	// use SVD to compute rotation and then the translation.
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_m, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d Z = Eigen::Matrix3d::Identity();
	Eigen::Matrix3d VUt = V*(U.transpose());
	Z.data()[8] = VUt.determinant();

	rotation = V*Z*(U.transpose());

	Eigen::Vector3d S = svd.singularValues();
	scale = (S.data()[0] * Z.data()[0] +
		S.data()[1] * Z.data()[4] +
		S.data()[2] * Z.data()[8]
		) / s_mv;

	translation = -scale * rotation * s_center + d_center;

	// compute error:
	double sum_err = 0.0f;
	for (int i = 0; i < expl_size; i++){
		Eigen::Vector3d diff = scale * rotation * src_pts[i] + translation - dst_pts[i];
		sum_err += sqrt(diff.data()[0] * diff.data()[0] + diff.data()[1] * diff.data()[1] + diff.data()[2] * diff.data()[2]);
	}
	err = sum_err / expl_size;

	return 1;
}

int RotationTransformation(
	const std::vector<Eigen::Vector3d>& src_pts, 
	const std::vector<Eigen::Vector3d>& dst_pts, 
	Eigen::Matrix3d & rMat)
{
	int expl_size = dst_pts.size();
	if (expl_size < 3 || src_pts.size() != expl_size)
		return 0;

	// Generate the covariance matrix
	Eigen::Matrix3d cov_m;
	cov_m.setZero();

	for (int i = 0; i < expl_size; i++) {
		Eigen::Vector3d ds = src_pts[i];
		Eigen::Vector3d dd = dst_pts[i];

		Eigen::MatrixXd dsTds = ds.transpose() * ds;
		Eigen::MatrixXd ddTdd = dd.transpose() * dd;

		cov_m += ds * dd.transpose();
	}
	cov_m /= expl_size;

	// use SVD to compute rotation and then the translation.
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_m, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d Z = Eigen::Matrix3d::Identity();
	Eigen::Matrix3d VUt = V * (U.transpose());
	Z.data()[8] = VUt.determinant();

	rMat = V * Z*(U.transpose());

	return 1;
}

int TranslationTransformation(
	const std::vector<Eigen::Vector3d>& src_pts, 
	const std::vector<Eigen::Vector3d>& dst_pts, 
	Eigen::Matrix3d & rotation, 
	Eigen::Vector3d & translation, 
	double & scale, 
	double & err)
{
	int expl_size = dst_pts.size();
	if (expl_size < 3 || src_pts.size() != expl_size)
		return 0;

	// compute center
	Eigen::Vector3d s_center, d_center;
	s_center.setZero();
	d_center.setZero();
	for (int i = 0; i < expl_size; i++) {
		s_center += src_pts[i];
		d_center += dst_pts[i];
	}
	s_center /= expl_size;
	d_center /= expl_size;

	// Generate the covariance matrix
	Eigen::Matrix3d cov_m;
	cov_m.setZero();
	double s_mv = 0.0;
	double d_mv = 0.0;

	for (int i = 0; i < expl_size; i++) {
		Eigen::Vector3d ds = src_pts[i] - s_center;
		Eigen::Vector3d dd = dst_pts[i] - d_center;

		Eigen::MatrixXd dsTds = ds.transpose() * ds;
		Eigen::MatrixXd ddTdd = dd.transpose() * dd;

		s_mv += dsTds.data()[0];
		d_mv += ddTdd.data()[0];
		cov_m += ds * dd.transpose();
	}

	d_mv /= expl_size;
	s_mv /= expl_size;
	cov_m /= expl_size;

	// use SVD to compute rotation and then the translation.
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_m, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d Z = Eigen::Matrix3d::Identity();
	Eigen::Matrix3d VUt = V * (U.transpose());
	Z.data()[8] = VUt.determinant();

	Eigen::Vector3d S = svd.singularValues();
	scale = (S.data()[0] * Z.data()[0] +
		S.data()[1] * Z.data()[4] +
		S.data()[2] * Z.data()[8]
		) / s_mv;

	translation = -scale * rotation * s_center + d_center;

	// compute error:
	double sum_err = 0.0f;
	for (int i = 0; i < expl_size; i++) {
		Eigen::Vector3d diff = scale * rotation * src_pts[i] + translation - dst_pts[i];
		sum_err += sqrt(diff.data()[0] * diff.data()[0] + diff.data()[1] * diff.data()[1] + diff.data()[2] * diff.data()[2]);
	}
	err = sum_err / expl_size;

	return 1;
}
