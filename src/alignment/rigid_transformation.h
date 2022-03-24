/**************************************************************************************/
/*                                                                                    */
/*  Transformation Library                                                            */
/*  https://github.com/keepdash/Transformation                                        */
/*                                                                                    */
/*  Copyright (c) 2017-2017, Wei Ye                                                   */
/*  All rights reserved.                                                              */
/*                                                                                    */
/**************************************************************************************/

#ifndef __TRANSFORMATION__ESTIMATION__
#define __TRANSFORMATION__ESTIMATION__

#include <vector>
#include <Eigen/Dense>

/************************************************************************/
/* 1. Rigid Transformation Estimation									*/
/************************************************************************/

/**
* Estimation rigid transformation of 2 point sets
* @param src_pts: the source point set
* @param dst_pts: the destiny point set
* @param: weight: the weight array of the point set, size should be equal to point set.
* @param: tfMat: the output 4*4 transformation matrix.
* @param: err: the output error of the estimation.
* @remark: if all points have same weight, set all values in [weight] to 1.0f.
*/
extern int RigidTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix4d &tfMat,
	double& err);

/**
* Estimation rigid transformation of 2 point sets
* @param src_pts: the source point set
* @param dst_pts: the destiny point set
* @param: weight: the weight array of the point set, size should be equal to point set.
* @param: rotation: the output 3*3 rotation matrix.
* @param: translation: the output 3*1 translation vector.
* @param: err: the output error of the estimation.
* @remark: if all points have same weight, set all values in [weight] to 1.0f.
*/
extern int RigidTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix3d &rotation,
	Eigen::Vector3d &translation,
	double& err);


extern int RigidTransformation(
	const std::vector<Eigen::Vector2d> &src_pts,
	const std::vector<Eigen::Vector2d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix2d &rotation,
	Eigen::Vector2d &translation,
	double& err);

/************************************************************************/
/* 2. Similarity Transformation Estimation                              */
/************************************************************************/

/**
* Estimation rigid transformation of 2 point sets
* @param src_pts: the source point set
* @param dst_pts: the destiny point set
* @param: weight: the weight array of the point set, size should be equal to point set.
* @param: tfMat: the output 4*4 transformation matrix.
* @param: err: the output error of the estimation.
* @remark: if all points have same weight, set all values in [weight] to 1.0f.
*/
extern int SimilarityTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix4d &tfMat,
	double& err);

/**
* Estimation rigid transformation of 2 point sets
* @param src_pts: the source point set
* @param dst_pts: the destiny point set
* @param: weight: the weight array of the point set, size should be equal to point set.
* @param: rotation: the output 3*3 rotation matrix.
* @param: translation: the output 3*1 translation vector.
* @param: scale: the output 1*1 scale scalar.
* @param: err: the output error of the estimation.
* @remark: if all points have same weight, set all values in [weight] to 1.0f.
*/
extern int SimilarityTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix3d &rotation,
	Eigen::Vector3d &translation,
	double& scale,
	double& err);


extern int SimilarityTransformation(
	const std::vector<Eigen::Vector2d> &src_pts,
	const std::vector<Eigen::Vector2d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix2d &rotation,
	Eigen::Vector2d &translation,
	double& scale,
	double& err);

extern int SimilarityTransformationRANSAC(
	const std::vector<Eigen::Vector2d> &src_pts,
	const std::vector<Eigen::Vector2d> &dst_pts,
	const std::vector<double> &weight,
	Eigen::Matrix2d &rotation,
	Eigen::Vector2d &translation,
	double& scale,
	double& err);

void RandVectorN(int v_min, int v_max, int N, int seed, std::vector<int>& values);

/************************************************************************/
/* 3. Rotation                                                   */
/************************************************************************/

/**
* Estimation rigid transformation of 2 point sets
* @param src_pts: the source point set
* @param dst_pts: the destiny point set
*/
extern int RotationTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	Eigen::Matrix3d &rMat);


/************************************************************************/
/* 4. Translation and scale given rotation                                                   */
/************************************************************************/

/**
* Estimation rigid transformation of 2 point sets
* @param src_pts: the source point set
* @param dst_pts: the destiny point set
*/
extern int TranslationTransformation(
	const std::vector<Eigen::Vector3d> &src_pts,
	const std::vector<Eigen::Vector3d> &dst_pts,
	Eigen::Matrix3d &rotation,
	Eigen::Vector3d &translation,
	double& scale,
	double& err);

#endif