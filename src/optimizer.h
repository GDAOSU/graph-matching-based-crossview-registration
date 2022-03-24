// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OBJECTSFM_OBJ_OPTIMIZER_H_
#define OBJECTSFM_OBJ_OPTIMIZER_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "ceres/ceres.h"
#include "basic_structs.h"

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SM;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SM_row;
typedef Eigen::MatrixXd DM;
typedef Eigen::Triplet<double> Tri;

namespace objectsfm {
class Camera;
class Point3D;

class BundleAdjuster
{
public:
	BundleAdjuster(std::vector<Camera*> &cams, std::vector<CameraModel*> &cam_models, std::vector<Point3D*> &pts);

	BundleAdjuster(std::vector<Camera*> &cams, CameraModel* cam_models, std::vector<Point3D*> &pts);

	BundleAdjuster(Camera* cam, CameraModel* cam_models, std::vector<Point3D*> &pts);

	~BundleAdjuster();

	void SetOptions(BundleAdjustOptions options);

	void RunOptimizetion(bool is_seed, double weight);

	// design for registration
	void RunOptimizetionRegistration(double weight, std::vector<Eigen::Vector3d>& pose_ori);

	void FindOutliersPoints();

	void UpdateParameters();

	void ReleaseBundleData();	

	// covariance
	void SetupJacobianBlocksOrder(std::vector<double*> &parameter_blocks);

	void CalcCovarianceCeres();

	void CalcCovarianceBlock();	

	void SetInitPrecision();	

	void ComputeScaleFromRight(const SM &A, SM &S);

	void ComputeSchurComplement(const int num_points, const SM &Ms, DM &invZs, SM &invAs, SM &Bs);

	void PseudoInv3x3(const SM & A, SM & invA);

	void PseudoInvNxN(const DM & A, DM & invA, double epsilon = std::numeric_limits<DM::Scalar>::epsilon());

	void MPPseudoInvNxN(const DM & A, DM & invA, double epsilon = std::numeric_limits<DM::Scalar>::epsilon());

	void ComputeUncertaintyOfPoints(const int num_pts, const SM &Ys, const DM &iZs, const SM &iAs, std::vector<Eigen::Matrix3d>& cov_pts);

	void UnscalePointsCovarianceMatrices(SM &S, std::vector<Eigen::Matrix3d> &cov_pts);

	void UnscaleCamsCovarianceMatrices(SM &S, DM &iZs);

	void WriteCovariance(std::string fold);

	void SaveMatrix(SM &A, std::string file);

	void SaveMatrix(DM &A, std::string file);

public:
	ceres::Problem problem_;

	ceres::Solver::Options options_;
	
	ceres::Solver::Summary summary_;

	ceres::Covariance *covariance_;

	std::vector<Camera*> cams_;

	std::vector<CameraModel*> cam_models_;

	std::vector<Point3D*> pts_;

	std::map<size_t, double*> paras_pts_, paras_cams_, paras_models_;

	int n_obs_;
	std::vector<double> weigth_obs_;
	SM init_cov_, init_precision_;
	double sigma_naut2_;
	std::map<size_t, Eigen::Matrix3d> cov_pts_;  // x y z
	std::map<size_t, Eigen::Matrix<double, 6, 6>> cov_cams_; // axis_x axis_y axis_z x y z
	std::map<size_t, Eigen::Matrix<double, 7, 7>> cov_cam_models_;  // f, k1, k2, dx, dy, p1, p2
	std::string distort_mode_;

	// Move the "center" of the reconstruction to the origin, where the center is determined by computing the 
	// marginal median of the points. The reconstruction is then scaled so that the median absolute deviation 
	// of the points measured from the origin is 100.0.
	void Normalize();

	void Perturb();
};


}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_OPTIMIZER_H_
