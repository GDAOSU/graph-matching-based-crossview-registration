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

#include "optimizer.h"

#include <fstream>
#include <iomanip>
#include "camera.h"
#include "structure.h"
#include "utils/basic_funcs.h"
#include "utils/reprojection_error_pose_cam_xyz.h"
#include "utils/reprojection_error_pose_xyz.h"
#include "utils/reprojection_error_pose_cam.h"
#include "utils/reprojection_error_pose.h"
#include "utils/reprojection_error_xyz.h"
#include "utils/z_error_pose_pose.h"
#include "utils/gps_error_pose_raletive_angle.h"
#include "utils/gps_error_pose_absolute.h"

namespace objectsfm {


BundleAdjuster::BundleAdjuster(std::vector<Camera*> &cams, std::vector<CameraModel*> &cam_models, std::vector<Point3D*> &pts)
{
	cams_ = cams;
	cam_models_ = cam_models;
	pts_ = pts;
}

BundleAdjuster::BundleAdjuster(std::vector<Camera*> &cams, CameraModel* cam_models, std::vector<Point3D*> &pts)
{
	cams_ = cams;
	cam_models_.push_back(cam_models);
	pts_ = pts;
}

BundleAdjuster::BundleAdjuster(Camera * cam, CameraModel * cam_models, std::vector<Point3D*>& pts)
{
	cams_.push_back(cam);
	cam_models_.push_back(cam_models);
	pts_ = pts;
}

BundleAdjuster::~BundleAdjuster()
{
}

void BundleAdjuster::SetOptions(BundleAdjustOptions options)
{
	options_.max_num_iterations = options.max_num_iterations;
	options_.minimizer_progress_to_stdout = options.minimizer_progress_to_stdout;
	options_.num_threads = options.num_threads;	
	distort_mode_ = options.distort_mode;
	if (options.ceres_solver_type == 5) {
		options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
	}
	else {
		options_.linear_solver_type = ceres::DENSE_SCHUR;
	}
	options_.function_tolerance = options.th_tolerance;
}

void BundleAdjuster::RunOptimizetion(bool is_initial_run, double weight)
{
	// for initial run, normalize the distribution of these 3D points
	if (is_initial_run)	{
		Normalize();
		Perturb();
	}
	// build the problem
	n_obs_ = 0;
	int num_pts = pts_.size();
	ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

	int count_edges = 0;
	for (size_t i = 0; i < num_pts; i += 1)
	{
		if (pts_[i]->is_bad_estimated_) {
			continue;
		}

		std::map<size_t, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
		std::map<size_t, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
		while (iter_cams != pts_[i]->cams_.end())
		{
			ceres::CostFunction* cost_function;			
			double w = pts_[i]->weight_;			
			if (pts_[i]->is_mutable_ && iter_cams->second->is_mutable_)
			{
				if (iter_cams->second->cam_model_->is_mutable_)
				{
					n_obs_++;
					weigth_obs_.push_back(w);
					count_edges++;
					cost_function = ReprojectionErrorPoseCamXYZ::Create(iter_pts->second(0), iter_pts->second(1), 
						w, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
					problem_.AddResidualBlock(cost_function, loss_function,
						iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->data);

					paras_pts_.insert(std::pair<size_t, double*>(pts_[i]->id_, pts_[i]->data));
					paras_cams_.insert(std::pair<size_t, double*>(iter_cams->second->id_, iter_cams->second->data));
					paras_models_.insert(std::pair<size_t, double*>(iter_cams->second->cam_model_->id_, iter_cams->second->cam_model_->data));
				}
				else
				{
					n_obs_++;
					weigth_obs_.push_back(w);
					count_edges++;
					cost_function = ReprojectionErrorPoseXYZ::Create(iter_pts->second(0), iter_pts->second(1), 
						iter_cams->second->cam_model_->data, w, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, pts_[i]->data);
					paras_pts_.insert(std::pair<size_t, double*>(pts_[i]->id_, pts_[i]->data));
					paras_cams_.insert(std::pair<size_t, double*>(iter_cams->second->id_, iter_cams->second->data));					
				}
				
			}
			else if (pts_[i]->is_mutable_ && !iter_cams->second->is_mutable_)
			{
				n_obs_++;
				weigth_obs_.push_back(w);
				count_edges++;
				cost_function = ReprojectionErrorXYZ::Create(iter_pts->second(0), iter_pts->second(1), 
					iter_cams->second->data, iter_cams->second->cam_model_->data, w, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
				problem_.AddResidualBlock(cost_function, loss_function, pts_[i]->data);
				paras_pts_.insert(std::pair<size_t, double*>(pts_[i]->id_, pts_[i]->data));
			}
			else if (!pts_[i]->is_mutable_ && iter_cams->second->is_mutable_)
			{
				if (iter_cams->second->cam_model_->is_mutable_)
				{
					n_obs_++;
					weigth_obs_.push_back(w);
					count_edges++;
					cost_function = ReprojectionErrorPoseCam::Create(iter_pts->second(0), iter_pts->second(1), 
						pts_[i]->data, w, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, iter_cams->second->cam_model_->data);
					paras_cams_.insert(std::pair<size_t, double*>(iter_cams->second->id_, iter_cams->second->data));
					paras_models_.insert(std::pair<size_t, double*>(iter_cams->second->cam_model_->id_, iter_cams->second->cam_model_->data));
				}
				else
				{
					n_obs_++;
					weigth_obs_.push_back(w);
					count_edges++;
					cost_function = ReprojectionErrorPose::Create(iter_pts->second(0), iter_pts->second(1), 
						pts_[i]->data, iter_cams->second->cam_model_->data, w, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data);
					paras_cams_.insert(std::pair<size_t, double*>(iter_cams->second->id_, iter_cams->second->data));					
				}
			}

			//std::cout << iter_cams->second->cam_model_->f_hyp_ << std::endl;

			iter_cams++;
			iter_pts++;
		}
	}

	//std::cout << "count edges " << count_edges << std::endl;

	// solve the problem
	ceres::Solve(options_, &problem_, &summary_);

	//std::cout << "come on!!!" << std::endl;
}

void BundleAdjuster::RunOptimizetionRegistration(double weight, std::vector<Eigen::Vector3d>& pose_ori)
{
	// build the problem
	int num_cams = cams_.size();
	int num_pts = pts_.size();
	ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

	// step1: reprojection error
	int count_edges = 0;
	for (size_t i = 0; i < num_pts; i += 10)
	{
		if (pts_[i]->is_bad_estimated_) {
			continue;
		}
		if (pts_[i]->cams_.size() < 3) {
			pts_[i]->weight_ = 1.0;
		}
		if (pts_[i]->cams_.size() >= 3) {
			pts_[i]->weight_ = weight;
		}

		std::map<size_t, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
		std::map<size_t, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
		while (iter_cams != pts_[i]->cams_.end())
		{
			//
			ceres::CostFunction* cost_function;
			if (pts_[i]->is_mutable_ && iter_cams->second->is_mutable_)
			{
				if (iter_cams->second->cam_model_->is_mutable_)
				{
					count_edges++;
					cost_function = ReprojectionErrorPoseCamXYZ::Create(iter_pts->second(0), iter_pts->second(1), 
						pts_[i]->weight_, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
					problem_.AddResidualBlock(cost_function, loss_function,
						iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->data);
				}
				else
				{
					count_edges++;
					cost_function = ReprojectionErrorPoseXYZ::Create(iter_pts->second(0), iter_pts->second(1),
						iter_cams->second->cam_model_->data, pts_[i]->weight_, iter_cams->second->cam_model_->f_hyp_, distort_mode_);
					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, pts_[i]->data);
				}

			}
			else if (pts_[i]->is_mutable_ && !iter_cams->second->is_mutable_)
			{
				count_edges++;
				cost_function = ReprojectionErrorXYZ::Create(iter_pts->second(0), iter_pts->second(1),
					iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->weight_, iter_cams->second->cam_model_->f_hyp_, distort_mode_);

				problem_.AddResidualBlock(cost_function, loss_function, pts_[i]->data);
			}
			else if (!pts_[i]->is_mutable_ && iter_cams->second->is_mutable_)
			{
				if (iter_cams->second->cam_model_->is_mutable_)
				{
					count_edges++;
					cost_function = ReprojectionErrorPoseCam::Create(iter_pts->second(0), iter_pts->second(1), pts_[i]->data, 
						pts_[i]->weight_, iter_cams->second->cam_model_->f_hyp_, distort_mode_);

					problem_.AddResidualBlock(cost_function, loss_function,
						iter_cams->second->data, iter_cams->second->cam_model_->data);
				}
				else
				{
					count_edges++;
					cost_function = ReprojectionErrorPose::Create(iter_pts->second(0), iter_pts->second(1),
						pts_[i]->data, iter_cams->second->cam_model_->data, pts_[i]->weight_, iter_cams->second->cam_model_->f_hyp_, distort_mode_);

					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data);
				}
			}

			iter_cams++;
			iter_pts++;
		}
	}

	// step2: absolute pose error
	int count_edges2 = 0;
	for (int i = 0; i < num_cams; i++) {
		ceres::CostFunction* cost_function;
		cost_function = GPSErrorPoseAbsolute::Create(pose_ori[i](0), pose_ori[i](1), pose_ori[i](2), 10.0);
		problem_.AddResidualBlock(cost_function, loss_function, cams_[i]->data);
		count_edges2++;
	}
	std::cout << "count_edges1 " << count_edges << " count_edges2 " << count_edges2 << std::endl;

	// solve the problem
	ceres::Solve(options_, &problem_, &summary_);
}

void BundleAdjuster::FindOutliersPoints()
{

}

void BundleAdjuster::UpdateParameters()
{
	for (size_t i = 0; i < cams_.size(); i++)
	{
		cams_[i]->UpdatePoseFromData();
	}

	for (size_t i = 0; i < cam_models_.size(); i++)
	{
		cam_models_[i]->UpdataModelFromData();
	}
}

void BundleAdjuster::ReleaseBundleData()
{
}

void BundleAdjuster::SetupJacobianBlocksOrder(std::vector<double*>& parameter_blocks)
{
	std::cout << "pts: " << paras_pts_.size() << "cams: " << paras_cams_.size() << "models: " << paras_models_.size() << std::endl;
	for (auto &pt : paras_pts_)
		parameter_blocks.push_back(pt.second);
	for (auto &cam : paras_cams_)
		parameter_blocks.push_back(cam.second);
	for (auto &model : paras_models_)
		parameter_blocks.push_back(model.second);
}

void BundleAdjuster::CalcCovarianceCeres()
{
	// setup Jacobian blocks order
	std::vector<double*> parameter_blocks;
	SetupJacobianBlocksOrder(parameter_blocks);

	// Jacobian
	double cost = 0.0;
	ceres::Problem::EvaluateOptions eval_opt;
	eval_opt.parameter_blocks = parameter_blocks;
	eval_opt.apply_loss_function = true;
	ceres::CRSMatrix J = ceres::CRSMatrix();	
	problem_.Evaluate(eval_opt, &cost, NULL, NULL, &J);

	// covariance blocks
	ceres::Covariance::Options cov_opt;
	covariance_ = new ceres::Covariance(cov_opt);
	std::vector<std::pair<const double*, const double*> > covariance_blocks;
	for (auto &pt : paras_pts_) {
		covariance_blocks.push_back(std::make_pair(pt.second, pt.second));
	}		
	for (auto &cam : paras_cams_)
		covariance_blocks.push_back(std::make_pair(cam.second, cam.second));

	covariance_->Compute(covariance_blocks, &problem_);
	int count = 0;
	for (auto &pt : paras_pts_) {
		double *cov_pt_data;
		covariance_->GetCovarianceBlock(pt.second, pt.second, cov_pt_data);
		Eigen::Matrix3d cov_pt_temp;
		for (size_t i = 0; i < 3; i++) {
			for (size_t j = 0; j < 3; j++) {
				cov_pt_temp(i, j) = cov_pt_data[i * 3 + j];
			}
		}
		if (count % 10000 == 0) {
			std::cout << cov_pt_temp << std::endl;
		}
		cov_pts_.insert(std::make_pair(pt.first, cov_pt_temp));
	}
	for (auto &cam : paras_cams_) {
		Eigen::Matrix<double, 6, 6> cov_cam_temp;
		double *cov_cam_data;
		covariance_->GetCovarianceBlock(cam.second, cam.second, cov_cam_data);
		for (size_t i = 0; i < 3; i++) {
			for (size_t j = 0; j < 3; j++) {
				cov_cam_temp(i, j) = cov_cam_data[i * 3 + j];
			}
		}
		std::cout << cov_cam_temp << std::endl;
		cov_cams_.insert(std::make_pair(cam.first, cov_cam_temp));
	}
}


void BundleAdjuster::CalcCovarianceBlock()
{
	// setup Jacobian blocks order
	std::vector<double*> parameter_blocks;
	SetupJacobianBlocksOrder(parameter_blocks);
	std::cout << parameter_blocks.size() << std::endl;

	// Jacobian
	double cost = 0.0;
	ceres::Problem::EvaluateOptions eval_opt;
	eval_opt.parameter_blocks = parameter_blocks;
	eval_opt.apply_loss_function = true;
	ceres::CRSMatrix JJ = ceres::CRSMatrix();
	std::vector<double> residuals;
	problem_.Evaluate(eval_opt, &cost, &residuals, NULL, &JJ);
	SM J = Eigen::Map<SM_row>(JJ.num_rows, JJ.num_cols, JJ.cols.size(), JJ.rows.data(), JJ.cols.data(), JJ.values.data());
	SM s;
	ComputeScaleFromRight(J, s);
	J = J * s;

	// block inversion
	SM M(J.transpose() * init_precision_ * J);
	//std::string file = "F:\\M.txt";
	//SaveMatrix(M, file);

	DM invZ;  // M = [A B; B^T D], invZ = (D - B.transpose() * invA * B)^-1
	SM invA, B;
	ComputeSchurComplement(paras_pts_.size(), M, invZ, invA, B);

	// uncertainty of pts
	SM Y = invA * B;
	std::vector<Eigen::Matrix3d> cov_pts;
	ComputeUncertaintyOfPoints(paras_pts_.size(), Y, invZ, invA, cov_pts);
	UnscalePointsCovarianceMatrices(s, cov_pts);	
	//std::cout << cov_pts[0] << std::endl;
	//std::cout << cov_pts[100] << std::endl;

	// uncertainty of cams
	UnscaleCamsCovarianceMatrices(s, invZ);
	//std::string file3 = "F:\\Zinv_scale.txt";
	//SaveMatrix(invZ, file3);
	//std::cout << invZ.block(invZ.rows() - 7, invZ.rows() - 7, 7, 7) << std::endl;

	// calculate variance factor as the sum of the squared residuals	
	Eigen::VectorXd v(residuals.size());
	for (size_t i = 0; i < residuals.size(); i++) {
		v(i) = residuals[i];
	}
	sigma_naut2_ = v.transpose()*init_precision_*v;
	sigma_naut2_ /= (residuals.size() - paras_pts_.size() * 3 - paras_cams_.size() * 6 - paras_models_.size() * 7);
	std::cout << "sigma_naut square: " << sigma_naut2_ << std::endl;

	// 
	int count_pts = 0;
	for (auto iter : paras_pts_) {
		size_t id = iter.first;
		cov_pts_.insert(std::make_pair(id, cov_pts[count_pts]));
		count_pts++;
	}

	int count_cams = 0;
	for (auto iter : paras_cams_) {
		size_t id = iter.first;
		Eigen::Matrix<double, 6, 6> cov_cam_temp = invZ.block(count_cams * 6, count_cams * 6, 6, 6);
		cov_cams_.insert(std::make_pair(id, cov_cam_temp));
		count_cams++;
	}

	int count_cam_models = 0;
	for (auto iter : paras_models_) {
		size_t id = iter.first;
		Eigen::Matrix<double, 7, 7> cov_cam_model_temp = invZ.block(count_cams * 6 + count_cam_models * 7,
			count_cams * 6 + count_cam_models * 7, 7, 7);
		cov_cam_models_.insert(std::make_pair(id, cov_cam_model_temp));
		count_cam_models++;
	}
}


void BundleAdjuster::SetInitPrecision()
{
	std::cout << n_obs_ << std::endl;

	init_cov_ = SM(2 * n_obs_, 2 * n_obs_);
	init_precision_ = SM(2 * n_obs_, 2 * n_obs_);

	// compute the sum of squred residuals
	std::vector<double> reproj_errors(2 * n_obs_, 1.0);
	//reprojectionErrors(scene, reproj_errors);

	// set identity matrix from triplets
	std::vector<Tri> cov_triplets;
	std::vector<Tri> precision_triplets;
	cov_triplets.reserve(2 * n_obs_);
	precision_triplets.reserve(2 * n_obs_);
	for (int i = 0; i < n_obs_; ++i) {
		double w = weigth_obs_[i];
		cov_triplets.push_back(Tri(2 * i + 0, 2 * i + 0, reproj_errors[i] * reproj_errors[i]));
		cov_triplets.push_back(Tri(2 * i + 1, 2 * i + 1, reproj_errors[i] * reproj_errors[i]));
		precision_triplets.push_back(Tri(2 * i + 0, 2 * i + 0, w));
		precision_triplets.push_back(Tri(2 * i + 1, 2 * i + 1, w));
	}
	init_cov_.setFromTriplets(cov_triplets.begin(), cov_triplets.end());
	init_precision_.setFromTriplets(precision_triplets.begin(), precision_triplets.end());
}

void BundleAdjuster::ComputeScaleFromRight(const SM &A, SM &S)
{
	S.resize(A.cols(), A.cols());
	S.reserve(A.cols());
	std::vector<Tri> STriplets;
	for (int i = 0; i < A.cols(); ++i) {		// all columns
		double ss = 0;							// sum of squares
		for (int j = A.outerIndexPtr()[i]; j < A.outerIndexPtr()[i + 1]; ++j)  // all rows at column
			ss += A.valuePtr()[j] * A.valuePtr()[j];
		STriplets.push_back(Tri(i, i, 1 / sqrt(ss)));
	}
	S.setFromTriplets(STriplets.begin(), STriplets.end());
}



void BundleAdjuster::ComputeSchurComplement(const int num_points, const SM & Ms, DM & invZs, SM & invAs, SM & Bs)
{
	// Compose the submatrices Ms = [As Bs; Bs^T Ds]   
	// Matrix As is related to point parameters, Ds is related to cameras and Bs is related to both
	int num_pt_params = 3 * num_points;
	int num_cam_params = Ms.cols() - num_pt_params;
	std::cout << "num_pt_params " << num_pt_params << std::endl;
	std::cout << "num_cam_params " << num_cam_params << std::endl;

	SM As(num_pt_params, num_pt_params);
	SM Ds(num_cam_params, num_cam_params);
	invAs.resize(num_pt_params, num_pt_params);
	Bs.resize(num_pt_params, num_cam_params);
	std::vector<Tri> AsT, BsT, DsT;
	for (int i = 0; i < Ms.outerSize(); ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(Ms, i); it; ++it) {
			if (it.row() < num_pt_params && it.col() < num_pt_params)		// As
				AsT.push_back(Tri(it.row(), it.col(), it.value()));
			if (it.row() < num_pt_params && it.col() >= num_pt_params)		// Bs
				BsT.push_back(Tri(it.row(), it.col() - num_pt_params, it.value()));
			if (it.row() >= num_pt_params && it.col() >= num_pt_params)		// Ds
			{
				DsT.push_back(Tri(it.row() - num_pt_params, it.col() - num_pt_params, it.value()));
			}
		}
	}
	As.setFromTriplets(AsT.begin(), AsT.end());
	Bs.setFromTriplets(BsT.begin(), BsT.end());
	Ds.setFromTriplets(DsT.begin(), DsT.end());

	// Invert block diagonal matrix A
	PseudoInv3x3(As, invAs);

	// Compute Z
	DM Zs = (DM(Ds - Bs.transpose() * invAs * Bs));
	//std::string file = "F:\\Z.txt";
	//SaveMatrix(Zs, file);
	MPPseudoInvNxN(Zs, invZs, 0.0001);
	//std::string file2 = "F:\\Zinv_pesud.txt";
	//SaveMatrix(invZs, file2);
	DM Zsinv = Zs.inverse();
	//std::string file3 = "F:\\Zinv_inv.txt";
	//SaveMatrix(Zsinv, file3);
}

void BundleAdjuster::PseudoInv3x3(const SM & A, SM & invA)
{
	invA = A;		// clone all the indexes wich remain unchanged, i.e., the column and row indexes will be the same
	double const *val = A.valuePtr();
	double *Aval = invA.valuePtr();
	for (int i = 0; i < A.nonZeros(); i = i + 9) {
		Eigen::Matrix3d a;
		a << val[i], val[i + 1], val[i + 2],
			val[i + 3], val[i + 4], val[i + 5],
			val[i + 6], val[i + 7], val[i + 8];
		Eigen::Matrix3d a_inv = a.completeOrthogonalDecomposition().pseudoInverse();
		Aval[i + 0] = a_inv(0, 0);
		Aval[i + 1] = a_inv(0, 1);
		Aval[i + 2] = a_inv(0, 2);
		Aval[i + 3] = a_inv(1, 0);
		Aval[i + 4] = a_inv(1, 1);
		Aval[i + 5] = a_inv(1, 2);
		Aval[i + 6] = a_inv(2, 0);
		Aval[i + 7] = a_inv(2, 1);
		Aval[i + 8] = a_inv(2, 2);
	}
}

void BundleAdjuster::PseudoInvNxN(const DM & A, DM & invA, double epsilon)
{
	if (A.rows()<A.cols())
		return;

	Eigen::JacobiSVD< DM> svd = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	DM::Scalar tolerance = epsilon * std::max(A.cols(), A.rows()) * svd.singularValues().array().abs().maxCoeff();
	invA = svd.matrixV() * DM((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().
		array().inverse(), 0)).asDiagonal() * svd.matrixU().adjoint();
}

void BundleAdjuster::MPPseudoInvNxN(const DM & A, DM & invA, double epsilon)
{
	invA = A.completeOrthogonalDecomposition().pseudoInverse();
}

void BundleAdjuster::ComputeUncertaintyOfPoints(const int num_pts, const SM & Ys, const DM & iZs, const SM & iAs, std::vector<Eigen::Matrix3d>& cov_pts)
{
	cov_pts.resize(num_pts);

	// propagation to points is less numerical senzitive and we can compute it in floats to speed it up
	Eigen::SparseMatrix<float, Eigen::RowMajor> fYs = Eigen::SparseMatrix<float, Eigen::RowMajor>(Ys.cast<float>());
	Eigen::MatrixXf fiZs = iZs.cast<float>();
	float *iA = (float*)malloc(sizeof(float) * iAs.nonZeros());
	for (int i = 0; i < iAs.nonZeros(); ++i)
		iA[i] = static_cast<float>(iAs.valuePtr()[i]);

	// parallel sparse multiplication of needed blocks
	const int *rowsY = fYs.outerIndexPtr();
	const int *colsY = fYs.innerIndexPtr();
	const float *valuesY = fYs.valuePtr();
	const float *iZ = fiZs.data();

	int n = fiZs.cols();			// munber of camera parameters
	int nrowsY = fYs.rows();
	std::vector<std::vector<float>> h(nrowsY);

#pragma omp parallel for
	for (int i = 0; i < nrowsY; ++i) {				// rows of Y
		int row_from = rowsY[i];
		int row_to = rowsY[i + 1];
		int ncols = row_to - row_from;
		h[i].resize(ncols);

		for (int j = 0; j < ncols; ++j) {		// all columns on the row
			int col_id = colsY[row_from + j];
			float hi = 0;
			for (int k = 0; k < ncols; ++k) 	// multiplication of row j-th column iZ
				hi += valuesY[k + row_from] * iZ[n*col_id + colsY[k + row_from]];
			h[i][j] = hi;
		}
	}

#pragma omp parallel for
	for (int i = 0; i < num_pts; ++i) {				// each point m has values on three rows h
		std::vector<float> &r0 = h[3 * i + 0];
		std::vector<float> &r1 = h[3 * i + 1];
		std::vector<float> &r2 = h[3 * i + 2];
		int row_from = rowsY[3 * i];
		int row_to = rowsY[3 * i + 1];
		int ncols = row_to - row_from;

		float v00 = 0, v01 = 0, v02 = 0, v11 = 0, v12 = 0, v22 = 0;
		for (int j = 0; j < ncols; ++j) {
			v00 += r0[j] * valuesY[j + 0 * ncols + row_from];
			v01 += r0[j] * valuesY[j + 1 * ncols + row_from];
			v02 += r0[j] * valuesY[j + 2 * ncols + row_from];
			v11 += r1[j] * valuesY[j + 1 * ncols + row_from];
			v12 += r1[j] * valuesY[j + 2 * ncols + row_from];
			v22 += r2[j] * valuesY[j + 2 * ncols + row_from];
		}

		cov_pts[i].resize(3, 3);
		cov_pts[i] <<
			v00 + iA[9 * i + 0], v01 + iA[9 * i + 3], v02 + iA[9 * i + 6],
			v01 + iA[9 * i + 1], v11 + iA[9 * i + 4], v12 + iA[9 * i + 7],
			v02 + iA[9 * i + 2], v12 + iA[9 * i + 5], v22 + iA[9 * i + 8];		
	}

	free(iA);
}

void BundleAdjuster::UnscalePointsCovarianceMatrices(SM &S, std::vector<Eigen::Matrix3d> &cov_pts)
{
	double *s_ptr = S.valuePtr();
	for (int i = 0; i < cov_pts.size(); ++i) {
		Eigen::Vector3d s;
		s << s_ptr[3 * i], s_ptr[3 * i + 1], s_ptr[3 * i + 2];
		cov_pts[i] = Eigen::Matrix3d(s.asDiagonal() * cov_pts[i] * s.asDiagonal());
	}
}

void BundleAdjuster::UnscaleCamsCovarianceMatrices(SM & S, DM & iZs)
{
	Eigen::VectorXd s(S.rows());
	s.block(0, 0, S.rows(), 1) = S.diagonal();
	SM SS(s.block(s.rows() - iZs.rows(), 0, iZs.rows(), 1).asDiagonal());
	iZs = SS * iZs * SS;
}

void BundleAdjuster::WriteCovariance(std::string fold)
{
	// pts
	std::string file_pts = fold + "//" + "cov_pts.txt";
	std::string file_pts_show = fold + "//" + "cov_pts_show.txt";
	std::ofstream ofs(file_pts);
	std::ofstream ofs_show(file_pts_show);
	auto iter_pts = paras_pts_.begin();
	for (auto iter : cov_pts_) {
		size_t id = iter.first;
		Eigen::Matrix3d cov_temp = sigma_naut2_ * iter.second;
		ofs << id << " ";
		for (size_t i = 0; i < 3; i++) {
			for (size_t j = 0; j < 3; j++) {
				ofs << cov_temp(i, j) << " ";
			}
		}
		ofs << std::endl;
		ofs_show << iter_pts->second[0] << " " << iter_pts->second[1] << " " << iter_pts->second[2] << " " << sqrt(cov_temp(0, 0) + cov_temp(1, 1) + cov_temp(2, 2)) << std::endl;
		iter_pts++;
	}
	ofs.close();
	ofs_show.close();

	// cams
	std::string file_cams = fold + "//" + "cov_cams.txt";
	ofs = std::ofstream(file_cams);
	for (auto iter : cov_cams_) {
		size_t id = iter.first;
		Eigen::Matrix<double, 6, 6> cov_temp = sigma_naut2_ * iter.second;
		ofs << id << " ";
		for (size_t i = 0; i < 6; i++) {
			for (size_t j = 0; j < 6; j++) {
				ofs << cov_temp(i, j) << " ";
			}
		}
		ofs << std::endl;
	}
	ofs.close();

	// models
	std::string file_models = fold + "//" + "cov_cam_models.txt";
	ofs = std::ofstream(file_models);
	for (auto iter : cov_cam_models_) {
		size_t id = iter.first;
		Eigen::Matrix<double, 7, 7> cov_temp = sigma_naut2_ * iter.second;
		ofs << id << " ";
		for (size_t i = 0; i < 7; i++) {
			for (size_t j = 0; j < 7; j++) {
				ofs << cov_temp(i, j) << " ";
			}
		}
		ofs << std::endl;
	}
	ofs.close();
}

void BundleAdjuster::SaveMatrix(SM & A, std::string file)
{
	DM D = DM(A);
	std::ofstream ofs(file);
	ofs << std::fixed << std::setprecision(12);
	for (size_t i = 0; i < D.rows(); i++) {
		for (size_t j = 0; j < D.cols(); j++) {
			ofs << D(i, j) << " ";
		}
		ofs << std::endl;
	}
	ofs.close();
}

void BundleAdjuster::SaveMatrix(DM & A, std::string file)
{
	DM D = A;
	std::ofstream ofs(file);
	ofs << std::fixed << std::setprecision(12);
	for (size_t i = 0; i < D.rows(); i++) {
		for (size_t j = 0; j < D.cols(); j++) {
			ofs << D(i, j) << " ";
		}
		ofs << std::endl;
	}
	ofs.close();
}

void BundleAdjuster::Normalize()
{
	int num_pts = pts_.size();

	// Compute the marginal median of the geometry
	std::vector<double> mid_pt(3,0);
	for (int i = 0; i < num_pts; ++i)
	{
		mid_pt[0] += pts_[i]->data[0];
		mid_pt[1] += pts_[i]->data[1];
		mid_pt[2] += pts_[i]->data[2];
	}
	mid_pt[0] /= num_pts;
	mid_pt[1] /= num_pts;
	mid_pt[2] /= num_pts;

	double median_absolute_deviation = 0;
	for (int i = 0; i < num_pts; ++i)
	{
		double L1dis = abs(pts_[i]->data[0] - mid_pt[0])
			+ abs(pts_[i]->data[1] - mid_pt[1])
			+ abs(pts_[i]->data[2] - mid_pt[2]);
		median_absolute_deviation += L1dis;
	}
	median_absolute_deviation /= num_pts;

	// Scale so that the median absolute deviation of the resulting reconstruction is 100.
	double scale = 100.0 / median_absolute_deviation;
	for (int i = 0; i < num_pts; ++i)
	{
		pts_[i]->data[0] = scale * (pts_[i]->data[0] - mid_pt[0]);
		pts_[i]->data[1] = scale * (pts_[i]->data[1] - mid_pt[1]);
		pts_[i]->data[2] = scale * (pts_[i]->data[2] - mid_pt[2]);
	}

	Eigen::Vector3d mid_pt_eigen(mid_pt[0], mid_pt[1], mid_pt[2]);
	for (int i = 0; i < cams_.size(); ++i)
	{
		cams_[i]->SetACPose(cams_[i]->pos_ac_.a, scale*(cams_[i]->pos_ac_.c - mid_pt_eigen));
	}
}

void BundleAdjuster::Perturb()
{
	double rotation_sigma = 0.1;
	double translation_sigma = 0.5;
	double point_sigma = 0.5;

	// perturb the 3d points
	for (size_t i = 0; i < pts_.size(); i++)
	{
		pts_[i]->data[0] += math::randnormal<double>()*point_sigma;
		pts_[i]->data[1] += math::randnormal<double>()*point_sigma;
		pts_[i]->data[2] += math::randnormal<double>()*point_sigma;
	}

	// perturb the camera
	for (size_t i = 0; i < cams_.size(); i++)
	{
		if (rotation_sigma > 0.0)
		{
			Eigen::Vector3d noise;
			noise[0] = math::randnormal<double>()*rotation_sigma;
			noise[1] = math::randnormal<double>()*rotation_sigma;
			noise[2] = math::randnormal<double>()*rotation_sigma;
			cams_[i]->SetACPose(cams_[i]->pos_ac_.a+noise, cams_[i]->pos_ac_.c);
		}

		if (translation_sigma > 0.0)
		{
			Eigen::Vector3d noise;
			noise[0] = math::randnormal<double>()*translation_sigma;
			noise[1] = math::randnormal<double>()*translation_sigma;
			noise[2] = math::randnormal<double>()*translation_sigma;
			cams_[i]->SetRTPose(cams_[i]->pos_rt_.R, cams_[i]->pos_rt_.t+noise);
		}
	}
}


}  // namespace objectsfm
