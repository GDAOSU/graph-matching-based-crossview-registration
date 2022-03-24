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

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_
#define DIV 1048576
#define WIDTH 7

#include "crossview_alignment.h"

#include <fstream>
#include <filesystem>
#include <iomanip>
#include <windows.h>
#include <tchar.h>

#include <Eigen/Core>
#include "ceres/ceres.h"
#include <opencv2/opencv.hpp>

#include "utils/basic_funcs.h"
#include "utils/converter_utm_latlon.h"
#include "utils/ellipsoid_utm_info.h"
#include "utils/reprojection_error_pose_cam_xyz.h"
#include "utils/reprojection_error_pose_xyz.h"
#include "utils/happly.h"
#include "utils/nanoflann_all.hpp"
#include "utils/nanoflann_utils_all.h"

#include "alignment/registration_gps.h"
#include "alignment/registration_nogps.h"
#include "alignment/common_functions.h"

using namespace nanoflann;

namespace objectsfm {

	CrossviewAlignment::CrossviewAlignment() {
		th_mse_outliers_ = 5.0;
		offx_ = 324917.51;
		offy_ = 4431700.69;
		//offx_ = 0.0;
		//offy_ = 0.0;
		sate_info_ = new SateInfo;
		street_info_ = new StreetInfo;
	}

	CrossviewAlignment::~CrossviewAlignment()
	{
	}

	void CrossviewAlignment::SetSateInfo(std::string sate_dsm, std::string sate_ortho, std::string sate_mask_building, std::string sate_mask_ndvi)
	{
		sate_info_->file_dsm = sate_dsm;
		sate_info_->file_ortho = sate_ortho;
		sate_info_->file_mask_building = sate_mask_building;
		sate_info_->file_mask_ndvi = sate_mask_ndvi;

		std::string file_sate_tfw = sate_info_->file_dsm.substr(0, sate_info_->file_dsm.size() - 4) + ".tfw";
		std::ifstream iff(file_sate_tfw);
		float zeros;
		iff >> sate_info_->dx >> zeros >> zeros >> sate_info_->dy >> sate_info_->xs >> sate_info_->ys;
		iff.close();
	}

	void CrossviewAlignment::SetStreetInfo(std::string fold)
	{
		street_info_->fold = fold;
		street_info_->fold_sfm = fold + "\\sfm";
		street_info_->fold_dense = fold + "\\dense";
		street_info_->fold_align = fold + "\\align";
	}

	void CrossviewAlignment::RunAlign()
	{
		id_cams_ = 0;
		id_came_ = 10000;

		if (!std::experimental::filesystem::exists(street_info_->fold_align)) {
			std::experimental::filesystem::create_directory(street_info_->fold_align);
		}

		std::string file_street_models = street_info_->fold_sfm + "\\num_models.txt";
		std::string file_street_gps = street_info_->fold_sfm + "\\pos.txt";
		std::string fold_street_rgb = street_info_->fold_sfm + "\\rgb";

		bool use_gps = true;
		if (!std::experimental::filesystem::exists(file_street_gps)) {
			use_gps = false;
		}

		// align for each sfm model
		std::ifstream iff(file_street_models);
		int num_model = 0;
		iff >> num_model;
		iff.close();
		for (int i = 0; i < num_model; i++) {
			std::string fold_sfm_i = street_info_->fold_sfm + "\\" + std::to_string(i + 1);
			std::string fold_dense_i = street_info_->fold_dense + "\\" + std::to_string(i + 1);
			std::string fold_align_i = street_info_->fold_align + "\\" + std::to_string(i + 1);
			if (!std::experimental::filesystem::exists(fold_sfm_i)
				|| !std::experimental::filesystem::exists(fold_dense_i))
			{
				continue;
			}
			if (!std::experimental::filesystem::exists(fold_align_i)) {
				std::experimental::filesystem::create_directory(fold_align_i);
			}
			std::string file_full_bd = fold_align_i + "\\cam_pts.txt";
			if (std::experimental::filesystem::exists(file_full_bd)) {
				continue;
			}

			// one sparse model is dense reconstructed into several segs
			iff = std::ifstream(fold_sfm_i + "\\num_segs.txt");
			int num_seg = 0;
			iff >> num_seg;
			iff.close();

			// step1: read in dense		
			int dilation_step = 100;
			std::string file_street_dense_i = fold_align_i + "\\street_dense.txt";
			if (!std::experimental::filesystem::exists(file_street_dense_i)) {
				std::ofstream off(file_street_dense_i);
				for (int j = 0; j < num_seg; j++) {
					std::string file_ply = fold_dense_i + "\\openmvs" + std::to_string(j) + "\\result\\dense.ply";
					//std::string file_ply = fold_dense_i + "\\openmvs" + std::to_string(j) + "\\dense.ply";
					std::cout << file_ply << std::endl;
					if (!std::experimental::filesystem::exists(file_ply)) {
						continue;
					}
					happly::PLYData ply_reader(file_ply);
					std::vector<float> px = ply_reader.getElement("vertex").getProperty<float>("x");
					std::vector<float> py = ply_reader.getElement("vertex").getProperty<float>("y");
					std::vector<float> pz = ply_reader.getElement("vertex").getProperty<float>("z");
					std::vector<uchar> pr = ply_reader.getElement("vertex").getProperty<uchar>("red");
					std::vector<uchar> pg = ply_reader.getElement("vertex").getProperty<uchar>("green");
					std::vector<uchar> pb = ply_reader.getElement("vertex").getProperty<uchar>("blue");
					for (int m = 0; m < px.size(); m += dilation_step) {
						off << px[m] << " " << py[m] << " " << pz[m] << " ";
						off << int(pr[m]) << " " << int(pg[m]) << " " << int(pb[m]) << " " << j << std::endl;
					}
				}
				off.close();
			}
			if (std::experimental::filesystem::is_empty(file_street_dense_i)) {
				continue;
			}

			// step2: alignment the model
			std::string file_street_pose_i = fold_sfm_i + "\\street_pose.txt";
			if (use_gps) {
				RegistrationGPS gps_register;
				gps_register.PreSetPaths(fold_align_i, sate_info_->file_dsm, sate_info_->file_ortho, sate_info_->file_mask_building,
					sate_info_->file_mask_ndvi, file_street_dense_i, file_street_pose_i, file_street_gps);
				gps_register.Run();
			}
			else {
				RegistrationNoGPS nogps_register;
				nogps_register.PreSetPaths(fold_align_i, sate_info_->file_dsm, sate_info_->file_ortho, sate_info_->file_mask_building,
					sate_info_->file_mask_ndvi, file_street_dense_i, file_street_pose_i, file_street_gps);
				nogps_register.Run();
			}

			// step3: bundle
			if (!std::experimental::filesystem::exists(fold_align_i + "\\cam_pts1.txt")) {
				std::string file_street_bundle_i = fold_sfm_i + "\\street_bundle.txt";	
				std::string file_tfw = sate_info_->file_dsm.substr(0, sate_info_->file_dsm.size() - 4) + ".tfw";

				std::cout << "ConvertCamsPts" << std::endl;
				ConvertCamsPts(fold_street_rgb, file_street_bundle_i, file_street_pose_i, fold_align_i);
				
				Convert2Geo(file_tfw);
				WriteCameraPointsOut(fold_align_i + "\\cam_pts1.txt");

				FullBundleAdjustment();

				WriteCameraPointsOut(fold_align_i + "\\cam_pts2.txt");

				Convert2Geo(file_tfw);

				WriteCameraPointsOut(fold_align_i + "\\cam_pts3.txt");

				std::cout << "SaveOut" << std::endl;
				SaveUndistortedImage(fold_align_i, fold_street_rgb);

				SaveforOpenMVSNVM(fold_align_i);

				SaveforMSP(fold_align_i);
			}
		}
	}

	void CrossviewAlignment::ConvertCamsPts(std::string fold_img, std::string file_street_bundle, 
		std::string file_street_pose, std::string fold_out)
	{
		// read in bundle results
		ReadinImages(fold_img);

		ReadTempResultIn(file_street_bundle);

		// read in street pose
		std::vector<int> imgname_registed;
		ReadinCamNames(file_street_pose, imgname_registed);

		// read in trans parameters
		std::vector<std::vector<double>> trans_cams;
		ReadinTrans(fold_out + "\\xyz_alignment.txt", trans_cams);

		// transform the cam
		for (int i = 0; i < cams_sfm_.size(); i++) {
			int id_cam = i;
			cams_sfm_[id_cam]->Transformation(Eigen::Vector3d(trans_cams[id_cam][1], trans_cams[id_cam][2], trans_cams[id_cam][3]),
				Eigen::Vector3d(trans_cams[id_cam][4], trans_cams[id_cam][5], trans_cams[id_cam][6]),
				trans_cams[id_cam][0]);
		}

		// transform the pts
		std::vector<int> is_pts_cvted(pts_sfm_.size(), 0);
		for (int j = 0; j < pts_sfm_.size(); j++) {
			if (pts_sfm_[j]->is_bad_estimated_) {
				continue;
			}

			int id_avg = 0;
			int count = 0;
			for (auto iter = pts_sfm_[j]->cams_.begin(); iter != pts_sfm_[j]->cams_.end(); ++iter) {
				int id_cam = iter->second->id_;
				id_avg += id_cam;
				count++;
			}
			id_avg /= count;
			if (count<2) {
				continue;
			}

			pts_sfm_[j]->Transformation(Eigen::Vector3d(trans_cams[id_avg][1], trans_cams[id_avg][2], trans_cams[id_avg][3]),
				Eigen::Vector3d(trans_cams[id_avg][4], trans_cams[id_avg][5], trans_cams[id_avg][6]),
				trans_cams[id_avg][0]);
			is_pts_cvted[j] = 1;
		}

		for (int i = 0; i < pts_sfm_.size(); i++) {
			if (!is_pts_cvted[i]) {
				pts_sfm_[i]->is_bad_estimated_ = 1;
			}
		}
	}

	void CrossviewAlignment::Convert2Geo(std::string file_tfw) {		
		// read in tfw
		std::ifstream iff(file_tfw);
		float zeros;
		double xs, dx, ys, dy;
		//iff >> xs >> dx >> zeros >> ys >> zeros >> dy;
		iff >> dx >> zeros >> zeros >> dy >> xs >> ys;
		iff.close();		

		double s = abs(dx);
		Eigen::Vector3d a(CV_PI, 0.0, 0.0);
		Eigen::Vector3d t(xs - offx_, ys - offy_, 0.0);
		//Euler2Rotation(CV_PI, 0.0, 0.0, R);

		// do convertion
		for (int i = 0; i < cams_sfm_.size(); i++) {
			cams_sfm_[i]->Transformation(a, t, s);
		}
		for (int i = 0; i < pts_sfm_.size(); i++) {
			//std::cout << i << std::endl;
			pts_sfm_[i]->Transformation(a, t, s);
		}
	}

	void CrossviewAlignment::IncrementalBundleAdjustment()
	{
		cams_pos_ori_.resize(cams_sfm_.size());
		for (int i = 0; i < cams_sfm_.size(); i++) {
			cams_pos_ori_[i] = Eigen::Vector3d(cams_sfm_[i]->data[0], cams_sfm_[i]->data[1], cams_sfm_[i]->data[2]);
		}

		int size_step = 2;
		int num_step = cams_sfm_.size() / size_step + 1;
		
		for (int i = 0; i < num_step; i++) {
			cam_model_->SetMutable(false);
			int id_e = std::min(i* size_step, int(cams_sfm_.size()));
			int id_s = std::max(0, id_e - 100);
			if (i % 10 == 0 || id_e == cams_sfm_.size()) {
				id_s = 0;
				cam_model_->SetMutable(true);
			}
			ImmutableCamsPoints();

			for (int j = id_s; j < id_e; j++) {
				cams_sfm_[j]->SetMutable(true);
				for (auto iter = cams_sfm_[j]->pts_.begin(); iter != cams_sfm_[j]->pts_.end(); ++iter) {
					iter->second->SetMutable(true);
				}
			}

			int count_cams = 0, count_pts = 0;
			for (int j = 0; j < cams_sfm_.size(); j++) {
				if (cams_sfm_[j]->is_mutable_) {
					count_cams++;
				}
			}
			for (int j = 0; j < pts_sfm_.size(); j++) {
				if (pts_sfm_[j]->is_mutable_) {
					count_pts++;
				}
			}

			std::cout << "iter: " << i << std::endl;
			std::cout << "adjust cams: " << count_cams << std::endl;
			std::cout << "adjust pts: " << count_pts << " out of " << pts_sfm_.size() << std::endl;

			objectsfm::BundleAdjuster bundler(cams_sfm_, cam_model_, pts_sfm_);
			bundle_options_.th_tolerance = 1e-5;
			bundle_options_.ceres_solver_type = 3;
			bundle_options_.max_num_iterations = 50;
			bundler.SetOptions(bundle_options_);
			//bundler.RunOptimizetionRegistration(false, cams_pos_ori_);
			bundler.RunOptimizetion(false, 2.0);
			bundler.UpdateParameters();
			bundler.ReleaseBundleData();
			if (cam_model_->is_mutable_) {
				RemovePointOutliers();
			}

			std::cout << "f " << cam_model_->f_ << std::endl;
		}
	}

	void CrossviewAlignment::PartialBundleAdjustment()
	{
		cam_model_->SetMutable(false);

		//
		int size_step = 200;
		int num_step = cams_sfm_.size() / size_step + 0.5;
		size_step = cams_sfm_.size() / num_step;
		for (int i = 0; i < num_step; i++)
		{
			std::cout << "PartialBundleAdjustment " << i << std::endl;
			ImmutableCamsPoints();

			int ids_cam = i * size_step;
			int ide_cam = std::min((i + 1) * size_step, int(cams_sfm_.size()));

			for (int j = ids_cam; j < ide_cam; j++) {
				cams_sfm_[j]->SetMutable(true);
				for (auto iter = cams_sfm_[j]->pts_.begin(); iter != cams_sfm_[j]->pts_.end(); ++iter) {
					iter->second->SetMutable(true);
				}
			}

			int count_cams = 0, count_pts = 0;
			for (int j = 0; j < cams_sfm_.size(); j++) {
				if (cams_sfm_[j]->is_mutable_) {
					count_cams++;
				}
			}
			for (int j = 0; j < pts_sfm_.size(); j++) {
				if (pts_sfm_[j]->is_mutable_) {
					count_pts++;
				}
			}

			std::cout << "iter: " << i << std::endl;
			std::cout << "adjust cams: " << count_cams << std::endl;
			std::cout << "adjust pts: " << count_pts << " out of " << pts_sfm_.size() << std::endl;

			objectsfm::BundleAdjuster bundler(cams_sfm_, cam_model_, pts_sfm_);
			bundle_options_.th_tolerance = 1e-4;
			bundle_options_.ceres_solver_type = 3;
			bundle_options_.max_num_iterations = 50;
			bundler.SetOptions(bundle_options_);
			bundler.RunOptimizetion(false, 2.0);
			bundler.UpdateParameters();
			bundler.ReleaseBundleData();

			WriteCameraPointsOut("F:\\cam_pts" + std::to_string(i) + ".txt");

			//RemovePointOutliers();
		}
	}

	void CrossviewAlignment::FullBundleAdjustment()
	{
		int iter = 1;
		for (int i = 0; i < iter; i++) {
			cam_model_->SetMutable(false);

			// mutable all the cameras and points
			//MutableCamsPoints();
			float count_cams = 0;
			for (int j = id_cams_; j < id_came_; j++) {
				int count_obs = 0;
				for (auto iter = cams_sfm_[j]->pts_.begin(); iter != cams_sfm_[j]->pts_.end(); iter++) {
					if (!iter->second->is_bad_estimated_) {
						count_obs++;
					}
				}
				cams_sfm_[j]->SetWeigth(1000.0 / count_obs);
				cams_sfm_[j]->SetMutable(true);
				count_cams++;
			}			

			int count_pts = 0;
			for (size_t j = 0; j < pts_sfm_.size(); j++) {
				if (pts_sfm_[j]->is_bad_estimated_) {
					pts_sfm_[j]->is_mutable_ = false;
				}
				else {
					pts_sfm_[j]->is_mutable_ = true;
					count_pts++;
				}
			}

			std::cout << "adjust cams: " << count_cams << std::endl;
			std::cout << "adjust pts: " << count_pts << " out of " << pts_sfm_.size() << std::endl;

			// bundle adjustment 
			objectsfm::BundleAdjuster bundler(cams_sfm_, cam_model_, pts_sfm_);
			bundle_options_.th_tolerance = 1e-6;
			bundle_options_.ceres_solver_type = 3;
			bundle_options_.max_num_iterations = 30;
			bundler.SetOptions(bundle_options_);
			bundler.RunOptimizetion(false, 1.0);
			//bundler.RunOptimizetionRegistration(2.0, cams_pos_ori_);
			bundler.UpdateParameters();
			bundler.ReleaseBundleData();

			RemovePointOutliers();
		}
		
	}

	void CrossviewAlignment::ImmutableCamsPoints()
	{
		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			cams_sfm_[i]->SetMutable(false);

			std::map<size_t, Point3D*>::iterator iter = cams_sfm_[i]->pts_.begin();
			while (iter != cams_sfm_[i]->pts_.end()) {
				iter->second->SetMutable(false);
				iter++;
			}
		}
	}

	void CrossviewAlignment::MutableCamsPoints()
	{
		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			cams_sfm_[i]->SetMutable(true);

			std::map<size_t, Point3D*>::iterator iter = cams_sfm_[i]->pts_.begin();
			while (iter != cams_sfm_[i]->pts_.end()) {
				iter->second->SetMutable(true);
				iter++;
			}
		}
	}

	void CrossviewAlignment::RemovePointOutliers()
	{
		int count_outliers = 0;
		int count_outliers_new_add = 0;
		int count_new_add = 0;
		for (size_t i = 0; i < pts_sfm_.size(); i++) {
			if (!pts_sfm_[i]->is_mutable_) {
				continue;
			}
			count_new_add++;
			pts_sfm_[i]->is_bad_estimated_ = false;
			pts_sfm_[i]->Reprojection();
			if (std::sqrt(pts_sfm_[i]->mse_) > th_mse_outliers_) {
				pts_sfm_[i]->is_bad_estimated_ = true;
				count_outliers++;
			}
			//pts_sfm_[i]->is_new_added_ = false;
		}

		std::cout << "----------RemovePointOutliers: " << count_outliers << " of " << count_new_add << std::endl;
	}

	void CrossviewAlignment::AccuracySateBoundary()
	{
	}

	void CrossviewAlignment::AccuracyLidar(std::string file_lidar, std::string fold_street)
	{
		// read in lidar points
		std::vector<cv::Point3f> pts_lidar;
		std::ifstream iff(file_lidar);
		float x, y, z, idx;
		while (!iff.eof()) {
			iff >> x >> y >> z >> idx;
			pts_lidar.push_back(cv::Point3f(x, y, z));
		}
		iff.close();

		// read in street points	
		std::string file_streetpts = fold_street + "\\street_pts.txt";
		if (!std::experimental::filesystem::exists(file_streetpts)) {
			std::ofstream ofs(file_streetpts);
			if (!ofs.is_open()) {
				return;
			}
			ofs << std::fixed << std::setprecision(8);

			int dilation_step = 10;
			for (int i = 0; i < 100; i++) {
				std::string file_ply = fold_street + "\\openmvs" + std::to_string(i) + "\\dense.ply";
				if (!std::experimental::filesystem::exists(file_ply)) {
					continue;
				}
				std::cout << file_ply << std::endl;
				happly::PLYData ply_reader(file_ply);
				std::vector<float> px = ply_reader.getElement("vertex").getProperty<float>("x");
				std::vector<float> py = ply_reader.getElement("vertex").getProperty<float>("y");
				std::vector<float> pz = ply_reader.getElement("vertex").getProperty<float>("z");
				//std::vector<uchar> pr = ply_reader.getElement("vertex").getProperty<uchar>("red");
				//std::vector<uchar> pg = ply_reader.getElement("vertex").getProperty<uchar>("green");
				//std::vector<uchar> pb = ply_reader.getElement("vertex").getProperty<uchar>("blue");
				for (int m = 0; m < px.size(); m += dilation_step) {
					ofs << px[m] << " " << py[m] << " " << pz[m] << std::endl;
				}
			}
			ofs.close();
		}

		PointCloud3d<double> street_pts;
		std::ifstream ifs(file_streetpts);
		x, y, z;
		int count = 0;
		while (!ifs.eof()) {
			count++;
			ifs >> x >> y >> z;
			if (count % 2 == 0) {
				street_pts.pts.push_back(PointCloud3d<double>::PtData(x, y, z));
			}			
		}

		// street building detection
		PCAFunctions pcaer;
		int k = 30;
		std::vector<PCAInfo> pcaInfos;
		double scale, magnitd = 0.0;
		pcaer.Ori_PCA(street_pts, k, pcaInfos, scale, magnitd);

		std::vector<int> is_building_init(pcaInfos.size(), 0);
		double th_dev = cos(75.0 / 180.0*CV_PI);
		double vx = 0.0, vy = 0.0, vz = 1.0;
		for (size_t i = 0; i < pcaInfos.size(); i++) {
			double dev = vx * pcaInfos[i].normal(0) + vy * pcaInfos[i].normal(1) + vz * pcaInfos[i].normal(2);
			if (abs(dev) < th_dev) {
				is_building_init[i] = 1;
			}
		}

		std::vector<cv::Point3f> street_buildings;
		for (size_t i = 0; i < pcaInfos.size(); i++) {
			int count = 0;
			for (size_t j = 0; j < pcaInfos[i].idxAll.size(); j++) {
				int id = pcaInfos[i].idxAll[j];
				count += is_building_init[id];
			}
			if (count > k*0.75) {
				street_buildings.push_back(cv::Point3f(street_pts.pts[i].x, street_pts.pts[i].y, street_pts.pts[i].z));
			}
		}
		
		// quantitative assessment
		// kd tree for Lidar
		PointCloud3d<double> cloud;
		for (int i = 0; i < pts_lidar.size(); i++) {
			cloud.pts.push_back(PointCloud3d<double>::PtData(pts_lidar[i].x, pts_lidar[i].y, pts_lidar[i].z));
		}
		typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud3d<double> >, PointCloud3d<double>, 3/*dim*/ > P3D_kd_tree_t;
		P3D_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		index.buildIndex();

		// kd tree for dense
		PointCloud3d<double> cloud_dense;
		for (int i = 0; i < street_buildings.size(); i++) {
			cloud_dense.pts.push_back(PointCloud3d<double>::PtData(street_buildings[i].x, street_buildings[i].y, street_buildings[i].z));
		}
		P3D_kd_tree_t index_dense(3 /*dim*/, cloud_dense, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		index_dense.buildIndex();

		// closest dis for lidar
		std::vector<double> dis_min(pts_lidar.size());
		for (size_t i = 0; i < pts_lidar.size(); i++) {
			double *query_pt = new double[3];
			query_pt[0] = pts_lidar[i].x;
			query_pt[1] = pts_lidar[i].y;
			query_pt[2] = pts_lidar[i].z;

			double dis_temp = 0.0;
			size_t idx_temp = 0;
			nanoflann::KNNResultSet<double> result_set(1);
			result_set.init(&idx_temp, &dis_temp);
			index_dense.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));
			dis_min[i] = dis_temp;
		}

		// closest dis for dense
		std::vector<cv::Vec4f> inliers;
		for (int i = 0; i < street_buildings.size(); i++) {
			double *query_pt = new double[3];
			query_pt[0] = street_buildings[i].x;
			query_pt[1] = street_buildings[i].y;
			query_pt[2] = street_buildings[i].z;

			double dis_temp = 0.0;
			size_t idx_temp = 0;
			nanoflann::KNNResultSet<double> result_set(1);
			result_set.init(&idx_temp, &dis_temp);
			index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));
			if (dis_temp < dis_min[idx_temp] + 1.0) {
				inliers.push_back(cv::Vec4f(street_buildings[i].x, street_buildings[i].y, street_buildings[i].z, dis_temp));
			}			
		}

		// write out
		std::string file_buildings = fold_street + "\\inliers.txt";
		std::ofstream ofs(file_buildings);
		if (!ofs.is_open()) {
			return;
		}
		ofs << std::fixed << std::setprecision(8);
		double avg = 0.0;
		for (size_t i = 0; i < inliers.size(); i++) {
			avg += inliers[i].val[3];
			ofs << inliers[i].val[0] << " " << inliers[i].val[1] << " " << inliers[i].val[2] << " " << 20*inliers[i].val[3] << std::endl;
		}
		avg /= inliers.size();
		std::cout << avg << std::endl;
		ofs.close();
	}

	void CrossviewAlignment::Accuracy2D(std::string file_src, std::string file_dst, std::string file_out)
	{
		// read in lidar points
		std::vector<cv::Point3f> pts_src;
		PointCloud3d<double> src2d;
		std::ifstream iff(file_src);
		float x, y, z, idx;
		int r, g, b;
		while (!iff.eof()) {
			iff >> x >> y >> z >> r >> g >> b;
			pts_src.push_back(cv::Point3f(x, y, z));
			src2d.pts.push_back(PointCloud3d<double>::PtData(x, y, z));
		}
		iff.close();
		typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud3d<double> >, PointCloud3d<double>, 3/*dim*/ > P3D_kd_tree_t;
		P3D_kd_tree_t index_src(3 /*dim*/, src2d, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		index_src.buildIndex();

		// read in dst points
		std::vector<cv::Point3f> pts_dst;
		PointCloud3d<double> dst2d;
		iff = std::ifstream(file_dst);
		while (!iff.eof()) {
			iff >> x >> y >> z >> r >> g >> b;
			pts_dst.push_back(cv::Point3f(x, y, z));
			dst2d.pts.push_back(PointCloud3d<double>::PtData(x, y, z));
		}
		iff.close();
		P3D_kd_tree_t index_dst(3 /*dim*/, dst2d, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		index_dst.buildIndex();

		// closest dis for dst
		std::vector<double> min_dst(pts_dst.size());
		for (size_t i = 0; i < pts_dst.size(); i++) {
			double *query_pt = new double[3];
			query_pt[0] = pts_dst[i].x;
			query_pt[1] = pts_dst[i].y;
			query_pt[2] = pts_dst[i].z;

			double dis_temp = 0.0;
			size_t idx_temp = 0;
			nanoflann::KNNResultSet<double> result_set(1);
			result_set.init(&idx_temp, &dis_temp);
			index_src.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));
			min_dst[i] = dis_temp;
		}

		// closest dis for dense
		std::vector<cv::Vec4f> inliers;
		for (int i = 0; i < pts_src.size(); i++) {
			double *query_pt = new double[3];
			query_pt[0] = pts_src[i].x;
			query_pt[1] = pts_src[i].y;
			query_pt[2] = pts_src[i].z;

			double dis_temp = 0.0;
			size_t idx_temp = 0;
			nanoflann::KNNResultSet<double> result_set(1);
			result_set.init(&idx_temp, &dis_temp);
			index_dst.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));
			if (dis_temp < min_dst[idx_temp] + 100.0 && dis_temp < 200) {
			//if (dis_temp < min_dst[idx_temp] + 1.0 && dis_temp < 20) {
				inliers.push_back(cv::Vec4f(pts_src[i].x, pts_src[i].y, pts_src[i].z, dis_temp/2.0));
			}
		}

		// write out
		std::string file_buildings = file_out;
		std::ofstream ofs(file_buildings);
		if (!ofs.is_open()) {
			return;
		}
		ofs << std::fixed << std::setprecision(8);
		double avg = 0.0;
		for (size_t i = 0; i < inliers.size(); i++) {
			avg += inliers[i].val[3];
			ofs << inliers[i].val[0] << " " << inliers[i].val[1] << " " << inliers[i].val[2] << " " << inliers[i].val[3] << std::endl;
		}
		avg /= inliers.size();

		double mse = 0.0;
		for (size_t i = 0; i < inliers.size(); i++) {
			mse += std::pow(inliers[i].val[3] - avg, 2);
		}
		mse /= inliers.size();
		mse = sqrt(mse);

		std::cout << avg << " " << mse << std::endl;
		ofs.close();
	}

	void CrossviewAlignment::AccuracyMatching(std::string fold_test)
	{

	}

	void CrossviewAlignment::Euler2Rotation(double rx, double ry, double rz, Eigen::Matrix3d &R)
	{
		Eigen::Matrix3d Rx, Ry, Rz;
		Rx << 1.0, 0, 0,
			0, std::cos(rx), -std::sin(rx),
			0, std::sin(rx), std::cos(rx);

		Ry << std::cos(ry), 0, std::sin(ry),
			0, 1.0, 0,
			-std::sin(ry), 0, std::cos(ry);

		Rz << std::cos(rz), -std::sin(rz), 0,
			std::sin(rz), std::cos(rz), 0,
			0, 0, 1.0;

		R = Rz * Ry * Rx;
	}

	void CrossviewAlignment::Rotation2Euler(Eigen::Matrix3d R, double & rx, double & ry, double & rz) {
		//rx = std::atan2(-R(2, 1), R(2, 2));
		//ry = std::asin(R(2, 0));
		//rz = std::atan2(-R(1, 0), R(0, 0));

		rx = std::atan2(R(2, 1), R(2, 2));
		ry = std::asin(-R(2, 0));
		rz = std::atan2(R(1, 0), R(0, 0));

		if (rx != rx) { rx = 0.0; }
		if (ry != ry) { ry = 0.0; }
		if (rz != rz) { rz = 0.0; }
	}


	void CrossviewAlignment::Pts2Euler(std::vector<Eigen::Vector3d> pts_src, std::vector<Eigen::Vector3d> pts_dst,
		double &rx, double &ry, double &rz)
	{
		Eigen::Matrix3d cov_m;
		cov_m.setZero();
		double s_mv = 0.0;
		double d_mv = 0.0;

		for (int i = 0; i < pts_src.size(); i++) {
			Eigen::Vector3d ds = pts_src[i];
			Eigen::Vector3d dd = pts_dst[i];

			Eigen::MatrixXd dsTds = ds.transpose() * ds;
			Eigen::MatrixXd ddTdd = dd.transpose() * dd;

			s_mv += dsTds.data()[0];
			d_mv += ddTdd.data()[0];
			cov_m += ds * dd.transpose();
		}

		d_mv /= pts_src.size();
		s_mv /= pts_src.size();
		cov_m /= pts_src.size();

		// use SVD to compute rotation and then the translation.
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_m, Eigen::ComputeFullU | Eigen::ComputeFullV);

		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d Z = Eigen::Matrix3d::Identity();
		Eigen::Matrix3d VUt = V * (U.transpose());
		Z.data()[8] = VUt.determinant();

		Eigen::Matrix3d rotation = V * Z*(U.transpose());

		//
		Rotation2Euler(rotation, rx, ry, rz);
	}

	void CrossviewAlignment::ReadinImages(std::string fold)
	{
		std::vector<cv::String> image_paths_temp;
		cv::glob(fold + "/*.jpg", image_paths_temp, false);

		std::vector<int> img_names;
		for (size_t j = 0; j < image_paths_temp.size(); j++) {
			std::string path_i = std::string(image_paths_temp[j].c_str());
			int loc1 = path_i.find_last_of("\\") + 1;
			int loc2 = path_i.find_last_of(".");
			std::string name_i = path_i.substr(loc1, loc2 - loc1);
			img_names.push_back(std::stoi(name_i));
		}
		std::sort(img_names.begin(), img_names.end(), [](const int &lhs, const int &rhs) { return lhs < rhs; });

		// 
		cams_name_.resize(img_names.size());
		for (size_t i = 0; i < img_names.size(); i++) {
			cams_name_[i] = std::to_string(img_names[i]);
		}
	}

	void CrossviewAlignment::ReadinCamNames(std::string file, std::vector<int> &cam_names)
	{
		std::string file_pose = file;
		std::ifstream iff(file_pose);
		std::string temp;
		std::getline(iff, temp);
		std::getline(iff, temp);
		std::string name;
		float x, y, z, rx, ry, rz;
		std::vector<cv::Point3d> cams;
		while (!iff.eof()) {
			iff >> name >> x >> y >> z >> rx >> ry >> rz;
			name = name.substr(0, name.size() - 4);
			cam_names.push_back(std::stoi(name));
			cams.push_back(cv::Point3d(x, y, z));
		}
	}

	void CrossviewAlignment::WriteTempResultOut(std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}
		ofs.precision(20);

		// write out cam-models
		ofs << 1 << std::endl;
		ofs << 1 << " " << id_img_cur_ << std::endl;

		ofs << cam_model_->id_ << std::endl;
		ofs << cam_model_->cam_maker_ << std::endl;
		ofs << cam_model_->cam_model_ << std::endl;
		ofs << cam_model_->w_ << " " << cam_model_->h_ << std::endl;
		ofs << cam_model_->f_mm_
			<< " " << cam_model_->f_
			<< " " << cam_model_->f_hyp_
			<< " " << cam_model_->w_ / 2
			<< " " << cam_model_->h_ / 2
			<< " " << cam_model_->k1_
			<< " " << cam_model_->k2_
			<< " " << cam_model_->data[0]
			<< " " << cam_model_->data[1]
			<< " " << cam_model_->data[2]
			<< std::endl;

		// write out states
		ofs << 1 << std::endl;
		ofs << 1 << std::endl;

		// velocity
		ofs << 0 << " " << 0 << " " << 0 << " ";
		ofs << 0 << " " << 0 << " " << 0 << " ";
		ofs << 0 << " " << 0 << " " << 0 << " ";
		ofs << 0 << " " << 0 << " " << 0 << std::endl;

		// cam cur
		ofs << cams_sfm_[0]->id_ << std::endl;
		ofs << cams_sfm_[0]->id_img_ << std::endl;
		ofs << cams_sfm_[0]->cam_model_->id_ << std::endl;
		ofs << cams_sfm_[0]->is_mutable_ << std::endl;

		ofs << " " << cams_sfm_[0]->data[0]
			<< " " << cams_sfm_[0]->data[1]
			<< " " << cams_sfm_[0]->data[2]
			<< " " << cams_sfm_[0]->data[3]
			<< " " << cams_sfm_[0]->data[4]
			<< " " << cams_sfm_[0]->data[5]
			<< std::endl;

		// cam last
		ofs << cams_sfm_[0]->id_ << std::endl;
		ofs << cams_sfm_[0]->id_img_ << std::endl;
		ofs << cams_sfm_[0]->cam_model_->id_ << std::endl;
		ofs << cams_sfm_[0]->is_mutable_ << std::endl;

		ofs << " " << cams_sfm_[0]->data[0]
			<< " " << cams_sfm_[0]->data[1]
			<< " " << cams_sfm_[0]->data[2]
			<< " " << cams_sfm_[0]->data[3]
			<< " " << cams_sfm_[0]->data[4]
			<< " " << cams_sfm_[0]->data[5]
			<< std::endl;

		// write out pts_cur_
		ofs << 0 << std::endl;		

		// write out pts_last_
		ofs << 0 << std::endl;		

		// write out key cams
		ofs << cams_sfm_.size() << std::endl;
		for (size_t i = 0; i < cams_sfm_.size(); i++)
		{
			ofs << cams_sfm_[i]->id_ << std::endl;
			ofs << cams_sfm_[i]->id_img_ << std::endl;
			ofs << cams_sfm_[i]->cam_model_->id_ << std::endl;
			ofs << cams_sfm_[i]->is_mutable_ << std::endl;

			ofs << " " << cams_sfm_[i]->data[0]
				<< " " << cams_sfm_[i]->data[1]
				<< " " << cams_sfm_[i]->data[2]
				<< " " << cams_sfm_[i]->data[3]
				<< " " << cams_sfm_[i]->data[4]
				<< " " << cams_sfm_[i]->data[5]
				<< std::endl;

			ofs << cams_sfm_[i]->pts_.size() << std::endl;
			for (auto iter = cams_sfm_[i]->pts_.begin(); iter != cams_sfm_[i]->pts_.end(); ++iter)
			{
				ofs << iter->first << " " << iter->second->id_ << " ";
			}
			ofs << std::endl;
		}

		// write out key points
		ofs << pts_sfm_.size() << std::endl;
		for (size_t i = 0; i < pts_sfm_.size(); i++)
		{
			ofs << pts_sfm_[i]->id_ << std::endl;

			ofs << pts_sfm_[i]->is_mutable_
				<< " " << pts_sfm_[i]->is_bad_estimated_
				<< " " << pts_sfm_[i]->is_new_added_
				<< std::endl;

			ofs << pts_sfm_[i]->data[0]
				<< " " << pts_sfm_[i]->data[1]
				<< " " << pts_sfm_[i]->data[2]
				<< std::endl;

			ofs << pts_sfm_[i]->cams_.size() << std::endl;
			for (auto iter = pts_sfm_[i]->cams_.begin(); iter != pts_sfm_[i]->cams_.end(); ++iter)
			{
				ofs << iter->first << " " << iter->second->id_img_ << " ";
			}
			ofs << std::endl;

			ofs << pts_sfm_[i]->pts2d_.size() << std::endl;
			for (auto iter = pts_sfm_[i]->pts2d_.begin(); iter != pts_sfm_[i]->pts2d_.end(); ++iter)
			{
				ofs << iter->first << " " << iter->second(0) << " " << iter->second(1) << " ";
			}
			ofs << std::endl;

			ofs << pts_sfm_[i]->key_new_obs_ << std::endl;

			ofs << pts_sfm_[i]->mse_ << std::endl;
		}

		// write out cam names
		for (size_t i = 0; i < cams_name_.size(); i++)
		{
			ofs << cams_name_[i] << std::endl;
		}

		ofs.close();
	}

	void CrossviewAlignment::ReadTempResultIn(std::string path)
	{
		std::ifstream ifs(path);
		if (!ifs.is_open())
		{
			return;
		}

		// read cam-models
		float a, b;
		ifs >> a;
		ifs >> a >> id_img_cur_;

		cam_model_ = new CameraModel;
		ifs >> cam_model_->id_;
		std::string temp;
		std::getline(ifs, temp);
		std::getline(ifs, cam_model_->cam_maker_);
		std::getline(ifs, cam_model_->cam_model_);

		ifs >> cam_model_->w_ >> cam_model_->h_;
		ifs >> cam_model_->f_mm_
			>> cam_model_->f_
			>> cam_model_->f_hyp_
			>> cam_model_->px_
			>> cam_model_->py_
			>> cam_model_->k1_
			>> cam_model_->k2_
			>> cam_model_->data[0]
			>> cam_model_->data[1]
			>> cam_model_->data[2];

		// read slam
		ifs >> a;
		ifs >> b;

		// velocity
		ifs >> a >> a >> a;
		ifs >> a >> a >> a;
		ifs >> a >> a >> a;
		ifs >> a >> a >> a;

		// cam cur
		Camera* cam_cur_ = new Camera();
		ifs >> cam_cur_->id_;
		ifs >> cam_cur_->id_img_;
		int cam_model_id = 0;
		ifs >> cam_model_id;
		ifs >> cam_cur_->is_mutable_;
		ifs >> cam_cur_->data[0]
			>> cam_cur_->data[1]
			>> cam_cur_->data[2]
			>> cam_cur_->data[3]
			>> cam_cur_->data[4]
			>> cam_cur_->data[5];
		cam_cur_->AssociateCamereModel(cam_model_);
		cam_cur_->UpdatePoseFromData();

		// cam last
		Camera* cam_last_ = new Camera();
		ifs >> cam_last_->id_;
		ifs >> cam_last_->id_img_;
		ifs >> cam_model_id;
		ifs >> cam_last_->is_mutable_;
		ifs >> cam_last_->data[0]
			>> cam_last_->data[1]
			>> cam_last_->data[2]
			>> cam_last_->data[3]
			>> cam_last_->data[4]
			>> cam_last_->data[5];
		cam_last_->AssociateCamereModel(cam_model_);
		cam_last_->UpdatePoseFromData();

		// read pts cur
		int pts_cur_size = 0;
		ifs >> pts_cur_size;
		for (size_t i = 0; i < pts_cur_size; i++)
		{
			Point3D* pt_temp = new Point3D();

			ifs >> pt_temp->id_;
			ifs >> pt_temp->is_mutable_
				>> pt_temp->is_bad_estimated_
				>> pt_temp->is_new_added_;

			ifs >> pt_temp->data[0]
				>> pt_temp->data[1]
				>> pt_temp->data[2];

			size_t key;
			double x, y;
			ifs >> key >> x >> y;
			pt_temp->pts2d_.insert(std::pair<size_t, Eigen::Vector2d>(key, Eigen::Vector2d(x, y)));
			pt_temp->cams_.insert(std::pair<size_t, Camera*>(key, cam_cur_));
			cam_cur_->AddPoints(pt_temp, key);

			ifs >> pt_temp->key_new_obs_;
			ifs >> pt_temp->mse_;
		}

		// read pts last	
		int pts_last_size = 0;
		ifs >> pts_last_size;		
		for (size_t i = 0; i < pts_last_size; i++)
		{
			Point3D* pt_temp = new Point3D();

			ifs >> pt_temp->id_;
			ifs >> pt_temp->is_mutable_
				>> pt_temp->is_bad_estimated_
				>> pt_temp->is_new_added_;

			ifs >> pt_temp->data[0]
				>> pt_temp->data[1]
				>> pt_temp->data[2];

			size_t key;
			double x, y;
			ifs >> key >> x >> y;
			pt_temp->pts2d_.insert(std::pair<size_t, Eigen::Vector2d>(key, Eigen::Vector2d(x, y)));
			pt_temp->cams_.insert(std::pair<size_t, Camera*>(key, cam_last_));
			cam_last_->AddPoints(pt_temp, key);

			ifs >> pt_temp->key_new_obs_;
			ifs >> pt_temp->mse_;
		}


		// read cams
		int cams_size = 0;
		ifs >> cams_size;
		cams_sfm_.resize(cams_size);
		std::vector<int> cam_model_info(cams_size);
		std::vector<std::vector<std::pair<size_t, size_t>>> cams_pts_info(cams_size);
		for (size_t i = 0; i < cams_sfm_.size(); i++)
		{
			cams_sfm_[i] = new Camera;
			ifs >> cams_sfm_[i]->id_;
			ifs >> cams_sfm_[i]->id_img_;

			int cam_model_id = 0;
			ifs >> cam_model_info[i];

			ifs >> cams_sfm_[i]->is_mutable_;

			ifs >> cams_sfm_[i]->data[0]
				>> cams_sfm_[i]->data[1]
				>> cams_sfm_[i]->data[2]
				>> cams_sfm_[i]->data[3]
				>> cams_sfm_[i]->data[4]
				>> cams_sfm_[i]->data[5];

			int num_cam_pts = 0;
			ifs >> num_cam_pts;
			size_t key, id;
			for (size_t j = 0; j<num_cam_pts; ++j)
			{
				ifs >> key >> id;
				cams_pts_info[i].push_back(std::pair<size_t, size_t>(key, id));
			}
		}

		// read points
		int pts_size;
		ifs >> pts_size;
		pts_sfm_.resize(pts_size);
		std::vector<std::vector<std::pair<size_t, size_t>>> pts_cams_info(pts_size);
		for (size_t i = 0; i < pts_sfm_.size(); i++)
		{
			pts_sfm_[i] = new Point3D();

			ifs >> pts_sfm_[i]->id_;

			ifs >> pts_sfm_[i]->is_mutable_
				>> pts_sfm_[i]->is_bad_estimated_
				>> pts_sfm_[i]->is_new_added_;

			ifs >> pts_sfm_[i]->data[0]
				>> pts_sfm_[i]->data[1]
				>> pts_sfm_[i]->data[2];

			int num_pts_cams;
			ifs >> num_pts_cams;
			size_t key, id;
			for (size_t j = 0; j<num_pts_cams; ++j)
			{
				ifs >> key >> id;
				pts_cams_info[i].push_back(std::pair<size_t, size_t>(key, id));
			}

			int num_pts_obs;
			ifs >> num_pts_obs;
			double x, y;
			for (size_t j = 0; j<num_pts_cams; ++j)
			{
				ifs >> key >> x >> y;
				pts_sfm_[i]->pts2d_.insert(std::pair<size_t, Eigen::Vector2d>(key, Eigen::Vector2d(x, y)));
			}

			ifs >> pts_sfm_[i]->key_new_obs_;
			ifs >> pts_sfm_[i]->mse_;
		}
		ifs.close();

		// association
		std::map<int, Camera*> cams_map_;
		for (size_t i = 0; i < cams_sfm_.size(); i++)
		{
			cams_map_.insert(std::pair<int, Camera*>(cams_sfm_[i]->id_img_, cams_sfm_[i]));
		}

		std::map<int, Point3D*> pts_map_;
		for (size_t i = 0; i < pts_sfm_.size(); i++)
		{
			pts_map_.insert(std::pair<int, Point3D*>(pts_sfm_[i]->id_, pts_sfm_[i]));
		}

		// cam
		for (size_t i = 0; i < cams_sfm_.size(); i++)
		{
			int id_cam_model = cam_model_info[i];
			cams_sfm_[i]->AssociateCamereModel(cam_model_);
			cams_sfm_[i]->UpdatePoseFromData();

			for (size_t j = 0; j < cams_pts_info[i].size(); j++)
			{
				size_t idx_local = cams_pts_info[i][j].first;
				size_t id_pts3d = cams_pts_info[i][j].second;
				cams_sfm_[i]->AddPoints(pts_map_[id_pts3d], idx_local); // to do
			}
		}

		// points
		for (size_t i = 0; i < pts_sfm_.size(); i++)
		{
			for (size_t j = 0; j < pts_cams_info[i].size(); j++)
			{
				size_t idx_local = pts_cams_info[i][j].first;
				size_t id_cam = pts_cams_info[i][j].second;
				pts_sfm_[i]->cams_.insert(std::pair<size_t, Camera*>(idx_local, cams_map_[id_cam]));
			}
		}
	}

	void CrossviewAlignment::ReadinTrans(std::string path, std::vector<std::vector<double>>& trans)
	{
		std::ifstream ff(path);
		int id;
		std::vector<double> trans_temp(7);
		while (!ff.eof()) {
			ff >> id;
			for (int i = 0; i < trans_temp.size(); i++) {
				ff >> trans_temp[i];
			}
			trans.push_back(trans_temp);
		}
	}

	void CrossviewAlignment::WriteCameraPointsOut(std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}
		ofs << std::fixed << std::setprecision(8);
		double scale = (cams_sfm_[0]->pos_ac_.c - cams_sfm_[1]->pos_ac_.c).norm() / 100;

		//float span = 1000;
		for (size_t i = 0; i < pts_sfm_.size(); i++) {
			if (pts_sfm_[i]->is_bad_estimated_) {
				continue;
			}
			//if (abs(pts_sfm_[i]->data[0]) > span || abs(pts_sfm_[i]->data[1]) > span || abs(pts_sfm_[i]->data[2]) > span) {
			//	continue;
			//}
			ofs << pts_sfm_[i]->data[0] << " " << pts_sfm_[i]->data[1] << " " << pts_sfm_[i]->data[2]
				<< " " << 0 << " " << 0 << " " << 0 << std::endl;
		}

		//double scale = 0.01;
		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			std::vector<Eigen::Vector3d> axis(3);
			axis[0] = cams_sfm_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(1, 0, 0);
			axis[1] = cams_sfm_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 1, 0);
			axis[2] = cams_sfm_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 0, 1);

			std::vector<Eigen::Vector3d> cam_pts;
			GenerateCamera3D(cams_sfm_[i]->pos_ac_.c, axis, cams_sfm_[i]->cam_model_->f_,
				cams_sfm_[i]->cam_model_->w_, cams_sfm_[i]->cam_model_->h_, scale, cam_pts);

			for (size_t j = 0; j < cam_pts.size(); j++)
			{
				//if (abs(cam_pts[j](0)) > span || abs(cam_pts[j](1)) > span || abs(cam_pts[j](2)) > span)
				//{
				//	continue;
				//}

				ofs << cam_pts[j](0) << " " << cam_pts[j](1) << " " << cam_pts[j](2)
					<< " " << 255 << " " << 0 << " " << 0 << std::endl;
			}
		}

		// trajectory
		for (size_t i = 0; i < cams_all_.size(); i++) {
			std::vector<Eigen::Vector3d> axis(3);
			axis[0] = cams_all_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(1, 0, 0);
			axis[1] = cams_all_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 1, 0);
			axis[2] = cams_all_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 0, 1);

			std::vector<Eigen::Vector3d> cam_pts;
			GenerateCamera3D(cams_all_[i]->pos_ac_.c, axis, cams_all_[i]->cam_model_->f_,
				cams_all_[i]->cam_model_->w_, cams_all_[i]->cam_model_->h_, scale, cam_pts);

			for (size_t j = 0; j < cam_pts.size(); j++)
			{
				ofs << cam_pts[j](0) << " " << cam_pts[j](1) << " " << cam_pts[j](2)
					<< " " << 255 << " " << 0 << " " << 0 << std::endl;
			}
		}

		ofs.close();
	}

	void CrossviewAlignment::SaveUndistortedImage(std::string fold, std::string fold_rgb)
	{
		// undistortion
		cv::Mat K(3, 3, CV_64FC1);
		K.at<double>(0, 0) = cams_sfm_[0]->cam_model_->f_, K.at<double>(0, 1) = 0.0, K.at<double>(0, 2) = cams_sfm_[0]->cam_model_->w_/2.0;
		K.at<double>(1, 0) = 0.0, K.at<double>(1, 1) = cams_sfm_[0]->cam_model_->f_, K.at<double>(1, 2) = cams_sfm_[0]->cam_model_->h_/2.0;
		K.at<double>(2, 0) = 0.0, K.at<double>(2, 1) = 0.0, K.at<double>(2, 2) = 1.0;

		cv::Mat dist(1, 5, CV_64FC1);
		dist.at<double>(0, 0) = cams_sfm_[0]->cam_model_->k1_;
		dist.at<double>(0, 1) = cams_sfm_[0]->cam_model_->k2_;
		dist.at<double>(0, 2) = 0.0;
		dist.at<double>(0, 3) = 0.0;
		dist.at<double>(0, 4) = 0.0;

		//		
		std::string fold_image = fold + "\\undistort_image";
		if (!std::experimental::filesystem::exists(fold_image)) {
			std::experimental::filesystem::create_directory(fold_image);
		}

		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			int id_img = cams_sfm_[i]->id_img_;		
			//std::cout << id_img << std::endl;

			// write out image
			std::string path_in = fold_rgb + "\\" + cams_name_[id_img] + ".jpg";
			if (!std::experimental::filesystem::exists(path_in)) {
				continue;
			}

			std::string path_out = fold_image + "\\" + cams_name_[id_img] + ".jpg";
			if (std::experimental::filesystem::exists(path_out)) {
				continue;
			}

			cv::Mat img = cv::imread(path_in);
			std::cout << path_in << std::endl;
			cv::Mat img_undistort;
			std::cout << K << std::endl;
			std::cout << dist << std::endl;
			cv::undistort(img, img_undistort, K, dist);
			cv::imwrite(path_out, img_undistort);
		}
	}

	void CrossviewAlignment::SaveforOpenMVSNVM(std::string fold)
	{
		int sample_step = 1;

		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_sfm_[i]->id_, i));
		}

		int n_seg = int(float(cams_sfm_.size()) / 500 + 0.5);
		int step = (cams_sfm_.size() - 1) / n_seg;
		for (size_t k = 0; k < n_seg; k++)
		{
			int cam_ids = k * step;
			int cam_ide = cam_ids + step;
			if (cam_ide > cams_sfm_.size()) {
				cam_ide = cams_sfm_.size();
			}

			std::string fold_openmvs = fold + "\\openmvs" + std::to_string(k);

			if (!std::experimental::filesystem::exists(fold_openmvs)) {
				std::experimental::filesystem::create_directory(fold_openmvs);
			}
			std::string fold_img = fold_openmvs + "\\image";
			if (!std::experimental::filesystem::exists(fold_img)) {
				std::experimental::filesystem::create_directory(fold_img);
			}
			std::string fold_rgb = fold + "\\undistort_image";

			// step0: find pts belonging to the clusters
			std::vector<int> pt_ids;
			for (size_t i = 0; i < pts_sfm_.size(); i++) {
				if (pts_sfm_[i]->is_bad_estimated_) {
					continue;
				}

				auto it = pts_sfm_[i]->cams_.begin();
				int count = 0;
				while (it != pts_sfm_[i]->cams_.end()) {
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide && (iter->second - cam_ids) % sample_step == 0) {
						count++;
					}
					it++;
				}
				if (count > 2) {
					pt_ids.push_back(i);
				}
			}

			// step1: save nvm
			std::string file_para = fold_openmvs + "\\sparse.nvm";
			std::ofstream ff(file_para);
			ff << std::fixed << std::setprecision(12);

			// cams
			int n_cams_temp = 0;
			for (size_t i = cam_ids; i < cam_ide; i++) {
				if ((i - cam_ids) % sample_step == 0) {
					n_cams_temp++;
				}
			}

			ff << "NVM_V3_R9T\n" << n_cams_temp << '\n';
			for (size_t i = cam_ids; i < cam_ide; i++) {
				if ((i - cam_ids) % sample_step != 0) {
					continue;
				}

				int id_img = cams_sfm_[i]->id_img_;
				std::string name = "image\\" + cams_name_[id_img] + ".jpg";

				ff << name << " " << cams_sfm_[0]->cam_model_->f_ << " ";
				ff << cams_sfm_[i]->pos_rt_.R(0, 0) << " " << cams_sfm_[i]->pos_rt_.R(0, 1) << " " << cams_sfm_[i]->pos_rt_.R(0, 2) << " "
				   << cams_sfm_[i]->pos_rt_.R(1, 0) << " " << cams_sfm_[i]->pos_rt_.R(1, 1) << " " << cams_sfm_[i]->pos_rt_.R(1, 2) << " "
				   << cams_sfm_[i]->pos_rt_.R(2, 0) << " " << cams_sfm_[i]->pos_rt_.R(2, 1) << " " << cams_sfm_[i]->pos_rt_.R(2, 2) << " ";
				ff << cams_sfm_[i]->pos_rt_.t(0) << " " << cams_sfm_[i]->pos_rt_.t(1) << " " << cams_sfm_[i]->pos_rt_.t(2) << " ";
				ff << 0.0 << " 0" << std::endl;
			}

			// pts
			ff << pt_ids.size() << std::endl;
			for (size_t i = 0; i < pt_ids.size(); i++) {
				int pt_id = pt_ids[i];
				ff << pts_sfm_[pt_id]->data[0] << " " << pts_sfm_[pt_id]->data[1] << " " << pts_sfm_[pt_id]->data[2] << " ";
				ff << 255 << " " << 255 << " " << 255 << " ";

				// obs
				int count_cams = 0;
				auto it = pts_sfm_[pt_id]->cams_.begin();
				while (it != pts_sfm_[pt_id]->cams_.end()) {
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide && (iter->second - cam_ids) % sample_step == 0) {
						count_cams++;
					}
					it++;
				}

				ff << count_cams << " ";
				auto it1 = pts_sfm_[pt_id]->pts2d_.begin();
				auto it2 = pts_sfm_[pt_id]->cams_.begin();
				while (it1 != pts_sfm_[pt_id]->pts2d_.end()) {
					int idx_pt = it1->first;
					int id_cam = it2->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide && (iter->second - cam_ids) % sample_step == 0) {
						float x = it1->second(0);
						float y = it1->second(1);
						ff << (iter->second - cam_ids) / sample_step << " " << 0 << " " << float(x) << " " << float(y) << " ";
					}
					it1++;  it2++;
				}
				ff << std::endl;
			}

			// images
			for (size_t i = cam_ids; i < cam_ide; i++) {
				if ((i - cam_ids) % sample_step != 0) {
					continue;
				}
				int id_img = cams_sfm_[i]->id_img_;
				std::string name = cams_name_[id_img] + ".jpg";				
				std::string file_img_in = fold_rgb + "\\" + name;
				std::string file_img_out = fold_img + "\\" + name;
				cv::Mat img = cv::imread(file_img_in);
				cv::imwrite(file_img_out, img);
			}

			ff.close();
		}
	}

	void CrossviewAlignment::OpenMVSNVMConvert(std::string fold)
	{

	}

	void CrossviewAlignment::SaveforOpenMVSBundle(std::string fold)
	{
		//int ncluster =1;
		int ncluster = cams_sfm_.size() / 500;
		if (ncluster == 0) {
			ncluster = 1;
		}
		int nstep = cams_sfm_.size() / ncluster;

		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_sfm_[i]->id_, i));
		}

		for (size_t k = 0; k < 1; k++)
		{
			int cam_ids = k * nstep;
			int cam_ide = (k + 1) * nstep;
			if (cam_ide > cams_sfm_.size()) {
				cam_ide = cams_sfm_.size();
			}

			std::string fold_openmvs = fold + "\\openmvs" + std::to_string(k);

			if (!std::experimental::filesystem::exists(fold_openmvs)) {
				std::experimental::filesystem::create_directory(fold_openmvs);
			}
			std::string fold_img = fold_openmvs + "\\image";
			if (!std::experimental::filesystem::exists(fold_img)) {
				std::experimental::filesystem::create_directory(fold_img);
			}			
			std::string fold_rgb = fold + "\\undistort_image";

			// step0: find pts belonging to the clusters
			std::vector<int> pt_ids;
			for (size_t i = 0; i < pts_sfm_.size(); i++)
			{
				if (pts_sfm_[i]->is_bad_estimated_) {
					continue;
				}

				auto it = pts_sfm_[i]->cams_.begin();
				while (it != pts_sfm_[i]->cams_.end())
				{
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						pt_ids.push_back(i);
						break;
					}
					it++;
				}
			}


			// step1: save bundle
			std::string file_para = fold_openmvs + "\\bundle.out";
			std::ofstream ff(file_para);
			ff << std::fixed << std::setprecision(8);

			// cams
			ff << "# Bundle file v0.3" << std::endl;
			ff << cam_ide - cam_ids << " " << pt_ids.size() << std::endl;
			for (size_t i = cam_ids; i < cam_ide; i++)
			{
				ff << cams_sfm_[0]->cam_model_->f_ << " " << cams_sfm_[0]->cam_model_->k1_ << " " << cams_sfm_[0]->cam_model_->k2_ << std::endl;
				ff << cams_sfm_[i]->pos_rt_.R(0, 0) << " " << cams_sfm_[i]->pos_rt_.R(0, 1) << " " << cams_sfm_[i]->pos_rt_.R(0, 2) << " "
					<< cams_sfm_[i]->pos_rt_.R(1, 0) << " " << cams_sfm_[i]->pos_rt_.R(1, 1) << " " << cams_sfm_[i]->pos_rt_.R(1, 2) << " "
					<< cams_sfm_[i]->pos_rt_.R(2, 0) << " " << cams_sfm_[i]->pos_rt_.R(2, 1) << " " << cams_sfm_[i]->pos_rt_.R(2, 2) << std::endl;
				ff << cams_sfm_[i]->pos_rt_.t(0) << " " << cams_sfm_[i]->pos_rt_.t(1) << " " << cams_sfm_[i]->pos_rt_.t(2) << std::endl;
			}

			// points
			for (size_t i = 0; i < pt_ids.size(); i++)
			{
				int pt_id = pt_ids[i];
				ff << pts_sfm_[pt_id]->data[0] << " " << pts_sfm_[pt_id]->data[1] << " " << pts_sfm_[pt_id]->data[2] << " ";
				ff << 255 << " " << 255 << " " << 255 << " ";
				int count_cams = 0;
				auto it = pts_sfm_[pt_id]->cams_.begin();
				while (it != pts_sfm_[pt_id]->cams_.end())
				{
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						count_cams++;
					}
					it++;
				}
				ff << count_cams << std::endl;

				auto it1 = pts_sfm_[pt_id]->pts2d_.begin();
				auto it2 = pts_sfm_[pt_id]->cams_.begin();
				while (it1 != pts_sfm_[pt_id]->pts2d_.end())
				{
					int idx_pt = it1->first;
					int id_cam = it2->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						int x = it1->second(0);
						int y = -it1->second(1);
						ff << iter->second - cam_ids << " " << idx_pt << " " << float(x) << " " << -float(y) << std::endl;
					}
					it1++;  it2++;
				}
			}
			ff.close();

			// step2: image list
			std::string file_list = fold_openmvs + "\\bundle-list.txt";
			std::ofstream ff2(file_list);
			for (size_t i = cam_ids; i < cam_ide; i++) {
				int id_img = cams_sfm_[i]->id_img_;
				std::string name = cams_name_[id_img] + ".jpg";
				ff2 << name << std::endl;
				std::string file_img_in = fold_rgb + "\\" + name;
				std::string file_img_out = fold_img + "\\" + name;
				//cv::Mat img = cv::imread(file_img_in);
				//cv::imwrite(file_img_out, img);
			}
			ff2.close();
		}
	}

	void CrossviewAlignment::SaveforCMVS(std::string fold)
	{
		int ncluster = cams_sfm_.size() / 500;
		if (ncluster == 0) {
			ncluster = 1;
		}
		int nstep = cams_sfm_.size() / ncluster;

		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_sfm_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_sfm_[i]->id_, i));
		}

		for (size_t k = 2; k < 3; k++)
		{
			int cam_ids = k * nstep;
			int cam_ide = (k + 1) * nstep;
			if (cam_ide > cams_sfm_.size()) {
				cam_ide = cams_sfm_.size();
			}

			std::string fold_cmvs = fold + "\\cmvs" + std::to_string(k);

			if (!std::experimental::filesystem::exists(fold_cmvs)) {
				std::experimental::filesystem::create_directory(fold_cmvs);
			}
			std::string fold_img = fold_cmvs + "\\visualize";
			if (!std::experimental::filesystem::exists(fold_img)) {
				std::experimental::filesystem::create_directory(fold_img);
			}
			std::string fold_txt = fold_cmvs + "\\txt";
			if (!std::experimental::filesystem::exists(fold_txt)) {
				std::experimental::filesystem::create_directory(fold_txt);
			}
			std::string fold_rgb = fold + "\\undistort_image";

			// step0: find pts belonging to the clusters
			std::vector<int> pt_ids;
			for (size_t i = 0; i < pts_sfm_.size(); i++)
			{
				if (pts_sfm_[i]->is_bad_estimated_) {
					continue;
				}

				auto it = pts_sfm_[i]->cams_.begin();
				while (it != pts_sfm_[i]->cams_.end())
				{
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						pt_ids.push_back(i);
						break;
					}
					it++;
				}
			}


			// step1: save bundle
			std::string file_para = fold_cmvs + "\\bundle.rd.out";

			//
			std::ofstream ff(file_para);
			ff << std::fixed << std::setprecision(8);

			// cams
			ff << "# Bundle file v0.3" << std::endl;
			ff << cam_ide - cam_ids << " " << pt_ids.size() << std::endl;
			for (size_t i = cam_ids; i < cam_ide; i++)
			{
				ff << cams_sfm_[0]->cam_model_->f_ << " " << cams_sfm_[0]->cam_model_->k1_ << " " << cams_sfm_[0]->cam_model_->k2_ << std::endl;
				ff << cams_sfm_[i]->pos_rt_.R(0, 0) << " " << cams_sfm_[i]->pos_rt_.R(0, 1) << " " << cams_sfm_[i]->pos_rt_.R(0, 2) << " "
					<< cams_sfm_[i]->pos_rt_.R(1, 0) << " " << cams_sfm_[i]->pos_rt_.R(1, 1) << " " << cams_sfm_[i]->pos_rt_.R(1, 2) << " "
					<< cams_sfm_[i]->pos_rt_.R(2, 0) << " " << cams_sfm_[i]->pos_rt_.R(2, 1) << " " << cams_sfm_[i]->pos_rt_.R(2, 2) << std::endl;
				ff << cams_sfm_[i]->pos_rt_.t(0) << " " << cams_sfm_[i]->pos_rt_.t(1) << " " << cams_sfm_[i]->pos_rt_.t(2) << std::endl;
			}

			// points
			for (size_t i = 0; i < pt_ids.size(); i++)
			{
				int pt_id = pt_ids[i];
				ff << pts_sfm_[pt_id]->data[0] << " " << pts_sfm_[pt_id]->data[1] << " " << pts_sfm_[pt_id]->data[2] << " ";
				ff << 255 << " " << 255 << " " << 255 << " ";
				int count_cams = 0;
				auto it = pts_sfm_[pt_id]->cams_.begin();
				while (it != pts_sfm_[pt_id]->cams_.end())
				{
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						count_cams++;
					}
					it++;
				}
				ff << count_cams << std::endl;

				auto it1 = pts_sfm_[pt_id]->pts2d_.begin();
				auto it2 = pts_sfm_[pt_id]->cams_.begin();
				while (it1 != pts_sfm_[pt_id]->pts2d_.end())
				{
					int idx_pt = it1->first;
					int id_cam = it2->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						int x = it1->second(0);
						int y = it1->second(1);
						ff << iter->second - cam_ids << " " << idx_pt << " " << float(x) << " " << float(y) << std::endl;
					}
					it1++;  it2++;
				}
			}
			ff.close();

			// step2: save txt and image
			for (size_t i = cam_ids; i < cam_ide; i++)
			{
				std::string name = std::to_string(i - cam_ids);
				name = std::string(8 - name.length(), '0') + name;

				// txt
				Eigen::Matrix3d R = cams_sfm_[i]->pos_rt_.R;
				Eigen::Vector3d t = cams_sfm_[i]->pos_rt_.t;
				cv::Mat M = (cv::Mat_<double>(3, 4) << R(0, 0), R(0, 1), R(0, 2), t(0),
					R(1, 0), R(1, 1), R(1, 2), t(1),
					R(2, 0), R(2, 1), R(2, 2), t(2));
				cv::Mat K = (cv::Mat_<double>(3, 3) << cams_sfm_[0]->cam_model_->f_, 0, cams_sfm_[0]->cam_model_->px_,
					0, cams_sfm_[0]->cam_model_->f_, cams_sfm_[0]->cam_model_->py_,
					0, 0, 1);
				cv::Mat P = K * M;

				std::ofstream ff(fold_txt + "\\" + name + ".txt");
				ff << std::fixed << std::setprecision(8);
				ff << "CONTOUR" << std::endl;
				for (size_t m = 0; m < 3; m++) {
					for (size_t n = 0; n < 4; n++) {
						ff << P.at<double>(m, n) << " ";
					}
					ff << std::endl;
				}
				ff.close();

				// image
				int id_img = cams_sfm_[i]->id_img_;
				std::string file_img_in = fold_rgb + "\\" + cams_name_[id_img] + ".jpg";
				std::string file_img_out = fold_img + "\\" + name + ".jpg";
				cv::Mat img = cv::imread(file_img_in);
				cv::imwrite(file_img_out, img);
			}
		}
	}

	void CrossviewAlignment::SaveforMSP(std::string fold)
	{
		std::string fold_msp = fold + "\\msp";
		if (!std::experimental::filesystem::exists(fold_msp)) {
			std::experimental::filesystem::create_directory(fold_msp);
		}

		std::string file_msp = fold_msp + "\\pose.qin";

		std::ofstream ff(file_msp);
		ff << std::fixed << std::setprecision(12);

		ff << cams_sfm_.size() << std::endl;
		double pixel_mm = 0.005;
		ff << cams_sfm_[0]->cam_model_->f_ * pixel_mm << " "
			<< cams_sfm_[0]->cam_model_->dx_ * pixel_mm << " "
			<< cams_sfm_[0]->cam_model_->dy_ * pixel_mm << " "
			<< pixel_mm << " " << pixel_mm << " " << cols << " " << rows << std::endl;

		Eigen::Matrix3d R_cv2ph;
		R_cv2ph << 1.0, 0.0, 0.0,
			0.0, cos(CV_PI), -sin(CV_PI),
			0.0, sin(CV_PI), cos(CV_PI);

		for (size_t i = 0; i < cams_sfm_.size(); i++)
		{
			int id_img = cams_sfm_[i]->id_img_;
			ff << cams_name_[id_img] + ".jpg" << " " << cams_sfm_[i]->pos_ac_.c(0) << " " << cams_sfm_[i]->pos_ac_.c(1) << " " << cams_sfm_[i]->pos_ac_.c(2) << " ";

			// convert cv coordinate system to photogrammetry system by rotating around x-axis for pi
			Eigen::Matrix3d Rph = R_cv2ph * cams_sfm_[i]->pos_rt_.R;
			double rx = 0.0, ry = 0.0, rz = 0.0;
			rotation::RotationMatrixToEulerAngles(Rph, rx, ry, rz);
			ff << rx << " " << ry << " " << rz;
			if (i < cams_sfm_.size() - 1) {
				ff << std::endl;
			}
		}
		ff.close();
	}

}  // namespace objectsfm
