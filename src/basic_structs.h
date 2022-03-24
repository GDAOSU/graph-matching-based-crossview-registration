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

#ifndef OBJECTSFM_BASIC_STRUCTS_H_
#define OBJECTSFM_BASIC_STRUCTS_H_

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX_

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_

#include <vector>
#include <string>
#include <map>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "utils/nanoflann_utils_all.h"
#include "utils/nanoflann_all.hpp"

using namespace nanoflann;

namespace objectsfm 
{
	struct ImageInfo
	{
		int rows = 0, cols = 0;
		float zoom_ratio = 1.0;
		float f_mm = 0.0, f_pixel = 0.0;
		float gps_latitude = 0.0, gps_longitude = 0.0, gps_attitude = 0.0;
		float rx = 0.0, ry = 0.0, rz = 0.0;
		
		std::string cam_maker, cam_model;
		std::string name;
	};

	struct CameraModel
	{
	public:
		CameraModel() 
		{
			f_mm_ = 0;
			f_ = 0;
			is_mutable_ = true;
			k1_ = 0.0, k2_ = 0.0;
			dx_ = 0.0, dy_ = 0.0; 
			p1_ = 0.0, p2_ = 0.0;
		};

		CameraModel(int id, int h, int w, double f_mm, double f, std::string cam_maker, std::string cam_model )
		{
			f_mm_ = f_mm;
			f_ = f;
			if (f > 0 ) {
				f_hyp_ = f;
			}
			else {
				f_hyp_ = MAX_(w, h)*1.2;
			}
			
			w_ = w, h_ = h;
			px_ = w_ / 2.0, py_ = h_ / 2.0;
			k1_ = 0.0, k2_ = 0.0;
			dx_ = 0.0, dy_ = 0.0;
			p1_ = 0.0, p2_ = 0.0;
			id_ = id;
			cam_maker_ = cam_maker;
			cam_model_ = cam_model;
			num_cams_ = 0;
			is_mutable_ = true;

			UpdateDataFromModel();
		}

		void SetIntrisicParas(double f, double px, double py)
		{
			f_ = f;
			px_ = px;
			py_ = py;
			UpdateDataFromModel();
		}

		void SetDistortionParas(double k1, double k2)
		{
			k1_ = k1;
			k2_ = k2;
			UpdateDataFromModel();
		}

		void SetDistortionParas(double k1, double k2, double dx, double dy)
		{
			k1_ = k1;
			k2_ = k2;
			dx_ = dx;
			dy_ = dy;
			UpdateDataFromModel();
		}

		void SetFocalLength(double f)
		{
			f_ = f;
			UpdateDataFromModel();
		}

		void UpdateDataFromModel()
		{
			data[0] = f_;
			data[1] = k1_;
			data[2] = k2_;
			data[3] = dx_;
			data[4] = dy_;
			data[5] = p1_;
			data[6] = p2_;
		}

		void UpdataModelFromData()
		{			
			f_ = data[0];
			k1_ = data[1];
			k2_ = data[2];
			dx_ = data[3];
			dy_ = data[4];
			p1_ = data[5];
			p2_ = data[6];
		}

		void AddCamera(int idx)
		{
			idx_cams_.push_back(idx);
			num_cams_++;
		}

		void SetMutable(bool is_mutable)
		{
			is_mutable_ = is_mutable;
		}

		int id_;
		std::string cam_maker_, cam_model_;
		int w_, h_;       // cols and rows
		double f_mm_, f_, f_hyp_, px_, py_;  // focal length and principle point
		double k1_, k2_, dx_, dy_, p1_, p2_;   // distrotion paremeters
		double data[7];    // for bundle adjustment, {f_, k1_, k2_, dcx_, dcy_, p1_, p2_}, respectively
		int num_cams_;
		std::vector<int> idx_cams_;
		bool is_mutable_;
	};

	struct RTPoseRelative
	{
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
	};

	// [Xc 1] = [R|t]*[Xw 1], converts a 3D point in the world-3d coordinate system
	// into the camera-3d coordinate system
	struct RTPose
	{
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
	};

	// Another form of camera pose designed for bundle adjustment
	struct ACPose
	{
		Eigen::Vector3d a;  // R = Rodrigues(a)
		Eigen::Vector3d c;  // Xc = R*(Xw-c) since Xc = R*Xw+t, we have t = -R*c and c = -R'*t
	};

	struct SfMOptions
	{
		// image reading
		std::string input_fold;
		std::string output_fold;

		// feature detection
		int num_image_voc = 500;
		int size_image = 2000*1500;
		std::string feature_type = "vlsift"; // vlsift ezsift cudasift
		bool resize = false;

		// feature matching
		std::string init_matching_type = "all";  // all, gps, bow
		std::string fine_matching_type = "flann_cpu"; // flann_cpu, sift_gpu, cascade_hash, cross_check_cpu, cross_check_gpu
		int knn = 50;		

		// share intrinsic
		bool use_same_camera = false;
		std::string distortion_mode = "fk1k2";

		// to generat the global index of the keypoint on each image
		// id_pt_global = id_pt_image + id_image * label_max_per_image
		size_t idx_max_per_image = 1000000;

		// localization
		int th_seedpair_structures = 20;		
		int th_max_failure_localization = 5;
		int th_min_2d3d_corres = 10;

		// bundle
		bool minimizer_progress_to_stdout = false;
		int th_max_iteration_full_bundle = 50;
		int th_max_iteration_partial_bundle = 50;
		float th_step_full_bundle_adjustment = 0.1;
		float th_step_partial_bundle_adjustment = 0.01;

		// outliers
		double th_mse_localization = 7.0;
		double th_mse_reprojection = 5.0;
		double th_mse_outliers = 3.0;

		// triangulate angle
		double th_angle_small = 1.0 / 180.0*3.1415;
		double th_angle_large = 2.0 / 180.0*3.1415;
	};


	//
	struct DatabaseImageOptions
	{
		// voc
		int num_image_voc = 500;
		int fbow_k = 16;  // number of children for each node
		int fbow_l = 8;     // number of levels of the voc-tree

		bool resize = false;
		bool is_sequential = false;
		float size_image = 2000*1500;
		std::string feature_type = "vlsift"; // vlsift ezsift cudasift
	};

	//
	struct GraphOptions
	{
		std::string sim_graph_type = "WordNumber";  // BoWDistance WordNumber
		std::string init_matching_type = "exhaustive"; // all, gps, bow
		std::string fine_matching_type = "flann_cpu"; //  flann_cpu, sift_gpu, cascade_hash, cross_check_cpu, cross_check_gpu		
		std::string priori_file = "";
		int knn = 50;
	};

	// 
	struct BundleAdjustOptions
	{
		int max_num_iterations = 100;
		bool minimizer_progress_to_stdout = false;
		int num_threads = 8;
		int ceres_solver_type = 5; // 5 ITERATIVE_SCHUR  3 DENSE_SCHUR
		double th_tolerance = 1e-6;
		bool cal_cov = false;
		std::string distort_mode = "fk1k2"; // fk1 fk1k2, fk1k2dxdy
		bool reset_distort_paras_;
	};

	// 
	struct AccuracyOptions
	{
		std::string data_type = "COLMAP";

		// the covariance matrix of observations [UNIT, VARIANCE_FACTOR, SQUARED_RESIDUALS, STRUCTURE_TENSOR]
		std::string cov_type = "UNIT";

		// algorithm for inversion of Schur complement matrix [SVD_QR_ITERATION, SVD_DEVIDE_AND_CONQUER, LHUILLIER, TE-INVERSION, NBUP]
		std::string alg_type = "NBUP";
	};

	struct SiftMatchingOptions {
		// Number of threads for feature matching and geometric verification.
		int num_threads = -1;

		// Whether to use the GPU for feature matching.
		bool use_gpu = true;

		// Index of the GPU used for feature matching. For multi-GPU matching,
		// you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
		std::string gpu_index = "-1";

		// Maximum distance ratio between first and second best match.
		double max_ratio = 0.6;

		// Maximum distance to best match.
		double max_distance = 0.7;

		// Whether to enable cross checking in matching.
		bool cross_check = true;

		// Maximum number of matches.
		int max_num_matches = 32768;

		// Maximum epipolar error in pixels for geometric verification.
		double max_error = 4.0;

		// Confidence threshold for geometric verification.
		double confidence = 0.999;

		// Minimum/maximum number of RANSAC iterations. Note that this option
		// overrules the min_inlier_ratio option.
		int min_num_trials = 30;
		int max_num_trials = 10000;

		// A priori assumed minimum inlier ratio, which determines the maximum
		// number of iterations.
		double min_inlier_ratio = 0.25;

		// Minimum number of inliers for an image pair to be considered as
		// geometrically verified.
		int min_num_inliers = 15;

		// Whether to attempt to estimate multiple geometric models per image pair.
		bool multiple_models = false;

		// Whether to perform guided matching, if geometric verification succeeds.
		bool guided_matching = false;

		bool Check() const;
	};

	//
	struct DenseOptions
	{
		int disp_size = 128;
		float uniqueness = 0.96;
		int idx_max_per_image = 1000000;

	};

	struct SimilarityTrans
	{
		double s;
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
	};

	//
	struct ListKeyPoint
	{
		std::vector<cv::KeyPoint> pts;
	};

	struct ListFloat
	{
		std::vector<float> data;
	};

	struct QuaryResult
	{
		int ids[10];
	};

	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, SiftList<float> >, SiftList<float>, 128/*dim*/ > my_kd_tree_t;

}
#endif //OBJECTSFM_BASIC_STRUCTS_H_