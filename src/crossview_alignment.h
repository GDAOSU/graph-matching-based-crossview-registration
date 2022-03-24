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

#ifndef OBJECTSFM_SYSTEM_CROSSVIEW_ALIGNMENT_H_
#define OBJECTSFM_SYSTEM_CROSSVIEW_ALIGNMENT_H_

#include "basic_structs.h"
#include "camera.h"
#include "structure.h"
#include "optimizer.h"

namespace objectsfm {

	// the system is designed to handle sfm from internet images, in which not
	// all the focal lengths are known
	struct SateInfo
	{
		std::string file_dsm;
		std::string file_ortho;
		std::string file_mask_building;
		std::string file_mask_ndvi;
		double dx, dy, xs, ys;
	};

	struct StreetInfo
	{
		std::string fold;
		std::string fold_sfm;
		std::string fold_dense;
		std::string fold_align;
	};

	class CrossviewAlignment
	{
	public:
		CrossviewAlignment();
		~CrossviewAlignment();

		void SetSateInfo(std::string sate_dsm, std::string sate_ortho, std::string sate_mask_building, std::string sate_mask_ndvi);

		void SetStreetInfo(std::string fold);

		void RunAlign();

		// convert cams and pts
		void ConvertCamsPts(std::string fold_img, std::string file_street_bundle,
			std::string file_street_pose, std::string fold_out);

		void Convert2Geo(std::string file_tfw);

		// full bundle
		void IncrementalBundleAdjustment();

		void PartialBundleAdjustment();

		void FullBundleAdjustment();

		void ImmutableCamsPoints();

		void MutableCamsPoints();

		void RemovePointOutliers();

		// accuracy analysis
		void AccuracySateBoundary();

		void AccuracyLidar(std::string file_lidar, std::string fold_street);

		void Accuracy2D(std::string file_dst, std::string file_src, std::string file_out);

		void AccuracyMatching(std::string fold_test);

		//
		void Euler2Rotation(double rx, double ry, double rz, Eigen::Matrix3d &R);

		void Rotation2Euler(Eigen::Matrix3d R, double &rx, double &ry, double &rz);

		void Pts2Euler(std::vector<Eigen::Vector3d> pts_src, std::vector<Eigen::Vector3d> pts_dst, 
			double &rx, double &ry, double &rz);

		void ReadinImages(std::string fold);

		void ReadinCamNames(std::string path, std::vector<int> &cam_names);

		void WriteTempResultOut(std::string path);

		void ReadTempResultIn(std::string path);

		void ReadinTrans(std::string path, std::vector<std::vector<double>> &trans);

		void WriteCameraPointsOut(std::string path);
		
		// save
		void SaveUndistortedImage(std::string fold, std::string fold_rgb);

		void SaveforOpenMVSNVM(std::string fold);

		void OpenMVSNVMConvert(std::string fold);

		void SaveforOpenMVSBundle(std::string fold);

		void SaveforCMVS(std::string fold);

		void SaveforMSP(std::string fold);

	public:		
		SateInfo* sate_info_;
		StreetInfo* street_info_;
		BundleAdjustOptions bundle_options_;

		double th_mse_outliers_ = 2.0;
		int rows, cols;
		double fx, fy, cx, cy;  // initial intrinsic parameters from slam

		CameraModel *cam_model_;          // camera models
		std::vector<std::string> cams_name_;  // camera image name
		std::vector<Camera*> cams_sfm_;  // cameras from sfm
		std::vector<Point3D*> pts_sfm_;  // 3d points from sfm
		std::vector<Camera*> cams_all_;  // cameras from sfm
		std::map<int, int> img_cam_map_; // first, image id; second, corresponding camera id
		std::vector<cv::Point3d> cams_gps_;  // camera pose from gps
		int id_img_cur_;
		std::vector<Eigen::Vector3d> cams_pos_ori_;
		double offx_, offy_;
		int id_cams_, id_came_;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_SYSTEM_INCREMENTAL_SFM_SYSTEM_H_
