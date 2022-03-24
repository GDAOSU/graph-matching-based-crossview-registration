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

#ifndef OBJECTSFM_EPI_GEO_H_
#define OBJECTSFM_EPI_GEO_H_

#include <opencv2/opencv.hpp>

#include "basic_structs.h"
#include "camera.h"
#include "structure.h"


namespace objectsfm {


class EpipolarGeometry
{
public:
	EpipolarGeometry();
	~EpipolarGeometry();

	// the main pipeline
	void Run(std::string fold_in, std::string fold_out, bool given_pose);

	void SearchImagePaths();

	void ReadinPoseFile(std::string pose_file);

	void ReadinSfMFile(std::string sfm_file);

	std::vector<int> EliminateRedudentImages();

	void GenerateAllDisparity();

	void SGMDense(int id_left, int id_right, bool lr_shift, cv::Mat &disparity, cv::Mat &Hl, cv::Mat &Hr);

	void SGMDenseBlock(int id_left, int id_right, cv::Mat &disparity, cv::Mat &Hl, cv::Mat &Hr);

	void ELASDense(int id_left, int id_right, cv::Mat &disparity, cv::Mat &Hl, cv::Mat &Hr);

	void EpipolarRectification(cv::Mat &img1, cv::Mat K1, cv::Mat R1, cv::Mat t1,
		                       cv::Mat &img2, cv::Mat K2, cv::Mat R2, cv::Mat t2, 
		                       bool write_out, cv::Mat &P1_, cv::Mat &P2_);

	void EpipolarRectification2(int idx1, int idx2, cv::Mat &img1, cv::Mat &img2, cv::Mat &H1, cv::Mat &H2);	

	//void EpipolarRectification2(std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, cv::Mat &H1, cv::Mat &H2);

	void DisparityFusion(int th_num_ray, float th_angle_min, float th_proj);

	void DisparityFusion(int id1, 
		std::vector<int> id2_list, std::vector<cv::Mat> &disparity_list, std::vector<cv::Mat> H_list,
		int th_num_ray, float th_angle_min, float th_proj, int step_out, std::string file_out);

	void SavePoseFile(std::string sfm_file);

	void Depth2Points(std::string fold);

	void LeftRightSift(cv::Mat &img);

	bool ReadinFlow(int idx, cv::Mat & flowx, cv::Mat & flowy);

private:
	double f_, k1_, k2_, rows_, cols_;

	std::string fold_img_, fold_output_;
	std::vector<std::string> names_;

	CameraModel* cam_model_;
	std::vector<Camera*> cams_;  // cameras from sfm
	std::vector<Point3D*> pts_;  // 3d points from sfm
	std::vector<int> key_imgs_;

	std::vector<int> good_frames_;

};

}  // namespace objectsfm

#endif  // OBJECTSFM_SYSTEM_DENSE_H_
