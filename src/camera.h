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

#ifndef OBJECTSFM_OBJ_CAMERA_H_
#define OBJECTSFM_OBJ_CAMERA_H_

#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>

#include <opencv2/opencv.hpp>

#include "basic_structs.h"
#include "structure.h"

namespace objectsfm {

class Point3D;

// The camera observes the 3D structures then generate the image
class Camera
{
public:
	Camera();
	~Camera();

	void AssociateImage(int id_img);

	void AssociateCamereModel(CameraModel *cam_model);

	// Pose representations
	void SetRTPose(Eigen::Matrix3d R, Eigen::Vector3d t);

	void SetRTPose(RTPose abs_pose1, RTPoseRelative rel_pose2to1);

	void SetACPose(Eigen::Vector3d a, Eigen::Vector3d c);

	void SetGPS(Eigen::Vector3d llt);

	void Transformation(Eigen::Matrix3d R, Eigen::Vector3d t, double scale);

	void Transformation(Eigen::Vector3d angles, Eigen::Vector3d t, double scale);

	void UpdateDataFromPose();

	void UpdatePoseFromData();

	void SetFocalLength(double f);	

	// given the focal length, observe a 3D point, return centralized 2d image point
	bool Observe3DPoint(Eigen::Vector3d pt_w, Eigen::Vector2d &pt2d);

	// add a 3D point to the camera
	void AddPoints(Point3D* pt, size_t idx);

	void AddVisibleCamera(int id_visible_cam);

	void SetMutable(bool is_mutable);

	void SetUsable(bool is_usable);

	void SetID(int id);

	void SetWeigth(double w);

	// 
	int id_;
	int id_img_;
	bool is_mutable_, is_usable_, is_new_;
	CameraModel *cam_model_;
	double w_;
	Eigen::Vector3d llt_;

	RTPose pos_rt_;
	ACPose pos_ac_;
	double data[6]; // for bundle adjustment, [0,1,2] are the angle-axis, [3,4,5] are the translation t.

	Eigen::Matrix<double, 3, 3> K;
	Eigen::Matrix<double, 3, 4> M, P;
	std::map<size_t, Point3D*> pts_;
	std::vector<int> visible_cams_;	
};


}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_CAMERA_H_
