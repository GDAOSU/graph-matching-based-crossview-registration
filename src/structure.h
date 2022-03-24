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

#ifndef OBJECTSFM_OBJ_STRUCTURE_H_
#define OBJECTSFM_OBJ_STRUCTURE_H_

#include <Eigen/Core>
#include <vector>
#include <map>
#include "basic_structs.h"

namespace objectsfm {

class Camera;

class Point3D
{
public:
	Point3D();
	~Point3D();

	static void Trianglate(Camera *cam1, Eigen::Vector2d pt1, Camera *cam2, Eigen::Vector2d pt2, Eigen::Vector3d &pt3d);

	static double ReprojectToCam(Camera *cam, Eigen::Vector3d pt3d, Eigen::Vector2d pt_obs, Eigen::Vector2d &pt_est);

	static double IntersectionAngle(Eigen::Vector3d pt3d, Camera *cam1, Camera *cam2);


	void SetID(int id);

	void SetData(double x, double y, double z);

	void AddObservation(Camera *cam, double x, double y, size_t id_global, size_t id_local);

	void AddObservation(Camera *cam, double x, double y, size_t id_global);

	void RemoveObservation(size_t id_global);

	bool Trianglate(double th_error, double th_angle);

	bool Trianglate2(double th_error, double th_angle);

	bool Trianglate_Undist(double th_error, double th_angle);

	bool Trianglate3(double th_error, double th_angle);

	void Reprojection();

	void ReprojectionUndistorted();

	double ReprojectToCam(Camera *cam, double obs_x, double obs_y);

	bool SufficientTriangulationAngle(double th_angle_triangulation);

	void SetMutable(bool is_mutable);

	void CalculateWeight(std::string mode);

	void SetWeight(float w);

	void GetIntersectionAngle(double &angle_max, double &angle_min, double &angle_avg);

	void Transformation(Eigen::Matrix3d R, Eigen::Vector3d t, double scale);

	void Transformation(Eigen::Vector3d angles, Eigen::Vector3d t, double scale);

	void Undistortion();

	int id_;
	bool is_mutable_, is_bad_estimated_, is_new_added_;

	// data
	double data[3];   // x y z
	int color[3];	
	std::map<size_t, Camera*> cams_;
	std::map<size_t, size_t> ids_local_;
	std::map<size_t, Eigen::Vector2d> pts2d_;
	std::map<size_t, Eigen::Vector2d> pts2d_undistorted_;
	int key_new_obs_;

	// re-projection error and weight
	double mse_, mse_undistorted_;
	double weight_;
};

struct Point3DNew
{
	Point3D* pt;
	int id_cam1, id_cam2;
	int id_pt1;
	int id_pt2;
};

}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_STRUCTURE_H_
