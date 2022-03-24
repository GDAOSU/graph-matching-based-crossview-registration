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

#ifndef OBJECTSFM_CAMERA_CALIBRATION_H_
#define OBJECTSFM_CAMERA_CALIBRATION_H_

#include <Eigen/Core>
#include <vector>
#include <map>

#include "basic_structs.h"

namespace objectsfm 
{
	class Calibration
	{
	public:
		Calibration();
		~Calibration();

		static void UndistortedPts(std::vector<Eigen::Vector2d> pts, std::vector<Eigen::Vector2d> &pts_undistorted, CameraModel* cam_model);

		static void UndistortedPts(Eigen::Vector2d pt, Eigen::Vector2d &pt_undistorted, CameraModel* cam_model);

	};


}
#endif //OBJECTSFM_CAMERA_CALIBRATION_H_