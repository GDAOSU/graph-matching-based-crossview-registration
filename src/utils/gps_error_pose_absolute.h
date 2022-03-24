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

#ifndef OBJECTSFM_OPTIMIZER_GPS_ERROR_POSE_ABSOLUTE_H_
#define OBJECTSFM_OPTIMIZER_GPS_ERROR_POSE_ABSOLUTE_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace objectsfm
{
	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t.
	struct GPSErrorPoseAbsolute
	{
		GPSErrorPoseAbsolute(const double x, const double y, const double z, const double weight)
			: x(x), y(y), z(z), weight(weight){}

		template <typename T>
		bool operator()(const T* const pose, T* residuals) const
		{
			const T x_predict = pose[3];
			const T y_predict = pose[4];
			const T z_predict = pose[5];

			// The error is the difference between the predicted and observed position.
			residuals[0] = weight * (x_predict - x);
			residuals[1] = weight * (y_predict - y);
			residuals[2] = weight * (z_predict - z);   // z is not as accurate as xy
			//std::cout << residuals[0] << " " << residuals[1] << " " << residuals[2] << std::endl;
			return true;
		}
		
		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x, const double y, const double z, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<GPSErrorPoseAbsolute, 3, 6>(
				new GPSErrorPoseAbsolute(x, y, z, weight)));
		}

		const double x, y, z;
		const double weight;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_OPTIMIZER_SNAVELY_REPROJECTION_ERROR_H_
