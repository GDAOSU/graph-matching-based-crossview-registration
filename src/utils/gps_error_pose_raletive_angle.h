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

#ifndef OBJECTSFM_OPTIMIZER_GPS_ERROR_POSE_RELATIVE_ANGLE_H_
#define OBJECTSFM_OPTIMIZER_GPS_ERROR_POSE_RELATIVE_ANGLE_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace objectsfm
{
	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t.
	struct GPSErrorPoseRelativeAngle
	{
		GPSErrorPoseRelativeAngle(const double angle1, const double angle2, const double angle3, const double weight)
			: angle1(angle1), angle2(angle2), angle3(angle3), weight(weight){}

		template <typename T>
		bool operator()(const T* const pose1, const T* const pose2, const T* const pose3,
			T* residuals) const
		{
			//T R[9];
			//ceres::AngleAxisToRotationMatrix(const T* angle_axis, T* R);


			// calculate the angles 
			const T vx12 = pose2[3] - pose1[3];
			const T vy12 = pose2[4] - pose1[4];
			const T vz12 = pose2[5] - pose1[5];
			const T l12 = sqrt(vx12*vx12 + vy12 * vy12 + vz12 * vz12);

			const T vx13 = pose3[3] - pose1[3];
			const T vy13 = pose3[4] - pose1[4];
			const T vz13 = pose3[5] - pose1[5];
			const T l13 = sqrt(vx13*vx13 + vy13 * vy13 + vz13 * vz13);

			const T vx23 = pose3[3] - pose2[3];
			const T vy23 = pose3[4] - pose2[4];
			const T vz23 = pose3[5] - pose2[5];
			const T l23 = sqrt(vx23*vx23 + vy23 * vy23 + vz23 * vz23);

			// Compute the angels
			const T angle1_predict = acos((vx12 * vx13 + vy12 * vy13 + vz12 * vz13) / l12 / l13);
			const T angle2_predict = acos((-vx12 * vx23 - vy12 * vy23 - vz12 * vz23) / l12 / l23);
			const T angle3_predict = acos((vx13 * vx23 + vy13 * vy23 + vz13 * vz23) / l13 / l23);


			// The error is the difference between the predicted and observed position.
			residuals[0] = weight * abs(angle1_predict - angle1);
			residuals[1] = weight * abs(angle2_predict - angle2);
			residuals[2] = weight * abs(angle3_predict - angle3);

			//const T th = T(300.0);
			//if (residuals[0] > th || residuals[1] > th || residuals[2] > th)
			//{
			//	std::cout << angle1_predict << " " << angle1 << std::endl;
			//	std::cout << angle2_predict << " " << angle2 << std::endl;
			//	std::cout << angle3_predict << " " << angle3 << std::endl;
			//}

			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double angle1, const double angle2, const double angle3, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<GPSErrorPoseRelativeAngle, 3, 6, 6, 6>(
				new GPSErrorPoseRelativeAngle(angle1, angle2, angle3, weight)));
		}

		const double angle1, angle2, angle3;
		const double weight;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_OPTIMIZER_SNAVELY_REPROJECTION_ERROR_H_
