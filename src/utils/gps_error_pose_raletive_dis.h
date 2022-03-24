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

#ifndef OBJECTSFM_OPTIMIZER_GPS_ERROR_POSE_RELATIVE_DIS_H_
#define OBJECTSFM_OPTIMIZER_GPS_ERROR_POSE_RELATIVE_DIS_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace objectsfm
{
	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t.
	struct GPSErrorPoseRelativeDis
	{
		GPSErrorPoseRelativeDis(const double ratio12, const double ratio23, const double weight)
			: ratio12(ratio12), ratio23(ratio23), weight(weight){}

		template <typename T>
		bool operator()(const T* const pose1, const T* const pose2, const T* const pose3,
			T* residuals) const
		{
			// calculate the angles 
			const T vx1 = pose2[3] - pose1[3];
			const T vy1 = pose2[4] - pose1[4];
			//const T vz1 = pose2[5] - pose1[5];
			const T dis1 = sqrt(vx1 * vx1 + vy1 * vy1);

			const T vx2 = pose3[3] - pose2[3];
			const T vy2 = pose3[4] - pose2[4];
			//const T vz2 = pose3[5] - pose2[5];
			const T dis2 = sqrt(vx2 * vx2 + vy2 * vy2);

			const T vx3 = pose1[3] - pose3[3];
			const T vy3 = pose1[4] - pose3[4];
			//const T vz3 = pose1[5] - pose3[5];
			const T dis3 = sqrt(vx3 * vx3 + vy3 * vy3);

			if (dis1 == T(0) || dis2 == T(0) || dis3 == T(0))
			{
				residuals[0] = T(100.0);
				residuals[1] = T(100.0);
			}
			else
			{
				const T ratio12_pred = dis1 / dis2;
				const T ratio23_pred = dis2 / dis3;

				// The error is the difference between the predicted and observed position.
				residuals[0] = weight * abs(ratio12_pred - ratio12);
				residuals[1] = weight * abs(ratio23_pred - ratio23);
			}
			
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double ratio12, const double ratio23, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<GPSErrorPoseRelativeDis, 2, 6, 6, 6>(
				new GPSErrorPoseRelativeDis(ratio12, ratio23, weight)));
		}

		const double ratio12, ratio23;
		const double weight;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_OPTIMIZER_SNAVELY_REPROJECTION_ERROR_H_
