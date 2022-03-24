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

#ifndef OBJECTSFM_OPTIMIZER_Z_ERROR_POSE_POSE_H_
#define OBJECTSFM_OPTIMIZER_Z_ERROR_POSE_POSE_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace objectsfm
{
	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t.
	struct ZErrorPosePose
	{
		ZErrorPosePose(const double weight) : weight(weight){}

		template <typename T>
		bool operator()(const T* const pose1, const T* const pose2,
			T* residuals) const
		{
			residuals[0] = pose1[3] - pose2[3];
			residuals[1] = pose1[4] - pose2[4];
			residuals[2] = pose1[5] - pose2[5];
			residuals[0] = abs(weight * T(1000.0) * residuals[0]);
			residuals[1] = abs(weight * T(1000.0) * residuals[1]);
			residuals[2] = abs(weight * T(1000.0) * residuals[2]);


			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double weight) {
			return (new ceres::AutoDiffCostFunction<ZErrorPosePose, 3, 6, 6>(new ZErrorPosePose(weight)));
		}

		const double weight;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_OPTIMIZER_SNAVELY_REPROJECTION_ERROR_H_
