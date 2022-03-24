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

#ifndef OBJECTSFM_OPTIMIZER_REPROJECTION_ERROR_POSE_XYZ_H_
#define OBJECTSFM_OPTIMIZER_REPROJECTION_ERROR_POSE_XYZ_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace objectsfm
{
	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t.
	// xyz: is the x, y, z coordinates of the 3D point
	struct ReprojectionErrorPoseXYZ
	{
		ReprojectionErrorPoseXYZ(const double observed_x, const double observed_y, const double* cam, const double weight, const double f_init, const std::string mode)
			: observed_x(observed_x), observed_y(observed_y), cam(cam), weight(weight), f_init(f_init), mode(mode) {}

		template <typename T>
		bool operator()(const T* const pose,
			const T* const xyz,
			T* residuals) const
		{
			// pose[0,1,2] are the angle-axis rotation.
			T p[3];
			ceres::AngleAxisRotatePoint(pose, xyz, p);

			// pose[3,4,5] are the translation.
			p[0] += pose[3];
			p[1] += pose[4];
			p[2] += pose[5];

			// Compute the center of distortion. The sign change comes from
			// the camera model that Noah Snavely's Bundler assumes, whereby
			// the camera coordinate system has a negative z axis.
			const T xp = p[0] / p[2];
			const T yp = p[1] / p[2];

			// Apply second and fourth order radial distortion
			T predicted_x = T(0.0);
			T predicted_y = T(0.0);
			if (mode == "fk1k2") {
				const T focal = T(cam[0]);
				const T k1 = T(cam[1]);
				const T k2 = T(cam[2]);
				const T dx = T(cam[3]);
				const T dy = T(cam[4]);
				const T r2 = xp * xp + yp * yp;
				const T distortion = 1.0 + k1 * r2 + k2 * r2 * r2;
				predicted_x = focal * distortion * xp;
				predicted_y = focal * distortion * yp;
			}
			else if (mode == "fk1k2dxdy") {
				const T focal = T(cam[0]);
				const T k1 = T(cam[1]);
				const T k2 = T(cam[2]);
				const T dx = T(cam[3]);
				const T dy = T(cam[4]);
				const T r2 = xp * xp + yp * yp;
				const T distortion = 1.0 + k1 * r2 + k2 * r2 * r2;
				predicted_x = focal * distortion * xp + dx;
				predicted_y = focal * distortion * yp + dy;

			}
			else if (mode == "fk1k2dxdyp1p2") {
				const T focal = T(cam[0]);
				const T k1 = T(cam[1]);
				const T k2 = T(cam[2]);
				const T dx = T(cam[3]);
				const T dy = T(cam[4]);
				const T p1 = T(cam[5]);
				const T p2 = T(cam[6]);
				const T r2 = xp * xp + yp * yp;
				const T distortion = 1.0 + k1 * r2 + k2 * r2 * r2;
				predicted_x = focal * (xp * distortion + T(2.0)*p1*xp*yp + p2 * (r2 + T(2.0) * xp*xp)) + dx;
				predicted_y = focal * (yp * distortion + T(2.0)*p2*xp*yp + p1 * (r2 + T(2.0) * yp*yp)) + dy;
			}

			// The error is the difference between the predicted and observed position.
			residuals[0] = weight * abs(predicted_x - observed_x) + abs(T(cam[0]) - T(f_init)) / T(f_init) * 0.1;
			residuals[1] = weight * abs(predicted_y - observed_y) + abs(T(cam[0]) - T(f_init)) / T(f_init) * 0.1;

			if (abs(residuals[0]) + abs(residuals[1]) > T(1000)) {
				residuals[0] = T(100.0);
				residuals[1] = T(100.0);
			}
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double *cam, const double weight, const double f_init, const std::string mode)
		{
			return (new ceres::AutoDiffCostFunction<ReprojectionErrorPoseXYZ, 2, 6, 3>(
				new ReprojectionErrorPoseXYZ(observed_x, observed_y, cam, weight, f_init, mode)));
		}

		const double observed_x;
		const double observed_y;
		const double *cam;
		const double weight;
		const double f_init;
		const std::string mode;
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_OPTIMIZER_REPROJECTION_ERROR_POSE_XYZ_H_
