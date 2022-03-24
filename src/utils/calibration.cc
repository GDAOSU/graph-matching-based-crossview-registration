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

#include "calibration.h"

#include "utils/find_polynomial_roots_companion_matrix.h"

namespace objectsfm
{
	Calibration::Calibration()
	{
	}
	Calibration::~Calibration()
	{
	}

	void Calibration::UndistortedPts(std::vector<Eigen::Vector2d> pts, std::vector<Eigen::Vector2d>& pts_undistorted, CameraModel * cam_model)
	{
		double k1 = cam_model->k1_;
		double k2 = cam_model->k2_;
		double f = cam_model->f_;
		double dcx = cam_model->dx_;
		double dcy = cam_model->dy_;
		if (k1 == 0 || k2 == 0 || f == 0) {
			pts_undistorted = pts;
			return;
		}

		Eigen::VectorXd polynomial(6);
		polynomial[0] = pow(k2, 2);
		polynomial[1] = 2 * k1 * k2;
		polynomial[2] = 2 * k2 + pow(k1, 2);
		polynomial[3] = 2 * k1;
		polynomial[4] = 1.0;

		Eigen::VectorXd real(5), imaginary(5);
		pts_undistorted.resize(pts.size());
		for (size_t i = 0; i < pts.size(); i++)
		{
			double u = pts[i][0] - dcx;
			double v = pts[i][1] - dcy;

			// undistortion
			/* u = f * d * x      v = f * d * y      d = 1 + k1 * r2 + k2 * r4
			so, u2+v2 = f2*d2*(x*x+ y*y) = f2*d2*r2 = f2*(1 + k1 * r2 + k2 * r4)2*r2
			*/
			polynomial[5] = -(u*u + v * v) / (f*f);
			FindPolynomialRootsCompanionMatrix(polynomial, &real, &imaginary);
			double r2 = 0;
			for (size_t j = 0; j < 5; j++)
			{
				if (imaginary[j] == 0) {
					r2 = real[j];
				}
			}
			if (r2 != 0) {
				double d = 1 + k1 * r2 + k2 * r2 * r2;
				pts_undistorted[i][0] = u / d;
				pts_undistorted[i][1] = v / d;
			}
		}
	}

	void Calibration::UndistortedPts(Eigen::Vector2d pt, Eigen::Vector2d & pt_undistorted, CameraModel * cam_model)
	{
		double k1 = cam_model->k1_;
		double k2 = cam_model->k2_;
		double f = cam_model->f_;
		double dcx = cam_model->dx_;
		double dcy = cam_model->dy_;

		if (k1 == 0 || k2 == 0 || f == 0) {
			pt_undistorted = pt;
			return;
		}

		Eigen::VectorXd polynomial(6);
		polynomial[0] = pow(k2, 2);
		polynomial[1] = 2 * k1 * k2;
		polynomial[2] = 2 * k2 + pow(k1, 2);
		polynomial[3] = 2 * k1;
		polynomial[4] = 1.0;

		//
		double u = pt[0] - dcx;
		double v = pt[1] - dcy;
		polynomial[5] = -(u*u + v * v) / (f*f);
		Eigen::VectorXd real(5), imaginary(5);
		FindPolynomialRootsCompanionMatrix(polynomial, &real, &imaginary);
		double r2 = 0;
		for (size_t j = 0; j < 5; j++) {
			if (imaginary[j] == 0) {
				r2 = real[j];
			}
		}

		double d = 1.0;
		if (r2 != 0) {
			d = 1.0 + k1 * r2 + k2 * r2 * r2;
		}
		pt_undistorted[0] = u / d;
		pt_undistorted[1] = v / d;
	}
};
