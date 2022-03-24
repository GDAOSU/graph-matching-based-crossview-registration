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

#ifndef REGISTRATION_ERROR_TRANS_H_
#define REGISTRATION_ERROR_TRANS_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t, [6] is the scale	
	struct AbsoluteErrorTran
	{
		AbsoluteErrorTran(const double x, const double y, const double z, 
		const double xw, const double yw, const double zw, const double weight)
			: x(x), y(y), z(z), xw(xw), yw(yw), zw(zw),weight(weight) {}

		template <typename T>
		bool operator()(const T* const tran,
			T* residuals) const
		{
			// pose[0,1,2] are the angle-axis rotation.
			T xyz[3];
			xyz[0] = T(x);
			xyz[1] = T(y);
			xyz[2] = T(z);

			T p[3];
			ceres::AngleAxisRotatePoint(tran, xyz, p);

			// pose[3,4,5] are the translation.
			T s = tran[6];
			p[0] = s * p[0] + tran[3];
			p[1] = s * p[1] + tran[4];
			p[2] = s * p[2] + tran[5];
		
			// The error is the difference between the predicted and observed position.
			residuals[0] = weight*(p[0] - xw);
			residuals[1] = weight*(p[1] - yw);
			residuals[2] = weight*(p[2] - zw);
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x, const double y, const double z, 
		const double xw, const double yw, const double zw, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<AbsoluteErrorTran, 3, 7>(
				new AbsoluteErrorTran(x, y, z, xw, yw, zw, weight)));
		}

		const double x,y,z;
		const double xw,yw,zw;		
		const double weight;
	};


    // pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t, [6] is the scale	
	struct RelativeErrorTranTran
	{
		RelativeErrorTranTran(const double x1, const double y1, const double z1, 
		const double x2, const double y2, const double z2, const double weight)
			: x1(x1), y1(y1), z1(z1), x2(x2), y2(y2), z2(z2),weight(weight) {}

		template <typename T>
		bool operator()(const T* const tran1, const T* const tran2,
			T* residuals) const
		{
			// tran1
			T xyz1[3];
			xyz1[0] = T(x1);
			xyz1[1] = T(y1);
			xyz1[2] = T(z1);
			T p1[3];
			ceres::AngleAxisRotatePoint(tran1, xyz1, p1);
			T s1 = tran1[6];
			p1[0] = s1 * p1[0] + tran1[3];
			p1[1] = s1 * p1[1] + tran1[4];
			p1[2] = s1 * p1[2] + tran1[5];

			// tran2
			T xyz2[3];
			xyz2[0] = T(x2);
			xyz2[1] = T(y2);
			xyz2[2] = T(z2);
			T p2[3];
			ceres::AngleAxisRotatePoint(tran2, xyz2, p2);
			T s2 = tran2[6];
			p2[0] = s2 * p2[0] + tran2[3];
			p2[1] = s2 * p2[1] + tran2[4];
			p2[2] = s2 * p2[2] + tran2[5];

			// The error is the difference between the predicted and observed position.
			residuals[0] = weight*(p1[0] - p2[0]);
			residuals[1] = weight*(p1[1] - p2[1]);
			residuals[2] = weight*(p1[2] - p2[2]);
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x1, const double y1, const double z1, 
		const double x2, const double y2, const double z2, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<RelativeErrorTranTran, 3, 7, 7>(
				new RelativeErrorTranTran(x1, y1, z1, x2, y2, z2, weight)));
		}

		const double x1,y1,z1;
		const double x2,y2,z2;		
		const double weight;
	};


	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t, [6] is the scale	
	struct DirectionErrorTran
	{
		DirectionErrorTran(const double x, const double y, const double z,
			const double xw, const double yw, const double zw, const double weight)
			: x(x), y(y), z(z), xw(xw), yw(yw), zw(zw), weight(weight) {}

		template <typename T>
		bool operator()(const T* const tran,
			T* residuals) const
		{
			// pose[0,1,2] are the angle-axis rotation.
			T xyz[3];
			xyz[0] = T(x);
			xyz[1] = T(y);
			xyz[2] = T(z);

			T p[3];
			ceres::AngleAxisRotatePoint(tran, xyz, p);
			T N = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
			N = sqrt(N);
			p[0] /= N;
			p[1] /= N;
			p[2] /= N;			

			// The error is the difference between the predicted and observed position.
			residuals[0] = weight * (p[0] - xw);
			residuals[1] = weight * (p[1] - yw);
			residuals[2] = weight * (p[2] - zw);
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x, const double y, const double z,
			const double xw, const double yw, const double zw, const double weight)
		{
			return (new ceres::AutoDiffCostFunction<DirectionErrorTran, 3, 7>(
				new DirectionErrorTran(x, y, z, xw, yw, zw, weight)));
		}

		const double x, y, z;
		const double xw, yw, zw;
		const double weight;
	};



	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t, [6] is the scale	
	struct DisErrorTran
	{
		DisErrorTran(const double x, const double y, const int n, const double* pts)
			: x(x), y(y), n(n), pts(pts) {}

		template <typename T>
		bool operator()(const T* const tran,
			T* residuals) const
		{
			// pose[0,1,2] are the angle-axis rotation.
			T s = tran[0];
			T r = tran[1];
			T tx = tran[2];
			T ty = tran[3];

			T cos_r = cos(r);
			T sin_r = sin(r);
			T xnew = s * (cos_r*T(x) - sin_r * T(y)) + tx;
			T ynew = s * (sin_r*T(x) + cos_r * T(y)) + ty;

			residuals[0] = T(0.0);
			for (size_t i = 0; i < n; i++) {
				T dis_i = abs(xnew - pts[2 * i + 0]) + abs(ynew - pts[2 * i + 1]);
				residuals[0] += (1.0 - 1.0 / (exp(dis_i / 5.0))) / 1000.0;
			}
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x, const double y, const int n, const double* pts)
		{
			return (new ceres::AutoDiffCostFunction<DisErrorTran, 1, 4>(
				new DisErrorTran(x, y, n, pts)));
		}

		const double x, y;
		const int n;
		const double* pts;
	};


	// pose: [0,1,2] are the angle-axis, [3,4,5] are the translation t, [6] is the scale	
	struct RelativeErrorTran
	{
		RelativeErrorTran(const double w1, const double w2, const double w3, const double w4)
			: w1(w1), w2(w2), w3(w3), w4(w4) {}

		template <typename T>
		bool operator()(const T* const tran1, const T* const tran2,
			T* residuals) const
		{
			residuals[0] = (tran1[0] - tran2[0]) * w1
				+ (tran1[1] - tran2[1]) * w2
				+ (tran1[2] - tran2[2]) * w3
				+ (tran1[3] - tran2[3]) * w4;
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double w1, const double w2, const double w3, const double w4)
		{
			return (new ceres::AutoDiffCostFunction<RelativeErrorTran, 1, 4, 4>(
				new RelativeErrorTran(w1, w2, w3, w4)));
		}

		const double w1, w2, w3, w4;
	};



	struct RelativeErrorTran2
	{
		RelativeErrorTran2(const double x, const double y, const double w)
			: x(x), y(y), w(w) {}

		template <typename T>
		bool operator()(const T* const tran1, const T* const tran2,
			T* residuals) const
		{
			T x1 = tran1[0] * (cos(tran1[1])*T(x) - sin(tran1[1]) * T(y)) + tran1[2];
			T y1 = tran1[0] * (sin(tran1[1])*T(x) + cos(tran1[1]) * T(y)) + tran1[3];
			T x2 = tran2[0] * (cos(tran2[1])*T(x) - sin(tran2[1]) * T(y)) + tran2[2];
			T y2 = tran2[0] * (sin(tran2[1])*T(x) + cos(tran2[1]) * T(y)) + tran2[3];

			residuals[0] = (abs(x1 - x2) + abs(y1 - y2))*w;
			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x, const double y, const double w)
		{
			return (new ceres::AutoDiffCostFunction<RelativeErrorTran2, 1, 4, 4>(
				new RelativeErrorTran2(x, y, w)));
		}

		const double x, y, w;
	};
#endif  // REGISTRATION_ERROR_TRANS_H_
