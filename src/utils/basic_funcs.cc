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

#include "basic_funcs.h"

namespace objectsfm
{
	namespace rotation
	{
		// This algorithm comes from "Quaternion Calculus and Fast Animation",
		// Ken Shoemake, 1987 SIGGRAPH course notes
		void RotationMatrixToQuaternion(Eigen::Matrix3d R, double *quaternion)
		{
			const double trace = R(0, 0) + R(1, 1) + R(2, 2);
			if (trace >= 0.0)
			{
				double t = sqrt(trace + double(1.0));
				quaternion[0] = double(0.5) * t;
				t = double(0.5) / t;
				quaternion[1] = (R(2, 1) - R(1, 2)) * t;
				quaternion[2] = (R(0, 2) - R(2, 0)) * t;
				quaternion[3] = (R(1, 0) - R(0, 1)) * t;
			}
			else
			{
				int i = 0;
				if (R(1, 1) > R(0, 0))
				{
					i = 1;
				}

				if (R(2, 2) > R(i, i))
				{
					i = 2;
				}

				const int j = (i + 1) % 3;
				const int k = (j + 1) % 3;
				double t = sqrt(R(i, i) - R(j, j) - R(k, k) + double(1.0));
				quaternion[i + 1] = double(0.5) * t;
				t = double(0.5) / t;
				quaternion[0] = (R(k, j) - R(j, k)) * t;
				quaternion[j + 1] = (R(j, i) + R(i, j)) * t;
				quaternion[k + 1] = (R(k, i) + R(i, k)) * t;
			}
		}		

		void QuaternionToRotationMatrix(double * q, Eigen::Matrix3d & R)
		{
			R(0, 0) = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3];
			R(0, 1) = 2 * q[1] * q[2] - 2 * q[0] * q[3];
			R(0, 2) = 2 * q[3] * q[1] + 2 * q[0] * q[2];
			R(1, 0) = 2 * q[1] * q[2] + 2 * q[0] * q[3];
			R(1, 1) = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3];
			R(1, 2) = 2 * q[2] * q[3] - 2 * q[0] * q[1];
			R(2, 0) = 2 * q[3] * q[1] - 2 * q[0] * q[2];
			R(2, 1) = 2 * q[2] * q[3] + 2 * q[0] * q[1];
			R(2, 2) = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2];
		}

		void QuaternionToAngleAxis(const double* quaternion, double* angle_axis)
		{
			const double& q1 = quaternion[1];
			const double& q2 = quaternion[2];
			const double& q3 = quaternion[3];
			const double sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

			// For quaternions representing non-zero rotation, the conversion
			// is numerically stable.
			if (sin_squared_theta > double(0.0))
			{
				const double  sin_theta = sqrt(sin_squared_theta);
				const double& cos_theta = quaternion[0];

				// If cos_theta is negative, theta is greater than pi/2, which
				// means that angle for the angle_axis vector which is 2 * theta
				// would be greater than pi.
				//
				// While this will result in the correct rotation, it does not
				// result in a normalized angle-axis vector.
				//
				// In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
				// which is equivalent saying
				//
				//   theta - pi = atan(sin(theta - pi), cos(theta - pi))
				//              = atan(-sin(theta), -cos(theta))
				//
				const double two_theta = double(2.0) * ((cos_theta < 0.0)
					? atan2(-sin_theta, -cos_theta)
					: atan2(sin_theta, cos_theta));
				const double k = two_theta / sin_theta;
				angle_axis[0] = q1 * k;
				angle_axis[1] = q2 * k;
				angle_axis[2] = q3 * k;
			}
			else
			{
				// For zero rotation, sqrt() will produce NaN in the derivative since
				// the argument is zero.  By approximating with a Taylor series,
				// and truncating at one term, the value and first derivatives will be
				// computed correctly when Jets are used.
				const double k(2.0);
				angle_axis[0] = q1 * k;
				angle_axis[1] = q2 * k;
				angle_axis[2] = q3 * k;
			}
		}

		void RotationMatrixToAngleAxis(Eigen::Matrix3d R, Eigen::Vector3d &axis)
		{
			double quaternion[4];
			RotationMatrixToQuaternion(R, quaternion);
			double angle_axis[3];
			QuaternionToAngleAxis(quaternion, angle_axis);
			axis = Eigen::Vector3d(angle_axis[0], angle_axis[1], angle_axis[2]);
		}

		void AngleAxisToRotationMatrix(Eigen::Vector3d &angle_axis, Eigen::Matrix3d &R)
		{
			static const double kOne = double(1.0);
			const double theta2 = angle_axis.dot(angle_axis);
			if (theta2 > double(std::numeric_limits<double>::epsilon()))
			{
				// We want to be careful to only evaluate the square root if the
				// norm of the angle_axis vector is greater than zero. Otherwise
				// we get a division by zero.
				const double theta = sqrt(theta2);
				const double wx = angle_axis[0] / theta;
				const double wy = angle_axis[1] / theta;
				const double wz = angle_axis[2] / theta;

				const double costheta = cos(theta);
				const double sintheta = sin(theta);

				R(0, 0) = costheta + wx * wx*(kOne - costheta);
				R(1, 0) = wz * sintheta + wx * wy*(kOne - costheta);
				R(2, 0) = -wy * sintheta + wx * wz*(kOne - costheta);
				R(0, 1) = wx * wy*(kOne - costheta) - wz * sintheta;
				R(1, 1) = costheta + wy * wy*(kOne - costheta);
				R(2, 1) = wx * sintheta + wy * wz*(kOne - costheta);
				R(0, 2) = wy * sintheta + wx * wz*(kOne - costheta);
				R(1, 2) = -wx * sintheta + wy * wz*(kOne - costheta);
				R(2, 2) = costheta + wz * wz*(kOne - costheta);
			}
			else
			{
				// Near zero, we switch to using the first order Taylor expansion.
				R(0, 0) = kOne;
				R(1, 0) = angle_axis[2];
				R(2, 0) = -angle_axis[1];
				R(0, 1) = -angle_axis[2];
				R(1, 1) = kOne;
				R(2, 1) = angle_axis[0];
				R(0, 2) = angle_axis[1];
				R(1, 2) = -angle_axis[0];
				R(2, 2) = kOne;
			}
		}

		void AngleAxisRotatePoint(const double* angle_axis, const double* pt, double* result)
		{
			const double theta2 = angle_axis[0] * angle_axis[0]
				+ angle_axis[1] * angle_axis[1]
				+ angle_axis[2] * angle_axis[2];
			if (theta2 > double(std::numeric_limits<double>::epsilon()))
			{
				// Away from zero, use the rodriguez formula
				//
				//   result = pt costheta +
				//            (w x pt) * sintheta +
				//            w (w . pt) (1 - costheta)
				//
				// We want to be careful to only evaluate the square root if the
				// norm of the angle_axis vector is greater than zero. Otherwise
				// we get a division by zero.
				//
				const double theta = sqrt(theta2);
				const double costheta = cos(theta);
				const double sintheta = sin(theta);
				const double theta_inverse = double(1.0) / theta;

				const double w[3] = { angle_axis[0] * theta_inverse,
					angle_axis[1] * theta_inverse,
					angle_axis[2] * theta_inverse };

				// Explicitly inlined evaluation of the cross product for
				// performance reasons.
				const double w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
					w[2] * pt[0] - w[0] * pt[2],
					w[0] * pt[1] - w[1] * pt[0] };
				const double tmp =
					(w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (double(1.0) - costheta);

				result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
				result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
				result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
			}
			else
			{
				// Near zero, the first order Taylor approximation of the rotation
				// matrix R corresponding to a vector w and angle w is
				//
				//   R = I + hat(w) * sin(theta)
				//
				// But sintheta ~ theta and theta * w = angle_axis, which gives us
				//
				//  R = I + hat(w)
				//
				// and actually performing multiplication with the point pt, gives us
				// R * pt = pt + w x pt.
				//
				// Switching to the Taylor expansion near zero provides meaningful
				// derivatives when evaluated using Jets.
				//
				// Explicitly inlined evaluation of the cross product for
				// performance reasons.
				const double w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
					angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
					angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };

				result[0] = pt[0] + w_cross_pt[0];
				result[1] = pt[1] + w_cross_pt[1];
				result[2] = pt[2] + w_cross_pt[2];
			}
		}

		void EulerAnglesToRotationMatrix(double rx, double ry, double rz, Eigen::Matrix3d & R)
		{
			Eigen::Matrix3d Rx, Ry, Rz;
			Ry << std::cos(ry), 0, std::sin(ry),
				0, 1.0, 0,
				-std::sin(ry), 0, std::cos(ry);

			Rx << 1.0, 0, 0,
				0, std::cos(rx), -std::sin(rx),
				0, std::sin(rx), std::cos(rx);

			Rz << std::cos(rz), -std::sin(rz), 0,
				std::sin(rz), std::cos(rz), 0,
				0, 0, 1.0;

			R = Rz * Ry * Rx;
		}

		void RotationMatrixToEulerAngles(Eigen::Matrix3d R, double & rx, double & ry, double & rz)
		{
			rx = std::atan2(-R(2, 1), R(2, 2));
			ry = std::asin(R(2, 0));
			rz = std::atan2(-R(1, 0), R(0, 0));

			if (rx != rx) { rx = 0.0; }
			if (ry != ry) { ry = 0.0; }
			if (rz != rz) { rz = 0.0; }
		}

		Eigen::Matrix3d SkewSymmetricMatrix(Eigen::Vector3d v)
		{
			Eigen::Matrix3d S;
			S << 0, -v(2), v(1),
				v(2), 0, -v(0),
				-v(1), v(0), 0;
			return S;
		}

		Eigen::Matrix3d Projection2Fundamental(Eigen::Matrix3d K1, Eigen::Matrix3d R1, Eigen::Vector3d t1,
			Eigen::Matrix3d K2, Eigen::Matrix3d R2, Eigen::Vector3d t2)
		{
			Eigen::Matrix3d R12 = R1 * R2.transpose();
			Eigen::Vector3d t12 = -R12 * t2 + t1;

			Eigen::Matrix3d t12x;
			t12x << 0, -t12(2), t12(1),
				t12(2), 0, -t12(0),
				-t12(1), t12(0), 0;

			//Eigen::Matrix3d temp1 = K1.transpose();
			return K1.transpose().inverse()*t12x*R12*K2.inverse();
		}
	}

	namespace math
	{
		void RandVectorAll(int v_min, int v_max, std::vector<int>& values)
		{
			for (size_t i = v_min; i < v_max; i++)
			{
				values.push_back(i);
			}
			std::random_shuffle(values.begin(), values.end());
		}

		void RandVectorN(int v_min, int v_max, int N, int seed, std::vector<int>& values)
		{
			//std::vector<int> values_all;
			//for (size_t i = v_min; i < v_max; i++)
			//{
			//	values_all.push_back(i);
			//}
			//std::random_shuffle(values_all.begin(), values_all.end());

			//for (size_t i = 0; i < N; i++)
			//{
			//	values.push_back(values_all[i]);
			//}
			srand(seed);

			int n = v_max - v_min;
			for (size_t i = 0; i < N; i++)
			{
				bool is_in = true;
				while (is_in)
				{
					int v = rand() % n + v_min;

					is_in = false;
					for (size_t j = 0; j < values.size(); j++) {
						if (values[j] == v) {
							is_in = true;
							break;
						}
					}
					if (!is_in) {
						values.push_back(v);
					}
				}
			}
		}

		float simOfBows(std::map<uint32_t, float>* w1, std::map<uint32_t, float>* w2)
		{
			auto w1_end = w1->end();
			auto w2_end = w2->end();

			auto w1_it = w1->begin();
			auto w2_it = w2->begin();

			double score = 0;

			while (w1_it != w1_end && w2_it != w2_end)
			{
				const auto& vi = w1_it->second;
				const auto& wi = w2_it->second;

				if (w1_it->first == w2_it->first)
				{
					score += vi * wi;

					// move v1 and v2 forward
					++w1_it;
					++w2_it;
				}
				else if (w1_it->first < w2_it->first)
				{
					// move v1 forward
					w1_it = w1->lower_bound(w2_it->first);
					// v1_it = (first element >= v2_it.id)
				}
				else
				{
					// move v2 forward
					w2_it = w2->lower_bound(w1_it->first);
					// v2_it = (first element >= v1_it.id)
				}
			}

			// ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
			//		for all i | v_i != 0 and w_i != 0 )
			// (Nister, 2006)
			if (score >= 1) // rounding errors
				score = 1.0;
			else
				score = 1.0 - sqrt(1.0 - score); // [0..1]

			return score;
		}

		float simOfBows(std::vector<int> &words1, std::vector<int> &words2)
		{
			std::vector<int> word12;
			word12.resize(words1.size() + words2.size());

			int i = 0, j = 0, k = 0;
			while (i < words1.size() && j < words2.size())
			{
				if (words1[i] < words2[j])
				{
					word12[k] = words1[i];
					i++;
				}
				else
				{
					word12[k] = words2[j];
					j++;
				}
				k++;
			}
			while (i < words1.size())
			{
				word12[k] = words1[i];
				i++;
				k++;
			}
			while (j < words2.size())
			{
				word12[k] = words2[j];
				j++;
				k++;
			}

			// unique
			int count_unique = 0;
			int v = -1;
			for (size_t i = 0; i < word12.size(); i++)
			{
				if (word12[i] != v)
				{
					v = word12[i];
					count_unique++;
				}
			}
			int count_same = words1.size() + words2.size() - count_unique;

			return float(count_same) / sqrt(float(words1.size()*words2.size()));
		}

		void keep_unique_idx_vector(std::vector<int> &data, std::vector<int> &unique_idx)
		{
			std::vector<std::pair<int, int>> data_new(data.size());
			for (size_t i = 0; i < data.size(); i++)
			{
				data_new[i].first = i;
				data_new[i].second = data[i];
			}
			std::sort(data_new.begin(), data_new.end(), [](const std::pair<int, int> &lhs, const std::pair<int, int> &rhs) { return lhs.second < rhs.second; });

			bool is_unique = true;
			int v = data_new[0].second;
			for (auto iter = data_new.begin(); iter != data_new.end(); ++iter)
			{
				if (iter->second != v)
				{
					if (is_unique)
					{
						unique_idx.push_back(iter->first);
					}
					v = iter->second;
					is_unique = true;
				}
				else
				{
					is_unique = false;
				}
			}
		}

		float vector_dot_float(float * ptr1, float * ptr2, int n)
		{
			float sim = 0.0;
			for (size_t i = 0; i < n; i++)
			{
				sim += (*ptr1++) * (*ptr2++);
			}
			return sim;
		}

		void same_in_vectors(std::vector<int>& v1, std::vector<int>& v2, std::vector<int>& vsame)
		{
			std::vector<int> v12;
			v12.resize(v1.size() + v2.size());

			int i = 0, j = 0, k = 0;
			while (i < v1.size() && j < v2.size())
			{
				if (v1[i] < v2[j])
				{
					v12[k] = v1[i];
					i++;
				}
				else
				{
					v12[k] = v2[j];
					j++;
				}
				k++;
			}
			while (i < v1.size())
			{
				v12[k] = v1[i];
				i++;
				k++;
			}
			while (j < v2.size())
			{
				v12[k] = v2[j];
				j++;
				k++;
			}

			// unique
			int v = -1;
			for (size_t i = 0; i < v12.size(); i++)
			{
				if (v12[i] != v)
				{
					v = v12[i];
				}
				else
				{
					vsame.push_back(v);
				}
			}
		}

		std::vector<int> vector_subtract(int num, std::vector<int> data2)
		{
			std::sort(data2.begin(), data2.end());

			std::vector<int> data3;
			for (int i = 0; i < num; i++)
			{
				int v1 = i;
				bool found = false;
				for (int j = 0; j < data2.size(); j++)
				{
					int v2 = data2[j];
					if (v2 == v1)
					{
						found = true;
						break;
					}
					if (v2 > v1)
					{
						found = false;
						break;
					}
				}
				if (!found)
				{
					data3.push_back(v1);
				}
			}
			return data3;
		}

		void vector_avg_denoise(std::vector<float>& data, int &count, float &result)
		{
			float avg = 0.0;
			for (size_t i = 0; i < data.size(); i++) {
				avg += data[i];
			}
			avg /= data.size();

			float sigma = 0.0;
			for (size_t i = 0; i < data.size(); i++) {
				sigma += pow(data[i] - avg, 2);
			}
			sigma = sqrt(sigma / data.size());
			if (sigma == 0) {
				result = avg;
				count = data.size();
				return;
			}

			result = 0.0;
			count = 0;
			for (size_t i = 0; i < data.size(); i++)
			{
				float t = (data[i] - avg) / sigma;
				if (t < 2.0) {
					result += data[i];
					count++;
				}
			}
			result /= count;
		}

	}

	void FindCorrespondences(std::vector<int>& v1, std::vector<int>& v2, std::vector<int>& index)
	{
		index.resize(v1.size(), -1);

		for (size_t i = 0; i < v1.size(); i++)
		{
			for (size_t j = 0; j < v2.size(); j++)
			{
				if (v1[i] == v2[j])
				{
					index[i] = j;
					break;
				}
			}

		}
	}

	void GenerateLine3D(Eigen::Vector3d pt1, Eigen::Vector3d pt2, int num, std::vector<Eigen::Vector3d>& points)
	{
		for (size_t i = 0; i < num; i++)
		{
			points.push_back((pt1*i + pt2 * (num - i)) / num);
		}
	}

	void GenerateCamera3D(Eigen::Vector3d c, std::vector<Eigen::Vector3d> axis, double f, double w, double h,
		double scale, std::vector<Eigen::Vector3d> &points)
	{
		double x0 = c(0);
		double y0 = c(1);
		double z0 = c(2);

		// the  vertexs
		double focal_w = 100 * scale;
		Eigen::Vector3d pt_central = c + focal_w * axis[2]; // z axis
		Eigen::Vector3d offset_x_half = axis[0] * w / f * focal_w / 2.0;
		Eigen::Vector3d offset_y_half = axis[1] * h / f * focal_w / 2.0;

		Eigen::Vector3d pt_lu = pt_central + offset_x_half + offset_y_half;
		Eigen::Vector3d pt_ld = pt_central + offset_x_half - offset_y_half;
		Eigen::Vector3d pt_rd = pt_central - offset_x_half - offset_y_half;
		Eigen::Vector3d pt_ru = pt_central - offset_x_half + offset_y_half;

		// draw lines
		int num_pt_line = 100;
		GenerateLine3D(c, pt_lu, num_pt_line, points);
		GenerateLine3D(c, pt_ld, num_pt_line, points);
		GenerateLine3D(c, pt_rd, num_pt_line, points);
		GenerateLine3D(c, pt_ru, num_pt_line, points);

		GenerateLine3D(pt_lu, pt_ld, num_pt_line, points);
		GenerateLine3D(pt_ld, pt_rd, num_pt_line, points);
		GenerateLine3D(pt_rd, pt_ru, num_pt_line, points);
		GenerateLine3D(pt_ru, pt_lu, num_pt_line, points);
	}

};
