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

#ifndef OBJECTSFM_CAMERA_BASIC_FUNCS_H_
#define OBJECTSFM_CAMERA_BASIC_FUNCS_H_

#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>
#include <map>  

namespace objectsfm 
{
	namespace math
	{
		template <class T>
		void median(std::vector<T>& data, T& mid)
		{
			mid = 0;
			for (size_t i = 0; i < data.size(); i++)
			{
				mid += data[i];
			}
			mid /= data.size();
		}

		template <class T>
		void sum(std::vector<T>& data, T& sum)
		{
			sum = 0;
			for (size_t i = 0; i < data.size(); i++)
			{
				sum += data[i];
			}
		}

		template <class T>
		void sum(T* data, int n, float& sum)
		{
			sum = 0.0;
			for (size_t i = 0; i < n; i++)
			{
				sum += data[i];
			}
		}

		template <class T>
		T randnormal()
		{
			T x1, x2, w;
			do
			{
				x1 = 2.0 * static_cast<T>(std::rand()) / RAND_MAX - 1.0;
				x2 = 2.0 * static_cast<T>(rand()) / RAND_MAX - 1.0;
				w = x1 * x1 + x2 * x2;
			} while (w >= 1.0 || w == 0.0);

			w = std::sqrt((-2.0 * log(w)) / w);
			return x1 * w;
		}

		// max and min
		template <class T>
		void get_max_min(T* X, int numpt, T &maxv, T &minv)
		{
			if (numpt <= 0)
				return;
			maxv = minv = X[0];

			for (int p = 1; p < numpt; p++)
			{
				if (X[p] > maxv)
					maxv = X[p];
				else if (X[p] < minv)
					minv = X[p];
			}
		}

		template <class T>
		void get_max(std::vector<T> &X, int numpt, T &maxv)
		{
			if (numpt <= 0)
				return;
			maxv = X[0];

			for (int p = 1; p < numpt; p++)
			{
				if (X[p] > maxv)
					maxv = X[p];
			}
		}

		// unique of vector
		template <class T>
		void unique_vector(std::vector<T> &data)
		{
			std::vector<T> data_new;
			std::sort(data.begin(), data.end());

			T v = data[0];
			data_new.push_back(v);
			for (std::vector<T>::iterator iter = data.begin(); iter != data.end(); ++iter)
			{
				if (*iter != v)
				{
					v = *iter;
					data_new.push_back(v);
				}
			}

			data = data_new;
		}

		template <class T>
		void keep_unique_vector(std::vector<T> &data)
		{
			std::vector<T> data_new;
			std::sort(data.begin(), data.end());

			T v = data[0];
			bool is_unique = true;
			for (std::vector<T>::iterator iter = data.begin(); iter != data.end(); ++iter)
			{
				if (*iter != v)
				{
					if (is_unique)
					{
						data_new.push_back(v);
					}
					v = *iter;
					is_unique = true;
				}
				else
				{
					is_unique = false;
				}
			}

			data = data_new;
		}

		// 
		template <class T>
		T quadratic_solution(T a, T b, T c)
		{
			T delta = std::sqrt(b*b - 4 * a*c);
			return (-b + delta) / (2 * a);
		}

		template <class T>
		void vector_dot(std::vector<T> &vec1, std::vector<T> vec2)
		{
			for (size_t i = 0; i < vec1.size(); i++)
			{
				vec1[i] = vec1[i] * vec2[i];
			}
		}

		template <class T>
		T vector_sum(std::vector<T> &vec)
		{
			T sum_value = 0;
			for (size_t i = 0; i < vec.size(); i++)
			{
				sum_value += vec[i];
			}
			return sum_value;
		}

		void RandVectorAll(int v_min, int v_max, std::vector<int> &values);

		void RandVectorN(int v_min, int v_max, int N, int seed, std::vector<int> &values);

		float simOfBows(std::map<uint32_t, float>* word1, std::map<uint32_t, float>* word2);

		float simOfBows(std::vector<int> &words1, std::vector<int> &words2);

		void keep_unique_idx_vector(std::vector<int> &data, std::vector<int> &unique_idx);

		float vector_dot_float(float* ptr1, float* ptr2, int n);

		std::vector<int> vector_subtract(int num, std::vector<int> data2);

		void vector_avg_denoise(std::vector<float> &data, int &count, float &result);

		void same_in_vectors(std::vector<int> &v1, std::vector<int> &v2, std::vector<int> &vsame);
	}

	namespace rotation 
	{
		// This algorithm comes from "Quaternion Calculus and Fast Animation",
		// Ken Shoemake, 1987 SIGGRAPH course notes
		void RotationMatrixToQuaternion(Eigen::Matrix3d R, double *quaternion);		

		void QuaternionToRotationMatrix(double *quaternion, Eigen::Matrix3d &R);

		void RotationMatrixToAngleAxis(Eigen::Matrix3d R, Eigen::Vector3d &axis);

		void QuaternionToAngleAxis(const double* quaternion, double* angle_axis);
		
		void AngleAxisToRotationMatrix(Eigen::Vector3d &angle_axis, Eigen::Matrix3d &R);
		
		void AngleAxisRotatePoint(const double* angle_axis, const double* pt, double* result);

		// R = Rz*Ry*Rx
		void EulerAnglesToRotationMatrix(double rx, double ry, double rz, Eigen::Matrix3d &R); 

		void RotationMatrixToEulerAngles(Eigen::Matrix3d R, double &rx, double &ry, double &rz);

		Eigen::Matrix3d SkewSymmetricMatrix(Eigen::Vector3d v);

		Eigen::Matrix3d Projection2Fundamental(Eigen::Matrix3d K1, Eigen::Matrix3d R1, Eigen::Vector3d t1,
			Eigen::Matrix3d K2, Eigen::Matrix3d R2, Eigen::Vector3d t2);
	}

	namespace memory
	{
		// pointer vector release
		template <class T>
		void ReleasePointerVector(std::vector<T> &ptr_vector)
		{
			for (size_t i = 0; i < ptr_vector.size(); i++)
			{
				delete ptr_vector[i];
			}
		}
	}

	// normalize image points
	//void NormalizeImagePoints(std::vector<Eigen::Vector2d> pts, std::vector<Eigen::Vector2d>& pts_norm, Eigen::Matrix3d &matrix_norm);
	void FindCorrespondences(std::vector<int> &v1, std::vector<int> &v2, std::vector<int> &index);

	void GenerateLine3D(Eigen::Vector3d pt1, Eigen::Vector3d pt2, int num, std::vector<Eigen::Vector3d> &points);

	void GenerateCamera3D(Eigen::Vector3d c, std::vector<Eigen::Vector3d> axis, double f, double w, double h,
		double scale, std::vector<Eigen::Vector3d> &points);
}
#endif //OBJECTSFM_CAMERA_BASIC_FUNCS_H_