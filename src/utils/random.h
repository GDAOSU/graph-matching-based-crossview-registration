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

#ifndef OBJECTSFM_UITL_RANDOM_H_
#define OBJECTSFM_UITL_RANDOM_H_

#include <Eigen/Core>
#include <random>

namespace objectsfm {

	// A wrapper around the c++11 random generator utilities. This allows for a
	// thread-safe random number generator that may be easily instantiated and
	// passed around as an object.
	class RandomNumberGenerator {
	public:
		// Creates the random number generator using the current time as the seed.
		RandomNumberGenerator();

		// Creates the random number generator using the given seed.
		explicit RandomNumberGenerator(const unsigned seed);

		// Seeds the random number generator with the given value.
		void Seed(const unsigned seed);

		// Get a random double between lower and upper (inclusive).
		double RandDouble(const double lower, const double upper);

		// Get a random float between lower and upper (inclusive).
		float RandFloat(const float lower, const float upper);

		// Get a random double between lower and upper (inclusive).
		int RandInt(const int lower, const int upper);

		// Generate a number drawn from a gaussian distribution.
		double RandGaussian(const double mean, const double std_dev);

		// Return eigen types with random initialization. These are just convenience
		// methods. Methods without min and max assign random values between -1 and 1
		// just like the Eigen::Random function.
		Eigen::Vector2d RandVector2d(const double min, const double max);
		Eigen::Vector2d RandVector2d();
		Eigen::Vector3d RandVector3d(const double min, const double max);
		Eigen::Vector3d RandVector3d();
		Eigen::Vector4d RandVector4d(const double min, const double max);
		Eigen::Vector4d RandVector4d();

		inline double Rand(const double lower, const double upper) {
			return RandDouble(lower, upper);
		}

		inline float Rand(const float lower, const float upper) {
			return RandFloat(lower, upper);
		}

		// Sets an Eigen type with random values between -1.0 and 1.0. This is meant
		// to replace the Eigen::Random() functionality.
		template <typename Derived>
		void SetRandom(Eigen::MatrixBase<Derived>* b) {
			for (int r = 0; r < b->rows(); r++) {
				for (int c = 0; c < b->cols(); c++) {
					(*b)(r, c) = Rand(-1.0, 1.0);
				}
			}
		}

	};

}  // namespace objectsfm

#endif  //
