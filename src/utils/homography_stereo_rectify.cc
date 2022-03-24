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

#include "homography_stereo_rectify.h"

#include "homography_stereo_rectify_util.h"

namespace objectsfm
{
	HomoStereoRectify::HomoStereoRectify()
	{
	}

	HomoStereoRectify::~HomoStereoRectify()
	{
	}


	void HomoStereoRectify::Run(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, int rows, int cols, cv::Mat & H1, cv::Mat & H2)
	{
		// step1: estimate F
		float th_epipolar = 1.0;
		std::vector<uchar> status_f(pts1.size());
		cv::Mat F = cv::findFundamentalMat(pts1, pts2, status_f, cv::FM_RANSAC, th_epipolar);

		//  step2: calculate the epipole, which is the left-null vector of F
		cv::Mat e_mat;
		cv::SVD::solveZ(F, e_mat);
		
		Mat A, B, Ap, Bp;
		Mat e_x = crossProductMatrix(e_mat);

		/****************** PROJECTIVE **************************/

		// Get A,B matrix for minimizing z
		obtainAB(rows, cols, e_x, A, B);
		obtainAB(rows, cols, F, Ap, Bp);

		// Get initial guess for z
		Vec3d z = getInitialGuess(A, B, Ap, Bp);

		// Optimizes the z solution
		optimizeRoot(A, B, Ap, Bp, z);

		// Get w
		Mat w = e_x * Mat(z);
		Mat wp = F * Mat(z);

		w /= w.at<double>(2, 0);
		wp /= wp.at<double>(2, 0);

		// Get final H_p and Hp_p matrix for projection
		Mat H_p = Mat::eye(3, 3, CV_64F);
		H_p.at<double>(2, 0) = w.at<double>(0, 0);
		H_p.at<double>(2, 1) = w.at<double>(1, 0);

		Mat Hp_p = Mat::eye(3, 3, CV_64F);
		Hp_p.at<double>(2, 0) = wp.at<double>(0, 0);
		Hp_p.at<double>(2, 1) = wp.at<double>(1, 0);

		/****************** SIMILARITY **************************/

		// Get the translation term
		double vp_c = getTranslationTerm(rows, cols, H_p, Hp_p);

		// Get the H_r and Hp_r matrix directly
		Mat H_r = Mat::zeros(3, 3, CV_64F);

		H_r.at<double>(0, 0) = F.at<double>(2, 1) - w.at<double>(1, 0) * F.at<double>(2, 2);
		H_r.at<double>(1, 0) = F.at<double>(2, 0) - w.at<double>(0, 0) * F.at<double>(2, 2);

		H_r.at<double>(0, 1) = w.at<double>(0, 0) * F.at<double>(2, 2) - F.at<double>(2, 0);
		H_r.at<double>(1, 1) = H_r.at<double>(0, 0);

		H_r.at<double>(1, 2) = F.at<double>(2, 2) + vp_c;
		H_r.at<double>(2, 2) = 1.0;

		Mat Hp_r = Mat::zeros(3, 3, CV_64F);

		Hp_r.at<double>(0, 0) = wp.at<double>(1, 0) * F.at<double>(2, 2) - F.at<double>(1, 2);
		Hp_r.at<double>(1, 0) = wp.at<double>(0, 0) * F.at<double>(2, 2) - F.at<double>(0, 2);

		Hp_r.at<double>(0, 1) = F.at<double>(0, 2) - wp.at<double>(0, 0) * F.at<double>(2, 2);
		Hp_r.at<double>(1, 1) = Hp_r.at<double>(0, 0);

		Hp_r.at<double>(1, 2) = vp_c;
		Hp_r.at<double>(2, 2) = 1.0;

		/******************* SHEARING ***************************/
		Mat H_1 = H_r * H_p;
		Mat H_2 = Hp_r * Hp_p;

		Mat H_s, Hp_s;

		// Get shearing transforms with the method described on the paper
		getShearingTransforms(rows, cols, H_1, H_2, H_s, Hp_s);

		/****************** RECTIFY IMAGES **********************/
		Mat H = H_s * H_r * H_p;
		Mat Hp = Hp_s * Hp_r * Hp_p;

		H1 = H;
		H2 = Hp;

		// shift to center
		vector<Point2d> corners_all(4), corners_all_1(4), corners_all_2(4);
		corners_all[0] = Point2d(0, 0);
		corners_all[1] = Point2d(cols, 0);
		corners_all[2] = Point2d(cols, rows);
		corners_all[3] = Point2d(0, rows);
		perspectiveTransform(corners_all, corners_all_1, H1);
		perspectiveTransform(corners_all, corners_all_2, H2);

		cv::Point2d center1(0, 0), center2(0, 0);
		for (size_t i = 0; i < 4; i++) {
			center1 += corners_all_1[i];
			center2 += corners_all_2[i];
		}
		center1 *= 1.0 / 4.0;
		center2 *= 1.0 / 4.0;

		double shiftx1 = cols / 2 - center1.x;
		double shiftx2 = cols / 2 - center1.x;
		double shifty = rows / 2 - (center1.y + center2.y) / 2;

		cv::Mat H1shift = (cv::Mat_<double>(3,3) << 1, 0, shiftx1,
			0, 1, shifty,
			0, 0, 1);
		cv::Mat H2shift = (cv::Mat_<double>(3, 3) << 1, 0, shiftx2,
			0, 1, shifty,
			0, 0, 1);
		H1 = H1shift * H1;
		H2 = H2shift * H2;
	}
};
