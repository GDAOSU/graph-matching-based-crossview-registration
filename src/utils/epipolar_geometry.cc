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

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_

#include "epipolar_geometry.h"

#include <fstream>
#include <filesystem>
#include <iomanip>
#include <omp.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "utils/basic_funcs.h"
#include "utils/homography_stereo_rectify.h"

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

namespace objectsfm {

	EpipolarGeometry::EpipolarGeometry()
	{
	}

	EpipolarGeometry::~EpipolarGeometry()
	{
	}

	void EpipolarGeometry::Run(std::string fold_in, std::string fold_out, bool given_pose)
	{
		fold_img_ = fold_in + "\\visualize";
		fold_output_ = fold_out;

		// step1: read in
		SearchImagePaths();
		
		std::string file_pose = fold_in + "\\dense.out";
		ReadinSfMFile(file_pose);
		
		// step2: generate disparity for all images
		GenerateAllDisparity();

		// step3: disparity fusion
		int th_num_ray = 5;
		float th_angle_min = 3.0 / 180.0*CV_PI;
		float th_proj = 5.0;
		DisparityFusion(th_num_ray, th_angle_min, th_proj);
	}

	void EpipolarGeometry::SearchImagePaths()
	{
		std::string image_format_ = "jpg";
		std::vector<cv::String> image_paths_temp;
		cv::glob(fold_img_ + "/*." + image_format_, image_paths_temp, false);

		std::vector<int> img_names;
		for (size_t j = 0; j < image_paths_temp.size(); j++) {
			std::string path_i = std::string(image_paths_temp[j].c_str());
			int loc1 = path_i.find_last_of("\\") + 1;
			int loc2 = path_i.find_last_of(".");
			std::string name_i = path_i.substr(loc1, loc2 - loc1);
			img_names.push_back(std::stoi(name_i));
		}
		std::sort(img_names.begin(), img_names.end(), [](const int &lhs, const int &rhs) { return lhs < rhs; });

		// 
		names_.resize(img_names.size());
		for (size_t i = 0; i < img_names.size(); i++) {
			names_[i] = std::to_string(img_names[i]) + "." + image_format_;
		}

		cv::Mat temp = cv::imread(fold_img_ + "\\" + names_[0], 0);
		rows_ = temp.rows;
		cols_ = temp.cols;
	}

	void EpipolarGeometry::ReadinPoseFile(std::string sfm_file)
	{
		// in the same format as sure
		std::ifstream ff(sfm_file);
		std::string temp;
		for (size_t i = 0; i < 8; i++) {
			std::getline(ff, temp);
		}

		std::string name;
		int w, h;
		double k1, k2, k3, p1, p2;
		while (!ff.eof())
		{
			cv::Mat_<double> K(3, 3);
			cv::Mat_<double> R(3, 3);
			cv::Mat_<double> t(3, 1);

			ff >> name >> w >> h;
			for (size_t i = 0; i < 3; i++) {
				for (size_t j = 0; j < 3; j++) {
					ff >> K(i, j);
				}
			}
			ff >> k1 >> k2 >> k3 >> p1 >> p2;

			for (size_t i = 0; i < 3; i++) {
				ff >> t(i, 0);
			}

			for (size_t i = 0; i < 3; i++) {
				for (size_t j = 0; j < 3; j++) {
					ff >> R(i, j);
				}
			}

			names_.push_back(name);
			//Ks.push_back(cv::Mat(K));
			//Rs.push_back(cv::Mat(R));
			//ts.push_back(cv::Mat(t));
		}
	}

	void EpipolarGeometry::ReadinSfMFile(std::string sfm_file)
	{
		int num_imgs = names_.size();

		std::ifstream ff(sfm_file);

		// all cameras
		cams_.resize(num_imgs);
		for (size_t i = 0; i < num_imgs; i++) {
			cams_[i] = new Camera();
		}

		Eigen::Matrix3d R;
		Eigen::Vector3d t;
		int num_cams = 0;
		ff >> num_cams;
		for (size_t i = 0; i < num_cams; i++)
		{
			ff >> f_ >> k1_ >> k2_;			
			int id_img = 0;
			ff >> id_img;
			if (i == 0) {
				cam_model_ = new CameraModel(i, rows_, cols_, 0.0, f_, "lu", "lu");
			}
			cams_[id_img]->AssociateImage(id_img);
			cams_[id_img]->AssociateCamereModel(cam_model_);

			ff >> R(0, 0) >> R(0, 1) >>  R(0, 2);
			ff >> R(1, 0) >> R(1, 1) >>  R(1, 2);
			ff >> R(2, 0) >> R(2, 1) >>  R(2, 2);
			ff >> t(0) >> t(1) >> t(2);
			cams_[id_img]->SetRTPose(R, t);
			cams_[id_img]->SetID(id_img);
		}

		// key frames
		int num_key_imgs = 0;
		ff >> num_key_imgs;
		key_imgs_.resize(num_key_imgs);
		int t1, t2;
		for (size_t i = 0; i < num_key_imgs; i++) {
			ff >> key_imgs_[i] >> t1 >> t2;
		}

		// points
		int num_pts = 0;
		ff >> num_pts;
		pts_.resize(num_pts);
		double x, y, z;
		int r, g, b;
		for (size_t i = 0; i < num_pts; i++)
		{	
			pts_[i] = new Point3D();
			ff >> pts_[i]->data[0] >> pts_[i]->data[1] >> pts_[i]->data[2];
			ff >> r >> g >> b;

			int count_cams = 0;
			ff >> count_cams;

			std::map<int, Camera*> obs_cams_;
			std::map<int, Eigen::Vector2d> obs_pts2d_;
			int id_img, id_pt_global;
			float x, y;
			for (size_t j = 0; j < count_cams; j++)
			{
				ff >> id_img >> id_pt_global >> x >> y;				
				pts_[i]->AddObservation(cams_[id_img], x, y, id_pt_global, id_pt_global);
				cams_[id_img]->AddPoints(pts_[i], id_pt_global);
			}
		}
		ff.close();
	}

	std::vector<int> EpipolarGeometry::EliminateRedudentImages()
	{
		int num_img = names_.size();

		// step1: calculate the redundency of each image
		std::vector<int> pts_imgs(num_img, 0);
		std::vector<float> reduncency_imgs(num_img, 0);
		for (size_t i = 0; i < pts_.size(); i++)
		{
			int nobs = pts_[i]->cams_.size();
			for (auto iter = pts_[i]->cams_.begin(); iter != pts_[i]->cams_.end(); iter++)
			{
				int id_img = iter->second->id_img_;
				pts_imgs[id_img]++;
				reduncency_imgs[id_img] += nobs;
			}
		}

		for (size_t i = 0; i < num_img; i++) {
			if (pts_imgs[i] >0) {
				reduncency_imgs[i] /= pts_imgs[i];
			}
		}

		// step2: select keyframes according to the redundency
		std::vector<int> good_dense_frames;
		int i = 0;
		while (i < key_imgs_.size())
		{
			int id_img = key_imgs_[i];
			if (reduncency_imgs[id_img]>0)
			{
				good_dense_frames.push_back(id_img);
				i += int(reduncency_imgs[id_img] / 2 + 0.5);
			}
			else {
				i++;
				continue;
			}
			
		}

		return good_dense_frames;
	}

	void EpipolarGeometry::GenerateAllDisparity()
	{
		// 
		int shift_constant = 128;

		int n_img = names_.size();
		//for (size_t k = 0; k < n_img - 1; k++) {
		for (size_t k = 0; k < 2000; k++) {
			std::cout << "------Disparity " << k << std::endl;
			int id1 = k;
			int id2 = k + 1;
			if (!cams_[id1]->cam_model_->f_ || !cams_[id2]->cam_model_->f_) {
				continue;
			}

			cv::Mat disparity1, H1, H2;
			SGMDense(id1, id2, true, disparity1, H1, H2);  // disparity is CV_32FC1
														   
            // remove disparity boundaries
			cv::Mat dx(rows_, cols_, CV_32F, cv::Scalar(0));
			cv::Mat dy(rows_, cols_, CV_32F, cv::Scalar(0));
			cv::Sobel(disparity1, dx, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
			cv::Sobel(disparity1, dy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
			cv::Mat dd = abs(dx) + abs(dy);
			cv::Mat mask;
			cv::threshold(dd, mask, 8, 1, 0);
			mask.convertTo(mask, CV_32F);
			disparity1 = disparity1.mul(1.0 - mask);
		    
			// calculate correspondence offset x and y map
			double a1 = H1.at<double>(0, 0), a2 = H1.at<double>(0, 1), a3 = H1.at<double>(0, 2);
			double a4 = H1.at<double>(1, 0), a5 = H1.at<double>(1, 1), a6 = H1.at<double>(1, 2);
			double a7 = H1.at<double>(2, 0), a8 = H1.at<double>(2, 1), a9 = H1.at<double>(2, 2);

			cv::Mat H2_inv = H2.inv();
			double b1 = H2_inv.at<double>(0, 0), b2 = H2_inv.at<double>(0, 1), b3 = H2_inv.at<double>(0, 2);
			double b4 = H2_inv.at<double>(1, 0), b5 = H2_inv.at<double>(1, 1), b6 = H2_inv.at<double>(1, 2);
			double b7 = H2_inv.at<double>(2, 0), b8 = H2_inv.at<double>(2, 1), b9 = H2_inv.at<double>(2, 2);

			cv::Mat offsetx(rows_, cols_, CV_8UC1, cv::Scalar(0));
			cv::Mat offsety(rows_, cols_, CV_8UC1, cv::Scalar(0));
#pragma omp parallel
			for (int i = 0; i < rows_; i++)
			{
				uchar* ptrx = offsetx.ptr<uchar>(i);
				uchar* ptry = offsety.ptr<uchar>(i);
				float* ptrd = (float*)disparity1.data;
				int loc = 0;
				for (int j = 0; j < cols_; j++)
				{
					int x1 = j;
					int y1 = i;

					// get matches
					double t1 = a1 * x1 + a2 * y1 + a3;
					double t2 = a4 * x1 + a5 * y1 + a6;
					double t3 = a7 * x1 + a8 * y1 + a9;
					int px1 = t1 / t3;
					int py1 = t2 / t3;
					if (px1<0 || px1>cols_ - 1 || py1<0 || py1>rows_ - 1) {
						ptrx++; ptry++;
						continue;
					}

					int loc_now = py1 * cols_ + px1;
					ptrd += (loc_now - loc);
					loc = loc_now;
					float d = *ptrd;
					if (d == 0) {
						ptrx++; ptry++;
						continue;
					}

					int px2 = px1 - d;
					int py2 = py1;
					t1 = b1 * px2 + b2 * py2 + b3;
					t2 = b4 * px2 + b5 * py2 + b6;
					t3 = b7 * px2 + b8 * py2 + b9;
					int x2 = t1 / t3;
					int y2 = t2 / t3;
					if (x2<0 || x2>cols_ - 1 || y2<0 || y2>rows_ - 1) {
						ptrx++; ptry++;
						continue;
					}

					int offx = x2 - x1 + shift_constant;
					int offy = y2 - y1 + shift_constant;
					if (offx <0 || offx>255|| offy <0 || offy>255) {
						ptrx++; ptry++;
						continue;
					}

					*ptrx = offx;
					*ptry = offy;
					ptrx++; ptry++;
				}
			}

			// write out
			//cv::imwrite("F:\\offsetx.bmp", offsetx);
			//cv::imwrite("F:\\offsety.bmp", 10*offsety);
			std::string path = fold_output_ + "//" + std::to_string(id1) + "_denseflow";
			std::ofstream ofs(path, std::ios::binary);
			ofs.write((const char*)(offsetx.data), offsetx.elemSize() * offsetx.total());
			ofs.write((const char*)(offsety.data), offsety.elemSize() * offsety.total());
			ofs.close();
		}
	}

	void EpipolarGeometry::SGMDense(int id_left, int id_right, bool lr_shift, cv::Mat &disparity, cv::Mat &Hl, cv::Mat &Hr)
	{
		std::string file_left = fold_img_ + "\\" + names_[id_left];
		cv::Mat left = cv::imread(file_left, 0);

		std::string file_right = fold_img_ + "\\" + names_[id_right];
		cv::Mat right = cv::imread(file_right, 0);

		// generate epipolar image
		EpipolarRectification2(id_left, id_right, left, right, Hl, Hr);

		// left-rigth sift
		if (lr_shift) {
			cv::Mat Hsift = (cv::Mat_<double>(3, 3) << -1.0, 0, cols_,
				0, 1, 0,
				0, 0, 1);
			LeftRightSift(left);
			LeftRightSift(right);
			Hl = Hsift * Hl;
			Hr = Hsift * Hr;
		}

		if (0)
		{
			cv::Mat left_re, right_re;
			cv::warpPerspective(left, left_re, Hl.inv(), cv::Size(cols_, rows_));
			cv::warpPerspective(right, right_re, Hr.inv(), cv::Size(cols_, rows_));
			cv::imwrite("F:\\left_re.jpg", left_re);
			cv::imwrite("F:\\right_re.jpg", right_re);
			int a = 0;
		}
		// get disparity via sgm
		ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
		ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
		//ASSERT_MSG(options_.disp_size == 64 || options_.disp_size == 128, "disparity size must be 64 or 128.");

		int bits = 0;
		switch (left.type()) {
		case CV_8UC1: bits = 8; break;
		case CV_16UC1: bits = 16; break;
		default:
			std::cerr << "invalid input image color format" << left.type() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		//cv::resize(left, left, cv::Size(left.cols / 2, left.rows / 2));
		//cv::resize(right, right, cv::Size(right.cols / 2, right.rows / 2));

		//sgm::StereoSGM::Parameters para(10, 120, options_.uniqueness, false);
		//sgm::StereoSGM ssgm(left.cols, left.rows, options_.disp_size, bits, 8, sgm::EXECUTE_INOUT_HOST2HOST, para);

		disparity = cv::Mat(cv::Size(left.cols, left.rows), CV_8UC1);
		//ssgm.execute(left.data, right.data, disparity.data);
		disparity.convertTo(disparity, CV_32FC1);

		if (0)
		{
			std::string file_left1 = "F:\\epi_" + std::to_string(id_left) + "_" + std::to_string(id_right) + "_l.jpg";
			std::string file_right1 = "F:\\epi_" + std::to_string(id_left) + "_" + std::to_string(id_right) + "_r.jpg";
			std::string file_d = "F:\\d_" + std::to_string(id_left) + "_" + std::to_string(id_right) + ".jpg";
			cv::imwrite(file_left1, left);
			cv::imwrite(file_right1, right);
			cv::imwrite(file_d, 10 * disparity);
		}
	}

	void EpipolarGeometry::SGMDenseBlock(int id_left, int id_right, cv::Mat & disparity, cv::Mat & Hl, cv::Mat & Hr)
	{
		std::string file_left = fold_img_ + "\\" + names_[id_left];
		cv::Mat left = cv::imread(file_left, 0);

		std::string file_right = fold_img_ + "\\" + names_[id_right];
		cv::Mat right = cv::imread(file_right, 0);

		int bits = 0;
		switch (left.type()) {
		case CV_8UC1: bits = 8; break;
		case CV_16UC1: bits = 16; break;
		default:
			std::cerr << "invalid input image color format" << left.type() << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// step1: generate epipolar image
		EpipolarRectification2(id_left, id_right, left, right, Hl, Hr);

		// step2: left-rigth shift
		cv::Mat Hsift = (cv::Mat_<double>(3, 3) << -1.0, 0, cols_,
			0, 1, 0,
			0, 0, 1);
		LeftRightSift(left);
		LeftRightSift(right);
		Hl = Hsift * Hl;
		Hr = Hsift * Hr;

		// step3: disparity with different shifts
		int nstep = 4;
		int size_step = 32;
		std::vector<cv::Mat> disparities(nstep + 1);
		for (size_t i = 1; i <= nstep; i++)
		{
			cv::Mat right_i(rows_, cols_, CV_8UC1, cv::Scalar(0));
			cv::Rect roi1(cv::Point(0, 0), cv::Size(cols_ - i * size_step, rows_));
			cv::Rect roi2(cv::Point(i * size_step, 0), cv::Size(cols_ - i * size_step, rows_));
			right(roi1).copyTo(right_i(roi2));
			cv::imwrite("F:\\right" + std::to_string(i) + ".jpg", right_i);

			// sgm
			//sgm::StereoSGM::Parameters para(10, 120, options_.uniqueness, false);
			//sgm::StereoSGM ssgm(left.cols, left.rows, options_.disp_size, bits, 8, sgm::EXECUTE_INOUT_HOST2HOST, para);

			disparities[i] = cv::Mat(cv::Size(left.cols, left.rows), CV_8UC1, cv::Scalar(0));
			//ssgm.execute(left.data, right_i.data, disparities[i].data);

			cv::imwrite("F:\\disparity" + std::to_string(i) + ".jpg", disparities[i]);
			int a = 0;
		}

		// step4: mergeing disparity
		disparity = cv::Mat(cv::Size(left.cols, left.rows), CV_8UC1, cv::Scalar(0));

	}

	void EpipolarGeometry::ELASDense(int id_left, int id_right, cv::Mat & disparity, cv::Mat & Hl, cv::Mat & Hr)
	{
		std::string file_left = fold_img_ + "\\" + names_[id_left];
		cv::Mat left = cv::imread(file_left, 0);

		std::string file_right = fold_img_ + "\\" + names_[id_right];
		cv::Mat right = cv::imread(file_right, 0);

		// generate epipolar image
		EpipolarRectification2(id_left, id_right, left, right, Hl, Hr);

		// left-rigth sift
		cv::Mat Hsift = (cv::Mat_<double>(3, 3) << -1.0, 0, cols_,
			0, 1, 0,
			0, 0, 1);
		LeftRightSift(left);
		LeftRightSift(right);
		Hl = Hsift * Hl;
		Hr = Hsift * Hr;

		// get disparity via elas
		const int32_t dims[3] = { left.cols,left.rows,left.cols };
		float* d_left = (float*)malloc(left.cols*left.rows * sizeof(float));
		float* d_right = (float*)malloc(left.cols*left.rows * sizeof(float));

		//Elas::parameters param(Elas::ROBOTICS);
		//param.disp_min = 0;
		//param.disp_max = 600;
		//param.postprocess_only_left = true;

		//Elas elas(param);
		//elas.process(left.data, right.data, d_left, d_right, dims);

		//
		disparity = cv::Mat(rows_, cols_, CV_32FC1, cv::Scalar(0));
		float* ptr = (float*)disparity.data;
		for (size_t i = 0; i < rows_; i++) {
			for (size_t j = 0; j < cols_; j++) {
				*ptr++ = *d_left++;
			}
		}

		//cv::Mat B;
		//disparity.convertTo(B, CV_8UC1);
		//cv::imwrite("F:\\elas.jpg", disparity);
	}

	void EpipolarGeometry::EpipolarRectification(cv::Mat &img1, cv::Mat K1, cv::Mat R1, cv::Mat t1,
		cv::Mat &img2, cv::Mat K2, cv::Mat R2, cv::Mat t2,
		bool write_out, cv::Mat &P1_, cv::Mat &P2_)
	{
		int cols = img1.cols;
		int rows = img2.rows;
		cv::Size img_size(cols, rows);

		//
		cv::Mat R = R2* R1.inv();  // R2=R*R1, t2=R*t1+t  OpenCV::stereoCalibrate
		cv::Mat t = -R * t1 + t2;
		cv::Mat dist(5, 1, CV_64FC1, cv::Scalar(0));

		//
		cv::Mat R1_, R2_, Q_;
		cv::stereoRectify(K1, dist, K2, dist, img_size, R, t, R1_, R2_, P1_, P2_, Q_, cv::CALIB_ZERO_DISPARITY, -1, img_size);

		cv::Mat mapx1, mapy1, mapx2, mapy2;
		cv::initUndistortRectifyMap(K1, dist, R1_, P1_, img_size, CV_32FC1, mapx1, mapy1);
		cv::initUndistortRectifyMap(K2, dist, R2_, P2_, img_size, CV_32FC1, mapx2, mapy2);

		cv::Mat img_rectified1, img_rectified2;
		cv::remap(img1, img_rectified1, mapx1, mapy1, cv::INTER_LINEAR);
		cv::remap(img2, img_rectified2, mapx2, mapy2, cv::INTER_LINEAR);
		img1 = img_rectified1;
		img2 = img_rectified2;

		if (write_out)
		{
			cv::imwrite("F:\\img_rectified1.bmp", img_rectified1);
			cv::imwrite("F:\\img_rectified2.bmp", img_rectified2);
		}
	}

	void EpipolarGeometry::EpipolarRectification2(int idx1, int idx2, cv::Mat &img1, cv::Mat &img2, cv::Mat &H1, cv::Mat &H2)
	{
		// get correspondences
		Eigen::Matrix<double, 3, 3> K;
		K << f_, 0.0, cols_ / 2.0,
			0.0, f_, rows_ / 2.0,
			0.0, 0.0, 1.0;
		
		Eigen::Matrix<double, 3, 4> P1, P2;
		P1 = K * cams_[idx1]->M;
		P2 = K * cams_[idx2]->M;
		
		// find closest key frame
		float dis_best = 1000000.0;
		int idx_best = 0;
		for (size_t i = 0; i < key_imgs_.size(); i++) {
			int idx_key = key_imgs_[i];
			float dis = abs(idx1 + idx2 - 2 * idx_key);
			if (dis < dis_best) {
				dis_best = dis;
				idx_best = idx_key;
			}
		}

		std::vector<cv::Point2f> pts1, pts2;
		for (auto iter = cams_[idx_best]->pts_.begin(); iter != cams_[idx_best]->pts_.end(); iter++)
		{
			Eigen::Vector4d X(iter->second->data[0], iter->second->data[1], iter->second->data[2], 1.0);
			Eigen::Vector3d x1 = P1 * X;
			Eigen::Vector3d x2 = P2 * X;

			pts1.push_back(cv::Point2f(x1(0) / x1(2), x1(1) / x1(2)));
			pts2.push_back(cv::Point2f(x2(0) / x2(2), x2(1) / x2(2)));
		}

		// calculate F matrix
		//float th_epipolar = 1.0;
		//std::vector<uchar> status_f(pts1.size());
		//cv::Mat F = cv::findFundamentalMat(pts1, pts2, status_f, cv::FM_RANSAC, th_epipolar);
		//cv::stereoRectifyUncalibrated(pts1, pts2, F, cv::Size(cols_, rows_), H1, H2);
		
		HomoStereoRectify::Run(pts1, pts2, rows_, cols_, H1, H2);

		// warp
		cv::warpPerspective(img1, img1, H1, cv::Size(cols_, rows_));
		cv::warpPerspective(img2, img2, H2, cv::Size(cols_, rows_));
	}


	void EpipolarGeometry::DisparityFusion(int th_num_ray, float th_angle_min, float th_proj)
	{
		int step = 2;
		int span = 10;
		int shift_constant = 128;
		int rows_half = rows_ / 2;
		int cols_half = cols_ / 2;
		int num_img = names_.size();
		for (size_t i = 0; i < key_imgs_.size(); i++)
		{
			int id_key = key_imgs_[i];
			int id_end = MIN(id_key + span, num_img);
			int n_corres = id_end - id_key + 1;

			// step1: initialize 3d points
			std::vector<std::vector<cv::Point2f>> matches(n_corres);
			std::vector<bool> is_lost;
			cv::Mat flowx_cur, flowy_cur;  // CV_32F
			if (!ReadinFlow(id_key, flowx_cur, flowy_cur)) {
				continue;
			}
			float* ptrx = (float*)flowx_cur.data;
			float* ptry = (float*)flowy_cur.data;
			int loc = 0;
			for (size_t y = 0; y < rows_; y += step) {
				for (size_t x = 0; x < cols_; x += step) {
					int loc_cur = y * cols_ + x;
					ptrx += (loc_cur - loc);
					ptry += (loc_cur - loc);
					loc = loc_cur;
					if (*ptrx == -shift_constant && *ptry == -shift_constant) {
						continue;
					}

					int x2 = x + *ptrx;
					int y2 = y + *ptry;
					if (x2<0 || x2>cols_ - 1 || y2<0 || y2>rows_ - 1) {
						continue;
					}
					matches[0].push_back(cv::Point(x, y));
					matches[1].push_back(cv::Point(x2, y2));
					is_lost.push_back(false);
				}
			}
			for (size_t j = 2; j < matches.size(); j++) {
				matches[j].resize(matches[0].size());
			}
		
			for (size_t id = id_key+1; id < id_end; id++)
			{
				std::cout << id_key << " " << id << std::endl;
				int l = id - id_key;
				if (!ReadinFlow(id, flowx_cur, flowy_cur)) {
					break;
				}

				ptrx = (float*)flowx_cur.data;
				ptry = (float*)flowy_cur.data;
				loc = 0;
				for (size_t j = 0; j < matches[l].size(); j++)
				{
					if (is_lost[j]) {
						continue;
					}

					int x = matches[l][j].x;
					int y = matches[l][j].y;
					int loc_cur = y * cols_ + x;
					ptrx += (loc_cur - loc);
					ptry += (loc_cur - loc);
					loc = loc_cur;
					if (*ptrx == -shift_constant && *ptry == -shift_constant) {
						is_lost[j] = true;
						continue;
					}

					int x2 = x + *ptrx;
					int y2 = y + *ptry;
					if (x2<0 || x2>cols_ - 1 || y2<0 || y2>rows_ - 1) {
						is_lost[j] = true;
						continue;
					}
					matches[l + 1][j] = cv::Point(x2, y2);
				}
			}

			// do triangulation
			std::vector<cv::Point3d> pts3d(matches[0].size());
			std::vector<cv::Scalar> colors(matches[0].size());
			std::vector<int> id_good_pt((matches[0].size()));
			cv::Mat img_rgb = cv::imread(fold_img_ + "\\" + names_[id_key]);
#pragma omp parallel
			for (int m = 0; m < matches[0].size(); m++) {
				Point3D pt_j;
				for (int n = 0; n < matches.size(); n++) {
					if (matches[n][m].x != 0 && matches[n][m].y != 0) {
						pt_j.AddObservation(cams_[id_key + n], matches[n][m].x-cols_half, matches[n][m].y-rows_half, n, n);
					}
				}
				if (pt_j.pts2d_.size() < th_num_ray) {
					continue;
				}
				//if (abs(matches[0][m].x - 399)<3 && abs(matches[0][m].y - 817)<3)
				//{
				//	std::cout << matches[0][m] << std::endl;
				//	std::cout << matches[1][m] << std::endl;
				//	std::cout << matches[2][m] << std::endl;
				//	std::cout << matches[3][m] << std::endl;
				//}
				if (pt_j.Trianglate3(th_proj, th_angle_min)) {
					pts3d[m] = cv::Point3d(pt_j.data[0], pt_j.data[1], pt_j.data[2]);					
					uchar* ptr_color = (uchar*)img_rgb.data + 3 * int(matches[0][m].y*cols_ + matches[0][m].x);
					colors[m] = cv::Scalar(ptr_color[2], ptr_color[1], ptr_color[0]);
					id_good_pt[m] = 1;
				}
			}

			// write out
			std::string file_out = fold_output_ + "\\" + names_[id_key].substr(0, names_[id_key].size() - 4) + ".txt";
			std::ofstream ff(file_out);
			//ff << std::fixed << std::setprecision(8);
			for (size_t j = 0; j < pts3d.size(); j++) {
				if (id_good_pt[j]) {
					ff << pts3d[j].x << " " << pts3d[j].y << " " << pts3d[j].z << " ";
					ff << int(colors[j].val[0]) << " " << int(colors[j].val[1]) << " " << int(colors[j].val[2]) << std::endl;
				}
			}
			ff.close();
		}
	}

	void EpipolarGeometry::DisparityFusion(int id1, 
		std::vector<int> id2_list, std::vector<cv::Mat>& disparity_list,  std::vector<cv::Mat> H_list,
		int th_num, float th_angle, float th_proj, int step_out, std::string file_out)
	{
		int cols = disparity_list[0].cols;
		int rows = disparity_list[0].rows;

		cv::Mat K = (cv::Mat_<double>(3, 3) << f_, 0, cols / 2,
			0, f_, rows / 2,
			0, 0, 1);

		cv::Mat M = (cv::Mat_<double>(3, 4) << cams_[id1]->M(0, 0), cams_[id1]->M(0, 1), cams_[id1]->M(0, 2), cams_[id1]->M(0, 3),
			                                   cams_[id1]->M(1, 0), cams_[id1]->M(1, 1), cams_[id1]->M(1, 2), cams_[id1]->M(1, 3),
			                                   cams_[id1]->M(2, 0), cams_[id1]->M(2, 1), cams_[id1]->M(2, 2), cams_[id1]->M(2, 3));
		cv::Mat P = K * M;

		float xc = cams_[id1]->pos_ac_.c(0);
		float yc = cams_[id1]->pos_ac_.c(1);
		float zc = cams_[id1]->pos_ac_.c(2);

		step_out = 4;

		//
		std::vector<cv::Mat> H_inv_list(H_list.size());
		for (size_t i = 0; i < H_list.size(); i++){
			H_inv_list[i] = H_list[i].inv();
		}

		std::vector<float*> ptr_list(disparity_list.size());
		std::vector<int> loc_list(disparity_list.size());
		for (size_t i = 0; i < disparity_list.size(); i++)
		{
			ptr_list[i] = (float*)disparity_list[i].data;
			loc_list[i] = 0;
		}

		// generate point cloud
		std::vector<cv::Point3d> pts3d;
		std::vector<cv::Scalar> colors;
		cv::Mat img_rgb = cv::imread(fold_img_ + "\\" + names_[id1]);
		uchar* ptr_rgb = img_rgb.data;
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				cv::Mat x_src = (cv::Mat_<double>(3, 1) << j, i, 1.0);

				// get matches
				std::vector<cv::Point2f> matches(disparity_list.size());
				for (size_t m = 0; m < disparity_list.size(); m++)
				{
					cv::Mat x1 = H_list[2 * m] * x_src;
					int px = x1.at<double>(0) / x1.at<double>(2);
					int py = x1.at<double>(1) / x1.at<double>(2);
					if (px<0 || px>cols - 1 || py<0 || py>rows - 1) {
						continue;
					}
					int loc = py * cols + px;
					ptr_list[m] += (loc - loc_list[m]);
					loc_list[m] = loc;
					float d = *(ptr_list[m]);
					if (d == 0) {
						continue;
					}

					cv::Mat x2 = (cv::Mat_<double>(3, 1) << px - d, py, 1.0);
					cv::Mat x_dst = H_inv_list[2 * m + 1] * x2;
					float xx = x_dst.at<double>(0) / x_dst.at<double>(2);
					float yy = x_dst.at<double>(1) / x_dst.at<double>(2);
					if (xx<0 || xx>cols - 1 || yy<0 || yy>rows - 1) {
						continue;
					}

					matches[m] = cv::Point2f(xx, yy);
				}

				// triangulation
				int count_num = 0;
				for (size_t m = 0; m < matches.size(); m++) {
					if (!(matches[m].x == 0 && matches[m].y == 0)) {
						count_num++;
					}
				}

				if (count_num >= th_num)
				{
					Point3D pt_temp;
					pt_temp.AddObservation(cams_[id1], double(j) - cols / 2, double(i) - rows / 2, 0, 0);
					for (size_t m = 0; m < matches.size(); m++) {
						if (!(matches[m].x == 0 && matches[m].y == 0)) {
							int id_cam = id2_list[m];
							pt_temp.AddObservation(cams_[id_cam], matches[m].x - cols / 2, matches[m].y - rows / 2, m + 1, m + 1);
						}
						
					}
					if (pt_temp.Trianglate3(th_proj, th_angle)) {
						pts3d.push_back(cv::Point3d(pt_temp.data[0], pt_temp.data[1], pt_temp.data[2]));
						colors.push_back(cv::Scalar(ptr_rgb[0], ptr_rgb[1], ptr_rgb[2]));
					}
				}
				ptr_rgb += 3;
			}

		}

		// write out
		std::ofstream ff(file_out);
		//ff << std::fixed << std::setprecision(8);
		for (size_t i = 0; i < pts3d.size(); i++)
		{
			ff << pts3d[i].x << " " << pts3d[i].y << " " << pts3d[i].z << " ";
			ff << int(colors[i].val[2]) << " " << int(colors[i].val[1]) << " " << int(colors[i].val[0]) << std::endl;
		}
		ff.close();
	}

	void EpipolarGeometry::SavePoseFile(std::string sfm_file)
	{
		//
		FILE * fp;
		fp = fopen(sfm_file.c_str(), "w+");
		fprintf(fp, "%s\n", "fileName imageWidth imageHeight");
		fprintf(fp, "%s\n", "camera matrix K [3x3]");
		fprintf(fp, "%s\n", "radial distortion [3x1]");
		fprintf(fp, "%s\n", "tangential distortion [2x1]");
		fprintf(fp, "%s\n", "camera position t [3x1]");
		fprintf(fp, "%s\n", "camera rotation R [3x3]");
		fprintf(fp, "%s\n\n", "camera model P = K [R|-Rt] X");


		//for (size_t i = 1; i < Rs_new.size(); i++)
		//{
		//	fprintf(fp, "%s %d %d\n", names[i], cols, rows);
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", Ks[i].at<double>(0, 0), 0, Ks[i].at<double>(0, 2));
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", 0, Ks[i].at<double>(1, 1), Ks[i].at<double>(1, 2));
		//	fprintf(fp, "%d %d %d\n", 0, 0, 1);
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", 0, 0, 0);
		//	fprintf(fp, "%.8lf %.8lf\n", 0, 0);
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", ts_new[i].at<double>(0, 0), ts_new[i].at<double>(1, 0), ts_new[i].at<double>(2, 0));
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", Rs_new[i].at<double>(0, 0), Rs_new[i].at<double>(0, 1), Rs_new[i].at<double>(0, 2));
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", Rs_new[i].at<double>(1, 0), Rs_new[i].at<double>(1, 1), Rs_new[i].at<double>(1, 2));
		//	fprintf(fp, "%.8lf %.8lf %.8lf\n", Rs_new[i].at<double>(2, 0), Rs_new[i].at<double>(2, 1), Rs_new[i].at<double>(2, 2));
		//}

		//fclose(fp);
	}

	void EpipolarGeometry::Depth2Points(std::string fold)
	{
		double scale = 20.0;
		double focal = 900.95001200;

		std::string fold_point = fold + "\\pointcloud";
		if (!std::experimental::filesystem::exists(fold_point)) {
			std::experimental::filesystem::create_directory(fold_point);
		}

		// 
		std::string fold_rgb = fold + "\\visualize";
		std::string fold_depth = fold + "\\depth";
		std::string fold_txt = fold + "\\txt";

		std::vector<cv::String> img_paths;
		cv::glob(fold_rgb + "/*.jpg", img_paths, false);
		for (size_t i = 0; i < img_paths.size(); i++)
		{
			std::string path_i = std::string(img_paths[i].c_str());
			int idxs = path_i.find_last_of("\\");
			int idxe = path_i.find_last_of(".");
			std::string name_i = path_i.substr(idxs + 1, idxe - idxs - 1);

			std::string path_rgb = fold_rgb + "\\" + name_i + ".jpg";
			std::string path_depth = fold_depth + "\\" + name_i + ".png";
			std::string path_txt = fold_txt + "\\" + name_i + ".txt";

			cv::Mat depth = cv::imread(path_depth, -1);
			//cv::Mat rgb = cv::imread(path_rgb);

			std::ifstream ff(path_txt);
			cv::Mat R = cv::Mat(3, 3, CV_64FC1);
			cv::Mat t = cv::Mat(3, 1, CV_64FC1);
			for (size_t m = 0; m < 3; m++)
			{
				for (size_t n = 0; n < 3; n++)
				{
					double v = 0;
					ff >> v;
					*((double*)R.data + m * 3 + n) = v;
				}
			}
			for (size_t m = 0; m < 3; m++)
			{
				double v = 0;
				ff >> v;
				*((double*)t.data + m) = v;
			}
			ff.close();

			//
			double px = depth.cols / 2;
			double py = depth.rows / 2;
			short* ptr_depth = (short*)depth.data;
			//uchar* ptr_rgb = rgb.data;
			std::vector<cv::Point3d> pts;
			std::vector<cv::Point3d> colors;
			for (size_t m = 0; m < depth.rows; m++)
			{
				for (size_t n = 0; n < depth.cols; n++)
				{
					if (*ptr_depth != 0)
					{
						double z = *ptr_depth / scale;
						double x = z * (n - px) / focal;
						double y = z * (m - py) / focal;
						cv::Mat Xc = (cv::Mat_<double>(3, 1) << x, y, z);
						cv::Mat Xw = R.inv()*(Xc - t);

						pts.push_back(cv::Point3d(Xw.at<double>(0, 0), Xw.at<double>(1, 0), Xw.at<double>(2, 0)));
						//colors.push_back(cv::Point3d(ptr_rgb[0], ptr_rgb[1], ptr_rgb[2]));
						colors.push_back(cv::Point3d(0, 0, 0));
					}
					ptr_depth++;
					//ptr_rgb++;
				}
			}

			// write out
			std::string path_point = fold_point + "\\" + name_i + ".txt";
			std::ofstream of(path_point);
			for (size_t m = 0; m < pts.size(); m += 2)
			{
				of << pts[m].x << " " << pts[m].y << " " << pts[m].z << " ";
				of << colors[m].x << " " << colors[m].y << " " << colors[m].z << std::endl;
			}
			of.close();
		}
	}

	void EpipolarGeometry::LeftRightSift(cv::Mat & img)
	{
		int cols = img.cols;
		int rows = img.rows;
		cv::Mat img_new(rows, cols, CV_8U);
		for (size_t i = 0; i < rows; i++)
		{
			uchar* ptr = img.ptr<uchar>(i);
			uchar* ptr_new = img_new.ptr<uchar>(i) + cols;
			for (size_t j = 0; j < cols; j++) {
				*ptr_new-- = *ptr++;
			}
		}
		img = img_new.clone();
	}

	bool EpipolarGeometry::ReadinFlow(int idx, cv::Mat & flowx, cv::Mat & flowy)
	{
		std::string path = fold_output_ + "//" + std::to_string(idx) + "_denseflow";
		if (!std::experimental::filesystem::exists(path)) {
			return false;
		}

		int h = rows_;
		int w = cols_;
		int v_offset = 128;
		flowx = cv::Mat(h, w, CV_8U, cv::Scalar(0));
		flowy = cv::Mat(h, w, CV_8U, cv::Scalar(0));

		std::ifstream ifs(path, std::ios::binary);
		ifs.read((char*)(flowx.data), flowx.elemSize() * flowx.total());
		ifs.read((char*)(flowy.data), flowy.elemSize() * flowy.total());
		ifs.close();

		flowx.convertTo(flowx, CV_32F);
		flowy.convertTo(flowy, CV_32F);
		flowx -= v_offset;
		flowy -= v_offset;

		return true;
	}

}  // namespace objectsfm
