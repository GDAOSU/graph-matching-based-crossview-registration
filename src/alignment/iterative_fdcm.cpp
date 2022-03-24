#include "iterative_fdcm.h"

#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "common_functions.h"  // for normal calculation, cv::flann + pca is not stable

IterativeFDCM::IterativeFDCM(void)
{
}

IterativeFDCM::~IterativeFDCM(void)
{
}

void IterativeFDCM::Run(std::vector<cv::Point2d> pts_src, std::vector<double> weight, std::vector<cv::Vec4f> lines_dst,
	std::vector<cv::Vec4f> hyps_coarse, int rows, int cols, double th_offset, std::vector<cv::Vec6f> &hyps_refined)
{
	// step1: convert line into edge map
	cv::Mat edge_map = cv::Mat::ones(rows, cols, CV_8UC1) * 255;
	std::vector<cv::Point2d> line_centers;
	std::vector<double> line_angles;
	std::vector<double> line_lens;
	for (int i = 0; i < lines_dst.size(); i++)
	{
		double xs = lines_dst[i].val[0];
		double ys = lines_dst[i].val[1];
		double xe = lines_dst[i].val[2];
		double ye = lines_dst[i].val[3];
		double cx = (xs + xe) / 2;
		double cy = (ys + ye) / 2;
		if (cx<0 || cx>cols - 1 || cy<0 || cy>rows - 1) {
			continue;
		}

		double dx = xs - xe;
		double dy = ys - ye;
		double len = sqrt(dx * dx + dy * dy);
		double k1 = (len - 2.0) / len;
		double k2 = 2.0 / len;

		// to avoid line connection refer to: opencv distanceTransform
		cv::line(edge_map, cv::Point(k1*xs + k2 * xe, k1*ys + k2 * ye), cv::Point(k2*xs + k1 * xe, k2*ys + k1 * ye), cv::Scalar(0));
		line_centers.push_back(cv::Point2d(cx, cy));
		line_angles.push_back(atan(dy / (dx + 0.00001)));
		line_lens.push_back(len);
	}
	//cv::imwrite("F:\\edge_map.bmp", edge_map);

	// step2: distance transformation
	cv::Mat dist_map, labels_map;
	cv::distanceTransform(edge_map, dist_map, labels_map, CV_DIST_L2, CV_DIST_MASK_5, cv::DIST_LABEL_CCOMP);
	
	/*cv::Mat labels_map_new;
	labels_map.convertTo(labels_map_new, CV_8U, 1, 0);
	cv::imwrite("F:\\labels_map_new.bmp", labels_map_new);*/
	
	std::vector<int> label_id_info(100000);
	for (int i = 0; i < line_centers.size(); i++) {
		int label = labels_map.at<int>(line_centers[i].y, line_centers[i].x);
		label_id_info[label] = i;
	}
	cv::Mat dirc_map(rows, cols, CV_32FC1);
	float* ptr_dirc = (float*)dirc_map.data;
	int* ptr_label = (int*)labels_map.data;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int label = *ptr_label;
			int id_line = label_id_info[label];
			*ptr_dirc = line_angles[id_line];

			ptr_label++; ptr_dirc++;
		}
	}

	// step3: calculate the gradient of the distance map
	cv::Mat dist_gx, dist_gy;
	cv::Sobel(dist_map, dist_gx, dist_map.type(), 1, 0, 3);
	cv::Sobel(dist_map, dist_gy, dist_map.type(), 0, 1, 3);
	dist_gx /= 4.0;
	dist_gy /= 4.0;

	// step4: calculate pts normal
	std::vector<cv::Point2d> pts_normals;
	std::vector<double> pts_angles;
	//PtNormal(pts_src, 20, pts_normals, pts_angles);
	PCAFunctions::PCA2d(pts_src, 20, pts_normals, pts_angles);
	//std::cout << pts_angles[0] << " " << pts_normals[0] << " " << pts_src[0] << std::endl;
	//std::cout << pts_angles[100] << " " << pts_normals[100] << " " << pts_src[100] << std::endl;

	//cv::Mat img_pts;
	//ShowResult(pts_src, 0.3, img_pts);
	//cv::imwrite("F:\\img_pts.bmp", img_pts);

	// step5: traverse each hypothesis
	std::vector<double> scores(hyps_coarse.size());
	std::vector<cv::Point2d> offsets(hyps_coarse.size());
#pragma omp parallel for
	for (int i = 0; i < hyps_coarse.size(); i++) {
		if (i%100000==0) {
			std::cout << i << std::endl;
		}
		

		double scale_i = hyps_coarse[i].val[0];
		double angle_i = hyps_coarse[i].val[1];
		double tx_i = hyps_coarse[i].val[2];
		double ty_i = hyps_coarse[i].val[3];

		double vcos = cos(angle_i);
		double vsin = sin(angle_i);

		// convert the pts
		std::vector<cv::Point2d> pts_cur(pts_src.size());
		std::vector<double> angle_cur = pts_angles;
		for (int m = 0; m < pts_src.size(); m++) {
			pts_cur[m].x = scale_i * (vcos * pts_src[m].x - vsin * pts_src[m].y) + tx_i;
			pts_cur[m].y = scale_i * (vsin * pts_src[m].x + vcos * pts_src[m].y) + ty_i;
			angle_cur[m] += angle_i;
		}

		// iterative FDCM
		MatchIterative(pts_cur, weight, angle_cur, th_offset, dist_map, dist_gx, dist_gy, dirc_map, scores[i], offsets[i]);
	}

	// update hyps
	hyps_refined.resize(hyps_coarse.size());
	for (int i = 0; i < hyps_coarse.size(); i++) {
		hyps_refined[i].val[0] = scores[i];
		hyps_refined[i].val[1] = hyps_coarse[i].val[0];
		hyps_refined[i].val[2] = hyps_coarse[i].val[1];
		hyps_refined[i].val[3] = hyps_coarse[i].val[2] + offsets[i].x;
		hyps_refined[i].val[4] = hyps_coarse[i].val[3] + offsets[i].y;
		//hyps_refined[i].val[3] = hyps_coarse[i].val[2];
		//hyps_refined[i].val[4] = hyps_coarse[i].val[3];
	}

	//// find out the best score
	//double score_best = 1000000000.0;
	//int idx_best = 0;
	//for (int i = 0; i < trans_hyps.size(); i++) {
	//	double score_weighted = trans_hyps[i].val[0] / (1.0 + trans_hyps[i].val[1]);
	//	//double score_weighted = trans_hyps[i].val[0];
	//	if (score_weighted<score_best) {
	//		score_best = score_weighted;
	//		idx_best = i;
	//	}
	//}
	//std::cout << trans_hyps[idx_best] << std::endl;

	//double vcos = cos(trans_hyps[idx_best].val[2]);
	//double vsin = sin(trans_hyps[idx_best].val[2]);
	//std::vector<cv::Point2d> pts_final(pts_src.size());
	//for (int m = 0; m < pts_src.size(); m++) {
	//	pts_final[m].x = trans_hyps[idx_best].val[1] * (vcos * pts_src[m].x - vsin * pts_src[m].y) + trans_hyps[idx_best].val[3];
	//	pts_final[m].y = trans_hyps[idx_best].val[1] * (vsin * pts_src[m].x + vcos * pts_src[m].y) + trans_hyps[idx_best].val[4];
	//}
	//cv::Mat img_show_final = edge_map.clone();
	//ShowResult(pts_final, img_show_final);
	//cv::imwrite("F:\\img_show_final.bmp", img_show_final);
	//int a = 0;
}

void IterativeFDCM::GetDistMap(std::vector<cv::Vec4f> lines_dst, cv::Mat & dist_map, cv::Mat &mask_map)
{
	int rows = dist_map.rows;
	int cols = dist_map.cols;

	// step1: convert line into edge map
	cv::Mat edge_map = cv::Mat::ones(rows, cols, CV_8UC1) * 255;
	std::vector<cv::Point2d> line_centers;
	std::vector<double> line_angles;
	std::vector<double> line_lens;
	for (int i = 0; i < lines_dst.size(); i++)
	{
		double xs = lines_dst[i].val[0];
		double ys = lines_dst[i].val[1];
		double xe = lines_dst[i].val[2];
		double ye = lines_dst[i].val[3];
		double cx = (xs + xe) / 2;
		double cy = (ys + ye) / 2;
		if (cx<0 || cx>cols - 1 || cy<0 || cy>rows - 1) {
			continue;
		}

		double dx = xs - xe;
		double dy = ys - ye;
		double len = sqrt(dx * dx + dy * dy);
		double k1 = (len - 2.0) / len;
		double k2 = 2.0 / len;

		// to avoid line connection refer to: opencv distanceTransform
		cv::line(edge_map, cv::Point(k1*xs + k2 * xe, k1*ys + k2 * ye), cv::Point(k2*xs + k1 * xe, k2*ys + k1 * ye), cv::Scalar(0));
		line_centers.push_back(cv::Point2d(cx, cy));
		line_angles.push_back(atan(dy / (dx + 0.00001)));
		line_lens.push_back(len);
	}
	mask_map = 255 - edge_map;
	cv::dilate(mask_map, mask_map, cv::Mat(9, 9, mask_map.type()));

	//cv::imwrite("F:\\edge_map.bmp", edge_map);

	// step2: distance transformation
	cv::Mat labels_map;
	cv::distanceTransform(edge_map, dist_map, labels_map, CV_DIST_L2, CV_DIST_MASK_5, cv::DIST_LABEL_CCOMP);

}

void IterativeFDCM::MatchIterative(std::vector<cv::Point2d> pts, std::vector<double> weight, std::vector<double> pt_angles, double th_offset,
	cv::Mat &dist_map, cv::Mat &dist_gx, cv::Mat &dist_gy, cv::Mat &angle_map,
	double &score, cv::Point2d &offset)
{
	score = 0.0;
	offset = cv::Point2d(0, 0);

	double lambda = 10.0;
	int rows = dist_map.rows;
	int cols = dist_map.cols;
	int panelty = 100;

	float* ptr_dist = (float*)dist_map.data;
	float* ptr_gx = (float*)dist_gx.data;
	float* ptr_gy = (float*)dist_gy.data;
	float* ptr_angle = (float*)angle_map.data;
	int loc_pre = 0;
	double score_pre = 10000000000.0;
	double score_cur = 1000000000.0;	

	double count_weighted = 0;
	double offset_dis = 0;
	while (score_pre - score_cur > score_pre * 1e-3 && offset_dis<th_offset) {
		score_pre = score_cur;
		score_cur = 0;

		// step1: find the score and gradient
		std::vector<double> scores(pts.size(), -1);
		std::vector<double> gx(pts.size(), 0);
		std::vector<double> gy(pts.size(), 0);
		for (int i = 0; i < pts.size(); i++) {
			int x = pts[i].x;
			int y = pts[i].y;
			if (x<0 || x>cols - 1 || y<0 || y>rows - 1) {
				continue;
			}
			int loc_cur = y * cols + x;
			int loc_dev = loc_cur - loc_pre;
			loc_pre = loc_cur;
			ptr_dist += loc_dev;
			ptr_gx += loc_dev;
			ptr_gy += loc_dev;
			ptr_angle += loc_dev;

			double dev_angle = abs(*ptr_angle - pt_angles[i]);
			dev_angle = abs(dev_angle - int(dev_angle / CV_PI + 0.5)*CV_PI);

			scores[i] = *ptr_dist + lambda * dev_angle;
			gx[i] = *ptr_gx;
			gy[i] = *ptr_gy;
		}

		// step2: update the points
		count_weighted = 0;
		double score_avg = 0.0;
		double gx_avg = 0.0, gy_avg = 0.0;
		for (int i = 0; i < pts.size(); i++) {
			if (scores[i]>0 && scores[i]<10) {
				score_avg += scores[i] * weight[i];
				gx_avg += gx[i] * weight[i];
				gy_avg += gy[i] * weight[i];
				count_weighted += weight[i];

				score_cur += scores[i];
			}
			else {
				score_cur += panelty;
			}			
		}

		// we aim to reduce scores into zero, so the step can be score_cur/2
		score_avg /= count_weighted;
		gx_avg /= count_weighted;
		gy_avg /= count_weighted;
		double step = score_avg / 2;
		cv::Point2d dxy(-gx_avg * step, -gy_avg * step);
		for (int i = 0; i < pts.size(); i++) {
			pts[i] += dxy;
		}

		offset += dxy;
		offset_dis = sqrt(offset.x*offset.x + offset.y*offset.y);

		//std::cout << score_pre << " " << score_cur << std::endl;
	}

	score = count_weighted;
}

void IterativeFDCM::MatchIterative2(std::vector<int> pts, int xmax, int ymax, 
	std::vector<double> weight, cv::Mat mask_map, ::Mat &dist_map, int winsize, cv::Mat &score_map)
{
	int rows = dist_map.rows;
	int cols = dist_map.cols;
	int rows2 = rows - ymax;
	int cols2 = cols - xmax;

	//
	std::vector<int> offsets(pts.size());
	for (int i = 1; i < pts.size(); i++) {
		offsets[i] = pts[i] - pts[i - 1];
	}

	// get scores
	score_map = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(100000.0));

	for (int i = 0; i < rows2; i++) {
		float* ptr_s = (float*)score_map.data + i * cols;
		uchar* ptr_m = mask_map.data + i * cols;
		for (int j = 0; j < cols2; j++) {
			if (*ptr_m) {
				float* ptr_temp = (float*)dist_map.data + i * cols + j + pts[0];
				double score = *ptr_temp;
				for (int m = 1; m < offsets.size(); m++) {
					ptr_temp += offsets[m];
					score += *ptr_temp;
				}
				*ptr_s = score;
			}
			ptr_s++;
			ptr_m++;
		}		
	}

	//cv::resize(score_map, score_map, cv::Size(cols / winsize, rows / winsize));
}

void IterativeFDCM::MatchIterative(std::vector<cv::Point2d> pts, std::vector<double> weight, double win_size, 
	cv::Mat & dist_map, cv::Mat & dist_gx, cv::Mat & dist_gy, double & score, cv::Point2d & offset)
{
	score = 0.0;
	offset = cv::Point2d(0, 0);

	double lambda = 10.0;
	int rows = dist_map.rows;
	int cols = dist_map.cols;
	int panelty = 100;

	float* ptr_dist = (float*)dist_map.data;
	float* ptr_gx = (float*)dist_gx.data;
	float* ptr_gy = (float*)dist_gy.data;
	int loc_pre = 0;
	double score_pre = 10000000000.0;
	double score_cur = 1000000000.0;

	double count_weighted = 0;
	double offset_dis = 0;
	while (score_pre - score_cur > score_pre * 1e-3 && offset_dis<win_size) {
		score_pre = score_cur;
		score_cur = 0;

		// step1: find the score and gradient
		std::vector<double> scores(pts.size(), -1);
		std::vector<double> gx(pts.size(), 0);
		std::vector<double> gy(pts.size(), 0);
		for (int i = 0; i < pts.size(); i++) {
			int x = pts[i].x;
			int y = pts[i].y;
			if (x<0 || x>cols - 1 || y<0 || y>rows - 1) {
				continue;
			}
			int loc_cur = y * cols + x;
			int loc_dev = loc_cur - loc_pre;
			loc_pre = loc_cur;
			ptr_dist += loc_dev;
			ptr_gx += loc_dev;
			ptr_gy += loc_dev;

			scores[i] = *ptr_dist;
			gx[i] = *ptr_gx;
			gy[i] = *ptr_gy;
		}

		// step2: update the points
		count_weighted = 0;
		double score_avg = 0.0;
		double gx_avg = 0.0, gy_avg = 0.0;
		for (int i = 0; i < pts.size(); i++) {
			if (scores[i]>0 && scores[i]<win_size) {
				score_avg += scores[i] * weight[i];
				gx_avg += gx[i] * weight[i];
				gy_avg += gy[i] * weight[i];
				count_weighted += weight[i];

				score_cur += scores[i];
			}
			else {
				score_cur += panelty;
			}
		}

		// we aim to reduce scores into zero, so the step can be score_cur/2
		score_avg /= count_weighted;
		gx_avg /= count_weighted;
		gy_avg /= count_weighted;
		double step = score_avg / 2;
		cv::Point2d dxy(-gx_avg * step, -gy_avg * step);
		for (int i = 0; i < pts.size(); i++) {
			pts[i] += dxy;
		}

		offset += dxy;
		offset_dis = sqrt(offset.x*offset.x + offset.y*offset.y);
	}

	score = count_weighted;
}

void IterativeFDCM::PtNormal(std::vector<cv::Point2d>& pts, int k, std::vector<cv::Point2d>& normals, std::vector<double> &angles)
{
	k = 16;
	// kd tree
	cv::Mat pts_mat(pts.size(), 2, CV_32F);
	float* ptr = (float*)pts_mat.data;
	for (int i = 0; i < pts.size(); i++) {
		ptr[0] = pts[i].x;
		ptr[1] = pts[i].y;
		ptr += 2;
	}
	cv::flann::Index kdindex(pts_mat, cv::flann::KDTreeIndexParams(4)); // the paramter is the circular paramter for contructing kd tree.

	// knn
	normals.resize(pts.size());
	angles.resize(pts.size());
	for (int i = 0; i < pts.size(); i++)
	{
		cv::Mat pti_mat(1, 2, CV_32F);
		pti_mat.at<float>(0, 0) = pts[i].x;
		pti_mat.at<float>(0, 1) = pts[i].y;
		cv::Mat mindices, mdists;
		kdindex.knnSearch(pti_mat, mindices, mdists, k, cv::flann::SearchParams(32));
		
		// pca
		cv::Mat data(k, 2, CV_32F);
		for (int j = 0; j < k; j++) {
			int id = mindices.at<int>(j);
			data.at<float>(j, 0) = pts[id].x;
			data.at<float>(j, 1) = pts[id].y;
		}
		cv::PCA pca_analysis(data, cv::Mat(), CV_PCA_DATA_AS_ROW);
		normals[i].x = pca_analysis.eigenvectors.at<float>(1, 0);
		normals[i].y = pca_analysis.eigenvectors.at<float>(1, 1);
		angles[i] = atan(pca_analysis.eigenvectors.at<float>(0, 1) / pca_analysis.eigenvectors.at<float>(0, 0));

		if (i == 100) {
			std::cout << data << std::endl;
		}

		//std::cout << data << std::endl;
		//std::cout << pca_analysis.eigenvectors << std::endl;
		//std::cout << angles[i] << std::endl;

		int a = 0;
	}
}

void IterativeFDCM::ShowResult(std::vector<cv::Point2d>& pts, double scale, cv::Mat & img)
{
	//
	std::vector<cv::Point2d> pts_all = pts;

	// xy to image
	double xmin = 1000000000.0;
	double xmax = -1000000000.0;
	double ymin = 1000000000.0;
	double ymax = -1000000000.0;
	for (size_t m = 0; m < pts_all.size(); m++)
	{
		if (pts_all[m].x < xmin)xmin = pts_all[m].x;
		if (pts_all[m].x > xmax)xmax = pts_all[m].x;
		if (pts_all[m].y < ymin)ymin = pts_all[m].y;
		if (pts_all[m].y > ymax)ymax = pts_all[m].y;
	}

	int rows = (ymax - ymin) / scale + 1;
	int cols = (xmax - xmin) / scale + 1;
	img = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
	double cx = 0, cy = 0;
	for (size_t m = 0; m < pts_all.size(); m++)
	{
		int px = (pts_all[m].x - xmin) / scale;
		int py = (pts_all[m].y - ymin) / scale;
		cx += px;
		cy += py;

		//cv::circle(img, cv::Point(px, py), 1, cv::Scalar(0, 0, 255));
		int loc = 3 * (py*cols + px);
		uchar* ptr = img.data + loc;
		ptr[0] = 0;
		ptr[1] = 0;
		ptr[2] = 255;
	}
	cv::circle(img, cv::Point2d(cx / pts_all.size(), cy / pts_all.size()), 3, cv::Scalar(255, 0, 0));
}

void IterativeFDCM::ShowResult(std::vector<cv::Point2d>& pts, cv::Mat & img)
{
	cv::Mat img_rgb;
	img.convertTo(img_rgb, CV_8UC3);
	for (int i = 0; i < pts.size(); i++) {
		cv::circle(img_rgb, pts[i], 2, cv::Scalar(0, 0, 255));
	}
	img = img_rgb;
}
