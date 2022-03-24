#ifndef _ITERATIVE_FDCM_H_
#define _ITERATIVE_FDCM_H_
#pragma once

#include <opencv2/opencv.hpp>

class IterativeFDCM
{
public:
	IterativeFDCM(void);
	~IterativeFDCM(void);

	void Run(std::vector<cv::Point2d> pts_src, std::vector<double> weight, std::vector<cv::Vec4f> lines_dst,
		std::vector<cv::Vec4f> hyps_coarse, int rows, int cols, double th_offset, std::vector<cv::Vec6f> &hyps_refined);

	static void GetDistMap(std::vector<cv::Vec4f> lines, cv::Mat &dist_map, cv::Mat &mask_map);

	void MatchIterative(std::vector<cv::Point2d> pts, std::vector<double> weight, std::vector<double> angles, double th_offset,
		cv::Mat &dist_map, cv::Mat &dist_gx, cv::Mat &dist_gy, cv::Mat &angle_map,
		double &score, cv::Point2d &offset);

	static void MatchIterative2(std::vector<int> pts, int xmax, int ymax, std::vector<double> weight, 
		cv::Mat mask_map, cv::Mat &dist_map, int winsize, cv::Mat &score_map);

	static void MatchIterative(std::vector<cv::Point2d> pts, std::vector<double> weight, double win_size,
		cv::Mat &dist_map, cv::Mat &dist_gx, cv::Mat &dist_gy, double &score, cv::Point2d &offset);

	void PtNormal(std::vector<cv::Point2d> &pts, int k, std::vector<cv::Point2d> &normals, std::vector<double> &angles);

	void ShowResult(std::vector<cv::Point2d> &pts, double scale, cv::Mat &img);

	void ShowResult(std::vector<cv::Point2d> &pts, cv::Mat &img);
};

#endif // _CANNY_LINE_H_

