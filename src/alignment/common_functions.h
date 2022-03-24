#ifndef _COMMON_FUNCTIONS_
#define _COMMON_FUNCTIONS_
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>

#include "utils/nanoflann_all.hpp"
#include "utils/nanoflann_utils_all.h"

using namespace cv;
using namespace std;

class LineFunctions
{
public:
	LineFunctions(void){};
	~LineFunctions(void){};

public:
	static void lineFitting( int rows, int cols, std::vector<cv::Point> &contour, double thMinimalLineLength, std::vector<std::vector<cv::Point2d> > &lines );

	static void subDivision( std::vector<std::vector<cv::Point> > &straightString, std::vector<cv::Point> &contour, int first_index, int last_index
		, double min_deviation, int min_size  );

	static void subDivision(std::vector<std::vector<int> > &straightString, std::vector<cv::Point> &contour, int first_index, int last_index
		, double min_deviation, int min_size);

	static void lineFittingSVD( cv::Point *points, int length, std::vector<double> &parameters, double &maxDev );
};

struct PCAInfo
{
	double lambda0, scale;
	cv::Matx31d normal, planePt;
	std::vector<int> idxAll, idxIn;

	PCAInfo &operator =(const PCAInfo &info)
	{
		this->lambda0 = info.lambda0;
		this->normal = info.normal;
		this->idxIn = info.idxIn;
		this->idxAll = info.idxAll;
		this->scale = scale;
		return *this;
	}
};


struct PCAInfo2d
{
	double lambda0;
	cv::Matx21d normal;
	std::vector<int> idx_neigh;

	PCAInfo2d &operator =(const PCAInfo2d &info)
	{
		this->lambda0 = info.lambda0;
		this->normal = info.normal;
		this->idx_neigh = info.idx_neigh;
		return *this;
	}
};

class PCAFunctions 
{
public:
	PCAFunctions(void){};
	~PCAFunctions(void){};

	void Ori_PCA(PointCloud3d<double> &cloud, int k, std::vector<PCAInfo> &pcaInfos, double &scale, double &magnitd );

	static void Ori_PCA2d(PointCloud2d<double> &cloud, int k, std::vector<PCAInfo2d> &normals, double &scale);

	static void PCA2d(std::vector<cv::Point2d> &cloud, int k, std::vector<double> &angles);

	static void PCA2d(std::vector<cv::Point2d> &cloud, int k, std::vector<cv::Point2d>& normals, std::vector<double> &angles);

	static void PCA2d(std::vector<Eigen::Vector2d> &cloud, int k, std::vector<double> &angles, std::vector<double> &carvture);

	static void Orthogonality(std::vector<Eigen::Vector2d> &cloud, int k, std::vector<double> &score);

	void PCASingle( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo );

	void MCMD_OutlierRemoval( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo );

	double meadian( std::vector<double> dataset );
};

#endif //_COMMON_FUNCTIONS_
