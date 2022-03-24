#ifndef _Registration_GPS_
#define _Registration_GPS_
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>

class RegistrationGPS
{
public:
	RegistrationGPS();

	~RegistrationGPS(void);

	void Run();

	// pre-processing
	void PreSetPaths(std::string fold, std::string sate_dsm, std::string sate_ortho,
		std::string sate_mask_building, std::string sate_mask_ndvi,
		std::string street_dense, std::string street_pose, std::string street_gps);	

	void PreProssSatellite();

	void PreProssStreetview();

	void AssignPt2Cams();

	void VerticalBuildingDetection();

	void VerticalRefinement();

	void BuildingRefinement();

	// XY alignment via building matching	
	void AlignmentXY();

	void StreetBuildingDetection(std::vector<std::vector<Eigen::Vector2d>> &pts_buildings, std::vector<cv::Vec2i> &cams_buildings);

	void TrajectorySegmentation(std::vector<Eigen::Vector2d> cams, std::vector<std::vector<int>> &cams_segs);

	void StreetBuildingSegmentation(std::vector<Eigen::Vector2d>& pts, float th_dev, std::vector<std::vector<Eigen::Vector2d>>& segments_pts,
		std::vector<std::vector<int>>& segments_ids);

	void GlobalAlignment(std::vector<std::vector<Eigen::Vector2d>> src_list, std::vector<Eigen::Vector2d> dst_list, std::vector<cv::Vec6d> &trans_list);

	void HypothesesViaGPS(std::vector<Eigen::Vector2d> &cams_street,  std::vector<Eigen::Vector2d> &cams_gps, std::string file_trans);

	void CostTransformations(std::vector<std::vector<Eigen::Vector2d>> &src_list, std::vector<Eigen::Vector2d> &dst_list,
		std::vector<cv::Vec6f> &trans, std::vector<std::vector<float>> &costs);

	// Z alignment
	void AlignmentZ();

	void DetectGround();

	// XYZ alignment
	void AlignmentXYZ();

	// supporting functions
	void PointOrthogonality(std::vector<Eigen::Vector2d> &pts, std::vector<double> &orthogonality);

	void Rotation2Euler(Eigen::Matrix3d R, double &rx, double &ry, double &rz);

	void Euler2Rotation(double rx, double ry, double rz, Eigen::Matrix3d &R);

	void StaticVector(std::vector<double> &data, double &v_avg, double &v_min, double &v_max);

	void UniqueVector(std::vector<int> &data);

	void PCANormal(std::vector<Eigen::Vector2d> &pts, Eigen::Vector2d &normal, double &curve);

	void ShowResult(std::vector<Eigen::Vector2d> &pts, double scale, cv::Mat &img);

	void ShowResult(std::vector<Eigen::Vector2d> &pts, cv::Mat &img);

	void ShowResult(std::vector<Eigen::Vector2d> &src, cv::Vec6f paras, std::vector<Eigen::Vector2d> &dst, cv::Mat &img);

	void AlignViaXYOffset(std::vector<Eigen::Vector2d> &src, std::vector<Eigen::Vector2d> &dst, Eigen::Vector2d &offset, double score);

public:
	std::string fold_;
	std::string file_dsm_, file_ortho_;
	std::string file_mask_building_, file_mask_ndvi_, file_mask_road_float_;
	std::string file_street_dense_, file_street_pose_, file_street_gps_;

	//
	double scale_, gsd_;

	// sate
	double xs_, ys_, dx_, dy_;
	int sate_rows_, sate_cols_;
	int ellipsoid_id_ = 23; // WGS-84
	std::string zone_id_ = "17N";

	// alignment
	cv::Point3d dir_vertial_;	

};

#endif //_RegistrationGPS_
