#include "registration_nogps.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iomanip>
#include <experimental/filesystem>
#include <Eigen/Geometry>

#include "common_functions.h"
#include "converter_utm_latlon.h"
#include "tinytiffreader.h"
#include "tinytiffwriter.h"
#include "rigid_transformation.h"
#include "ceres/ceres.h"
#include "resigtration_error_trans.h"
#include "utils/basic_funcs.h"
#include "utils/nanoflann_all.hpp"
#include "utils/nanoflann_utils_all.h"
#include "GCoptimization.h"


#define VALUE_NON_ROAD -3
#define VALUE_ROAD -2
#define VALUE_OSM -1

namespace fs = std::experimental::filesystem;
using namespace nanoflann;

RegistrationNoGPS::RegistrationNoGPS()
{
}


RegistrationNoGPS::~RegistrationNoGPS(void)
{
}

void RegistrationNoGPS::Run()
{
	// if no gps, the scale need to be specified
	scale_ = 0.62;
	gsd_ = 0.5;

	PreProssSatellite();

	PreProssStreetview();

	// xy alignment
	std::cout << "---AlignmentXY" << std::endl;
	AlignmentXY();

	// z alignment
	std::cout << "---AlignmentZ" << std::endl;
	AlignmentZ();

	// xyz alignment
	std::cout << "---AlignmentXYZ" << std::endl;
	AlignmentXYZ();
}

void RegistrationNoGPS::PreSetPaths(std::string fold, std::string sate_dsm, std::string sate_ortho,
	std::string sate_mask_building, std::string sate_mask_ndvi,
	std::string street_dense, std::string street_pose, std::string street_gps)
{
	fold_ = fold;
	file_dsm_ = sate_dsm;
	file_ortho_ = sate_ortho;
	file_mask_building_ = sate_mask_building;
	file_mask_ndvi_ = sate_mask_ndvi;
	file_street_dense_ = street_dense;
	file_street_pose_ = street_pose;
	file_street_gps_ = street_gps;
	file_mask_road_float_ = fold + "\\mask_road.bin";
}


void RegistrationNoGPS::PreProssSatellite()
{
	std::string file_sate_tfw = file_dsm_.substr(0, file_dsm_.size() - 4) + ".tfw";
	// read in tfw
	std::ifstream iff(file_sate_tfw);
	float zeros;
	//iff >> xs_ >> dx_ >> zeros >> ys_ >> zeros >> dy_;
	iff >> dx_ >> zeros >> zeros >> dy_ >> xs_ >> ys_;
	iff.close();
}

void RegistrationNoGPS::PreProssStreetview()
{
	AssignPt2Cams();

	VerticalBuildingDetection();

	VerticalRefinement();

	BuildingRefinement();
}

void RegistrationNoGPS::AssignPt2Cams()
{
	std::string file_dense_new = fold_ + "\\street_dense_cam.txt";
	if (std::experimental::filesystem::exists(file_dense_new)) {
		file_street_dense_ = file_dense_new;
		return;
	}

	// read in dense pts
	std::string file_dense = file_street_dense_;
	std::ifstream iff = std::ifstream(file_dense);
	float x, y, z, idx;
	int r, g, b;
	int count = 0;
	std::vector<cv::Vec4f> pts;
	std::vector<cv::Vec3i> colors;
	std::vector<std::vector<int>> segids(100);
	while (!iff.eof()) {
		iff >> x >> y >> z >> r >> g >> b >> idx;
		count++;
		if (count % 5 == 0) {
			count = 0;
			pts.push_back(cv::Vec4f(x, y, z, idx));
			colors.push_back(cv::Vec3i(r, g, b));
			segids[int(idx)].push_back(pts.size() - 1);
		}
	}
	iff.close();

	// read in cams
	std::string file_pose = file_street_pose_;
	iff = std::ifstream(file_pose);
	std::string temp;
	std::getline(iff, temp);
	std::getline(iff, temp);
	std::string name;
	float rx, ry, rz;
	std::vector<cv::Point3d> cams;
	while (!iff.eof()) {
		iff >> name >> x >> y >> z >> rx >> ry >> rz;
		cams.push_back(cv::Point3d(x, y, z));
	}

	// step1: sort pt segs via cam trajectory
	std::vector<cv::Vec4f> segid_camid;
	for (int i = 0; i < segids.size(); i++) {
		if (!segids[i].size()) {
			continue;
		}

		// closest cam for each point
		std::vector<int> idcams;
		for (int j = 0; j < segids[i].size(); j++) {
			int id_pt = segids[i][j];
			double dis_min = 10000000000.0;
			int idx_min = 0;
			for (int m = 0; m < cams.size(); m += 10) {
				double dx = pts[id_pt].val[0] - cams[m].x;
				double dy = pts[id_pt].val[1] - cams[m].y;
				double dz = pts[id_pt].val[2] - cams[m].z;
				double dis = abs(dx) + abs(dy) + abs(dz);
				if (dis < dis_min) {
					dis_min = dis;
					idx_min = m;
				}
			}
			idcams.push_back(idx_min);
		}

		// determine the cam seg
		UniqueVector(idcams);

		// 
		double avg = 0.0;
		for (int j = 0; j < idcams.size(); j++) {
			avg += idcams[j];
		}
		avg /= idcams.size();

		double sigma = 0.0;
		for (int j = 0; j < idcams.size(); j++) {
			sigma += (idcams[j] - avg)*(idcams[j] - avg);
		}
		sigma = sqrt(sigma / idcams.size());

		double id_avg = 0.0, id_min = 1000000, id_max = 0;
		int count_avg = 0;
		for (int j = 0; j < idcams.size(); j++) {
			if (abs(idcams[j] - avg) < sigma) {
				id_avg += idcams[j];
				count_avg++;
				if (idcams[j] < id_min) {
					id_min = idcams[j];
				}
				if (idcams[j] > id_max) {
					id_max = idcams[j];
				}
			}
		}
		id_avg = int(id_avg / count_avg);
		segid_camid.push_back(cv::Vec4f(id_avg, i, id_min, id_max));
	}

	// sort the segments
	std::sort(segid_camid.begin(), segid_camid.end(), [](const cv::Vec4f& lhs, const cv::Vec4f& rhs) { return lhs.val[0] < rhs.val[0]; });

	// step2: find assign each pt with closest cams
	for (int i = 0; i < segid_camid.size(); i++) {
		int id_seg = segid_camid[i].val[1];
		int ids_cam = std::max(int(segid_camid[i].val[2] - cams.size() / 10), 0);
		int ide_cam = std::min(int(segid_camid[i].val[3] + cams.size() / 10), int(cams.size()));

		for (int j = 0; j < segids[id_seg].size(); j++) {
			int id_pt = segids[id_seg][j];
			double x = pts[id_pt].val[0];
			double y = pts[id_pt].val[1];
			double z = pts[id_pt].val[2];
			double dis_min = 1000000000.0;
			int idx_min = 0;
			for (int m = ids_cam; m < ide_cam; m++) {
				double dx = cams[m].x - x;
				double dy = cams[m].y - y;
				double dz = cams[m].z - z;
				double dis = abs(dx) + abs(dy) + abs(dz);
				if (dis < dis_min) {
					dis_min = dis;
					idx_min = m;
				}
			}

			//
			pts[id_pt].val[3] = idx_min;
		}
	}

	//step3: write out
	std::ofstream off(file_dense_new);
	for (int i = 0; i < pts.size(); i++) {
		off << pts[i].val[0] << " " << pts[i].val[1] << " " << pts[i].val[2] << " ";
		off << colors[i].val[0] << " " << colors[i].val[1] << " " << colors[i].val[2] << " " << pts[i].val[3] << std::endl;
	}
	off.close();

	file_street_dense_ = file_dense_new;
}

void RegistrationNoGPS::VerticalBuildingDetection()
{
	std::string file_vertical = fold_ + "\\vertical.txt";
	std::string file_building = fold_ + "\\street_building.txt";
	if (std::experimental::filesystem::exists(file_vertical) && std::experimental::filesystem::exists(file_building)) {
		std::ifstream ff(file_vertical);
		ff >> dir_vertial_.x >> dir_vertial_.y >> dir_vertial_.z;
		ff.close();
		return;
	}

	// read in dense pts
	std::string file_dense = file_street_dense_;
	std::ifstream iff = std::ifstream(file_dense);
	float x, y, z, idx;
	int r, g, b;
	int count = 0;
	PointCloud3d<double> street_pts;
	std::vector<cv::Point3i> colors_pts;
	std::vector<int> cam_idx_pts;
	while (!iff.eof()) {
		iff >> x >> y >> z >> r >> g >> b >> idx;
		count++;
		if (count % 1 == 0) {
			count = 0;
			street_pts.pts.push_back(PointCloud3d<double>::PtData(x, y, z));
			cam_idx_pts.push_back(idx);
			colors_pts.push_back(cv::Point3i(r, g, b));
		}
	}
	iff.close();

	// calculate normal of points
	PCAFunctions pcaer;
	int k = 30;
	std::vector<PCAInfo> pcaInfos;
	double scale, magnitd = 0.0;
	pcaer.Ori_PCA(street_pts, k, pcaInfos, scale, magnitd);

	// find the domain directions
	float step_grid = 2.0 / 180.0*CV_PI;
	int nlon = 2 * CV_PI / step_grid;
	int nlat = CV_PI / step_grid;
	std::vector<std::vector<int>> lon_lat_gird_count(nlat);
	for (size_t i = 0; i < nlat; i++) {
		lon_lat_gird_count[i].resize(nlon);
	}

	for (size_t i = 0; i < pcaInfos.size(); i++)
	{
		double lat = acos(pcaInfos[i].normal(2));
		double lon = atan2(pcaInfos[i].normal(0), pcaInfos[i].normal(1)) + CV_PI;
		int glat = lat / step_grid;
		if (glat == nlat) glat = nlat - 1;
		int glon = lon / step_grid;
		if (glon == nlon) glon = nlon - 1;
		lon_lat_gird_count[glat][glon]++;
	}

	std::vector<cv::Point2i> dir_clusters;
	for (int i = 0; i < nlat; i++) {
		for (int j = 0; j < nlon; j++) {
			if (lon_lat_gird_count[i][j] > pcaInfos.size() / 1000)
			{
				int i1 = (i - 1 + nlat) % nlat;
				int i2 = (i + 1 + nlat) % nlat;
				int j1 = (j - 1 + nlon) % nlon;
				int j2 = (j + 1 + nlon) % nlon;
				std::vector<int> ii = { i1, i, i2 };
				std::vector<int> jj = { j1, j, j2 };
				bool is_non_max = false;
				for (size_t m = 0; m < ii.size(); m++) {
					for (size_t n = 0; n < jj.size(); n++) {
						if (lon_lat_gird_count[ii[m]][jj[n]] > lon_lat_gird_count[i][j]) {
							is_non_max = true;
							break;
						}
					}
					if (is_non_max) {
						break;
					}
				}

				if (!is_non_max) {
					dir_clusters.push_back(cv::Point2i(i, j));
				}
			}
		}
	}

	// find the best
	double th_cos_paral = cos(5.0 / 180.0*CV_PI);
	double th_cos_ortho = cos(85.0 / 180.0*CV_PI);
	double s_best = 0.0;
	int idx_best = 0;
	cv::Matx31d vv_vertical(0, -1.0, 0.0);
	for (int i = 0; i < dir_clusters.size(); i++) {
		double lon = dir_clusters[i].y*step_grid;
		double lat = dir_clusters[i].x*step_grid;
		cv::Matx31d vv;
		vv(0) = sin(lat)*sin(lon);
		vv(1) = sin(lat)*cos(lon);
		vv(2) = cos(lat);
		double w = abs(vv_vertical.dot(vv));
		double si = 0;
		for (int j = 0; j < pcaInfos.size(); j++) {
			double cos_value = abs(vv.dot(pcaInfos[j].normal));
			if (cos_value < th_cos_ortho || cos_value > th_cos_paral) {
				//si++;
				si += w;
			}
		}
		if (si > s_best) {
			s_best = si;
			idx_best = i;
		}
	}

	double lon = dir_clusters[idx_best].y*step_grid;
	double lat = dir_clusters[idx_best].x*step_grid;
	dir_vertial_.x = sin(lat)*sin(lon);
	dir_vertial_.y = sin(lat)*cos(lon);
	dir_vertial_.z = cos(lat);
	if (dir_vertial_.y < 0) {
		dir_vertial_ *= -1;
	}

	// detect building
	std::vector<int> is_building_init(pcaInfos.size(), 0);
	double th_dev = cos(75.0 / 180.0*CV_PI);
	for (size_t i = 0; i < pcaInfos.size(); i++)
	{
		double dev = dir_vertial_.x*pcaInfos[i].normal(0)
			+ dir_vertial_.y*pcaInfos[i].normal(1)
			+ dir_vertial_.z*pcaInfos[i].normal(2);
		if (abs(dev)<th_dev) {
			is_building_init[i] = 1;
		}
	}

	std::vector<int> is_building(pcaInfos.size(), 0);
	for (size_t i = 0; i < pcaInfos.size(); i++) {
		int count = 0;
		for (size_t j = 0; j < pcaInfos[i].idxAll.size(); j++) {
			int id = pcaInfos[i].idxAll[j];
			count += is_building_init[id];
		}
		if (count > k*0.75) {
			is_building[i] = 1;
		}
	}
	is_building = is_building_init;

	// write out
	std::ofstream ff(file_vertical);
	ff << std::setprecision(12) << std::fixed;
	ff << dir_vertial_.x << " " << dir_vertial_.y << " " << dir_vertial_.z;
	ff.close();

	// write out buildings
	ff = std::ofstream(file_building);
	ff << std::setprecision(12) << std::fixed;
	for (size_t i = 0; i < pcaInfos.size(); i++) {
		if (is_building[i]) {
			ff << street_pts.pts[i].x << " " << street_pts.pts[i].y << " " << street_pts.pts[i].z << " " ;
			ff << colors_pts[i].x << " " << colors_pts[i].y << " " << colors_pts[i].z << " " ;
			ff << cam_idx_pts[i] << std::endl;
		}
	}
	ff.close();
}

void RegistrationNoGPS::VerticalRefinement()
{
	std::string file_refined = fold_ + "\\vertical_refined.txt";
	if (std::experimental::filesystem::exists(file_refined)) {
		std::ifstream ff(file_refined);
		ff >> dir_vertial_.x >> dir_vertial_.y >> dir_vertial_.z;
		ff.close();
		return;
	}

	// read in building points
	std::string file_building = fold_ + "\\street_building.txt";
	std::vector<Eigen::Vector3d> pts_building;
	std::vector<cv::Point3i> pts_colors;
	std::ifstream ff(file_building);
	float x, y, z, idx;
	int r, g, b;
	while (!ff.eof()) {
		ff >> x >> y >> z >> r >> g >> b >> idx;
		pts_building.push_back(Eigen::Vector3d(x, y, z));
		pts_colors.push_back(cv::Point3i(r, g, b));
	}
	ff.close();

	// get lon and lat for the direction vector
	double lat = acos(dir_vertial_.z);
	double lon = atan2(dir_vertial_.x, dir_vertial_.y) + CV_PI;

	int n = 10;
	double step = 4.0 / 180.0 * CV_PI / n;
	double lon_best, lat_best;
	float score_best = 0.0;
	cv::Mat img_best;
	for (int i = 0; i < n; i++)
	{
		double lon_temp = (i - n / 2)*step + lon;
		if (lon_temp<0) { lon_temp += 2 * CV_PI; }
		if (lon_temp>2 * CV_PI) { lon_temp -= 2 * CV_PI; }

		for (int j = 0; j < n; j++)
		{
			double lat_temp = (j - n / 2)*step + lat;
			if (lat_temp<0) { lat_temp += 2 * CV_PI; }
			if (lat_temp>2 * CV_PI) { lat_temp -= 2 * CV_PI; }

			double x = sin(lat_temp)*sin(lon_temp);
			double y = sin(lat_temp)*cos(lon_temp);
			double z = cos(lat_temp);
			Eigen::Vector3d vz(x, y, z);
			Eigen::Vector3d vx(1.0, 1.0, -(x + y) / z);
			vx /= vx.norm();
			//Eigen::Vector3d vy = vz.cross(vx); // right hand rule
			Eigen::Vector3d vy = vz.cross(vx); // right hand rule
			vy /= vy.norm();
			Eigen::Vector3d pt_ref = pts_building[0];

			// projection to xy
			std::vector<cv::Point2d> pts2d(pts_building.size());
			for (int m = 0; m < pts_building.size(); m++) {
				Eigen::Vector3d dev = pts_building[m] - pt_ref;
				pts2d[m].x = vx.dot(dev);
				pts2d[m].y = vy.dot(dev);
			}

			// xy to image
			double xmin = 1000000000.0;
			double xmax = -1000000000.0;
			double ymin = 1000000000.0;
			double ymax = -1000000000.0;
			for (size_t m = 0; m < pts2d.size(); m++)
			{
				if (pts2d[m].x < xmin)xmin = pts2d[m].x;
				if (pts2d[m].x > xmax)xmax = pts2d[m].x;
				if (pts2d[m].y < ymin)ymin = pts2d[m].y;
				if (pts2d[m].y > ymax)ymax = pts2d[m].y;
			}

			double scale = std::max(ymax - ymin, xmax - xmin) / 1000;
			int rows = (ymax - ymin) / scale + 1;
			int cols = (xmax - xmin) / scale + 1;
			cv::Mat img(rows, cols, CV_32FC1, cv::Scalar(0.0));
			float* ptr = (float*)img.data;
			int loc = 0;
			for (size_t m = 0; m < pts2d.size(); m++)
			{
				int px = (pts2d[m].x - xmin) / scale;
				int py = (pts2d[m].y - ymin) / scale;
				int loc_temp = py * cols + px;
				ptr += (loc_temp - loc);
				loc = loc_temp;
				*ptr += 1;
			}

			// calculate the score
			float score = img.dot(img);
			//std::cout << "score " << score << std::endl;
			if (score > score_best) {
				score_best = score;
				lon_best = lon_temp;
				lat_best = lat_temp;
				img_best = img.clone();
			}
		}
	}

	// 
	std::cout << "dir_vertial_ former " << dir_vertial_ << std::endl;
	dir_vertial_.x = sin(lat_best)*sin(lon_best);
	dir_vertial_.y = sin(lat_best)*cos(lon_best);
	dir_vertial_.z = cos(lat_best);
	if (dir_vertial_.y < 0) {
		dir_vertial_ *= -1;
	}

	// write out
	std::ofstream off(file_refined);
	off << std::setprecision(12) << std::fixed;
	off << dir_vertial_.x << " " << dir_vertial_.y << " " << dir_vertial_.z;
	off.close();

	std::cout << "dir_vertial_ refined " << dir_vertial_ << std::endl;
	cv::imwrite(fold_ + "\\overlapped_best.bmp", 255 * img_best);
}

void RegistrationNoGPS::BuildingRefinement()
{
	// read in buildings
	std::string file_building = fold_ + "\\street_building.txt";
	std::string file_building_out = fold_ + "\\street_building_refined.txt";
	if (std::experimental::filesystem::exists(file_building_out)) {
		return;
	}

	// read in all cam pose
	std::string file_pose = file_street_pose_;
	std::ifstream iff(file_pose);
	std::string temp;
	std::getline(iff, temp);
	std::getline(iff, temp);
	std::string name;
	float x, y, z, rx, ry, rz;
	std::vector<cv::Point3d> cams;
	while (!iff.eof()) {
		iff >> name >> x >> y >> z >> rx >> ry >> rz;
		cams.push_back(cv::Point3d(x, y, z));
	}

	// read in vertical direction
	Eigen::Vector3d vz(dir_vertial_.x, dir_vertial_.y, dir_vertial_.z);

	// step1: refine building via normal
	PointCloud3d<double> building_pts;
	std::vector<cv::Point3i> colors_pts;
	std::vector<int> idx_pts;
	std::ifstream ff(file_building);
	float idx = 0;
	int r, g, b;
	while (!ff.eof()) {
		ff >> x >> y >> z >> r >> g >> b >> idx;
		building_pts.pts.push_back(PointCloud3d<double>::PtData(x, y, z));
		colors_pts.push_back(cv::Point3i(r, g, b));
		idx_pts.push_back(idx);
	}
	ff.close();

	std::vector<int> ks;
	ks.push_back(10);
	ks.push_back(20);
	ks.push_back(30);
	PCAFunctions pcaer;

	float th_angle = cos(75.0 / 180.0*CV_PI);
	std::vector<int> scores_pts(building_pts.pts.size(), 0);
	for (int i = 0; i < ks.size(); i++) {
		std::vector<PCAInfo> pcaInfos;
		double scale, magnitd = 0.0;
		pcaer.Ori_PCA(building_pts, ks[i], pcaInfos, scale, magnitd);

		cv::Matx31d vz_mat(vz(0), vz(1), vz(2));
		for (int j = 0; j < pcaInfos.size(); j++) {
			double v_cos = abs(pcaInfos[j].normal.dot(vz_mat));
			if (v_cos < th_angle) {
				scores_pts[j]++;
			}
		}
	}

	std::vector<Eigen::Vector3d> pts;
	std::vector<cv::Point3i> colors;
	std::vector<float> idxs;
	for (int i = 0; i < scores_pts.size(); i++) {
		if (scores_pts[i] >= 2) {
			pts.push_back(Eigen::Vector3d(building_pts.pts[i].x, building_pts.pts[i].y, building_pts.pts[i].z));
			colors.push_back(colors_pts[i]);
			idxs.push_back(idx_pts[i]);
		}
	}


	// step2: refine building via z height
	std::vector<double> znew;
	std::vector<std::pair<int, int>> pts_cam;
	std::vector<std::vector<int>> cam_pts(cams.size());
	for (size_t i = 0; i < pts.size(); i++)
	{
		// find closest cam		
		int idx_min = idxs[i];
		Eigen::Vector3d dev_z = pts[i] - Eigen::Vector3d(cams[idx_min].x, cams[idx_min].y, cams[idx_min].z);
		znew.push_back(vz.dot(dev_z));
		pts_cam.push_back(std::pair<int, int>(idx_min, i));
		cam_pts[idx_min].push_back(i);
	}

	std::vector<cv::Vec2i> pts_cam_inlier;
	for (int i = 0; i < cams.size(); i++) {
		std::vector<cv::Vec2f> z_idx;
		for (int j = 0; j < cam_pts[i].size(); j++) {
			int idx_pt = cam_pts[i][j];
			z_idx.push_back(cv::Vec2f(znew[idx_pt], idx_pt));
		}
		std::sort(z_idx.begin(), z_idx.end(), [](const cv::Vec2f& lhs, cv::Vec2f& rhs) { return lhs.val[0] < rhs.val[0]; });

		int idx_s = int(z_idx.size()*0.01);
		int idx_e = std::min(int(z_idx.size()*0.99), int(z_idx.size()) - 1);
		if (idx_e <= idx_s) {
			continue;
		}
		double zmin = z_idx[idx_s].val[0];
		double zmax = z_idx[idx_e].val[0];
		double zlow = 0.7*zmin + 0.3*zmax;
		double zhigh = 0.3*zmin + 0.7*zmax;
		//std::cout << zmin << " " << zmax << std::endl;

		//
		for (int j = 0; j < z_idx.size(); j++) {
			if (z_idx[j].val[0] > zlow && z_idx[j].val[0] < zhigh) {
				//if (z_idx[j].val[0] < zhigh) {
				pts_cam_inlier.push_back(cv::Vec2i(z_idx[j].val[1], i));
			}
		}
	}

	// write out
	std::ofstream off(file_building_out);
	for (size_t j = 0; j < pts_cam_inlier.size(); j++) {
		int id_pt = pts_cam_inlier[j].val[0];
		int id_cam = pts_cam_inlier[j].val[1];
		off << pts[id_pt](0) << " " << pts[id_pt](1) << " " << pts[id_pt](2) << " ";
		off << colors[id_pt].x << " " << colors[id_pt].y << " " << colors[id_pt].z << " ";
		off << id_cam << std::endl;
	}
	off.close();
}


void RegistrationNoGPS::AlignmentXY()
{
	//////////////////////////////////
	// step1: read in temprary results
	//////////////////////////////////

	// read in vetical
	std::string file_refined = fold_ + "\\vertical_refined.txt";
	std::ifstream ff1(file_refined);
	ff1 >> dir_vertial_.x >> dir_vertial_.y >> dir_vertial_.z;
	ff1.close();
	Eigen::Vector3d vz(dir_vertial_.x, dir_vertial_.y, dir_vertial_.z);
	Eigen::Vector3d vx(1.0, 1.0, -(vz(0) + vz(1)) / vz(2));  vx /= vx.norm();
	Eigen::Vector3d vy = vz.cross(vx);  vy /= vy.norm();

	// read in all poses
	std::string file_pose = file_street_pose_;
	std::ifstream iff(file_pose);
	std::string temp;
	std::getline(iff, temp);
	std::getline(iff, temp);
	std::string name;
	float x, y, z, rx, ry, rz, idx;
	std::vector<Eigen::Vector2d> cams;
	while (!iff.eof()) {
		iff >> name >> x >> y >> z >> rx >> ry >> rz;
		Eigen::Vector3d t(x, y, z);
		cams.push_back(Eigen::Vector2d(t.dot(vx), t.dot(vy)));
	}
	iff.close();
	int ncams = cams.size();

	// read in building pts
	std::string file_building = fold_ + "\\street_building_refined.txt";
	std::vector<Eigen::Vector3d> pts_building_3d;
	std::vector<Eigen::Vector2d> pts_building;
	std::vector<int> pts_building_idxcam;
	std::ifstream ff(file_building);
	int r, g, b;
	while (!ff.eof()) {
		ff >> x >> y >> z >> r >> g >> b >> idx;
		//ff >> x >> y >> z >> idx;
		Eigen::Vector3d t(x, y, z);
		pts_building_3d.push_back(t);
		pts_building.push_back(Eigen::Vector2d(t.dot(vx), t.dot(vy)));
		pts_building_idxcam.push_back(idx);
	}
	ff.close();

	////////////////////////////////////////
	// step2: cross-view building extraction
	////////////////////////////////////////

	// street-view buildings
	std::vector<std::vector<int>> trajectorys;
	TrajectorySegmentation(cams, trajectorys);
	int ntraj = trajectorys.size();

	std::vector<std::vector<Eigen::Vector2d>> street_buildings;
	std::vector<std::vector<int>> street_buildings_id;
	std::vector<std::vector<Eigen::Vector2d>> street_cams;
	std::vector<cv::Vec2i> street_cams_id;
	for (size_t i = 0; i < ntraj; i++) {
		int id_cams = trajectorys[i][0];
		int id_came = trajectorys[i][trajectorys[i].size() - 1];

		// get cams
		std::vector<Eigen::Vector2d> cams_i;
		for (int j = id_cams; j <= id_came; j++) {
			cams_i.push_back(cams[j]);
		}

		// get pts
		std::vector<Eigen::Vector2d> pts_i;
		std::vector<int> pts_id_i;
		for (int j = 0; j < pts_building_idxcam.size(); j++) {
			if (pts_building_idxcam[j] >= id_cams && pts_building_idxcam[j] <= id_came) {
				pts_i.push_back(pts_building[j]);
				pts_id_i.push_back(j);
			}
		}
		if (pts_i.size() < 1000) {
			continue;
		}

		float th_dev = 1.0;
		std::vector<std::vector<Eigen::Vector2d>> segs_pts_i;
		std::vector<std::vector<int>> segs_ids_i;
		StreetBuildingSegmentation(pts_i, th_dev, segs_pts_i, segs_ids_i);

		std::vector<std::vector<int>> segs_ids_i_3d(segs_ids_i.size());
		std::vector<cv::Vec4i> segs_ids_avg(segs_ids_i.size());
		for (size_t j = 0; j < segs_ids_i.size(); j++) {
			int id_cam_avg = 0, id_cam_min = 10000000, id_cam_max = -id_cam_min;
			for (size_t m = 0; m < segs_ids_i[j].size(); m++) {
				int id_pt_temp = pts_id_i[segs_ids_i[j][m]];
				segs_ids_i_3d[j].push_back(id_pt_temp);
				int id_cam_temp = pts_building_idxcam[id_pt_temp];
				id_cam_avg += id_cam_temp;
				if (id_cam_temp < id_cam_min) {
					id_cam_min = id_cam_temp;
				}
				if (id_cam_temp > id_cam_max) {
					id_cam_max = id_cam_temp;
				}
			}
			id_cam_avg /= segs_ids_i[j].size();
			segs_ids_avg[j] = cv::Vec4i(j, id_cam_avg, id_cam_min, id_cam_max);
		}
		std::sort(segs_ids_avg.begin(), segs_ids_avg.end(), [](const cv::Vec4i& lhs, const cv::Vec4i& rhs) { return lhs.val[1] < rhs.val[1]; });

		for (size_t j = 0; j < segs_ids_avg.size(); j++) {
			street_buildings.push_back(segs_pts_i[segs_ids_avg[j].val[0]]);
			street_buildings_id.push_back(segs_ids_i_3d[segs_ids_avg[j].val[0]]);

			std::vector<Eigen::Vector2d> segs_cams;
			for (size_t m = segs_ids_avg[j].val[2]; m < segs_ids_avg[j].val[3]; m++) {
				segs_cams.push_back(cams[m]);
			}
			street_cams.push_back(segs_cams);
			street_cams_id.push_back(cv::Vec2i(segs_ids_avg[j].val[2], segs_ids_avg[j].val[3]));
		}
	}

	if (1) {
		int rows = 3000;
		int cols = 3000;
		float xmin = -cols / 4;
		float ymin = -rows / 2;
		cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
		std::string file_buildings = fold_ + "\\building_pts.txt";
		std::ofstream ff(file_buildings);
		ff << std::setprecision(12) << std::fixed;
		for (size_t m = 0; m < street_buildings.size(); m++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			int xc = 0, yc = 0;
			for (int j = 0; j < street_buildings[m].size(); j++) {
				int x = int(street_buildings[m][j](0) - xmin);
				int y = int(street_buildings[m][j](1) - ymin);
				xc += x;
				yc += y;
				cv::circle(img, cv::Point(x, y), 1, cv::Scalar(r, g, b), 1);
				int id = street_buildings_id[m][j];
				ff << pts_building_3d[id](0) << " " << pts_building_3d[id](1) << " " << pts_building_3d[id](2) << " " << r << " " << g << " " << b << std::endl;
			}
			xc /= street_buildings[m].size();
			yc /= street_buildings[m].size();
			cv::putText(img, std::to_string(m), cv::Point(xc, yc), 1, 2, cv::Scalar(r, g, b), 2);
		}
		ff.close();

		cv::imwrite(fold_ + "\\buildings_street.bmp", img);
	}

	// over-view buildings
	std::string file_mask_building = file_mask_building_;
	cv::Mat mask_building = cv::imread(file_mask_building, 0);
	std::vector<std::vector<Eigen::Vector2d>> sate_buildings;
	SateBuildingSegmentation(mask_building, sate_buildings);

	if (1) {
		// sate
		cv::Mat img(mask_building.rows, mask_building.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (size_t m = 0; m < sate_buildings.size(); m++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			int xc = 0, yc = 0;
			for (int j = 0; j < sate_buildings[m].size(); j++) {
				int x = int(sate_buildings[m][j](0));
				int y = int(sate_buildings[m][j](1));
				xc += x;
				yc += y;
				cv::circle(img, cv::Point(x, y), 2, cv::Scalar(r, g, b), 1);
			}
			xc /= sate_buildings[m].size();
			yc /= sate_buildings[m].size();
			cv::putText(img, std::to_string(m), cv::Point(xc, yc), 1, 2, cv::Scalar(r, g, b), 2);
		}
		cv::imwrite(fold_ + "\\buildings_sate.bmp", img);;
	}

	//////////////////////////////////
	// step3: hypotheses generation 
	//////////////////////////////////
	HypothesesViaBuildingMatching(street_buildings, sate_buildings);

	//////////////////////////////////
	// step4: global registration 
	//////////////////////////////////
	std::string file_trans = fold_ + "\\xy_alignment.txt";
	std::vector<cv::Vec6d> trans_xy;
	if (!std::experimental::filesystem::exists(file_trans)) {		
		GlobalAlignment(street_buildings, sate_buildings, trans_xy);

		// write out
		std::ofstream off(file_trans);
		off << std::setprecision(12) << std::fixed;
		for (size_t i = 0; i < trans_xy.size(); i++) {
			off << street_cams_id[i].val[0] << " " << street_cams_id[i].val[1] << " ";
			off << scale_ << " " << trans_xy[i].val[3] << " " << trans_xy[i].val[4] << " " << trans_xy[i].val[5] << std::endl;
		}
		off.close();
	}
}

void RegistrationNoGPS::TrajectorySegmentation(std::vector<Eigen::Vector2d> cams, std::vector<std::vector<int>>& cams_segs)
{
	// step1: calculate the normal of each point
	int span = 10;
	std::vector<Eigen::Vector2d> normals(cams.size());
	for (int i = 0; i < cams.size(); i++) {
		int id1 = std::max(i - span, 0);
		int id2 = std::min(i + span, int(cams.size()) - 1);
		normals[i] = cams[id2] - cams[id1];
		normals[i] /= normals[i].norm();
		//std::cout << i << " " << normals[i].transpose() << std::endl;
	}

	// step2: find the corners
	double th_cos = cos(80.0 / 180.0*CV_PI);
	Eigen::Vector2d normal_s = normals[0];
	std::vector<int> seg_temp;
	seg_temp.push_back(0);
	int seg_size = 300;
	for (int i = 1; i < normals.size(); i++) {
		double cos_v = abs(normal_s.dot(normals[i]));
		if ((cos_v < th_cos && seg_temp.size()>seg_size) || (i == (normals.size() - 1))) {
			cams_segs.push_back(seg_temp);
			normal_s = cams[seg_temp[0]] - cams[seg_temp[seg_temp.size() - 1]];
			normal_s /= normal_s.norm();
			normal_s = Eigen::Vector2d(-normal_s(1), normal_s(0));
			seg_temp.clear();
		}
		seg_temp.push_back(i);
	}
}

void RegistrationNoGPS::StreetBuildingSegmentation(std::vector<Eigen::Vector2d>& pts, float th_dev, 
	std::vector<std::vector<Eigen::Vector2d>>& segments_pts, std::vector<std::vector<int>>& segments_ids)
{
	// step1: scaling
	std::vector<Eigen::Vector2d> pts_new(pts.size());
	double xmin = 1000000.0, xmax = -xmin;
	double ymin = 1000000.0, ymax = -ymin;
	for (size_t i = 0; i < pts.size(); i++) {
		pts_new[i] = scale_ * pts[i];
		if (pts_new[i](0) < xmin) { xmin = pts_new[i](0); }
		if (pts_new[i](0) > xmax) { xmax = pts_new[i](0); }
		if (pts_new[i](1) < ymin) { ymin = pts_new[i](1); }
		if (pts_new[i](1) > ymax) { ymax = pts_new[i](1); }
	}

	int cols = xmax - xmin;
	int rows = ymax - ymin;
	std::map<int, int> pts_map;
	for (size_t i = 0; i < pts_new.size(); i++) {
		int x = int(pts_new[i](0) - xmin);
		int y = int(pts_new[i](1) - ymin);
		pts_map.insert(std::pair<int, int>(y*cols + x, i));
	}

	if (0) {
		cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (auto iter = pts_map.begin(); iter != pts_map.end(); iter++) {
			int id = iter->second;
			int x = int(pts_new[id](0) - xmin);
			int y = int(pts_new[id](1) - ymin);
			cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 1);
		}
		cv::imwrite("F:\\img_pts1.bmp", img);
	}

	// step2: segmentation
	PointCloud2d<double> cloud;
	std::vector<int> cloud_ids;
	for (auto iter = pts_map.begin(); iter != pts_map.end(); iter++) {
		int id = iter->second;
		cloud.pts.push_back(PointCloud2d<double>::PtData(pts_new[id](0), pts_new[id](1)));
		cloud_ids.push_back(id);
	}
	
	typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud2d<double> >, PointCloud2d<double>, 2/*dim*/ > my_kd_tree_t;
	my_kd_tree_t index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	double radius1 = 2.0;
	double radius2 = 5.0;
	int count_inliers = 0;
	std::vector<int> is_outliers(cloud.pts.size(), 0);
	std::vector<std::vector<int>> idx_neighs(cloud.pts.size());
	std::vector<Eigen::Vector2d> normals(cloud.pts.size());
	std::vector<cv::Vec2f> carvtures(cloud.pts.size());
	for (size_t i = 0; i < cloud.pts.size(); i++) {
		double *query_pt = new double[2];
		query_pt[0] = cloud.pts[i].x;  
		query_pt[1] = cloud.pts[i].y;

		std::vector<std::pair<size_t, double> > idx_dis_result1;
		RadiusResultSet<double, size_t> result_set1(radius1, idx_dis_result1);
		index.findNeighbors(result_set1, query_pt, nanoflann::SearchParams());

		std::vector<std::pair<size_t, double> > idx_dis_result2;
		RadiusResultSet<double, size_t> result_set2(radius2, idx_dis_result2);
		index.findNeighbors(result_set2, query_pt, nanoflann::SearchParams());

		// judge1
		if (idx_dis_result2.size() - idx_dis_result1.size() <= 1) {
			carvtures[i] = cv::Vec2f(i, 1.0);
			is_outliers[i] = 1;
			continue;
		}

		// normal
		std::vector<Eigen::Vector2d> pts_temp;
		for (size_t j = 0; j < idx_dis_result2.size(); j++) {
			int id_temp = idx_dis_result2[j].first;
			idx_neighs[i].push_back(id_temp);
			pts_temp.push_back(Eigen::Vector2d(cloud.pts[id_temp].x, cloud.pts[id_temp].y));
		}
		Eigen::Vector2d normal_temp;
		double curve_temp;
		PCANormal(pts_temp, normal_temp, curve_temp);
		normals[i] = normal_temp;
		carvtures[i] = cv::Vec2f(i, curve_temp);
	}
	std::sort(carvtures.begin(), carvtures.end(), [](const cv::Vec2f& lhs, const cv::Vec2f& rhs) { return lhs.val[1] < rhs.val[1]; });


	if (0) {
		cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (int j = 0; j < cloud.pts.size(); j++) {
			if (!is_outliers[j]) {
				int x = int(cloud.pts[j].x - xmin);
				int y = int(cloud.pts[j].y - ymin);
				cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 1);
			}			
		}
		cv::imwrite("F:\\img_pts2.bmp", img);
	}

	// step3: line fitting
	std::vector<std::vector<int>> line_segs;
	std::vector<int> is_used = is_outliers;
	for (size_t i = 0; i < carvtures.size(); i++) {
		int id = carvtures[i].val[0];
		if (is_used[id]) {
			continue;
		}

		std::vector<int> seg_temp;
		seg_temp.push_back(id);
		int count = 0;
		while (count < seg_temp.size()) {
			int id0 = seg_temp[count];
			Eigen::Vector2d v_normal = normals[id0];
			for (size_t j = 0; j < idx_neighs[id0].size(); j++) {
				int id1 = idx_neighs[id0][j];
				if (is_used[id1]) {
					continue;
				}
				Eigen::Vector2d v_dis(cloud.pts[id0].x - cloud.pts[id1].x, cloud.pts[id0].y - cloud.pts[id1].y);
				double dis_ortho = v_normal.dot(v_dis);
				if (abs(dis_ortho)<2.0) {
					seg_temp.push_back(id1);
					is_used[id1] = 1;
				}
			}
			count++;
		}

		if (seg_temp.size() > 30) {
			double xmin = 10000000, xmax = -xmin;
			double ymin = 10000000, ymax = -ymin;
			for (size_t j = 0; j < seg_temp.size(); j++) {
				int id = seg_temp[j];
				if (cloud.pts[id].x < xmin) { xmin = cloud.pts[id].x; }
				if (cloud.pts[id].x > xmax) { xmax = cloud.pts[id].x; }
				if (cloud.pts[id].y < ymin) { ymin = cloud.pts[id].y; }
				if (cloud.pts[id].y > ymax) { ymax = cloud.pts[id].y; }
			}
			if (abs(xmax - xmin) + abs(ymax - ymin)>30) {
				line_segs.push_back(seg_temp);
			}
		}
	}

	if (0) {
		cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (size_t m = 0; m < line_segs.size(); m++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			for (int j = 0; j < line_segs[m].size(); j++) {
				int id = line_segs[m][j];
				int x = int(cloud.pts[id].x - xmin);
				int y = int(cloud.pts[id].y - ymin);
				cv::circle(img, cv::Point(x, y), 1, cv::Scalar(r, g, b), 1);
			}
		}
		
		cv::imwrite("F:\\img_pts3.bmp", img);
	}

	// step4: line merging to fit for building
	std::vector<int> cloud_seg_ids;
	PointCloud2d<double> cloud_seg;
	std::vector<int> pts_lineid;
	for (size_t i = 0; i < line_segs.size(); i++) {
		for (size_t j = 0; j < line_segs[i].size(); j++) {
			int id_temp = line_segs[i][j];
			cloud_seg.pts.push_back(cloud.pts[id_temp]);
			cloud_seg_ids.push_back(cloud_ids[id_temp]);
			pts_lineid.push_back(i);
			line_segs[i][j] = pts_lineid.size() - 1;
		}
	}
	my_kd_tree_t index_seg(2 /*dim*/, cloud_seg, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index_seg.buildIndex();

	double radius_seg = 100;
	std::vector<std::vector<int>> idx_neighs_seg(cloud_seg.pts.size());
	for (size_t i = 0; i < cloud_seg.pts.size(); i++) {
		double *query_pt = new double[2];
		query_pt[0] = cloud_seg.pts[i].x;
		query_pt[1] = cloud_seg.pts[i].y;

		std::vector<std::pair<size_t, double> > idx_dis_result;
		RadiusResultSet<double, size_t> result_set(radius_seg, idx_dis_result);
		index_seg.findNeighbors(result_set, query_pt, nanoflann::SearchParams());
		for (size_t j = 0; j < idx_dis_result.size(); j++) {
			int id_temp = idx_dis_result[j].first;
			idx_neighs_seg[i].push_back(id_temp);
		}
	}

	if (0) {
		cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (size_t i = 0; i < cloud_seg.pts.size(); i++) {
			if (abs(cloud_seg.pts[i].x - xmin - 166)<5 && abs(cloud_seg.pts[i].y - ymin - 795)<5) {
				for (int j = 0; j < idx_neighs_seg[i].size(); j++) {
					int idd = idx_neighs_seg[i][j];
					int x = int(cloud_seg.pts[idd].x - xmin);
					int y = int(cloud_seg.pts[idd].y - ymin);
					cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 1);
				}
			}
		}

		cv::imwrite("F:\\img_pts4.bmp", img);
	}

	int id_new = line_segs.size() + 1;
	bool do_merge = true;
	while (do_merge) {
		do_merge = false;
		for (size_t i = 0; i < line_segs.size(); i++) {
			for (size_t j = 0; j < line_segs[i].size(); j++) {
				int id0 = line_segs[i][j];				
				for (int m = 0; m < idx_neighs_seg[id0].size(); m++) {
					int id1 = idx_neighs_seg[id0][m];
					if (pts_lineid[id1] != pts_lineid[id0]) {
						int label1 = pts_lineid[id1];
						int label2 = pts_lineid[id0];
						do_merge = true;
						for (size_t n = 0; n < pts_lineid.size(); n++) {
							if (pts_lineid[n] == label1 || pts_lineid[n] == label2) {
								pts_lineid[n] = id_new;
							}
						}
						id_new++;
					}
					if (do_merge) { break; }
				}
				if (do_merge) { break; }
			}
			if (do_merge) { break; }
		}
	}

	std::vector<std::vector<int>> line_clusters(id_new);
	for (size_t i = 0; i < line_segs.size(); i++) {
		int id = line_segs[i][0];
		int id_cluster = pts_lineid[id];
		line_clusters[id_cluster].push_back(i);
	}

	for (size_t i = 0; i < line_clusters.size(); i++) {
		if (line_clusters[i].size()) {
			std::vector<Eigen::Vector2d> seg_pts_temp;
			std::vector<int> seg_ids_temp;
			double xmin = 10000000, xmax = -xmin;
			double ymin = 10000000, ymax = -ymin;
			for (size_t j = 0; j < line_clusters[i].size(); j++) {
				int id_line = line_clusters[i][j];
				for (size_t m = 0; m < line_segs[id_line].size(); m++) {
					int id_pt = line_segs[id_line][m];
					seg_pts_temp.push_back(Eigen::Vector2d(cloud_seg.pts[id_pt].x, cloud_seg.pts[id_pt].y));
					seg_ids_temp.push_back(cloud_seg_ids[id_pt]);
					if (cloud_seg.pts[id_pt].x < xmin) { xmin = cloud_seg.pts[id_pt].x; }
					if (cloud_seg.pts[id_pt].x > xmax) { xmax = cloud_seg.pts[id_pt].x; }
					if (cloud_seg.pts[id_pt].y < ymin) { ymin = cloud_seg.pts[id_pt].y; }
					if (cloud_seg.pts[id_pt].y > ymax) { ymax = cloud_seg.pts[id_pt].y; }
				}
			}
			if (abs(xmax - xmin) + abs(ymax - ymin)>60) {
				segments_pts.push_back(seg_pts_temp);
				segments_ids.push_back(seg_ids_temp);
			}			
		}
	}

	if (0) {
		cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (size_t m = 0; m < segments_pts.size(); m++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			int xc = 0, yc = 0;
			for (int j = 0; j < segments_pts[m].size(); j++) {
				int x = int(segments_pts[m][j](0) - xmin);
				int y = int(segments_pts[m][j](1) - ymin);
				xc += x;
				yc += y;
				cv::circle(img, cv::Point(x, y), 1, cv::Scalar(r, g, b), 1);
			}
			xc /= segments_pts[m].size();
			yc /= segments_pts[m].size();
			cv::putText(img, std::to_string(m), cv::Point(xc, yc), 1, 2, cv::Scalar(r, g, b), 2);
		}

		cv::imwrite("F:\\img_pts5.bmp", img);
	}
}

void RegistrationNoGPS::SateBuildingSegmentation(cv::Mat & mask_building, std::vector<std::vector<Eigen::Vector2d>>& segments)
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask_building, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	
	for (size_t i = 0; i < contours.size(); i++) {
		if (contours[i].size()>100 && contours[i].size()<4000) {
			std::vector<Eigen::Vector2d> seg_temp;
			for (size_t j = 0; j < contours[i].size(); j++) {
				seg_temp.push_back(Eigen::Vector2d(contours[i][j].x, contours[i][j].y));
			}
			segments.push_back(seg_temp);
		}
	}
}

void RegistrationNoGPS::GlobalAlignment(std::vector<std::vector<Eigen::Vector2d>> src_list, 
	std::vector<std::vector<Eigen::Vector2d>> dst_list, std::vector<cv::Vec6d>& src_trans)
{
	std::vector<std::vector<Eigen::Vector2d>> src_list_ori = src_list;
	std::vector<std::vector<Eigen::Vector2d>> dst_list_ori = dst_list;

	// step1: read in hypotheses
	std::string file_trans_filter = fold_ + "\\trans_filtered.txt";
	std::ifstream iff = std::ifstream(file_trans_filter);	
	std::vector<cv::Vec6f> trans;
	double id1, id2;
	double score, angle, tx, ty;
	while (!iff.eof()) {
		iff >> id1 >> id2 >> score >> angle >> tx >> ty;
		trans.push_back(cv::Vec6f(id1, id2, score, angle, tx, ty));
	}
	iff.close();

	// step2: graph cut
	int n_node = src_list.size();
	int n_label = trans.size();

	// a. data term
	std::vector<std::vector<float>> costs;
	CostTransformations(src_list_ori, dst_list_ori, trans, costs);
	int* data_term = new int[n_node*n_label];
	for (int i = 0; i < n_node; i++) {
		for (int j = 0; j < n_label; j++) {
			data_term[i*n_label + j] = costs[i][j] + 1;
		}
	}

	// b. smooth term
	int count_pts = 0;
	for (size_t i = 0; i < src_list_ori.size(); i++) {
		count_pts += src_list_ori[i].size();
	}

	double th_angle = 20.0 / 180.0*CV_PI;
	double th_t = 50.0;
	int penalty = std::max(count_pts / 100, 20);
	std::cout << "penalty " << penalty << std::endl;
	int* smooth_term = new int[n_label*n_label];
	int count = 0;
	for (int i = 0; i < n_label; i++) {
		for (int j = 0; j < n_label; j++) {
			smooth_term[count] = 0;
			if (i == j) {
				count++;
				continue;
			}

			double dev_angle = abs(trans[i].val[3] - trans[j].val[3]);
			dev_angle = std::min(dev_angle, CV_PI * 2 - dev_angle);
			double dev_t = abs(trans[i].val[4] - trans[j].val[4]) + abs(trans[i].val[5] - trans[j].val[5]);
			double p_angle = std::max(dev_angle, th_angle) / th_angle * penalty + 1;
			double p_t = std::max(dev_t, th_t) / th_t * penalty + 1;
			smooth_term[count] = p_angle + p_t;
			count++;
		}
	}

	GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(n_node, n_label);
	gc->setDataCost(data_term);
	gc->setSmoothCost(smooth_term);

	// c. neighbors
	int n_neighbors = 1;
	for (int i = 0; i < src_list_ori.size() - 1; i++) {
		int ids = i + 1;
		int ide = std::min(i + n_neighbors, int(src_list_ori.size() - 1));
		for (int j = ids; j <= ide; j++) {
			gc->setNeighbors(i, j);
		}
	}

	printf("\nBefore optimization energy is %d", gc->compute_energy());
	gc->expansion(10);
	printf("\nAfter optimization energy is %d\n", gc->compute_energy());

	src_trans.resize(src_list.size());
	for (size_t ii = 0; ii < src_list.size(); ii++) {
		int l = gc->whatLabel(ii);
		src_trans[ii] = trans[l];
		std::cout << l << " " << src_trans[ii] << std::endl;
	}
}

void RegistrationNoGPS::HypothesesViaBuildingMatching(std::vector<std::vector<Eigen::Vector2d>> &src_list, std::vector<std::vector<Eigen::Vector2d>> &dst_list)
{
	// step1: hypotheses generation
	std::vector<std::vector<Eigen::Vector2d>> src_list_ori = src_list;
	std::vector<std::vector<Eigen::Vector2d>> dst_list_ori = dst_list;

	// src center
	std::vector<Eigen::Vector2d> c_srcs(src_list.size());
	for (size_t i = 0; i < src_list.size(); i++) {
		c_srcs[i] = Eigen::Vector2d(0, 0);
		for (size_t j = 0; j < src_list[i].size(); j++) {
			c_srcs[i] += src_list[i][j];
		}
		c_srcs[i] /= src_list[i].size();

		for (size_t j = 0; j < src_list[i].size(); j++) {
			src_list[i][j] -= c_srcs[i];
		}
	}

	// rotated srcs
	int th_size = 200;
	float step_angle = 3.0 / 180.0*CV_PI;
	int n_angle = 2 * CV_PI / step_angle;
	std::vector<std::vector<std::vector<Eigen::Vector2d>>> src_rotated(src_list.size());
	for (size_t i = 0; i < src_list.size(); i++) {
		src_rotated[i].resize(n_angle);
		for (int j = 0; j < n_angle; j++) {
			// rotate src
			float angle_j = j * step_angle;
			Eigen::Matrix2d R;
			R << cos(angle_j), -sin(angle_j), sin(angle_j), cos(angle_j);

			src_rotated[i][j].resize(th_size);
			float step = float(src_list[i].size() - 2) / th_size;
			for (int m = 0; m < th_size; m++) {
				int id = int(m*step);
				src_rotated[i][j][m] = R * src_list[i][id];
			}
		}
	}

	// dst box
	std::vector<cv::Vec4d> bbox_dsts(dst_list.size());
	for (size_t i = 0; i < dst_list.size(); i++) {
		double xmin = 10000000, ymin = xmin, xmax = -xmin, ymax = -ymin;
		for (size_t j = 0; j < dst_list[i].size(); j++) {
			if (dst_list[i][j](0) < xmin) xmin = dst_list[i][j](0);
			if (dst_list[i][j](0) > xmax) xmax = dst_list[i][j](0);
			if (dst_list[i][j](1) < ymin) ymin = dst_list[i][j](1);
			if (dst_list[i][j](1) > ymax) ymax = dst_list[i][j](1);
		}
		bbox_dsts[i] = cv::Vec4d(xmin, ymin, xmax, ymax);
		for (size_t j = 0; j < dst_list[i].size(); j++) {
			dst_list[i][j] -= Eigen::Vector2d(xmin, ymin);
		}
	}

	// matching
	std::string file_trans = fold_ + "\\trans_ori.txt";;
	int offset = 10;
	int num_sample = 10;
	if (!std::experimental::filesystem::exists(file_trans)) {
		std::vector<std::vector<cv::Vec6f>> matches(dst_list.size());
		for (int k = 0; k < dst_list.size(); k++) {
			std::cout << k << std::endl;

			// distance transformation
			std::vector<std::vector<cv::Point>> contour(1);
			for (int i = 0; i < dst_list[k].size(); i++) {
				contour[0].push_back(cv::Point(dst_list[k][i](0) + offset, dst_list[k][i](1) + offset));
			}

			int cols = bbox_dsts[k].val[2] - bbox_dsts[k].val[0] + 2 * offset;
			int rows = bbox_dsts[k].val[3] - bbox_dsts[k].val[1] + 2 * offset;
			cv::Mat edge_map(rows, cols, CV_8UC1, cv::Scalar(255));
			cv::drawContours(edge_map, contour, -1, cv::Scalar(0));
			cv::Mat dist_map, labels_map;
			cv::distanceTransform(edge_map, dist_map, labels_map, CV_DIST_L2, CV_DIST_MASK_5, cv::DIST_LABEL_CCOMP);
			//cv::imwrite("F:\\edge_map.bmp", edge_map);
			//cv::imwrite("F:\\dist_map.bmp", 10 * dist_map);
			//cv::imwrite("F:\\labels_map.bmp", labels_map);

			// matching
			for (int i = 0; i < src_list.size(); i++) {
				std::vector<cv::Vec6f>  trans_temp(n_angle);
#pragma omp parallel for
				for (int j = 0; j < n_angle; j++) {
					//j = 1.185688138008 / step_angle;
					double error_min = 100000000.0;
					double offx_min = 0.0, offy_min = 0.0;
					for (int m = 0; m < num_sample; m++) {
						int id1 = m * dst_list[k].size() / num_sample;
						for (int n = 0; n < src_rotated[i][j].size(); n += 2) {
							int id2 = n;
							double offx = dst_list[k][id1](0) - src_rotated[i][j][id2](0);
							double offy = dst_list[k][id1](1) - src_rotated[i][j][id2](1);

							// calculate error
							double error_temp = 0;
							int loc = 0;
							float* ptr_dist = (float*)dist_map.data;
							for (int p = 0; p < src_rotated[i][j].size(); p++) {
								int x = src_rotated[i][j][p](0) + offx + offset;
								int y = src_rotated[i][j][p](1) + offy + offset;
								if (x<0 || y<0 || x >= cols || y >= rows) {
									error_temp += 5;
								}
								else {
									int loc_cur = y * cols + x;
									ptr_dist += loc_cur - loc;
									loc = loc_cur;
									float dis = *ptr_dist;
									if (dis > 5.0) {
										error_temp += 1;
									}
								}																
							}
							if (error_temp < error_min) {
								error_min = error_temp;
								offx_min = offx;
								offy_min = offy;
							}
						}
					}

					float angle = j * step_angle;
					Eigen::Matrix2d R;
					R << cos(angle), -sin(angle), sin(angle), cos(angle);
					Eigen::Vector2d t(offx_min, offy_min);
					t = -R * c_srcs[i] + t + Eigen::Vector2d(bbox_dsts[k].val[0], bbox_dsts[k].val[1]);
					trans_temp[j] = cv::Vec6f(i, k, error_min, angle, t(0), t(1));

					if (0) {
						cv::Mat img_match = dist_map.clone();
						img_match = 10 * img_match;
						img_match.convertTo(img_match, CV_8U);
						cv::cvtColor(img_match, img_match, CV_GRAY2BGR);
						for (size_t p = 0; p < src_rotated[i][j].size(); p++) {
							int x = src_rotated[i][j][p](0) + offx_min + offset;
							int y = src_rotated[i][j][p](1) + offy_min + offset;
							cv::circle(img_match, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), 1);
						}
						cv::imwrite("F:\\DevelopCenter\\papers\\ISPRS2019\\experimentals\\building_matching\\" + std::to_string(j) + ".bmp", img_match);
					}

					if (0) {
						int id_src = i;
						int id_dst = k;
						std::string file_src = "F:\\src" + std::to_string(id_src) + ".txt";
						std::string file_dst = "F:\\dst" + std::to_string(id_dst) + ".txt";
						std::string file_rst = "F:\\rst" + std::to_string(id_src) + ".txt";

						std::ofstream off1(file_src);
						std::ofstream off2(file_rst);
						off1 << std::setprecision(12) << std::fixed;
						off2 << std::setprecision(12) << std::fixed;
						for (size_t p = 0; p < src_rotated[i][j].size(); p++) {
							off1 << src_rotated[i][j][p](0) << " " << src_rotated[i][j][p](1) << " " << 1.0 << std::endl;
							off2 << src_rotated[i][j][p](0) + offx_min << " " << src_rotated[i][j][p](1) + offy_min << " " << 1.0 << std::endl;
						}
						off1.close();
						off2.close();

						std::ofstream off3(file_dst);
						off3 << std::setprecision(12) << std::fixed;
						for (size_t p = 0; p < dst_list[id_dst].size(); p++) {
							off3 << dst_list[id_dst][p](0) << " " << dst_list[id_dst][p](1) << " " << 1.0 << std::endl;
						}
						off3.close();
					}
				}

				// sort
				// std::sort(trans_temp.begin(), trans_temp.end(), [](const cv::Vec6f& lhs, const cv::Vec6f& rhs) { return lhs.val[0] < rhs.val[0]; });
				for (size_t i = 0; i < trans_temp.size(); i++) {
					matches[k].push_back(trans_temp[i]);
				}
			}
		}

		// write out
		std::ofstream off(file_trans);
		off << std::setprecision(12) << std::fixed;
		for (size_t i = 0; i < matches.size(); i++) {
			int tt = int(matches[i].size());
			for (size_t j = 0; j < tt; j++) {
				for (size_t n = 0; n < 6; n++) {
					off << matches[i][j].val[n] << " ";
				}
				off << std::endl;
			}
		}
	}

	// step2: hypotheses filtering
	std::ifstream iff(file_trans);
	std::vector<cv::Vec6f> trans;
	double id1, id2;
	double score, angle, tx, ty;
	while (!iff.eof()) {
		iff >> id1 >> id2 >> score >> angle >> tx >> ty;
		trans.push_back(cv::Vec6f(id1, id2, score, angle, tx, ty));
	}
	iff.close();

	// step2: filter transformation via neighboring points
	std::string file_trans_filter = fold_ + "\\trans_filtered.txt";
	if (!std::experimental::filesystem::exists(file_trans_filter)) {
		HypothesesFiltering(src_list_ori, dst_list_ori, trans);
	}
}

void RegistrationNoGPS::HypothesesFiltering(std::vector<std::vector<Eigen::Vector2d>>& src_list, std::vector<std::vector<Eigen::Vector2d>>& dst_list, std::vector<cv::Vec6f>& trans)
{
	// build kdtree 
	PointCloud2d<double> cloud;
	for (size_t i = 0; i < dst_list.size(); i++) {
		for (size_t j = 0; j < dst_list[i].size(); j++) {
			cloud.pts.push_back(PointCloud2d<double>::PtData(dst_list[i][j](0), dst_list[i][j](1)));
		}

	}
	typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud2d<double> >, PointCloud2d<double>, 2/*dim*/ > my_kd_tree_t;
	my_kd_tree_t index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	// calculate the error for each trans
	int span = 3;
	float th_dis = 10.0;
	std::vector<int> is_ok(trans.size(), 0);
#pragma omp parallel for
	for (int i = 0; i < trans.size(); i++) {
		if (i % 1000 == 0) {
			std::cout << i << std::endl;
		}

		int id_src = trans[i].val[0];
		double angle = trans[i].val[3];
		double tx = trans[i].val[4];
		double ty = trans[i].val[5];
		Eigen::Matrix2d R;
		R << cos(angle), -sin(angle), sin(angle), cos(angle);
		Eigen::Vector2d t(tx, ty);

		int id_s = id_src;
		if (id_s >= src_list.size() - span) {
			id_s = src_list.size() - span;
		}
		int id_e = id_s + span;

		std::vector<Eigen::Vector2d> pts;
		for (size_t j = id_s; j < id_e; j++) {
			for (size_t m = 0; m < src_list[j].size(); m++) {
				pts.push_back(R*src_list[j][m] + t);
			}
		}

		int count_inliers = 0;
		for (size_t j = 0; j < pts.size(); j++) {
			double *query_pt = new double[2];
			query_pt[0] = pts[j](0);
			query_pt[1] = pts[j](1);
			double dis_temp = 0.0;
			size_t idx_temp = 0;
			nanoflann::KNNResultSet<double> result_set(1);
			result_set.init(&idx_temp, &dis_temp);
			index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));
			double dx = cloud.pts[idx_temp].x - pts[j](0);
			double dy = cloud.pts[idx_temp].y - pts[j](1);
			if (abs(dx) + abs(dy) < th_dis) {
				count_inliers++;
			}
		}
		//if (count_inliers > pts.size()*0.75) {
		if (count_inliers > pts.size()*0.5) {
			is_ok[i] = 1;
		}
	}

	//
	std::vector<cv::Vec6f> trans_new;
	for (size_t i = 0; i < trans.size(); i++) {
		if (is_ok[i]) {
			trans_new.push_back(trans[i]);
		}
	}
	std::cout << "trans before " << trans.size();
	std::cout << " trans after " << trans_new.size() << std::endl;
	trans = trans_new;
}

void RegistrationNoGPS::AlignmentZ()
{
	std::string file_z_align = fold_ + "\\z_alignment.txt";
	if (std::experimental::filesystem::exists(file_z_align)) {
		return;
	}

	// step1: detect ground pts
	DetectGround();

	// read in vetical
	std::string file_refined = fold_ + "\\vertical_refined.txt";
	std::ifstream ff1(file_refined);
	ff1 >> dir_vertial_.x >> dir_vertial_.y >> dir_vertial_.z;
	ff1.close();
	Eigen::Vector3d vz(dir_vertial_.x, dir_vertial_.y, dir_vertial_.z);
	Eigen::Vector3d vx(1.0, 1.0, -(vz(0) + vz(1)) / vz(2));  vx /= vx.norm();
	Eigen::Vector3d vy = vz.cross(vx);  vy /= vy.norm();

	// read in ground
	std::string file_ground = fold_ + "\\ground.txt";
	std::vector<Eigen::Vector3d> pts_ground;
	std::ifstream ff(file_ground);
	float x, y, z;
	while (!ff.eof()) {
		ff >> x >> y >> z;
		Eigen::Vector3d t(x, y, z);
		pts_ground.push_back(Eigen::Vector3d(t.dot(vx), t.dot(vy), t.dot(vz)));
	}
	ff.close();

	// read in cams
	std::string file_pose = file_street_pose_;
	std::ifstream iff(file_pose);
	std::string temp;
	std::getline(iff, temp);
	std::getline(iff, temp);
	std::string name;
	float rx, ry, rz;
	std::vector<Eigen::Vector3d> cams;
	while (!iff.eof()) {
		iff >> name >> x >> y >> z >> rx >> ry >> rz;
		Eigen::Vector3d t(x, y, z);
		cams.push_back(Eigen::Vector3d(t.dot(vx), t.dot(vy), t.dot(vz)));
	}
	iff.close();
	//std::cout << cams[0] << std::endl;
	//std::cout << cams[cams.size() - 50] << std::endl;

	// read in transformation
	std::string file_trans = fold_ + "\\xy_alignment.txt";
	std::ifstream fff(file_trans);
	std::vector<cv::Vec2i> idcam_trans;
	std::vector<double> s_trans, angles_trans;
	std::vector<Eigen::Vector2d> t_trans;
	int ids, ide;
	double s, angle, tx, ty;
	while (!fff.eof()) {
		fff >> ids >> ide >> s >> angle >> tx >> ty;
		idcam_trans.push_back(cv::Vec2i(ids, ide));
		s_trans.push_back(s);
		angles_trans.push_back(angle);
		t_trans.push_back(Eigen::Vector2d(tx, ty));
	}
	fff.close();

	// step1: ground pts from satellite
	std::string file_mask_road = "";
	if (!std::experimental::filesystem::exists(file_mask_road)) {
		std::string file_mask_building = file_mask_building_;
		std::string file_mask_ndvi = file_mask_ndvi_;
		cv::Mat mask_build = cv::imread(file_mask_building, 0);
		cv::Mat mask_ndvi = cv::imread(file_mask_ndvi, 0);
		cv::Mat mask_road = 255 - mask_build - mask_ndvi;
		cv::imwrite(file_mask_road, mask_road);
	}
	cv::Mat mask_road = cv::imread(file_mask_road, 0);
	mask_road.convertTo(mask_road, CV_32FC1);
	mask_road /= 255;

	cv::Mat dsm(mask_road.rows, mask_road.cols, CV_32FC1, cv::Scalar(0.0));
	std::string file_sate_dsm = file_dsm_;
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(file_sate_dsm.c_str());
	TinyTIFFReader_getSampleData(tiffr, (float*)dsm.data, 0);
	TinyTIFFReader_close(tiffr);
	dsm = dsm.mul(mask_road);
	dsm = -dsm;
	dsm /= 0.5;
	int sate_span = 20;

	std::vector<std::vector<Eigen::Vector3d>> cam_pts_sate(cams.size());
	std::vector<double> cam_scales(cams.size());
	for (int i = 0; i < cams.size(); i++) {
		// interpolate for the trans
		double w_avg = 0.0, angle_avg = 0.0, s_avg = 0.0;
		Eigen::Vector2d t_avg(0, 0);
		for (int j = 0; j < idcam_trans.size(); j++) {
			double dis_min = std::min(abs(i - idcam_trans[j].val[0]), abs(i - idcam_trans[j].val[1]));
			double w = exp(-dis_min / 100.0);
			w_avg += w;
			angle_avg += angles_trans[j] * w;
			s_avg += s_trans[j] * w;
			t_avg += t_trans[j] * w;
		}
		angle_avg /= w_avg;
		s_avg /= w_avg;
		t_avg /= w_avg;
		Eigen::Matrix2d R_avg;
		R_avg << cos(angle_avg), -sin(angle_avg), sin(angle_avg), cos(angle_avg);

		cam_scales[i] = s_avg;

		// convert to the satellite image
		Eigen::Vector2d pt_sate = s_avg * R_avg*Eigen::Vector2d(cams[i](0), cams[i](1)) + t_avg;
		int xs = std::max(int(pt_sate(0) - sate_span), 0);
		int xe = std::min(int(pt_sate(0) + sate_span), mask_road.cols);
		int ys = std::max(int(pt_sate(1) - sate_span), 0);
		int ye = std::min(int(pt_sate(1) + sate_span), mask_road.rows);

		// find the road pixels		
		std::vector<Eigen::Vector3d> pts_road;
		for (int y = ys; y < ye; y++) {
			float* ptr = dsm.ptr<float>(y) + xs;
			for (int x = xs; x < xe; x++) {
				if (*ptr != 0 && (*ptr == *ptr)) {
					pts_road.push_back(Eigen::Vector3d(x, y, *ptr));
				}
				ptr++;
			}
		}
		if (!pts_road.size()) {
			continue;
		}
		std::sort(pts_road.begin(), pts_road.end(), [](const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) { return lhs(2) > rhs(2); });
		int idxs = pts_road.size()*0.1;
		int idxe = pts_road.size()*0.6;
		//std::vector<double>tt;
		for (int j = idxs; j < idxe; j++) {
			cam_pts_sate[i].push_back(pts_road[j]);
			//tt.push_back(pts_road[j](2));
		}
	}

	// step2: ground pts from streetview
	int num_sample = cams.size() * 10;
	int step = std::max(int(pts_ground.size() / num_sample), 1);
	std::vector<std::pair<int, int>> pt_cam;
	for (size_t i = 0; i < pts_ground.size(); i += step) {
		double dis_min = 100000000.0;
		int idx_min = 0;
		for (size_t j = 0; j < cams.size(); j++) {
			double dis = abs(cams[j](0) - pts_ground[i](0)) + abs(cams[j](1) - pts_ground[i](1));
			if (dis < dis_min) {
				dis_min = dis;
				idx_min = j;
			}
		}
		pt_cam.push_back(std::pair<int, int>(i, idx_min));
	}

	std::vector<std::vector<Eigen::Vector3d>> cam_pts_street(cams.size());
	for (size_t i = 0; i < pt_cam.size(); i++) {
		int id_pt = pt_cam[i].first;
		int id_cam = pt_cam[i].second;
		cam_pts_street[id_cam].push_back(pts_ground[id_pt]);
	}

	// step3: weight z offset
	std::vector<cv::Vec2f> cam_devz;
	std::vector<cv::Vec2f> cam_z;
	if (0) {
		int span = cams.size() / 4;
		float sigma = cams.size() / 8;
		for (int i = 0; i < cams.size(); i++) {
			int ids_cam = max(i - span, 0);
			int ide_cam = min(i + span, int(cams.size() - 1));

			double w_avg = 0.0;
			double z_street_avg = 0.0, z_sate_avg = 0.0;
			for (int j = ids_cam; j < ide_cam; j++) {
				double w = exp(-abs(i - j) / sigma);
				double z_street = 0.0, z_sate = 0.0;
				if (!cam_pts_street[j].size() || !cam_pts_sate[j].size()) {
					continue;
				}
				for (int m = 0; m < cam_pts_street[j].size(); m++) {
					z_street += cam_pts_street[j][m](2);
				}
				for (int m = 0; m < cam_pts_sate[j].size(); m++) {
					z_sate += cam_pts_sate[j][m](2);
				}
				z_street /= cam_pts_street[j].size();
				z_sate /= cam_pts_sate[j].size();

				z_street_avg += z_street * w;
				z_sate_avg += z_sate * w;
				w_avg += w;
			}
			if (w_avg>0) {
				z_sate_avg /= w_avg;
				z_street_avg /= w_avg;
				cam_devz.push_back(cv::Vec2f(i, z_sate_avg - cam_scales[i] * z_street_avg));
				cam_z.push_back(cv::Vec2f(i, z_sate_avg));
			}
			//std::cout << i << " " << z_sate_avg << " " << z_street_avg << std::endl;
		}
	}
	else {
		for (int i = 0; i < cams.size(); i++) {
			if (!cam_pts_street[i].size() || !cam_pts_sate[i].size()) {
				continue;
			}

			double z_street_avg = 0.0, z_sate_avg = 0.0;
			for (int m = 0; m < cam_pts_street[i].size(); m++) {
				z_street_avg += cam_pts_street[i][m](2);
			}
			for (int m = 0; m < cam_pts_sate[i].size(); m++) {
				z_sate_avg += cam_pts_sate[i][m](2);
			}
			z_street_avg /= cam_pts_street[i].size();
			z_sate_avg /= cam_pts_sate[i].size();
			cam_devz.push_back(cv::Vec2f(i, z_sate_avg - cam_scales[i] * z_street_avg));
			cam_z.push_back(cv::Vec2f(i, z_sate_avg));
			//std::cout << i << " " << z_sate_avg << " " << z_street_avg << std::endl;
		}
	}

	double devz_avg = 0.0;
	for (int i = 0; i < cam_devz.size(); i++) {
		devz_avg += cam_devz[i].val[1];
	}
	devz_avg /= cam_devz.size();

	double devz_sigma = 0.0;
	for (int i = 0; i < cam_devz.size(); i++) {
		devz_sigma += pow(cam_devz[i].val[1] - devz_avg, 2);
	}
	devz_sigma = sqrt(devz_sigma / cam_devz.size());

	// write out
	std::ofstream off(file_z_align);
	off << std::setprecision(12) << std::fixed;
	for (int i = 0; i < cam_devz.size(); i++) {
		if (abs(cam_devz[i].val[1] - devz_avg) / devz_sigma >1.0) {
			//continue;
		}
		//off << int(cam_devz[i].val[0]) << " " << cam_devz[i].val[1];
		off << int(cam_devz[i].val[0]) << " " << cam_z[i].val[1];
		if (i<cam_devz.size() - 1) {
			off << std::endl;
		}
	}
	off.close();
}

void RegistrationNoGPS::DetectGround()
{
	std::string file_ground = fold_ + "\\ground.txt";
	if (std::experimental::filesystem::exists(file_ground)) {
		return;
	}

	// read in vetical
	std::string file_refined = fold_ + "\\vertical_refined.txt";
	std::ifstream ff1(file_refined);
	ff1 >> dir_vertial_.x >> dir_vertial_.y >> dir_vertial_.z;
	ff1.close();
	Eigen::Vector3d vz(dir_vertial_.x, dir_vertial_.y, dir_vertial_.z);
	Eigen::Vector3d vx(1.0, 1.0, -(vz(0) + vz(1)) / vz(2));  vx /= vx.norm();
	Eigen::Vector3d vy = vz.cross(vx);  vy /= vy.norm();

	// read in dense pts
	PointCloud3d<double> pts_dense;
	std::string file_dense = file_street_dense_;
	std::ifstream iff = std::ifstream(file_dense);
	float x, y, z, idx;
	int r, g, b;
	int count = 0;
	while (!iff.eof()) {
		iff >> x >> y >> z >> r >> g >> b >> idx;
		count++;
		if (count % 1 == 0) {
			count = 0;
			pts_dense.pts.push_back(PointCloud3d<double>::PtData(x, y, z));
		}
	}
	iff.close();

	// calculate normal of points
	PCAFunctions pcaer;
	int k = 30;
	std::vector<PCAInfo> pcaInfos;
	double scale, magnitd = 0.0;
	pcaer.Ori_PCA(pts_dense, k, pcaInfos, scale, magnitd);

	// detect ground
	std::vector<int> is_ground(pcaInfos.size(), 0);
	double th_dev = cos(5.0 / 180.0*CV_PI);
	for (size_t i = 0; i < pcaInfos.size(); i++)
	{
		double dev = dir_vertial_.x*pcaInfos[i].normal(0)
			+ dir_vertial_.y*pcaInfos[i].normal(1)
			+ dir_vertial_.z*pcaInfos[i].normal(2);
		if (abs(dev)>th_dev) {
			is_ground[i] = 1;
		}
	}

	// write out ground
	std::ofstream ff = std::ofstream(file_ground);
	ff << std::setprecision(12) << std::fixed;
	for (int i = 0; i < pcaInfos.size(); i++) {
		if (is_ground[i]) {
			int count_inliers = 0;
			for (int j = 0; j < pcaInfos[i].idxAll.size(); j++) {
				int id = pcaInfos[i].idxAll[j];
				if (is_ground[id]) {
					count_inliers++;
				}
			}
			if (count_inliers > 0.75*pcaInfos[i].idxAll.size()) {
				ff << pts_dense.pts[i].x << " " << pts_dense.pts[i].y << " " << pts_dense.pts[i].z << std::endl;
			}
		}
	}
	ff.close();
}

void RegistrationNoGPS::AlignmentXYZ()
{
	std::string file_xyz_align = fold_ + "\\xyz_alignment.txt";
	if (std::experimental::filesystem::exists(file_xyz_align)) {
		return;
	}

	std::string file_xy_align = fold_ + "\\xy_alignment.txt";
	std::string file_z_align = fold_ + "\\z_alignment.txt";
	std::string file_vertical = fold_ + "\\vertical_refined.txt";

	// read in vetical	
	std::ifstream ff(file_vertical);
	ff >> dir_vertial_.x >> dir_vertial_.y >> dir_vertial_.z;
	ff.close();
	Eigen::Vector3d vz(dir_vertial_.x, dir_vertial_.y, dir_vertial_.z);
	Eigen::Vector3d vx(1.0, 1.0, -(vz(0) + vz(1)) / vz(2));  vx /= vx.norm();
	Eigen::Vector3d vy = vz.cross(vx);  vy /= vy.norm();

	// read in xy alignment parameters
	ff = std::ifstream(file_xy_align);
	std::vector<int> idcam_xy;
	std::vector<cv::Vec4f> paras_xy;
	std::vector<Eigen::Matrix2d> Rs_xy;
	int ids, ide;
	double s, angle, tx, ty, tz;
	while (!ff.eof()) {
		ff >> ids >> ide >> s >> angle >> tx >> ty;
		for (int i = ids; i <= ide; i += 10) {
			idcam_xy.push_back(i);
			paras_xy.push_back(cv::Vec4f(s, angle, tx, ty));
			Eigen::Matrix2d Rtemp;
			Rtemp << cos(angle), -sin(angle), sin(angle), cos(angle);
			Rs_xy.push_back(Rtemp);
		}		
	}
	ff.close();

	// read in z alignment parameters
	ff = std::ifstream(file_z_align);
	std::vector<int> idcam_z;
	std::vector<double> paras_z;
	int id;
	float dz;
	while (!ff.eof()) {
		ff >> id >> dz;
		idcam_z.push_back(id);
		paras_z.push_back(dz);
	}
	ff.close();

	// read in cams
	std::string file_pose = file_street_pose_;
	std::ifstream iff(file_pose);
	std::string temp;
	std::getline(iff, temp);
	std::getline(iff, temp);
	std::string name;
	double x, y, z, rx, ry, rz;
	std::vector<Eigen::Vector3d> cams;
	std::vector<Eigen::Vector2d> cams_2d;
	std::vector<int> cams_imgid;
	while (!iff.eof()) {
		iff >> name >> x >> y >> z >> rx >> ry >> rz;
		Eigen::Vector3d t(x, y, z);
		cams.push_back(t);
		cams_2d.push_back(Eigen::Vector2d(t.dot(vx), t.dot(vy)));
		name = name.substr(0, name.size() - 4);
		cams_imgid.push_back(std::stoi(name));
	}
	iff.close();

	// step1: calculate the converted xyz of each cam

	// xy cvt
	double sigma_xy = 50.0;
	std::vector<Eigen::Vector2d> cams_2d_cvt(cams_2d.size());
	for (int i = 0; i < cams_2d.size(); i++) {
		Eigen::Vector2d pt = cams_2d[i];
		Eigen::Vector2d pt_cvt(0, 0);
		double w_total = 0.0;
		for (int m = 0; m < idcam_xy.size(); m++) {
			Eigen::Vector2d p2_m = paras_xy[m].val[0] * Rs_xy[m] * pt + Eigen::Vector2d(paras_xy[m].val[2], paras_xy[m].val[3]);
			double w_temp = exp(-abs(i - idcam_xy[m]) / sigma_xy);
			pt_cvt += w_temp * p2_m;
			w_total += w_temp;
		}
		pt_cvt /= w_total;
		cams_2d_cvt[i] = pt_cvt;
	}

	// z cvt
	double sigma_z = 100.0;
	std::vector<double> z_cvt(cams_2d.size());
	for (int i = 0; i < cams_2d.size(); i++)
	{
		double z_avg = 0.0;
		double z_weight = 0.0;
		for (int m = 0; m < idcam_z.size(); m++) {
			double w_temp = exp(-abs(i - idcam_z[m]) / sigma_z);
			z_avg += w_temp * paras_z[m];
			z_weight += w_temp;
		}
		z_cvt[i] = z_avg / z_weight;
	}


	// step2: calculate trans paras for each cam
	Eigen::Matrix3d Rc2xy; // rotation matrix from camera coordinate to plane (xy) coordinate
	Rc2xy << vx(0), vx(1), vx(2), vy(0), vy(1), vy(2), vz(0), vz(1), vz(2);

	int span = 50;	
	Eigen::Matrix3d R;
	std::vector<std::vector<double>> paras_xyz(cams.size());
	for (int i = 0; i < cams.size(); i++) {
		// xy 
		std::vector<Eigen::Vector2d> pts1, pts2;
		std::vector<double> weight;
		int id = i;
		int idcams = std::max(id - span, 0);
		int idcame = std::min(id + span, int(cams.size()));
		for (int j = idcams; j < idcame; j++) {			
			pts1.push_back(cams_2d[j]);
			pts2.push_back(cams_2d_cvt[j]);
			weight.push_back(1.0);
		}

		double s2d;
		Eigen::Matrix2d R2d;
		Eigen::Vector2d t2d;
		double residual = 0.0;
		SimilarityTransformation(pts1, pts2, weight, R2d, t2d, s2d, residual);
		Eigen::Matrix3d Rxy2s;
		Rxy2s << R2d(0, 0), R2d(0, 1), 0, R2d(1, 0), R2d(1, 1), 0, 0, 0, 1;
		R = Rxy2s * Rc2xy;
		Rotation2Euler(R, rx, ry, rz);
		s = s2d;
		tx = t2d(0);
		ty = t2d(1);

		// z		
		Eigen::Vector3d cam_cvt = s * R * cams[i];
		double tz = z_cvt[i] - cam_cvt(2) + 1.0; // 1.0 is the height of the camera

		paras_xyz[i].push_back(s);
		paras_xyz[i].push_back(rx);
		paras_xyz[i].push_back(ry);
		paras_xyz[i].push_back(rz);
		paras_xyz[i].push_back(tx);
		paras_xyz[i].push_back(ty);
		paras_xyz[i].push_back(tz);
	}

	if (0) {
		std::string file_cam = "F:\\result\\test.txt";
		std::ofstream fff(file_cam);
		for (size_t i = 0; i < cams.size(); i++)
		{
			double ss = paras_xyz[i][0];
			Eigen::Matrix3d RR;
			Euler2Rotation(paras_xyz[i][1], paras_xyz[i][2], paras_xyz[i][3], RR);
			Eigen::Vector3d ptt = ss * RR * cams[i] + Eigen::Vector3d(paras_xyz[i][4], paras_xyz[i][5], paras_xyz[i][6]);
			fff << ptt(0) << " " << ptt(1) << " " << ptt(2) << std::endl;
		}
		fff.close();
	}

	// write out
	std::ofstream off(file_xyz_align);
	off << std::setprecision(12) << std::fixed;
	for (int i = 0; i < paras_xyz.size(); i++) {
		off << i << " ";
		for (int j = 0; j < paras_xyz[i].size(); j++) {
			off << paras_xyz[i][j] << " ";
		}
		off << std::endl;
	}
	off.close();
}



void RegistrationNoGPS::PointOrthogonality(std::vector<Eigen::Vector2d>& pts, std::vector<double>& orthogonality)
{
	// calculate angles for each point
	std::vector<double> angles, carvture;
	PCAFunctions::PCA2d(pts, 10, angles, carvture);

	// calculate the orthogonality of each point
	int nbin = 180;
	float step_bin = 2 * CV_PI / nbin;
	std::vector<std::vector<int>> bins(nbin);
	std::vector<int> pt_bins(pts.size());
	for (int i = 0; i < angles.size(); i++) {
		int idbin = std::min(int(angles[i] / step_bin), nbin - 1);
		bins[idbin].push_back(i);
		pt_bins[i] = idbin;
	}

	orthogonality.resize(pts.size());
	double orthogonality_max = 0.0;
	for (int i = 0; i < pts.size(); i++) {
		int idbin = pt_bins[i];

		std::vector<int> ids;
		int id1 = (idbin - nbin / 4 + nbin) % nbin;
		for (int j = -2; j <= 2; j++) {
			int id = id1 + j;
			if (id >= 0 && id<nbin) {
				ids.push_back(id);
			}
		}
		int id2 = (idbin + nbin / 4 + nbin) % nbin;
		for (int j = -2; j <= 2; j++) {
			int id = id2 + j;
			if (id >= 0 && id<nbin) {
				ids.push_back(id);
			}
		}

		double dis_min = 10000000.0;
		for (int m = 0; m < ids.size(); m++) {
			for (int n = 0; n < bins[ids[m]].size(); n++) {
				int id_pt = bins[ids[m]][n];
				double dis = (pts[i] - pts[id_pt]).norm();
				if (dis < dis_min) {
					dis_min = dis;
				}
			}
		}
		orthogonality[i] = 1.0 / (1.0 + dis_min);
		if (orthogonality[i] > orthogonality_max) {
			orthogonality_max = orthogonality[i];
		}
	}

	for (size_t i = 0; i < orthogonality.size(); i++) {
		orthogonality[i] /= orthogonality_max;
	}

	if (0)
	{
		// show
		std::vector<cv::Point2d> pts_all;
		for (size_t i = 0; i < pts.size(); i++) {
			pts_all.push_back(cv::Point2d(pts[i](0), pts[i](1)));
		}

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

		double scale = 0.1;
		int rows = (ymax - ymin) / scale + 1;
		int cols = (xmax - xmin) / scale + 1;
		cv::Mat img = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
		for (size_t m = 0; m < pts_all.size(); m++)
		{
			int px = (pts_all[m].x - xmin) / scale;
			int py = (pts_all[m].y - ymin) / scale;
			if (abs(px - 688)<2 && abs(py - 1375)<2)
			{
				int a = 0;
			}
			//cv::circle(img, cv::Point(px, py), 1, cv::Scalar(0, 0, 255));
			int loc = 3 * (py*cols + px);
			uchar* ptr = img.data + loc;
			ptr[0] = int(255);
			ptr[1] = int(255);
			ptr[2] = int(255);

			//if (carvture[m]>0) {
			cv::circle(img, cv::Point(px, py), 3, cv::Scalar(0, 0, 1000 * orthogonality[m]), 2);
			//}
		}
		cv::imwrite("F:\\carvture.bmp", img);
	}
}

// R = Rz*Ry*Rx
void RegistrationNoGPS::Rotation2Euler(Eigen::Matrix3d R, double & rx, double & ry, double & rz) {
	//rx = std::atan2(-R(2, 1), R(2, 2));
	//ry = std::asin(R(2, 0));
	//rz = std::atan2(-R(1, 0), R(0, 0));

	rx = std::atan2(R(2, 1), R(2, 2));
	ry = std::asin(-R(2, 0));
	rz = std::atan2(R(1, 0), R(0, 0));

	if (rx != rx) { rx = 0.0; }
	if (ry != ry) { ry = 0.0; }
	if (rz != rz) { rz = 0.0; }
}

void RegistrationNoGPS::Euler2Rotation(double rx, double ry, double rz, Eigen::Matrix3d & R)
{
	Eigen::Matrix3d Rx, Ry, Rz;
	Rx << 1.0, 0, 0,
		0, std::cos(rx), -std::sin(rx),
		0, std::sin(rx), std::cos(rx);

	Ry << std::cos(ry), 0, std::sin(ry),
		0, 1.0, 0,
		-std::sin(ry), 0, std::cos(ry);

	Rz << std::cos(rz), -std::sin(rz), 0,
		std::sin(rz), std::cos(rz), 0,
		0, 0, 1.0;

	R = Rz * Ry * Rx;
}

void RegistrationNoGPS::UniqueVector(std::vector<int> &data)
{
	std::vector<int> data_new;
	std::sort(data.begin(), data.end());

	int v = data[0];
	data_new.push_back(v);
	for (std::vector<int>::iterator iter = data.begin(); iter != data.end(); ++iter) {
		if (*iter != v) {
			v = *iter;
			data_new.push_back(v);
		}
	}

	data = data_new;
}

void RegistrationNoGPS::PCANormal(std::vector<Eigen::Vector2d>& pts, Eigen::Vector2d & normal, double & curve)
{
	int npts = pts.size();

	Eigen::Vector2d pt_avg(0, 0);
	for (int j = 0; j < npts; ++j) {		
		pt_avg += pts[j];
	}
	pt_avg /= npts;

	cv::Matx22d cov;
	for (int j = 0; j < npts; ++j) {
		Eigen::Vector2d pt_dev = pts[j] - pt_avg;
		cov(0, 0) += pt_dev(0)*pt_dev(0);
		cov(0, 1) += pt_dev(0)*pt_dev(1);
		cov(1, 0) += pt_dev(1)*pt_dev(0);
		cov(1, 1) += pt_dev(1)*pt_dev(1);
	}
	cov *= 1.0 / npts;

	// eigenvector
	cv::Matx22d e_vecs;
	cv::Matx21d e_vals;
	cv::eigen(cov, e_vals, e_vecs);

	//pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0];
	curve = e_vals(1) / e_vals(0);
	normal = Eigen::Vector2d(e_vecs(1, 0), e_vecs(1, 1));
}


void RegistrationNoGPS::ShowResult(std::vector<Eigen::Vector2d>& pts, double scale, cv::Mat & img)
{
	//
	std::vector<cv::Point2d> pts_all;
	for (size_t i = 0; i < pts.size(); i++) {
		pts_all.push_back(cv::Point2d(pts[i](0), pts[i](1)));
	}

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
	for (size_t m = 0; m < pts_all.size(); m++)
	{
		int px = (pts_all[m].x - xmin) / scale;
		int py = (pts_all[m].y - ymin) / scale;
		cv::circle(img, cv::Point(px, py), 1, cv::Scalar(0, 0, 255));
		cv::putText(img, std::to_string(m), cv::Point(px, py), 1, 1, cv::Scalar(255, 0, 0));
		//int loc = 3 * (py*cols + px);
		//uchar* ptr = img.data + loc;
		//ptr[0] = 0;
		//ptr[1] = 0;
		//ptr[2] = 255;
	}
}

void RegistrationNoGPS::ShowResult(std::vector<Eigen::Vector2d>& pts, cv::Mat & img)
{
	for (size_t m = 0; m < pts.size(); m++)
	{
		int px = pts[m](0);
		int py = pts[m](1);
		cv::circle(img, cv::Point(px, py), 2, cv::Scalar(0, 0, 255));
	}
}

void RegistrationNoGPS::ShowResult(std::vector<Eigen::Vector2d>& src_ori, cv::Vec6f paras, std::vector<Eigen::Vector2d>& dst, cv::Mat & img)
{
	std::vector<Eigen::Vector2d> src(src_ori.size());
	double angle_j = paras.val[3];
	Eigen::Matrix2d R_j;
	R_j << cos(angle_j), -sin(angle_j), sin(angle_j), cos(angle_j);
	Eigen::Vector2d t(paras.val[4], paras.val[5]);
	for (size_t m = 0; m < src_ori.size(); m++) {
		src[m] = R_j * src_ori[m] + t;
	}

	double xmin = 10000.0, xmax = -xmin;
	double ymin = 10000.0, ymax = -ymin;
	for (size_t i = 0; i < src.size(); i++) {
		if (src[i](0) < xmin)xmin = src[i](0);
		if (src[i](0) > xmax)xmax = src[i](0);
		if (src[i](1) < ymin)ymin = src[i](1);
		if (src[i](1) > ymax)ymax = src[i](1);
	}

	for (size_t i = 0; i < dst.size(); i++) {
		if (dst[i](0) < xmin)xmin = dst[i](0);
		if (dst[i](0) > xmax)xmax = dst[i](0);
		if (dst[i](1) < ymin)ymin = dst[i](1);
		if (dst[i](1) > ymax)ymax = dst[i](1);
	}

	double w = xmax - xmin;
	double h = ymax - ymin;
	double tt = max(w, h);
	double s = 1000 / tt;
	int cols = w * s + 10;
	int rows = h * s + 10;
	img = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (size_t i = 0; i < src.size(); i++) {
		int x = (src[i](0) - xmin)*s;
		int y = (src[i](1) - ymin)*s;
		cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0, 0, 255));
	}

	for (size_t i = 0; i < dst.size(); i++) {
		int x = (dst[i](0) - xmin)*s;
		int y = (dst[i](1) - ymin)*s;
		cv::circle(img, cv::Point(x, y), 2, cv::Scalar(255, 0, 0));
	}
}

void RegistrationNoGPS::AlignViaXYOffset(std::vector<Eigen::Vector2d>& src, std::vector<Eigen::Vector2d>& dst, Eigen::Vector2d & offset, double score)
{
	// get the bbox of dst
	double xmin = 10000000.0, xmax = -xmin;
	double ymin = 10000000.0, ymax = -ymin;
	for (size_t i = 0; i < dst.size(); i++) {
		if (dst[i](0) < xmin) xmin = dst[i](0);
		if (dst[i](0) > xmax) xmax = dst[i](0);
		if (dst[i](1) < ymin) ymin = dst[i](1);
		if (dst[i](1) > ymax) ymax = dst[i](1);
	}

	// build grid
	int gridx = (xmax - xmin) / gsd_ + 1;
	int gridy = (ymax - ymin) / gsd_ + 1;
	std::vector<std::vector<int>> G(gridy);
	for (size_t i = 0; i < gridy; i++) {
		G[i].resize(gridx, 0);
	}

	for (size_t i = 0; i < dst.size(); i++) {
		int gx = (dst[i](0) - xmin) / gsd_;
		int gy = (dst[i](1) - ymin) / gsd_;
		G[gy][gx] = 1;
	}

	std::vector<Eigen::Vector2i> src_int(src.size());
	for (size_t i = 0; i < src.size(); i++) {
		src_int[i](0) = src[i](0) / gsd_;
		src_int[i](1) = src[i](1) / gsd_;
	}

	// searching
	int score_max = 0;
	int offx_max = 0, offy_max = 0;
	for (int i = 0; i < gridy; i++) {
		for (int j = 0; j < gridx; j++) {
			int score_temp = 0;
			for (int m = 0; m < src_int.size(); m++) {
				int x = src_int[m](0) + j;
				int y = src_int[m](1) + i;
				if (x<0 || x>= gridx || y<0 || y>=gridy) {
					continue;
				}
				if (G[y][x]>0) {
					score_temp++;
				}
			}
			if (score_temp > score_max) {
				score_max = score_temp;
				offx_max = j;
				offy_max = i;
			}
		}
	}

	offset = Eigen::Vector2d(xmin + offx_max * gsd_, ymin + offy_max * gsd_);
	score = score_max;
}

void RegistrationNoGPS::CostTransformations(std::vector<std::vector<Eigen::Vector2d>>& src_list, std::vector<std::vector<Eigen::Vector2d>>& dst_list, 
	std::vector<cv::Vec6f>& trans, std::vector<std::vector<float>>& costs)
{
	// build kdtree 
	PointCloud2d<double> cloud;
	for (size_t i = 0; i < dst_list.size(); i++) {
		for (size_t j = 0; j < dst_list[i].size(); j++) {
			cloud.pts.push_back(PointCloud2d<double>::PtData(dst_list[i][j](0), dst_list[i][j](1)));
		}

	}
	typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud2d<double> >, PointCloud2d<double>, 2/*dim*/ > my_kd_tree_t;
	my_kd_tree_t index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	// calculate the error for each trans
	int span = 1;
	float th_dis = 8.0;
	costs.resize(src_list.size());
	for (size_t k = 0; k < src_list.size(); k++) {
		costs[k].resize(trans.size());
		std::wcout << "cost " << k << std::endl;
#pragma omp parallel for
		for (int i = 0; i < trans.size(); i++) {
			int id_src = trans[i].val[0];
			double angle = trans[i].val[3];
			double tx = trans[i].val[4];
			double ty = trans[i].val[5];
			Eigen::Matrix2d R;
			R << cos(angle), -sin(angle), sin(angle), cos(angle);
			Eigen::Vector2d t(tx, ty);

			int id_s = k;
			if (id_s >= src_list.size() - span) {
				id_s = src_list.size() - span;
			}
			int id_e = id_s + span;

			std::vector<Eigen::Vector2d> pts;
			for (size_t j = id_s; j < id_e; j++) {
				for (size_t m = 0; m < src_list[j].size(); m++) {
					pts.push_back(R*src_list[j][m] + t);
				}
			}

			int count_inliers = 0;
			for (size_t j = 0; j < pts.size(); j++) {
				double *query_pt = new double[2];
				query_pt[0] = pts[j](0);
				query_pt[1] = pts[j](1);
				double dis_temp = 0.0;
				size_t idx_temp = 0;
				nanoflann::KNNResultSet<double> result_set(1);
				result_set.init(&idx_temp, &dis_temp);
				index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));
				double dx = cloud.pts[idx_temp].x - pts[j](0);
				double dy = cloud.pts[idx_temp].y - pts[j](1);
				if (abs(dx) + abs(dy) < th_dis) {
					count_inliers++;
				}
			}
			costs[k][i] = pts.size() - count_inliers;
		}
	}
}
