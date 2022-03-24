#include <stdio.h>
#include <fstream>
#include <filesystem>

#include "cv.h"
#include "highgui.h"
#include "crossview_alignment.h"
#include "tinytiffreader.h"
#include "tinytiffwriter.h"

using namespace cv;
using namespace std;


void DSM2Pts()
{
	int cols = 1008;
	int rows = 655;
	std::string file_dsm = "F:\\DevelopCenter\\PointCloudRegistration\\data\\whole\\DSM.tif";
	std::string file_rgb = "F:\\DevelopCenter\\PointCloudRegistration\\data\\whole\\Ortho.bmp";
	std::string file_pts = "F:\\DevelopCenter\\PointCloudRegistration\\data\\whole\\pts.txt";
	cv::Mat img_rgb = cv::imread(file_rgb, 1);
	cv::Mat dsm(rows, cols, CV_32FC1, cv::Scalar(0.0));
	std::string file_sate_dsm = file_dsm;
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(file_sate_dsm.c_str());
	TinyTIFFReader_getSampleData(tiffr, (float*)dsm.data, 0);
	TinyTIFFReader_close(tiffr);

	//
	double dx_ = 0.5;
	double dy_ = -0.5;
	double xs_ = 324917.51;
	double ys_ = 4431700.69;
	double offx_ = 324917.51;
	double offy_ = 4431700.69;
	std::ofstream off(file_pts);
	off << std::fixed << std::setprecision(8);
	float* ptr = (float*)dsm.data;
	uchar* ptr_rgb = img_rgb.data;
	int count = 0;
	for (int i = 0; i < dsm.rows; i++) {
		for (int j = 0; j < dsm.cols; j++) {
			count++;
			if (*ptr >0) {
				off << j * dx_ + xs_ << " " << i * dy_ + ys_ << " " << *ptr << " " << int(ptr_rgb[0]) << " " << int(ptr_rgb[1]) << " " << int(ptr_rgb[2]) << std::endl;
			}
			ptr++;
			ptr_rgb += 3;
		}
	}
	off.close();
}


void main()
{
	objectsfm::CrossviewAlignment registor;

	// satellite
	std::string fold_sate = "F:\\DevelopCenter\\PointCloudRegistration\\data\\campus\\";
	std::string sate_dsm = fold_sate + "DSM.tif";
	std::string sate_ortho = fold_sate + "Ortho.bmp";
	std::string sate_mask_building = fold_sate + "mask_building.bmp";	
	std::string sate_mask_ndvi = fold_sate + "ndvi.bmp";
	registor.SetSateInfo(sate_dsm, sate_ortho, sate_mask_building, sate_mask_ndvi);
	
	// street	
	std::string fold_sfm = "D:\\gopro_result\\6_2019\\02_10\\GX010029\\";	
	//std::string fold_sfm = "D:\\gopro_result\\6_2019\\02_10\\GX010487\\";	
	registor.SetStreetInfo(fold_sfm);
	
	registor.RunAlign();
}


int test(int argc, char *argv[])
{
	objectsfm::CrossviewAlignment registor;

	// satellite
	std::string fold_sate = argv[1];
	std::string sate_dsm = fold_sate + "dsm_patch0.tif";
	std::string sate_ortho = fold_sate + "ortho_patch0.bmp";
	std::string sate_mask_building = fold_sate + "mask_building.bmp";
	std::string sate_mask_road = fold_sate + "mask_road.bmp";
	std::string sate_mask_ndvi = fold_sate + "ndvi.bmp";
	registor.SetSateInfo(sate_dsm, sate_ortho, sate_mask_building, sate_mask_ndvi);

	// street	
	std::string fold_sfm = argv[2];
	registor.SetStreetInfo(fold_sfm);

	std::cout << fold_sate << std::endl;
	std::cout << fold_sfm << std::endl;

	registor.RunAlign();

	return 1;
}