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

#include "local_orientation.h"

namespace objectsfm {

LocalOrientation::LocalOrientation()
{
}

LocalOrientation::~LocalOrientation()
{
}

void LocalOrientation::CalOrientation(cv::Mat & img, cv::Point pt, float &dx, float &dy)
{
	// step1: corp the image around the input pt
	int win_half = 20;
	int xmin = MAX(pt.x - win_half, 0);
	int ymin = MAX(pt.y - win_half, 0);
	int xmax = MIN(pt.x + win_half, img.cols - 1);
	int ymax = MIN(pt.y + win_half, img.rows - 1);

	int cols = xmax - xmin;
	int rows = ymax - ymin;
	if (cols < win_half + 1 || rows < win_half + 1) return;
	cv::Rect rect(xmin, ymin, cols, rows);
	cv::Mat img_cropped_ = img(rect);

	// step2: region grow to collect neighboring pixels
	int x_offset[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int y_offset[8] = { 1, 0, -1, -1, -1, 0, 1, 1 };
	int loc_offset[8];
	for (size_t i = 0; i < 8; i++)
	{
		loc_offset[i] = y_offset[i] * cols + x_offset[i];
	}

	cv::Mat mask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));	
	int y_seed = pt.y - ymin;
	int x_seed = pt.x - xmin;
	std::vector<cv::Point> cluster;
	cluster.push_back(cv::Point(x_seed, y_seed));
	mask.at<uchar>(y_seed, x_seed) = 1;

	int count = 0;
	while (count < cluster.size())
	{
		int x0 = cluster[count].x;
		int y0 = cluster[count].y;
		if (x0 < 1 || x0>cols - 2 || y0<1 || y0>rows - 2)
		{
			count++;
			continue;
		}

		for (size_t j = 0; j < 8; j++)
		{
			int x = x0 + x_offset[j];
			int y = y0 + y_offset[j];
			if (!mask.at<uchar>(y, x) && img_cropped_.at<uchar>(y,x))
			{
				cluster.push_back(cv::Point(x, y));
				mask.at<uchar>(y, x) = 1;
			}
		}
		if (cluster.size() > 30)
		{
			break;
		}
		count++;
	}

	// step3: orientation fitting via svd
	if (cluster.size() < 2) return;

	cv::Matx21d h_mean(0, 0);
	for (int i = 0; i < cluster.size(); ++i)
	{
		h_mean += cv::Matx21d(cluster[i].x, cluster[i].y);
	}
	h_mean *= (1.0 / cluster.size());

	cv::Matx22d h_cov(0, 0, 0, 0);
	for (int i = 0; i < cluster.size(); ++i)
	{
		cv::Matx21d hi = cv::Matx21d(cluster[i].x, cluster[i].y);
		h_cov += (hi - h_mean) * (hi - h_mean).t();
	}
	h_cov *= (1.0 / cluster.size());

	// eigenvector
	cv::Matx22d h_cov_evectors;
	cv::Matx21d h_cov_evals;
	cv::eigen(h_cov, h_cov_evals, h_cov_evectors);
	cv::Matx21d normal = h_cov_evectors.row(1).t();

	dx =  normal(1);
	dy = -normal(0);
}


}  // namespace objectsfm
