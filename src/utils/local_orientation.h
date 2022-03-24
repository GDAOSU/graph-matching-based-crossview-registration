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

#ifndef OBJECTSFM_OBJ_ORIENTATION_H_
#define OBJECTSFM_OBJ_ORIENTATION_H_

#include <vector>
#include <opencv2/opencv.hpp>

namespace objectsfm {

// calculate the gradient orientation of a pixel 
class LocalOrientation
{
public:
	LocalOrientation();
	~LocalOrientation();

	static void CalOrientation(cv::Mat &img, cv::Point pt, float &dx, float &dy);
};


}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_ORIENTATION_H_
