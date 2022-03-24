
#ifndef __HOUGH_LINE_H_
#define __HOUGH_LINE_H_

#include <vector>
#include <opencv2/opencv.hpp>

struct boundingbox_t
{
	int x;                              
	int y;                            
	int width;                        
	int height;                         
};

struct line_float_t
{
	float startx;
	float starty;
	float endx;
	float endy;
};

/*
@function    HoughLineDetector
@param       [in]      src:						  image,single channel
@param       [in]      w:                         width of image
@param       [in]      h:                         height of image
@param       [in]      scaleX:                    downscale factor in X-axis
@param       [in]      scaleY:                    downscale factor in Y-axis
@param       [in]      CannyLowThresh:            lower threshold for the hysteresis procedure in canny operator
@param       [in]      CannyHighThresh:           higher threshold for the hysteresis procedure in canny operator
@param       [in]      HoughRho:                  distance resolution of the accumulator in pixels
@param       [in]      HoughTheta:                angle resolution of the accumulator in radians
@param       [in]      MinThetaLinelength:        standard: for standard and multi-scale hough transform, minimum angle to check for lines.
												  propabilistic: minimum line length. Line segments shorter than that are rejected
@param       [in]      MaxThetaGap:               standard: for standard and multi-scale hough transform, maximum angle to check for lines
												  propabilistic: maximum allowed gap between points on the same line to link them
@param       [in]      HoughThresh:               accumulator threshold parameter. only those lines are returned that get enough votes ( >threshold ).
@param       [in]      _type:                     hough line method: HOUGH_LINE_STANDARD or HOUGH_LINE_PROBABILISTIC
@param       [in]      bbox:                      boundingbox to detect
@param       [in/out]  lines:                     result
@return：										  0:ok; 1:error
@brief：     _type: HOUGH_LINE_STANDARD:		  standard hough line algorithm
					HOUGH_LINE_PROBABILISTIC	  probabilistic hough line algorithm
					
When HOUGH_LINE_STANDARD runs, the line points might be the position outside the image coordinate

standard:		try (src,w,h,scalex,scaley,70,150, 1, PI/180, 0, PI, 100, HOUGH_LINE_STANDARD, bbox, line)
propabilistic:  try (src,w,h,scalex,scaley,70,150, 1, PI/180, 30, 10, 80, HOUGH_LINE_STANDARD, bbox, line)
*/
class HoughLiner
{
public:
	HoughLiner();
	~HoughLiner();

	static int HoughLineDetector( cv::Mat &edge_map, int _type, std::vector<line_float_t> &lines,
		double hough_rho, double hough_theta, double hough_thresh, double min_theta_linelength, double max_theta_gap);

	static int HoughLineDetector(unsigned char *src, int w, int h,
		float scaleX, float scaleY, float CannyLowThresh, float CannyHighThresh,
		float HoughRho, float HoughTheta, float MinThetaLinelength, float MaxThetaGap, int HoughThresh,
		int _type, boundingbox_t bbox, std::vector<line_float_t> &lines);
};



#endif /* HOUGH_H */
