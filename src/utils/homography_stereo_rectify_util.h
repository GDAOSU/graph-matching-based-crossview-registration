#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace cv;
using namespace std;

#define INF 2147483647.0

enum detector_id{
    ORB,
    BRISK
};


enum descriptor_id{
    BRUTE_FORCE,
    FLANN_BASE
};


/**
 * Returns a string showing the Mat type associated to the Mat type code.
 * Taken from http://stackoverflow.com/a/17820615/3370561
 * @param[in]  type Mat.type() integer.
 * @return      String like "CV_64F" or "CV_32UC3".
 */
string type2str(int type);


/**
 * Shows an image.
 * @param[in] img  Mat object that will be drawn in a new window.
 * @param[in] name Name of the window that will be created.
 */
void draw(Mat img, string name);


void obtainAB(int rows, int cols, const Mat &mult_mat, Mat &A, Mat &B);


/**
 * Returns an antisymmetric matrix representing the cross product with \p elem.
 * @param[in]  elem Element whose cross product matrix should be computed.
 * @return      A Mat object that represents the antisymmetric matrix
 * associated with the cross product of the vector \p elem.
 */
Mat crossProductMatrix(Vec3d elem);


/**
 * Maximizes the addend from equation 11 in the paper given the A,B or A',B'
 * matrices
 * @param[in]  A First parameter matrix.
 * @param[in]  B Second parameter matrix.
 * @return The Z vector that maximizes the addend.
 */
Vec3d maximize(Mat &A, Mat &B);


/**
 * Computes initial guess of z as explained in section 5.2 of the paper.
 * @param[in]  A  A matrix.
 * @param[in]  B  B matrix.
 * @param[in]  Ap A' matrix.
 * @param[in]  Bp B' matrix.
 * @return    Initial guess of z.
 */
Vec3d getInitialGuess(Mat &A, Mat &B, Mat &Ap, Mat &Bp);


/**
 * Manual fundamental Mat for paper example
 * @param  good_matches_1 Matches from first image.
 * @param  good_matches_2 Matches from second image.
 * @return                Fundamental Matrix
 */
Mat manualFundMat( vector<Point2d> &good_matches_1,
                    vector<Point2d> &good_matches_2);


/**
 * Computes the v'_c translation from equations 16 and 17 to be used in the
 * similarity transforms.
 * @param[in]  img_1 First image.
 * @param[in]  img_2 Second image.
 * @param[in]  H_p   First projective transform: H_p
 * @param[in]  Hp_p  Second projective transform: H'_p.
 * @return       Translation on Y axis.
 */
double getTranslationTerm(int rows, int cols, const Mat &H_p,
                          const Mat &Hp_p);


/**
 * Computes the minimum Y coordinate of the image \p after the \p homography is
 * applied.
 * @param[in]  img        Input image.
 * @param[in]  homography Homography.
 * @return Minimum Y coordinate of \p img after \p homography is applied.
 */
double getMinYCoord(int rows, int cols, const Mat &homography);


/**
 * Computes the auxiliary S matrix to later compute shearing transform.
 * @param[in]  img        Input image.
 * @param[in]  homography Rectifying homography for the image; i.e., H_r*H_p;
 * @return            S matrix.
 */
Mat getS(int rows, int cols, const Mat &homography);


/**
 * Computes both shearing transforms given the projective and similiarity
 * transforms of the two images.; i.e., it computes H_s and H'_s matrices given
 * H_r*H_p and H'_r*H'_p.
 * @param[in] img_1 First image.
 * @param[in] img_2 Second image.
 * @param[in] H_1   Rectifying homography for the 1st image; i.e., H_r*H_p;
 * @param[in] H_2   Rectifying homography for the 2nd image; i.e., H'_r*H'_p;
 * @param[out] H_s   Computed H_s shearing transform.
 * @param[out] Hp_s  Computed H'_s shearing transform.
 */
void getShearingTransforms(int rows, int cols,
                          const Mat &H_1, const Mat &H_2,
                          Mat &H_s, Mat &Hp_s);


/**
* Computes the Cholesky decomposition; i.e., given the symmetric
* definite-positive matrix \p A, an upper triangular matrix \p L is computed
* such that A = L^T.L.
* the .
* @param[in]  A Input matrix.
* @param[out]  L Computed upper triangular matrix.
* @return True if the decomposition exists; false if it does not exist.
*/
bool choleskyCustomDecomp(const Mat &A, Mat &L);


/**
 * Checks whether an image is inverted after an homography is applied.
 * @param[in]  img        Image to be tested. Mat object.
 * @param[in]  homography Homography to be applied. Mat object.
 * @return            A boolean value showing whether the image would be inverted
 *                      after the homography is applied.
 */
bool isImageInverted(int rows, int cols, const Mat &homography);

/**
 * Improves the estimated root of the polynomial described in Eq.11 of the
 * Loop&Zhang paper. All parameters follow their notation.
 * @param[in] A  A matrix.
 * @param[in] B  B matrix.
 * @param[in] Ap A' matrix.
 * @param[in] Bp B' matrix.
 * @param[out] z  Initial guess. The root given in section 5.2 of the paper
 * should be used.
 */
void optimizeRoot(const Mat &A, const Mat &B,
                  const Mat &Ap, const Mat &Bp,
                  Vec3d &z);

#endif
