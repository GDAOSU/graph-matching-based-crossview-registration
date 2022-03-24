/*
 * FILE: simtestlib.h, v.1.7.1, 4/28/2014
 * Author: Tomas V. Arredondo
 *
 * SimTSTLib: A simple yet flexible implementation of some test functions in C++.
 *
 * To change the function and range being optimized go tchange
 * FUNCTION_LOWER_RANGE, FUNCTION_UPPER_RANGE and FUNCTION definitions.
 *
 * DISCLAIMER: No liability is assumed by the author for any use made
 * of this program.
 * DISTRIBUTION: Any use may be made of this program, as long as the
 * clear acknowledgment is made to the author in code and runtime executables
 */
#ifndef SIMTSTLIB_H
#define SIMTSTLIB_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>

using namespace std;

namespace simtstlib
{
	// my function
	double alignment_fitness_fn(std::vector<double> x, int *img, int rows, int cols, float *pts, float dx, float dy, int num_pts);


	//
	/*******************************************************************/
	// The following random2 functions uses rand() which has a period of
	// RAND_MAX and could cause problems... user beware.
	/*******************************************************************/
	// Seed random2 generator (Note: random2 is not recommended see below)
#define randomize_rand()srand(2000)

	/* float random2(): uses rand() to generate a float random number between [0-1] (inclusive).*/
	/* Note that rand( ) based functions are not recommended.*/
	float random2();

	/* float random2(int num): generates an integer random number between [0-num) (num non inclusive)*/
	float random2(int num);

	/* float random2(float max, float min): uses rand() to generate a float random number */
	/* between [min,max] (both inclusive). Note that rand( ) based functions are not recommended.*/
	float random2(float max, float min);

	/******************************************************************************************/
	// ran2 function and definitions from NR. For comercial use you must get permision
	// from NR.
	// Note that in this code NR stands for Numerical Recipes which is copyrighted software
	// and is included here for academic completeness only.

	// If NR software is used by you in any application then must contact NR to obtain a license.

	// Numerical Recipes Software
	// P.O. Box 243,
	// Cambridge, MA 02238
	// http://www.nr.com/
	/******************************************************************************************/

	/*******************************************************************************************
	float ran2(long *idum)
	Numerical Recipes in C: The Art of Scientific Computing. William H. Press - Saul A. Teukolsky - William T. Vetterling - Brian P. Flanner
	Long period (> 2 × 10^18) random number generator of L’Ecuyer with Bays-Durham shuffle and added safeguards.
	Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint values).
	Call with idum a negative integer to initialize; thereafter, do not alter idum between successive deviates in a sequence.
	RNMX should approximate the largest floating value that is less than 1.
	*******************************************************************************************/
	float ran2(long *idum);

	/* float ran2(int num): generates an integer random number between [0-num) (num non inclusive)*/
	float ran2(int num);

	/* float ran2(float max, float min): uses rand() to generate a float random number */
	/* between [min,max] (both inclusive). */
	float ran2(float max, float min);

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

	/* flip flips a coin, with probability (prob) returns a 1 else a 0 */
	int flip(double prob);
}
#endif // SIMTSTLIB_TEST_FUNCTIONS_H
