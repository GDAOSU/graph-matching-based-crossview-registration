#include <stdio.h>
#include <fstream>
#include <filesystem>

#include "cv.h"
#include "highgui.h"
#include "registration_gps.h"
#include "registration_nogps.h"

using namespace cv;
using namespace std;

void test()
{
	std::ifstream iff("F:\\Database\\GoPro\\02_10\\GX010029\\1\\msp\\pose.txt");
	std::ofstream off("F:\\Database\\GoPro\\02_10\\GX010029\\1\\msp\\pose2.txt");
	char buffer[10000];
	iff.getline(buffer, 10000);
	off << buffer << std::endl;
	iff.getline(buffer, 10000);
	off << buffer << std::endl;
	int count = 0;
	while (!iff.eof())
	{
		iff.getline(buffer, 10000);
		count++;
		if (count % 2 == 0) {
			off << buffer << std::endl;
			count = 0;
		}
	}
	iff.close();
	off.close();
}
