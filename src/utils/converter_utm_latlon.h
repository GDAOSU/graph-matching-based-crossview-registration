//LatLong- UTM conversion..h
//definitions for lat/long to UTM and UTM to lat/lng conversions

#ifndef CONVERTER_UTM_LATLON_H_
#define CONVERTER_UTM_LATLON_H_

#include <string.h>

void LLtoUTM(int ReferenceEllipsoid, const double Lat, const double Long, 
	double &UTMNorthing, double &UTMEasting, char* UTMZone);

void UTMtoLL(int ReferenceEllipsoid, const double UTMNorthing, const double UTMEasting, const char* UTMZone,
	double& Lat,  double& Long );

char UTMLetterDesignator(double Lat);

#endif // CONVERTER_UTM_LATLON_H_

