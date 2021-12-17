/**************************************************************************************************************
* EDLines source codes.
* Copyright (C) Cuneyt Akinlar & Cihan Topal
* E-mails of the authors: cuneytakinlar@gmail.com, cihantopal@gmail.com
*
* Please cite the following papers if you use EDLines library:
*
* [1] C. Akinlar and C. Topal, “EDLines: A Real-time Line Segment Detector with a False Detection Control,”
*     Pattern Recognition Letters, 32(13), 1633-1642, DOI: 10.1016/j.patrec.2011.06.001 (2011).
*
* [2] C. Akinlar and C. Topal, “EDLines: Realtime Line Segment Detection by Edge Drawing (ED),”
*     IEEE Int’l Conf. on Image Processing (ICIP), Sep. 2011.
**************************************************************************************************************/

#ifndef _EDLines_
#define _EDLines_

#include "ED.h"
#include "EDColor.h"
#include "NFA.h"

#define SS 0
#define SE 1
#define ES 2
#define EE 3

// light weight struct for Start & End coordinates of the line segment
struct LS {
	cv::Point2d start;
	cv::Point2d end;

	LS(cv::Point2d _start, cv::Point2d _end)
	{
		start = _start;
		end = _end;
	}
};

struct LS2 {
	cv::Point2d start1;
	cv::Point2d start2;
	cv::Point2d end1;
	cv::Point2d end2;

	LS2(cv::Point2d _start1, cv::Point2d _end1, cv::Point2d _start2, cv::Point2d _end2)
	{
		start1 = _start1;
		start2 = _start2;
		end1 = _end1;
		end2 = _end2;
	}
};

struct RT {
	float r;
	float theta;

	RT(float _r, float _theta)
	{
		r = _r;
		theta = _theta;
	}
};

struct SERT {
	cv::Point2d start;
	cv::Point2d end;
	float r;
	float theta;

	SERT(cv::Point2d _start, cv::Point2d _end, float _r, float _theta)
	{
		start = _start;
		end = _end;
		r = _r;
		theta = _theta;
	}
};

struct SERTD {
	cv::Point2d start;
	cv::Point2d end;
	float r;
	float theta;
	float d;

	SERTD(cv::Point2d _start, cv::Point2d _end, float _r, float _theta, float _d)
	{
		start = _start;
		end = _end;
		r = _r;
		theta = _theta;
		d = _d;
	}
};

struct MatchSegment {
	cv::Point2d start1;
	cv::Point2d start2;
	cv::Point2d end1;
	cv::Point2d end2;
	float r1;
	float r2;
	float theta1;
	float theta2;
	float d1_start;
	float d1_end;
	float d2_start;
	float d2_end;
	float space_start;
	float space_end;
	float inv_space_start;
	float inv_space_end;
	float x_center_start;
	float x_center_end;

	MatchSegment(cv::Point2d _start1, cv::Point2d _end1, cv::Point2d _start2, cv::Point2d _end2, 
				float _r1, float _theta1, float _d1_start, float _d1_end, 
				float _r2, float _theta2, float _d2_start, float _d2_end, 
				float _space_start, float _space_end, 
				float _inv_space_start, float _inv_space_end,
				float _x_center_start, float _x_center_end)
	{
		start1 = _start1;
		start2 = _start2;
		end1 = _end1;
		end2 = _end2;
		r1 = _r1;
		r2 = _r2;
		theta1 = _theta1;
		theta2 = _theta2;
		d1_start = _d1_start;
		d1_end = _d1_end;
		d2_start = _d2_start;
		d2_end = _d2_end;
		space_start = _space_start;
		space_end = _space_end;
		inv_space_start = _inv_space_start;
		inv_space_end = _inv_space_end;
		x_center_start=_x_center_start;
		x_center_end=_x_center_end;
	}
};

struct LineSegment {
	double a, b;          // y = a + bx (if invert = 0) || x = a + by (if invert = 1)
	int invert;

	double sx, sy;        // starting x & y coordinates
	double ex, ey;        // ending x & y coordinates

	int segmentNo;        // Edge segment that this line belongs to
	int firstPixelIndex;  // Index of the first pixel within the segment of pixels
	int len;              // No of pixels making up the line segment

	LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len) {
		a = _a;
		b = _b;
		invert = _invert;
		sx = _sx;
		sy = _sy;
		ex = _ex;
		ey = _ey;
		segmentNo = _segmentNo;
		firstPixelIndex = _firstPixelIndex;
		len = _len;
	}
}; 

struct LineSegment2 {

	double sx, sy;        // starting x & y coordinates
	double ex, ey;        // ending x & y coordinates


	LineSegment2(double _sx, double _sy, double _ex, double _ey) {
		sx = _sx;
		sy = _sy;
		ex = _ex;
		ey = _ey;
	}
}; 


class EDLines : public ED {
public:
	EDLines(cv::Mat srcImage, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	EDLines(ED obj, double _line_error = 1.0, int _min_line_len =15, double _max_distance_between_two_lines = 10.0, double _max_error = 1.3);
	EDLines(EDColor obj, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	EDLines();

	std::vector<LS> getLines();
	int getLinesNo();
	cv::Mat getLineImage();
	cv::Mat drawOnImage();

	// EDCircle uses this one 
	static void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo, std::vector<LineSegment> &lines, int min_line_len = 6, double line_error = 1.0);

private:
	std::vector<LineSegment> lines;
	std::vector<LineSegment> invalidLines;
	std::vector<LS> linePoints;
	int linesNo;
	int min_line_len;
	double line_error;
	double max_distance_between_two_lines;
	double max_error;
	double prec;
	NFALUT *nfa;
	

	int ComputeMinLineLength();
	void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo);
	void JoinCollinearLines();
	
	void ValidateLineSegments();
	bool ValidateLineSegmentRect(int *x, int *y, LineSegment *ls);
	bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex);
	
	static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
	static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut);
	static void LineFit(double *x, double *y, int count, double &a, double &b, int invert);
	static void LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert);
	static double ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich);
	static void UpdateLineParameters(LineSegment *ls);
	static void EnumerateRectPoints(double sx, double sy, double ex, double ey,int ptsx[], int ptsy[], int *pNoPoints);

	// Utility math functions
	
};

#endif 
