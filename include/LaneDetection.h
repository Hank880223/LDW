#pragma once
#ifndef LANEDETECTION_H_
#define LANEDETECTION_H_
#include "EDLib.h"
#include <iostream>
#include <sys/time.h>
#include "RTKalman.h"
#include <tuple>

class LANEDETECTION {

    
    
public:
    
    ED testED;
    float y_bias = 150.f;
    int LCurveStartx = 9999;
    int LCurveEndx = 0;
    int LCurveStarty = 9999;
    int LCurveEndy = 0;
    int RCurveStartx = 9999;
    int RCurveEndx = 0;
    int RCurveStarty = 9999;
    int RCurveEndy = 0;

    std::vector<LS> getlines(const cv::Mat& bgr);
    std::vector<SERT> Combinelines(std::vector<LS> lines);
    std::vector<MatchSegment> Trackinglines(std::vector<SERT> AllLines);
    std::vector<LS> Kalmanlines(std::vector<MatchSegment> TrackingRoadLines);
    std::vector<LS> KalmanLeftlines(std::vector<MatchSegment> TrackingRoadLines);
    std::vector<LS> KalmanRightlines(std::vector<MatchSegment> TrackingRoadLines);
    std::vector<LS> Leftlines(std::vector<MatchSegment> TrackingRoadLines);
    std::vector<LS> Rightlines(std::vector<MatchSegment> TrackingRoadLines);
    std::vector<cv::Point> LeftCurvePts(std::vector<MatchSegment> TrackingRoadLines);
    std::vector<cv::Point> RightCurvePts(std::vector<MatchSegment> TrackingRoadLines);
    void Update_PreviousRoadLines(std::vector<LS> Leftlines, std::vector<LS> Rightlines);
    cv::Mat polynomial_curve_fit_X(std::vector<cv::Point>& key_point, int n);
    void k_init();

private:

    vector<RTK> Lline_track;
	vector<RTK> Rline_track;
	//RTKalman::Lkf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
	//RTKalman::Rkf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
	RTKalman Ltrk = RTKalman(RTK(196.6f, 42.25f));
	RTKalman Rtrk = RTKalman(RTK(234.5f, -51.05f));
	Mat Lpredict;
	Mat Rpredict;
	Mat LUpdatedvalue;
	Mat RUpdatedvalue;


	std::vector<MatchSegment> PreviousRoadLines;
	int tracking_success_count = 0;
	int tracking_success_flag = 0;
	int tracking_fail_count = 0;
	int tracking_fail_flag = 0;

	KalmanStructTypedef left_line_pts_x_start;
    KalmanStructTypedef left_line_pts_x_end;
    KalmanStructTypedef left_line_pts_y_start;
    KalmanStructTypedef left_line_pts_y_end;
	KalmanStructTypedef right_line_pts_x_start;
    KalmanStructTypedef right_line_pts_x_end;
    KalmanStructTypedef right_line_pts_y_start;
    KalmanStructTypedef right_line_pts_y_end;
    
    cv::Mat HM = cv::Mat::ones(3, 3, CV_64FC1);
    
    float flm = 0.f;
    float frm = 0.f;
    float flb = 0.f;
    float frb = 0.f;

    float x_value(float y, float m, float b);
    float m_value(float start_x, float start_y, float end_x, float end_y);
    float b_value(float start_x, float start_y, float end_x, float end_y);
    
    void kalman_init(KalmanStructTypedef *kalmanFilter, float init_x, float init_p);
	float kalman_filter(KalmanStructTypedef *kalmanFilter, float newMeasured);
    //bool compareY(const Point2f& p1, const Point2f& p2);
    
};
#endif // !MOBILEFACENET_H_