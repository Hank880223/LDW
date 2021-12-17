/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: xwwang@openailab.com
 * Author: stevenwudi@fiture.com
 * 
 * original model: https://github.com/ultralytics/yolov5
 */

#include "yolov5sv6.h"
#include "LaneDetection.h"
#include "SortTracking.h"

using namespace cv;

int main(int argc, char* argv[])
{
    
    //Mat img=imread("0992.jpg",1);//载入3通道的彩色图像
    cv::VideoCapture cap("car.mp4");
    cv::Mat img;
    cv::Mat result_img;
	//cap.set(cv::CAP_PROP_FRAME_WIDTH , 640);
	//cap.set(cv::CAP_PROP_FRAME_HEIGHT , 480);
    
    cv::Mat yolov5sv6_result;
    //YOlOV5SV6 yolov5sv6("yolov5n.uint8.tmfile", "NPU");
    YOlOV5SV6 yolov5sv6("yolov5n.opt.tmfile", "CPU");
    LANEDETECTION lane;
    ED testED;
    lane.k_init();
    SORTTRACKING sorttracking;
    sorttracking.H(51,358,292,128,333,128,628,357, -188,0,-188,8000,188,8000,188,0);
    lane.y_bias = 150.f;
    lane.testED.H(48,358-lane.y_bias,300,120-lane.y_bias,330,120-lane.y_bias,628,357-lane.y_bias, -188,0,-188,8000,188,8000,188,0);
    
    while(1){
        if(KalmanTracker::kf_count == 100){
            KalmanTracker::kf_count = 0;
        }
        cap >> img;
        resize(img, img, Size(640, 360));        
        
        std::vector<LS> getlines = lane.getlines(img);
        cv:: Mat edgeImg = lane.testED.getEdgeImage();
        imshow("edgeImg", edgeImg);
        std::vector<SERT> Combinelines = lane.Combinelines(getlines);
        
        
        std::vector<MatchSegment> Trackinglines = lane.Trackinglines(Combinelines);
        std::vector<LS> Leftlines = lane.Leftlines(Trackinglines);
        std::vector<LS> Rightlines = lane.Rightlines(Trackinglines);
        if(Trackinglines.size()>0 && Leftlines.size()>0 && Rightlines.size()>0){
            lane.Update_PreviousRoadLines(Leftlines, Rightlines);
        }
        for(int i = 0; i < Leftlines.size(); i++){
			line(img, Point(Leftlines[i].start.x, Leftlines[i].start.y + lane.y_bias), Point(Leftlines[i].end.x, Leftlines[i].end.y + lane.y_bias), Scalar(0,0,255), 2, LINE_AA, 0);
		}
        for(int i = 0; i < Rightlines.size(); i++){
			line(img, Point(Rightlines[i].start.x, Rightlines[i].start.y + lane.y_bias), Point(Rightlines[i].end.x, Rightlines[i].end.y + lane.y_bias), Scalar(255,0,0), 2, LINE_AA, 0);
		}

        /*
        std::vector<LS> KalmanLeftlines = lane.KalmanLeftlines(Trackinglines);
        std::vector<LS> KalmanRightlines = lane.KalmanRightlines(Trackinglines);
        //std::vector<LS> Kalmanlines = lane.Kalmanlines(Trackinglines);
        for(int i = 0; i < KalmanLeftlines.size(); i++){
			line(img, Point(KalmanLeftlines[i].start.x, KalmanLeftlines[i].start.y + lane.y_bias), Point(KalmanLeftlines[i].end.x, KalmanLeftlines[i].end.y + lane.y_bias), Scalar(0,0,255), 2, LINE_AA, 0);
		}
        for(int i = 0; i < KalmanRightlines.size(); i++){
			line(img, Point(KalmanRightlines[i].start.x, KalmanRightlines[i].start.y + lane.y_bias), Point(KalmanRightlines[i].end.x, KalmanRightlines[i].end.y + lane.y_bias), Scalar(255,0,0), 2, LINE_AA, 0);
		}
        */
        //std::vector<cv::Point> LeftCurvePts = lane.LeftCurvePts(Trackinglines);
        //std::vector<cv::Point> RightCurvePts = lane.RightCurvePts(Trackinglines);

        /*
        cv::Mat LA;
		cv::Mat RA;
		LA = lane.polynomial_curve_fit_X(LeftCurvePts, 2);
		RA = lane.polynomial_curve_fit_X(RightCurvePts, 2);
        std::vector<cv::Point> LeftCurvePts_Fitted;
		std::vector<cv::Point> RightCurvePts_Fitted;
        for (int i = lane.LCurveStarty + lane.y_bias; i < lane.LCurveEndy + lane.y_bias; i++)
		{
			double j = LA.at<double>(0, 0) + LA.at<double>(1, 0) * i +
				LA.at<double>(2, 0)*std::pow(i, 2);
			if(j>0 && j<640) LeftCurvePts_Fitted.push_back(cv::Point(j, i));
		}
		cv::polylines(img, LeftCurvePts_Fitted, false, cv::Scalar(255, 0, 255), 2, 8, 0);

		for (int i = lane.RCurveStarty + lane.y_bias; i < lane.RCurveEndy + lane.y_bias; i++)
		{
			double j = RA.at<double>(0, 0) + RA.at<double>(1, 0) * i +
				RA.at<double>(2, 0)*std::pow(i, 2);
			if(j>0 && j<640) RightCurvePts_Fitted.push_back(cv::Point(j, i));
		}
		cv::polylines(img, RightCurvePts_Fitted, false, cv::Scalar(255, 0, 255), 2, 8, 0);
        */
        
        /*
        std::vector<Object> objects;
        objects = yolov5sv6.process(img);
        //yolov5sv6_result = yolov5sv6.draw_objects(img, objects);
        std::vector<TrackingBox> ImageTrackingResult;
        ImageTrackingResult = sorttracking.TrackingResult(objects);
        result_img = sorttracking.DarwTrackingResult(img, ImageTrackingResult);
        */
        imshow("detection", img);
        
        cv::waitKey(1);
        //yolov5sv6.clear_object();
        //ImageTrackingResult.clear();
        

        getlines.clear();
		Combinelines.clear();
		Trackinglines.clear();
		Leftlines.clear();
		Rightlines.clear();

    }
    yolov5sv6.~YOlOV5SV6();
    return 0;
}