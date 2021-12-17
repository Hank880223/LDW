#include "LaneDetection.h"
using namespace cv;
using namespace std;

float LANEDETECTION::x_value(float y, float m, float b){
	return ((y - b) / m);
}

float LANEDETECTION::m_value(float start_x, float start_y, float end_x, float end_y){
	return ((start_y - end_y)/(start_x -  end_x));
}

float LANEDETECTION::b_value(float start_x, float start_y, float end_x, float end_y){
	return (((end_x * start_y) - (start_x * end_y)) / (end_x - start_x));
}



cv::Mat LANEDETECTION::polynomial_curve_fit_X(std::vector<cv::Point>& key_point, int n)
{
    cv::Mat A;
    
    //Number of key points
    int N = key_point.size();

    //creat matrix X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                    std::pow(key_point[k].y, i + j);
            }
        }
    }

    //creat matrix Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                std::pow(key_point[k].y, i) * key_point[k].x;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //solve matrix A with LU
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return A;
}

bool compareY(const Point2f& p1, const Point2f& p2){
	if(p1.y < p2.y) return true;
	if(p1.y > p2.y) return false;
}

std::vector<LS> LANEDETECTION::getlines(const cv::Mat& bgr){
    cv::Mat bgrimg = bgr.clone();
    cv::Mat grayimg;
    cv::cvtColor(bgrimg, grayimg, COLOR_BGR2GRAY);
    cv::Mat ROIImg = grayimg(Rect(0,y_bias,639, (359 - y_bias)));
    testED = ED(ROIImg, SOBEL_OPERATOR, 36, 8, 5, 10, 1, true); // apply ED algorithm
    EDLines testEDLines = EDLines(testED);

    std::vector<LS> lines = testEDLines.getLines();
    int noLines = testEDLines.getLinesNo();

	std::cout << "Number of line segments: " << noLines << std::endl;
    
    return lines;
}

std::vector<SERT> LANEDETECTION::Combinelines(std::vector<LS> lines){
    
    float l1m = m_value(250.f, 0.f, 0.f, 70.f);
    float l1b = b_value(250.f, 0.f, 0.f, 70.f);
    float l2m = m_value(250.f, 0.f, 15.f, (359 - y_bias));
    float l2b = b_value(250.f, 0.f, 15.f, (359 - y_bias));

    float r1m = m_value(390.f, 0.f, 639.f, 70.f);
    float r1b = b_value(390.f, 0.f, 639.f, 70.f);
    float r2m = m_value(390.f, 0.f, 624.f, (359 - y_bias));
    float r2b = b_value(390.f, 0.f, 624.f, (359 - y_bias));

    std::vector<cv::Point> LeftCombineLines;
    std::vector<cv::Point> RightCombineLines;
    std::vector<RT> LeftRT;
    std::vector<RT> RightRT;
    std::vector<RT> ALLRT;
    std::vector<SERT> AllLines;

    int LeftCombineLinesNos;
    int RightCombineLinesNos;
    int AllLinesNos;
    for (int i = 0; i < lines.size(); i++){
        float x = lines[i].start.x - lines[i].end.x;
        float y = lines[i].start.y - lines[i].end.y;
        float r = sqrt(x*x+y*y);
        
        /*
        if(r < 20 && lines[i].end.y >100){
            cout << "r " << r <<endl;
            lines.erase(lines.begin()+i);
            i--;
        }
        */
    }
    Mat thresholdlineImage1 = Mat(360, 640, CV_8UC3, Scalar(255,255,255));
    line(thresholdlineImage1, Point(250, y_bias), Point(0, 240), Scalar(0,255,0), 1, LINE_AA, 0);
    line(thresholdlineImage1, Point(390, y_bias), Point(639, 240), Scalar(0,255,0), 1, LINE_AA, 0);
    line(thresholdlineImage1, Point(250, y_bias), Point(15, 359), Scalar(255,0,0), 1, LINE_AA, 0);
    line(thresholdlineImage1, Point(390, y_bias), Point(624, 359), Scalar(255,0,0), 1, LINE_AA, 0);
    line(thresholdlineImage1, Point(250, y_bias), Point(70, 359), Scalar(0,0,255), 1, LINE_AA, 0);
    line(thresholdlineImage1, Point(390, y_bias), Point(569, 359), Scalar(0,0,255), 1, LINE_AA, 0);
    for (int i = 0; i < lines.size(); i++) {
        
            line(thresholdlineImage1, Point(lines[i].start.x, lines[i].start.y + y_bias), Point(lines[i].end.x, lines[i].end.y + y_bias), Scalar(0,0,0), 1, LINE_AA, 0);
    }
    imshow("thresholdlineImage1",thresholdlineImage1);


    for (int i = 0; i < lines.size(); i++) {
        float m0 = (lines[i].start.y - lines[i].end.y)/(lines[i].start.x -  lines[i].end.x);
        //float b0 = ((lines[i].end.x * lines[i].start.y) - (lines[i].start.x * lines[i].end.y)) / (lines[i].end.x - lines[i].start.x);
        //float y0 = b0;
        //float x0 = (-b0) / m0;
        float y0 = lines[i].start.y;
        float x0 = lines[i].start.x;
        float deg0 = atan (-1.f/m0) * 180.f / CV_PI;
        float theta0 = deg0;
        float r0 = x0 * cos(theta0 * CV_PI / 180.f ) + y0 * sin(theta0 * CV_PI / 180.f );
        
        for (int j = i+1; j < lines.size(); j++) {
            float m1 = (lines[j].start.y - lines[j].end.y)/(lines[j].start.x -  lines[j].end.x);
            //float b1 = ((lines[j].end.x * lines[j].start.y) - (lines[j].start.x * lines[j].end.y)) / (lines[j].end.x - lines[j].start.x);
            //float y1 = b1;
            //float x1 = (-b1) / m1;
            float y1 = lines[j].start.y;
            float x1 = lines[j].start.x;
            float deg1 = atan (-1.f/m1) * 180.f / CV_PI;
            float theta1 = deg1;
            float r1 = x1 * cos(theta1 * CV_PI / 180.f) + y1 * sin(theta1 * CV_PI / 180.f);
            
            if(abs(theta0 - theta1) < 5.f && abs(r0 - r1) < 15.f && (lines[j].start.x < 640.f/2.f || lines[j].end.x < 640.f/2.f) && theta1 > -20.f && theta1 < 90.f ){

                float l1x = x_value(y1, l1m, l1b);
                float l2x = x_value(y1, l2m, l2b);
                if(x1 < l1x) continue;
                if(x1 < l2x) continue;
                if(x1 < l2x || theta0 <20.f) continue;
                LeftRT.push_back(RT(r1, theta1));
                lines.erase(lines.begin()+j);
                j--;
            }

            if(abs(theta0 - theta1) < 5.f && abs(r0 - r1) < 15.f && (lines[j].start.x > 640.f/2.f || lines[j].end.x > 640.f/2.f)  && theta1 < 20.f && theta1 > -90.f){
                
                float r1x = x_value(y1, r1m, r1b);
                float r2x = x_value(y1, r2m, r2b);
                if(x1 > r1x) continue;
                if(x1 > r2x) continue;
                if(x1 > r2x || theta0 > -20.f) continue;
                RightRT.push_back(RT(r1, theta1));
                lines.erase(lines.begin()+j);
                j--;
                
            }
            
        }

        LeftCombineLinesNos = (int)LeftRT.size(); // get the total number of anchor points
        RightCombineLinesNos = (int)RightRT.size(); // get the total number of anchor points

        if(LeftCombineLinesNos > 0) 
        {
            LeftRT.push_back(RT(r0, theta0));
            float left_r_avg = 0.f;
            float left_theta_avg = 0.f;
            for(int k = 0; k < LeftRT.size(); k++){
                left_r_avg += LeftRT[k].r;
                left_theta_avg += LeftRT[k].theta;
            }
            
            left_r_avg /= LeftRT.size();
            left_theta_avg /= LeftRT.size();
            float left_x_pt = 0.f;
            float left_y_pt = 0.f;

            if(left_theta_avg < 90.f && left_theta_avg > 0.f){	
                left_x_pt = left_r_avg / cos(left_theta_avg * CV_PI / 180.f);
                left_y_pt = left_r_avg / sin(left_theta_avg * CV_PI / 180.f);
                float left_m = -left_y_pt / left_x_pt;
                float left_b = -(left_x_pt * left_y_pt) / (-left_x_pt);
                float left_y = left_b;
                float left_x = ((359 - y_bias) - left_b) / left_m;
                AllLines.push_back(SERT(Point(left_x_pt, 0), Point(left_x, (359 - y_bias)), left_r_avg, left_theta_avg));
            }
            else{
                left_x_pt = left_r_avg / cos((left_theta_avg) * CV_PI / 180.f);
                left_y_pt = left_r_avg / sin((left_theta_avg) * CV_PI / 180.f);
                float left_y_shift = y_bias +abs(left_y_pt);
                float left_m = -(left_y_pt) / left_x_pt;
                float left_b = ((left_x_pt*2 * (left_y_pt + left_y_shift - y_bias)) - (left_x_pt * (left_y_shift - y_bias))) / (left_x_pt*2 - left_x_pt);
                float left_y = left_b;
                float left_x = ((359 - y_bias) - left_b) / left_m;
                AllLines.push_back(SERT(Point(left_x_pt, 0), Point(left_x, (359 - y_bias)), left_r_avg, left_theta_avg));
            }
            
        }
        else if(lines[i].start.x < 640.f/2.f || lines[i].end.x < 640.f/2.f)
        {
            float l1x = x_value(lines[i].start.y, l1m, l1b);
            float l2x = x_value(lines[i].start.y, l2m, l2b);
            if(lines[i].start.x < l1x) continue;
            if(lines[i].start.x < l2x && theta0 <20.f) continue;
            LeftRT.push_back(RT(r0, theta0));
            float left_x_pt = 0.f;
            float left_y_pt = 0.f;
            if(LeftRT[0].theta < 90.f && LeftRT[0].theta > 0.f){	
                left_x_pt = LeftRT[0].r / cos(LeftRT[0].theta * CV_PI / 180.f);
                left_y_pt = LeftRT[0].r / sin(LeftRT[0].theta * CV_PI / 180.f);
                float left_m = -left_y_pt / left_x_pt;
                float left_b = -(left_x_pt * left_y_pt) / (-left_x_pt);
                float left_y = left_b;
                float left_x = ((359 - y_bias) - left_b) / left_m;
                AllLines.push_back(SERT(Point(left_x_pt, 0), Point(left_x, (359 - y_bias)), LeftRT[0].r, LeftRT[0].theta));
            }
            else{
                left_x_pt = LeftRT[0].r / cos((LeftRT[0].theta) * CV_PI / 180.f);
                left_y_pt = LeftRT[0].r / sin((LeftRT[0].theta) * CV_PI / 180.f);
                float left_y_shift = y_bias +abs(left_y_pt);
                float left_m = -(left_y_pt) / left_x_pt;
                float left_b = ((left_x_pt*2 * (left_y_pt + left_y_shift - y_bias)) - (left_x_pt * (left_y_shift - y_bias))) / (left_x_pt*2 - left_x_pt);
                float left_y = left_b;
                float left_x = ((359 - y_bias) - left_b) / left_m;
                AllLines.push_back(SERT(Point(left_x_pt, 0), Point(left_x, (359 - y_bias)), LeftRT[0].r, LeftRT[0].theta));

            }

        }

        if(RightCombineLinesNos > 0) 
        {
            RightRT.push_back(RT(r0, theta0));
            float right_r_avg = 0.f;
            float right_theta_avg = 0.f;
            for(int k = 0; k < RightRT.size(); k++){
                right_r_avg += RightRT[k].r;
                right_theta_avg += RightRT[k].theta;
            }
            
            right_r_avg /= RightRT.size();
            right_theta_avg /= RightRT.size();
            float right_x_pt = 0.f;
            float right_y_pt = 0.f;

            if(right_theta_avg > -90.f && right_theta_avg < 0.f){
                right_x_pt = right_r_avg / cos((right_theta_avg) * CV_PI / 180.f);
                right_y_pt = right_r_avg / sin((right_theta_avg) * CV_PI / 180.f);
                float right_y_shift = y_bias+abs(right_y_pt);
                float right_m = -(right_y_pt) / right_x_pt;
                float right_b = ((right_x_pt*2 * (right_y_pt + right_y_shift - y_bias)) - (right_x_pt * (right_y_shift - y_bias))) / (right_x_pt*2 - right_x_pt);
                float right_y = right_b;
                float right_x = ((359 - y_bias) - right_b) / right_m;
                AllLines.push_back(SERT(Point(right_x_pt, 0), Point(right_x, (359 - y_bias)), right_r_avg, right_theta_avg));
            }
            else{
                right_x_pt = right_r_avg / cos((right_theta_avg) * CV_PI / 180.f);
                right_y_pt = right_r_avg / sin((right_theta_avg) * CV_PI / 180.f);
                float right_m = -right_y_pt / right_x_pt;
                float right_b = -(right_x_pt * right_y_pt) / (-right_x_pt);
                float right_y = right_b;
                float right_x = ((359 - y_bias) - right_b) / right_m;
                AllLines.push_back(SERT(Point(right_x_pt, 0), Point(right_x, (359 - y_bias)), right_r_avg, right_theta_avg));
            }
            
        }
        else if(lines[i].start.x > 640.f/2.f && lines[i].end.x > 640.f/2.f)
        {
            float r1x = x_value(lines[i].start.y, r1m, r1b);
            float r2x = x_value(lines[i].start.y, r2m, r2b);
            //std::cout << "theta0 " << theta0 << std::endl;
            if(lines[i].start.x > r1x) continue;
            if(lines[i].start.x > r2x && theta0 > -20.f) continue;
            RightRT.push_back(RT(r0, theta0));
            float right_x_pt = 0.f;
            float right_y_pt = 0.f;
            
            if(RightRT[0].theta > -90.f && RightRT[0].theta < 0.f){
                right_x_pt = RightRT[0].r / cos((RightRT[0].theta) * CV_PI / 180.f);
                right_y_pt = RightRT[0].r / sin((RightRT[0].theta) * CV_PI / 180.f);
                float right_y_shift = y_bias+abs(right_y_pt);
                float right_m = -(right_y_pt) / right_x_pt;
                float right_b = ((right_x_pt*2 * (right_y_pt + right_y_shift - y_bias)) - (right_x_pt * (right_y_shift - y_bias))) / (right_x_pt*2 - right_x_pt);
                float right_y = right_b;
                float right_x = ((359 - y_bias) - right_b) / right_m;
                AllLines.push_back(SERT(Point(right_x_pt, 0), Point(right_x, (359 - y_bias)), RightRT[0].r, RightRT[0].theta));
            }
            else if(RightRT[0].theta < 20.f && RightRT[0].theta > 0.f)
            {
                right_x_pt = RightRT[0].r / cos((RightRT[0].theta) * CV_PI / 180.f);
                right_y_pt = RightRT[0].r / sin((RightRT[0].theta) * CV_PI / 180.f);
                float right_m = -right_y_pt / right_x_pt;
                float right_b = -(right_x_pt * right_y_pt) / (-right_x_pt);
                float right_y = right_b;
                float right_x = ((359 - y_bias) - right_b) / right_m;
                AllLines.push_back(SERT(Point(right_x_pt, 0), Point(right_x, (359 - y_bias)), RightRT[0].r, RightRT[0].theta));
            }
        }
        LeftRT.clear();
        RightRT.clear();
    }



    return AllLines;
}

std::vector<MatchSegment> LANEDETECTION::Trackinglines(std::vector<SERT> AllLines){
    std::vector<MatchSegment> RoadLines;
	std::vector<MatchSegment> TrackingRoadLines;

    /*
    HM.at<double>(0,0) = 3.739862280030613;
    HM.at<double>(0,1) = -0.08851745041492516;
    HM.at<double>(0,2) = -1154.035195103293;
    HM.at<double>(1,0) = 1.942214879979242e-15;
    HM.at<double>(1,1) = 34.31228297333891;
    HM.at<double>(1,2) = -1338.179035960218;
    HM.at<double>(2,0) = 2.115629703434065e-18;
    HM.at<double>(2,1) = 0.02307103760814556;
    HM.at<double>(2,2) = 1;
    */
    /*
    HM.at<double>(0,0) = 3.739862280030613;
    HM.at<double>(0,1) = -0.08851745041492516;
    HM.at<double>(0,2) = -1154.035195103293;
    HM.at<double>(1,0) = 1.942214879979242e-15;
    HM.at<double>(1,1) = 34.31228297333891;
    HM.at<double>(1,2) = -1338.179035960218;
    HM.at<double>(2,0) = 2.115629703434065e-18;
    HM.at<double>(2,1) = 0.02307103760814556;
    HM.at<double>(2,2) = 1;
    */
    
    for (int i = 0; i < AllLines.size(); i++) {
        double line0_start = testED.line_width((AllLines[i].start.x), AllLines[i].start.y, 1.f);
        double line0_end = testED.line_width((AllLines[i].end.x), AllLines[i].end.y, 1.f);
        for (int j = i+1; j < AllLines.size(); j++) {
            double line1_start = testED.line_width((AllLines[j].start.x), AllLines[j].start.y, 1.f);
            double line1_end = testED.line_width((AllLines[j].end.x), AllLines[j].end.y, 1.f);
            float space_start = abs(line0_start - line1_start);
            float space_end = abs(line0_end - line1_end);
            float inv_space_start = abs(abs(line0_start) - abs(line1_start));
            float inv_space_end = abs(abs(line0_end) - abs(line1_end));
            float x_center_start = (AllLines[j].start.x + AllLines[i].start.x) / 2.f;
            float x_center_end = (AllLines[j].end.x + AllLines[i].end.x) / 2.f;

            if((space_end > 300.f && space_end < 600.f) && (space_start > 300.f && space_start < 600.f)){
                if(line0_end>line1_end && AllLines[j].start.x < AllLines[i].start.x && AllLines[j].end.x < AllLines[i].end.x){
                    RoadLines.push_back(MatchSegment(AllLines[j].start, AllLines[j].end, AllLines[i].start, AllLines[i].end, 
                                                    AllLines[j].r, AllLines[j].theta, line1_start, line1_end,
                                                    AllLines[i].r, AllLines[i].theta, line0_start, line0_end,
                                                    space_start, space_end, 
                                                    inv_space_start, inv_space_end,
                                                    x_center_start, x_center_end));
                }
                else if(line0_end<line1_end && AllLines[j].start.x > AllLines[i].start.x && AllLines[j].end.x > AllLines[i].end.x){
                    RoadLines.push_back(MatchSegment(AllLines[i].start, AllLines[i].end, AllLines[j].start, AllLines[j].end, 
                                                    AllLines[i].r, AllLines[i].theta, line0_start, line0_end,
                                                    AllLines[j].r, AllLines[j].theta, line1_start, line1_end,
                                                    space_start, space_end, 
                                                    inv_space_start, inv_space_end,
                                                    x_center_start, x_center_end));
                }
            }
        }
    }

    Mat showimg2 = Mat(360, 640, CV_8UC3, Scalar(255,255,255));
    for(int i = 0; i < RoadLines.size(); i++){
        line(showimg2, Point(RoadLines[i].start1.x, RoadLines[i].start1.y + y_bias), Point(RoadLines[i].end1.x, RoadLines[i].end1.y + y_bias), Scalar(50,128,255), 2, LINE_AA, 0);
        line(showimg2, Point(RoadLines[i].start2.x, RoadLines[i].start2.y + y_bias), Point(RoadLines[i].end2.x, RoadLines[i].end2.y + y_bias), Scalar(50,128,255), 2, LINE_AA, 0);
    }
    cv::imshow("showimg2",showimg2);

    if(tracking_success_flag == 0){
        if(PreviousRoadLines.size()==0){
            float min_center = 9999.f;
            int min_num = 0;
            for(int i = 0; i < RoadLines.size(); i++){
                float x_center = abs(RoadLines[i].x_center_end - 320.f);
                if(x_center < min_center){
                    min_center = x_center;
                    min_num = i;
                }
            }
            if(RoadLines.size() > 0)
                PreviousRoadLines.push_back(RoadLines[min_num]);
        }
        else{
            float min_space_err_rate = 9999.f;
            float min_inv_space_err_rate = 9999.f;
            float min_lr_err_rate = 9999.f;
            float min_theta_err_rate = 9999.f;
            int min_num = 99;
            float space_err_rate = 9999.f;
            float r1_err_rate = 9999.f;
            float r2_err_rate = 9999.f;
            float theat1_err_rate = 9999.f;
            float theat2_err_rate = 9999.f;
            float min_center = 9999.f;
            
            if(PreviousRoadLines.size()>0 ){
                for(int i = 0; i < RoadLines.size(); i++){
                    float x_center = abs(RoadLines[i].x_center_end - 320.f);
                    float p_x_center = abs(PreviousRoadLines[0].x_center_end - 320.f);
                    
                    if(x_center < min_center
                        //&& abs(abs(PreviousRoadLines[0].theta1) - abs(RoadLines[i].theta1))<15.f 
                        //&& abs(abs(PreviousRoadLines[0].theta2) - abs(RoadLines[i].theta2))<15.f
                        //&& abs(abs(PreviousRoadLines[0].r1) - abs(RoadLines[i].r1))<10.f
                        //&& abs(abs(PreviousRoadLines[0].r2) - abs(RoadLines[i].r2))<10.f
                        //&& abs(p_x_center - x_center) < 50.f
                        &&(RoadLines[i].end1.x < 320 && RoadLines[i].end2.x > 320)
                        ){
                        min_center = x_center;
                        min_num = i;
                    }
                }
            }

            for(int i =0; i < PreviousRoadLines.size(); i++){
                if(RoadLines.size() > 0 && (RoadLines[min_num].space_end <= PreviousRoadLines[i].space_end)){
                    space_err_rate = abs((RoadLines[min_num].space_end - PreviousRoadLines[i].space_end ) / PreviousRoadLines[i].space_end);
                }
                else if(RoadLines.size() > 0 && (RoadLines[min_num].space_end >= PreviousRoadLines[i].space_end)){
                    space_err_rate = abs((RoadLines[min_num].space_end - PreviousRoadLines[i].space_end ) / RoadLines[min_num].space_end);
                }
                if(RoadLines.size() > 0 && (RoadLines[min_num].r1 <= PreviousRoadLines[i].r1)){
                r1_err_rate = abs((PreviousRoadLines[i].r1 - RoadLines[min_num].r1) / PreviousRoadLines[i].r1);
                }
                else if(RoadLines.size() > 0 && (RoadLines[min_num].r1 >= PreviousRoadLines[i].r1)){
                    r1_err_rate = abs((RoadLines[min_num].r1 - PreviousRoadLines[i].r1 ) / RoadLines[min_num].r1);
                }

                if(RoadLines.size() > 0 && (RoadLines[min_num].r2 <= PreviousRoadLines[i].r2)){
                    r2_err_rate = abs((PreviousRoadLines[i].r2 - RoadLines[min_num].r2) / PreviousRoadLines[i].r2);
                }
                else if(RoadLines.size() > 0 && (RoadLines[min_num].r2 >= PreviousRoadLines[i].r2)){
                    r2_err_rate = abs((RoadLines[min_num].r2 - PreviousRoadLines[i].r2 ) / RoadLines[min_num].r2);
                }

                if(RoadLines.size() > 0 && (RoadLines[min_num].theta1 <= PreviousRoadLines[i].theta1)){
                    theat1_err_rate = abs((PreviousRoadLines[i].theta1 - RoadLines[min_num].theta1) / PreviousRoadLines[i].theta1);
                }
                else if(RoadLines.size() > 0 && (RoadLines[min_num].theta1 >= PreviousRoadLines[i].theta1)){
                    theat1_err_rate = abs((RoadLines[min_num].theta1 - PreviousRoadLines[i].theta1 ) / RoadLines[min_num].theta1);
                }

                if(RoadLines.size() > 0 && (RoadLines[min_num].theta2 <= PreviousRoadLines[i].theta2)){
                    theat2_err_rate = abs((PreviousRoadLines[i].theta2 - RoadLines[min_num].theta2) / PreviousRoadLines[i].theta2);
                }
                else if(RoadLines.size() > 0 && (RoadLines[min_num].theta2 >= PreviousRoadLines[i].theta2)){
                    theat2_err_rate = abs((RoadLines[min_num].theta2 - PreviousRoadLines[i].theta2 ) / RoadLines[min_num].theta2);
                }

            }
            if(RoadLines.size() > 0 && space_err_rate < 0.1f && r1_err_rate <0.1f && r2_err_rate <0.1f && theat1_err_rate < 0.2f && theat2_err_rate < 0.2f){
                PreviousRoadLines.clear();
                PreviousRoadLines.push_back(RoadLines[min_num]);
                tracking_success_count++;
            }
            else{
                PreviousRoadLines.clear();
                tracking_success_count = 0;
                tracking_success_flag = 0;
            }
            if(tracking_success_count == 10) {
                //TrackingRoadLines.push_back(RoadLines[min_num]);
                tracking_success_flag = 1;
            }
        }
    }
    else if(tracking_success_flag == 1){
        float min_space_err_rate = 9999.f;
        float min_inv_space_err_rate = 9999.f;
        float theta1_err = 9999.f;
        float theta2_err = 9999.f;
        int inv_space_min_num = 0;
        int space_min_num = 0;
        int min_num = 99;
        float space_err_rate = 9999.f;
        float inv_space_err_rate = 9999.f;
        float r1_err_rate = 9999.f;
        float r2_err_rate = 9999.f;
        int space_flag = 0;
        float min_center = 9999.f;

        if(PreviousRoadLines.size()>0 ){
            for(int i = 0; i < RoadLines.size(); i++){
                float x_center = abs(RoadLines[i].x_center_end - 320.f);
                float p_x_center = abs(PreviousRoadLines[0].x_center_end - 320.f);
                if( x_center < min_center 
                    && abs(PreviousRoadLines[0].inv_space_end - RoadLines[i].inv_space_end) < 50.f
                    && abs(p_x_center - x_center) < 50.f
                    && (RoadLines[i].end1.x < 320 && RoadLines[i].end2.x > 320)
                    ){
                    min_center = x_center;
                    min_num = i;
                }
            }
        }
        
        for(int i =0; i < PreviousRoadLines.size(); i++){
            if(RoadLines.size() > 0 && (RoadLines[min_num].space_end <= PreviousRoadLines[i].space_end)){
                space_err_rate = abs((RoadLines[min_num].space_end - PreviousRoadLines[i].space_end ) / PreviousRoadLines[i].space_end);
            }
            else if(RoadLines.size() > 0 && (RoadLines[min_num].space_end >= PreviousRoadLines[i].space_end)){
                space_err_rate = abs((RoadLines[min_num].space_end - PreviousRoadLines[i].space_end ) / RoadLines[min_num].space_end);
            }
        }
    

        //&& r1_err_rate <0.3f && r2_err_rate <0.3f && RoadLines[min_num].theta1 > 0.f && RoadLines[min_num].theta2 < -0.f
        if(RoadLines.size() > 0 && space_err_rate < 0.1f && RoadLines[min_num].theta1 > -20.f && RoadLines[min_num].theta2 < 20.f){
            PreviousRoadLines.clear();
            PreviousRoadLines.push_back(RoadLines[min_num]);
            TrackingRoadLines.push_back(RoadLines[min_num]);
            tracking_fail_count = 0;
            tracking_fail_flag = 0;
        }
        else{

            if(tracking_fail_count == 10){				
                PreviousRoadLines.clear();
                tracking_success_count = 0;
                tracking_success_flag = 0;
                tracking_fail_count = 0;
                //waitKey(1000);
            }
            else if( tracking_fail_count < 10 && PreviousRoadLines.size()>0){
                TrackingRoadLines.push_back(PreviousRoadLines[0]);
                tracking_fail_count++;
                tracking_fail_flag = 0;
            }

        }
        //std::cout << "tracking_fail_count " << tracking_fail_count << std::endl;
    }
    return TrackingRoadLines;
}

std::vector<LS> LANEDETECTION::Kalmanlines(std::vector<MatchSegment> TrackingRoadLines){

    flm = 0.f;
    frm = 0.f;
    flb = 0.f;
    frb = 0.f;

    std::vector<LS> FinalRoadLines;

    Lpredict = Ltrk.predict();
    Rpredict = Rtrk.predict();
    //std::cout << "Lpredict " << Lpredict << std::endl;
    for(int i = 0; i < TrackingRoadLines.size(); i++){
        /*
        std::cout << "T r1  " << TrackingRoadLines[i].r1 << std::endl;
        std::cout << "T t1 " << TrackingRoadLines[i].theta1 << std::endl;
        std::cout << "T r2  " << TrackingRoadLines[i].r2 << std::endl;
        std::cout << "T t2 " << TrackingRoadLines[i].theta2 << std::endl;
        */
        Ltrk.update(RTK(TrackingRoadLines[i].r1, TrackingRoadLines[i].theta1));
        LUpdatedvalue = Ltrk.get_state();
        float LRK = LUpdatedvalue.at<float>(0,0);
        float LTK = LUpdatedvalue.at<float>(1,0);
        
        float lx_pt = 0;
        float ly_pt = 0;
        float ly_shift = 0;
        float lm = 0;
        float lb = 0;
        float ly = 0;
        float lx = 0;
        
        if(LTK < 90.f && LTK > 0.f){	
            lx_pt = LRK / cos(LTK * CV_PI / 180.f);
            ly_pt = LRK / sin(LTK * CV_PI / 180.f);
            lm = -ly_pt / lx_pt;
            lb = -(lx_pt * ly_pt) / (-lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
        else{
            lx_pt = LRK / cos((LTK) * CV_PI / 180.f);
            ly_pt = LRK / sin((LTK) * CV_PI / 180.f);
            ly_shift = y_bias + abs(ly_pt);
            lm = -(ly_pt) / lx_pt;
            lb = ((lx_pt*2 * (ly_pt + ly_shift - y_bias)) - (lx_pt * (ly_shift - y_bias))) / (lx_pt*2 - lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }

        Rtrk.update(RTK(TrackingRoadLines[i].r2, TrackingRoadLines[i].theta2));
        RUpdatedvalue = Rtrk.get_state();
        float RRK = RUpdatedvalue.at<float>(0,0);
        float RTK = RUpdatedvalue.at<float>(1,0);
        
        float rx_pt = 0;
        float ry_pt = 0;
        float ry_shift = 0;
        float rm = 0;
        float rb = 0;
        float ry = 0;
        float rx = 0;

        if(RTK > -90.f && RTK < 0.f){	
            rx_pt = RRK / cos((RTK) * CV_PI / 180.f);
            ry_pt = RRK / sin((RTK) * CV_PI / 180.f);
            ry_shift = y_bias + abs(ry_pt);
            rm = -(ry_pt) / rx_pt;
            rb = ((rx_pt*2 * (ry_pt + ry_shift - y_bias)) - (rx_pt * (ry_shift - y_bias))) / (rx_pt*2 - rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        else{
            rx_pt = RRK / cos(RTK * CV_PI / 180.f);
            ry_pt = RRK / sin(RTK * CV_PI / 180.f);
            rm = -ry_pt / rx_pt;
            rb = -(rx_pt * ry_pt) / (-rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        flm = lm;
        frm = rm;
        flb = lb;
        frb = rb;
    }

    if(TrackingRoadLines.size()==0){
        
        float LRK = Lpredict.at<float>(0,0);
        float LTK = Lpredict.at<float>(1,0);
        Ltrk.update(RTK(LRK, LTK));
        LUpdatedvalue = Ltrk.get_state();
        float lx_pt = 0;
        float ly_pt = 0;
        float ly_shift = 0;
        float lm = 0;
        float lb = 0;
        float ly = 0;
        float lx = 0;

        if(LTK < 90.f && LTK > 0.f){	
            lx_pt = LRK / cos(LTK * CV_PI / 180.f);
            ly_pt = LRK / sin(LTK * CV_PI / 180.f);
            lm = -ly_pt / lx_pt;
            lb = -(lx_pt * ly_pt) / (-lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
        else{
            lx_pt = LRK / cos((LTK) * CV_PI / 180.f);
            ly_pt = LRK / sin((LTK) * CV_PI / 180.f);
            ly_shift = y_bias + abs(ly_pt);
            lm = -(ly_pt) / lx_pt;
            lb = ((lx_pt*2 * (ly_pt + ly_shift - y_bias)) - (lx_pt * (ly_shift - y_bias))) / (lx_pt*2 - lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }

        float RRK = Rpredict.at<float>(0,0);
        float RTK1 = Rpredict.at<float>(1,0);
        Rtrk.update(RTK(RRK, RTK1));
        RUpdatedvalue = Rtrk.get_state();
        float rx_pt = 0;
        float ry_pt = 0;
        float ry_shift = 0;
        float rm = 0;
        float rb = 0;
        float ry = 0;
        float rx = 0;

        if(RTK1 > -90.f && RTK1 < 0.f){	
            rx_pt = RRK / cos((RTK1) * CV_PI / 180.f);
            ry_pt = RRK / sin((RTK1) * CV_PI / 180.f);
            ry_shift = y_bias + abs(ry_pt);
            rm = -(ry_pt) / rx_pt;
            rb = ((rx_pt*2 * (ry_pt + ry_shift - y_bias)) - (rx_pt * (ry_shift - y_bias))) / (rx_pt*2 - rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        else{
            rx_pt = RRK / cos(RTK1 * CV_PI / 180.f);
            ry_pt = RRK / sin(RTK1 * CV_PI / 180.f);
            rm = -ry_pt / rx_pt;
            rb = -(rx_pt * ry_pt) / (-rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
    }

    return FinalRoadLines;
}

std::vector<LS> LANEDETECTION::Leftlines(std::vector<MatchSegment> TrackingRoadLines){

    flm = 0.f;
    flb = 0.f;
    
    std::vector<LS> FinalLeftRoadLines;

    for(int i = 0; i < TrackingRoadLines.size(); i++){

        float LRK = TrackingRoadLines[i].r1;
        float LTK = TrackingRoadLines[i].theta1;

        float lx_pt = 0;
        float ly_pt = 0;
        float ly_shift = 0;
        float lm = 0;
        float lb = 0;
        float ly = 0;
        float lx = 0;
        

        if(LTK < 90.f && LTK > 0.f){	
            lx_pt = LRK / cos(LTK * CV_PI / 180.f);
            ly_pt = LRK / sin(LTK * CV_PI / 180.f);
            lm = -ly_pt / lx_pt;
            lb = -(lx_pt * ly_pt) / (-lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            lx_pt = kalman_filter(&left_line_pts_x_start, lx_pt);
            lx = kalman_filter(&left_line_pts_x_end, lx);
            FinalLeftRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));  
        }
        else{
            lx_pt = LRK / cos((LTK) * CV_PI / 180.f);
            ly_pt = LRK / sin((LTK) * CV_PI / 180.f);
            ly_shift = y_bias + abs(ly_pt);
            lm = -(ly_pt) / lx_pt;
            lb = ((lx_pt*2 * (ly_pt + ly_shift - y_bias)) - (lx_pt * (ly_shift - y_bias))) / (lx_pt*2 - lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            lx_pt = kalman_filter(&left_line_pts_x_start, lx_pt);
            lx = kalman_filter(&left_line_pts_x_end, lx);
            FinalLeftRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
        flm = lm;
        flb = lb;
    }

    return FinalLeftRoadLines;
}

std::vector<LS> LANEDETECTION::Rightlines(std::vector<MatchSegment> TrackingRoadLines){

    frm = 0.f;
    frb = 0.f;

    std::vector<LS> FinalRightRoadLines;

    for(int i = 0; i < TrackingRoadLines.size(); i++){

        float RRK = TrackingRoadLines[i].r2;
        float RTK = TrackingRoadLines[i].theta2;
        
        float rx_pt = 0;
        float ry_pt = 0;
        float ry_shift = 0;
        float rm = 0;
        float rb = 0;
        float ry = 0;
        float rx = 0;

        if(RTK > -90.f && RTK < 0.f){	
            rx_pt = RRK / cos((RTK) * CV_PI / 180.f);
            ry_pt = RRK / sin((RTK) * CV_PI / 180.f);
            ry_shift = y_bias + abs(ry_pt);
            rm = -(ry_pt) / rx_pt;
            rb = ((rx_pt*2 * (ry_pt + ry_shift - y_bias)) - (rx_pt * (ry_shift - y_bias))) / (rx_pt*2 - rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            rx_pt = kalman_filter(&right_line_pts_x_start, rx_pt);
            rx = kalman_filter(&right_line_pts_x_end, rx);
            FinalRightRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        else{
            rx_pt = RRK / cos(RTK * CV_PI / 180.f);
            ry_pt = RRK / sin(RTK * CV_PI / 180.f);
            rm = -ry_pt / rx_pt;
            rb = -(rx_pt * ry_pt) / (-rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            rx_pt = kalman_filter(&right_line_pts_x_start, rx_pt);
            rx = kalman_filter(&right_line_pts_x_end, rx);
            FinalRightRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        frm = rm;
        frb = rb;
    }

    return FinalRightRoadLines;
}

std::vector<LS> LANEDETECTION::KalmanLeftlines(std::vector<MatchSegment> TrackingRoadLines){

    flm = 0.f;
    flb = 0.f;

    std::vector<LS> FinalLeftRoadLines;

    Lpredict = Ltrk.predict();
    //std::cout << "Lpredict " << Lpredict << std::endl;
    for(int i = 0; i < TrackingRoadLines.size(); i++){
        /*
        std::cout << "T r1  " << TrackingRoadLines[i].r1 << std::endl;
        std::cout << "T t1 " << TrackingRoadLines[i].theta1 << std::endl;
        std::cout << "T r2  " << TrackingRoadLines[i].r2 << std::endl;
        std::cout << "T t2 " << TrackingRoadLines[i].theta2 << std::endl;
        */
        Ltrk.update(RTK(TrackingRoadLines[i].r1, TrackingRoadLines[i].theta1));
        LUpdatedvalue = Ltrk.get_state();
        float LRK = LUpdatedvalue.at<float>(0,0);
        float LTK = LUpdatedvalue.at<float>(1,0);

        float lx_pt = 0;
        float ly_pt = 0;
        float ly_shift = 0;
        float lm = 0;
        float lb = 0;
        float ly = 0;
        float lx = 0;
        
        if(LTK < 90.f && LTK > 0.f){	
            lx_pt = LRK / cos(LTK * CV_PI / 180.f);
            ly_pt = LRK / sin(LTK * CV_PI / 180.f);
            lm = -ly_pt / lx_pt;
            lb = -(lx_pt * ly_pt) / (-lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalLeftRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
        else{
            lx_pt = LRK / cos((LTK) * CV_PI / 180.f);
            ly_pt = LRK / sin((LTK) * CV_PI / 180.f);
            ly_shift = y_bias + abs(ly_pt);
            lm = -(ly_pt) / lx_pt;
            lb = ((lx_pt*2 * (ly_pt + ly_shift - y_bias)) - (lx_pt * (ly_shift - y_bias))) / (lx_pt*2 - lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalLeftRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
        flm = lm;
        flb = lb;
    }

    if(TrackingRoadLines.size()==0){
        
        float LRK = Lpredict.at<float>(0,0);
        float LTK = Lpredict.at<float>(1,0);
        Ltrk.update(RTK(LRK, LTK));
        LUpdatedvalue = Ltrk.get_state();
        float lx_pt = 0;
        float ly_pt = 0;
        float ly_shift = 0;
        float lm = 0;
        float lb = 0;
        float ly = 0;
        float lx = 0;

        if(LTK < 90.f && LTK > 0.f){	
            lx_pt = LRK / cos(LTK * CV_PI / 180.f);
            ly_pt = LRK / sin(LTK * CV_PI / 180.f);
            lm = -ly_pt / lx_pt;
            lb = -(lx_pt * ly_pt) / (-lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalLeftRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
        else{
            lx_pt = LRK / cos((LTK) * CV_PI / 180.f);
            ly_pt = LRK / sin((LTK) * CV_PI / 180.f);
            ly_shift = y_bias + abs(ly_pt);
            lm = -(ly_pt) / lx_pt;
            lb = ((lx_pt*2 * (ly_pt + ly_shift - y_bias)) - (lx_pt * (ly_shift - y_bias))) / (lx_pt*2 - lx_pt);
            ly = lb;
            lx = ((359 - y_bias) - lb) / lm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalLeftRoadLines.push_back(LS(Point(lx_pt, 0), Point(lx, (359 - y_bias))));
        }
    }

    return FinalLeftRoadLines;
}

std::vector<LS> LANEDETECTION::KalmanRightlines(std::vector<MatchSegment> TrackingRoadLines){


    frm = 0.f;
    frb = 0.f;

    std::vector<LS> FinalRightRoadLines;

    Rpredict = Rtrk.predict();
    //std::cout << "Lpredict " << Lpredict << std::endl;
    for(int i = 0; i < TrackingRoadLines.size(); i++){
        /*
        std::cout << "T r1  " << TrackingRoadLines[i].r1 << std::endl;
        std::cout << "T t1 " << TrackingRoadLines[i].theta1 << std::endl;
        std::cout << "T r2  " << TrackingRoadLines[i].r2 << std::endl;
        std::cout << "T t2 " << TrackingRoadLines[i].theta2 << std::endl;
        */
        Rtrk.update(RTK(TrackingRoadLines[i].r2, TrackingRoadLines[i].theta2));
        RUpdatedvalue = Rtrk.get_state();
        float RRK = RUpdatedvalue.at<float>(0,0);
        float RTK = RUpdatedvalue.at<float>(1,0);
        
        float rx_pt = 0;
        float ry_pt = 0;
        float ry_shift = 0;
        float rm = 0;
        float rb = 0;
        float ry = 0;
        float rx = 0;

        if(RTK > -90.f && RTK < 0.f){	
            rx_pt = RRK / cos((RTK) * CV_PI / 180.f);
            ry_pt = RRK / sin((RTK) * CV_PI / 180.f);
            ry_shift = y_bias + abs(ry_pt);
            rm = -(ry_pt) / rx_pt;
            rb = ((rx_pt*2 * (ry_pt + ry_shift - y_bias)) - (rx_pt * (ry_shift - y_bias))) / (rx_pt*2 - rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRightRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        else{
            rx_pt = RRK / cos(RTK * CV_PI / 180.f);
            ry_pt = RRK / sin(RTK * CV_PI / 180.f);
            rm = -ry_pt / rx_pt;
            rb = -(rx_pt * ry_pt) / (-rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRightRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        frm = rm;
        frb = rb;
    }

    if(TrackingRoadLines.size()==0){
        
        float RRK = Rpredict.at<float>(0,0);
        float RTK1 = Rpredict.at<float>(1,0);
        Rtrk.update(RTK(RRK, RTK1));
        RUpdatedvalue = Rtrk.get_state();
        float rx_pt = 0;
        float ry_pt = 0;
        float ry_shift = 0;
        float rm = 0;
        float rb = 0;
        float ry = 0;
        float rx = 0;

        if(RTK1 > -90.f && RTK1 < 0.f){	
            rx_pt = RRK / cos((RTK1) * CV_PI / 180.f);
            ry_pt = RRK / sin((RTK1) * CV_PI / 180.f);
            ry_shift = y_bias + abs(ry_pt);
            rm = -(ry_pt) / rx_pt;
            rb = ((rx_pt*2 * (ry_pt + ry_shift - y_bias)) - (rx_pt * (ry_shift - y_bias))) / (rx_pt*2 - rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRightRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
        else{
            rx_pt = RRK / cos(RTK1 * CV_PI / 180.f);
            ry_pt = RRK / sin(RTK1 * CV_PI / 180.f);
            rm = -ry_pt / rx_pt;
            rb = -(rx_pt * ry_pt) / (-rx_pt);
            ry = rb;
            rx = ((359 - y_bias) - rb) / rm;
            //if(left_x < 0.f) AllLines.push_back(LS(Point(left_x_pt, 0), Point(0, left_b)));
            FinalRightRoadLines.push_back(LS(Point(rx_pt, 0), Point(rx, (359 - y_bias))));
        }
    }

    return FinalRightRoadLines;
}

std::vector<cv::Point> LANEDETECTION::LeftCurvePts(std::vector<MatchSegment> TrackingRoadLines){
    Mat ROIedgeImg = testED.getEdgeImage();
    std::vector<cv::Point> LeftCurvePts;
    LCurveStartx = 9999;
    LCurveEndx = 0;
    LCurveStarty = 9999;
    LCurveEndy = 0;

    if(TrackingRoadLines.size() > 0){
        int lx1,ly_f,lx2;
        std::vector<cv::Point> LSortPts;
        int lfirst_flag = 0;
        
        for(int i = (359 - y_bias) - 1 ; i > 0; i-=8){
            ly_f = i;
            if(lfirst_flag==0){
                    lx1 = (ly_f - flb)/flm;
                    lfirst_flag = 1;
            }
            
            for(int j = i - 8 ; j < i + 1; j++){
                uchar* data = ROIedgeImg.ptr<uchar>(j);
                for(int k = lx1 - 30 ; k <= lx1 + 31; k++){
                    uchar val = ROIedgeImg.at<uchar>(j,k);
                    if(data[k] > 250 && k >0 && k<640 && j > 0 && j<(359 - y_bias)){
                        if(k < LCurveStartx) LCurveStartx = k;
                        if(k > LCurveEndx) LCurveEndx = k;
                        if(j < LCurveStarty) LCurveStarty = j;
                        if(j > LCurveEndy) LCurveEndy = j;
                        LSortPts.push_back(cv::Point(k, j));
                        
                    }
                }
            }

            
            sort(LSortPts.begin(), LSortPts.end(), compareY);
            //std::cout << "SortPts size" << SortPts.size() << std::endl;
            int lmini_x = 9999;
            int lmini_n = 0;
            int lavg_x = 0;
            int lx_count = 0;
            if(LSortPts.size() > 0){

                //std::cout << "LSortPts miden " << LSortPts[LSortPts.size()/2] << std::endl;
                for (int k = 0; k < LSortPts.size(); k++)
                {
                    if(LSortPts[k].y==LSortPts[LSortPts.size()/2].y){
                        //std::cout << "LSortPts " << LSortPts[k] << std::endl;
                        if(abs(LSortPts[k].x - lx1) < lmini_x){
                            lmini_x = abs(LSortPts[k].x - lx1);
                            lmini_n = k;
                        }
                        lavg_x += LSortPts[k].x;
                        lx_count++;
                    }
                }
                lavg_x /= lx_count;
                //cv::rectangle(edgeImg,Rect(lavg_x-30,i-8,61,9),Scalar(0,255,0),1,1,0);
                LeftCurvePts.push_back(cv::Point(lavg_x, LSortPts[lmini_n].y + y_bias));
                lx1 = lavg_x;
                lfirst_flag = 1;
                LSortPts.clear();
            }
            else{
                lfirst_flag = 0;
                LSortPts.clear();
            }
        }
    }
    LeftCurvePts.erase(LeftCurvePts.begin());

    return LeftCurvePts;
}

std::vector<cv::Point> LANEDETECTION::RightCurvePts(std::vector<MatchSegment> TrackingRoadLines){
    Mat ROIedgeImg = testED.getEdgeImage();
    std::vector<cv::Point> RightCurvePts;
    RCurveStartx = 9999;
    RCurveEndx = 0;
    RCurveStarty = 9999;
    RCurveEndy = 0;

    if(TrackingRoadLines.size() > 0){
        int rx1,ry_f,rx2;
        std::vector<cv::Point> RSortPts;
        int rfirst_flag = 0;
        
        for(int i = (359 - y_bias) - 1 ; i > 0; i-=8){

            
            ry_f = i;
            if(rfirst_flag==0){
                    rx1 = (ry_f - frb)/frm;
                    rfirst_flag = 1;
            }

            for(int j = i - 8 ; j < i + 1; j++){
                uchar* data = ROIedgeImg.ptr<uchar>(j);
                
                for(int k = rx1 - 20 ; k <= rx1 + 21; k++){
                    uchar val = ROIedgeImg.at<uchar>(j,k);
                    if(data[k] > 250 && k >0 && k<640 && j > 0 && j<(359 - y_bias)){
                        if(k < RCurveStartx) RCurveStartx = k;
                        if(k > RCurveEndx) RCurveEndx = k;
                        if(j < RCurveStarty) RCurveStarty = j;
                        if(j > RCurveEndy) RCurveEndy = j;
                        RSortPts.push_back(cv::Point(k, j));
                        
                    }
                }
            }

            sort(RSortPts.begin(), RSortPts.end(), compareY);
            
            int rmini_x = 9999;
            int rmini_n = 0;
            if(RSortPts.size() > 0){
                //std::cout << "x" << x << std::endl;
                //std::cout << "SortPts miden" << SortPts[SortPts.size()/2] << std::endl;
                for (int k = 0; k < RSortPts.size(); k++)
                {
                    if(RSortPts[k].y==RSortPts[RSortPts.size()/2].y){
                        //std::cout << "RSortPts " << RSortPts[k] << std::endl;
                        if(abs(RSortPts[k].x - rx1) < rmini_x){
                            rmini_x = abs(RSortPts[k].x - rx1);
                            rmini_n = k;
                        }
                    }
                }
                //cv::rectangle(edgeImg,Rect(RSortPts[rmini_n].x-20,i-8,41,9),Scalar(0,255,0),1,1,0);
                RightCurvePts.push_back(cv::Point(RSortPts[rmini_n].x, RSortPts[rmini_n].y + y_bias));
                rx1 = RSortPts[rmini_n].x;
                rfirst_flag = 1;
                RSortPts.clear();
            }
            else{
                rfirst_flag = 0;
                RSortPts.clear();
            }

        }

    }
    
    RightCurvePts.erase(RightCurvePts.begin());

    return RightCurvePts;
}

void LANEDETECTION::Update_PreviousRoadLines(std::vector<LS> Leftlines, std::vector<LS> Rightlines){

    float left_m = (Leftlines[0].start.y - Leftlines[0].end.y)/(Leftlines[0].start.x -  Leftlines[0].end.x);
    float left_y = Leftlines[0].start.y;
    float left_x = Leftlines[0].start.x;
    float left_theta = atan (-1.f/left_m) * 180.f / CV_PI;
    float left_r = left_x * cos(left_theta * CV_PI / 180.f ) + left_y * sin(left_theta * CV_PI / 180.f );

    float right_m = (Rightlines[0].start.y - Rightlines[0].end.y)/(Rightlines[0].start.x -  Rightlines[0].end.x);
    float right_y = Rightlines[0].start.y;
    float right_x = Rightlines[0].start.x;
    float right_theta = atan (-1.f/right_m) * 180.f / CV_PI;
    float right_r = right_x * cos(right_theta * CV_PI / 180.f ) + right_y * sin(right_theta * CV_PI / 180.f );

    double left_d1_start = testED.line_width((Leftlines[0].start.x), Leftlines[0].start.y, 1.f);
    double left_d1_end = testED.line_width((Leftlines[0].end.x), Leftlines[0].end.y, 1.f);
    double right_d2_start = testED.line_width((Rightlines[0].start.x), Rightlines[0].start.y, 1.f);
    double right_d2_end = testED.line_width((Rightlines[0].end.x), Rightlines[0].end.y, 1.f);
    float space_start = abs(right_d2_start - left_d1_start);
    float space_end = abs(right_d2_end - left_d1_end);
    float inv_space_start = abs(abs(right_d2_start) - abs(left_d1_start));
    float inv_space_end = abs(abs(right_d2_end) - abs(left_d1_end));
    float x_center_start = (Leftlines[0].start.x + Rightlines[0].start.x) / 2.f;
    float x_center_end = (Leftlines[0].end.x + Rightlines[0].end.x) / 2.f;

    PreviousRoadLines[0].start1 = Leftlines[0].start;
	PreviousRoadLines[0].start2 = Rightlines[0].start;
	PreviousRoadLines[0].end1 = Leftlines[0].end;
	PreviousRoadLines[0].end2 = Rightlines[0].end;
	//PreviousRoadLines[0].r1 = left_r;
	//PreviousRoadLines[0].r2 = right_r;
	//PreviousRoadLines[0].theta1 = left_theta;
	//PreviousRoadLines[0].theta2 = right_theta;
	//PreviousRoadLines[0].d1_start = left_d1_start;
	//PreviousRoadLines[0].d1_end = left_d1_end;
	//PreviousRoadLines[0].d2_start = right_d2_start;
	//PreviousRoadLines[0].d2_end = right_d2_end;
	//PreviousRoadLines[0].space_start = space_start;
	//PreviousRoadLines[0].space_end = space_end;
	//PreviousRoadLines[0].inv_space_start = inv_space_start;
	//PreviousRoadLines[0].inv_space_end = inv_space_end;
	//PreviousRoadLines[0].x_center_start = x_center_start;
	//PreviousRoadLines[0].x_center_end = x_center_end;

}
void LANEDETECTION::k_init(){
    kalman_init(&left_line_pts_x_start,10,10);
    kalman_init(&left_line_pts_x_end,10,10);
    kalman_init(&left_line_pts_y_start,10,10);
    kalman_init(&left_line_pts_y_end,10,10);
    kalman_init(&right_line_pts_x_start,10,10);
    kalman_init(&right_line_pts_x_end,10,10);
    kalman_init(&right_line_pts_y_start,10,10);
    kalman_init(&right_line_pts_y_end,10,10);
}

void LANEDETECTION::kalman_init(KalmanStructTypedef *kalmanFilter, float init_x, float init_p)
{
    kalmanFilter->x = init_x;//待測量的初始值，如有中值一般設成中值
    kalmanFilter->p = init_p;//後驗狀態估計值誤差的方差的初始值（不要爲0問題不大）
    kalmanFilter->A = 1;
    kalmanFilter->H = 1;
    kalmanFilter->q = KALMAN_Q;//預測（過程）噪聲方差 影響收斂速率，可以根據實際需求給出
    kalmanFilter->r = KALMAN_R;//測量（觀測）噪聲方差 可以通過實驗手段獲得
}

float LANEDETECTION::kalman_filter(KalmanStructTypedef *kalmanFilter, float newMeasured)
{
    /* Predict */
    kalmanFilter->x = kalmanFilter->A * kalmanFilter->x;//%x的先驗估計由上一個時間點的後驗估計值和輸入信息給出
    kalmanFilter->p = kalmanFilter->A * kalmanFilter->A * kalmanFilter->p + kalmanFilter->q;  /*計算先驗均方差 p(n|n-1)=A^2*p(n-1|n-1)+q */

    /* Correct */
    kalmanFilter->gain = kalmanFilter->p * kalmanFilter->H / (kalmanFilter->p * kalmanFilter->H * kalmanFilter->H + kalmanFilter->r);
    kalmanFilter->x = kalmanFilter->x + kalmanFilter->gain * (newMeasured - kalmanFilter->H * kalmanFilter->x);//利用殘餘的信息改善對x(t)的估計，給出後驗估計，這個值也就是輸出
    kalmanFilter->p = (1 - kalmanFilter->gain * kalmanFilter->H) * kalmanFilter->p;//%計算後驗均方差

    return kalmanFilter->x;
}