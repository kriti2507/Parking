#include "image_properties.h"

//ros libraries
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Path.h"
#include "std_msgs/Int8MultiArray.h"
#include <std_msgs/Float64.h>
#include <std_msgs/Int8.h>
#include "std_msgs/Float64MultiArray.h"

//C++ libraries
#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>
#include <algorithm>

//GRANSAC header files
// #include "ransac/GRANSAC_CUDA.h"
#include "ransac/AbstractModel.hpp"
#include "ransac/QuadraticModel.hpp"
#include "ransac/LineModel.hpp"
#include "ransac/GRANSAC.h"

//Kalman filter libraries
#include "kalman/laneTracker.hpp"
#include "kalman/laneTrackerQuadratic.cpp"

//OpenCV libraries
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <utility>
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cuda.h>
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;


int tau=10, thresh=60;
int lane_dist=2,iterations=500,min_score=200;
int frame_number=1, loop_count;
string WINDOW = "Occupancy-Grid";

ros::Publisher pub_Lanedata;
ros::Publisher path_pub;
ros::Publisher pub1;
ros::Publisher pub_coeff;

int stop_threshold=15;
int stop_left=0, stop_right=0;
int stop_far_left=0, stop_far_right=0;

int class_left=0, class_right=0;
int class_far_left=0, class_far_right=0;


Mat mask_img,kalman_img, result_or, final_adding, published_result;

int area = 100;
std::vector<std::vector<Point> > extreme;
 std::vector<Vec4i> hierarchy;
//coefficients is a 4X5 vector
//4 corresponds to 4 lanes
//Each lane has (found status, class, coefficient of y^2, coefficient of y, coefficient of 1)
//found=1 if found, else 0
//class = 0(dashed lane), 1(single solid lane), 2(double solid lane)
//std::vector<std::vector<double>> coefficients;

struct myclass {
    bool operator() (vector<cv::Point> pt1, vector<cv::Point> pt2) { return (pt1[0].y < pt2[0].y);}
} myobject;

//logic for sorting
bool less_by_y(const cv::Point& lhs, const cv::Point& rhs){
  return lhs.y < rhs.y;
}

//To determine the bottom most point of a lane
double x_base(std::vector<GRANSAC::VPFloat> Parameters){
    return Parameters[0]*400*400 + Parameters[1]*400 + Parameters[2];
}

GRANSAC::VPFloat Slope(int x0, int y0, int x1, int y1)
{
    return (GRANSAC::VPFloat)(y1 - y0) / (x1 - x0);
}

void DrawFullLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar color, int LineWidth)
{
    GRANSAC::VPFloat slope = Slope(a.x, a.y, b.x, b.y);

    cv::Point p(0, 0), q(img.cols, img.rows);

    p.y = -(a.x - p.x) * slope + a.y;
    q.y = -(b.x - q.x) * slope + b.y;

    cv::line(img, p, q, color, LineWidth, 8, 0);
}

//Input: Grayscale of IPM image
//Output: Binary image containing lane candidate points
//Applies step-edge kernel with paramters "tau" (width of lane ion pixels) and "thresh" (thresholding for the step-edge-kernel score of every pixel) 

// For vertical detection
// Mat lane_filter_image(Mat src)
// {
//     Mat Score_Image=Mat::zeros(src.rows, src.cols, CV_8U);
//     Mat binary_inliers_image=Mat::zeros(src.rows, src.cols, CV_8U);
//     namedWindow(WINDOW,CV_WINDOW_AUTOSIZE);
//     createTrackbar("tau",WINDOW, &tau, 100);
//     tau = getTrackbarPos("tau",WINDOW);
//     createTrackbar("thresh",WINDOW, &thresh, 255);
//     thresh = getTrackbarPos("thresh",WINDOW);

//     for(int j=0; j<src.rows; j++){
//         unsigned char* ptRowSrc = src.ptr<uchar>(j);
//         unsigned char* ptRowSI = Score_Image.ptr<uchar>(j);
    
//         //Step-edge kernel
//         for(int i = tau; i< src.cols-tau; i++){
//             if(ptRowSrc[i]!=0){
//                 int aux = 2*ptRowSrc[i];
//                 aux += -ptRowSrc[i-tau];
//                 aux += -ptRowSrc[i+tau];
//                 aux += -2*abs((int)(ptRowSrc[i-tau]-ptRowSrc[i+tau]));
//                 aux = (aux<0)?(0):(aux);
//                 aux = (aux>255)?(255):(aux);

//                 ptRowSI[i] = (unsigned char)aux;
//             }
//         }
//     }
//     //Thresholding to form binary image. White points are lane candidate points
//     binary_inliers_image=Score_Image>thresh;
//     return binary_inliers_image;
// }

//For horizontal detection
Mat lane_filter_image(Mat src)
{
    Mat Score_Image=Mat::zeros(src.rows, src.cols, CV_8U);
    Mat binary_inliers_image=Mat::zeros(src.rows, src.cols, CV_8U);
    namedWindow(WINDOW,CV_WINDOW_AUTOSIZE);
    createTrackbar("tau",WINDOW, &tau, 100);
    tau = getTrackbarPos("tau",WINDOW);
    createTrackbar("thresh",WINDOW, &thresh, 255);
    thresh = getTrackbarPos("thresh",WINDOW);

    for(int j=0; j<src.cols; j++){
        unsigned char* ptColSrc = src.ptr<uchar>(j);
        unsigned char* ptColSI = Score_Image.ptr<uchar>(j);
    
        //Step-edge kernel
        for(int i = tau; i< src.rows-tau; i++){
            if(ptColSrc[i]!=0){
                int aux = 2*ptColSrc[i];
                aux += -ptColSrc[i-tau];
                aux += -ptColSrc[i+tau];
                aux += -2*abs((int)(ptColSrc[i-tau]-ptColSrc[i+tau]));
                aux = (aux<0)?(0):(aux);
                aux = (aux>255)?(255):(aux);

                ptColSI[i] = (unsigned char)aux;
            }
        }
    }
    //Thresholding to form binary image. White points are lane candidate points
    binary_inliers_image=Score_Image>thresh;
    return binary_inliers_image;
}


Mat ransac_fitting(Mat LaneFilter)
{
    if(LaneFilter.rows>0)
    {
        //lane_filter takes grayscale image as input
        Mat src2=lane_filter_image(LaneFilter);
        Mat Final_Lane_Filter = Mat::zeros(src2.rows, src2.cols, CV_8UC1);
        Mat result= Mat::zeros(Size(src2.cols,src2.rows), CV_8UC1);

        int dilation_size=1;
        Mat element = getStructuringElement( MORPH_RECT,
                                           Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           Point( dilation_size, dilation_size ) );
        
        // imshow("Lane filter",src2);
        //dilating to merge nearby lane blobs and prevent small lane blobs from disappearing when the image is down-sized
        dilate( src2,src2, element );

        //Area thresholding
        std::vector<std::vector<Point> > contour_area_threshold;

        findContours(src2, contour_area_threshold, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        createTrackbar("area",WINDOW, &area, 100);
        area = getTrackbarPos("area",WINDOW);
        for (int i = 0; i < contour_area_threshold.size(); ++i)
        {
            if (contourArea(contour_area_threshold[i]) >= area)
            {
                drawContours(Final_Lane_Filter, contour_area_threshold, i, Scalar(255),-1);
            }
        }
        
    
        // resize(blank,blank,Size(200,240),CV_INTER_LINEAR);  //If front cameras are used
        // copyMakeBorder(blank,blank,160,0,0,0,BORDER_CONSTANT,Scalar(0));    

        //Occupancy grid is of size 200X400 where 20 pixels corresponds to 1metre
        //So we resize the image by half and add 160 pixels padding to the top
        imshow("Final_right",Final_Lane_Filter);
        waitKey(1);
        Final_Lane_Filter.copyTo(kalman_img);
        //To store if the corresponding lanes are detected
        int found_fl=0,found_fr=0,found_1=0,found_2=0;
        //To store coefficients of quadratic equation of lane: x = ay^2 + by + c
        std::vector<GRANSAC::VPFloat> Parameters_fl, Parameters_fr, Parameters_1, Parameters_2;
        for(int n_lanes=0;n_lanes<4;n_lanes++)
        {
            vector< vector<GRANSAC::VPFloat>> SlopeIntercept;
            std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
            vector<Point> locations;   // output, locations of non-zero pixels
            //Locations is the vector of all lane candidate points
            findNonZero(Final_Lane_Filter, locations);
            //Stores top-most and bottom-most candidate points for drawing lanes in the end
            //Converting locations to a data-type compatible with GRANSAC package
            for(int i=0;i<contour_area_threshold.size();++i)
            {
                vector<GRANSAC::VPFloat> Params;
                for (int j = 0; j < contour_area_threshold[i].size(); ++j)
                {
                    cv::Point Pt=contour_area_threshold[i][j];
                    std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(Pt.x, Pt.y);
                    CandPoints.push_back(CandPt);
                }
            
            //GRANSAC parameters
            //A candidate lane point is considered an inlier if it is within lane_dist pixels away from the quadratic line
            createTrackbar("lane_dist",WINDOW, &lane_dist, 100);
            lane_dist = getTrackbarPos("lane_dist",WINDOW);
            //Number of iterations to run RANSAC
            createTrackbar("iterations",WINDOW, &iterations, 1000);
            iterations = getTrackbarPos("iterations",WINDOW);
            //Best-fit line is considered a lane if its inliers are more than min_score
            createTrackbar("min_score",WINDOW, &min_score, 1000);
            min_score = getTrackbarPos("min_score",WINDOW);
		    GRANSAC::RANSAC<Line2DModel, 2> Estimator;
			Estimator.Initialize(20, 100); // Threshold, iterations
			int start = cv::getTickCount();
			Estimator.Estimate(CandPoints);
			int end = cv::getTickCount();
			std::cout << "RANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;

			auto BestInliers = Estimator.GetBestInliers();
			if (BestInliers.size() > 0)
			{
				for (auto& Inlier : BestInliers)
				{
					auto RPt = std::dynamic_pointer_cast<Point2D>(Inlier);
					cv::Point Pt(floor(RPt->m_Point2D[0]), floor(RPt->m_Point2D[1]));
					// cv::circle(result, Pt, 5, cv::Scalar(255), -1);
				}
			}

			auto BestLine = Estimator.GetBestModel();
			if (BestLine)
			{
				auto BestLinePt1 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[0]);
				auto BestLinePt2 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[1]);
				if (BestLinePt1 && BestLinePt2)
				{
					cv::Point Pt1(BestLinePt1->m_Point2D[0], BestLinePt1->m_Point2D[1]);
					cv::Point Pt2(BestLinePt2->m_Point2D[0], BestLinePt2->m_Point2D[1]);
					// DrawFullLine(result, Pt1, Pt2, cv::Scalar(255), 2);
                    GRANSAC::VPFloat slope= Slope(Pt1.x, Pt1.y, Pt2.x, Pt2.y);
                    Params.push_back(slope);
                    Params.push_back(GRANSAC::VPFloat(Pt1.y- Pt1.x*slope));
				}
			}
            SlopeIntercept.push_back(Params);
		}
        //for all contours fitt lines and find intersections
        vector<Point2f> IntersectionPts;
        for(int w=0;w<SlopeIntercept.size();w++)
        {
            GRANSAC::VPFloat m1= SlopeIntercept[w][0];
            GRANSAC::VPFloat c1= SlopeIntercept[w][1];
            for(int v=w+1;v<SlopeIntercept.size(); v++)
            {
                GRANSAC::VPFloat m2= SlopeIntercept[v][0];
                GRANSAC::VPFloat c2= SlopeIntercept[v][1];
                //finding point of intersection
                Point2f P1= Point2f(float((c2-c1)/(m1-m2)), float((m1*c2-m2*c1)/(m1-m2)));
                IntersectionPts.push_back(P1);
            }
        }
        for(int h=0;h<IntersectionPts.size();h++)
        {
            //checking in a radius of 100 pixels in src2 image to see if the point calculated is right
            //force fitting it to a point which is nearest
            vector<Point> Neighbours;
            for(int j1=-5;j1<5;j1++)
            {
                for(int i1=-5;i1<5;i1++)
                {
                    Neighbours.push_back(Point(IntersectionPts[h].x-j1, IntersectionPts[h].y-i1));
                }
            }
            int count=0;
            for(int n=0;n<Neighbours.size();n++)
            {
                if(Neighbours[n].x<600 && Neighbours[n].x>0&& Neighbours[n].y<400 && Neighbours[n].y>0)
                {
                Scalar intensity= src2.at<uchar>(Point(Neighbours[n].x, Neighbours[n].y));
                if(intensity.val[0]!=0)
                {
                    count++;
                }
            }
            }
            if(count<40)
            {
                IntersectionPts.erase(IntersectionPts.begin()+h);
                h--;
            }
        }

                for(int u=0;u<IntersectionPts.size();u++)
        {
            Point2f P1= IntersectionPts[u];
            if(P1.x>0 && P1.y>0 &&P1.x<600 &&P1.y<600)
            {
            for(int z=u+1;z<IntersectionPts.size();z++)
            {
                Point2f P2= IntersectionPts[z];
                if(P2.x>0 && P2.y>0 &&P2.x<600 &&P2.y<400)
                {
                
                float distance= sqrt((P2.x-P1.x)*(P2.x-P1.x)+(P2.y-P1.y)*(P2.y-P1.y));
                if(distance< 30)
                {
                    IntersectionPts.erase(IntersectionPts.begin()+z);
                    z--;
                }
            }
            }
        }
    }

    //printing points
    for(int k=0;k<IntersectionPts.size();k++)
    {
        cout<<IntersectionPts[k]<<endl;
        circle(src2, IntersectionPts[k], 5, Scalar(255), -1);
    }
    if(IntersectionPts.size()>0)
    {
        std_msgs::Int8 msg;
        msg.data = 1;
        pub1.publish(msg);
    }
    else
    {
        std_msgs::Int8 msg;
        msg.data = 0;
        pub1.publish(msg);
    }
	}
    return result;
    imshow("points", src2);
    waitKey(1);
    //Decision making part
    
}
}

void ransac_fit_Image(const sensor_msgs::ImageConstPtr& msg){
    //Extract image from message
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        if (enc::isColor(msg->encoding))
            cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
        else
            cv_ptr = cv_bridge::toCvShare(msg, enc::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    //src is 400X480. Since lane_filter works better if the image is larger, we convert it to 200X240 only after passing it through lane_filter
    Mat gray,src = cv_ptr->image;
    cvtColor(src, gray, CV_BGR2GRAY);
    medianBlur( gray, gray, 5 );
    Mat src2=lane_filter_image(gray); //applying lane filter on original image
    Mat not_rotated_result=ransac_fitting(src2);
     cv::imshow("not_rotated_result",not_rotated_result);
    if(frame_number<=6 && frame_number!=1)
    {
        bitwise_or(not_rotated_result, final_adding, final_adding);
        frame_number=frame_number+1;
    }
    if(frame_number==1)
        {
            final_adding= Mat::zeros(src.rows,src.cols,CV_8U);
            frame_number=frame_number+1;
        }
    
    else if(frame_number>6)
    {
        frame_number=1;
        loop_count=0;
        published_result=final_adding;
        final_adding= Mat::zeros(src.rows, src.cols,CV_8U);
        std::vector<std::vector<Point> > contour;
        findContours(final_adding, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        
        extreme.clear();
        //Extremes of all lane blobs to be used by Path Planning module
        for (int i = 0; i < contour.size(); i++){
            Point extTop   = *min_element(contour[i].begin(), contour[i].end(),[](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
            Point extBot   = *max_element(contour[i].begin(), contour[i].end(),[](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
            vector<cv::Point> v{extTop,extBot};
            extreme.push_back(v);
}
}
    cv::imshow("Gray",gray);
    waitKey(1);
}

void endprocessing()
{

    nav_msgs::Path gui_path;
    geometry_msgs::PoseStamped pose;
    for (int i=0; i<extreme.size(); i++){
        pose.pose.position.x = extreme[i][0].x;
        pose.pose.position.y = extreme[i][0].y;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = 0.0;
        pose.pose.orientation.w = 1.0;
        //plan.push_back(pose);
        gui_path.poses.push_back(pose);
    }

    for (int i=0; i<extreme.size(); i++){
          pose.pose.position.x = extreme[i][1].x;
          pose.pose.position.y = extreme[i][1].y;
          pose.pose.position.z = 0.0;
          pose.pose.orientation.x = 0.0;
          pose.pose.orientation.y = 0.0;
          pose.pose.orientation.z = 0.0;
          pose.pose.orientation.w = 1.0;
          gui_path.poses.push_back(pose);
    }
    path_pub.publish(gui_path);
    gui_path.poses.clear();

    nav_msgs::OccupancyGrid Final_Grid;

    Final_Grid.info.map_load_time = ros::Time::now();
    Final_Grid.header.frame_id = "lane";
    Final_Grid.info.resolution = (float)map_width/(100*(float)occ_grid_widthr);
    Final_Grid.info.width = 600;
    Final_Grid.info.height = 400;

    Final_Grid.info.origin.position.x = 0;
    Final_Grid.info.origin.position.y = 0;
    Final_Grid.info.origin.position.z = 0;

    Final_Grid.info.origin.orientation.x = 0;
    Final_Grid.info.origin.orientation.y = 0;
    Final_Grid.info.origin.orientation.z = 0;
    Final_Grid.info.origin.orientation.w = 1;

    for (int i = 0; i < final_adding.rows; ++i)
    {
        for (int j = 0; j < final_adding.cols; ++j)
        {
            if ( final_adding.at<uchar>(i,j) > 0)
            {
                Final_Grid.data.push_back(2);
            }
            else
                Final_Grid.data.push_back(final_adding.at<uchar>(i,j));
        }
    }
    pub_Lanedata.publish(Final_Grid);
}

int main(int argc, char **argv)
{
    
    
    ros::init(argc, argv, "Lane_Occupancy_Grid");
    ros::NodeHandle nh;
    
    image_transport::ImageTransport it(nh);

    image_transport::Subscriber sub = it.subscribe("/camera/linknetlane", 1, ransac_fit_Image);    
    
    pub_coeff = nh.advertise<std_msgs::Float64MultiArray>("/Lane_coefficients", 1);
    pub1= nh.advertise<std_msgs::Int8>("/CheckParkingSPots", 1);
    pub_Lanedata = nh.advertise<nav_msgs::OccupancyGrid>("/Lane_Occupancy_Grid", 1);
    ros::Rate loop_rate(10);
    ros::NodeHandle p;
    path_pub = p.advertise<nav_msgs::Path>("/lane_coord", 1);



    while(ros::ok())
    {   
        ros::spinOnce();
        endprocessing();
        loop_rate.sleep();
    }
    ROS_INFO("videofeed::occupancygrid.cpp::No error.");
}
