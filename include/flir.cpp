#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <ctime>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>
#include <logger.h>
#include <yolo.h>
#include <config.h>

/*

g++ -std=c++11 flir.cpp -I /usr/local/include/opencv4/ -I /usr/local/flycapture2/include -L /usr/local/lib/ -L /usr/local/flycapture2/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lflycapture -lflycapturevideo -o flir_exe

g++ FlyCapture2Test.cpp -o FlyCapture2Test -L../../lib -lflycapture  -Wl,-rpath-link=../../lib

*/

namespace flir
{
    struct TempCoordinate
    {
        int x, y, R, G, B;
        double temp;
    };

    class ThermalFrame : public cv::Mat{};

    class FLIR
    {
        public:
            static const int defaultWidth=160, defaultHeight=120;
            double frameSafetyFactorX = config::flir::frameSafetyFactor::x;
            double frameSafetyFactorY = config::flir::frameSafetyFactor::y;
            
            int cameraID, width, height;
            int xmin, xmax, ymin, ymax;
            bool enableResize;
            cv::VideoCapture cap;
            int refreshThreshold = config::flir::refreshThresh;
            int refreshCounter = 0;

            FLIR(int width, int height)
            {
                logger::log("FLIR,Constructor,Instantiating","start");
                if((width > this->defaultWidth) && (height > this->defaultHeight))
                    this->enableResize = true;
                else
                {
                    if(!((width==this->defaultWidth) && (height==this->defaultHeight)))
                        logger::log("FLIR,Constructor,invalid Width and Height values,validValue=(w:"+std::to_string(width)+", h:"+std::to_string(height)+")","constructing");
                }

                this->width = width;
                this->height = height;
                this->xmin = (int)(this->frameSafetyFactorX*this->width);
                this->xmax = (int)((1.-this->frameSafetyFactorX)*this->width);
                this->ymin = (int)(this->frameSafetyFactorY*this->height);
                this->ymax = (int)((1.-this->frameSafetyFactorY)*this->height);
                std::stringstream ss; std::string s;
                ss << "FLIR,Constructor,setFrameLimit" << ",xmin=" << this->xmin << ",xmax=" << this->xmax << ",ymin=" << this->ymin << ",ymax=" << this->ymax;
                ss >> s;
                logger::debug(s,"DONE");

                if(!this->getCamera(0)) logger::debug("FLIR,Constructor,Instantiating,no matched camera id fro FLIR","FAILED");

                logger::log("FLIR,Constructor,Instantiating","end");
            }

            // ~FLIR(){}

            bool getCamera(int id)
            {
                if(0 <= id && id <= 10)
                {
                    try
                    {
                        logger::log("FLIR,scanCameraID,id="+std::to_string(id),"start");
                        this->cap.open(id, cv::VideoCaptureAPIs::CAP_V4L2);
                        if(this->cap.isOpened())
                        {
                            cv::Mat frame;
                            this->cap.read(frame);
                            if(frame.size[1]==this->defaultWidth && frame.size[0]==this->defaultHeight)
                            {
                                this->setCapY16();
                                // this->setCapRGB();
                                logger::log("FLIR,scanCameraID,id="+std::to_string(id),"valid");
                                return true;
                            }
                        }
                        logger::log("FLIR,scanCameraID,id="+std::to_string(id),"invalid");
                        this->cap.release();
                        return this->getCamera(id+1);
                    }
                    catch(std::exception& e)
                    {
                        logger::error("FLIR, Constructor", e.what());
                    }
                }
                return false;
            }

            void setCapY16()
            {
                logger::debug("FLIR,setCapFormat,format=Y16","START");
                this->cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','1','6',' '));
                this->cap.set(cv::CAP_PROP_CONVERT_RGB, false);
                logger::debug("FLIR,setCapFormat,format=Y16","END");
            }

            void setCapRGB()
            {
                logger::debug("FLIR,setCapFormat,format=RGB-UYVY","START");
                this->cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U','Y','V','Y'));
                this->cap.set(cv::CAP_PROP_CONVERT_RGB, true);
                logger::debug("FLIR,setCapFormat,format=RGB-UYVY","END");
            }

            cv::Mat getFrame()
            {
                logger::debug("FLIR,getFrame","START");
                cv::Mat frame, croppedFrame;
                this->cap.read(frame);
                try
                {
                    if(this->enableResize) cv::resize(frame, frame, cv::Size(this->width, this->height), 0., 0., cv::INTER_NEAREST);
                    croppedFrame = frame(cv::Rect(0,0,this->width,(int)(0.98*this->height)));
                }
                catch(const std::exception& e)
                {
                    logger::error(e);
                    logger::debug("FLIR,getFrame","RETRY");
                    return this->getFrame();
                }
                logger::debug("FLIR,getFrame","END");
                return croppedFrame;
            }

            double getTemp(cv::Mat frame, cv::Point center, int roiWidth, int roiHeight)
            {
                cv::Rect roi(center.x, center.y, center.x+roiWidth, center.y+roiHeight);
                return this->getTemp(frame, roi);
            }

            double getTemp(cv::Mat frame, cv::Rect box)
            {
                logger::debug("FLIR,getTemp","START");
                double temp = 0.;
                try
                {
                    int xcenter = (int)((box.x+box.width)/2);
                    int ycenter = (int)((box.y+box.height)/2);
                    if(this->xmin<xcenter && xcenter<this->xmax && this->ymin<ycenter && ycenter<this->ymax)
                    {
                        cv::Mat croppedFrame = frame(box);
                        double minVal, maxVal;
                        cv::minMaxLoc(croppedFrame, &minVal, &maxVal);
                        temp = (maxVal - 27315) / 100.0;
                        logger::debug("FLIR,getTemp,maxVal="+std::to_string(maxVal)+",temp="+std::to_string(temp),"END");
                    }
                    else
                    {
                        logger::error("FLIR,getTemp,InvalidROI,box out of range","END");
                    }
                }
                catch(const std::exception& e)
                {
                    logger::error(e);
                    logger::error("FLIR,getTemp,InvalidROI,box out of range","END");
                    return 0.;
                }
                return temp;
            }

            cv::Mat colorize(cv::Mat frame16)
            {
                logger::debug("FLIR,colorize","START");
                cv::Mat Img_Destination8Bit_Gray(this->width,this->height,CV_8UC1);
                cv::Mat colorFrame, outNorm;
                double minVal, maxVal;
                cv::minMaxLoc(frame16, &minVal, &maxVal);
                frame16.convertTo(Img_Destination8Bit_Gray, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
                cv::normalize(Img_Destination8Bit_Gray, outNorm, 0, 255., cv::NORM_MINMAX);
                cv::applyColorMap(outNorm, colorFrame, cv::COLORMAP_JET);
                logger::debug("FLIR,colorize","END");
                return colorFrame;
            }

            void scanMinMax(cv::Mat frame, TempCoordinate &tcMin, TempCoordinate &tcMax)
            {
                logger::debug("FLIR,scanMinMAX","START");
                cv::Rect roi(this->xmin, this->ymin, this->xmax-this->xmin, this->ymax-this->ymin);
                cv::Mat croppedFrame = frame(roi);

                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(croppedFrame, &minVal, &maxVal, &minLoc, &maxLoc);
                tcMin.temp = (minVal - 27315) / 100.0;
                tcMin.x = minLoc.x+this->xmin;
                tcMin.y = minLoc.y+this->ymin;
                tcMax.temp = (maxVal - 27315) / 100.0; 
                tcMax.x = maxLoc.x+this->xmin;
                tcMax.y = maxLoc.y+this->ymin;
                logger::debug("FLIR,scanMinMAX,minVal="+std::to_string(minVal)+",maxVal="+std::to_string(maxVal),"END");
            }

            double calibrateTemp(double temp, double distance)
            {
                logger::debug("FLIR,calibrateTemp,temp="+std::to_string(temp)+",dist="+std::to_string(distance),"START");
                double minDist = config::flir::distPolyRange[0];
                double maxDist = config::flir::distPolyRange[1];
                double minTemp = config::flir::tempPolynom[0];
                double maxTemp = config::flir::tempPolyRange[1];
                std::vector<double> polyDist = config::flir::distPolynom;
                std::vector<double> polyTemp = config::flir::tempPolynom;

                if(minTemp <= temp && temp <= maxTemp && minDist <= distance && distance <= maxDist)
                {
                    // distance correction
                    double calTemp = temp+((polyDist[0]*distance*distance)+(polyDist[1]*distance)+polyDist[2]);

                    // temperature correction
                    calTemp = (calTemp*polyTemp[0]*polyTemp[0])+(calTemp*polyTemp[1])+polyTemp[2];

                    logger::debug("FLIR,calibrateTemp,temp="+std::to_string(calTemp),"END");
                    return calTemp;
                }
                logger::debug("FLIR,calibrateTemp,outOfCalibrationRange,temp="+std::to_string(temp),"END");
                return temp;
            }
    };
}