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
#include <librealsense2/rs.hpp>
#include "logger.h"

namespace camera
{
    // framesize from realsense camera
    struct FrameSize
    {
        int colorWidth;
        int colorHeight;
        int depthWidth;
        int depthHeight;
    };

    // class to hold color and depth frames
    class Frames
    {
        public:
            cv::Mat colorFrame;
            rs2::depth_frame depthFrame = rs2::frameset().get_depth_frame();
            Frames(cv::Mat colorFrame, rs2::depth_frame depthFrame)
            {
                this->colorFrame = colorFrame;
                this->depthFrame = depthFrame;
            }
    };

    // class to handle realsense camera
    class RealSenseCamera
    {
        public:
            rs2::pipeline pipe;
            rs2::config config;
            rs2::frameset rsFrame;
            cv::Mat frame;
            camera::FrameSize frameSize;
            rs2::depth_frame depthFrame = rsFrame.get_depth_frame();

            RealSenseCamera(camera::FrameSize &frameSize)
            {
                logger::log("RealSenseCamera,Instantiation","START");
                this->frameSize = frameSize;

                logger::log("RealSenseCamera,Instantiation","enabling color stream");
                this->config.enable_stream(RS2_STREAM_COLOR, this->frameSize.colorWidth, this->frameSize.colorHeight, RS2_FORMAT_BGR8, 0);

                logger::log("RealSenseCamera,Instantiation","enabling depth stream");
                this->config.enable_stream(RS2_STREAM_DEPTH, this->frameSize.depthWidth, this->frameSize.depthHeight, RS2_FORMAT_Z16, 0);

                logger::log("RealSenseCamera,Instantiation","starting pipeline");
                rs2::pipeline_profile selection = this->pipe.start(this->config);

                logger::log("RealSenseCamera,Instantiation","enabling auto exposure");
                rs2::device selected_device = selection.get_device();
                rs2::depth_stereo_sensor depth_sensor = selected_device.first<rs2::depth_stereo_sensor>();
                depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

                logger::log("RealSenseCamera,Instantiation","END");
            }

            void read()
            {
                logger::debug("RealSenseCamera,read","START");
                this->rsFrame = this->pipe.wait_for_frames();
                rs2::frame colorFrame = this->rsFrame.get_color_frame();
                this->frame = cv::Mat(cv::Size(this->frameSize.colorWidth, this->frameSize.colorHeight), CV_8UC3, (void*) colorFrame.get_data(), cv::Mat::AUTO_STEP);
                this->depthFrame = this->rsFrame.get_depth_frame();
                logger::debug("RealSenseCamera,read","END");
                // return camera::Frames(this->frame, this->depthFrame);
            }
    };
}