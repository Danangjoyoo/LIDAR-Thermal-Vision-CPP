/* 

g++ -std=c++11 main.cpp -I /home/rnd/dev/dnn1/include -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lopencv_cudaobjdetect -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_dnn_objdetect -lopencv_dnn -lpthread -o main

# flir lidar scanner (centerpoint)
g++ -std=c++11 /home/rnd/dev/dnn1/main.cpp -I /home/rnd/Documents -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_dnn_objdetect -lopencv_dnn -lpthread  -o /home/rnd/dev/dnn1/center_scan

*/

#include <fstream>
#include <iomanip>
#include <opencv2/dnn_superres.hpp>
#include "visioncam.h"

struct FrameSize
{
    int colorWidth;
    int colorHeight;
    int depthWidth;
    int depthHeight;
    int thermalWidth;
    int thermalHeight;
};

class Scales
{
    public:
        double x;
        double y;
        Scales(double x, double y)
        {
            this->x = x;
            this->y = y;
        }
};

void yoloDetection();
void run(YoloClassifier &classifier);
void drawBoxes(YoloClassifier &classifier, cv::Mat &frame);
void drawBoxesWithDepth(YoloClassifier &classifier, cv::Mat &frame, rs2::depth_frame depthFrame);
void drawBoxesWithDepthROI(YoloClassifier &classifier, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize);
void drawBoxesWithDepthROI_plusDepthColor(YoloClassifier &classifier, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize);
float getValidDistanceInBoxArea(rs2::depth_frame depthFrame, cv::Point centerPoint, int xRange, int yRange);
void runFlir();
void newRun();
void flirScan();
Scales getCalibratedDepthScale(FrameSize &fsize);
std::vector<std::vector<int>> getCalibratedThermalValues(FrameSize &fsize);
cv::Point convertColorToThermal(int x, int y, FrameSize &fsize);
cv::Rect convertColorToThermal(flir::FLIR flir, cv::Rect box, FrameSize &fsize);
cv::Point convertThermalToDepth(flir::FLIR flir, cv::Point thermalCenter, FrameSize fsize, bool &inRange);
void drawBoxesAndTrack(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, FrameSize &fsize);
void drawBoxesAndTrackWithDepth(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize);
void drawBoxesAndTrackWithDepth_plusThermalColor(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize);
void drawTrackAndShowAll(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize, bool showThermal, bool showDepth);
tracker::TrackedObject *trackBoxes(YoloClassifier &classifier, int idx, flir::FLIR &flir, cv::Mat &frame);
void visionCam();
void runVisionCam(YoloClassifier &classifier);
void visionCamHelp();
void flirXlidarScan();
void visionCamSuperResolution();
void runVisionCamSuperResolution(YoloClassifier &classifier, cv::dnn_superres::DnnSuperResImpl superRes);

int main(int argc, char* argv[])
{
    try
    {
        argparser::set(argc, argv);
        argparser::KwargsMap mappedArgs;

        if(mappedArgs.checkArgs("-h") || mappedArgs.checkArgs("--help"))
        {
            visionCamHelp();
            return 1;
        }
        
        if(argc>1)
        {
            if(mappedArgs.checkArgs("-mode"))
            {
                // set logger -l <log_level>
                if(mappedArgs.checkArgs("-l"))
                {
                    int logLevel = std::stoi(mappedArgs.getArgs("-l"));
                    logger::setLevel(logLevel);
                }
                else logger::setLevel(config::logger::level);

                // set logger --logger
                if(mappedArgs.checkArgs("--log"))
                    logger::enable(true, "./log.txt");
                else
                    logger::enable(config::logger::enable, "./log.txt");

                logger::log("Main","START");

                // select mode
                switch(std::stoi(mappedArgs.getArgs("-mode")))
                {
                case (0): visionCam(); break;
                case (1): flirScan(); break;
                case (2): flirXlidarScan(); break;
                case (3): visionCamSuperResolution(); break;
                default: break;
                }
            }
            else 
            {
                std::cout << " Please insert mode number -mode" << std::endl;
                std::cout << " use -h/--help to see help" << std::endl;
                // visionCamHelp();
            }
        }
        else
        {
            std::cout << " Please insert mode number -mode" << std::endl;
            std::cout << " use -h/--help to see help" << std::endl;
            // visionCamHelp();
        }
        
        // visionCam();
        // yoloDetection();
        // flirScan();
        // runFlir();
        // flirXlidarScan();
        // newRun();
    }
    catch(const std::exception& e)
    {
        logger::error(e);
    }

    logger::log("Main","END");
    logger::enable(false);

    return 1;
}


void runFlir()
{
    flir::FLIR flir = flir::FLIR(640, 480);
    // flir::FLIR flir = flir::FLIR(160, 120);
    cv::Mat frame;

    while(1)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        frame = flir.getFrame();

        auto t1 = std::chrono::high_resolution_clock::now();
        int fps = 1e9/(t1-t0).count();
        std::string fpsText = "FPS: "+std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,240), 2);

        cv::imshow("flir", frame);
        if(cv::waitKey(5) > 0) break;
    }
}


void flirScan()
{
    flir::FLIR flir = flir::FLIR(640, 480);
    cv::Mat frame;
    usleep(1e6);

    while(1)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        frame = flir.getFrame();
        flir::TempCoordinate tcmax, tcmin;
        flir.scanMinMax(frame, tcmin, tcmax);

        cv::Mat colorFrame;
        frame = flir.colorize(frame);

        if(tcmax.x && tcmax.y)
        {
            cv::circle(frame, cv::Point(tcmax.x, tcmax.y), 10, cv::Scalar(0,0,0), 2);
            cv::circle(frame, cv::Point(tcmax.x, tcmax.y), 5, cv::Scalar(255,255,255), -1);

            std::stringstream ss; std::string s;
            ss << "Temp=" << tcmax.temp << ",pos=(x:" << tcmax.x << ",y:" << tcmax.y << ")"; ss >> s;
            std::stringstream ss1; std::string s1;
            ss1 << std::setprecision(4) << tcmax.temp; ss1 >> s1;
            cv::putText(frame, s, cv::Point(tcmax.x-70, tcmax.y-20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,0), 1);
            cv::putText(frame, s1+" C", cv::Point(tcmax.x-20, tcmax.y+30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
        }

        if(tcmin.x && tcmin.y)
        {
            cv::circle(frame, cv::Point(tcmin.x, tcmin.y), 10, cv::Scalar(255,255,255), 2);
            cv::circle(frame, cv::Point(tcmin.x, tcmin.y), 5, cv::Scalar(0,0,0), -1);
            std::stringstream ss; std::string s;
            ss << "Temp=" << tcmin.temp << ",pos=(x:" << tcmin.x << ",y:" << tcmin.y << ")"; ss >> s;
            std::stringstream ss1; std::string s1;
            ss1 << std::setprecision(4) << tcmin.temp; ss1 >> s1;
            cv::putText(frame, s, cv::Point(tcmin.x-70, tcmin.y-20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1);
            cv::putText(frame, s1+" C", cv::Point(tcmin.x-20, tcmin.y+30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        int fps = 1e9/(t1-t0).count();
        std::string fpsText = "FPS: "+std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,240), 2);
        cv::imshow("flir", frame);
        if(cv::waitKey(5) > 0) break;
    }
}


void flirXlidarScan()
{
    rs2::pipeline pl;
    rs2::config customConfig;
    FrameSize frameSize;

    frameSize.colorWidth = config::camera::colorFrame.width;
    frameSize.colorHeight = config::camera::colorFrame.height;
    frameSize.depthWidth = config::camera::depthFrame.width;
    frameSize.depthHeight = config::camera::depthFrame.height;
    frameSize.thermalWidth = config::camera::thermalFrame.width;
    frameSize.thermalHeight = config::camera::thermalFrame.height;

    flir::FLIR flir = flir::FLIR(frameSize.thermalWidth, frameSize.thermalHeight);

    customConfig.enable_stream(RS2_STREAM_DEPTH, frameSize.depthWidth, frameSize.depthHeight, RS2_FORMAT_Z16, 0);
    rs2::frameset rsFrame;

    rs2::pipeline_profile selection = pl.start(customConfig);
    rs2::device selected_device = selection.get_device();
    rs2::depth_stereo_sensor depth_sensor = selected_device.first<rs2::depth_stereo_sensor>();
    depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

    cv::Mat frame;

    usleep(1e6);

    while(1)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        rsFrame = pl.wait_for_frames();
        rs2::depth_frame depthFrame = rsFrame.get_depth_frame();
        frame = flir.getFrame();
        flir::TempCoordinate tcmax, tcmin;
        // flir.scanMinMax(frame, tcmin, tcmax);
        tcmax.x = (int)(flir.width*1./2);
        tcmax.y = (int)(flir.height*1./2);
        tcmax.temp = flir.getTemp(frame, cv::Rect(tcmax.x-15, tcmax.y-15, 30, 30));

        cv::Mat colorFrame;
        frame = flir.colorize(frame);

        rs2::colorizer colormap;
        rs2::video_frame vframe = colormap.colorize(depthFrame);
        cv::Mat cdFrame = cv::Mat(cv::Size(frameSize.depthWidth, frameSize.depthHeight), CV_8UC3, (void*) vframe.get_data(), cv::Mat::AUTO_STEP);

        if(tcmax.x && tcmax.y)
        {
            // cv::circle(frame, cv::Point(tcmax.x, tcmax.y), 7, cv::Scalar(0,0,0), 2);
            cv::rectangle(frame, cv::Rect(tcmax.x-15, tcmax.y-15, 30, 30), cv::Scalar(0,0,0), 2);
            cv::circle(frame, cv::Point(tcmax.x, tcmax.y), 3, cv::Scalar(255,255,255), -1);

            std::stringstream ss; std::string s;
            ss << std::setprecision(4) << tcmax.temp; ss >> s;
            // cv::putText(frame, s+" C", cv::Point(tcmax.x-30, tcmax.y-20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 2);
            cv::putText(frame, "(ori)"+s+" C", cv::Point(tcmax.x-30, tcmax.y-20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 2);
            
            bool inRange = true;
            cv::Point depthCenter = convertThermalToDepth(flir, cv::Point(tcmax.x, tcmax.y), frameSize, inRange);
            depthCenter.x -= 30;
            depthCenter.y -= 10;

            float distance = getValidDistanceInBoxArea(depthFrame, cv::Point(depthCenter.x, depthCenter.y), 5, 5);
            int distanceInCm = (int)(distance*100);
            std::string distanceText = std::to_string(distanceInCm)+" cm";
            cv::putText(frame, distanceText, cv::Point(tcmax.x-30, tcmax.y+30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 2);
            cv::circle(cdFrame, cv::Point(depthCenter.x, depthCenter.y), 10, cv::Scalar(0,0,0), 2);
            cv::circle(cdFrame, cv::Point(depthCenter.x, depthCenter.y), 5, cv::Scalar(255,255,255), -1);
            
            double calTemp = flir.calibrateTemp(tcmax.temp, distanceInCm*1.);
            std::stringstream ss1; std::string s1;
            ss1 << std::setprecision(4) << calTemp; ss1 >> s1;
            cv::putText(frame, "(cal)"+s1+" C", cv::Point(tcmax.x-35, tcmax.y-40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 2);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        int fps = 1e9/(t1-t0).count();
        std::string fpsText = "FPS: "+std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,240), 2);
        cv::imshow("flir", frame);
        cv::imshow("depth", cdFrame);
        if(cv::waitKey(5) > 0) break;
    }
}


void visionCamSuperResolution()
{
    std::vector<cv::String> availableClasses = config::objectDetection::models::classes;
    std::vector<cv::Scalar> colors = config::objectDetection::models::colors;

    logger::log("Program,Retrieving config file","START");
    // YOLO config file
    cv::String cfgFile(config::objectDetection::models::configfile);
    logger::log("Program,Retrieving config file","END");

    logger::log("Program,Retrieving DNN model","START");
    // TRAINED YOLO MODEL
    cv::String model(config::objectDetection::models::modelfile);
    logger::log("Program,Retrieving DNN model","END");

    logger::log("Program,generatingNeuralNet","START");
    // darknet dnn object
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfgFile, model);
    logger::log("Program,generatingNeuralNet","END");

    logger::log("Program,SetComputationDevice,target=CUDA","START");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    logger::log("Program,SetComputationDevice,target=CUDA","END");

    logger::log("Program,superResolution,SettingUp,target=EDSR","START");
    cv::dnn_superres::DnnSuperResImpl superRes;
    superRes.readModel("super-res-models/EDSR_x2.pb");
    superRes.setModel("edsr",2);
    superRes.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    superRes.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    logger::log("Program,superResolution,SettingUp,target=EDSR","END");

    float confidenceThresh = config::objectDetection::confidenceThreshold;

    // instantiation
    YoloClassifier ml(net, availableClasses, colors, confidenceThresh, config::objectDetection::models::inputSize);

    runVisionCamSuperResolution(ml, superRes);

}

void runVisionCamSuperResolution(YoloClassifier &classifier, cv::dnn_superres::DnnSuperResImpl superRes)
{

    logger::log("Main,run","START");
    int fps = 0, sumFps = 0;
    double avgFps = 0;
    
    rs2::pipeline pl;
    rs2::config customConfig;
    FrameSize frameSize;

    classifier.targetSize = cv::Size(config::camera::colorFrame.width, config::camera::colorFrame.height);
    // frameSize.colorWidth = 424;
    // frameSize.colorHeight = 240;

    frameSize.colorWidth = config::camera::colorFrame.width;
    frameSize.colorHeight = config::camera::colorFrame.height;
    frameSize.depthWidth = config::camera::depthFrame.width;
    frameSize.depthHeight = config::camera::depthFrame.height;
    frameSize.thermalWidth = config::camera::thermalFrame.width;
    frameSize.thermalHeight = config::camera::thermalFrame.height;

    logger::info("Main,run,colorFrame,size=(w:"+std::to_string(config::camera::colorFrame.width)+",h:"+std::to_string(config::camera::colorFrame.height)+")","DONE");
    logger::info("Main,run,depthFrame,size=(w:"+std::to_string(config::camera::depthFrame.width)+",h:"+std::to_string(config::camera::depthFrame.height)+")","DONE");
    logger::info("Main,run,thermalFrame,size=(w:"+std::to_string(config::camera::thermalFrame.width)+",h:"+std::to_string(config::camera::thermalFrame.height)+")","DONE");

    flir::FLIR flir = flir::FLIR(frameSize.thermalWidth, frameSize.thermalHeight);
    tracker::enableGlobalTempUpdate(config::objectTracking::useGlobalTemperatureUpdate);

    customConfig.enable_stream(RS2_STREAM_COLOR, 424, 240, RS2_FORMAT_BGR8, 0);
    // customConfig.enable_stream(RS2_STREAM_COLOR, frameSize.colorWidth, frameSize.colorHeight, RS2_FORMAT_BGR8, 0);
    customConfig.enable_stream(RS2_STREAM_DEPTH, frameSize.depthWidth, frameSize.depthHeight, RS2_FORMAT_Z16, 0);
    rs2::frameset rsFrame;

    rs2::pipeline_profile selection = pl.start(customConfig);
    rs2::device selected_device = selection.get_device();
    rs2::depth_stereo_sensor depth_sensor = selected_device.first<rs2::depth_stereo_sensor>();
    depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

    cv::Mat frame;
    cv::String windowName("Camera Vision");
    cv::namedWindow(windowName, cv::WindowFlags::WINDOW_FULLSCREEN);
    cv::setWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_FULLSCREEN, cv::WindowFlags::WINDOW_FULLSCREEN);

    std::vector<std::vector<int>> thermalProjectedFrame = getCalibratedThermalValues(frameSize);

    argparser::KwargsMap mappedArgs;
    bool showThermal = mappedArgs.checkArgs("-t");
    bool showDepth = mappedArgs.checkArgs("-d");

    logger::log("Main,run,looping","START");
    while(1)
    {
        logger::debug("Main,run,innerLoop","START");
        auto t0 = std::chrono::high_resolution_clock::now();

        logger::debug("Main,run,innerLoop,waitForFrames","START");
        // if(!pl.try_wait_for_frames(&rsFrame)) continue;
        rsFrame = pl.wait_for_frames();
        logger::debug("Main,run,innerLoop,waitForFrames","END");

        logger::debug("Main,run,innerLoop,getColorFrame","START");
        rs2::frame colorFrame = rsFrame.get_color_frame();
        logger::debug("Main,run,innerLoop,getColorFrame","END");

        logger::debug("Main,run,innerLoop,getDepthFrame","START");
        rs2::depth_frame depthFrame = rsFrame.get_depth_frame();
        logger::debug("Main,run,innerLoop,getDepthFrame","END");

        logger::debug("Main,run,innerLoop,MatricesConversion,frameType=color","START");
        frame = cv::Mat(cv::Size(424, 240), CV_8UC3, (void*) colorFrame.get_data(), cv::Mat::AUTO_STEP);
        // frame = cv::Mat(cv::Size(frameSize.colorWidth, frameSize.colorHeight), CV_8UC3, (void*) colorFrame.get_data(), cv::Mat::AUTO_STEP);
        logger::debug("Main,run,innerLoop,MatricesConversion,frameType=color","END");

        logger::debug("Main,run,innerLoop,resize","START");
        // superRes.upsample(frame, frame);
        // logger::debug("Main,run,innerLoop,resize,superResolution","DONE");
        cv::resize(frame, frame, classifier.targetSize, 0., 0., cv::INTER_CUBIC);
        logger::debug("Main,run,innerLoop,resize,fitToScreen","DONE");

        logger::debug("Main,run,innerLoop,detect","START");
        classifier.detect(frame);
        logger::debug("Main,run,innerLoop,detect","END");

        // ####### write your desired draws below (after classifier.detect) ####### 

        if(config::flir::showFrameBoundary)
        {
            cv::putText(frame, "Thermal Update Box", cv::Point(thermalProjectedFrame[0][0], thermalProjectedFrame[0][1]-15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(250,5,5), 1);
            cv::rectangle(frame, cv::Rect(thermalProjectedFrame[0][0], thermalProjectedFrame[0][1], thermalProjectedFrame[1][0], thermalProjectedFrame[1][1]), cv::Scalar(250,5,5), 2);
        }

        logger::debug("Main,run,innerLoop,draw","START");
        if(mappedArgs.checkArgs("-vc"))
        {
            switch(std::stoi(mappedArgs.getArgs("-vc")))
            {
                case (0): drawBoxes(classifier, frame); break;
                case (1): drawBoxesWithDepthROI(classifier, frame, depthFrame, frameSize); break;
                case (2): drawBoxesWithDepthROI_plusDepthColor(classifier, frame, depthFrame, frameSize); break;
                case (3): drawBoxesAndTrack(classifier, flir, frame, frameSize); break;
                case (4): drawBoxesAndTrackWithDepth(classifier, flir, frame, depthFrame, frameSize); break;
                case (5): drawBoxesAndTrackWithDepth_plusThermalColor(classifier, flir, frame, depthFrame, frameSize); break;
                case (6): drawTrackAndShowAll(classifier, flir, frame, depthFrame, frameSize, showThermal, showDepth); break;
                default: break;
            }
        }
        else
        {
            // drawBoxes(classifier, frame);
            // drawBoxesWithDepthROI(classifier, frame, depthFrame, frameSize);
            // drawBoxesWithDepthROI_plusDepthColor(classifier, frame, depthFrame, frameSize);
            // drawBoxesAndTrack(classifier, flir, frame, frameSize);
            // drawBoxesAndTrackWithDepth(classifier, flir, frame, depthFrame, frameSize);
            // drawBoxesAndTrackWithDepth_plusThermalColor(classifier, flir, frame, depthFrame, frameSize);
            drawTrackAndShowAll(classifier, flir, frame, depthFrame, frameSize, showThermal, showDepth);
        }
        logger::debug("Main,run,innerLoop,draw","END");

        auto t1 = std::chrono::high_resolution_clock::now();
        int fps = 1e9/(t1-t0).count();
        std::string fpsText = "FPS: "+std::to_string(fps);

        logger::debug("Main,run,innerLoop,frameRendering","START");
        // cv::resize(frame, frame, classifier.targetSize, 0., 0., cv::INTER_CUBIC);
        cv::putText(frame, fpsText, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(5,5,240), 2);
        cv::putText(frame, "count: "+std::to_string(tracker::totalValidObject), cv::Point(20,frame.rows-80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(5,5,240), 2);
        cv::imshow(windowName, frame);
        logger::debug("Main,run,innerLoop,frameRendering","END");

        if(cv::waitKey(1)>0) break;
        logger::debug("Main,run,innerLoop","END");
    }

    logger::log("Main,run,looping","END");
    cv::destroyAllWindows();
    logger::log("Main,run","END");
}


void visionCam()
{
    std::vector<cv::String> availableClasses = config::objectDetection::models::classes;
    std::vector<cv::Scalar> colors = config::objectDetection::models::colors;

    logger::log("Program,Retrieving config file","START");
    // YOLO config file
    cv::String cfgFile(config::objectDetection::models::configfile);
    logger::log("Program,Retrieving config file","END");

    logger::log("Program,Retrieving DNN model","START");
    // TRAINED YOLO MODEL
    cv::String model(config::objectDetection::models::modelfile);
    logger::log("Program,Retrieving DNN model","END");

    logger::log("Program,generatingNeuralNet","START");
    // darknet dnn object
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfgFile, model);
    logger::log("Program,generatingNeuralNet","END");

    logger::log("Program,SetComputationDevice,target=CUDA","START");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    logger::log("Program,SetComputationDevice,target=CUDA","END");

    float confidenceThresh = config::objectDetection::confidenceThreshold;

    // instantiation
    YoloClassifier ml(net, availableClasses, colors, confidenceThresh, config::objectDetection::models::inputSize);

    runVisionCam(ml);

}

void runVisionCam(YoloClassifier &classifier)
{

    logger::log("Main,run","START");
    int fps = 0, sumFps = 0;
    double avgFps = 0;
    
    rs2::pipeline pl;
    rs2::config customConfig;
    FrameSize frameSize;

    frameSize.colorWidth = config::camera::colorFrame.width;
    frameSize.colorHeight = config::camera::colorFrame.height;
    frameSize.depthWidth = config::camera::depthFrame.width;
    frameSize.depthHeight = config::camera::depthFrame.height;
    frameSize.thermalWidth = config::camera::thermalFrame.width;
    frameSize.thermalHeight = config::camera::thermalFrame.height;

    logger::info("Main,run,colorFrame,size=(w:"+std::to_string(config::camera::colorFrame.width)+",h:"+std::to_string(config::camera::colorFrame.height)+")","DONE");
    logger::info("Main,run,depthFrame,size=(w:"+std::to_string(config::camera::depthFrame.width)+",h:"+std::to_string(config::camera::depthFrame.height)+")","DONE");
    logger::info("Main,run,thermalFrame,size=(w:"+std::to_string(config::camera::thermalFrame.width)+",h:"+std::to_string(config::camera::thermalFrame.height)+")","DONE");

    flir::FLIR flir = flir::FLIR(frameSize.thermalWidth, frameSize.thermalHeight);
    tracker::enableGlobalTempUpdate(config::objectTracking::useGlobalTemperatureUpdate);

    customConfig.enable_stream(RS2_STREAM_COLOR, frameSize.colorWidth, frameSize.colorHeight, RS2_FORMAT_BGR8, 0);
    customConfig.enable_stream(RS2_STREAM_DEPTH, frameSize.depthWidth, frameSize.depthHeight, RS2_FORMAT_Z16, 0);
    rs2::frameset rsFrame;

    rs2::pipeline_profile selection = pl.start(customConfig);
    rs2::device selected_device = selection.get_device();
    rs2::depth_stereo_sensor depth_sensor = selected_device.first<rs2::depth_stereo_sensor>();
    depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

    cv::Mat frame;
    cv::String windowName("Camera Vision");
    cv::namedWindow(windowName, cv::WindowFlags::WINDOW_FULLSCREEN);
    cv::setWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_FULLSCREEN, cv::WindowFlags::WINDOW_FULLSCREEN);

    std::vector<std::vector<int>> thermalProjectedFrame = getCalibratedThermalValues(frameSize);

    argparser::KwargsMap mappedArgs;
    bool showThermal = mappedArgs.checkArgs("-t");
    bool showDepth = mappedArgs.checkArgs("-d");

    logger::log("Main,run,looping","START");
    while(1)
    {
        logger::debug("Main,run,innerLoop","START");
        auto t0 = std::chrono::high_resolution_clock::now();

        logger::debug("Main,run,innerLoop,waitForFrames","START");
        rsFrame = pl.wait_for_frames();
        logger::debug("Main,run,innerLoop,waitForFrames","END");

        logger::debug("Main,run,innerLoop,getColorFrame","START");
        rs2::frame colorFrame = rsFrame.get_color_frame();
        logger::debug("Main,run,innerLoop,getColorFrame","END");

        logger::debug("Main,run,innerLoop,getDepthFrame","START");
        rs2::depth_frame depthFrame = rsFrame.get_depth_frame();
        logger::debug("Main,run,innerLoop,getDepthFrame","END");

        logger::debug("Main,run,innerLoop,MatricesConversion,frameType=color","START");
        frame = cv::Mat(cv::Size(frameSize.colorWidth, frameSize.colorHeight), CV_8UC3, (void*) colorFrame.get_data(), cv::Mat::AUTO_STEP);
        logger::debug("Main,run,innerLoop,MatricesConversion,frameType=color","END");

        logger::debug("Main,run,innerLoop,detect","START");
        classifier.detect(frame);
        logger::debug("Main,run,innerLoop,detect","END");

        // ####### write your desired draws below (after classifier.detect) ####### 

        if(config::flir::showFrameBoundary)
        {
            cv::putText(frame, "Thermal Update Box", cv::Point(thermalProjectedFrame[0][0], thermalProjectedFrame[0][1]-15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(250,5,5), 1);
            cv::rectangle(frame, cv::Rect(thermalProjectedFrame[0][0], thermalProjectedFrame[0][1], thermalProjectedFrame[1][0], thermalProjectedFrame[1][1]), cv::Scalar(250,5,5), 2);
        }

        logger::debug("Main,run,innerLoop,draw","START");
        if(mappedArgs.checkArgs("-vc"))
        {
            switch(std::stoi(mappedArgs.getArgs("-vc")))
            {
                case (0): drawBoxes(classifier, frame); break;
                case (1): drawBoxesWithDepthROI(classifier, frame, depthFrame, frameSize); break;
                case (2): drawBoxesWithDepthROI_plusDepthColor(classifier, frame, depthFrame, frameSize); break;
                case (3): drawBoxesAndTrack(classifier, flir, frame, frameSize); break;
                case (4): drawBoxesAndTrackWithDepth(classifier, flir, frame, depthFrame, frameSize); break;
                case (5): drawBoxesAndTrackWithDepth_plusThermalColor(classifier, flir, frame, depthFrame, frameSize); break;
                case (6): drawTrackAndShowAll(classifier, flir, frame, depthFrame, frameSize, showThermal, showDepth); break;
                default: break;
            }
        }
        else
        {
            // ## chose default operation

            // drawBoxes(classifier, frame);
            // drawBoxesWithDepthROI(classifier, frame, depthFrame, frameSize);
            // drawBoxesWithDepthROI_plusDepthColor(classifier, frame, depthFrame, frameSize);
            // drawBoxesAndTrack(classifier, flir, frame, frameSize);
            // drawBoxesAndTrackWithDepth(classifier, flir, frame, depthFrame, frameSize);
            // drawBoxesAndTrackWithDepth_plusThermalColor(classifier, flir, frame, depthFrame, frameSize);
            drawTrackAndShowAll(classifier, flir, frame, depthFrame, frameSize, showThermal, showDepth);
        }
        logger::debug("Main,run,innerLoop,draw","END");

        auto t1 = std::chrono::high_resolution_clock::now();
        int fps = 1e9/(t1-t0).count();
        std::string fpsText = "FPS: "+std::to_string(fps);
        logger::debug("Main,run,innerLoop,frameRendering","START");
        
        if(false) 
        {
            for(int i = 0; i >= 0; i++)
            {
                if(i*config::objectTracking::movingThreshold > frameSize.colorWidth) break;
                cv::line(frame,
                    cv::Point(i*config::objectTracking::movingThreshold,0),
                    cv::Point(i*config::objectTracking::movingThreshold,frameSize.colorHeight),
                    cv::Scalar(250,250,250), 1);
            }
        }
        
        // cv::resize(frame, frame, classifier.targetSize, 0., 0., cv::INTER_CUBIC);
        cv::putText(frame, fpsText, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(5,5,240), 2);
        cv::putText(frame, "count: "+std::to_string(tracker::totalValidObject), cv::Point(20,frame.rows-80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(5,5,240), 2);
        cv::imshow(windowName, frame);
        logger::debug("Main,run,innerLoop,frameRendering","END");

        if(cv::waitKey(1)>0) break;
        logger::debug("Main,run,innerLoop","END");
    }

    logger::log("Main,run,looping","END");
    cv::destroyAllWindows();
    logger::log("Main,run","END");
}

void visionCamHelp()
{
    std::string datas = "", s;
    std::ifstream file;
    file.open("visioncam_help.txt");
    while(std::getline(file,s)) std::cout << s << std::endl;
    file.close();
    std::cout << datas << std::endl;
}

void yoloDetection()
{
    // available classes
    // std::vector<cv::String> availableClasses = {
    //     "mask",
    //     "no_mask",
    //     "not_visible",
    //     "misplaced"
    // };
    std::vector<cv::String> availableClasses = {
        "face",
        "face",
        "face",
        "face"
    };

    // detection colors
    // std::vector<cv::Scalar> colors = {
    //     cv::Scalar(5, 255, 5),
    //     cv::Scalar(5, 5, 255),
    //     cv::Scalar(20, 250, 250),
    //     cv::Scalar(20, 250, 250)
    // };
    std::vector<cv::Scalar> colors = {
        cv::Scalar(5, 255, 5),
        cv::Scalar(5, 255, 5),
        cv::Scalar(5, 255, 5),
        cv::Scalar(5, 255, 5)
    };


    logger::log("Program,Retrieving config file","START");
    // YOLO config file
    cv::String cfgFile("yolo/facemask-yolov4-tiny.cfg");
    logger::log("Program,Retrieving config file","END");

    logger::log("Program,Retrieving DNN model","START");
    // TRAINED YOLO MODEL
    cv::String model("yolo/facemask-yolov4-tiny_best.weights");
    logger::log("Program,Retrieving DNN model","END");

    logger::log("Program,generatingNeuralNet","START");
    // darknet dnn object
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfgFile, model);
    logger::log("Program,generatingNeuralNet","END");

    logger::log("Program,SetComputationDevice,target=CUDA","START");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    logger::log("Program,SetComputationDevice,target=CUDA","END");

    std::vector<int> inputSize = {512, 416, 320, 224, 128};

    float confidenceThresh = 0.2;

    // instantiation
    YoloClassifier ml(net, availableClasses, colors, confidenceThresh, inputSize[3]);

    run(ml);

}

void run(YoloClassifier &classifier)
{
    // cv::Mat thermalFrame;

    logger::log("Main,run","START");
    int fps = 0, sumFps = 0;
    double avgFps = 0;
    
    rs2::pipeline pl;
    rs2::config customConfig;
    FrameSize frameSize;

    // frameSize.colorWidth = 1920;
    // frameSize.colorHeight = 1080;
    frameSize.colorWidth = 1280;
    frameSize.colorHeight = 720;
    // frameSize.colorWidth = 640;
    // frameSize.colorHeight = 480;

    // frameSize.depthWidth = 1280;
    // frameSize.depthHeight = 720;
    frameSize.depthWidth = 640;
    frameSize.depthHeight = 480;

    // frameSize.thermalWidth = 640;
    // frameSize.thermalHeight = 480;
    frameSize.thermalWidth = 320;
    frameSize.thermalHeight = 240;

    flir::FLIR flir = flir::FLIR(frameSize.thermalWidth, frameSize.thermalHeight);
    tracker::enableGlobalTempUpdate(true);

    customConfig.enable_stream(RS2_STREAM_COLOR, frameSize.colorWidth, frameSize.colorHeight, RS2_FORMAT_BGR8, 0);
    customConfig.enable_stream(RS2_STREAM_DEPTH, frameSize.depthWidth, frameSize.depthHeight, RS2_FORMAT_Z16, 0);
    rs2::frameset rsFrame;

    rs2::pipeline_profile selection = pl.start(customConfig);
    rs2::device selected_device = selection.get_device();
    rs2::depth_stereo_sensor depth_sensor = selected_device.first<rs2::depth_stereo_sensor>();
    depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

    cv::Mat frame;
    cv::String windowName("Camera Vision");
    cv::namedWindow(windowName, cv::WindowFlags::WINDOW_FULLSCREEN);

    logger::log("Main,run,looping","START");
    while(1)
    {
        logger::debug("Main,run,innerLoop","START");
        auto t0 = std::chrono::high_resolution_clock::now();

        logger::debug("Main,run,innerLoop,waitForFrames","START");
        rsFrame = pl.wait_for_frames();
        logger::debug("Main,run,innerLoop,waitForFrames","END");

        logger::debug("Main,run,innerLoop,getColorFrame","START");
        rs2::frame colorFrame = rsFrame.get_color_frame();
        logger::debug("Main,run,innerLoop,getColorFrame","END");

        logger::debug("Main,run,innerLoop,getDepthFrame","START");
        rs2::depth_frame depthFrame = rsFrame.get_depth_frame();
        logger::debug("Main,run,innerLoop,getDepthFrame","END");

        logger::debug("Main,run,innerLoop,MatricesConversion,frameType=color","START");
        frame = cv::Mat(cv::Size(frameSize.colorWidth, frameSize.colorHeight), CV_8UC3, (void*) colorFrame.get_data(), cv::Mat::AUTO_STEP);
        logger::debug("Main,run,innerLoop,MatricesConversion,frameType=color","END");

        logger::debug("Main,run,innerLoop,detect","START");
        classifier.detect(frame);
        logger::debug("Main,run,innerLoop,detect","END");

        // cv::rectangle(frame, cv::Rect(flir::FLIR::thermalOffsetX, flir::FLIR::thermalOffsetY, flir::FLIR::calibratedWidth, flir::FLIR::calibratedHeight), cv::Scalar(250,5,5), 2);

        logger::debug("Main,run,innerLoop,draw","START");
        // drawBoxes(classifier, frame);
        // drawBoxesWithDepthROI(classifier, frame, depthFrame, frameSize);
        // drawBoxesWithDepthROI_plusDepthColor(classifier, frame, depthFrame, frameSize);
        // drawBoxesAndTrack(classifier, flir, frame, frameSize);
        drawBoxesAndTrackWithDepth(classifier, flir, frame, depthFrame, frameSize);
        // drawBoxesAndTrackWithDepth_plusThermalColor(classifier, flir, frame, depthFrame, frameSize);
        logger::debug("Main,run,innerLoop,draw","END");

        auto t1 = std::chrono::high_resolution_clock::now();
        int fps = 1e9/(t1-t0).count();
        std::string fpsText = "FPS: "+std::to_string(fps);

        logger::debug("Main,run,innerLoop,frameRendering","START");
        cv::putText(frame, fpsText, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(5,5,240), 2);
        cv::putText(frame, "count: "+std::to_string(tracker::totalValidObject), cv::Point(20,frame.rows-80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(5,5,240), 2);
        // cv::resize(frame, frame, classifier.targetSize, 0., 0., cv::INTER_NEAREST);
        cv::imshow(windowName, frame);
        logger::debug("Main,run,innerLoop,frameRendering","END");

        // thermalFrame = flir.getFrame();
        // cv::imshow("flir", thermalFrame);

        if(cv::waitKey(1)>0) break;
        logger::debug("Main,run,innerLoop","END");
    }

    logger::log("Main,run,looping","END");
    cv::destroyAllWindows();
    logger::log("Main,run","END");
}

bool checkRectangleCollision(YoloClassifier &classifier, int idx)
{
    logger::debug("Main,checkRectangleCollision","START");
    cv::Rect targetBox = classifier.postProcVar.boxes[idx];
    std::vector<cv::Point> corner = {
        cv::Point(targetBox.x, targetBox.y),
        cv::Point(targetBox.x+targetBox.width, targetBox.y),
        cv::Point(targetBox.x, targetBox.y+targetBox.height),
        cv::Point(targetBox.x+targetBox.width, targetBox.y+targetBox.height)
    };
    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        if(i!=idx)
        {
            cv::Rect box = classifier.postProcVar.boxes[i];
            for(cv::Point j : corner)
            {
                bool horizontalCollide = box.x < j.x && j.x < box.x+box.width;
                bool verticalCollide = box.y < j.y && j.y < box.y+box.height;
                if(horizontalCollide && verticalCollide)
                {
                    logger::debug("Main,checkRectangleCollision,value=true","END");
                    return true;
                }
            }
        }
    }
    logger::debug("Main,checkRectangleCollision,value=false","END");
    return false;
}

void drawBoxes(YoloClassifier &classifier, cv::Mat &frame)
{
    logger::debug("Main,drawBoxes","START");
    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        if(i>0)
        {
            if(checkRectangleCollision(classifier, i)) continue;
        }

        int idx = classifier.postProcVar.indices[i];
        cv::Rect box = classifier.postProcVar.boxes[i];

        cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

        std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
        std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;

        cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 15);
        cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
        cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
    }
    logger::debug("Main,drawBoxes","END");
}

void drawBoxesWithDepth(YoloClassifier &classifier, cv::Mat &frame, rs2::depth_frame depthFrame)
{
    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        if(i>0)
        {
            if(checkRectangleCollision(classifier, i)) continue;
        }

        int idx = classifier.postProcVar.indices[i];
        cv::Rect box = classifier.postProcVar.boxes[i];

        cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

        std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
        std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;

        float distance = depthFrame.get_distance(classifier.postProcVar.centerPoints[i].x, classifier.postProcVar.centerPoints[i].y);
        int distanceInCm = (int)(distance*100);
        std::string distanceText = "Dist: "+std::to_string(distanceInCm)+" cm";

        cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 30);
        cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
        cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
        cv::putText(frame, distanceText, cv::Point(box.x+2, box.y+box.height+17), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
    }
}

float getValidDistanceInBoxArea(rs2::depth_frame depthFrame, cv::Point centerPoint, int xRange, int yRange)
{
    logger::debug("Main,getValidDistanceInBoxArea","START");
    int w = depthFrame.get_width();
    int h = depthFrame.get_height();
    std::vector<float> allDist;
    float maxDist = 0.f;

    if((0 < centerPoint.x-xRange && centerPoint.x+xRange < w ) && (0 < centerPoint.y-yRange && centerPoint.y+yRange < h))
    {
        int x0 = centerPoint.x-xRange;
        int y0 = centerPoint.y-yRange;
        for(int x = 0; x < xRange*2; x++)
        {
            for(int y = 0; y < yRange*2; y++)
            {
                float d = depthFrame.get_distance(x+x0, y+y0);
                allDist.push_back(d);
            }
        }

        for(float dist : allDist)
        {
            if(dist>maxDist) maxDist=dist;
        }
    }
    logger::debug("Main,getValidDistanceInBoxArea,maxValue="+std::to_string(maxDist),"END");
    return maxDist;
}

Scales getCalibratedDepthScale(FrameSize &fsize)
{
    logger::debug("Main,getValidDistanceInBoxArea","START");

    /*
    // calibrasi manual (color 1920-1080, depth 1280-720)
    double scaleX = 0.336*2;
    double scaleY = 0.354*2;

    // calibrasi manual (color 1920-1080, depth 640-480)
    double scaleX = 0.15625*2;
    double scaleY = 0.3125*2;

    // calibrasi manual (color 640-480, depth 1280-720)
    double scaleX = 0.345*2;
    double scaleY = 0.257*2;
    
    // calibrasi manual (color 640-480, depth 640-480)
    double scaleX = 0.3125*2;
    double scaleY = 0.3125*2;
    */

    if(fsize.colorWidth == 1920 && fsize.colorHeight == 1080)
    {
        if(fsize.depthWidth == 1280 && fsize.depthHeight == 720)
            return Scales(0.336*2, 0.354*2);
        else if(fsize.depthWidth == 640 && fsize.depthHeight == 480)
            return Scales(0.42*2, 0.31215*2);
    }
    else if(fsize.colorWidth == 1280 && fsize.colorHeight == 720)
    {
        if(fsize.depthWidth == 1280 && fsize.depthHeight == 720)
            return Scales(0.348*2, 0.34*2);
        else if(fsize.depthWidth == 640 && fsize.depthHeight == 480)
            return Scales(0.422*2, 0.3125*2);
    }
    else if(fsize.colorWidth == 640 && fsize.colorHeight == 480)
    {
        if(fsize.depthWidth == 1280 && fsize.depthHeight == 720)
            return Scales(0.258*2, 0.345*2);
        else if(fsize.depthWidth == 640 && fsize.depthHeight == 480)
            return Scales(0.3125*2, 0.3125*2);
    }
    return Scales(0.36*2, 0.36*2);
}

std::vector<std::vector<int>> getCalibratedThermalValues(FrameSize &fsize)
{
    // format [[offsetX, offsetY], [calWidth, calHeight]]
    if(fsize.colorWidth == 1920 && fsize.colorHeight == 1080)
    {
        // return std::vector<std::vector<int>>({std::vector<int>({300, 175}), std::vector<int>({1440, 1050})});
        return std::vector<std::vector<int>>({std::vector<int>({350, 120}), std::vector<int>({1435, 960})});
    }
    else if(fsize.colorWidth == 1280 && fsize.colorHeight == 720)
    {
        // return std::vector<std::vector<int>>({std::vector<int>({185, 75}), std::vector<int>({965, 705})});
        return std::vector<std::vector<int>>({std::vector<int>({225, 75}), std::vector<int>({970, 690})});
    }
    else if(fsize.colorWidth == 640 && fsize.colorHeight == 480)
    {
        // return std::vector<std::vector<int>>({std::vector<int>({185, 75}), std::vector<int>({965, 705})});
        return std::vector<std::vector<int>>({std::vector<int>({225, 75}), std::vector<int>({970, 690})});
    }
    logger::debug("Main,getCalibratedThermalValues,ColorFrame size is not registered","EXITTING");
    return std::vector<std::vector<int>>({});
}

cv::Point convertColorToThermal(int x, int y, FrameSize &fsize)
{
    std::vector<std::vector<int>> calThermal = getCalibratedThermalValues(fsize);
    int thermalX = (int)(((x-calThermal[0][0])*1./calThermal[1][0])*fsize.thermalWidth);
    int thermalY = (int)(((y-calThermal[0][1])*1./calThermal[1][1])*fsize.thermalHeight);
    return cv::Point(thermalX, thermalY);
}

cv::Rect convertColorToThermal(flir::FLIR flir, cv::Rect box, FrameSize &fsize)
{
    std::vector<std::vector<int>> calThermal = getCalibratedThermalValues(fsize);
    int thermalX = (int)(((box.x-calThermal[0][0])*1./calThermal[1][0])*fsize.thermalWidth) + flir.xmin;
    int thermalY = (int)(((box.y-calThermal[0][1])*1./calThermal[1][1])*fsize.thermalHeight) + flir.ymin;
    // int thermalWidth = (int)(box.width*fsize.thermalWidth/fsize.colorWidth);
    int thermalHeight = (int)(box.height*fsize.thermalHeight/fsize.colorHeight);
    int thermalWidth = thermalHeight;
    return cv::Rect(thermalX, thermalY, thermalWidth, thermalHeight);
}

cv::Point convertThermalToDepth(flir::FLIR flir, cv::Point thermalCenter, FrameSize fsize, bool &inRange)
{
    logger::debug("Main,convertThermalToDepth","START");
    std::vector<std::vector<int>> calThermal = getCalibratedThermalValues(fsize);
    int colorX = (((thermalCenter.x - flir.xmin)*1. / fsize.thermalWidth)*1.*calThermal[1][0])+calThermal[0][0];
    int colorY = (((thermalCenter.y - flir.ymin)*1. / fsize.thermalHeight)*1.*calThermal[1][1])+calThermal[0][1];

    Scales scale = getCalibratedDepthScale(fsize);
    
    int offsetX = ((fsize.depthWidth-(fsize.depthWidth*scale.x))/2);
    int offsetY = ((fsize.depthHeight-(fsize.depthHeight*scale.y))/2);

    int depthX = (int)((colorX*1.) / (1.*fsize.colorWidth) * (fsize.depthWidth*scale.x)) + offsetX;
    int depthY = (int)((colorY*1.) / (1.*fsize.colorHeight) * (fsize.depthHeight*scale.y)) + offsetY;
    inRange = (offsetX < depthX && depthX < fsize.depthWidth-offsetX) || (offsetY < depthY && depthY < fsize.depthHeight-offsetY);

    std::stringstream ss; std::string s;
    ss << "Main,convertThermalToDepth,depth=(x:" << depthX << ",y:" << depthY << "),inRange=" << inRange?"true":"false" ; ss >> s;
    logger::debug(s,"END");
    return cv::Point(depthX, depthY);
}

void drawBoxesWithDepthROI(YoloClassifier &classifier, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize)
{
    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        if(i>0)
        {
            if(checkRectangleCollision(classifier, i)) continue;
        }

        int idx = classifier.postProcVar.indices[i];
        cv::Rect box = classifier.postProcVar.boxes[i];

        cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

        std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
        std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;

        Scales scale = getCalibratedDepthScale(fsize);

        int offsetX = ((fsize.depthWidth-(fsize.depthWidth*scale.x))/2);
        int offsetY = ((fsize.depthHeight-(fsize.depthHeight*scale.y))/2);

        int depthX = (int)((classifier.postProcVar.centerPoints[i].x*1.) / (1.*fsize.colorWidth) * (fsize.depthWidth*scale.x)) + offsetX;
        int depthY = (int)((classifier.postProcVar.centerPoints[i].y*1.) / (1.*fsize.colorHeight) * (fsize.depthHeight*scale.y)) + offsetY;

        if((offsetX < depthX && depthX < fsize.depthWidth-offsetX) || (offsetY < depthY && depthY < fsize.depthHeight-offsetY))
        {
            // float distance = 123.f;
            // float distance = depthFrame.get_distance(depthX, depthY);
            // float distance = depthFrame.get_distance(depthX, depthY+10);
            float distance = getValidDistanceInBoxArea(depthFrame, cv::Point(depthX, depthY+10), 5, 5);
            int distanceInCm = (int)(distance*100);
            std::string distanceText = "Dist: "+std::to_string(distanceInCm)+" cm";
            cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 30);
            cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
            cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
            cv::putText(frame, distanceText, cv::Point(box.x+2, box.y+box.height+17), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
        }
        else
        {
            cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 15);
            cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
            cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
        }
    }
}

void drawBoxesWithDepthROI_plusDepthColor(YoloClassifier &classifier, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize)
{
    rs2::colorizer colormap;
    rs2::video_frame vframe = colormap.colorize(depthFrame);
    cv::Mat cdFrame = cv::Mat(cv::Size(fsize.depthWidth, fsize.depthHeight), CV_8UC3, (void*) vframe.get_data(), cv::Mat::AUTO_STEP);

    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        if(i>0)
        {
            if(checkRectangleCollision(classifier, i)) continue;
        }
            
        int idx = classifier.postProcVar.indices[i];
        cv::Rect box = classifier.postProcVar.boxes[i];

        cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

        std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
        std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;

        Scales scale = getCalibratedDepthScale(fsize);

        int offsetX = ((fsize.depthWidth-(fsize.depthWidth*scale.x))/2);
        int offsetY = ((fsize.depthHeight-(fsize.depthHeight*scale.y))/2);

        int depthX = (int)((classifier.postProcVar.centerPoints[i].x*1.) / (1.*fsize.colorWidth) * (fsize.depthWidth*scale.x)) + offsetX;
        int depthY = (int)((classifier.postProcVar.centerPoints[i].y*1.) / (1.*fsize.colorHeight) * (fsize.depthHeight*scale.y)) + offsetY;

        cv::circle(cdFrame, cv::Point(depthX, depthY+10), 10, cv::Scalar(2,2,255), -1);
        cv::circle(cdFrame, cv::Point(depthX, depthY+10), 30, cv::Scalar(2,2,255), 3);

        if((offsetX < depthX && depthX < fsize.depthWidth-offsetX) || (offsetY < depthY && depthY < fsize.depthHeight-offsetY))
        {
            // float distance = depthFrame.get_distance(depthX, depthY);
            // float distance = depthFrame.get_distance(depthX, depthY+10);
            float distance = getValidDistanceInBoxArea(depthFrame, cv::Point(depthX, depthY+10), 5, 5);
            int distanceInCm = (int)(distance*100);
            std::string distanceText = "Dist: "+std::to_string(distanceInCm)+" cm";
            cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 30);
            cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
            cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
            cv::putText(frame, distanceText, cv::Point(box.x+2, box.y+box.height+17), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
            // std::string boxCenterText = "x: "+std::to_string(depthX)+", y: "+std::to_string(depthY);
            // cv::putText(frame, boxCenterText, cv::Point(box.x, box.y+box.height-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(2,255,2), 2);
        }
        else
        {
            cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 15);
            cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
            cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
        }
    }
    
    cv::imshow("colored",cdFrame);
}

void drawBoxesAndTrack(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, FrameSize &fsize)
{
    tracker::refreshAll();
    cv::Mat thermalFrame = flir.getFrame();
    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {   
        tracker::TrackedObject *obj = trackBoxes(classifier, i, flir, frame);

        if(!obj->drawn)
        {
            obj->drawn = true;
            int idx = classifier.postProcVar.indices[i];
            cv::Rect box = classifier.postProcVar.boxes[i];

            cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

            std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
            std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;

            cv::Rect textBackground = cv::Rect(box.x, box.y+box.height-10, 11.f*classifyText.size(), 15);
            cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
            cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);

            // temperature show
            cv::Rect thermalROI = convertColorToThermal(flir, classifier.postProcVar.boxes[i], fsize);
            obj->updateTemp(thermalFrame, thermalROI);

            std::string tempText; std::stringstream t;
            t << std::setprecision(4) << obj->temp; t >> tempText;
            std::string objectText = "ID: "+std::to_string(obj->id)+" | Temp: "+tempText+" C";
            cv::putText(frame, objectText, cv::Point(box.x+5, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(25,255,5), 2);
        }
    }
}

void drawBoxesAndTrackWithDepth(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize)
{
    tracker::refreshAll();
    bool needTempUpdate = tracker::checkTempUpdateAll()>=0? true : false;
    cv::Mat thermalFrame;
    if(needTempUpdate) thermalFrame = flir.getFrame();

    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        tracker::TrackedObject *obj = trackBoxes(classifier, i, flir, frame);

        if(!obj->drawn)
        {
            obj->drawn = true;
            
            // drawing boxes
            logger::debug("Main,drawBoxesAndTrackWithDepth,drawBoxes,id="+std::to_string(obj->id),"START");
            
            int idx = classifier.postProcVar.indices[i];
            cv::Rect box = classifier.postProcVar.boxes[i];

            cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

            // properties write
            std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
            std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;
            logger::debug("Main,drawBoxesAndTrackWithDepth,drawBoxes,id="+std::to_string(obj->id),"END");

            // object ID
            cv::putText(frame, "ID: "+std::to_string(obj->id), cv::Point(box.x+5, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(25,255,5), 2);

            // temperature show
            logger::debug("Main,drawBoxesAndTrackWithDepth,processingTemp,id="+std::to_string(obj->id),"START");
            if(needTempUpdate)
            {
                cv::Rect thermalROI = convertColorToThermal(flir, classifier.postProcVar.boxes[i], fsize);
                obj->updateTemp(thermalFrame, thermalROI);
            }

            std::string tempText;
            if(obj->temp>0)
            {
                std::string tt; std::stringstream t;
                t << std::setprecision(4) << obj->temp; t >> tt;
                tempText = "Temp: "+tt+" C";
            }
            else tempText = "Temp: retrieving..";

            logger::debug("Main,drawBoxesAndTrackWithDepth,processingTemp,id="+std::to_string(obj->id),"END");

            // distance estimation
            logger::debug("Main,drawBoxesAndTrackWithDepth,distanceEstimation,id="+std::to_string(obj->id),"START");

            Scales depthScale = getCalibratedDepthScale(fsize);

            int depthOffsetX = ((fsize.depthWidth-(fsize.depthWidth*depthScale.x))/2);
            int depthOffsetY = ((fsize.depthHeight-(fsize.depthHeight*depthScale.y))/2);

            int depthX = (int)((classifier.postProcVar.centerPoints[i].x*1.) / (1.*fsize.colorWidth) * (fsize.depthWidth*depthScale.x)) + depthOffsetX;
            int depthY = (int)((classifier.postProcVar.centerPoints[i].y*1.) / (1.*fsize.colorHeight) * (fsize.depthHeight*depthScale.y)) + depthOffsetY;

            if((depthOffsetX < depthX && depthX < fsize.depthWidth-depthOffsetX) || (depthOffsetY < depthY && depthY < fsize.depthHeight-depthOffsetY))
            {
                float distance = getValidDistanceInBoxArea(depthFrame, cv::Point(depthX, depthY+10), 5, 5);
                int distanceInCm = (int)(distance*100);
                std::string distanceText = "Dist: "+std::to_string(distanceInCm)+" cm";
                cv::Rect textBackground = cv::Rect(box.x, box.y+box.height, 10.f*(tempText.size()), 45);
                cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
                cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, distanceText, cv::Point(box.x+2, box.y+box.height+27), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, tempText, cv::Point(box.x+2, box.y+box.height+42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }
            else
            {
                cv::Rect textBackground = cv::Rect(box.x, box.y+box.height, 10.f*tempText.size(), 30);
                cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
                cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, tempText, cv::Point(box.x+2, box.y+box.height+27), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }

            logger::debug("Main,drawBoxesAndTrackWithDepth,distanceEstimation,id="+std::to_string(obj->id),"END");
        }
    }
}

void drawBoxesAndTrackWithDepth_plusThermalColor(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize)
{
    tracker::refreshAll();
    cv::Mat thermalFrame = flir.getFrame();
    cv::Mat colored = flir.colorize(thermalFrame);
    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        tracker::TrackedObject *obj = trackBoxes(classifier, i, flir, frame);

        if(!obj->drawn)
        {
            obj->drawn = true;

            int idx = classifier.postProcVar.indices[i];
            cv::Rect box = classifier.postProcVar.boxes[i];

            cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

            std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
            std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;
            
            // temperature show
            cv::Rect thermalROI = convertColorToThermal(flir, classifier.postProcVar.boxes[i], fsize);
            obj->updateTemp(thermalFrame, thermalROI);
            cv::rectangle(colored, thermalROI, cv::Scalar(0,5,255),1);

            std::string objectText = "ID: "+std::to_string(obj->id);
            cv::putText(frame, objectText, cv::Point(box.x+5, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(25,255,5), 1);

            std::string tempText; int tempTextLen = 150;
            if(obj->temp>0)
            {
                std::string tt; std::stringstream t;
                t << std::setprecision(4) << obj->temp; t >> tt;
                tempText = "Temp: "+tt+" C";
                tempTextLen = 130;
            }
            else tempText = "Temp: retrieving..";

            // distance estimation
            Scales depthScale = getCalibratedDepthScale(fsize);

            int depthOffsetX = ((fsize.depthWidth-(fsize.depthWidth*depthScale.x))/2);
            int depthOffsetY = ((fsize.depthHeight-(fsize.depthHeight*depthScale.y))/2);

            int depthX = (int)((classifier.postProcVar.centerPoints[i].x*1.) / (1.*fsize.colorWidth) * (fsize.depthWidth*depthScale.x)) + depthOffsetX;
            int depthY = (int)((classifier.postProcVar.centerPoints[i].y*1.) / (1.*fsize.colorHeight) * (fsize.depthHeight*depthScale.y)) + depthOffsetY;

            if((depthOffsetX < depthX && depthX < fsize.depthWidth-depthOffsetX) || (depthOffsetY < depthY && depthY < fsize.depthHeight-depthOffsetY))
            {
                float distance = getValidDistanceInBoxArea(depthFrame, cv::Point(depthX, depthY+10), 5, 5);
                int distanceInCm = (int)(distance*100);
                std::string distanceText = "Dist: "+std::to_string(distanceInCm)+" cm";
                cv::Rect textBackground = cv::Rect(box.x, box.y+box.height, 10.f*tempText.size(), 45);
                cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
                cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, distanceText, cv::Point(box.x+2, box.y+box.height+27), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, tempText, cv::Point(box.x+2, box.y+box.height+42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }
            else
            {
                cv::Rect textBackground = cv::Rect(box.x, box.y+box.height, 10.f*tempText.size(), 30);
                cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
                cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, tempText, cv::Point(box.x+2, box.y+box.height+42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }
        }
    }
    cv::imshow("Thermal",colored);
}

void drawTrackAndShowAll(YoloClassifier &classifier, flir::FLIR &flir, cv::Mat &frame, rs2::depth_frame depthFrame, FrameSize &fsize, bool showThermal, bool showDepth)
{
    tracker::refreshAll();
    cv::Mat thermalFrame = flir.getFrame();
    cv::Mat ctFrame;
    if(showThermal) ctFrame = flir.colorize(thermalFrame);

    cv::Mat cdFrame;
    if(showDepth)
    {
        rs2::colorizer colormap;
        rs2::video_frame vframe = colormap.colorize(depthFrame);
        cdFrame = cv::Mat(cv::Size(fsize.depthWidth, fsize.depthHeight), CV_8UC3, (void*) vframe.get_data(), cv::Mat::AUTO_STEP);
    }

    for(size_t i=0;i<classifier.postProcVar.indices.size();i++)
    {
        tracker::TrackedObject *obj = trackBoxes(classifier, i, flir, frame);

        if(!obj->drawn)
        {
            obj->drawn = true;

            int idx = classifier.postProcVar.indices[i];
            cv::Rect box = classifier.postProcVar.boxes[i];

            cv::rectangle(frame, box, classifier.detectionColors[classifier.postProcVar.classIds[idx]], 2);

            std::string percentConfidence = std::to_string((int)(classifier.postProcVar.confidences[idx]*100))+'%';
            std::string classifyText = classifier.availableClasses[classifier.postProcVar.classIds[idx]]+": "+percentConfidence;
            
            // temperature show
            cv::Rect thermalROI = convertColorToThermal(flir, classifier.postProcVar.boxes[i], fsize);
            obj->updateTemp(thermalFrame, thermalROI);
            if(showThermal) cv::rectangle(ctFrame, thermalROI, cv::Scalar(255,255,255),2);

            std::string objectText = "ID: "+std::to_string(obj->id);
            cv::putText(frame, objectText, cv::Point(box.x+5, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(25,255,5), 1);

            std::string tempText = "Temp: retrieving..";
            int tempTextLen = 150;

            // distance estimation
            Scales depthScale = getCalibratedDepthScale(fsize);

            int depthOffsetX = ((fsize.depthWidth-(fsize.depthWidth*depthScale.x))/2);
            int depthOffsetY = ((fsize.depthHeight-(fsize.depthHeight*depthScale.y))/2);

            int depthX = (int)((classifier.postProcVar.centerPoints[i].x*1.) / (1.*fsize.colorWidth) * (fsize.depthWidth*depthScale.x)) + depthOffsetX;
            int depthY = (int)((classifier.postProcVar.centerPoints[i].y*1.) / (1.*fsize.colorHeight) * (fsize.depthHeight*depthScale.y)) + depthOffsetY;

            if((depthOffsetX < depthX && depthX < fsize.depthWidth-depthOffsetX) || (depthOffsetY < depthY && depthY < fsize.depthHeight-depthOffsetY))
            {
                float distance = getValidDistanceInBoxArea(depthFrame, cv::Point(depthX, depthY+10), 5, 5);
                int distanceInCm = (int)(distance*100);

                // calibrate temp
                if(obj->temp>0)
                {
                    std::string tt; std::stringstream t;
                    obj->temp = obj->tempUpdated==true?flir.calibrateTemp(obj->temp, distanceInCm*1.):obj->temp;
                    t << std::setprecision(4) << obj->temp; t >> tt;
                    tempText = "Temp: "+tt+" C";
                    tempTextLen = 130;
                }

                std::string distanceText = "Dist: "+std::to_string(distanceInCm)+" cm";
                cv::Rect textBackground = cv::Rect(box.x, box.y+box.height, tempTextLen, 45);
                cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
                cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, distanceText, cv::Point(box.x+2, box.y+box.height+27), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, tempText, cv::Point(box.x+2, box.y+box.height+42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);

            }
            else
            {
                // calibrate temp
                if(obj->temp>0)
                {
                    std::string tt; std::stringstream t;
                    t << std::setprecision(4) << obj->temp; t >> tt;
                    tempText = "Temp: "+tt+" C";
                    tempTextLen = 130;
                }

                cv::Rect textBackground = cv::Rect(box.x, box.y+box.height, tempTextLen, 30);
                cv::rectangle(frame, textBackground, cv::Scalar(255,255,255), -1);
                cv::putText(frame, classifyText, cv::Point(box.x+2, box.y+box.height+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
                cv::putText(frame, tempText, cv::Point(box.x+2, box.y+box.height+42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }

            // draw distance
            cv::Rect depthBox(depthX-(box.width*depthScale.x/4), depthY-(box.height*depthScale.y/4), box.width*depthScale.x/2, box.height*depthScale.y/2);
            if(showDepth) cv::rectangle(cdFrame, depthBox, cv::Scalar(255,255,255), 2);
        }
    }
    if(showDepth) cv::imshow("Stereo (LIDAR)",cdFrame);
    if(showThermal) cv::imshow("Thermal",ctFrame);
}

tracker::TrackedObject *trackBoxes(YoloClassifier &classifier, int idx, flir::FLIR &flir, cv::Mat &frame)
{
    cv::Point center = classifier.postProcVar.centerPoints[idx];
    
    // check in the last frameThresh, is there any points around the moving thresh?
    int objectId = tracker::verifyAll(center);
    if(objectId >= 0)
    {
        logger::debug("Tracker,verify,id="+std::to_string(objectId)+",count="+std::to_string(tracker::objectId),"valid");
        return &tracker::trackedList[objectId];
    }
    else
    {
        // register the object
        logger::debug("Tracker,verify,id="+std::to_string(objectId)+",count="+std::to_string(tracker::objectId),"invalid");
        // tracker::TrackedObject *obj = new tracker::TrackedObject(center, flir);
        // tracker::addTracker(*obj);
        // return obj;
        tracker::TrackedObject *obj = tracker::createTracker(center, flir);
        return obj;
    }
}

