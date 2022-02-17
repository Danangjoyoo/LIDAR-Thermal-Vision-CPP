#include </usr/local/cuda/include/cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <ctime>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <opencv2/core/cuda.hpp>
#include <librealsense2/rs.hpp>
#include <logger.h>

class PureCenter
{
    public:
        float x, y;
        PureCenter(float x, float y){this->x=x;this->y=y;}
};

struct PostProcessVariable
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point> centerPoints;
    std::vector<int> indices;
    std::vector<PureCenter> pureCenter;
};

class YoloClassifier
{
    public:
        cv::dnn::Net *net;
        std::vector<std::string> availableClasses;
        std::vector<cv::Scalar> detectionColors;
        std::vector<cv::String> outputLayers;
        PostProcessVariable postProcVar;

        float confidenceThresh;
        int targetInputSizeIdx, targetInputSize;
        bool GPUInitialized = false;
        cv::Size targetSize = cv::Size(1920, 1080);
        std::vector<cv::Point> objectCenterPoints;

        YoloClassifier(
            cv::dnn::Net &net, 
            std::vector<std::string> availableClasses,
            std::vector<cv::Scalar> detectionColors,
            float confidenceThresh, int targetInputSize
            )
        {
            logger::log("YoloClassifier,Instantiation","START");
            this->net = &net;
            this->availableClasses = availableClasses;
            this->detectionColors = detectionColors;
            this->confidenceThresh = confidenceThresh;
            this->outputLayers = this->getOutLayerNames(*this->net);
            this->targetInputSize = targetInputSize;
            // this->targetInputSizeIdx = targetInputSizeIdx;
            logger::info("YoloClassifier,Instantiation,property","totalClasses="+std::to_string(availableClasses.size()));
            logger::info("YoloClassifier,Instantiation,property","confidenceThresh="+std::to_string(confidenceThresh));
            logger::info("YoloClassifier,Instantiation,property","inputSize="+std::to_string(targetInputSize));
            logger::log("YoloClassifier,Instantiation","END");
        }

        void detect(cv::Mat &frame)
        {
            logger::debug("YoloClassifier,detect","START");
            if(!this->GPUInitialized) logger::log("YoloClassifier,detect,GPUInitialization","START");

            cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255., cv::Size(this->targetInputSize, this->targetInputSize), cv::Scalar(), true, false);
            
            // cv::Mat clone;
            // cv::resize(frame, clone, cv::Size(320,240));
            // cv::Mat blob = cv::dnn::blobFromImage(clone, 1/255., cv::Size(this->targetInputSize, this->targetInputSize), cv::Scalar(), true, false);
            
            this->net->setInput(blob);

            std::vector<cv::Mat> outputs;
            this->net->forward(outputs, this->outputLayers);
            if(!this->GPUInitialized)
            {
                this->GPUInitialized = true;
                logger::log("YoloClassifier,detect,GPUInitialization","END");
            }
            this->postProcess(frame, outputs);
            logger::debug("YoloClassifier,detect","END");
        }

        std::vector<std::vector<float>> verticalStack(std::vector<cv::Mat> inBlobs)
        {
            logger::debug("YoloClassifier,detect,verticalStack","START");
            cv::Mat scores = inBlobs[0];
            cv::Size scoreSize = scores.size();

            cv::Mat geometry = inBlobs[1];
            cv::Size geoSize = geometry.size();
            
            std::vector<std::vector<float>> results;
            for(int i = 0; i < scoreSize.height+geoSize.height; i++)
            {
                std::vector<float> res;
                for(int ii = 0; ii < scoreSize.width; ii++)
                {
                    if(i < scoreSize.height)
                        res.push_back(scores.at<float>(i, ii));
                    else
                        res.push_back(geometry.at<float>(i+scoreSize.height, ii));
                }
                results.push_back(res);

                for(int ii = 0; ii < scoreSize.width; ii++)
                {
                    if(ii>4)
                    {
                        if(i < scoreSize.height)
                            std::cout << std::to_string(scores.at<float>(i, ii))+", ";
                        else
                            std::cout << std::to_string(geometry.at<float>(i+scoreSize.height, ii))+", ";
                    }
                }
                std::cout << std::endl;
            }
            logger::debug("YoloClassifier,detect,verticalStack","END");
            return results;
        }

        std::vector<cv::String> getOutLayerNames(cv::dnn::Net &net)
        {
            std::vector<cv::String> outLayerNames;
            std::vector<cv::String> layerNames = net.getLayerNames();
            std::vector<int> outLayers = net.getUnconnectedOutLayers();
            for(int i=0;i<outLayers.size();i++)
                outLayerNames.push_back(layerNames[outLayers[i] - 1]);
            return outLayerNames;
        }

        void postProcess(cv::Mat &frame, std::vector<cv::Mat> outputs)
        {
            logger::debug("YoloClassifier,detect,postProcess","START");
            PostProcessVariable newVar;
            this->postProcVar = newVar;

            for(size_t i = 0; i < outputs.size(); i++)
            {
                float *data = (float*)outputs[i].data;
                for(int ii = 0; ii < outputs[i].rows; ii++, data += outputs[i].cols)
                {
                    cv::Mat scores = outputs[i].row(ii).colRange(5, outputs[i].cols);
                    cv::Point classIdPoint;
                    double confidence;
                    cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                    if(confidence > this->confidenceThresh)
                    {
                        int centerX = (int)(data[0]*frame.cols);
                        int centerY = (int)(data[1]*frame.rows);
                        int width = (int)(data[2]*frame.cols);
                        int height = (int)(data[3]*frame.rows);
                        int topLeftX = centerX - width/2;
                        int topLeftY = centerY - height/2;

                        this->postProcVar.classIds.push_back(classIdPoint.x);
                        this->postProcVar.confidences.push_back((float)confidence);
                        this->postProcVar.boxes.push_back(cv::Rect(topLeftX, topLeftY, width, height));
                        this->postProcVar.centerPoints.push_back(cv::Point(centerX, centerY));
                        this->postProcVar.pureCenter.push_back(PureCenter(data[0], data[1]));
                    }
                }
            }

            cv::dnn::NMSBoxes(this->postProcVar.boxes, this->postProcVar.confidences, this->confidenceThresh, this->confidenceThresh*0.8f, this->postProcVar.indices);
            logger::debug("YoloClassifier,detect,postProcess","END");
        }
};