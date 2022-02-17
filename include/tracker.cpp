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
#include <logger.h>
#include <flir.h>
#include <yolo.h>
#include <config.h>


namespace tracker
{
    int objectId = 0;
    int totalValidObject = 0;
    int globalTempCount = 0;
    bool globalUpdateTemp = false;
    bool globalNeedUpdate = false;
    int globalTempResetCount = config::objectTracking::globalTemperatureResetCount;

    void enableGlobalTempUpdate(bool enable)
    {
        tracker::globalUpdateTemp = enable;
        if(enable) logger::log("Tracker,GlobalTempUpdate,enable=true","DONE");
        else logger::log("Tracker,GlobalTempUpdate,enable=false","DONE");
    }

    class TrackedObject
    {
        public:
            int moveThresh = config::objectTracking::movingThreshold;
            int registerThresh = config::objectTracking::registerThreshold;
            int deleteThresh = config::objectTracking::deleteThreshold;
            int tempResetCount = config::objectTracking::temperatureResetCount;
            int tempScanArea = config::objectTracking::temperatureScanArea;
            int updateTempAttempt = config::objectTracking::updateTemperatureAttempt;

            bool isAlive = true, showTemp=true, tempUpdated=false;
            double temp = 0.;
            int timeout, id, tempCount = 0, tempUpdateCount = 0;
            cv::Point lastCenter;
            flir::FLIR *flir;

            bool drawn = false, valid = false;
            int validCount = 0;

            TrackedObject(cv::Point center, flir::FLIR &flir)
            {
                logger::debug("Tracker,Constructor,create,id="+std::to_string(tracker::objectId),"START");
                this->id = tracker::objectId;
                this->timeout = this->deleteThresh;
                this->lastCenter = center;
                this->flir = &flir;;
                tracker::objectId++;
                logger::debug("Tracker,Constructor,create,id="+std::to_string(this->id),"SUCCEED");
            }

            void refresh()
            {
                if(this->isAlive)
                {
                    if(!tracker::globalUpdateTemp) this->tempCount++;
                    this->timeout--;
                    this->drawn = false;
                    if(this->timeout <= 0)
                    {
                        this->isAlive = false;
                        logger::debug("Tracker,TrackedObject,refresh,timeout reached,id="+std::to_string(this->id),"object deleted");
                    }
                    if(!this->valid)
                    {
                        if(this->validCount>=this->registerThresh)
                        {
                            this->valid = true;
                            tracker::totalValidObject++;
                            logger::debug("Tracker,TrackedObject,refresh,validation reached,id="+std::to_string(this->id),"object counted");
                        }
                    }
                    if(this->tempUpdateCount>this->updateTempAttempt && this->showTemp) this->showTemp = false;
                }
            }

            bool verify(cv::Point center)
            {
                bool xValid = this->lastCenter.x-this->moveThresh <= center.x && center.x <= this->lastCenter.x+this->moveThresh;
                bool yValid = this->lastCenter.y-this->moveThresh <= center.y && center.y <= this->lastCenter.y+this->moveThresh;

                if(xValid && yValid)
                {
                    logger::debug("Tracker,TrackedObject,verify,id="+std::to_string(this->id)+",lastCenter(x:"+std::to_string(this->lastCenter.x)+" y:"+std::to_string(this->lastCenter.y)+"), center(x:"+std::to_string(center.x)+" y:"+std::to_string(center.y)+")","valid");
                    this->validCount++;
                    this->lastCenter = center;
                    this->timeout = this->deleteThresh;
                    logger::debug("Tracker,TrackedObject,verify,id="+std::to_string(this->id)+",resetTimeout,value="+std::to_string(this->timeout),"DONE");
                    return true;
                }
                logger::debug("Tracker,TrackedObject,verify,id="+std::to_string(this->id)+",lastCenter(x:"+std::to_string(this->lastCenter.x)+" y:"+std::to_string(this->lastCenter.y)+"), center(x:"+std::to_string(center.x)+" y:"+std::to_string(center.y)+")","invalid");
                return false;
            }

            bool checkTempUpdate()
            {
                bool needUpdate = this->tempCount%this->tempResetCount==0 || this->tempCount <= 1;
                std::string needUpdateText = needUpdate?"true":"false";
                logger::debug("Tracker,checkTempUpdate,id="+std::to_string(this->id)+",status="+needUpdateText,"DONE");
                return needUpdate;
            }

            void updateTemp(cv::Mat frame, cv::Point center, int roiWidth, int roiHeight)
            {
                logger::debug("Tracker,TrackedObject,updateTemp,id="+std::to_string(this->id)+",count="+std::to_string(this->tempCount),"START");
                if(this->tempCount%this->tempResetCount==0 || this->tempCount<=1 || tracker::globalNeedUpdate)
                {
                    this->findTemp(frame, center, roiWidth, roiHeight);
                    this->tempUpdateCount++;
                    logger::debug("Tracker,TrackedObject,updateTemp,id="+std::to_string(this->id)+",count="+std::to_string(this->tempCount),"RESET");
                }
            }

            void updateTemp(cv::Mat frame, cv::Rect box)
            {
                logger::debug("Tracker,TrackedObject,updateTemp,id="+std::to_string(this->id)+",count="+std::to_string(this->tempCount),"START");
                if(this->tempCount%this->tempResetCount==0 || this->tempCount<=1 || tracker::globalNeedUpdate)
                {
                    this->findTemp(frame, box);
                    this->tempUpdateCount++;
                    logger::debug("Tracker,TrackedObject,updateTemp,id="+std::to_string(this->id)+",count="+std::to_string(this->tempCount),"RESET");
                }
            }

            void findTemp(cv::Mat frame , cv::Point center, int roiWidth, int roiHeight)
            {
                logger::debug("Tracker,TrackedObject,findTemp,id="+std::to_string(this->id),"START");
                double temp = this->flir->getTemp(frame, center, roiWidth, roiHeight);
                this->temp = temp>0?temp:this->temp;
                this->tempUpdated = temp>0;
                logger::debug("Tracker,TrackedObject,findTemp,id="+std::to_string(this->id)+",temp="+std::to_string(this->temp)+"C","END");
            }

            void findTemp(cv::Mat frame, cv::Rect box)
            {
                logger::debug("Tracker,TrackedObject,findTemp,id="+std::to_string(this->id),"START");
                double temp = this->flir->getTemp(frame, box);
                this->temp = temp>0?temp:this->temp;
                this->tempUpdated = temp>0;
                logger::debug("Tracker,TrackedObject,findTemp,id="+std::to_string(this->id)+",temp="+std::to_string(this->temp)+"C","END");
            }
    };
    
    std::vector<tracker::TrackedObject> trackedList;

    void addTracker(tracker::TrackedObject &obj)
    {
        tracker::trackedList.push_back(obj);
    }

    TrackedObject *createTracker(cv::Point &center, flir::FLIR &flir)
    {
        TrackedObject *obj = new TrackedObject(center, flir);
        tracker::addTracker(*obj);
        return obj;
    }

    int verifyAll(cv::Point center)
    {
        logger::debug("Tracker,verifyAll","START");
        for(int i = 0; i < tracker::trackedList.size(); i++)
        {
            if(tracker::trackedList[i].isAlive)
            {
                if(tracker::trackedList[i].verify(center))
                {
                    logger::debug("Tracker,verifyAll,object has been tracked,id="+std::to_string(i),"END");
                    return i;
                }
            }
        }
        logger::debug("Tracker,verifyAll,object not yet tracked","END");
        return -1;
    }

    void refreshAll()
    {
        logger::debug("Tracker,refreshAll","START");
        if(tracker::globalUpdateTemp) tracker::globalTempCount++;
        for(int i = 0; i < tracker::trackedList.size(); i++) tracker::trackedList[i].refresh();
        logger::debug("Tracker,refreshAll","END");
    }

    int checkTempUpdateAll()
    {
        logger::debug("Tracker,checkTemperatureUpdateAll","START");
        if(tracker::globalUpdateTemp)
        {
            if(tracker::globalTempCount%tracker::globalTempResetCount==0)
            {
                tracker::globalNeedUpdate = true;
                logger::debug("Tracker,checkTemperatureUpdateAll,needTemp=true,globalUpdate","END");
                return 1;
            }
        }
        else
        {
            for(int i = 0; i < tracker::trackedList.size(); i++)
            {
                if(tracker::trackedList[i].checkTempUpdate())
                {
                    logger::debug("Tracker,checkTemperatureUpdateAll,needTemp=true,firstHit at id="+std::to_string(i),"END");
                    return i;
                }
            }
        }
        logger::debug("Tracker,checkTemperatureUpdateAll,needTemp=false","END");
        return -1;
    }
}