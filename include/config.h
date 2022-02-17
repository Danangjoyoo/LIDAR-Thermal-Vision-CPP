#ifndef CONFIG_H_
#define CONFIG_H_

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace json = nlohmann;

namespace config
{
    json::json readConfig(std::string filepath)
    {
        std::ifstream i(filepath);
        json::json j;
        i >> j;
        return j;
    }

    std::string getValue(json::json jsonQuery)
    {
        std::stringstream ss; std::string s;
        ss << jsonQuery; ss >> s;
        return s;
    }

    std::string getString(json::json jsonQuery)
    {
        std::string s = config::getValue(jsonQuery);
        s.replace(s.begin(),s.begin()+1,"");
        s.replace(s.end()-1,s.end(),"");
        return s;
    }

    int getInt(json::json jsonQuery)
    {
        return std::stoi(config::getValue(jsonQuery));
    }

    double getDouble(json::json jsonQuery)
    {
        return std::stod(config::getValue(jsonQuery));
    }

    bool getBool(json::json jsonQuery)
    {
        if(config::getValue(jsonQuery)=="true") return true;
        else return false;
    }

    std::string configFilePath = "/home/rnd/dev/dnn1/config.json";

    json::json rawConfig = config::readConfig(configFilePath);

    namespace camera
    {
        class Frame
        {
            public:
                int width, height;
                Frame(int width, int height)
                {
                    this->width = width;
                    this->height = height;
                }
        };
        json::json cameraConfig = config::rawConfig["camera"];

        Frame colorFrame(config::getInt(cameraConfig["colorFrame"]["w"]), config::getInt(cameraConfig["colorFrame"]["h"]));
        Frame depthFrame(config::getInt(cameraConfig["depthFrame"]["w"]), config::getInt(cameraConfig["depthFrame"]["h"]));
        Frame thermalFrame(config::getInt(cameraConfig["thermalFrame"]["w"]), config::getInt(cameraConfig["thermalFrame"]["h"]));
    };


    namespace objectDetection
    {
        std::vector<std::string> _getAvailableClasses(json::json jsonQueryList)
        {
            std::vector<std::string> classes;
            for(int i = 0; i >= 0; i++)
            {
                std::string val = config::getValue(jsonQueryList[i]);
                if(val!="null") classes.push_back(config::getString(jsonQueryList[i]["name"]));
                else break;
            }
            return classes;
        }

        std::vector<cv::Scalar> _getAvailableColors(json::json jsonQueryList)
        {
            std::vector<cv::Scalar> res;
            for(int i = 0; i >= 0; i++)
            {
                std::string val = config::getValue(jsonQueryList[i]);
                if(val!="null")
                {
                    json::json rgbConfig = jsonQueryList[i]["color"];
                    int r = config::getInt(rgbConfig[0]);
                    int g = config::getInt(rgbConfig[1]);
                    int b = config::getInt(rgbConfig[2]);
                    res.push_back(cv::Scalar(r,g,b));
                }
                else break;
            }
            return res;
        }

        json::json objDetConfig = config::rawConfig["objectDetection"];

        double confidenceThreshold = config::getDouble(objDetConfig["confidenceThreshold"]);
        std::string network = config::getString(objDetConfig["network"]);
        namespace models
        {
            std::string network = objectDetection::network;
            int idx = config::getInt(objDetConfig["frameworkIndex"]);
            json::json networkConfig = objDetConfig["framework"][network][idx];
            int inputSize = config::getInt(networkConfig["inputSize"]);
            std::string configfile = config::getString(networkConfig["config"]);
            std::string modelfile = config::getString(networkConfig["model"]);
            std::string classfile = config::getString(networkConfig["classfile"]);
            std::vector<std::string> classes = _getAvailableClasses(config::readConfig(classfile)["object"]);
            std::vector<cv::Scalar> colors = _getAvailableColors(config::readConfig(classfile)["object"]);
        }
    }

    namespace objectTracking
    {
        json::json objTrackConfig = config::rawConfig["objectTracking"];
        int movingThreshold = config::getInt(objTrackConfig["movingThreshold"]);
        int temperatureResetCount = config::getInt(objTrackConfig["temperatureResetCount"]);
        int temperatureScanArea = config::getInt(objTrackConfig["temperatureScanArea"]);
        int updateTemperatureAttempt = config::getInt(objTrackConfig["updateTemperatureAttempt"]);
        int registerThreshold = config::getInt(objTrackConfig["registerThreshold"]);
        int deleteThreshold = config::getInt(objTrackConfig["deleteThreshold"]);
        int globalTemperatureResetCount = config::getInt(objTrackConfig["globalTemperatureResetCount"]);
        bool useGlobalTemperatureUpdate = config::getBool(objTrackConfig["useGlobalTemperatureUpdate"]);
    }

    int determineLoggerLevel(std::string levelStr)
    {
        std::vector<std::string> levels = {"INFO","LOG","DEBUG","ERROR","CRITICAL"};
        for(int i=0;i<levels.size();i++)
            if(levels[i]==levelStr) return i;
        return 0;
    }

    namespace logger
    {
        json::json loggerConfig = config::rawConfig["logger"];
        bool enable = config::getBool(loggerConfig["enable"]);
        int level = config::determineLoggerLevel(config::getString(loggerConfig["level"]));
        std::string outfile = config::getString(loggerConfig["outfile"]);
    }

    namespace flir
    {
        std::vector<double> _getPolynom(json::json jsonQueryList)
        {
            double x2 = config::getDouble(jsonQueryList[0]);
            double x1 = config::getDouble(jsonQueryList[1]);
            double x0 = config::getDouble(jsonQueryList[2]);
            return std::vector<double>({x2,x1,x0});
        }

        std::vector<double> _getRange(json::json jsonQueryList)
        {
            double min = config::getDouble(jsonQueryList[0]);
            double max = config::getDouble(jsonQueryList[1]);
            return std::vector<double>({min,max});
        }

        json::json flirConfig = config::rawConfig["flir"];
        int refreshThresh = config::getDouble(flirConfig["refreshThresh"]);
        bool showFrameBoundary = config::getBool(flirConfig["showFrameBoundary"]);
        namespace frameSafetyFactor
        {
            double x = config::getDouble(flirConfig["frameSafetyFactor"]["x"]);
            double y = config::getDouble(flirConfig["frameSafetyFactor"]["y"]);
        }
        std::vector<double> distPolynom = flir::_getPolynom(flirConfig["distancePolynom"]);
        std::vector<double> distPolyRange = flir::_getRange(flirConfig["distancePolynomRange"]);
        std::vector<double> tempPolynom = flir::_getPolynom(flirConfig["tempPolynom"]);
        std::vector<double> tempPolyRange = flir::_getRange(flirConfig["tempPolynomRange"]);
    }
}

#endif