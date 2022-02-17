#include <iostream>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <stdexcept>


#pragma once

namespace logger
{
    void _baseLogger(int level, std::string process, std::string message);
    void _baseLogger(int level, const std::exception& e);
    void info(std::string process, std::string message);
    void log(std::string process, std::string message);
    void debug(std::string process, std::string message);
    void error(const std::exception& e);
    void error(std::string process, std::string message);
    void critical(std::string process, std::string message);
    void critical(const std::exception& e);
    std::string getTimestamp();
    void showConfig();

    bool enableStatus = false;
    
    int _level = 0;
    std::string allLevels[5] = {"INFO", "LOG", "DEBUG", "ERROR", "CRITICAL"};

    enum LEVEL
    {
        INFO = 0,
        LOG = 1,
        DEBUG = 2,
        ERROR = 3,
        CRITICAL = 4
    };

    LEVEL LEVEL_INFO = INFO;
    LEVEL LEVEL_LOG = LOG;
    LEVEL LEVEL_DEBUG = DEBUG;
    LEVEL LEVEL_ERROR = ERROR;
    LEVEL LEVEL_CRITICAL = CRITICAL;

    std::ofstream outfile;
    std::string outfileName = "";
    auto start = std::chrono::high_resolution_clock::now();

    void setLevel(int level)
    {
        logger::_level = level;
    }

    void enable(bool stat)
    {
        if(stat)
        {
            logger::enableStatus = stat;
            if(logger::outfileName=="")
            {
                logger::outfileName = "./log.txt";
                logger::outfile.open(logger::outfileName);
                logger::info("Logger","Message=Using default outputfilename");
                logger::info("Logger,Initialized","outfile="+logger::outfileName);
            }
            else
            {
                logger::outfile.open(logger::outfileName);
                logger::info("Logger,Initialized","outfile="+logger::outfileName);
            }
            logger::showConfig();
        }
        else
        {
            logger::info("ClosingLogger","outfile="+logger::outfileName);
            logger::enableStatus = stat;
            logger::outfile.close();
        }
    }

    void enable(bool stat, std::string filename)
    {
        logger::outfileName = filename;
        logger::enable(stat);
    }

    void showConfig()
    {
        std::time_t tnow = std::chrono::high_resolution_clock::to_time_t(logger::start);
        std::string date = std::ctime(&tnow);
        date.replace(date.end()-2, date.end(), "");
        std::string enableText = logger::enableStatus?"true":"false";
        std::cout << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] << LOG CONFIGURATION >> " << std::endl;
        std::cout << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] enabled : " << enableText << std::endl;
        std::cout << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] level   : " << logger::allLevels[logger::_level] << std::endl;
        std::cout << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] outfile : " << logger::outfileName << std::endl;
        std::cout << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] start_t : " << date << std::endl;
        logger::outfile << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] << LOG CONFIGURATION >> " << std::endl;
        logger::outfile << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] enabled : " << logger::enableStatus << std::endl;
        logger::outfile << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] level   : " << logger::allLevels[logger::_level] << std::endl;
        logger::outfile << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] outfile : " << logger::outfileName << std::endl;
        logger::outfile << "["+logger::getTimestamp()+"]" << "[LOGGER_INFO] start_t : " << date << std::endl;
    }

    std::string getTimestamp()
    {
        try
        {
            auto now = std::chrono::high_resolution_clock::now();
            std::time_t tnow = std::chrono::high_resolution_clock::to_time_t(now);

            // Linux

            std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
            long milli = (double)ms.count();
            std::string milliStamp = std::to_string(milli);
            milliStamp.replace(milliStamp.begin(), milliStamp.end()-3, "");

            std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
            long micro = (double)us.count();
            std::string microStamp = std::to_string(micro);
            microStamp.replace(microStamp.begin(), microStamp.end()-3, "");

            std::string timestamp = std::ctime(&tnow);
            timestamp.replace(timestamp.begin(), timestamp.begin()+11,"");
            timestamp.replace(timestamp.end()-6, timestamp.end(), "."+milliStamp+"."+microStamp);

            // Windows
            // std::chrono::minutes m = std::chrono::duration_cast<std::chrono::minutes>(now-logger::start);
            // long min = (double)m.count();
            // std::string minStamp = std::to_string(min);

            // std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(now-logger::start);
            // long sec = (double)s.count();
            // std::string secStamp = std::to_string(sec);

            // std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(now-logger::start);
            // long milli = (double)ms.count();
            // std::string milliStamp = std::to_string(milli);
            // milliStamp.replace(milliStamp.begin(), milliStamp.end()-3, "");

            // std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(now-logger::start);
            // long micro = (double)us.count();
            // std::string microStamp = std::to_string(micro);
            // microStamp.replace(microStamp.begin(), microStamp.end()-3, "");

            // std::string timestamp = minStamp+":"+secStamp+"."+milliStamp+"."+microStamp;
            
            return timestamp;
        }
        catch(const std::exception& e)
        {
            return "timestampError";
        }        
    }

    void _baseLogger(int level, std::string process, std::string message)
    {
        if(logger::enableStatus)
        {
            if(level <= logger::_level)
            {
                std::string timestamp = logger::getTimestamp();
                std::string logText = "["+timestamp+"]["+logger::allLevels[level]+"] "+process+","+message;
                // std::string logText = ""+timestamp+";"+logger::allLevels[level]+";"+process+","+message;
                std::cout << logText << std::endl;
                logger::outfile << "\n"+logText;
            }
        }
    }

    void _baseLogger(int level, const std::exception& e)
    {
        if(logger::enableStatus)
        {
            if(level <= logger::_level)
            {
                std::string timestamp = logger::getTimestamp();
                std::string logText = "["+timestamp+"]["+logger::allLevels[level]+"] Exception=";
                // std::string logText = ""+timestamp+";"+logger::allLevels[level]+";Exception=";
                std::cout << logText << e.what() << std::endl;
                logger::outfile << "\n"+logText << e.what();
            }
        }
    }

    void info(std::string process, std::string message)
    {
        logger::_baseLogger(logger::LEVEL_INFO, process, message);
    }

    void log(std::string process, std::string message)
    {
        logger::_baseLogger(logger::LEVEL_LOG, process, message);
    }

    void debug(std::string process, std::string message)
    {
        logger::_baseLogger(logger::LEVEL_DEBUG, process, message);
    }

    void error(const std::exception& e)
    {
        logger::_baseLogger(logger::LEVEL_ERROR, e);
    }

    void error(std::string process, std::string message)
    {
        logger::_baseLogger(logger::LEVEL_ERROR, process, message);
    }

    void critical(std::string process, std::string message)
    {
        logger::_baseLogger(logger::LEVEL_CRITICAL, process, message);
    }

    void critical(const std::exception& e)
    {
        logger::_baseLogger(logger::LEVEL_CRITICAL, e);
    }
}