#include <iostream>
#include <vector>

#pragma once

namespace argparser
{
    int argc = 0;
    char *argv[100];

    void set(int argc, char* argv[])
    {
        argparser::argc = argc;
        for(int i = 0; i < argc; i++)
        {
            argparser::argv[i] = argv[i];
        }
    }

    std::string getKey(std::string val)
    {
        if(val[0] == '-')
        {
            std::string cloneVal = val;
            std::string nohifinVal;
            if(val[1] == '-')
            {
                nohifinVal = cloneVal.replace(cloneVal.begin(), cloneVal.begin()+2, "");
            }
            else
            {
                nohifinVal = cloneVal.replace(cloneVal.begin(), cloneVal.begin()+1, "");
            }
            return nohifinVal;
        }
        return "";
    }

    enum HifinType
    {
        SINGLE = 0,
        DOUBLE = 1
    };
    
    class KwargsMap
    {
        public:
            std::vector<std::string> keys;
            std::vector<std::string> args;
            std::vector<std::string> hifin;
            int totalArgs = 0;

            bool validateArgs(std::string key)
            {
                return key.length()>0;
            }

            KwargsMap()
            {
                argparser::KwargsMap::getMappedKwargs(*this);
            };

            static void getMappedKwargs(KwargsMap &mapped)
            {
                // format [[keys],[hifin],[args]]
                std::vector<std::string> keys;
                std::vector<std::string> args;
                std::vector<std::string> hifin;
                int currentConfig = 0;
                bool getThis = false;
                for(int i = 0; i < argparser::argc; i++)
                {
                    std::string val = argparser::argv[i];
                    if(!getThis)
                    {
                        if(val[0] == '-')
                        {
                            getThis = true;
                            std::string cloneVal = val;
                            std::string nohifinVal;
                            currentConfig = i;
                            if(val[1] == '-')
                            {
                                hifin.push_back("--");
                                nohifinVal = cloneVal.replace(cloneVal.begin(), cloneVal.begin()+2, "");
                            }
                            else
                            {
                                hifin.push_back("-");
                                nohifinVal = cloneVal.replace(cloneVal.begin(), cloneVal.begin()+1, "");
                            }
                            keys.push_back(cloneVal);
                        }

                        if(getThis)
                        {
                            if(i == argparser::argc-1)
                            {
                                mapped.totalArgs++;
                                args.push_back("-1");
                            }
                            else
                            {
                                if(argparser::argv[currentConfig+1][0]=='-')
                                {
                                    mapped.totalArgs++;
                                    getThis = false;
                                    args.push_back("-1");
                                }
                            }
                        }
                    }
                    else
                    {
                        mapped.totalArgs++;
                        args.push_back(argparser::argv[currentConfig+1]);
                        getThis = false;
                    }
                }
                mapped.keys = keys;
                mapped.args = args;
                mapped.hifin = hifin;
            }

            std::string getArgs(int hifinType, std::string key)
            {
                std::string hif = hifinType==0?"-":"--";
                for(int i = 0; i < this->totalArgs; i++)
                {
                    if(this->hifin[i] == hif && this->keys[i] == key) return this->args[i];
                }
                return "";
            }

            std::string getArgs(std::string key)
            {
                if(!validateArgs(key) || key[0]!='-') return "";
                std::string hif = key[1]=='-'?"--":"-";
                key = getKey(key);
                for(int i = 0; i < this->totalArgs; i++)
                {
                    if(this->hifin[i] == hif && this->keys[i] == key) return this->args[i];
                }
                return "";
            }

            bool checkArgs(int hifinType, std::string key)
            {
                return this->getArgs(hifinType, key)!="";
            }

            bool checkArgs(std::string key)
            {
                return this->getArgs(key)!="";
            }
    };
}