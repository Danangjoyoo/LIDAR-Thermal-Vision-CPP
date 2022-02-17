#!/bin/sh

echo "[Executor] Initializing"
secs=$((0))
while [ $secs -gt 0 ]; do
   echo "[Executor] Vision Cam Started in $secs \r"
   secs=$(($secs-1))
   sleep 1
done

# echo "[Executor] Compiling Start"
cd /home/jetson/dev/dnn1

lastMod=$(stat -c%Y main)

# g++ -std=c++11 /home/jetson/dev/dnn1/main.cpp -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lopencv_cudaobjdetect -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_dnn_objdetect -lopencv_dnn -o /home/jetson/dev/dnn1/main
# g++ -std=c++11 /home/jetson/dev/dnn1/main.cpp -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_dnn_objdetect -lopencv_dnn -o /home/jetson/dev/dnn1/main
# g++ -std=c++11 /home/jetson/dev/dnn1/main.cpp -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_dnn_objdetect -lopencv_dnn -lpthread  -o /home/jetson/dev/dnn1/main

/home/jetson/dev/dnn1/compile_cam.sh

nowMod=$(stat -c%Y main)

# echo "$nowMod ======= $lastMod"
if [ $nowMod != $lastMod ]
then
    echo "[Executor] Compiling Done"
    echo "[Executor] Vision Cam Start"
    ./main
    echo "[Executor] Vision Cam Done"
else
    echo "[Executor] Compiling Error"
    echo "[Executor] Aborted"
fi

echo "press ENTER to exit"
read exitMsg