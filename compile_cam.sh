#!/bin/sh

echo "[Executor] Compiling Start"

# g++ -std=c++11 /home/rnd/dev/dnn1/main.cpp -I /home/rnd/Documents -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_dnn_objdetect -lopencv_dnn -lopencv_dnn_superres -lpthread  -o /home/rnd/dev/dnn1/visioncam

g++ -std=c++11 ./main.cpp \
    -I /home/rnd/Documents \
    -I /usr/local/include/opencv4/ \
    -I ./include \
    -L /usr/local/lib/ \
    -L /usr/local/cuda/lib64 \
    -lcuda -lcudart -lrealsense2 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio \
    -lopencv_dnn_objdetect -lopencv_dnn -lopencv_dnn_superres -lpthread \
    -o ./visioncam