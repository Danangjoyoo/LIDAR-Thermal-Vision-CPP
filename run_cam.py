import os, sys, time

startUpTime = 5

if len(sys.argv)>1:
    if sys.argv[1] == "run":
        t0 = time.time()
        i = startUpTime
        while time.time()-t0 < startUpTime:
            time.sleep(1); i -= 1
            print("Starting Camera Vision in {:>3}".format(f"{i}s"), end="\r")
        print("Starting Camera Vision in {:>3}".format(f"{i}s"))
        print("[ScriptRunner] Changing Directory ")
        os.chdir("/home/jetson/dev/dnn1")
        print("[ScriptRunner] Compiling Start")
        os.system("g++ -std=c++11 main.cpp -I /usr/local/include/opencv4/ -L /usr/local/lib/ -L /usr/local/cuda/lib64 -lcuda -lcudart -lrealsense2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lopencv_cudaobjdetect -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_dnn_objdetect -lopencv_dnn -o main")
        print("[ScriptRunner] Compiling Done")
        print("[ScriptRunner] Running Vision Cam")
        os.system("./main")
    elif sys.argv[1] == "--skip-wait":
        os.system('lxterminal --command="python3.9 /home/jetson/dev/dnn1/run_cam.py run"')
else:
    time.sleep(3)
    os.system('lxterminal --command="python3.9 /home/jetson/dev/dnn1/run_cam.py run"')