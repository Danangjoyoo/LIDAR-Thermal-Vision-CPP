import cv2, os, sys, time

def go(id):
    print("camID", id)
    cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
    now = time.perf_counter()
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        # frame = cv2.resize(frame, (320, 240))
        cv2.imshow("aaa",frame)
        if cv2.waitKey(5)>0: break;
        t = now
        now = time.perf_counter()
        print("FPS:{}==".format(int(1/(now-t))),end="\r")
    cap.release()
    cv2.destroyAllWindows()

if len(sys.argv) > 1:
    go(int(sys.argv[1]))
else:
    go(0)

