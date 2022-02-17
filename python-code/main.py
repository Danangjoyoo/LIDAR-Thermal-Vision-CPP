import cv2
import numpy as np
import time
import json

class Detector:
    def __init__(self, net, availableClasses, detectionColors, confidenceThreshold, targetInputSizeIdx=2):
        self.net = net
        self.availableClasses = availableClasses
        self.detectionColors = detectionColors
        ln = self.net.getLayerNames()
        self.layerNames = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # self.layerNames = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.confidenceThreshold = confidenceThreshold
        self.targetInputSizeIdx = targetInputSizeIdx
    
    def run(self, deviceID = 0):
        cap = cv2.VideoCapture(deviceID)
        while cap.isOpened():
            t0 = time.perf_counter()
            
            _, frame = cap.read()
            frame = self.detect(frame)

            t1 = time.perf_counter()
            fps = int(1/(t1-t0))
            cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 10, 10), 2)

            cv2.imshow("Detection Window", frame)
            if cv2.waitKey(5) > 0: break
        cap.release()
        cv2.destroyAllWindows()

    def detect(self, originalFrame):
        frame = originalFrame.copy()

        # detecting
        blobSize = [416, 320, 224, 128]

        blob = cv2.dnn.blobFromImage(frame, 1/255, (blobSize[self.targetInputSizeIdx], blobSize[self.targetInputSizeIdx]), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layerNames)
        outputs = np.vstack(outputs)

        # drawing
        H, W = frame.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > self.confidenceThreshold:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                # p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidenceThreshold, self.confidenceThreshold-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.detectionColors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y+h), color, 2)

                text = "{}: {:.0f}%".format(self.availableClasses[classIDs[i]], confidences[i]*100)
                
                # text upper inside the box
                # cv.putText(img, text, (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # text upper outside the box
                # cv.rectangle(img, (x,y-10), (x+(13*len(text)), y+10), (255,255,255), -1)
                # cv.putText(img, text, (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

                # text lower outside the box
                cv2.rectangle(frame, (x,y-10+h-10), (x+(13*len(text)), y+h), (255,255,255), -1)
                cv2.putText(frame, text, (x+5, y+5+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        return frame


# read from object.json for names and colors
with open("../yolo/object.json","r") as f:
    availableObject = json.load(f)
classes = list(availableObject["object"].keys())
colors = list(availableObject["object"].values())

confidenceThresh = 0.5 # above 50% will passed
net = cv2.dnn.readNetFromDarknet(
    '../yolo/facemask-yolov4-tiny.cfg',
    '../yolo/facemask-yolov4-tiny_best.weights'
    )
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

detector = Detector(net, classes, colors, confidenceThresh, 2)
detector.run(deviceID=1)