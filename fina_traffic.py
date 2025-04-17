# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from flask import Flask, render_template, Response, redirect, url_for
import pyttsx3
app = Flask(__name__)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args([])) 

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["sign.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["model.weights"])
configPath = os.path.sep.join(["model.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream
vs = cv2.VideoCapture('input.mp4')

def detect_output_result(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            # extract the class ID and confidence 
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions probability 
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    
    if len(idxs) > 0:
        
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            
            color = [int(c) for c in COLORS[classIDs[i]]]

            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

           
            if LABELS[classIDs[i]] == "traffic light":
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, "traffic light", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
##                engine = pyttsx3.init()
##                engine.say('traffic light')
##                engine.runAndWait()

            if LABELS[classIDs[i]] == "stop sign":
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, "stop sign", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
##                engine = pyttsx3.init()
##                engine.say('stop sign')
##                engine.runAndWait()

    return frame

def generate():
    global vs
    while True:
        if vs is not None:
            grabbed, frame = vs.read()
            if not grabbed:
                break

            frame = detect_output_result(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global vs
    if vs is None:
        vs = cv2.VideoCapture('input.mp4')  # Change the video file name here
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global vs
    if vs is not None:
        vs.release()
        vs = None
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
