from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import dlib
import os
import numpy as np

from keras.models import model_from_json
from keras.models import load_model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

filepath = 'names.txt'
people = []
with open(filepath) as fp:
     line = fp.readline()
     while line:
        people.append(line.strip())
        line = fp.readline()
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 0)

    for rect in rects:
        faceAligned = fa.align(frame, gray_frame, rect)
        faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
        faceAligned = np.array(faceAligned)
        faceAligned = faceAligned.astype('float32')
        faceAligned /= 255.0
        faceAligned= np.expand_dims([faceAligned], axis=4)

        Y_pred =  loaded_model.predict(faceAligned)
        for index, value in enumerate(Y_pred[0]):
          result = people[index] + ': ' + str(int(value * 100)) + '%'
          cv2.putText(frame, result, (10, 15 * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)
        result = np.argmax(Y_pred, axis=1)
        cv2.putText(frame, people[result[0]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Frame", frame)

cv2.destroyAllWindows()
vs.stop()