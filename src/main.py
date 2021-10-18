import cv2
import numpy as np

videoCapture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('C:\\Users\\joel.488714\\face_swap\\src\\images\\carajoel2.png')

if videoCapture.isOpened():
    rval, frame = videoCapture.read()
else:
    rval = False

while rval:
    rval, frame = videoCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facez = faceCascade.detectMultiScale(gray, 1.3, 5)

    print(facez)

    for (x, y, w, h) in facez:

        cropImg = frame[y:y+h, x:x+w]

        height = cropImg.shape[0]
        width = cropImg.shape[1]

        resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

        frame[y:y+h, x:x+w] = resized
    cv2.imshow('Camerax', frame)

    key = cv2.waitKey(20)

    if key == 27:
        break

videoCapture.release()
cv2.destroyWindow("Camera")