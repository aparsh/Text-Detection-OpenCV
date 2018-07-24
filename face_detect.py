import numpy as np
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

def get_face(img,gray):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    max_area = 0
    face_tuple = None
    for (x,y,w,h) in faces:
        if w*h > max_area:
            max_area = w*h
            face_tuple = (x,y,w,h)
    if face_tuple[2]*face_tuple[3] == 0:
        return None
    else:
        return face_tuple