import face_recognition
import cv2
import numpy as np
import sys

# Ask for path to image
if(len(sys.argv) < 2):
    print("Enter One Picture Path for Face Detection.")
    exit(0)

image = cv2.imread(sys.argv[1])
face_locations = face_recognition.face_locations(image)
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0,0,255),2)
cv2.imshow('image', image)
while True:
    k = cv2.waitKey(1000) # change the value from the original 0 (wait forever) to something appropriate
    if k == 27:
        cv2.destroyAllWindows()
        break        
    if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
        break        
cv2.destroyAllWindows()
