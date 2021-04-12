import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('video.mp4')

try:
    if not os.path.exists('frames'):
        os.makedirs('frames')
except OSError:
    print ('Unable to create the images directory')

currentFrame = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Unable to recieve frame or video ended. Exiting ...")
        break;

    # grayscale image output
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)
    # if cv2.waitKey(1) == ord('q'):
    #     break

    # Saves image of the current frame in jpg file
    name = './frames/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
