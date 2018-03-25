import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys
from skimage import exposure

def detectWink(frame, location, ROI, cascade,w,h):
    sharpness=cv2.Laplacian(ROI, cv2.CV_64F).var()
    # print("Face sharpness: ",sharpness)
    ''' if sharpness < 20:
        kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
        ROI = cv2.filter2D(ROI, -1, kernel) '''
    eyes = cascade.detectMultiScale(
        ROI, 1.1014 ,15, 0|cv2.CASCADE_SCALE_IMAGE, (10, 20)) 
    
    ''' if w < 100 and h < 100:
        eyes = cascade.detectMultiScale(
        ROI, 3.0 ,4, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10))   '''
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    return len(eyes) == 1    # number of eyes is one

def detect(frame, faceCascade, eyesCascade):
    

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # possible frame pre-processing:
    

    ''' clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(30,30))
    gray_frame = clahe.apply(gray_frame) '''
    #gray_frame = cv2.medianBlur(gray_frame, 5)

    
    

    #gray_frame = cv2.GaussianBlur(gray_frame,(3,3),0)

    #gray_frame = frame
    blur_res = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    
    
    #if blur_res > 250 :
        #print(blur_res)
        #gray_frame = cv2.equalizeHist(gray_frame)
        #gray_frame = cv2.GaussianBlur(gray_frame,(1,1),0)
       

        #gray_frame = exposure.adjust_gamma(gray_frame, 2)
    #print(blur_res)

    #gray_frame = exposure.adjust_gamma(gray_frame, 2)

    #else :

    
    rows, cols = gray_frame.shape
    total=0
    for i in range(0, rows) :
        for j in range(0, cols) :
            total+= gray_frame[i, j]
    
    avg=total/(rows*cols)
    # print("Average: " + str(avg))   
    if avg > 190 or avg < 65: 


        gray_frame = cv2.equalizeHist(gray_frame)
    
    gray_frame = cv2.GaussianBlur(gray_frame,(3,3),0)


        #cv2.imshow("Sharpen", gray_frame)

    
    #gray_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2RGB)

    scaleFactor = 1.14 # range is from 1 to ..
    minNeighbors = 11   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (30,30) # range is from (0,0) to ..
    
    faces = faceCascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)

    detected = 0
    for f in faces:
        
        x, y, w, h = f[0], f[1], f[2], f[3]
        h1 = (int)(h/1.5)
        #print("dimension :" , w , h)
        faceROI = gray_frame[y:y+h1, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade,w,h):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h1), (255, 0, 0), 2)
            #print("detected",f)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h1), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)

