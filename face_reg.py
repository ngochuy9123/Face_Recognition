import cv2
import os
import pickle
import face_recognition
import numpy as np

print("Loading Encoding File ...")
file  = open('EncodeFile.p',"rb")
encodeListKnowWithIds = pickle.load(file)
file.close()
encodeListKnow,studentIds = encodeListKnowWithIds
# print(studentIds)
print("Encode File Loaded")

vid = cv2.VideoCapture(0) 
vid.set(3,640)
vid.set(4,480)

  
while(True): 
      
    # Capture the video frame 
    # by frame 
    success, img = vid.read() 
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)

        print(matches)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print("Match Index",matchIndex)
        if matches[matchIndex]:
            print(studentIds[matchIndex])
            label = studentIds[matchIndex]

            y1,x2,y2,x1 = faceLoc
            print(y1,x2,y2,x1)
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            # bbox = 55+x1,162+y1,x2-x1,y2-y1
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)

            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

    # Display the resulting frame 
    cv2.imshow('frame', img) 
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Destroy all the windows 
cv2.destroyAllWindows() 