import cv2
import face_recognition
import pickle
import os




def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def encoding():

    folderPath = "Images"
    pathList = os.listdir(folderPath)
    imgList = []
    studentIds = []
    for path in pathList:
        imgList.append(cv2.imread(os.path.join(folderPath,path)))
        studentIds.append(os.path.splitext(path)[0])
    print(studentIds)

    print("Encode Start ....")
    encodeListKnow = findEncodings(imgList)
    encodeListKnowWithIds = [encodeListKnow,studentIds]
    print(encodeListKnow)
    print("Encode Complete")


    file = open("EncodeFile.p","wb")
    pickle.dump(encodeListKnowWithIds,file)
    file.close
    print("File Save")

