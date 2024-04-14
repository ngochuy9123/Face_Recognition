from flask import Flask, Response, render_template,request
from flask import Flask, flash, request, redirect, url_for
import json
import os
import cv2
import pickle
import face_recognition
import numpy as np
import encodeGenerator
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

def video_detection():
    while(True): 
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
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),3)

                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [0,255,0], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img


def generate_frames():
    yolo_output = video_detection()
    for detection in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def hello():
    return 'Hello, World!'


UPLOAD_FOLDER = 'Images' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/encodeimg', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        employeeId =""

        f = request.files['file']
        if f.filename == '':
            print('No file selected')
            return render_template('encodeImage.html')

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], employeeId)
        os.makedirs(upload_path, exist_ok=True) 

        filename = secure_filename(f.filename)

        f.save(os.path.join(upload_path, filename))
        encodeGenerator.encoding()
        print(f'File {filename} uploaded successfully for employee {employeeId}')
        return render_template('encodeImage.html') 

    return render_template('encodeImage.html')



if __name__ == '__main__':
    app.run()