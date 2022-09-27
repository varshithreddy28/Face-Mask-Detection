from flask import Flask, render_template, request, flash, redirect
import jyserver.Flask as jsf
import cv2
import numpy as np
from keras.models import load_model

model = load_model('FaceMask.h5')

# ioT
from controler import doorAutomate

import urllib.request

url = 'http://192.168.0.16/400x296.jpg'

app = Flask(__name__)


def getOutPut():
    text = ''
    for i in range(0, 2):
        imgResponse = urllib.request.urlopen(url)
        fileName = "capture" + str(i) + ".jpg"
        print(fileName)
        urllib.request.urlretrieve(url, fileName)
    imgnp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    imgnp = cv2.imdecode(imgnp, -1)
    frame = imgnp
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        faces = frame[y:y + h, x:x + w]

    try:
        img = cv2.resize(faces, dsize=(300, 300))
    except:
        img = cv2.resize(frame, dsize=(300, 300))
    img = np.array([img])
    result = model.predict(img)

    if result[0][0] == 1:
        # With Mask
        doorAutomate(0)
        return "With Mask"
    else:
        # WIth Out Mask
        doorAutomate(1)
        return "With Out Mask"


# doorAutomate(1)

# print(text)
# res = text
# retun res


@app.route('/', methods=["GET"])
def home():
    if request.method == 'GET':
        return render_template('index.html')


@app.route('/getPhoto', methods=["GET"])
def result():
    if request.method == 'GET':
        result = getOutPut()
        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run()