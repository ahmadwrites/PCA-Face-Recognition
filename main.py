import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib

#load training file saved in training.py which is in .yml format
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
names=[]
warningtime = 0

def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

statusStr = {
    '0': 'send unsucessfully',
    '-1': 'index incomplete',
    '-2': 'The server space is not supported. Please confirm that you support CURL or fsocket and contact your space provider to solve the problem or change the space',
    '30': 'password wrong',
    '40': 'account invaild',
    '41': 'balance low',
    '42': 'account update',
    '43': 'IP address limit',
    '50': 'Contain sensitive words'
}


def warning():
    smsapi = "http://api.smsbao.com/"
    # account
    user = '135****1900'
    # psw
    password = md5('*******')
    # content
    content = '[warning]\nreason: detect unknown people\nlocatiton:xxx'
    # sent to
    phone = '*******'

    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])

#image ready to recognize
def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#trun to gray
    face_detector=cv2.CascadeClassifier('C:/Users/Ahmad/.virtualenvs/PCA_Project/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # face recognition
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        #print('lable's id:',ids,'confidence', confidence)
        if confidence > 80:
            global warningtime
            warningtime += 1
            if warningtime > 100:
               warning()
               warningtime = 0
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result',img)
    #print('bug:',ids)

def name():
    path = './faces/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


cap=cv2.VideoCapture('1.mp4')
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
print(names)
