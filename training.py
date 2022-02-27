import os
import cv2
import sys
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    facesSamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    # detect face 
    # here i use a defalut classifier in opencv the path is depend where you install it 
    face_detector = cv2.CascadeClassifier('C:/Users/Ahmad/.virtualenvs/PCA_Project/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    # print arrays
    # imagePaths
    print('ordering_arrays:',imagePaths)
    #Iterate over the images in the list
    for imagePath in imagePaths:
        #open the image and make it black and white
        PIL_img=Image.open(imagePath).convert('L')
        #turn the image into arrays
       # PIL_img = cv2.resize(PIL_img, dsize=(400, 400))
        img_numpy=np.array(PIL_img,'uint8')
        #Get the face features in the image
        faces = face_detector.detectMultiScale(img_numpy)
        #get the id and name of each iamge
        id = int(os.path.split(imagePath)[1].split('.')[0])
        #prevent on face in image
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
        #print the face features and id
        #print('fs:', facesSamples)
        print('id:', id)
        #print('fs:', facesSamples[id])
    print('fs:', facesSamples)
    #print('face_example:',facesSamples[0])
    #print('id_info:',ids[0])
    return facesSamples,ids

if __name__ == '__main__':
    #image path depend on your computer's path
    path='./faces/'
    #get image arrays id and name
    faces,ids=getImageAndLabels(path)
    #get the training object
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    #recognizer.train(faces,names)#np.array(ids)
    recognizer.train(faces,np.array(ids))
    #save the train file
    #here trainer/trainer.yml can be edited whatever you want but only in .yml
    recognizer.write('trainer/trainer.yml')
    #save_to_file('names.txt',names)




