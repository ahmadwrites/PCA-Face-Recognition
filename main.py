import cv2
import numpy as np
import face_recognition
import os
import sys


def readImages(path):
    print("Reading images from " + path, end = "...")
    # Create array of array of images.
    images = []
    classNames = []
    # List all files in the directory and read points from text files one by one.
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:

            # Add to array of images.
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)

            if im is None:
                print("image:{} not read properly".format(imagePath))
            else:
                im = cv2.resize(im, (100, 100))
                # Convert to grayscale
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                # Reshape the shape to total pixels (10,000 because 100x100)
                im = im.reshape(10000)
                # Add image to list.
                images.append(im)
                classNames.append(os.path.splitext(filePath)[0])

    numImages = int(len(images))

    # Exit if no image found.
    if numImages == 0:
        print("No images found")
        sys.exit(0)

    images = np.asarray(images)
    images = images.transpose()

    print(str(numImages) + " files read.")
    return images, classNames


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    PATH = 'faces'
    NUM_EIGEN_FACES = 100

    # Read the images
    images, classNames = readImages(PATH)

    avg_face_vector = images.mean(axis=1)
    avg_face_vector = avg_face_vector.reshape(images.shape[0], 1)
    normalized_face_vector = images - avg_face_vector

    print("Calculating PCA...")
    covariance_matrix = np.cov(np.transpose(normalized_face_vector))
    print(covariance_matrix)

    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    print(eigen_vectors.shape)

    k_eigen_vectors = eigen_vectors[0:NUM_EIGEN_FACES, :]
    print(k_eigen_vectors.shape)

    eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
    print(eigen_faces.shape)

    weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
    print(weights)

    print("Done.")

    '''
    TEST IMAGE RECOGNITION WITH LIVE CAMERA IMPLEMENTATION
    '''

    while True:
        success, img = cap.read()

        imgC = cv2.resize(img, (0, 0), None, .25, .25)
        imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

        facesCurFrame = face_recognition.face_locations(imgC)

        for faceLoc in facesCurFrame:
            imgS = cv2.resize(img, (100, 100))
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
            imgS = imgS.reshape(10000, 1)
            imgS_normalized_face_vector = imgS - avg_face_vector
            imgS_weight = np.transpose(imgS_normalized_face_vector).dot(np.transpose(eigen_faces))
            index = np.argmin(np.linalg.norm(imgS_weight - weights, axis=1))

            # print(index.size)

            name = classNames[index].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)



