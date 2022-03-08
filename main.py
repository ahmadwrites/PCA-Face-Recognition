import cv2
import numpy as np
import face_recognition
import os
import sys


def readImages(path):
    print("Reading images from " + path, end = "...")
    # Create array of images.
    images = []
    classNames = []
    # List all files in the directory and read points from text files one by one.
    for filePath in os.listdir(path):
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
    '''
    INITIALIZATION
    '''
    PATH = 'faces'
    NUM_EIGEN_FACES = 30

    # Read the images
    images, classNames = readImages(PATH)

    avg_face_vector = images.mean(axis=1)
    avg_face_vector = avg_face_vector.reshape(images.shape[0], 1)
    normalized_face_vector = images - avg_face_vector

    print("Calculating PCA...", end=" ")
    covariance_matrix = np.cov(np.transpose(normalized_face_vector))
    # print(covariance_matrix)

    # Find eigen values and eigen vectors, then sort them
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    idx = eigen_values.argsort()[::-1]
    sorted_eigen_values = eigen_values[idx]
    sorted_eigen_vectors = eigen_vectors[:, idx]
    # print(eigen_vectors.shape)

    k_eigen_vectors = sorted_eigen_vectors[0:NUM_EIGEN_FACES, :]
    # print(k_eigen_vectors.shape)

    eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
    # print(eigen_faces.shape)

    weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
    # print(weights)

    print("Done.")

    '''
    TEST IMAGE RECOGNITION 
    '''

    test_url = "test/test17.jpg"
    test_img = cv2.imread(test_url)

    print("Searching for face location...", end=" ")
    detected_test_face = face_recognition.face_locations(test_img)
    print("Done.")
    if len(detected_test_face) == 0:
        print("No face detected.")
        sys.exit()

    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    test_img = cv2.resize(test_img, (100, 100))
    test_img = test_img.reshape(10000, 1)
    test_normalized_face_vector = test_img - avg_face_vector
    test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))

    print("Determining match...", end=" ")
    # print(np.linalg.norm(test_weight - weights, axis=1))

    # Euclidean distance
    # print((np.linalg.norm(test_weight - weights, axis=1)))

    index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
    print("Done.")

    # Match the detected index with the correct face/emotion
    if len(classNames[index].split(" - ")) == 2:
        matchingImage, matchingEmotion = classNames[index].split(" - ")
        print(f"Matching face: {matchingImage}")
        print(f"Matching emotion: {matchingEmotion}")
    else:
        print(f"Matching face: {classNames[index]}")

    print("Press any key on the photo window to quit the program.")
    # print(classNames)
    # print(index)

    '''
    SHOW BOTH IMAGES
    '''

    test_img_show = cv2.imread(test_url)
    test_img_show = cv2.resize(test_img_show, (300, 300))
    cv2.imshow('Test Image', test_img_show)

    myList = os.listdir(PATH)
    # print(myList)
    result_img_show = cv2.imread(f"faces/{myList[index]}")
    result_img_show = cv2.resize(result_img_show, (300, 300))
    cv2.imshow('Match Image', result_img_show)
    cv2.waitKey(0)

