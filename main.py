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


# Create data matrix from a list of images.
def createDataMatrix(images):
    print("Creating data matrix", end = " ... ")

    numImages = len(images)
    sz = images[0].shape
    # Data matrix.
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype = np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        # Each row get replaced with one flattened image.
        data[i,:] = image

    print("DONE")
    return data


if __name__ == '__main__':
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

    test_img = cv2.imread("test/test_bill.jpg")

    print("Searching for face location...")
    detected_test_face = face_recognition.face_locations(test_img)
    if len(detected_test_face) == 0:
        print("No face detected.")
        sys.exit()

    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    test_img = cv2.resize(test_img, (100, 100))
    test_img = test_img.reshape(10000, 1)
    test_normalized_face_vector = test_img - avg_face_vector
    test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))

    print("Determining match...")
    print(np.linalg.norm(test_weight - weights, axis=1))
    index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))

    print(f"Matching image: {classNames[index]}")



