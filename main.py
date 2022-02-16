import cv2
import numpy as np
import face_recognition
import os
import sys


def readImages(path):
    print("Reading images from " + path, end = "...")
    # Create array of array of images.
    images = []
    # List all files in the directory and read points from text files one by one.
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:

            # Add to array of images.
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)

            if im is None:
                print("image:{} not read properly".format(imagePath))
            else :
                im = cv2.resize(im, (100, 100))
                # Convert image to floating point.
                im = np.float32(im)/255.0
                # Add image to list.
                images.append(im)
                # Flip image.
                imFlip = cv2.flip(im, 1);
                # Append flipped image.
                images.append(imFlip)

    numImages = int(len(images) / 2)

    # Exit if no image found.
    if numImages == 0 :
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images


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
    NUM_EIGEN_FACES = 10

    # Read the images
    images = readImages(PATH)

    # Size of images
    sz = images[0].shape

    # Create the data matrix, each row represents 1 image
    data = createDataMatrix(images)

    # Compute the eigenvectors from the stack of images created.
    print("Calculating PCA ", end="...")

    # Get the mean and eigenVectors from PCA calculation
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)

    print("DONE")



