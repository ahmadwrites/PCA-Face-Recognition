# Face Recognition Using PCA
## Introduction
In this project, we use PCA to recognize the face of an unknown person captured 
through the live camera. The live camera obtains images frame by frame through OpenCV
and a face detection algorithm confirms if a face has been spotted. If so, the face
is taken and processed, then compared using different classifier algorithms e.g. 
euclidean distance and NCC in order to match it with the known faces. 

## Objectives
1. Initiation
2. Recognition
3. Display results

## Get Started
```
git clone https://github.com/ahmadwrites/PCA-Face-Recognition.git 
git chekckout pca
```

## Branches 
1. pca: Complete PCA implementation with comparing images
2. pca-camera: PCA implementation with open camera (less accurate)
3. main: Skeleton for project 
