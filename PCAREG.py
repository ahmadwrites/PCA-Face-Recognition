# Import required modules
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

#Define method PLT display grayscale images
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show()

#Read all pictures in a folder, input parameter is filename, return file address list
 n

def read_directory(directory_name):
    faces_addr = []
    for filename in os.listdir(directory_name):
        faces_addr.append(directory_name + "/" + filename)
    return faces_addr

# Read all face folder, save the image address in the list
faces = []
for i in range(1,42):
    faces_addr = read_directory(''+str(i))
    for addr in faces_addr:
        faces.append(addr)

# Read image data to generate list labels
    images = []
    labels = []
    for index, face in enumerate(faces):
            image = cv2.imread(face, 0)
            images.append(image)
            labels.append(int(index / 10 + 1))
    print(len(labels))
    print(len(images))
    print(type(images[0]))
    print(labels)

# Draw the last two sets of faces
#Create canvas and subgraph objects
fig, axes = plt.subplots(2,10
                       ,figsize=(15,4)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
#Fill in the image
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i+390],cmap="gray") #选择色彩的模式

# Image data transformation feature matrix
    image_data = []
    for image in images:
        data = image.flatten()
        image_data.append(data)
    print(image_data[0].shape)
# Convert to a NUMpy array
    X = np.array(image_data)
    y = np.array(labels)
    print(type(X))
    print(X.shape)
# Import the PCA module of SkLearn
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
# draw the eigenmatrix
import pandas as pd
data = pd.DataFrame(X)
data.head()
# Partition data set
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
# Training PCA model
pca=PCA(n_components=100)
pca.fit(x_train)
# Returns the test set and training set after dimensionality reduction
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print(x_train_pca.shape)
print(x_test_pca.shape)
V = pca.components_
V.shape
# 100 eigen faces
#Create canvas and subgraph objects
fig, axes = plt.subplots(10,10
                       ,figsize=(15,15)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
#Fill in the image
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i,:].reshape(112,92),cmap="gray") #选择色彩的模式
pca.explained_variance_ratio_
# Returns how much of the data carried by the feature is the original data
pca.explained_variance_ratio_.sum()
# Graph the number of eigens and the amount of information they carry
explained_variance_ratio = []
for i in range(1,151):
    pca=PCA(n_components=i).fit(x_train)
    explained_variance_ratio.append(pca.explained_variance_ratio_.sum())
plt.plot(range(1,151),explained_variance_ratio)
plt.show()
##############################################
##The images of training set and test set are projected into feature vector space, and then the clustering method (nearest neighbor or K-neighbor, etc.)
# is used to obtain the nearest image of each image in the test set, which can be classified.
##Cv2. Face. EigenFaceRecognizer_create () to create model of face recognition by image arrays and the corresponding labels to train model
##Predict () returns an array of two elements
##The first is a tag that identifies individuals,
##The second is confidence, the smaller the match, the higher the match, 0 means perfect match.
##GetEigenValues () obtained the eigenvalues
##GetEigenVectors ()
##The mean getMean ()
################################################


#Model creation and training
model = cv2.face.EigenFaceRecognizer_create()
model.train(x_train,y_train)
# predict
res = model.predict(x_test[0])
y_test[0]
# 测试数据集的准确率
ress = []
true = 0
for i in range(len(y_test)):
    res = model.predict(x_test[i])
#     print(res[0])
    if y_test[i] == res[0]:
        true = true+1
    else:
        print(i)

# 平均脸
mean = model.getMean()
print(mean)
meanFace = mean.reshape(112,92)
plt_show(meanFace)
# 降维
pca=PCA(n_components=100)
pca.fit(X)
X = pca.transform(X)
# 将所有数据都用作训练集
# 模型创建与训练
model = cv2.face.EigenFaceRecognizer_create()
model.train(X,y)
# plt显示彩色图片
def plt_show0(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
#输入图片识别
img = cv2.imread('directory_name')
plt_show0(img)
print(img.shape)
# 灰度处理
img = cv2.imread('directory_name',0)
plt_show(img)
imgs = []
imgs.append(img)
# 特征矩阵
image_data = []
for img in imgs:
    data = img.flatten()
    image_data.append(data)
test = np.array(image_data)
test.shape
# 用训练好的pca模型给图片降维
test = pca.transform(test)
test[0].shape
res = model.predict(test)
res
print('人脸识别结果：',res[0])