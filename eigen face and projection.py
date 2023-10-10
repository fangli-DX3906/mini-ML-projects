import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image


# eigen face and face projection

name = ['Anne', 'benjamin', 'clooney', 'keanu', 'Markle', 'ryan']
picSize = (80, 60)

face = []
for i in range(len(name)):
    face.append([])
    for j in range(5):
        facename = name[i] + str(j + 1)
        face_rgb = Image.open('./FilesHomework3/%s.jpg' % facename)
        faceResize = face_rgb.resize((60, 80))
        face_gray = faceResize.convert('L')
        face[i].append(np.array(face_gray, dtype=float))

# show full pics
plt.figure(figsize=(20, 20))
for i in range(6):
    for j in range(5):
        plt.subplot(6, 5, i * 5 + j + 1)
        plt.imshow(face[i][j], cmap=plt.get_cmap('gray'))
plt.show()

# average faces
plt.figure(figsize=(24, 4))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    faceAvg = sum(face[i]) / len(face[i])
    plt.imshow(faceAvg, cmap=plt.get_cmap('gray'))
plt.show()

# first 5 eigenfaces and 20 eigenvalues
# construct data matrix
imgData = []
for i in range(6):
    for j in range(5):
        faceRhpe = face[i][j].reshape((-1,))
        imgData.append(faceRhpe)
imgData = np.asarray(imgData)

# PCA
Cmat = np.dot(imgData.transpose(), imgData)
# Cmat = np.cov(imgData.T)
value, vector = np.linalg.eig(Cmat)
vector = vector.real
value = value.real

plt.figure(figsize=(20, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    eigenFace = vector[:, i].reshape(picSize)
    plt.imshow(eigenFace, cmap=plt.get_cmap('gray'))
plt.show()

plt.plot(value[:21], marker='D', color='r')
plt.title('First 20 Eigenvalues')
plt.xlim((-1, 20))
plt.show()

# reconstructing Anne1.jpg
plt.figure(figsize=(28, 4))
K = [1, 5, 10, 15, 20, 50, 100]
Anne1 = imgData[0]
for i in range(len(K)):
    faceRecnsct = 0.0
    for j in range(K[i]):
        plt.subplot(1, len(K), i + 1)
        weight = np.dot(vector[:, j], Anne1)
        faceRecnsct += weight * vector[:, j]
    faceRecnsct = faceRecnsct.reshape(picSize)
    plt.imshow(faceRecnsct, cmap=plt.get_cmap('gray'))
plt.show()

# projection of an arbitrary photo
selfie = Image.open('1.jpg')
selfie = selfie.resize((60, 80))
selfie = selfie.convert('L')
selfie = np.array(selfie, dtype=float)
plt.imshow(selfie, cmap=plt.get_cmap('gray'))
plt.show()
selfieRshpe = selfie.reshape((-1,))

# picking the first 20th eigenvectors as vector space
V = vector[:, :20]
weights = np.dot(selfieRshpe, V)
selfieRecnsct = np.dot(V, weights.transpose())
selfieRecnsct = selfieRecnsct.reshape(picSize)
plt.imshow(selfieRecnsct, cmap=plt.get_cmap('gray'))
plt.show()

# histogram
error = []
for i in range(len(name)):
    error.append([])
    for j in range(5):
        nu = np.linalg.norm(np.dot(imgData[i * 5 + j, :], V) - np.dot(selfieRshpe, V), ord=2)
        de = np.linalg.norm(np.dot(imgData[i * 5 + j, :], V), ord=2)
        error[i].append(nu / de)

avgeError = np.mean(np.array(error), axis=1)
print(name[np.argmin(avgeError)])

plt.figure(figsize=(24, 4))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.title(name[i])
    plt.bar(range(1, 6, 1), error[i])
plt.show()