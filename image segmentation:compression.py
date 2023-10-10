import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse


# implement k-means for image segmentation/compression


class KMnsForImgSgmtation:
    def __init__(self):
        pass

    def loadImg(self, path: str, reshape=True, showImg=True):
        image = Image.open(path).convert("RGB")
        if showImg:
            plt.imshow(image)
            plt.show()
        if reshape:
            imgdata = np.array(image)
            l, w, c = imgdata.shape
            self.imgSize = imgdata.shape
            self.imgResize = (l * w, c)
            self.imgdata = imgdata.reshape(self.imgResize)

    def assignToMean(self, meanVec: np.ndarray):
        distMat = []
        for mean in meanVec:
            diff = self.imgdata - np.tile(mean, (self.imgdata.shape[0], 1))
            distsq = diff ** 2
            dist = np.sum(distsq, axis=1)
            distMat.append(dist)
        distMat = np.asarray(distMat)
        thisCluster = np.argsort(distMat, axis=0)[0, :]
        return thisCluster

    def initMean(self, K):
        meanVec = []
        maxVal = np.max(self.imgdata, axis=0)
        minVal = np.min(self.imgdata, axis=0)
        for max, min in zip(maxVal, minVal):
            meanVec.append(np.random.randint(min, max, K))
        return np.asarray(meanVec).transpose()

    def meanUpdate(self, clstr: np.ndarray, K):
        meanVec = []
        idx = []
        for i in range(K):
            temp_mean = np.mean(self.imgdata[clstr == i, :], axis=0)
            meanVec.append(temp_mean)
            if np.isnan(temp_mean).any():
                idx.append(True)
            else:
                idx.append(False)
        meanVec = np.asarray(meanVec)
        meanVec[idx, :] = np.mean(meanVec[np.logical_not(idx), :], axis=0)
        return meanVec

    def Kmeans(self, K, epsilon=1e-3, maxIter=500):
        diff = 1
        iter = 0
        meanVec = self.initMean(K)
        while diff >= epsilon and iter <= maxIter:
            iter += 1
            thisCluster = self.assignToMean(meanVec)
            meanVecNew = self.meanUpdate(thisCluster, K)
            diff = np.sum((meanVec - meanVecNew) ** 2)
            meanVec = meanVecNew
        self.meanVec = meanVec
        self.cluster = thisCluster

    def rebuildImg(self, newName):
        img = np.zeros(self.imgResize)
        for i in range(self.meanVec.shape[0]):
            img[self.cluster == i, :] = self.meanVec[i]
        img = np.asarray(img)
        img = img.reshape(self.imgSize).astype(np.uint8)
        # print(img.shape)
        plt.imshow(img)
        plt.show()
        plt.imsave(newName, img)


# command line params
parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('K', type=int)
parser.add_argument('newName', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    image = KMnsForImgSgmtation()
    image.loadImg(args.name)
    image.Kmeans(K=args.K)
    image.rebuildImg(newName=args.newName)

    image.loadImg('113.png')
    image.Kmeans(2)
    image.rebuildImg('111_.png')
