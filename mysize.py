import cv2
import numpy as np
import Tools
import os
import glob

def getFile(path, fType='/*jpg'):
    return glob.glob(path+fType)

def mysize(img):
    img = cv2.resize(img, (3000,150))
    img1 = img[:, :760, :]
    img2 = img[:, 560:1320, :]
    img3 = img[:, 1120:1880, :]
    img4 = img[:, 1680:2440, :]
    img5 = img[:, 2240:, :]
    rimg = np.vstack((img1, img2, img3, img4, img5))
    return rimg


if __name__ == '__main__':
    Path = r'I:\pantographOnlineMonitoringItems\blockFallingDetectionDataAfterCutting(normalFigure)\up\type34\20210713'
    savePath = r'./resizedata/up'
    if os.path.exists(Path):
        imgsPath = getFile(Path)
        for imgPath in imgsPath:
            imgName = os.path.split(imgPath)[1]
            img = cv2.imread(imgPath)
            res = mysize(img)
            cv2.imwrite(savePath + '/' + imgName, res)
