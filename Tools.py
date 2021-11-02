import os
import cv2

from scipy import linalg
import numpy as np


def pathFile(path):
    if os.path.exists(path):
        return path
    else:
        os.makedirs(path)
    return path


def showImage1(winname, img, k=0.25):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, int(k * img.shape[1]), int(k * img.shape[0]))
    cv2.imshow(winname, img)
    cv2.waitKey(0)

def showImage(img, k=0.25):
    winname = 'default'
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, int(k * img.shape[1]), int(k * img.shape[0]))
    cv2.imshow(winname, img)
    cv2.waitKey(0)

def gamma(img, gama):
    img_norm = img / 255.0
    img_gamma = np.power(img_norm, gama) * 255.0
    img_gamma = img_gamma.astype(np.uint8)
    return img_gamma


def brightness_alpha(gray_img):

    r, c = gray_img.shape
    piexs_sum = r * c

    dark_points = (gray_img < 210)
    target_array = gray_img[dark_points]
    dark_sum = target_array.size

    return dark_sum/(piexs_sum)


def connectedDomainAnalysis(img, th, thx, thy):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    stats = np.delete(stats, 0, axis=0)
    for istat in stats:
        if istat[4] < th or (istat[2] < thx or istat[3] < thy):
            cv2.rectangle(img, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)

    return img

def connectedDomainAnalysis_accly(img, th, thx, thy, pt = 0):
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    stats = np.delete(stats, 0, axis=0)
    nb_components = nb_components - 1
    if pt:
        print(stats)
    for i in range(0, nb_components):
        istat = stats[i]
        if istat[4] < th or (istat[2] < thx or istat[3] < thy):
            img[labels == i + 1] = 0
    return img

    
def find_up_yp(h_list, width):
    w_sum = [np.sum(h_list[i: i + width]) for i in range(h_list.shape[0] - width)]
    index, _ = max(enumerate(w_sum), key=lambda x: x[1])
    return index


def find_down_yp(h_list, width):

    w_sum = [np.sum(h_list[i : i + width]) for i in reversed(range(h_list.shape[0]-width))]
    index, _ = max(enumerate(w_sum), key=lambda x: x[1])
    return h_list.shape[0] - index


def creat_kernel():
    kernel = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
         [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], np.uint8)
    return kernel

def gan(matrix, pointLis):
    res = []
    for point in pointLis:
        point = np.hstack((point,np.array([1])))
        res.append(np.dot(matrix,point).tolist())
    return res

def inverse_imgRotate(coordinate, matRotate):
    coefficient_matrix = matRotate[:,:2]
    constant_matrix = np.array([[coordinate[0],coordinate[1]]]).T - matRotate[:,-1:]
    inverse_coord = linalg.solve(coefficient_matrix, constant_matrix)
    return (int(inverse_coord[0]), int(inverse_coord[1]))


def main():
    import glob
    import shutil
    day_path = 'D:/Documents/projects/data/20210920_20210915/up/'
    vehicles_path = glob.glob(day_path + '20210915*')
    for v_path in vehicles_path:
        pans_path = glob.glob(v_path + '/*_2500W102*.jpg')
        for p_path in pans_path:
            shutil.copyfile(p_path, 'D:/Documents/projects/data/test20210920_20210915/{}.jpg'.format(p_path.split('\\')[-1]))


if __name__ == '__main__':
    main()