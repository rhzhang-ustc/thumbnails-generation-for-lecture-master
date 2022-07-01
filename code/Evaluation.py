import cv2
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
from fast_glcm import fast_glcm_entropy, fast_glcm_homogeneity, fast_glcm_std

# 提供对thumbnail的质量评价，包括明亮度、锐度、色彩的多样性、纹理、图像结构


def Thumbnail_Evaluation(path):
    print("target video :" + path)
    cap = cv2.VideoCapture(str(path))

    success, frame = cap.read()
    luminous = []
    sharpness = []
    uniformity = []
    texture = []

    while success:
        luminous.append(Luminance(frame))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness.append(Sharpness(frame_gray))
        uniformity.append(Uniformity(frame_gray))
        texture.append(fast_glcm_entropy(frame_gray, 0, 255, 8, 5))

        break

        success, frame = cap.read()


def Luminance(frame):
    b, g, r = cv2.split(frame)
    return np.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def Sharpness(frame_gray):
    return cv2.Laplacian(frame_gray, cv2.CV_64F).var()


def Uniformity(frame_gray):
    shape = frame_gray.shape
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])/(shape[0] * shape[1])
    hist = list(hist)
    hist.sort(reverse=True)
    hist = np.array(hist)
    return np.sum(hist[0:13])


if __name__ == "__main__":
    Thumbnail_Evaluation("./data/video/ponki_thumbnails.mp4")