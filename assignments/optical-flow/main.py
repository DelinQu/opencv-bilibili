import cv2.cv2 as cv
import numpy as np
import argparse
from LKOpticalFlow import LKOpticalFlow

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,help="path to input image")
    args = vars(ap.parse_args())

    LK = LKOpticalFlow()

    # img = cv.imread('/home/qdl/Pictures/head2.jpg')
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #
    # sift = cv.SIFT_create()
    # kp = sift.detect(gray, None)
    # sift_img = cv.drawKeypoints(gray, kp, img)
    # LK.cv_show('orignal',img)
    #
    # corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
    # sift_img = cv.drawKeypoints(gray, corners, img.copy())