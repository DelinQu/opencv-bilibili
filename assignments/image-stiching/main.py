import argparse
from Sticher import Sticher
import cv2.cv2 as cv

if __name__ == '__main__':
    # add arguments by argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--imageA', required=True,help='path to input image A')
    ap.add_argument('-b', '--imageB',required=True,help='path to input image B')
    args = vars(ap.parse_args())

    img2 = cv.imread(args['imageA'])
    img1 = cv.imread(args['imageB'])

    stich = Sticher()

    # detect features
    kp1, des1 = stich.detect(img1)
    kp2, des2 = stich.detect(img2)

    # match points
    matches = stich.match(des1, des2)

    # show matches
    stich.showMatches(img1,img2,kp1, kp2, matches)

    # Homography
    M = stich.findHomography(matches,kp1,kp2)

    # image stiching
    stimg = stich.imgStich(M,img1,img2)
    stich.cv_show('stiching image',stimg)