import numpy as np
import cv2.cv2 as cv

class Sticher:
    '''
    global consts
    '''
    HOM_MIN_NUM = 4
    RATIO = 0.75

    '''
    show image
    '''
    def cv_show(self,name:str,img:np.ndarray)->None:
        cv.imshow(name,img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    '''
    draw matche points and lines
    '''
    def showMatches(self,img1,img2,kp1, kp2, matches):
        img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.cv_show('match', img3)

    '''
    detect features
    '''
    def detect(self,img:np.ndarray):
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        bimg1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        kps, des = sift.detectAndCompute(bimg1, None)
        # kps = np.float32([kp.pt for kp in kps])
        return (kps,des)

    '''
    match the key points
    '''
    def match(self, des1, des2):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < self.RATIO * n.distance:
                good.append(m)
        return good

    '''
    find Homography matrix
    '''
    def findHomography(self,matches,kp1, kp2):
        if len(matches) > self.HOM_MIN_NUM:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)

        return M

    '''
    image stiching
    '''
    def imgStich(self,M,img1,img2):
        ret_img = cv.warpPerspective(img1,M,(img1.shape[1]+img2.shape[1],img1.shape[0]))
        ret_img[0:img2.shape[0],0:img2.shape[1]] = img2
        return ret_img





