import numpy as np
import cv2.cv2 as cv

class LKOpticalFlow:
    '''
    params
    '''
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    '''
    show images
    '''
    def cv_show(self,name,img):
        cv.imshow(name,img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    '''
    LK flow video
    '''
    def LKvideo(self,vpath):
        cap = cv.VideoCapture(vpath)
        ret, old_frame = cap.read()
        prevImg = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        prevPts = cv.goodFeaturesToTrack(prevImg, mask=None, **self.feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        while(True):
            ret, frame = cap.read()
            if(not ret):
                break
            nextImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            nextPts, status, err = cv.calcOpticalFlowPyrLK(prevImg,nextImg,prevPts,None,**self.lk_params)

            # Select good points
            if nextPts is not None:
                good_old = prevPts[status == 1]
                good_new = nextPts[status == 1]

            # draw the tracks
            for i,(old,new) in enumerate(zip(good_old,good_new)):
                xo, yo = old
                xn, yn = new

                mask = cv.line(mask, (int(xo), int(yo)), (int(xn), int(yn)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(xn), int(yn)), 5, color[i].tolist(), -1)

            img = cv.add(frame, mask)
            cv.imshow('frame', img)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            prevImg = nextImg.copy()
            prevPts = good_new.reshape(-1, 1, 2)

        cv.destroyAllWindows()