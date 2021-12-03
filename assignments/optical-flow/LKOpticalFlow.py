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
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))


