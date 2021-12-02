import argparse
import cv2.cv2 as cv
import numpy as np

from CardOCR import CardORC

if __name__ == '__main__':
    # add arguments by argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,help="path to input image")
    ap.add_argument("-t", "--template", required=True,help="path to template OCR-A image")
    args = vars(ap.parse_args())

    orc = CardORC()

    # templat
    img, bimg = orc.ImageProcess(args['template'], cv.THRESH_BINARY_INV)
    boxes = orc.findBoxContour(bimg,cv.RETR_EXTERNAL)
    orc.showRectangle('template',img,boxes)
    template = orc.boxTemplate(boxes,bimg)

    # image
    img, bimg = orc.ImageProcess(args['image'], cv.THRESH_BINARY)
    img, bimg = cv.resize(img,orc.IMAGE_SIZE), cv.resize(bimg,orc.IMAGE_SIZE)

    # get the area of numbers
    tophat = orc.morphologyEx(bimg)
    boxes = orc.findBoxContour(tophat,cv.RETR_TREE)
    boxes = orc.boxfilter(boxes)
    orc.showRectangle('boxed',img,boxes)

    # create mask area
    area = np.zeros(bimg.shape,dtype='uint8')
    for (x,y,w,h) in boxes:
        area[y:y+h,x:x+w] = cv.COLOR_RGB2YUV_I420
    bimg = cv.bitwise_and(bimg,area)
    # orc.cv_show('bimg',bimg)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closing = cv.morphologyEx(bimg, cv.MORPH_CLOSE, kernel, iterations=1)

    boxes = orc.findBoxContour(closing,cv.RETR_TREE)
    boxes = orc.digfilter(boxes)
    orc.showRectangle('boxes',img.copy(),boxes)

    # match
    nums = orc.match(bimg,boxes,template)
    print(nums)
    orc.showText('Number',img,boxes,nums)








