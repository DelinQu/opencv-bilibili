import cv2.cv2 as cv
import numpy as np

class CardORC:
    def __init__(self):
        pass

    # size of template
    IMAGE_SIZE = (800,500)
    TEMPLATE_SIZE = (57, 88)
    MAX_NUM = 16


    # the type of credit card
    FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }

    # show a image
    def cv_show(self,name:str,img:np.ndarray) -> None:
        cv.imshow(name,img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # basic image process include gray and binary
    def ImageProcess(self,impath:str,mode) -> np.ndarray:
        img = cv.imread(impath)
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 170, 255, mode)
        return img, thresh

    def findBoxContour(self,bimg:np.ndarray,model)->list:
        contours, hierarchy = cv.findContours(bimg, model, cv.CHAIN_APPROX_SIMPLE)
        boxes = [cv.boundingRect(item) for item in contours]
        return boxes

    # sort the boxes and return box template
    def boxTemplate(self,boxes:tuple,bimg:np.ndarray,method="L2R")->dict:
        reverse, i = False, 0
        if method in ("R2L", "B2T"):
            reverse = True
        if method in ("T2B", "B2T"):
            i = 1

        # find the minimal box (x,y,W,H)
        ordBoxes = sorted(boxes,key=lambda box: box[i], reverse=reverse)

        # map the idx -> contour
        template = []
        for i, item in enumerate(ordBoxes):
            x, y, w, h = item
            temp = bimg[y:y+h,x:x+w]
            template.append(cv.resize(temp,self.TEMPLATE_SIZE))

        return template

    # morphologyEx
    def morphologyEx(self,bimg:np.ndarray) -> np.ndarray:
        RecKernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 2))
        RecKernel2 = cv.getStructuringElement(cv.MORPH_RECT, (8, 5))
        opening = cv.morphologyEx(bimg, cv.MORPH_OPEN, RecKernel,iterations=1)
        dilation = cv.dilate(opening, RecKernel2, iterations=3)
        return dilation

    # select the reasonable box and sort them
    def boxfilter(self,boxes:list) -> list:
        newbox = []
        for item in boxes:
            x, y, w, h = item
            ar = w / float(h)
            if 2.85 < ar < 3.1 and (135 < w < 160) and (35 < h < 60):
                newbox.append((x, y, w, h))

        return sorted(newbox,key=lambda item:item[0])

    def digfilter(self,boxes:list) -> list:
        boxes = sorted(boxes,key= lambda item: item[2]*item[3], reverse=1)[0:self.MAX_NUM]
        return sorted(boxes,key= lambda item: item[0])

    # match the box
    def match(self,bimg:np.ndarray,boxes:list,template:list) -> list:
        ret = []
        for (x, y, w, h) in boxes:
            scores = []
            img = bimg[y:y+h,x:x+w]
            img = cv.resize(img,self.TEMPLATE_SIZE)
            for item in template:
                res = cv.matchTemplate(img, item, cv.TM_CCOEFF)
                scores.append(cv.minMaxLoc(res)[1])
            ret.append(np.argmax(scores))
        return ret

    # show the boxed orignal image
    def showRectangle(self,name:str,img:np.ndarray,boxes:list):
        boxImg = img.copy()
        for box in boxes:
            x, y, w, h = box
            cv.rectangle(boxImg, (x, y), (x + w, y + h), (0, 255, 0))

        self.cv_show(name, boxImg)

    def showText(self,name:str,img:np.ndarray,boxes:list,num:list):
        boxImg = img.copy()
        for i,box in enumerate(boxes):
            x, y, w, h = box
            cv.putText(boxImg, str(num[i]),(x, y-15),cv.FONT_HERSHEY_SIMPLEX,
                      0.7,(0, 0, 255),2)

        self.cv_show(name, boxImg)
