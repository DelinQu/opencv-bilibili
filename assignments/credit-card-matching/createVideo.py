import cv2.cv2 as cv
import numpy as np


def cv_show(name: str, img: np.ndarray) -> None:
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    img1 = cv.imread('/home/qdl/Downloads/cup_hidden.png')
    img2 = cv.imread('/home/qdl/Downloads/mountain_hidden.png')
    cv_show('cup_hidden',img1)
    cv_show('mountain_hidden',img2)

    SIZE = (400, 400)
    # img1 = cv.resize(img1,SIZE)
    # img2 = cv.resize(img2, SIZE)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('video.avi', fourcc, 20.0, SIZE)

    for i in range(100):
        frame = img1 if (i % 40 < 20) else img2
        cv.imshow('Recording...', frame)
        out.write(frame)
        if cv.waitKey(100) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    out.release()