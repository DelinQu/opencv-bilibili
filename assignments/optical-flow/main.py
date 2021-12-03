import argparse
from LKOpticalFlow import LKOpticalFlow

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,help="path to input image")
    args = vars(ap.parse_args())

    LK = LKOpticalFlow()
    LK.LKvideo(args['video'])