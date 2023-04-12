import cv2
import os

if __name__ == '__main__':
    PATH = '/home/andang/workspace/Computational_Vision/HL2/Hololens-Project/datasets/an_place/raw-04-12-08-33/depth_ahat/'
    im_paths = sorted(os.listdir(PATH))
    print(len(im_paths))

    for im_path in im_paths:
        im = cv2.imread(PATH+im_path)
        cv2.imshow('Visual', im)
        cv2.waitKey(100)
