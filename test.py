import numpy as np

import cv2 as cv

test_file = cv.imread(r'Video_filter_JN\background_images\pexels-aleksandar-pasaric-1619569.jpg')
# print(test_file)
Gaussian = cv.GaussianBlur(test_file,(21,21),0)
resized = cv.resize(Gaussian, (300,300))
cv.imshow('test',resized)
cv.waitKey(0)
cv.destroyAllWindows()
          