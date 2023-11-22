import numpy as np

import cv2 as cv
import os

directory = r'Video_filter_JN\background_images' 
# iterate over files in
# that directory
count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    test_file = cv.imread(f)
    # print(test_file)
    Gaussian = cv.GaussianBlur(test_file,(31,31),0)
    resized = cv.resize(Gaussian, (960,540))
    cv.imwrite(directory + "/result/" + "background_{}.jpg".format(count), resized)
    count += 1