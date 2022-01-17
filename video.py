import os
import cv2
path = './style_mixing/'

videowriter = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 60, (646, 646))
for fn in sorted(os.listdir(path)):
    img = cv2.imread(path + fn)
    videowriter.write(img)