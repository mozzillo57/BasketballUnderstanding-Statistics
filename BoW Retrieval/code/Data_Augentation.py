import cv2
import numpy as np
import os

path = r'../dataset/train'
#images = []
for folder in os.listdir(path):
    for file in  os.listdir(os.path.join(path, folder)):
        #images.append(os.path.join(path, os.path.join(folder, file)))
        img = cv2.imread(os.path.join(path, os.path.join(folder, file)))
        print('ok')

        # Mirror in x direction (flip horizontally)
        imgX = np.flip(img, axis=1)

        # Mirror in y direction (flip vertically)
        imgY = np.flip(img, axis=0)

        #cv2.imshow('imgX', imgX)
        #cv2.imshow('imgY', imgY)
        #cv2.waitKey(0)
        print(folder)
        cv2.imwrite('../dataset/train_augmented/'+ folder +'/flipY' + file, imgX)
        #cv2.imwrite('../dataset/train_augmented/'+ folder +'/flipY' + file, imgY)







