import cv2


def Open_JPG_OpenCV(path):
    img = cv2.imread(path, 1)
    return img


import PIL.Image as image


def Open_JPG_PIL(path):
    img = image.open(path)
    row, col = img.size
    return img


