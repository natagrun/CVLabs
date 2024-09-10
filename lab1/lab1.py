import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.line(img, (15, 20), (70, 50), color=(0, 255, 0),thickness=2)
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.circle(img,(200,200), 32,color=(255,0,0))
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.ellipse(img,(200, 200),(100,160), angle=45,startAngle=0,endAngle=360,color=(0,0,255) )
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.rectangle(img, (15, 20), (70, 50), color=(0,255,255))
    cv.imshow('df', img)
    cv.waitKey(1000)

    cv.destroyAllWindows()
