import cv2 as cv

# imread to read image from the hard drive. imread('url')
# imwrite to write image in hard disk. imwrite()
# imshow // Show Image
# Image !!
def readImage():

    im=cv.imread('pic.jpg')
    print(type(im))
    print(im.shape)
    cv.imshow('Zayyan',im)
    cv.waitKey(0)
#readImage()

# Video
def readVideo():
    frames=cv.VideoCapture('video.mp4')

    while True:
        isTrue,frame=frames.read()

        if isTrue:
            cv.imshow('video',frame)
            if cv.waitKey(20) and 0xFF==ord('d'):
                break
        else:
            break
    frames.release()

#readVideo()


# Convert BGR to GrayScale

def convertImageFromBGRToGray():
    im=cv.imread('pic.jpg')
    print("Shape before conversion ",im.shape)
    grayimg=cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    print("Shape After conversion ",grayimg.shape)
    cv.imshow('Zayyan',grayimg)
    cv.waitKey(0)
#convertImageFromBGRToGray()

# CannyEdgeDetection

import matplotlib.pyplot as plt
def EdgeDetection():
    im = cv.imread('pic.jpg')
    grayimg=cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    im_edge=cv.Canny(grayimg,50,100)

    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.title('Original Image')

    plt.subplot(1,3,2)
    plt.imshow(grayimg,cmap='gray')
    plt.title('Gray Scale Image')
    plt.subplot(1, 3, 3)
    plt.imshow(im_edge, cmap='gray')
    plt.title('Edge Detection Image')

    plt.show()
# EdgeDetection()

# Histogram - To tell wheather the picture is high contrast or low contrast
# High contrast is desirable

def Histogram():
    import matplotlib.pyplot as plt
    import numpy as np
    im = cv.imread('pic.jpg')
    histogram=cv.calcHist([im],[0],None,[256],[0,255]) # 0 kro to blue-- 1 kro to Green
    plt.figure(figsize=(8,5))
    plt.bar(np.arange(256), histogram.flatten(), width=1)
    plt.title('Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0,256])
    plt.show()

Histogram()