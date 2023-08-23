# Salt and Paper or Median
# Use Averaging filtering to fix this
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img=cv.imread('cameraman.jpg')

def manipulateImage(image, alpha, beta): # alpha is for contrast and beta is for brightness
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y, x] = np.clip(alpha * image[y, x] + beta, 0, 255)

    return new_image

def changeBrightness():
    bright = manipulateImage(img, 1, 30)
    dark = manipulateImage(img, 1, -30)

    plt.figure()
    plt.subplot(231), plt.imshow(dark, cmap='gray')
    plt.title('Dark')
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(bright, cmap='gray')
    plt.title('Bright')
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(234)
    plt.plot(cv.calcHist([dark], [0], None, [256], [0, 256])), plt.ylim((0, 1750))

    plt.subplot(235)
    plt.plot(cv.calcHist([img], [0], None, [256], [0, 256]))

    plt.subplot(236)
    plt.plot(cv.calcHist([bright], [0], None, [256], [0, 256]))
    plt.show()

# changeBrightness()   # Function Call


# Use Histogram Equalization to remove Over Exposure
# Convert it into high Contrast Image

def changeContrast():
    increase_contrast = manipulateImage(img, 1.35, 0)
    decrease_contrast = manipulateImage(img, 0.35, 0)

    # Compare the results
    plt.figure()
    plt.subplot(231), plt.imshow(decrease_contrast, cmap='gray')
    plt.title('Low Contrast')
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.grid(False), plt.xticks([]), plt.yticks([])


    plt.subplot(233),plt.imshow(increase_contrast, cmap='gray')
    plt.title('High Contrast')
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(234)
    plt.bar(range(256),cv.calcHist([decrease_contrast],[0],None,[256],[0,256]).ravel())
    plt.ylim((0, 1750))

    plt.subplot(235)
    plt.bar(range(256),cv.calcHist([img],[0],None,[256],[0,256]).ravel())
    plt.ylim((0, 1750))

    plt.subplot(236)
    plt.bar(range(256),cv.calcHist([increase_contrast],[0],None,[256],[0,256]).ravel())
    plt.ylim((0, 1750))
    plt.show()

changeContrast()  # Function Call
def HistogramEqualization():
    img_eq = cv.equalizeHist(img)

    grid = plt.GridSpec(3, 4, wspace=0.4, hspace=0.3)

    plt.subplot(grid[:2, :2])
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(grid[:2, 2:])
    plt.imshow(img_eq, cmap='gray')
    plt.title("Equalized Image")
    plt.grid(False), plt.xticks([]), plt.yticks([])

    plt.subplot(grid[2, :2])
    plt.bar(range(256),cv.calcHist([img], [0], None, [256], [0, 256]).ravel())

    plt.subplot(grid[2, 2:])
    plt.bar(range(256),cv.calcHist([img_eq], [0], None, [256], [0, 256]).ravel())
    plt.show()
# HistogramEqualization()   # Function Call


# Mask X x Y
# Image N x M
# Result (N-X-1)/stride x (M-Y-1)/stride

def AveragingFilter():
    # Defining a kernel using numpy.
    kernel_5 = np.ones((5, 5), np.float32) / 25
    kernel_3 = np.ones((3, 3), np.float32) / 9
    kernel_9 = np.ones((9, 9), np.float32) / 81

    # Another Way
    kernel_9UsingOtherMethod = cv.blur(img, (9, 9), -1)

    # Convolves an image with the kernel.
    # -1 means that the center of the kernel is located on the center pixel.
    # compare two kernel sizes.
    filtered_5 = cv.filter2D(img, -1, kernel_5)
    filtered_3 = cv.filter2D(img, -1, kernel_3)
    filtered_9 = cv.filter2D(img, -1, kernel_9)

    # plot the results in two subplots.
    fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(151), plt.imshow(img), plt.title('No Filter')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(152), plt.imshow(filtered_3), plt.title('3-by-3 filter')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(153), plt.imshow(filtered_5), plt.title('5-by-5 filter')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(154), plt.imshow(filtered_9), plt.title('9-by-9 filter')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(155), plt.imshow(kernel_9UsingOtherMethod), plt.title('9-by-9 filter using Simple Method')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.show()

# AveragingFilter()   # Function Call

def ImplementingPadding():
    top = 10;
    bottom = 5;
    left = 20;
    right = 5
    const = 100
    img2 = cv.copyMakeBorder(img, top, bottom, left, right,cv.BORDER_WRAP)
    img3 = cv.copyMakeBorder(img, top, bottom, left, right,cv.BORDER_REFLECT)
    img4 = cv.copyMakeBorder(img, top, bottom, left, right,cv.BORDER_REPLICATE)
    img5 = cv.copyMakeBorder(img, top, bottom, left, right,cv.BORDER_CONSTANT, const)

    # Display the images
    fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(221), plt.imshow(img2), plt.grid(False)
    plt.xticks([]), plt.yticks([]), plt.title('wrap')

    plt.subplot(222), plt.imshow(img3), plt.grid(False)
    plt.xticks([]), plt.yticks([]), plt.title('reflect')

    plt.subplot(223), plt.imshow(img4), plt.grid(False)
    plt.xticks([]), plt.yticks([]), plt.title('replicate')

    plt.subplot(224), plt.imshow(img5), plt.grid(False)
    plt.xticks([]), plt.yticks([]), plt.title('constant')

    plt.show()


# ImplementingPadding()    # Function Call

def UserDefinedKernals():
    # vertical gradient kernel
    # define a random kernel
    vertical_gd = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    # apply it.
    filter_v = cv.filter2D(img[:, :, 2], -1, vertical_gd)

    # show in a different colormap.
    plt.imshow(filter_v, cmap='gray'), plt.grid(False)
    plt.xticks([]), plt.yticks([])
    plt.show()
# UserDefinedKernals()