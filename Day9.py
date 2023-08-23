import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Median Filtering

img=cv.imread('cameraman.jpg')
girl=cv.imread('girl.jpg')

def addSaltAndPepperNoise(img):
    s_and_p = np.random.rand(img.shape[0], img.shape[1])

    # if we consider 5% salt and pepper noise, we'd like to have
    # 2.5% salt and 2.5% pepper. thus:
    salt = s_and_p > .975
    pepper = s_and_p < .025

    # in order to add some noise, we should turn off black (pepper) locations and
    # turn on white (white) locations.
    channel_2 = np.atleast_1d(img[:, :, 1])
    noisy = np.zeros_like(channel_2)

    for i in range(channel_2.shape[0] * channel_2.shape[1]):
        if salt.ravel()[i] == 1:
            noisy.ravel()[i] = 255
        elif pepper.ravel()[i] == 1:
            noisy.ravel()[i] = 0
        else:
            noisy.ravel()[i] = channel_2.ravel()[i]
    return noisy

def RemoveMedianFiltering(img):

    # apply median filter with size 3
    Median3 = cv.medianBlur(img, 3)
    Median5 = cv.medianBlur(img, 5)

    # Display the results
    fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(131), plt.xticks([]), plt.yticks([])
    plt.title("Salt And Pepper Noise")
    plt.imshow(img, cmap='gray'), plt.grid(False)
    plt.subplot(132), plt.xticks([]), plt.yticks([])
    plt.title("3x3")
    plt.imshow(Median3, cmap='gray'), plt.grid(False)
    plt.subplot(133), plt.xticks([]), plt.yticks([])
    plt.title("5x5")
    plt.imshow(Median5, cmap='gray'), plt.grid(False)
    plt.show()

# NoisyImage=addSaltAndPepperNoise(img)
# RemoveMedianFiltering(NoisyImage)
# For Image clean Image

# RemoveMedianFiltering(girl)
# For Noisy Image


def GaussianFiltering(ref):
    mean = 0
    sigma = 20.0
    gauss_noise = np.random.normal(mean, sigma, (ref.shape[0], ref.shape[1]))

    # Convert RGB image to Grayscale image using cvtColor()
    gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)

    # Add gaussian noise to the image
    g_noisy = gray + gauss_noise  # Gaussian noisy image

    # Showing gray image, noise image, and noisy image
    fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(131), plt.xticks([]), plt.yticks([])
    plt.imshow(gray, cmap='gray'), plt.grid(False)
    plt.subplot(132), plt.xticks([]), plt.yticks([])
    plt.imshow(gauss_noise, cmap='gray'), plt.grid(False)
    plt.subplot(133), plt.xticks([]), plt.yticks([])
    plt.imshow(g_noisy, cmap='gray'), plt.grid(False)
    plt.show()

# GaussianFiltering(img)

# For Bright we use  log
# For Dark we use antilog
# For Same we use identiy


# Image Arithmetics
import matplotlib.pyplot as plt
import numpy as np

def ImageArithmetics():
    skull = cv.imread('Skull.PNG')
    skullMask = cv.imread('skullMask.PNG')

    # plt.figure(figsize=(10, 10))
    # plt.subplot(131), plt.imshow(skull, cmap='gray')
    # plt.subplot(132), plt.imshow(skullMask - 20, cmap='gray')
    # plt.show()
    plt.figure(figsize=(10, 10))
    plt.subplot(131), plt.imshow(skullMask - skull, cmap='gray')
    plt.subplot(132), plt.imshow(-(skullMask - skull + 128), cmap='gray')
    plt.subplot(133), plt.imshow(skullMask - skull + 128, cmap='gray')
    plt.show()

# ImageArithmetics()

def TungstenShading():
    Tungsten = cv.imread('Tungsten.PNG')
    Shading = cv.imread('shading.PNG')

    plt.figure(figsize=(10, 10))
    plt.subplot(131), plt.imshow(Tungsten, cmap='gray')
    plt.title("Original")
    plt.subplot(132), plt.imshow(np.multiply(Tungsten, 1 / Shading), cmap='gray')
    plt.title("Shaded Image")
    plt.subplot(133), plt.imshow(Shading, cmap='gray')
    plt.title("Shade")
    s=np.multiply(Tungsten, 1 / Shading)
    print(s)

    plt.show()

# TungstenShading()


fox = cv.imread('fox.jpg')
huji = cv.imread('huji.png')

plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(fox, cmap='gray')
plt.title("Original")
plt.subplot(132), plt.imshow(np.multiply(fox, 1 / huji), cmap='gray')
plt.title("Shaded Image")
plt.subplot(133), plt.imshow(huji, cmap='gray')
plt.title("Shade")
s=np.multiply(fox, 1 / huji)
print(s)
cv.imshow("sad",np.multiply(fox, 1 / huji))

cv.waitKey(0)


