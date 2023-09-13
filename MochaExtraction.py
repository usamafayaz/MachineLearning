import cv2 as cv
import matplotlib.pyplot as plt

simple=cv.imread("simple.jpg")
mocha=cv.imread("mocha.jpg")

test=cv.imread("pic.jpg")
new_shape = (1600, 1512)
test = test.reshape(new_shape)

plt.figure(figsize=(10, 10))
plt.subplot(141), plt.imshow(simple)
plt.subplot(142), plt.imshow(mocha)
mochaFilter =mocha-simple
plt.subplot(143), plt.imshow(mochaFilter)
plt.subplot(144), plt.imshow(test+mochaFilter)
plt.show()

