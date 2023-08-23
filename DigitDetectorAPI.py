from flask import Flask,request, jsonify
import numpy as np
from PIL import Image
from keras.models import load_model
app = Flask(__name__)

def saveImage():
    import cv2 as cv
    from keras.datasets import mnist
    (trainData, trainLbl), (testData, testLbl) = mnist.load_data()
    cv.imwrite('testImage2.png', testData[2])
# saveImage()

def callAPI():
    @app.route('/hello', methods=['POST'])
    def getDigitsFromImage():
        img = request.files['file']
        print(type(img))
        model = load_model('model.hdf5')
        image = Image.open(img).convert('L')
        image = image.resize((28, 28))
        x = np.array(image)
        x = x.reshape((1, 784))
        x = x / 255.0
        prediction = model.predict(x)
        digit = np.argmax(prediction)
        print(digit)
        return str(digit)

    if __name__ == "__main__":
        app.run()
callAPI()


