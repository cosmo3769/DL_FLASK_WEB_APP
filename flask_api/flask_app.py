from flask import Flask, render_template, request

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json


json_file = open('model.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("weights.h5")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

app = Flask(__name__)
graph = tf.Graph()

@app.route('/')
def index():
    return render_template("index.html")

import re
import base64

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    global model, graph
    
    imgData = request.get_data()
    convertImage(imgData)
   
    x = cv2.imread('output.png', 0)
   
    x = cv2.resize(x, (28, 28))
   
    x = x.reshape(1, 28, 28, 1)

    # perform the prediction
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.argmax(out, axis=1)
    return str(response[0])


if __name__ == "__main__":

    # run the app locally on the given port
    app.run(host='0.0.0.0', port=80)
# optional if we want to run in debugging mode
    app.run(debug=True)
