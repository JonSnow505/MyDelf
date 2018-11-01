# Import Flask (and some helpers), create `app` object
from flask import Flask, request, jsonify
import skimage.io as io
app = Flask(__name__)

# Tell `app` that if someone asks for `/` (which is the main page)
# then run this function, and send back the return value
@app.route("/")
def hello():
    return "Hello World!"

# We'll support /resnet
@app.route("/resnet", methods=['POST'])
I = '456'
def myfunc():
    data = request.files['file']
    I = io.imread(data)
    io.imsave('image.jpg', I)
    return jsonify({'result': 123})

app.run(host='0.0.0.0', port=5000)
