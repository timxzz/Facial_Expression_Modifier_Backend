from flask import Flask, request, send_file
from flask import make_response 
import numpy as np
import scipy.misc as scm
import cv2
from io import BytesIO
from PIL import Image
from base64 import b64encode
from flask import jsonify

import os
import tensorflow as tf
import scipy.misc as scm
from StarGAN import StarGAN


class CustomError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

app = Flask(__name__)

@app.route("/")
def hello():
    return "I am online!"

@app.route('/image', methods=['POST'])
def gen_image():
    print("Received request")
    img_file = request.files.get('image')
    img_array = np.fromstring(img_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    print('Find face')
    try:
      face = find_face(img)
    except:
      raise CustomError("Couldn't find any face in the image.", status_code=400)

    print('Run Model')
    try:
      img_out = run_StarGAN(face)
    except:
      raise CustomError("Error occurred while running the model.", status_code=500)

    img_out.insert(0, face)

    exprID=[-1,0,1,2,3,4,5,6]
    exprName=['null', 'neutral', 'happy', 'sad', 'surprised', 'disgusted', 'angry', 'fearful']
    js = []
    encoded_img = []
    for i in range(8):
      pil_img = Image.fromarray(img_out[i].astype('uint8'))
      buff = BytesIO()
      pil_img.save(buff, format="png")
      encoded_img.append(b64encode(buff.getvalue()).decode("utf-8"))
      js.append({"expressionId": exprID[i],
                 "expressionName": exprName[i],
                 "image": "data:image/png;base64," + encoded_img[i]})

    res = make_response(jsonify( js ))
    res.headers.set('Access-Control-Allow-Origin', '*')
    return res

@app.errorhandler(CustomError)
def handle_custom_error(error):
    response = make_response(jsonify(error.to_dict()))
    response.status_code = error.status_code
    response.headers.set('Access-Control-Allow-Origin', '*')
    return response



def run_StarGAN(input_image):
    ''' config settings '''

    project_name = "StarGAN_Face_1_"
    train_flag = False

    '''-----------------'''

    # gpu_number = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" #args.gpu_number

    # with tf.device('/gpu:{0}'.format(gpu_number)):
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    #     config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            model = StarGAN(sess, project_name)
            out_images = model.test(input_image, train_flag)
        tf.reset_default_graph()

    return out_images

def find_face(input_image):

    faceDet = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("./face_detection_model/haarcascade_frontalface_alt_tree.xml")

    out_size = 128

    frame = input_image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_for_detection = gray

    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(color_for_detection, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        raise Exception("No face detected in the image.")

    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        color = frame[y:y+h, x:x+w] #Cut the frame to size
        out = cv2.resize(color, (out_size, out_size)) #Resize face so all images have same size

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    return out

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
