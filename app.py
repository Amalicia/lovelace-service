import numpy as np
import tensorflow as tf
import flask
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import os

from PIL import Image

app = flask.Flask(__name__)

TAG_MAPPING = {
	'none': 0,
	'epidural': 1,
	'intraparenchymal': 2,
	'intraventricular': 3,
	'subarachnoid': 4,
	'subdural': 5
}


def fbeta(y_pred, y_true, beta=2):
	y_pred = tf.keras.backend.clip(y_pred, 0, 1)

	true_pos = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)), axis=1)
	false_pos = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)), axis=1)
	false_neg = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)), axis=1)

	precision = true_pos / (true_pos + false_pos + tf.keras.backend.epsilon())
	recall = true_pos / (true_pos + false_neg + tf.keras.backend.epsilon())

	beta_squared = beta ** 2
	fbeta_score = tf.keras.backend.mean(
		(1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + tf.keras.backend.epsilon()))
	return fbeta_score


def getModel():
	model = tf.keras.models.load_model('model_2.h5', custom_objects={"fbeta": fbeta})
	return model


def process_image(image, target_size):
	if image.mode != "L":
		image = image.convert("L")

	image = image.resize(target_size)
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image.astype('uint8')
	return image


keras_model = getModel()

def make_predictions(model, image):
	inv_map = {v: k for k, v in TAG_MAPPING.items()}

	prediction = model.predict(image)
	rounded_pred = prediction.round()
	indexes = np.where(rounded_pred == 1.0)
	image_tags = [inv_map[i] for i in indexes[0]]
	if not image_tags:
		image_tags = ["Whoops, couldn't make a prediction :/"]
	return image_tags


@app.route('/predict', methods=['POST'])
def predict():
	file = flask.request.files['file']
	basepath = os.path.dirname(__file__)
	file_path = os.path.join(basepath, secure_filename(file.filename))
	file.save(file_path)

	image = Image.open(file_path)
	processed_img = process_image(image, target_size=(224, 224))

	prediction = make_predictions(keras_model, processed_img)

	response = {
		'prediction': prediction
	}
	os.remove(file_path)
	return flask.jsonify(response)


CORS(app, expose_headers='Authorization')

app.run(host='0.0.0.0')
