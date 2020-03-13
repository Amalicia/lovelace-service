import numpy as np
import tensorflow as tf
import flask
import base64
import io

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
	model = tf.keras.models.load_model('model.h5', custom_objects={"fbeta": fbeta})
	return model


def process_image(image, target_size):
	if image.mode != "L":
		image = image.convert("L")

	image = image.resize(target_size)
	image = tf.keras.preprocessing.image.img_to_array(image)
	return image


keras_model = getModel()

def make_predictions(model, image):
	inv_map = {v: k for k, v in TAG_MAPPING.items()}

	prediction = model.predict(image)
	rounded_pred = prediction.round()
	image_tags = [inv_map[i] for i in range(len(rounded_pred)) if rounded_pred[i] == 1.0]
	return image_tags


@app.route('/predict', methods=['POST'])
def predict():
	message = flask.request.get_json(force=True)
	encoded = message['blob']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_img = process_image(image, target_size=(224, 224))

	prediction_result = make_predictions(keras_model, processed_img)

	response = {
		'prediction' : {
			'values': prediction_result
		}
	}
	return flask.jsonify(response)

app.run(host='0.0.0.0')
