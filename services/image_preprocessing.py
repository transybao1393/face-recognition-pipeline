import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from services.image_super_resolution import ImageSuperResolution


class ImagePreProcessing():
	def __init__(self, 
	    taskName = "Denoising", 
		imageURL = "", 
		className = ""
	):
		# FIXME: taskName should be support string or array incase of multiple tasks
		print(">>> [IMAGE PRE-PROCESSING] running...")
		defaultTaskList = ['Denoising', 'Dehazing_Indoor', 'Dehazing_Outdoor', 'Deblurring', 'Deraining', 'Enhancement', 'Retouching']
		self.__IMAGE_URL = imageURL
		self.__CLASS_NAME = className
		print("image url", self.__IMAGE_URL)
		if taskName in defaultTaskList:
			self.__TASK_NAME = taskName
		else:
			self.__TASK_NAME = defaultTaskList[0]

	def exec(self):
		task = self.__TASK_NAME  # @param ["Denoising", "Dehazing_Indoor", "Dehazing_Outdoor", "Deblurring", "Deraining", "Enhancement", "Retouching"]
		model_handle_map = {
			"Denoising": [
				"https://tfhub.dev/sayakpaul/maxim_s-3_denoising_sidd/1",
				"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdQ4MA1MzeQTCGASVYtDX9vMKsTAYk_SKkrYy0uKFZxnVk37I1Wd_sRLzixiTQGhNMzHE&usqp=CAU",
			],
			"Dehazing_Indoor": [
				"https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-indoor/1",
				"https://github.com/google-research/maxim/raw/main/maxim/images/Dehazing/input/0003_0.8_0.2.png",
			],
			"Dehazing_Outdoor": [
				"https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-outdoor/1",
				"https://github.com/google-research/maxim/raw/main/maxim/images/Dehazing/input/1444_10.png",
			],
			"Deblurring": [
				"https://tfhub.dev/sayakpaul/maxim_s-3_deblurring_gopro/1",
				"https://github.com/google-research/maxim/raw/main/maxim/images/Deblurring/input/1fromGOPR0950.png",
			],
			"Deraining": [
				"https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1",
				"https://github.com/google-research/maxim/raw/main/maxim/images/Deraining/input/15.png",
			],
			"Enhancement": [
				"https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_lol/1",
				"https://github.com/google-research/maxim/raw/main/maxim/images/Enhancement/input/a4541-DSC_0040-2.png",
			],
			"Retouching": [
				"https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_fivek/1",
				"https://github.com/google-research/maxim/raw/main/maxim/images/Enhancement/input/a4541-DSC_0040-2.png",
			],
			"Super_Resolution": [
				"https://tfhub.dev/captain-pool/esrgan-tf2/1",
				"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAeNcI50g132EHUjNeiHFxHGQo26Zi36m8hEgH3QqowPDQgmJymHeeNNQ5q4jfzldJIvM&usqp=CAU",
			],
		}


		if task == "Super_Resolution":
			ImageSuperResolution(self.__CLASS_NAME + ".png", "./services/models/esrgan-super-resolution-model").exec()
		else:
			model_handle = model_handle_map[task]
			ckpt = model_handle[0]
			# image_url = model_handle[1] if self.__IMAGE_URL == "" else self.__IMAGE_URL
			image_path = tf.keras.utils.get_file(origin=model_handle[1]) if self.__IMAGE_URL == "" else self.__IMAGE_URL
			print(f"TF-Hub handle: {ckpt}")

			input_resolution = (256, 256)
			model = self.get_model(ckpt, input_resolution)
			final_pred_image = self.infer(image_path, model, input_resolution)
			
			# using PIL to save image
			final_pred_image = Image.fromarray((final_pred_image * 255).astype(np.uint8))
			final_pred_image.save(self.__CLASS_NAME + ".png") # save to processed image folder
		

	# Since the model was not initialized to take variable-length sizes (None, None, 3),
	# we need to be careful about how we are resizing the images.
	# From https://www.tensorflow.org/lite/examples/style_transfer/overview#pre-process_the_inputs
	def resize_image(self, image, target_dim):
		# Resize the image so that the shorter dimension becomes `target_dim`.
		shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
		short_dim = min(shape)
		scale = target_dim / short_dim
		new_shape = tf.cast(shape * scale, tf.int32)
		image = tf.image.resize(image, new_shape)

		# Central crop the image
		image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
		return image


	def process_image(self, image_path, target_dim=256):
		input_img = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
		input_img = tf.expand_dims(input_img, axis=0)
		input_img = self.resize_image(input_img, target_dim)
		return input_img
	
	def get_model(self, model_url: str, input_resolution: tuple) -> tf.keras.Model:
		inputs = tf.keras.Input((*input_resolution, 3))
		hub_module = hub.KerasLayer(model_url)
		outputs = hub_module(inputs)
		return tf.keras.Model(inputs, outputs)


	# Based on https://github.com/google-research/maxim/blob/main/maxim/run_eval.py
	def infer(self, image_path: str, model: tf.keras.Model, input_resolution=(256, 256)):
		preprocessed_image = self.process_image(image_path, input_resolution[0])

		preds = model.predict(preprocessed_image)
		if isinstance(preds, list):
			preds = preds[-1]
			if isinstance(preds, list):
				preds = preds[-1]

		preds = np.array(preds[0], np.float32)
		final_pred_image = np.array((np.clip(preds, 0.0, 1.0)).astype(np.float32))
		return final_pred_image
