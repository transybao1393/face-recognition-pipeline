from ..pipeline import Pipeline
import timeit
import shutil
from services.image_augemented import ImageAugmented
from services.image_preprocessing import ImagePreProcessing
import sys
import os
import tensorflow as tf
import numpy as np
from ...mtcnn.code.align import detect_face
from ...mtcnn.code import facenet
import random
from time import sleep

# Main class
class Training():
	def __init__(self, 
	      className: str, 
		  fileExtension: str, 
		  inputImagePath: str) -> None:
		self.__className = className
		self.__fileExtension = fileExtension
		self.__inputImagePath = inputImagePath
		pass

	def exec(self):
		# FIXME: Applying progress bar
		# TODO: Applying image face alignment (Pitch, Yall, Roll)
		# pipeline steps
		imagePreProcessingStep = ImagePreprocessing(
			className=self.__className, 
			fileExtension=self.__fileExtension, 
			inputImagePath=self.__inputImagePath)
		filesCopyStep = FlesCopying()
		faceCrop = FaceCrop()
		modelTraining = ModelTraining()
		modelValidation = ModelValidation()
		
		# pipeline building
		pipeline = (imagePreProcessingStep | filesCopyStep | faceCrop | modelTraining | modelValidation | modelValidation)
		# Iterate through pipeline
		try:
			# Iterate through pipeline
			for _ in pipeline:
				pass
		except StopIteration:
			return
		except KeyboardInterrupt:
			return
		finally:
			print(f"[INFO] End of pipeline")
		# return result

# First step
# Image preprocessing including image resolution improve, denoise, dehazing
class ImagePreprocessing(Pipeline):
	def __init__(self, 
				 className: str, 
				 fileExtension: str, 
				 inputImagePath: str
		) -> None:
		self.__isImagePreprocessingOk = False
		self.__isFileCopyingOk = False
		self.__isFaceCropOk = False
		self.__className = className,
		self.__fileExtension = fileExtension,
		self.__inputImagePath = inputImagePath

		pass

	def generator(self):
		data = {
			"isImagePreprocessingOk": self.__isImagePreprocessingOk,
			"isFileCopyingOk": self.__isFileCopyingOk,
			"isFaceCropOk": self.__isFaceCropOk,
			"className": self.__className,
			"fileExtension": self.__fileExtension
		}
		if self.filter(data):
			yield self.map(data)

	def map(self, data):
		# Image preprocessing code here
		start = timeit.default_timer()
		# Pipeline goal: input: 1 image, output: detection on webcam
		fileName = self.__className + self.__fileExtension
		# Step 1: image preprocessing (denoise, super resolution)
		# @param ["Denoising", "Dehazing_Indoor", "Dehazing_Outdoor", "Deblurring", "Deraining", "Enhancement", "Retouching"]
		ImagePreProcessing(imageURL=self.__inputImagePath, taskName="Denoising", className=self.__className).exec()
		# Step 2: image augmentation
		ImageAugmented(imgPath=fileName, saveToDir="preview", class_name="leo").exec() # save result images in preview folder
		stop = timeit.default_timer()
		print(">>> FINISHED on", (stop - start))

		return super().map(data)

# File preparation and copy to appropriate location
class FlesCopying(Pipeline):
	def __init__(self) -> None:
		self.__sourceFolder = "preview/"
		self.__destinationFolder = "mtcnn/training_data/raw/"
		pass

	def map(self, data):
		start = timeit.default_timer()
		sourcePath = self.__sourceFolder + data["className"]
		destinationPath = self.__destinationFolder + data["className"]
		try:
			shutil.copytree(sourcePath, destinationPath, dirs_exist_ok=False)
		except FileExistsError as err:
			print("[FileExistsError] with content", err)

		stop = timeit.default_timer()
		print(">>> [COPYING] FINISHED on", (stop - start))
		
		return super().map(data)

# Face dataset cropping in suitable dimensions
class FaceCrop(Pipeline):
	def __init__(self) -> None:
		self.__imageSize = 160
		self.__margin = 32
		self.__randomOrder = True
		self.__gpuMemoryFraction = 0.25
		self.__inputDir = "mtcnn/training_data/raw"
		self.__outputDir = "mtcnn/training_data/processed"
		self.__detectMultipleFaces = False
		pass

	def map(self, data):

		sleep(random.random())
		output_dir = os.path.expanduser(self.__outputDir)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# Store some git revision info in a text file in the log directory
		src_path,_ = os.path.split(os.path.realpath(__file__))
		facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
		dataset = facenet.get_dataset(self.__inputDir)
		print('Creating networks and loading parameters')
		
		with tf.Graph().as_default():
			#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
			sess = tf.compat.v1.Session()#config=tf.ConfigProto())#gpu_options=gpu_options, log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
		
		minsize = 20 # minimum size of face
		threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
		factor = 0.709 # scale factor

		# Add a random key to the filename to allow alignment using multiple processes
		random_key = np.random.randint(0, high=99999)
		bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
		
		with open(bounding_boxes_filename, "w") as text_file:
			nrof_images_total = 0
			nrof_successfully_aligned = 0
			if self.__randomOrder:
				random.shuffle(dataset)
			for cls in dataset:
				output_class_dir = os.path.join(output_dir, cls.name)
				if not os.path.exists(output_class_dir):
					os.makedirs(output_class_dir)
					if self.__randomOrder:
						random.shuffle(cls.image_paths)
				for image_path in cls.image_paths:
					nrof_images_total += 1
					filename = os.path.splitext(os.path.split(image_path)[1])[0]
					output_filename = os.path.join(output_class_dir, filename+'.png')
					print(image_path)
					if not os.path.exists(output_filename):
						try:
							import imageio
							img = imageio.imread(image_path)
						except (IOError, ValueError, IndexError) as e:
							errorMessage = '{}: {}'.format(image_path, e)
							print(errorMessage)
						else:
							if img.ndim<2:
								print('Unable to align "%s"' % image_path)
								text_file.write('%s\n' % (output_filename))
								continue
							if img.ndim == 2:
								img = facenet.to_rgb(img)
							img = img[:,:,0:3]
		
							bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
							nrof_faces = bounding_boxes.shape[0]
							if nrof_faces>0:
								det = bounding_boxes[:,0:4]
								det_arr = []
								img_size = np.asarray(img.shape)[0:2]
								if nrof_faces>1:
									if self.__detectMultipleFaces:
										for i in range(nrof_faces):
											det_arr.append(np.squeeze(det[i]))
									else:
										bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
										img_center = img_size / 2
										offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
										offset_dist_squared = np.sum(np.power(offsets,2.0),0)
										index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
										det_arr.append(det[index,:])
								else:
									det_arr.append(np.squeeze(det))

								for i, det in enumerate(det_arr):
									det = np.squeeze(det)
									bb = np.zeros(4, dtype=np.int32)
									bb[0] = np.maximum(det[0]-self.__margin/2, 0)
									bb[1] = np.maximum(det[1]-self.__margin/2, 0)
									bb[2] = np.minimum(det[2]+self.__margin/2, img_size[1])
									bb[3] = np.minimum(det[3]+self.__margin/2, img_size[0])
									cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
									from PIL import Image
									cropped = Image.fromarray(cropped)
									scaled = cropped.resize((self.__imageSize, self.__imageSize), Image.BILINEAR)
									nrof_successfully_aligned += 1
									filename_base, file_extension = os.path.splitext(output_filename)
									if self.__detectMultipleFaces:
										output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
									else:
										output_filename_n = "{}{}".format(filename_base, file_extension)
									imageio.imwrite(output_filename_n, scaled)
									text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
							else:
								print('Unable to align "%s"' % image_path)
								text_file.write('%s\n' % (output_filename))
								
		print('Total number of images: %d' % nrof_images_total)
		print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
		
		return super().map(data)
	

# Model training including model training, model saving
class ModelTraining(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)

# Model validation including accuracy, precision, recall, f1-score, confusion matrix
class ModelValidation(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)