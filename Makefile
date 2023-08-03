class_name := leo
image-preprocessing: # Step 1
	python main.py --input_image_path ${class_name}.jpg --class_name ${class_name}
copying: # Step 2
	python services/copying_files.py --copy_from_dir preview/${class_name} --copy_to_dir mtcnn/training_data/raw/${class_name}
face-crop: # Step 3
	python mtcnn/code/align_dataset_mtcnn.py mtcnn/training_data/raw mtcnn/training_data/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25
face-recognition-train: # Step 4
	python mtcnn/code/classifier.py TRAIN mtcnn/training_data/processed mtcnn/models/20180402-114759.pb mtcnn/models/facemodel.pkl --batch_size 1000
face-recognition-webcam: # Step 5
	python mtcnn/code/face_rec_cam.py --classifier_path mtcnn/models/facemodel.pkl --facenet_model_path mtcnn/models/20180402-114759.pb
face-recognition-video: # Step 5 (optional, video)
	python mtcnn/code/face_rec.py --path video/test1.mp4 --classifier_path mtcnn/models/facemodel.pkl --facenet_model_path mtcnn/models/20180402-114759.pb  