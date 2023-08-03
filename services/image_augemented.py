from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import shutil

class ImageAugmented():
    def __init__(self, imgPath, saveToDir, class_name) -> None:
        print(">>> [IMAGE AUGMENTATION] running...")
        self.__IMAGE_PATH = imgPath
        self.__SAVE_TO_DIR = saveToDir
        self.__CLASS_NAME = class_name
        pass

    def exec(self):
        datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


        # open method used to open different extension image file
        img = load_img(self.__IMAGE_PATH)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # TODO: Emotion handling
        # TODO: Age handling

        # Create folder before put image into
        # FIXME: Check if new directory is exist, if exist => copy to it, if not exist => create new one
        # Directory
        directory = self.__CLASS_NAME
        
        # Parent Directory path
        parent_dir = self.__SAVE_TO_DIR
        
        # Path
        newDir = self.__SAVE_TO_DIR + "/" + self.__CLASS_NAME
        shutil.rmtree(os.path.join(parent_dir, directory), ignore_errors=True)
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        print("Saved augmented images into directory", newDir)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=10, save_to_dir=newDir, save_prefix=self.__CLASS_NAME, save_format='jpeg'):
            i += 1
            if i > 50:
                break  # otherwise the generator would loop indefinitely