from services.image_augemented import ImageAugmented
from services.image_preprocessing import ImagePreProcessing
import sys
import argparse
import timeit


def main(args):
    print(">>> STARTED")

    start = timeit.default_timer()
    # FIXME: Applying pipeline
    # FIXME: Building cache to serve resume current process if any occurs
    # FIXME: Applying progress bar
    # FIXME: Applying error management
    # TODO: Applying image face alignment (Pitch, Yall, Roll)
    
    # Pipeline goal: input: 1 image, output: detection on webcam
    fileName = args.class_name + ".png"
    # Step 1: image preprocessing (denoise, super resolution)
    # @param ["Denoising", "Dehazing_Indoor", "Dehazing_Outdoor", "Deblurring", "Deraining", "Enhancement", "Retouching"]
    ImagePreProcessing(imageURL=args.input_image_path, taskName="Denoising", className=args.class_name).exec()
    
    # Step 2: image augmentation
    ImageAugmented(imgPath=fileName, saveToDir="preview", class_name="leo").exec() # save result images in preview folder
    
    stop = timeit.default_timer()
    print(">>> FINISHED on", (stop - start))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_image_path', type=str,
        help='Input image path')
    parser.add_argument('--class_name', type=str,
        help='Class name')
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))



