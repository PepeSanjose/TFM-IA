import cv2
import DataAugmentation
import FacialRecognition

#FASE 1: Images processing

# Specify the input folder containing the images
Source_folder = "img"
Target_folder = "img"

#FASE 1: Data augmentation
DataAugmentation.rotate_image(Source_folder, True)

# Call the function to process the images and save them in the output folder
FacialRecognition.crop_eyes_and_face(Source_folder, Target_folder)