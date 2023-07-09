import DataAugmentation
import FacialRecognition


#FASE 1: Classify images into Drowsy/Non Drowsy
Source_folder_drowsiness = "img_source"
Target_folder_drowsiness = "img_target"

FacialRecognition.classify_drowsiness(Source_folder_drowsiness, Target_folder_drowsiness)

#FASE 2: Data augmentation
Source_folder_aug = Target_folder_drowsiness 
Target_folder_aug = "img_aug"

DataAugmentation.rotate_image(Source_folder_aug, Target_folder_aug, False)

#FASE 3: Crop image
Source_folder_crop = Target_folder_aug
Target_folder_crop = "final_model"

FacialRecognition.crop_eyes_and_face(Source_folder_crop, Target_folder_crop)

