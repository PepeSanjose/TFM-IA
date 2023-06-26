import cv2
import DataAugmentation
import FacialRecognition


#FASE 1: Classify images into Drowsy/Non Drowsy
Source_folder_drowsiness = "C:/Users/egutierrezs/Desktop/Projectes/UEM/MODULO7/TFM/ProyectoIntermedio/tmp/DDD_Reduced_CarasSeparadas/TRAIN/Non Drowsy"
Target_folder_drowsiness = "C:/Users/egutierrezs/Desktop/Projectes/UEM/MODULO7/TFM/ProyectoIntermedio/tmp/DDD_Reduced_CarasSeparadas/TRAIN/Classified"

FacialRecognition.classify_drowsiness(Source_folder_drowsiness, Target_folder_drowsiness)

#FASE 2: Data augmentation
Source_folder_aug = Target_folder_drowsiness 
Target_folder_aug = "C:/Users/egutierrezs/Desktop/Projectes/UEM/MODULO7/TFM/ProyectoIntermedio/tmp/DDD_Reduced_CarasSeparadas/TRAIN/Augmentation"

DataAugmentation.rotate_image(Source_folder_aug, Target_folder_aug, False)

#FASE 3: Crop image
Source_folder_crop = Target_folder_aug
Target_folder_crop = "C:/Users/egutierrezs/Desktop/Projectes/UEM/MODULO7/TFM/ProyectoIntermedio/tmp/DDD_Reduced_CarasSeparadas/TRAIN/Crop"

FacialRecognition.crop_eyes_and_face(Source_folder_crop, Target_folder_crop)

