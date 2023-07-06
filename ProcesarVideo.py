import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time
import FacialRecognition
import pygame

# Cargar el modelo ResNet18 entrenado

modelResNet = models.resnet18(pretrained=False)
num_features = modelResNet.fc.in_features
modelResNet.fc = nn.Linear(num_features, 2)
modelResNet.load_state_dict(torch.load('Modelo/ResNet.pth',  map_location=torch.device('cpu')))
modelResNet.eval()


def reproducir_sonido(ruta_archivo):
    pygame.mixer.init()
    pygame.mixer.music.load(ruta_archivo)
    pygame.mixer.music.play()

# Llamar a la función para reproducir un archivo de sonido

# Función para procesar una imagen y obtener la predicción
def process_image(image):
    # Preprocesar la imagen
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convertir la imagen de numpy.ndarray a torch.Tensor
    input_img = preprocess(Image.fromarray(image)).unsqueeze(0)


    # Realizar la predicción
    with torch.no_grad():
        outputs = modelResNet(input_img)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()

def mostrar_webcam():
    # Crear un objeto de captura de video
    
  
    #num_consecutive_frames = 7
    drowsy_threshold = 4
    # Contadores
    drowsy_count = 0

    cap = cv2.VideoCapture(0)

    # Comprobar si la cámara está abierta correctamente
    if not cap.isOpened():
        raise IOError("No se puede abrir la cámara")

    # Bucle principal
    while True:
        time.sleep(1)
        # Leer el fotograma actual de la cámara
        ret, frame = cap.read()

        # Comprobar si se ha leído correctamente el fotograma
        if not ret:
            break

        # Mostrar el fotograma en una ventana llamada "Webcam"

        image = FacialRecognition.draw_face_rectangle(frame)

        cv2.imshow("Webcam", frame)

        #TO DO: LLAMAR A LA FUNCION DE ESTHER
        #predicted_label = FacialRecognition.eyes_closed(frame)
        
        predicted_label = process_image(frame)

        print(f"El valor de predicted_label es:{predicted_label}")

        if predicted_label ==1:  # Clase "Drowsy"
            drowsy_count += 1
            print(f"Ha cerrado los ojos, :{drowsy_count}")
            if drowsy_count >= drowsy_threshold: 
                reproducir_sonido("Sonidos/alert-long.mp3")

        else:
            drowsy_count = 0    
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Llamar al método para mostrar la webcam en tiempo real
