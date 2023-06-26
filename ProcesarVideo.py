import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Cargar el modelo ResNet18 entrenado
modelResNet = models.resnet18(pretrained=False)
num_features = modelResNet.fc.in_features
modelResNet.fc = nn.Linear(num_features, 2)
modelResNet.load_state_dict(torch.load('Modelo/ResNet.pth',  map_location=torch.device('cpu')))
modelResNet.eval()

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

# Ruta del archivo de video
video_path = 'Videos/2.mp4'

# Parámetros
frames_per_second = 1
#num_consecutive_frames = 20
drowsy_threshold = 20

# Contadores
drowsy_count = 0
frame_count = 0
total_drowsy = 0
total_nondrowsy = 0

# Procesamiento del video
cap = cv2.VideoCapture(video_path)
while True:
    # Leer el siguiente fotograma del video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Incrementar el contador de fotogramas
    frame_count += 1


    # Procesar la imagen actual
    if frame_count % frames_per_second == 0:
        predicted_label = process_image(frame)
        
        if predicted_label == 1:  # Clase "Drowsy"
            total_drowsy += 1
            drowsy_count += 1
            if drowsy_count >= drowsy_threshold:
                print("Somnolencia detectada")
                drowsy_count = 0
        else:
            total_nondrowsy += 1
            drowsy_count = 0
    
    # # Mostrar el fotograma procesado
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

print(f"TOTAL Drowsy: {total_drowsy} TOTAL NonDrowsy: {total_nondrowsy}")
