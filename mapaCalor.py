import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# Función para generar el mapa de calor
def generate_heatmap(model, img_path, target_class):
    # Cargar la imagen y preprocesarla
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_img = preprocess(img).unsqueeze(0)

    # Evaluar el modelo
    model.eval()
    input_img = input_img.to('cpu')
    output = model(input_img)

    # Modificar la capa target_layer para tener 3 canales
    target_layer = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1)

    # Obtener los features map
    activations = target_layer(input_img)
    activations = activations.detach()

    # Calcular los pesos para combinar los features map y los gradientes
    weights = torch.mean(activations, dim=(2, 3), keepdim=True)
    heatmap = torch.sum(weights * activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)  # ReLU

    # Normalizar el mapa de calor
    heatmap /= np.max(heatmap)

    # Redimensionar el mapa de calor a las dimensiones originales de la imagen
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Aplicar el mapa de calor a la imagen original
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

    return result

# Ruta de la imagen de prueba y clase objetivo
image_path = "img/imagen1.png"
target_class = 1  # Por ejemplo, 1 para "Drowsy"

modelResNet = models.resnet18(pretrained=False)
num_features = modelResNet.fc.in_features
modelResNet.fc = nn.Linear(num_features, 2)
modelResNet.load_state_dict(torch.load('Modelo/ResNet.pth', map_location=torch.device('cpu')))
modelResNet.eval()

# Generar el mapa de calor y visualizarlo
heatmap_img = generate_heatmap(modelResNet, image_path, target_class)
if heatmap_img is not None:
    resize = ResizeWithAspectRatio(heatmap_img, height=300) 
    cv2.imshow('resize', resize)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se pudo generar el mapa de calor.")

