import os
import dlib
import cv2
import math

def rotate_image(input_folder, output_folder, paint):

    # Cargamos el detector de rostros de DLib
    detector = dlib.get_frontal_face_detector()

    # Cargamos el modelo de puntos faciales de DLib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
    
    for filename in os.listdir(input_folder):
        print("Augmentation image")
        i= 0
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

         # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectamos los rostros en la imagen
        faces = detector(gray)

        # Iteramos sobre los rostros detectados
        for face in faces:
            # Obtenemos los puntos faciales del rostro
            landmarks = predictor(gray, face)

            # Iteramos sobre los puntos faciales y los dibujamos en la imagen
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                if paint:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Obtenemos las coordenadas de los ojos (puntos 36 y 45 en el modelo de 68 puntos de DLib)
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)

            # Calculamos el ángulo de la línea que une los ojos
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))

            # Obtenemos las dimensiones de la imagen
            height, width = image.shape[:2]

            # Calculamos el centro de la imagen
            center = (width // 2, height // 2)

            # Definimos la matriz de transformación de rotación
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Aplicamos la rotación a la imagen
            rotated_image = cv2.warpAffine(image, matrix, (width, height))
            # Save the processed image in the output folder
            output_filename = f"aug_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, rotated_image)
            i= i+1