
import cv2
import dlib
import os
import numpy as np
import time

threshold = 0.5
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

def draw_face_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) > 0:
        face = faces[0]  # Get the first detected face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


def eyes_closed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)

        left_eye = [
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(37).x, landmarks.part(37).y),
            (landmarks.part(38).x, landmarks.part(38).y),
            (landmarks.part(39).x, landmarks.part(39).y),
            (landmarks.part(40).x, landmarks.part(40).y),
            (landmarks.part(41).x, landmarks.part(41).y)
        ]

        right_eye = [
            (landmarks.part(42).x, landmarks.part(42).y),
            (landmarks.part(43).x, landmarks.part(43).y),
            (landmarks.part(44).x, landmarks.part(44).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(46).x, landmarks.part(46).y),
            (landmarks.part(47).x, landmarks.part(47).y)
        ]

        left_eye_status = calculate_eye_status(left_eye)
        right_eye_status = calculate_eye_status(right_eye)

        if left_eye_status == "closed" or right_eye_status == "closed":
            return True  # Eyes are closed

    return False  # Eyes are open


def calculate_eye_status(eye):
    eye_hull = cv2.convexHull(np.array(eye))
    eye_area = cv2.contourArea(eye_hull)
    eye_rect = cv2.boundingRect(eye_hull)
    eye_rect_area = eye_rect[2] * eye_rect[3]
    eye_aspect_ratio = eye_area / eye_rect_area

    if eye_aspect_ratio < threshold:
        return "closed"
    else:
        return "open"

def crop_eyes_and_face(image):
    # Load the shape predictor model for facial landmarks
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Predict the facial landmarks
        landmarks = predictor(gray, face)

        # Get the coordinates of the eyes and face
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        face_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 68)]

        # Find the bounding box for the eyes and face
        left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
        right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))
        face_bbox = cv2.boundingRect(np.array(face_points))

        # Expand the face bounding box to include the eyes
        face_bbox = (min(face_bbox[0], left_eye_bbox[0]),
                     min(face_bbox[1], left_eye_bbox[1]),
                     max(face_bbox[0] + face_bbox[2], right_eye_bbox[0] + right_eye_bbox[2]) - min(face_bbox[0], left_eye_bbox[0]),
                     max(face_bbox[1] + face_bbox[3], left_eye_bbox[1] + left_eye_bbox[3]) - min(face_bbox[1], left_eye_bbox[1]))

        # Crop the region of interest (ROI) for eyes and face
        eyes_and_face_roi = image[face_bbox[1]:face_bbox[1] + face_bbox[3],
                                  face_bbox[0]:face_bbox[0] + face_bbox[2]]

        # Show the region of interest
        cv2.imshow("Eyes and Face", eyes_and_face_roi)

    # Show the original image with the detected faces
    cv2.imshow("Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_eyes_and_face(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the shape predictor model for facial landmarks
    predictor_path = ''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        # Iterate over the detected faces
        for i, face in enumerate(faces):
            # Predict the facial landmarks
            landmarks = predictor(gray, face)

            # Get the coordinates of the eyes and face
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            face_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 68)]

            # Find the bounding box for the eyes and face
            left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
            right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))
            face_bbox = cv2.boundingRect(np.array(face_points))

            # Expand the face bounding box to include the eyes
            face_bbox = (min(face_bbox[0], left_eye_bbox[0]),
                         min(face_bbox[1], left_eye_bbox[1]),
                         max(face_bbox[0] + face_bbox[2], right_eye_bbox[0] + right_eye_bbox[2]) - min(face_bbox[0], left_eye_bbox[0]),
                         max(face_bbox[1] + face_bbox[3], left_eye_bbox[1] + left_eye_bbox[3]) - min(face_bbox[1], left_eye_bbox[1]))

            # Crop the region of interest (ROI) for eyes and face
            eyes_and_face_roi = image[face_bbox[1]:face_bbox[1] + face_bbox[3],
                                      face_bbox[0]:face_bbox[0] + face_bbox[2]]

            # Save the processed image in the output folder
            output_filename = f"processed_{i+1}_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, eyes_and_face_roi)

            print(f"Processed image {filename} and saved as {output_filename}")
                    # Pause for 1 second

    print("Image processing completed.")



def classify_drowsiness(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
         # Read the image
        #print(f"Imagen: {1}", filename)
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        # Save the image with a prefix according to eyes open/closed
        if eyes_closed(image):
            output_filename = f"D4_{filename}"
        else:
            output_filename = f"ND4_{filename}"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)

 

