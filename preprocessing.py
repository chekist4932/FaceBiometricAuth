import os
import cv2
from PIL import Image
from dlib import get_frontal_face_detector

origin_dataset_dir = r"C:\Users\GEORG\Downloads\FaceDataset"
prerender_dataset_frontal_dir = r"dataset\photo_dataset_frontal"
prerender_dataset_profile_dir = r"dataset\photo_dataset_profile"
path_to_face_frontal_detector = "source/haarcascade_frontalface_default.xml"
path_to_face_profile_detector = "source/haarcascade_profileface.xml"

photos = os.listdir(origin_dataset_dir)


def resize_without_deformation(image, size=(500, 500)):
    height, width, _ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left
    image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized_image = cv2.resize(image_with_border, size)
    return resized_image


total = 0
loss = 0
loss_name = []

face_detector = cv2.CascadeClassifier(path_to_face_profile_detector)

for photo_name in photos:

    photo_path = origin_dataset_dir + '\\' + photo_name

    image = cv2.imread(photo_path)
    image = resize_without_deformation(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_result = face_detector.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face_result) != 0:
        x, y, w, h = face_result[0]
        image = image[y:y + h, x:x + w]
        image = resize_without_deformation(image, size=(224, 244))

        save_path = prerender_dataset_profile_dir + f'\\{photo_name}.jpg'
        cv2.imwrite(save_path, image)
        total += 1


    else:
        loss += 1
        print(f"Face not found in - '{photo_name}'")
        loss_name.append(photo_name)
        continue

print(f'Total preprocessing: {total}')
print(f'Loss preprocessing: {loss}\nLoss names: {loss_name}')
