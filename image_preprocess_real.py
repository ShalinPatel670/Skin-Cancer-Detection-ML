import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import zipfile
import tqdm

input_image_path = r"C:\Users\16145\OneDrive\Desktop\skin_ml\540_images.zip"
output_zip_path = r"C:\Users\16145\OneDrive\Desktop\skin_ml\processed_images.zip"


def loadImagesFromZip(zip_path):
    images = []
    with zipfile.ZipFile(r"C:\Users\16145\OneDrive\Desktop\skin_ml\540_images.zip", 'r') as zip_file:
        for file_name in tqdm.tqdm(zip_file.namelist()):
            if file_name.endswith('.JPG'):
                with zip_file.open(file_name) as image_file:
                    image_data = image_file.read()
                    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                    images.append(image_np)
    return images

images = loadImagesFromZip(input_image_path)

print("# of Imgs: " + str(len(images)))


plt.imshow(images[0])
plt.show()

# Preprocessing
def preprocessing(data):
    processed_images = []
    for img in tqdm.tqdm(data):
        print('Original size:', img.shape)

        # Resizing
        height = 224
        width = 224
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

        # Grayscale conversion
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Normalization
        normalized_img = gray_img / 255.0

        # Contrast enhancement(adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(gray_img)

        processed_images.append(enhanced_img)


        processed_images.append(normalized_img)

    return processed_images


processed_images = preprocessing(images)

# Save processed images into a new zip file
with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
    for idx, img in enumerate(processed_images):
        img_filename = f"processed_image_{idx}.jpg"
        img_encoded = cv2.imencode('.jpg', img)[1]
        output_zip.writestr(img_filename, img_encoded.tobytes())

print("Processed images saved to:", output_zip_path)
