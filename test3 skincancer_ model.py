import os
import numpy as np
import pandas as pd
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate




# Load preprocessed clinical data and labels from CSV files
clinical_data = pd.read_csv(r"C:\Users\16145\OneDrive\Desktop\skin_ml\combined_preprocessed_clinicaldata_noheader_noid.csv") 
labels = pd.read_csv(r"C:\Users\16145\OneDrive\Desktop\skin_ml\metadata_labels_multiclass_numeric.csv") 

y_labels = []
with open(r"C:\Users\16145\OneDrive\Desktop\skin_ml\metadata_labels_multiclass_numeric.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        y_labels.extend(row)  

X_clinical = clinical_data.values
#y_labels = labels.values 

zip_file_path = r"C:\Users\16145\OneDrive\Desktop\skin_ml\processed_images.zip"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
     zip_ref.extractall('images')

# Preprocessed image paths and labels
image_directory = 'images'
image_filenames = os.listdir(image_directory)
image_paths = [os.path.join(image_directory, filename) for filename in image_filenames]
image_paths = image_paths[0:540]

# Load images and create a numpy array
X_images = []
for image_path in image_paths:
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    X_images.append(image)
X_images = np.array(X_images) / 255.0  # Normalize

# Split the data into train, validation, and test sets
X_images_train, X_images_temp, X_clinical_train, X_clinical_temp, y_train, y_temp = train_test_split(
    X_images, X_clinical, y_labels, test_size=0.3, random_state=42
)
X_images_val, X_images_test, X_clinical_val, X_clinical_test, y_val, y_test = train_test_split(
    X_images_temp, X_clinical_temp, y_temp, test_size=0.5, random_state=42
)

batch_size = 10
num_features = 17
# Reshape clinical data arrays to be 2D
X_clinical_train = X_clinical_train.reshape(-1, num_features)  # Replace `num_features` with the actual number of features in your clinical data
X_clinical_val = X_clinical_val.reshape(-1, num_features)
X_clinical_test = X_clinical_test.reshape(-1, num_features)

# Create TensorFlow Datasets using zip
train_dataset = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(X_images_train),
        tf.data.Dataset.from_tensor_slices(X_clinical_train),
        tf.data.Dataset.from_tensor_slices(y_train)
    )
).batch(batch_size)

val_dataset = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(X_images_val),
        tf.data.Dataset.from_tensor_slices(X_clinical_val),
        tf.data.Dataset.from_tensor_slices(y_val)
    )
).batch(batch_size)

test_dataset = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(X_images_test),
        tf.data.Dataset.from_tensor_slices(X_clinical_test),
        tf.data.Dataset.from_tensor_slices(y_test)
    )
).batch(batch_size)


train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_images_train, X_clinical_train), y_train)
).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_images_test, X_clinical_test), y_test)
).batch(batch_size)

# Make labels tensorflow-compatible
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

y_train = tf.convert_to_tensor(y_train_encoded, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val_encoded, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test_encoded, dtype=tf.float32)

image_data_generator = ImageDataGenerator(rescale=1.0/255.0)

# Create dataframes for train and test sets
train_df = pd.DataFrame({
    'image_path': [str(path) for path in X_images_train],
    'clinical_data': [data.tolist() for data in X_clinical_train],
    'label': y_train_encoded
})

test_df = pd.DataFrame({
    'image_path': [str(path) for path in X_images_test],
    'clinical_data': [data.tolist() for data in X_clinical_test],
    'label': y_test_encoded
})


num_classes = 3

image_input = Input(shape=(224, 224, 3))
clinical_input = Input(shape=(num_features,))

x_image = Conv2D(32, (3, 3), activation='relu')(image_input)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Conv2D(64, (3, 3), activation='relu')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Conv2D(128, (3, 3), activation='relu')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)
x_image = Flatten()(x_image)

x_clinical = Dense(512, activation='relu')(clinical_input)

combined = concatenate([x_image, x_clinical])

x = Dense(512, activation='relu')(combined)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[image_input, clinical_input], outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

num_epochs = 10
history = model.fit(
    x=[X_images_train, X_clinical_train],
    y=y_train,
    epochs=num_epochs,
    validation_data=([X_images_val, X_clinical_val], y_val),
    verbose=1
)

# Plot accuracy over epochs
plt.plot(np.arange(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(1, num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

# Test set
test_loss, test_accuracy = model.evaluate(
    x=[X_images_test, X_clinical_test],
    y=y_test,
    verbose=1
)

y_pred = model.predict([X_images_test, X_clinical_test])
y_pred_classes = np.argmax(y_pred, axis=1)

y_test_original = label_encoder.inverse_transform(y_test.numpy().astype(int))
y_pred_original = label_encoder.inverse_transform(y_pred_classes)

print("Classification Report:")
print(classification_report(y_test_original, y_pred_original))
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

conf_matrix = confusion_matrix(y_test_original, y_pred_original)
print("Confusion Matrix:")
print(conf_matrix)
