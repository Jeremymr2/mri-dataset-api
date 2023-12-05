import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet101
import requests
import boto3

# define classes name   
class_names = ['glioma','meningioma','notumor','pituitary']

url = 'https://datamining-final.s3.amazonaws.com/resnet101_weights.pickle'

# Descargar el archivo
response = requests.get(url)

resnet101 = ResNet101(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)

# Freeze model
for layer in resnet101.layers:
    layer.trainable = False

# build the entire model
x = resnet101.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(4, activation='softmax')(x)
resnet101_model = Model(inputs = resnet101.input, outputs = predictions)

# Verificar que la descarga fue exitosa
if response.status_code == 200:
    # Cargar los datos del archivo pickle directamente desde el contenido de la respuesta
    resnet101_weights = pickle.loads(response.content)
else:
    print("Error al descargar el archivo:", response.status_code)

resnet101_model.set_weights(resnet101_weights)

def predict(image_input):
    img = Image.open(image_input)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array]) 
    
    # generate predictions for samples
    predictions = resnet101_model.predict(img_array)
    print(predictions)

    # generate argmax for predictions
    class_id = np.argmax(predictions, axis = 1)
    print(class_id)

    # transform classes number into classes name
    return class_names[class_id.item()]