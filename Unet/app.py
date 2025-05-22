from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load the UNet model
new_model = tf.keras.models.load_model('Unet_model.keras')
# Define preprocessing functions
def resize(input_image):
    input_image = tf.image.resize(input_image, (128, 128), method="nearest")
    input_image.set_shape([128, 128, 3])  # Explicitly set the shape
    return input_image

def enhance_contrast(input_image):
    enhanced_image = tf.image.per_image_standardization(input_image)
    return enhanced_image

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

def load_image_and_preprocess(image_path):
    input_image = tf.io.read_file(image_path)
    input_image = tf.image.decode_png(input_image, channels=3)
    input_image = resize(input_image)
    input_image = enhance_contrast(input_image)
    input_image = normalize(input_image)
    return input_image

def display_2(display_list):
    plt.figure(figsize=(10, 8))
    title = ["Input Image","Predicted Tumor Region"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]), cmap = 'gray')
        plt.axis("off")
        plt.savefig('static/pred.jpg')

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image in dataset.take(num):
      pred_mask = new_model.predict(image)
      display_2([image[0], create_mask(pred_mask)])

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the uploaded file to the specified folder
        filename = file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        image_list = []
        image_list.append(image_path)
               
        # Get only files from the image directory
        # Create a TensorFlow dataset from the list of file paths
        image_dataset = tf.data.Dataset.from_tensor_slices(image_list)
        
        # Map the loading and preprocessing function to the dataset
        image_dataset = image_dataset.map(load_image_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        test_batches = image_dataset.take(1).batch(1)
       
        # Now, you can iterate through the dataset and predict
        # Take a single batch for demonstration
        show_predictions(test_batches.take(1),1)

        # Redirect to result page
        return redirect(url_for('result'))

    return redirect(request.url)

@app.route('/result')
def result():
    return render_template('result1.html')

if __name__ == '__main__':
    app.run()